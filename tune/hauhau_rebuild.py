"""Rebuild full-precision (HF safetensors) weights of HauhauCS's abliterated Qwen3.6-35B-A3B from its Q8_K_P GGUF, so
our trainer (which streams the bf16 HF checkpoint) can train adapters on it.

The GGUF was made by llama.cpp's converter, which transforms some tensors on the way (vendor/llama.cpp/conversion/
qwen.py): norm weights +1 (not the linear-attention norm), A_log -> -exp(A_log), conv1d squeezed, the experts'
gate_up_proj split into gate and up, and (in newer converter versions) the V heads of the linear-attention tensors
reordered from grouped to tiled. Every one of those is undone here. Whether this GGUF has the V-head reorder is not
known in advance, so `verify` tries both against the ORIGINAL Qwen weights: abliteration changes the weights only a
little, so the right inverse matches the original almost exactly and a wrong one does not. Most attention tensors are
stored in F16/F32 in this file (exact); the experts, output projections and embeddings are Q8_0 (~0.3 % error).
The vision tower and the MTP head are not in the GGUF and are copied from the original unchanged.

    uv run python hauhau_rebuild.py verify --layers 0,3,20,39
    uv run python hauhau_rebuild.py write --out ../models/hf/Qwen3.6-35B-A3B-hauhau
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor" / "llama.cpp" / "gguf-py"))
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

ORIG = Path("../models/hf/Qwen3.6-35B-A3B")
GGUF = Path("../models/base/hauhau/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf")
cfg = json.load(open(ORIG / "config.json"))
tc = cfg.get("text_config", cfg)
NK, NV, DK, DV = tc["linear_num_key_heads"], tc["linear_num_value_heads"], tc["linear_key_head_dim"], tc["linear_value_head_dim"]
R = NV // NK
QK = DK * NK  # q rows = k rows in in_proj_qkv


def reorder(t, dim, hd):
    """grouped -> tiled, exactly as the converter's _reorder_v_heads."""
    shape = list(t.shape); dim %= len(shape)
    t = t.reshape(shape[:dim] + [NK, R, hd] + shape[dim + 1:])
    p = list(range(t.ndim)); p[dim], p[dim + 1] = p[dim + 1], p[dim]
    return t.permute(*p).contiguous().reshape(shape)


def unreorder(t, dim, hd):
    """tiled -> grouped (inverse of reorder)."""
    shape = list(t.shape); dim %= len(shape)
    t = t.reshape(shape[:dim] + [R, NK, hd] + shape[dim + 1:])
    p = list(range(t.ndim)); p[dim], p[dim + 1] = p[dim + 1], p[dim]
    return t.permute(*p).contiguous().reshape(shape)


def plan(name):
    """HF tensor name -> (kind, layer, gguf base names) or None if it is not in the text GGUF."""
    if name == "model.language_model.embed_tokens.weight": return ("plain", None, ["token_embd.weight"])
    if name == "lm_head.weight": return ("plain", None, ["output.weight"])
    if name == "model.language_model.norm.weight": return ("norm", None, ["output_norm.weight"])
    m = re.match(r"model\.language_model\.layers\.(\d+)\.(.+)$", name)
    if not m: return None
    L, rest = int(m.group(1)), m.group(2)
    b = f"blk.{L}."
    table = {
        "input_layernorm.weight": ("norm", ["attn_norm.weight"]),
        "post_attention_layernorm.weight": ("norm", ["post_attention_norm.weight"]),
        "self_attn.q_proj.weight": ("plain", ["attn_q.weight"]), "self_attn.k_proj.weight": ("plain", ["attn_k.weight"]),
        "self_attn.v_proj.weight": ("plain", ["attn_v.weight"]), "self_attn.o_proj.weight": ("plain", ["attn_output.weight"]),
        "self_attn.q_norm.weight": ("norm", ["attn_q_norm.weight"]), "self_attn.k_norm.weight": ("norm", ["attn_k_norm.weight"]),
        "linear_attn.in_proj_qkv.weight": ("qkv", ["attn_qkv.weight"]),
        "linear_attn.in_proj_z.weight": ("rows_v", ["attn_gate.weight"]),
        "linear_attn.in_proj_a.weight": ("rows_1", ["ssm_alpha.weight"]),
        "linear_attn.in_proj_b.weight": ("rows_1", ["ssm_beta.weight"]),
        "linear_attn.out_proj.weight": ("cols_v", ["ssm_out.weight"]),
        "linear_attn.conv1d.weight": ("conv", ["ssm_conv1d.weight"]),
        "linear_attn.A_log": ("alog", ["ssm_a"]),
        "linear_attn.dt_bias": ("vec_1", ["ssm_dt.bias"]),
        "linear_attn.norm.weight": ("plain", ["ssm_norm.weight"]),
        "mlp.gate.weight": ("plain", ["ffn_gate_inp.weight"]),
        "mlp.shared_expert_gate.weight": ("shgate", ["ffn_gate_inp_shexp.weight"]),
        "mlp.shared_expert.gate_proj.weight": ("plain", ["ffn_gate_shexp.weight"]),
        "mlp.shared_expert.up_proj.weight": ("plain", ["ffn_up_shexp.weight"]),
        "mlp.shared_expert.down_proj.weight": ("plain", ["ffn_down_shexp.weight"]),
        "mlp.experts.gate_up_proj": ("gateup", ["ffn_gate_exps.weight", "ffn_up_exps.weight"]),
        "mlp.experts.down_proj": ("plain", ["ffn_down_exps.weight"]),
    }
    if rest not in table: return None
    kind, names = table[rest]
    return (kind, L, [b + n for n in names])


def to_gguf(kind, x, ro):
    """original HF tensor -> list of tensors in GGUF layout (the converter's forward transform); ro = V-head reorder."""
    x = x.float()
    if kind == "plain": return [x]
    if kind == "norm": return [x + 1]
    if kind == "shgate": return [x.reshape(-1)]
    if kind == "gateup": n = x.shape[-2] // 2; return [x[..., :n, :], x[..., n:, :]]
    if kind == "qkv":
        q, k, v = x[:QK], x[QK:2 * QK], x[2 * QK:]
        return [torch.cat([q, k, reorder(v, 0, DV) if ro else v], 0)]
    if kind == "rows_v": return [reorder(x, 0, DV) if ro else x]
    if kind == "rows_1": return [reorder(x, 0, 1) if ro else x]
    if kind == "cols_v": return [reorder(x, 1, DV) if ro else x]
    if kind == "vec_1": return [reorder(x.unsqueeze(-1), 0, 1).squeeze(-1) if ro else x]
    if kind == "alog":
        y = -torch.exp(x)
        return [reorder(y.unsqueeze(-1), 0, 1).squeeze(-1) if ro else y]
    if kind == "conv":
        d = x.squeeze()
        return [torch.cat([d[:2 * QK], reorder(d[2 * QK:], 0, DV) if ro else d[2 * QK:]], 0)]
    raise ValueError(kind)


def from_gguf(kind, ts, ro, like):
    """GGUF-layout tensors -> the HF tensor (inverse of to_gguf), shaped and typed like `like`."""
    if kind == "plain": y = ts[0]
    elif kind == "norm": y = ts[0] - 1
    elif kind == "shgate": y = ts[0].reshape(like.shape)
    elif kind == "gateup": y = torch.cat(ts, dim=-2)
    elif kind == "qkv":
        x = ts[0]; v = x[2 * QK:]
        y = torch.cat([x[:QK], x[QK:2 * QK], unreorder(v, 0, DV) if ro else v], 0)
    elif kind == "rows_v": y = unreorder(ts[0], 0, DV) if ro else ts[0]
    elif kind == "rows_1": y = unreorder(ts[0], 0, 1) if ro else ts[0]
    elif kind == "cols_v": y = unreorder(ts[0], 1, DV) if ro else ts[0]
    elif kind == "vec_1": y = unreorder(ts[0].unsqueeze(-1), 0, 1).squeeze(-1) if ro else ts[0]
    elif kind == "alog":
        a = unreorder(ts[0].unsqueeze(-1), 0, 1).squeeze(-1) if ro else ts[0]
        y = torch.log(-a)
    elif kind == "conv":
        d = ts[0]
        y = torch.cat([d[:2 * QK], unreorder(d[2 * QK:], 0, DV) if ro else d[2 * QK:]], 0).reshape(like.shape)
    else:
        raise ValueError(kind)
    assert tuple(y.shape) == tuple(like.shape), (kind, tuple(y.shape), tuple(like.shape))
    return y.to(like.dtype)


class Source:
    def __init__(self):
        self.r = GGUFReader(str(GGUF))
        self.t = {t.name: t for t in self.r.tensors}

    def get(self, name):
        t = self.t[name]
        a = dequantize(t.data, t.tensor_type) if t.tensor_type.name not in ("F32", "F16", "BF16") else np.asarray(t.data, dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def orig_index():
    return json.load(open(ORIG / "model.safetensors.index.json"))["weight_map"]


def load_orig(name, wm):
    with safe_open(str(ORIG / wm[name]), framework="pt") as f:
        return f.get_tensor(name)


def rel(a, b):
    return float((a - b).norm() / (b.norm() + 1e-12))


def verify(layers):
    src, wm = Source(), orig_index()
    names = [n for n in wm if plan(n) and (plan(n)[1] is None or plan(n)[1] in layers)]
    print(f"{len(names)} tensors to check (layers {sorted(layers)} + embeddings/norm/head)")
    votes = {True: 0, False: 0}
    rows = {}
    for n in names:
        kind, L, gn = plan(n)
        x = load_orig(n, wm)
        got = [src.get(g) for g in gn]
        errs = {}
        for ro in (True, False):
            want = to_gguf(kind, x, ro)
            try:
                errs[ro] = max(rel(g.reshape(w.shape), w) for g, w in zip(got, want))
            except RuntimeError:
                errs[ro] = float("inf")
        if kind in ("qkv", "rows_v", "rows_1", "cols_v", "vec_1", "alog", "conv") and abs(errs[True] - errs[False]) > 1e-3:
            votes[errs[True] < errs[False]] += 1
        best = min(errs.values())
        key = re.sub(r"layers\.\d+\.", "layers.N.", n)
        rows.setdefault(key, []).append((best, src.t[gn[0]].tensor_type.name))
    print(f"\nV-head reorder votes: present {votes[True]}, absent {votes[False]}")
    print(f"\n{'tensor':66s} {'type':5s} {'max rel. diff from original':>28s}")
    for k, v in sorted(rows.items()):
        print(f"{k:66s} {v[0][1]:5s} {max(e for e, _ in v) * 100:27.3f}%")
    return votes[True] >= votes[False]


def write(out, ro):
    src, wm = Source(), orig_index()
    out.mkdir(parents=True, exist_ok=True)
    for f in ORIG.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy2(f, out / f.name)
    shards = sorted(set(wm.values()))
    t0, replaced = time.time(), 0
    for si, shard in enumerate(shards):
        if (out / shard).exists():
            continue
        tensors = {}
        with safe_open(str(ORIG / shard), framework="pt") as f:
            for n in f.keys():
                x = f.get_tensor(n)
                p = plan(n)
                if p:
                    kind, L, gn = p
                    tensors[n] = from_gguf(kind, [src.get(g) for g in gn], ro, x).contiguous()
                    replaced += 1
                else:
                    tensors[n] = x.contiguous()
        save_file(tensors, str(out / (shard + ".tmp")), metadata={"format": "pt"})
        (out / (shard + ".tmp")).replace(out / shard)
        print(f"shard {si + 1}/{len(shards)} {shard}: {len(tensors)} tensors ({replaced} rebuilt so far, {(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"wrote {out} ({replaced} text tensors rebuilt from the GGUF, the rest copied from the original)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["verify", "write"])
    ap.add_argument("--layers", default="0,3,20,39")
    ap.add_argument("--out", default="../models/hf/Qwen3.6-35B-A3B-hauhau")
    ap.add_argument("--reorder", choices=["auto", "yes", "no"], default="auto")
    a = ap.parse_args()
    if a.mode == "verify":
        verify({int(x) for x in a.layers.split(",")})
    else:
        ro = {"yes": True, "no": False}.get(a.reorder)
        if ro is None:
            ro = verify({0, 1, 2})
            print(f"\nusing V-head reorder = {ro}\n")
        write(Path(a.out), ro)


if __name__ == "__main__":
    main()
