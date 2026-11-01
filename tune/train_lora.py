"""Train a micro-turn LoRA on the top layers of a large model on one consumer GPU (Qwen3.6-35B-A3B, Gemma 4 31B: the
model-specific parts are in profiles.py, picked from the checkpoint's config).

The idea (from the Codacus adapter): the lower layers are frozen, so their output for every training call never
changes. Phase A streams the bf16 HF checkpoint through the GPU one layer at a time and caches the hidden state at the
input of the first trained layer for every call (about 33 MB per 8k-token call in bf16). Phase B then trains LoRA
adapters on the last `--top` layers plus the final norm and lm_head, which fit in ~14 GB of VRAM.

    uv run --extra train python train_lora.py --hf ../models/hf/Qwen3.6-35B-A3B --calls out/calls_train.jsonl \
        --steps 282 --batch 4 --rank 32 --out out/adapter

Loss: cross-entropy on the assistant tokens of every tick (the reply plus its <|im_end|>), weighted by the tick's `w`,
normalised by the sum of weights. The adapter is saved as a PEFT adapter whose tensor names use the full HF module
path, so llama.cpp's convert_lora_to_gguf.py can map them (see export_gguf.py).
"""
import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from common import LISTEN, NOVOICE, SYSTEM, SYSTEMS, read_jsonl
from profiles import FORMATS, TRAIN, model_family

PREFIX = "model.language_model."  # Qwen and Gemma keep their text model under this name
EXPORT_PREFIX = PREFIX  # the same in the exported adapter, except for a text-only model ("model.", set in main)
SHARD_ALIASES = (("language_model.model.", PREFIX), ("language_model.lm_head.", "lm_head."),  # Ministral 3's names
                 # a text-only model (gpt-oss): "model.layers.", "model.embed_tokens.", "model.norm."
                 ("model.layers.", PREFIX + "layers."), ("model.embed_tokens.", PREFIX + "embed_tokens."), ("model.norm.", PREFIX + "norm."))


# ----------------------------------------------------------------------------------------------------------------------
# checkpoint access
# ----------------------------------------------------------------------------------------------------------------------
class Shards:
    def __init__(self, hf_dir):
        self.dir = Path(hf_dir)
        idx = json.load(open(self.dir / "model.safetensors.index.json"))
        # every tensor under the names Qwen and Gemma use; Ministral 3 keeps its text model under "language_model.model."
        # and its head as "language_model.lm_head."
        alias = lambda n: next((new + n[len(old):] for old, new in SHARD_ALIASES if n.startswith(old)), n)
        self.map = {alias(n): shard for n, shard in idx["weight_map"].items()}
        self.real = {alias(n): n for n in idx["weight_map"]}
        self.open = {}

    def get(self, name, device="cpu", dtype=None):
        shard = self.map[name]
        if shard not in self.open:
            self.open[shard] = safe_open(str(self.dir / shard), framework="pt", device="cpu")
        t = self.open[shard].get_tensor(self.real[name])
        if dtype is not None:
            t = t.to(dtype)
        return t.to(device)

    def names(self, prefix):
        return [n for n in self.map if n.startswith(prefix)]

    def close(self):
        self.open = {}


def load_module(module, shards, prefix, device, dtype=torch.bfloat16, rename=None):
    """Fill `module` (built on the meta device) with the checkpoint tensors under `prefix` (rename: checkpoint key ->
    module key, for a tensor that sits under another name, e.g. Gemma's tied output layer)."""
    # to_empty keeps the meta module's float32 parameters, so move to the target dtype before loading, otherwise
    # load_state_dict silently upcasts every weight to float32
    # buffers that are not in the checkpoint (non-persistent) come out of to_empty as uninitialised memory: a profile
    # must say what they hold (fixed_buffers), or loading stops. Gemma's embedding scale was one: every output was NaN.
    persistent = set(module.state_dict().keys())
    fixed = getattr(module, "fixed_buffers", {})
    unknown = [n for n, _ in module.named_buffers() if n not in persistent and n not in fixed and "inv_freq" not in n]
    if unknown:
        raise RuntimeError(f"{prefix}: buffers not in the checkpoint and not set by the profile: {unknown}")
    module.to_empty(device=device)
    module.to(dtype)
    for n, v in fixed.items():
        module.get_buffer(n).fill_(v)
    sd = {}
    for name in shards.names(prefix):
        key = name[len(prefix):]
        if rename is not None:
            if key not in rename:
                continue
            key = rename[key]
        t = shards.get(name)
        # keep the delta-net state parameters in float32 as the modelling code expects
        if key.endswith(("A_log", "dt_bias")):
            t = t.to(torch.float32)
        else:
            t = t.to(dtype)
        sd[key] = t
    missing, unexpected = module.load_state_dict(sd, strict=False)
    del sd
    # on Windows the shard mappings are copy-on-write and count as private memory; drop them after every module
    shards.close()
    missing = [m for m in missing if "inv_freq" not in m]
    if missing or unexpected:
        raise RuntimeError(f"{prefix}: missing {missing[:5]} unexpected {unexpected[:5]}")
    for n, p in module.named_parameters():
        if n.endswith(("A_log", "dt_bias")):
            p.data = p.data.float()
    return module


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def tokenize_call(tok, call, max_tokens, fmt=FORMATS["qwen3.6"]):
    """Returns (ids, weights): ids of the rendered call, and a per-token loss weight (0 outside assistant replies).
    A call marked timing_only (real recorded conversations, where the other side is a person chatting, not an
    assistant) keeps the loss on the first token of each reply only: the decision to listen ("<listen>") or to talk,
    not the person's actual words. A tick's `w1` weighs its first token (the decision to carry on rather than listen)
    on its own, the rest of the reply at `w`."""
    timing_only = bool(call.get("timing_only"))
    # every reply span includes its end marker (<|im_end|>, <turn|>): the model learns where a reply stops
    text, raw = fmt.call(call.get("system") or SYSTEM, [(t["heard"], t["said"]) for t in call["ticks"]])
    spans = [(s, e, float(t.get("w", 1.0)), t.get("w1")) for (s, e), t in zip(raw, call["ticks"])]
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"][:max_tokens]
    offs = enc["offset_mapping"][:max_tokens]
    w = torch.zeros(len(ids))
    si, last_span = 0, -1
    for i, (a, b) in enumerate(offs):
        while si < len(spans) and spans[si][1] <= a:
            si += 1
        if si < len(spans) and spans[si][0] <= a < spans[si][1]:
            if timing_only and si == last_span:
                continue
            w[i] = float(spans[si][3]) if si != last_span and spans[si][3] is not None else spans[si][2]
            last_span = si
    return torch.tensor(ids, dtype=torch.long), w


# ----------------------------------------------------------------------------------------------------------------------
# model pieces
# ----------------------------------------------------------------------------------------------------------------------
def _no_sdpa_gqa():
    """Make transformers expand the 2 key/value heads to 16 before attention. Passed to PyTorch as grouped-query
    attention, no fused kernel accepts them (this Windows build has no FlashAttention), so it falls back to the math
    kernel and builds the full T x T score matrix: ~8 GB at 8,000 tokens, which overflowed 24 GB in backward. Expanded,
    the memory-efficient kernel takes them in tiles (well under 1 GB)."""
    import transformers.integrations.sdpa_attention as sd
    sd.use_gqa_in_sdpa = lambda *args, **kwargs: False


def build_pieces(hf_dir, device="cuda"):
    """(config, text config, the family's profile with its layers/rotary/masks)"""
    _no_sdpa_gqa()
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(hf_dir)
    tcfg = cfg.get_text_config()  # the text part of a multimodal config (Qwen, Gemma, Ministral); a text-only one (gpt-oss) itself
    tcfg._attn_implementation = "sdpa"
    # Mixture-of-experts layers (Qwen3.6, Gemma 4 26B-A4B): transformers' default runs the experts one at a time; the
    # grouped matmul kernel computes the same sums in one call. One 26B-A4B layer at 3,000 tokens: forward 142 -> 23 ms,
    # forward + backward 1,850 -> 100 ms; outputs equal to bf16 rounding, input gradients cosine 0.99997
    # (check_experts.py). Training results are unchanged; it is only faster.
    if getattr(tcfg, "num_experts", None) and hasattr(torch, "_grouped_mm"):
        tcfg._experts_implementation = "grouped_mm"
    return cfg, tcfg, TRAIN[model_family(cfg.to_dict())](tcfg, device)


def check_cache_layers(cache, first_trained, top):
    """A cache holds the input of one layer, the first trained one, and layers.json says which. A cache from before
    it was written was made for the last 4 layers, the only setting then. Another --top needs its own cache."""
    marker = cache / "layers.json"
    if marker.exists():
        ok = json.loads(marker.read_text())["first_trained"] == first_trained
    else:
        ok = top == 4 or not any(cache.glob("call*.pt"))
    if not ok:
        sys.exit(f"{cache} was made for training other layers than the last {top}: that needs its own cache (a new run)")
    marker.write_text(json.dumps({"first_trained": first_trained}))


# ----------------------------------------------------------------------------------------------------------------------
# phase A: cache the input of the first trained layer for every call
# ----------------------------------------------------------------------------------------------------------------------
def phase_a(a, tcfg, P, shards, tok, calls, device):
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    first_trained = tcfg.num_hidden_layers - a.top
    check_cache_layers(cache, first_trained, a.top)
    todo = [i for i, c in enumerate(calls) if not (cache / f"call{i:05d}.pt").exists()]
    if not todo:
        print("phase A: cache complete")
        return
    print(f"phase A: caching {len(todo)} calls through layers 0..{first_trained - 1} in chunks of {a.chunk}")
    fmt = FORMATS[P.family]
    for ci in range(0, len(todo), a.chunk):
        chunk = todo[ci:ci + a.chunk]
        tc = time.time()
        hidden = {}
        weights = {}
        with torch.device("meta"):
            embed = P.embed()
        embed = load_module(embed, shards, PREFIX + "embed_tokens.", device)
        for i in chunk:
            ids, w = tokenize_call(tok, calls[i], a.max_tokens, fmt)
            with torch.no_grad():
                hidden[i] = embed(ids.to(device)[None]).to("cpu", copy=True).pin_memory()
            weights[i] = (ids, w)
        del embed
        ntok = sum(len(weights[i][0]) for i in chunk)
        ctxs = {}  # positions and masks per sequence length, shared by the calls of that length
        print(f" chunk {ci // a.chunk + 1}/{(len(todo) + a.chunk - 1) // a.chunk}: calls {chunk[0]}..{chunk[-1]}, "
              f"{ntok} tokens", flush=True)
        for li in range(first_trained):
            t0 = time.time()
            with torch.device("meta"):
                layer = P.layer(li)
            layer = load_module(layer, shards, f"{PREFIX}layers.{li}.", device).eval()
            with torch.no_grad():
                for i in chunk:
                    h = hidden[i].to(device, non_blocking=True)
                    h = P.run(layer, h, li, ctxs.setdefault(h.shape[1], P.context(h.shape[1], device)))
                    hidden[i].copy_(h.to("cpu"))
            del layer
            gc.collect()
            torch.cuda.empty_cache()
            kind = (getattr(tcfg, "layer_types", None) or ["full_attention"] * tcfg.num_hidden_layers)[li]  # Ministral: all full
            print(f"  layer {li:2d} ({kind:<16}) {time.time() - t0:6.1f} s", flush=True)
        for i in chunk:
            ids, w = weights[i]
            # written under a temporary name and renamed, so a stopped run never leaves a cut-off file that looks done
            tmp = cache / f"call{i:05d}.pt.tmp"
            torch.save({"hidden": hidden[i][0].clone(), "ids": ids, "w": w}, tmp)
            os.replace(tmp, cache / f"call{i:05d}.pt")
        del hidden, weights, ctxs
        gc.collect()
        print(f" chunk done in {(time.time() - tc) / 60:.1f} min", flush=True)
    print("phase A: done")


# ----------------------------------------------------------------------------------------------------------------------
# phase B: LoRA on the top layers
# ----------------------------------------------------------------------------------------------------------------------
class Top(torch.nn.Module):
    def __init__(self, tcfg, P, shards, first, device):
        super().__init__()
        self.tcfg = tcfg
        self.first = first
        self.P = P
        layers = []
        for li in range(first, tcfg.num_hidden_layers):
            with torch.device("meta"):
                layer = P.layer(li)
            layers.append(load_module(layer, shards, f"{PREFIX}layers.{li}.", device))
        self.layers = torch.nn.ModuleList(layers)
        with torch.device("meta"):
            self.norm = P.norm()
            self.lm_head = torch.nn.Linear(tcfg.hidden_size, tcfg.vocab_size, bias=False)
        load_module(self.norm, shards, PREFIX + "norm.", device)
        # Gemma ties the output layer to the input embedding: load that tensor under the lm_head's name
        load_module(self.lm_head, shards, P.lm_head_key, device, rename={"weight": "weight"})

    def hidden(self, h):
        # Gradient checkpointing per layer: only each layer's input is kept, its activations are recomputed during
        # backward. Without it an 8,000-token call through four MoE/delta-net layers overflows 24 GB and Windows
        # spills into system memory (~250 s per step instead of seconds).
        ctx = self.P.context(h.shape[1], h.device)
        for k, layer in enumerate(self.layers):
            li = self.first + k
            if torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(self.P.run, layer, h, li, ctx, use_reentrant=False)
            else:
                h = self.P.run(layer, h, li, ctx)
        return self.norm(h)


def weighted_ce(top, h, ids, w, chunk=1024):
    """Weighted next-token cross-entropy computed in chunks so the full logits are never materialised."""
    T = h.shape[1]
    tgt = ids[1:]
    wt = w[1:]
    total = h.new_zeros((), dtype=torch.float32)
    for s in range(0, T - 1, chunk):
        e = min(T - 1, s + chunk)
        ws = wt[s:e]
        if ws.sum() == 0:
            continue

        # targets and weights are bound now: checkpoint re-runs f during backward, after s and e have moved on
        def f(hs, t=tgt[s:e], ww=ws):
            logits = top.P.logits(top.lm_head, hs)  # Gemma: soft-capped as the model does
            ce = F.cross_entropy(logits.view(-1, logits.shape[-1]), t, reduction="none")
            return (ce * ww).sum()

        total = total + torch.utils.checkpoint.checkpoint(f, h[0, s:e], use_reentrant=False)
    return total / wt.sum().clamp(min=1e-6)


def rename_for_export(state, first):
    """PEFT names inside Top -> names llama.cpp's converter expects (base_model.model.<full HF path>)."""
    out = {}
    for k, v in state.items():
        # k like: base_model.model.layers.0.self_attn.q_proj.lora_A.weight
        k2 = k.replace("base_model.model.", "", 1)
        if k2.startswith("layers."):
            parts = k2.split(".")
            parts[1] = str(first + int(parts[1]))
            k2 = EXPORT_PREFIX + ".".join(parts)
        elif k2.startswith("lm_head.") or k2.startswith("norm."):
            k2 = (EXPORT_PREFIX if k2.startswith("norm.") else "") + k2
        out["base_model.model." + k2] = v.contiguous()
    return out


def parse_weight_map(text):
    """ "4.0:8.0,3.0:4.5" -> {4.0: 8.0, 3.0: 4.5}. Tick weights identify tick tags (see common.WEIGHTS), so this
    re-weights tags in an existing cache without recomputing it. 4.0 = hazard_start, 6.0 = start, 3.0 = barge_stop."""
    out = {}
    for part in (text or "").split(","):
        if part.strip():
            k, v = part.split(":")
            out[float(k)] = float(v)
    return out


def remap_weights(w, wmap):
    w = w.clone()
    for old, new in wmap.items():
        w[(w - old).abs() < 1e-4] = new
    return w


EXPERTS_FILE = "experts_lora.safetensors"  # ExpertLoRA's tensors (gpt-oss --experts), beside PEFT's file


def expert_loras(peft_top, first):
    """{name in the adapter file: tensor} of the ExpertLoRA modules in the trained layers (empty without --experts)."""
    out = {}
    for k, layer in enumerate(peft_top.base_model.model.layers):
        m = getattr(getattr(layer, "mlp", None), "experts", None)
        for n in getattr(m, "NAMES", ()):
            for ab in "AB":
                out[f"{EXPORT_PREFIX}layers.{first + k}.mlp.experts.{n}.lora_{ab}"] = getattr(m, f"{n}_{ab}")
    return out


def load_adapter(peft_top, adapter_dir, first):
    """Continue from an adapter saved by save_adapter (inverse of rename_for_export)."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    mine, f = expert_loras(peft_top, first), Path(adapter_dir) / EXPERTS_FILE
    if mine or f.exists():
        theirs = load_file(str(f)) if f.exists() else {}
        if set(theirs) != set(mine):
            raise RuntimeError(f"{adapter_dir}: expert adapters do not match this run's (--experts on one side only?)")
        with torch.no_grad():
            for n, p in mine.items():
                p.copy_(theirs[n])
    sd = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    inner = {}
    for k, v in sd.items():
        k2 = k.replace("base_model.model.", "", 1)
        if k2.startswith(EXPORT_PREFIX + "layers."):
            parts = k2[len(EXPORT_PREFIX):].split(".")
            parts[1] = str(int(parts[1]) - first)
            k2 = ".".join(parts)
        elif k2.startswith(EXPORT_PREFIX + "norm."):
            k2 = k2[len(EXPORT_PREFIX):]
        inner["base_model.model." + k2] = v
    res = set_peft_model_state_dict(peft_top, inner)
    unexpected = getattr(res, "unexpected_keys", []) or []
    missing = [m for m in (getattr(res, "missing_keys", []) or []) if "lora_" in m]
    if unexpected or missing:
        raise RuntimeError(f"init adapter mismatch: missing {missing[:3]} unexpected {unexpected[:3]}")
    print(f"continuing from {adapter_dir} ({len(inner)} tensors)")


def phase_b(a, tcfg, P, shards, calls, device):
    from peft import LoraConfig, get_peft_model
    first = tcfg.num_hidden_layers - a.top
    print(f"phase B: training LoRA rank {a.rank} on layers {first}..{tcfg.num_hidden_layers - 1}")
    top = Top(tcfg, P, shards, first, device)
    shards.close()
    for p in top.parameters():
        p.requires_grad_(False)
    if a.base_adapter:
        # --base-adapter: fold an existing adapter into these layers, frozen, and train a new one on top of it: an add-on
        # that only learns what the base adapter lacks (the app plays the two together at scale 1 each, which is the same
        # thing). The timing adapter keeps the timing; the add-on carries the persona and memory behaviour.
        bcfg = json.load(open(Path(a.base_adapter) / "adapter_config.json"))
        btargets = bcfg["target_modules"]
        bpeft = get_peft_model(top, LoraConfig(r=bcfg["r"], lora_alpha=bcfg["lora_alpha"], lora_dropout=0.0, target_modules=btargets, bias="none"))
        load_adapter(bpeft, a.base_adapter, first)
        top = bpeft.merge_and_unload()
        for p in top.parameters():
            p.requires_grad_(False)
        print(f"base adapter {a.base_adapter} folded in (frozen); training an add-on on top of it")
    targets = [t.strip() for t in (a.targets or P.targets).split(",") if t.strip()]
    lcfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, target_modules=targets, bias="none")
    peft_top = get_peft_model(top, lcfg)
    if a.experts:  # after PEFT, which would freeze them
        if not hasattr(P, "add_expert_lora"):
            sys.exit(f"--experts is for gpt-oss; {P.family} has its experts' adapters in --targets or none")
        for layer in peft_top.base_model.model.layers:
            P.add_expert_lora(layer, a.rank, a.alpha)
    peft_top.print_trainable_parameters()
    if a.init_adapter:
        load_adapter(peft_top, a.init_adapter, first)
    # --resume: continue a stopped run from its last saved adapter (saved every --save-every steps). The batch order is
    # replayed from the seed, so the run sees the same calls in the same order as if it had never stopped; the
    # optimizer's moment estimates restart from zero (train_state.pt keeps them when the run saved one).
    start = 0
    out_dir = Path(a.out)
    if a.resume and (out_dir / "adapter_model.safetensors").exists():
        log_file = out_dir.with_suffix(".log")
        done = [int(m.group(1)) for m in re.finditer(r"^step (\d+)/", log_file.read_text(), re.M)] if log_file.exists() else []
        start = (max(done) // a.save_every) * a.save_every if done else 0
        if start:
            load_adapter(peft_top, str(out_dir), first)
            print(f"resuming at step {start} of {a.steps}")
    params = [p for p in peft_top.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    def lr_factor(s):
        warm = min(1.0, (s + 1) / max(1, a.warmup))
        if a.lr_decay != "cosine" or s < a.warmup:
            return warm
        # cosine from the full rate down to lr_min_ratio of it over the steps after warm-up
        p = (s - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr_min_ratio + (1 - a.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    cache = Path(a.cache)
    files = sorted(cache.glob("call*.pt"))
    if a.heldout:
        files = [f for f in files if int(f.stem[4:]) < len(calls)]
    if a.fresh_w is not None:  # calls this recipe weighs 0 (an add-on borrowing a full run's cache) take no batch places
        files = [f for f in files if bool((a.fresh_w[int(f.stem[4:])][1] > 0).any())]
    # --val-calls: that many calls (picked by a fixed seed, the same on a resume) are never trained on; their loss every
    # --val-every steps shows whether more steps still help or the adapter has started to learn the training calls by heart
    val = sorted(random.Random(4242).sample(files, min(a.val_calls, len(files) // 10))) if a.val_calls else []
    files = [f for f in files if f not in set(val)]
    rng = random.Random(a.seed)
    for _ in range(start):  # replay the batch order up to the resume point
        rng.sample(files, min(a.batch, len(files)))
    state_file = out_dir / "train_state.pt"
    if start and state_file.exists():
        st = torch.load(state_file)
        if st.get("step") == start:
            opt.load_state_dict(st["opt"])
            print("  optimizer state restored")
    for _ in range(start):
        sched.step()
    wmap = parse_weight_map(a.weight_map)
    if wmap:
        print(f"loss weights remapped: {wmap}")
    log = open(Path(a.out).with_suffix(".log"), "a")
    t0 = time.time()
    for step in range(start, a.steps):
        batch = rng.sample(files, min(a.batch, len(files)))
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for f in batch:
            d = torch.load(f)
            h = d["hidden"][None].to(device, torch.bfloat16)
            if a.fresh_w is not None:  # --weights-from-calls: this call's weights as the calls file has them now
                ids0, w0 = a.fresh_w[int(f.stem[4:])]
                if not torch.equal(ids0, d["ids"]):
                    raise SystemExit(f"{f.name}: the calls file no longer matches this cache (different tokens)")
                d["w"] = w0
            ids, w = d["ids"].to(device), d["w"].to(device)
            if not bool((w > 0).any()):  # nothing to learn here (a piece of a persona call cut off from its memory moment)
                continue
            if wmap:
                w = remap_weights(w, wmap)
            # base weights are bf16, LoRA weights float32: peft casts the LoRA input to float32 and the result back
            hs = peft_top.base_model.model.hidden(h)
            loss = weighted_ce(peft_top.base_model.model, hs, ids, w) / len(batch)
            loss.backward()
            loss_sum += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        el = time.time() - t0
        msg = f"step {step + 1}/{a.steps} loss {loss_sum:.4f} lr {sched.get_last_lr()[0]:.2e} {el / (step + 1 - start):.1f} s/step"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
        if val and ((step + 1) % a.val_every == 0 or step + 1 == a.steps):
            vl = validation_loss(peft_top, val, a, device)
            vmsg = f"val {step + 1}/{a.steps} loss {vl:.4f} over {len(val)} held-out calls"
            print(vmsg, flush=True)
            log.write(vmsg + "\n")
            log.flush()
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            save_adapter(peft_top, lcfg, first, a)
            torch.save({"step": step + 1, "opt": opt.state_dict()}, out_dir / "train_state.pt")
    print("phase B: done")


def validation_loss(peft_top, val, a, device):
    """The weighted loss on the held-out calls (mean per call), without training on them."""
    total, n = 0.0, 0
    with torch.no_grad():
        for f in val:
            d = torch.load(f)
            w = a.fresh_w[int(f.stem[4:])][1] if a.fresh_w is not None else d["w"]
            if not bool((w > 0).any()):
                continue
            hs = peft_top.base_model.model.hidden(d["hidden"][None].to(device, torch.bfloat16))
            total += float(weighted_ce(peft_top.base_model.model, hs, d["ids"].to(device), w.to(device)))
            n += 1
    return total / max(1, n)


def save_adapter(peft_top, lcfg, first, a):
    from peft import get_peft_model_state_dict
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().to(torch.float32).cpu() for k, v in get_peft_model_state_dict(peft_top).items()}
    save_file(rename_for_export(sd, first), str(out / "adapter_model.safetensors"))
    experts = {k: v.detach().to(torch.float32).cpu().contiguous() for k, v in expert_loras(peft_top, first).items()}
    if experts:
        save_file(experts, str(out / EXPERTS_FILE))
    conf = {
        "peft_type": "LORA", "r": lcfg.r, "lora_alpha": lcfg.lora_alpha, "lora_dropout": lcfg.lora_dropout,
        "target_modules": list(lcfg.target_modules), "bias": "none", "task_type": "CAUSAL_LM",
        "base_model_name_or_path": str(a.hf), "talkme": {"top": a.top, "steps": a.steps, "rank": a.rank, "experts": a.experts},
    }
    json.dump(conf, open(out / "adapter_config.json", "w"), indent=1)
    print(f"  saved adapter to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="HF checkpoint dir (bf16 safetensors): Qwen3.6-35B-A3B or Gemma 4 31B")
    ap.add_argument("--calls", required=True)
    ap.add_argument("--heldout", default=None)
    ap.add_argument("--out", default="out/adapter")
    ap.add_argument("--cache", default="out/cache")
    ap.add_argument("--top", type=int, default=4, help="train the last N layers")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--targets", default=None, help="LoRA target modules (default: the model family's, see profiles.py)")
    ap.add_argument("--experts", action="store_true", help="gpt-oss: LoRA on the experts too, shared by a layer's experts (profiles.ExpertLoRA)")
    ap.add_argument("--steps", type=int, default=282)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--lr-decay", choices=["none", "cosine"], default="none", help="cosine: the rate falls to --lr-min-ratio of itself by the last step")
    ap.add_argument("--lr-min-ratio", type=float, default=0.1)
    ap.add_argument("--val-calls", type=int, default=0, help="calls held out of training; their loss is logged every --val-every steps")
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--chunk", type=int, default=64, help="phase A: calls cached per pass over the lower layers")
    ap.add_argument("--max-calls", type=int, default=0, help="use only the first N calls (for a quick test)")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-cache", action="store_true")
    ap.add_argument("--init-adapter", default=None, help="continue training from this adapter dir (saved by this script)")
    ap.add_argument("--base-adapter", default=None, help="fold this adapter dir in, frozen, and train a new adapter on top (an add-on)")
    ap.add_argument("--weight-map", default="", help="re-weight tick tags in the cache, e.g. '4.0:8.0' doubles hazard_start")
    ap.add_argument("--weights-from-calls", action="store_true",
                    help="train with the loss weights of --calls instead of the cached ones: another run's cache serves calls that "
                         "differ only in weights (the tokens are checked to match)")
    ap.add_argument("--skip-cache", action="store_true")
    ap.add_argument("--system", choices=sorted(SYSTEMS), default="assistant", help="system prompt every call is trained with (partner: run 9 on)")
    ap.add_argument("--resume", action="store_true", help="continue a stopped run from its last saved adapter in --out")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda"
    from transformers import AutoTokenizer
    cfg, tcfg, P = build_pieces(a.hf, device)
    tok = AutoTokenizer.from_pretrained(a.hf, **getattr(P, "tokenizer_kw", {}))
    shards = Shards(a.hf)
    global EXPORT_PREFIX  # the adapter's tensors are named as the checkpoint names its layers, for llama.cpp's converter
    if any(n.startswith("model.layers.") for n in shards.real.values()):
        EXPORT_PREFIX = "model."
    calls = read_jsonl(a.calls)
    for c in calls:
        c.setdefault("system", SYSTEMS[a.system])
    if a.max_calls:
        calls = calls[: a.max_calls]
    print(f"{len(calls)} calls; {P.family}; vocab {tcfg.vocab_size}; layers {tcfg.num_hidden_layers}; free VRAM "
          f"{torch.cuda.mem_get_info()[0] / 2**30:.1f} GB")
    if not a.skip_cache:
        phase_a(a, tcfg, P, shards, tok, calls, device)
    if a.only_cache:
        return
    a.fresh_w = None
    if a.weights_from_calls:
        fmt = FORMATS[P.family]
        a.fresh_w = {i: tokenize_call(tok, c, a.max_tokens, fmt) for i, c in enumerate(calls)}
        print(f"loss weights from {a.calls} ({len(calls)} calls)")
    phase_b(a, tcfg, P, shards, calls, device)


if __name__ == "__main__":
    main()
