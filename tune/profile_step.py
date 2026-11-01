"""Profiles one training call through the top layers: time and peak GPU memory of forward, loss and backward, per
layer type, so phase B speed problems can be pinned down.
    uv run --extra train python profile_step.py --hf ../models/hf/Qwen3.6-35B-A3B --cache out/cache_pub
"""
import argparse
import time
from pathlib import Path

import torch

import train_lora as T


def mem():
    return torch.cuda.max_memory_allocated() / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tokens", type=int, default=8000)
    ap.add_argument("--no-ckpt", action="store_true")
    a = ap.parse_args()
    cfg, tcfg, M = T.build_pieces(a.hf)
    shards = T.Shards(a.hf)
    dev = "cuda"
    top = T.Top(tcfg, M, shards, tcfg.num_hidden_layers - 4, dev)
    for p in top.parameters():
        p.requires_grad_(False)
    from peft import LoraConfig, get_peft_model
    pt = get_peft_model(top, LoraConfig(r=32, lora_alpha=64, target_modules=T.DEFAULT_TARGETS.split(","), bias="none"))
    for p in pt.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    print(f"weights loaded: {torch.cuda.memory_allocated() / 2**30:.2f} GB allocated")
    f = sorted(Path(a.cache).glob("call*.pt"))[0]
    d = torch.load(f)
    h = d["hidden"][None, : a.tokens].to(dev, torch.bfloat16)
    ids, w = d["ids"][: a.tokens].to(dev), d["w"][: a.tokens].to(dev)
    print(f"call {f.name}: {h.shape[1]} tokens")
    model = pt.base_model.model
    # per-layer forward timing without grad
    with torch.no_grad():
        x = h
        for k, layer in enumerate(model.layers):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
            x = T.run_layer(layer, x, model.rotary, tcfg.layer_types[model.first + k])
            torch.cuda.synchronize()
            print(f"  no-grad forward layer {model.first + k} ({tcfg.layer_types[model.first + k]}): {time.time() - t0:.2f} s, peak {mem():.2f} GB")
    for rep in range(2):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        if a.no_ckpt:
            x = h
            for k, layer in enumerate(model.layers):
                x = T.run_layer(layer, x, model.rotary, tcfg.layer_types[model.first + k])
            hs = model.norm(x)
        else:
            hs = model.hidden(h)
        torch.cuda.synchronize(); t1 = time.time(); m1 = mem()
        loss = T.weighted_ce(model, hs, ids, w)
        torch.cuda.synchronize(); t2 = time.time(); m2 = mem()
        loss.backward()
        torch.cuda.synchronize(); t3 = time.time(); m3 = mem()
        print(f"pass {rep + 1}: forward {t1 - t0:.2f} s (peak {m1:.2f} GB) | loss {t2 - t1:.2f} s (peak {m2:.2f} GB) | "
              f"backward {t3 - t2:.2f} s (peak {m3:.2f} GB) | loss {loss.item():.4f}")
        pt.zero_grad(set_to_none=True)


if __name__ == "__main__":
    main()
