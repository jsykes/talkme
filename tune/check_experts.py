"""Times one mixture-of-experts layer with each experts kernel transformers offers, and checks they agree: forward on a
call-sized input (as phase A caches), and forward + backward (as phase B trains). The kernels only reorder the same
sums, so the outputs should match to bf16 rounding.
    uv run --extra train python check_experts.py --hf ../models/hf/gemma-4-26B-A4B-it-heretic --layer 12"""
import argparse
import time

import torch

from train_lora import PREFIX, Shards, build_pieces, load_module

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True)
ap.add_argument("--layer", type=int, default=12)
ap.add_argument("--tokens", type=int, default=6000)
ap.add_argument("--impls", default="eager,grouped_mm,batched_mm")
a = ap.parse_args()

device = "cuda"
cfg, tcfg, P = build_pieces(a.hf, device)
shards = Shards(a.hf)
with torch.device("meta"):
    layer = P.layer(a.layer)
layer = load_module(layer, shards, f"{PREFIX}layers.{a.layer}.", device).eval()
torch.manual_seed(0)
x = torch.randn(1, a.tokens, tcfg.hidden_size, device=device, dtype=torch.bfloat16)
ctx = P.context(a.tokens, device)
ref = None
for impl in a.impls.split(","):
    tcfg._experts_implementation = None if impl == "eager" else impl
    try:
        with torch.no_grad():
            P.run(layer, x, a.layer, ctx)  # warm-up
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(3):
                y = P.run(layer, x, a.layer, ctx)
            torch.cuda.synchronize()
            fwd = (time.time() - t0) / 3
        xg = x.clone().requires_grad_(True)
        torch.cuda.synchronize()
        t0 = time.time()
        P.run(layer, xg, a.layer, ctx).float().pow(2).mean().backward()
        torch.cuda.synchronize()
        fb = time.time() - t0
        g = xg.grad.float()
        if impl == "eager":
            gref = g
        else:  # the gradient training relies on, not just the output
            cos = float(torch.nn.functional.cosine_similarity(g.flatten(), gref.flatten(), dim=0))
            print(f"{'':11} gradient vs eager: cosine {cos:.6f}, max diff {float((g - gref).abs().max()):.2e} (scale {float(gref.abs().max()):.2e})", flush=True)
        diff = "" if ref is None else f", max diff vs eager {float((y.float() - ref).abs().max()):.4f} (output scale {float(ref.abs().max()):.1f})"
        if ref is None:
            ref = y.float()
        print(f"{impl:11} forward {fwd * 1000:7.0f} ms, forward+backward {fb * 1000:7.0f} ms{diff}", flush=True)
    except Exception as e:
        print(f"{impl:11} failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
