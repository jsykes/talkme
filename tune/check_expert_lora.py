"""Checks the expert adapters of a gpt-oss adapter (train_lora.py --experts) against llama.cpp. The trainer's top layers
run from a training cache; llama-server runs the same tokens. Both run twice: with the whole adapter, and with only its
attention part (exported here, the adapter without experts_lora.safetensors). The difference between the two is what
the expert adapters do. The base GGUF is quantized, so the log-probabilities differ a little between the two programs,
but that difference has to agree. Use small expert adapters (a trained one, or random ones near zero): large ones push
the model far enough that quantization noise grows through the layers.
    uv run --extra train python check_expert_lora.py --hf ../models/hf/gpt-oss-20b-BF16 --adapter out/adapter_X \
        --gguf out/X.gguf --base-gguf ../models/base/gptoss/gpt-oss-20b-MXFP4.gguf --cache out/cache_X"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests
import torch

import train_lora as T

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True)
ap.add_argument("--adapter", required=True, help="the adapter dir train_lora.py saved")
ap.add_argument("--gguf", required=True, help="the same adapter exported (export_gguf.py)")
ap.add_argument("--base-gguf", required=True)
ap.add_argument("--cache", required=True, help="a training cache made for the adapter's layers")
ap.add_argument("--call", type=int, default=0)
ap.add_argument("--at", default="250,400,550,700,850,1000,1150,1300,1450,1600", help="token counts to compare the next-token distribution after")
ap.add_argument("--device", default="cuda", help="where the trainer's layers run (llama-server runs on the CPU)")
ap.add_argument("--port", type=int, default=8093)
a = ap.parse_args()

# the attention part alone: the adapter dir without its expert file, exported as usual
attn_dir, attn_gguf = Path(a.gguf).with_suffix(".attention"), Path(a.gguf).with_suffix(".attention.gguf")
attn_dir.mkdir(exist_ok=True)
for f in ("adapter_config.json", "adapter_model.safetensors"):
    shutil.copy(Path(a.adapter) / f, attn_dir / f)
subprocess.run([sys.executable, str(Path(__file__).parent / "export_gguf.py"), "--adapter", str(attn_dir), "--hf", a.hf, "--out", str(attn_gguf)],
               check=True, stdout=subprocess.DEVNULL)

conf = json.load(open(Path(a.adapter) / "adapter_config.json"))
cfg, tcfg, P = T.build_pieces(a.hf, a.device)
shards = T.Shards(a.hf)
if any(n.startswith("model.layers.") for n in shards.real.values()):
    T.EXPORT_PREFIX = "model."
first = tcfg.num_hidden_layers - conf["talkme"]["top"]
assert json.loads((Path(a.cache) / "layers.json").read_text())["first_trained"] == first, "the cache is for other layers"

from peft import LoraConfig, get_peft_model
top = T.Top(tcfg, P, shards, first, a.device).eval()
shards.close()
peft = get_peft_model(top, LoraConfig(r=conf["r"], lora_alpha=conf["lora_alpha"], target_modules=conf["target_modules"], bias="none"))
for p in peft.parameters():
    if p.requires_grad:
        p.data = p.data.float()  # as in training: the adapter in float32
experts = [P.add_expert_lora(layer, conf["r"], conf["lora_alpha"]) for layer in peft.base_model.model.layers]
T.load_adapter(peft, a.adapter, first)
d = torch.load(Path(a.cache) / f"call{a.call:05d}.pt")


def trainer(n, with_experts):
    for m in experts:
        m.scale = conf["lora_alpha"] / conf["r"] if with_experts else 0.0
    with torch.no_grad():
        hs = peft.base_model.model.hidden(d["hidden"][:n][None].to(a.device, torch.bfloat16))
        return torch.log_softmax(P.logits(peft.base_model.model.lm_head, hs[0, -1]), -1).cpu()


env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}  # llama-server on the CPU: the trainer's layers have the GPU
log = Path(a.gguf).with_suffix(".check.log")
server = subprocess.Popen([str(ROOT / "vendor" / "llama" / ("llama-server.exe" if os.name == "nt" else "llama-server")), "-m", a.base_gguf, "--lora", a.gguf, "--lora", str(attn_gguf),
                           "--lora-init-without-apply", "--port", str(a.port), "-c", "4096", "-ngl", "0", "--no-warmup"], env=env,
                          stdout=open(log, "w"), stderr=subprocess.STDOUT)
try:
    url = f"http://127.0.0.1:{a.port}"
    for _ in range(300):
        try:
            if requests.get(url + "/health", timeout=2).status_code == 200:
                break
        except requests.RequestException:
            pass
        if server.poll() is not None:
            raise SystemExit(f"llama-server stopped: see {log}")
        time.sleep(1)

    def llama(n, whole):
        r = requests.post(url + "/completion", json={"prompt": d["ids"][:n].tolist(), "n_predict": 1, "n_probs": 10, "temperature": 0, "cache_prompt": False,
                                                     "lora": [{"id": 0, "scale": 1.0 if whole else 0.0}, {"id": 1, "scale": 0.0 if whole else 1.0}]}).json()
        return {t["id"]: t["logprob"] for t in r["completion_probabilities"][0]["top_logprobs"]}

    # on the tokens both of llama.cpp's runs list: how the expert adapters move log p there, in each program
    xs, ys = [], []
    for n in (int(x) for x in a.at.split(",")):
        lw, ln = llama(n, True), llama(n, False)
        tw, tn = trainer(n, True), trainer(n, False)
        both = [i for i in lw if i in ln and max(lw[i], ln[i]) > math.log(0.01)]
        dl = [lw[i] - ln[i] for i in both]
        dt = [float(tw[i] - tn[i]) for i in both]
        xs += dl
        ys += dt
        print(f"after {n:>5} tokens, the expert adapters' change to log p: llama.cpp {' '.join(f'{v:+.2f}' for v in dl)}", flush=True)
        print(f"{'':>34} trainer   {' '.join(f'{v:+.2f}' for v in dt)}", flush=True)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    corr = cov / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys) or 1)
    print(f"\n{len(xs)} tokens: correlation {corr:.3f}; mean |change| llama.cpp {sum(map(abs, xs)) / len(xs):.3f}, "
          f"trainer {sum(map(abs, ys)) / len(ys):.3f}; mean |difference| {sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs):.3f}")
finally:
    server.terminate()
