"""Orpheus 3B (Canopy Labs, Apache 2.0), Q8_0 GGUF on our llama.cpp (port 8090) plus the SNAC 24 kHz decoder.
Voice "tara"; <laugh>/<sigh> tags. Streams: audio is decoded every 28 code tokens as they arrive.
   bash ../run_all.sh starts the server;   cd tts_bench/orpheus && uv run python ../run_orpheus.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEVICE, LINES, MODELS, Bench, text_for

b = Bench("orpheus-3b", "voice tara, <laugh>/<sigh> tags, Q8_0 on llama.cpp, streamed in 28-token windows")
import numpy as np
import requests
import torch
from snac import SNAC

URL = "http://127.0.0.1:8090"
snac = SNAC.from_pretrained(str(MODELS / "snac_24khz")).eval().to(DEVICE)
b.sr = 24000
tags = {"laugh": "<laugh>", "sigh": "<sigh>"}
# Orpheus's prompt: start-of-human, the text, end-of-text, end-of-human (canopylabs' reference code, as token ids)
START, END = [128259, 128000], [128009, 128260]
CODE0 = 128266  # first audio code token; each of the 7 codes in a frame is offset by a further 4096 * position


def tokenize(text):
    return requests.post(URL + "/tokenize", json={"content": text, "add_special": False}).json()["tokens"]


def decode(codes):
    """codes: a multiple of 7 audio codes -> float32 audio (SNAC's 3 layers: 1, 2 and 4 codes per frame)."""
    f = np.array(codes).reshape(-1, 7)
    l1 = f[:, 0]
    l2 = f[:, [1, 4]].reshape(-1)
    l3 = f[:, [2, 3, 5, 6]].reshape(-1)
    with torch.inference_mode():
        a = snac.decode([torch.tensor(x, device=DEVICE).unsqueeze(0) for x in (l1, l2, l3)])
    return a.squeeze().float().cpu().numpy()


def synth(text):
    body = {"prompt": START + tokenize(f"tara: {text}") + END, "n_predict": 1600, "temperature": 0.6, "top_p": 0.9,
            "repeat_penalty": 1.1, "stream": True, "return_tokens": True, "cache_prompt": False}
    codes, done = [], 0
    with requests.post(URL + "/completion", json=body, stream=True) as r:
        for raw in r.iter_lines():
            if not raw.startswith(b"data: "):
                continue
            j = json.loads(raw[6:])
            for t in j.get("tokens", []):
                if t >= CODE0:
                    codes.append((t - CODE0) - 4096 * (len(codes) % 7))
            whole = len(codes) // 7 * 7
            # every 4 new frames, decode what has arrived (the first window as soon as it exists: first audio)
            if whole - done >= 28:
                yield decode(codes[done:whole])
                done = whole
            if j.get("stop"):
                break
    whole = len(codes) // 7 * 7
    if whole > done:
        yield decode(codes[done:whole])


list(synth("Warming up."))
b.loaded()
for line in LINES:
    b.run(line, "whole", text_for(line, tags), synth)
    b.chunked(line, text_for(line), synth)
b.finish()
