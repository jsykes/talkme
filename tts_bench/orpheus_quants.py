"""Orpheus at Q8_0, Q6_K and Q4_K_M: the same four lines from Tara and Leo, with generation speed per quant.
Starts and stops its own llama-server on 8090 for each file. Writes out/voices/orpheus-<quant>-<speaker>/ and
out/voices/orpheus-quants.json; voices_page.py shows them.
   cd tts_bench/orpheus && uv run python ../orpheus_quants.py"""
import os
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import torch
from snac import SNAC

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from sample_voices_lines import LINES
from serve_common import trim

OUT = HERE / "out" / "voices"
URL = "http://127.0.0.1:8090"
snac = SNAC.from_pretrained(str(ROOT / "models" / "tts_bench" / "snac_24khz")).eval().to("cuda")
QUANTS = ["Q8_0", "Q6_K", "Q4_K_M"]
SPEAKERS = ["tara", "leo"]


def say(voice, text):
    toks = requests.post(URL + "/tokenize", json={"content": f"{voice}: {text}", "add_special": False}).json()["tokens"]
    r = requests.post(URL + "/completion", json={"prompt": [128259, 128000] + toks + [128009, 128260], "n_predict": 1600, "temperature": 0.6,
                                               "top_p": 0.9, "repeat_penalty": 1.1, "return_tokens": True, "cache_prompt": False}).json()
    codes = []
    for t in r["tokens"]:
        if t >= 128266:
            codes.append((t - 128266) - 4096 * (len(codes) % 7))
    f = np.array(codes[: len(codes) // 7 * 7]).reshape(-1, 7)
    with torch.inference_mode():
        a = snac.decode([torch.tensor(x, device="cuda").unsqueeze(0) for x in (f[:, 0], f[:, [1, 4]].reshape(-1), f[:, [2, 3, 5, 6]].reshape(-1))])
    return a.squeeze().float().cpu().numpy(), r["timings"]["predicted_per_second"]


results = []
for q in QUANTS:
    gguf = ROOT / "models" / "tts_bench" / "orpheus" / f"orpheus-3b-0.1-ft-{q}.gguf"
    srv = subprocess.Popen([str(ROOT / "vendor" / "llama" / ("llama-server.exe" if os.name == "nt" else "llama-server")), "-m", str(gguf), "-ngl", "99", "-c", "4096", "--port", "8090",
                            "--host", "127.0.0.1", "-np", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                if requests.get(URL + "/health", timeout=2).json().get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(1)
        say("tara", "Warming up.")
        tps, audio_s, wall = [], 0.0, 0.0
        for sp in SPEAKERS:
            d = OUT / f"orpheus-{q.lower()}-{sp}"
            d.mkdir(parents=True, exist_ok=True)
            for i, line in enumerate(LINES):
                t0 = time.time()
                a, rate = say(sp, line)
                wall += time.time() - t0
                audio_s += len(a) / 24000
                tps.append(rate)
                sf.write(d / f"{i}.wav", trim(a, 24000), 24000)
        res = {"quant": q, "size_gb": round(gguf.stat().st_size / 1e9, 2), "tokens_per_s": round(float(np.median(tps)), 1), "realtime_x": round(audio_s / wall, 2)}
        results.append(res)
        print(res, flush=True)
    finally:
        srv.kill()
        srv.wait()
        time.sleep(2)
(OUT / "orpheus-quants.json").write_text(json.dumps({"lines": LINES, "speakers": SPEAKERS, "results": results}, indent=1), encoding="utf-8")
