"""The same four lines from several voices of one engine, for picking a voice by ear. Writes
out/voices/<engine>-<voice>/<n>.wav and out/voices/<engine>.json.
  cd tts_bench/chatterbox && uv run python ../sample_voices.py chatterbox
  cd tts_bench/orpheus && uv run python ../sample_voices.py orpheus      (needs Orpheus's llama-server on 8090)"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "models" / "tts_bench"
OUT = HERE / "out" / "voices"
sys.path.insert(0, str(HERE))
from sample_voices_lines import LINES
from serve_common import trim

engine = sys.argv[1]
results = []
speed = []  # (seconds of audio, seconds to make it) per line
if engine in ("chatterbox", "chatterbox-original"):
    import time
    import torch
    if engine == "chatterbox":
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        model = ChatterboxTurboTTS.from_local(str(MODELS / "chatterbox-turbo"), device="cuda")
    else:  # the original 500M: slower, with an emotion dial (exaggeration, default 0.5)
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_local(str(MODELS / "chatterbox-original"), device="cuda")
    sr = model.sr
    model.generate("Warming up.")
    refs = [("built-in", None)] + [(f.stem, f) for f in sorted((MODELS / "chatterbox-voices").glob("*.wav"))]
    default = model.conds
    for name, ref in refs:
        if ref:
            model.prepare_conditionals(str(ref))
        else:
            model.conds = default
        for i, line in enumerate(LINES):
            t0 = time.time()
            with torch.inference_mode():
                a = model.generate(line).squeeze(0).float().cpu().numpy()
            speed.append((len(a) / sr, time.time() - t0))
            a = trim(a, sr)
            d = OUT / f"{engine}-{name}"
            d.mkdir(parents=True, exist_ok=True)
            sf.write(d / f"{i}.wav", a, sr)
        results.append({"voice": name, "dir": f"{engine}-{name}"})
        print(engine, name, flush=True)
else:
    import requests
    import torch
    from snac import SNAC
    snac = SNAC.from_pretrained(str(MODELS / "snac_24khz")).eval().to("cuda")
    sr = 24000
    URL = "http://127.0.0.1:8090"

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
        return trim(a.squeeze().float().cpu().numpy(), sr)

    for voice in ["tara", "leah", "jess", "mia", "zoe", "leo", "dan", "zac"]:
        d = OUT / f"orpheus-{voice}"
        d.mkdir(parents=True, exist_ok=True)
        for i, line in enumerate(LINES):
            sf.write(d / f"{i}.wav", say(voice, line), sr)
        results.append({"voice": voice, "dir": f"orpheus-{voice}"})
        print("orpheus", voice, flush=True)
rt = sum(x for x, _ in speed) / max(1e-9, sum(y for _, y in speed)) if speed else None
if rt:
    print(f"{engine}: {rt:.2f}x real time over {len(speed)} lines (whole lines)", flush=True)
(OUT / f"{engine}.json").write_text(json.dumps({"engine": engine, "lines": LINES, "voices": results, "realtime_x": rt}, indent=1), encoding="utf-8")
