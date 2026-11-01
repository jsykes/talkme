"""Chatterbox Turbo as an app voice (see serve_common.py). Started by the app; by hand:
   cd tts_bench/chatterbox && uv run python ../serve_chatterbox.py --port 8091 [--device cuda|cpu]"""
import argparse
import contextlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from serve_common import serve

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8091)
ap.add_argument("--device", default="cuda")
ap.add_argument("--model", default=str(HERE.parent / "models" / "tts_bench" / "chatterbox-turbo"))
ap.add_argument("--fp32", action="store_true", help="the speech model at full precision (default: 16-bit on the GPU)")
a = ap.parse_args()

import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS

model = ChatterboxTurboTTS.from_local(a.model, device=a.device)
# The speech model (T3) at 16-bit on the GPU: 2.6 GB at peak instead of 3.3, 0.31x real time instead of 0.24, and the
# same voice by ear (2026-10-28, docs/RESULTS.md "A lighter Chatterbox"). The library mixes 32- and 16-bit tensors, so
# everything runs under autocast; the rest of the model (S3Gen, the voice encoder) stays at full precision.
HALF = a.device == "cuda" and not a.fp32
if HALF:
    model.t3.half()


def precision():
    return torch.autocast("cuda", dtype=torch.float16) if HALF else contextlib.nullcontext()


def synth(text):
    with torch.inference_mode(), precision():
        return model.generate(text).squeeze(0).float().cpu().numpy()


BUILT_IN = model.conds
cache = {}  # reference clip -> its conditionals (a voice is prepared once, then switching is instant)


def set_voice(req):
    """{"ref": a clip to clone (5-15 s of clear speech), or null for the built-in voice}"""
    ref = req.get("ref")
    if not ref:
        model.conds = BUILT_IN
        return
    if ref not in cache:
        with precision():
            model.prepare_conditionals(ref)
        cache[ref] = model.conds
    model.conds = cache[ref]


synth("Warming up.")
serve("chatterbox", model.sr, synth, a.port, set_voice=set_voice)
