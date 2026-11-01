"""Chatterbox Turbo (Resemble AI, 350M, MIT): its built-in voice; [laugh]/[sigh] tags. No streaming: first audio =
the whole line.   cd tts_bench/chatterbox && uv run python ../run_chatterbox.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEVICE, LINES, MODELS, Bench, text_for

b = Bench("chatterbox-turbo", "built-in voice, [laugh]/[sigh] tags, no streaming")
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS

model = ChatterboxTurboTTS.from_local(str(MODELS / "chatterbox-turbo"), device=DEVICE)
b.sr = model.sr
tags = {"laugh": "[laugh]", "sigh": "[sigh]"}


def synth(text):
    with torch.inference_mode():
        yield model.generate(text).squeeze(0).float().cpu().numpy()


list(synth("Warming up."))
b.loaded()
for line in LINES:
    b.run(line, "whole", text_for(line, tags), synth)
    b.chunked(line, text_for(line), synth)
b.finish()
