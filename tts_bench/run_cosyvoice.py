"""CosyVoice 2 0.5B (Alibaba, Apache 2.0): voice cloned from the repo's own English-capable prompt clip, told the
mood of each line in words ("instruct2"), streamed out. Laugh and breath marks are its own: [laughter], [breath].
   cd tts_bench/cosyvoice && uv run python ../run_cosyvoice.py"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "cosyvoice" / "CosyVoice"), str(HERE / "cosyvoice" / "CosyVoice" / "third_party" / "Matcha-TTS")]
from common import DEVICE, LINES, MODELS, Bench, text_for

b = Bench("cosyvoice2", "voice from CosyVoice's cross_lingual_prompt.wav, mood told in words, [laughter]/[breath] marks, streamed")
from cosyvoice.cli.cosyvoice import CosyVoice2

cv = CosyVoice2(str(MODELS / "CosyVoice2-0.5B"), load_jit=False, load_trt=False, fp16=DEVICE == "cuda")
b.sr = cv.sample_rate
PROMPT = str(HERE / "cosyvoice" / "CosyVoice" / "asset" / "cross_lingual_prompt.wav")
tags = {"laugh": "[laughter]", "sigh": "[breath]"}
mood = {"text": ""}


def synth(text):
    instruct = f"Speak in a {mood['text']} way, like a friend on the phone.<|endofprompt|>"
    for out in cv.inference_instruct2(text, instruct, PROMPT, stream=True):
        yield out["tts_speech"].squeeze(0).float().cpu().numpy()


mood["text"] = "casual"
list(synth("Warming up."))
b.loaded()
for line in LINES:
    mood["text"] = line["mood"]
    b.run(line, "whole", text_for(line, tags), synth)
    b.chunked(line, text_for(line), synth)
b.finish()
