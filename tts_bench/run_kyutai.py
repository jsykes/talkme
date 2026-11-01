"""Kyutai TTS 1.6B (CC-BY 4.0): built for live chat, it reads text as it arrives and speaks as it goes. Voice: the
Unmute website's default. No sound tags.   cd tts_bench/kyutai && uv run python ../run_kyutai.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEVICE, LINES, MODELS, Bench, text_for

import os
os.environ.setdefault("NO_TORCH_COMPILE", "1")  # torch.compile needs the MSVC compiler on Windows, which isn't installed
b = Bench("kyutai-tts", "voice unmute-prod-website/default_voice, streams text in and audio out, no sound tags")
import numpy as np
import torch
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO, TTSModel

d = MODELS / "kyutai-tts-1.6b"
info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO, moshi_weights=d / "dsm_tts_1e68beda@240.safetensors",
                                   mimi_weights=d / "tokenizer-e351c8d8-checkpoint125.safetensors",
                                   tokenizer=d / "tokenizer_spm_8k_en_fr_audio.model", config_path=d / "config.json")
tts = TTSModel.from_checkpoint_info(info, n_q=32, temp=0.6, device=torch.device(DEVICE))
b.sr = tts.mimi.sample_rate
voice = tts.get_voice_path("unmute-prod-website/default_voice.wav")  # 256 KB voice embedding from kyutai/tts-voices
cond = tts.make_condition_attributes([voice], cfg_coef=2.0)


def synth(text):
    """Frames come out as they are generated; they are collected and yielded after each generate() step returns, so
    first audio is measured from the first decoded frame via a timestamp list."""
    out = []
    entries = tts.prepare_script([text], padding_between=1)

    def on_frame(frame):
        if (frame != -1).all():
            pcm = tts.mimi.decode(frame[:, 1:, :]).cpu().numpy()
            out.append(np.clip(pcm[0, 0], -1, 1))
            if len(out) == 1:
                synth.first_at = __import__("time").time()

    with torch.inference_mode(), tts.mimi.streaming(1):
        tts.generate([entries], [cond], on_frame=on_frame)
    yield from out


def timed(line, mode, text):
    import time
    t0 = time.time()
    synth.first_at = None
    b.run(line, mode, text, synth)
    if synth.first_at:  # the true first frame, not when the collected frames were handed back
        b.rows[-1]["first_audio_s"] = round(synth.first_at - t0, 3)


list(synth("Warming up."))
b.loaded()
for line in LINES:
    timed(line, "whole", text_for(line))
    b.chunked(line, text_for(line), synth)
b.finish()
