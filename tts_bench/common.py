"""Shared by the bake-off runners (each runs in its own environment): the lines, timing, GPU memory, output files.

Each runner synthesises every line of lines.json twice:
  whole: the line in one piece, streamed out where the model can (first audio = first chunk)
  chunked: the line in pieces of about three words, one after another, the way the app feeds its voice today
and writes out/<model>/<id>.<mode>.wav plus out/<model>.json with first-audio time, total time, audio length and peak
GPU memory.
"""
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
MODELS = ROOT.parent / "models" / "tts_bench"
OUT = ROOT / "out"
LINES = json.loads((ROOT / "lines.json").read_text(encoding="utf-8"))
# TTS_DEVICE=cpu TTS_SMOKE=1: a quick check that a runner works (first two lines, CPU) while the GPU is busy
DEVICE = os.environ.get("TTS_DEVICE", "cuda")
if os.environ.get("TTS_SMOKE"):
    LINES = LINES[:2]
    OUT = ROOT / "out" / "smoke"


def text_for(line, tags=None):
    """The line's text with {laugh}/{sigh} turned into the model's own tag (tags: {"laugh": "<laugh>"}), or removed."""
    tags = tags or {}
    t = re.sub(r"\{(\w+)\}", lambda m: tags.get(m.group(1), ""), line["text"])
    return re.sub(r"\s+", " ", t).strip()


def chunks(text, n=3):
    """About three words at a time, as the model writes them (punctuation stays with its word)."""
    w = text.split()
    return [" ".join(w[i:i + n]) for i in range(0, len(w), n)]


def gpu_used_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


class GpuPeak:
    """Polls total GPU memory in the background; .peak - .base is what the model added (llama-server included)."""
    def __init__(self):
        self.base = gpu_used_mb()
        self.peak = self.base
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop:
            self.peak = max(self.peak, gpu_used_mb())
            time.sleep(0.25)

    def stop(self):
        self._stop = True
        self._t.join()
        return self.peak - self.base


class Bench:
    def __init__(self, name, note=""):
        self.name, self.note = name, note
        self.dir = OUT / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.gpu = GpuPeak()
        self.t_load = time.time()

    def loaded(self):
        self.load_s = round(time.time() - self.t_load, 1)
        print(f"[{self.name}] loaded in {self.load_s} s, +{gpu_used_mb() - self.gpu.base} MB GPU", flush=True)

    def run(self, line, mode, text, synth):
        """synth(text) yields float32 audio chunks (any length) and returns nothing; the sample rate is self.sr."""
        t0 = time.time()
        first, parts = None, []
        for a in synth(text):
            a = np.asarray(a, dtype=np.float32).reshape(-1)
            if first is None and a.size:
                first = time.time() - t0
            parts.append(a)
        total = time.time() - t0
        audio = np.concatenate(parts) if parts else np.zeros(1, np.float32)
        sf.write(self.dir / f"{line['id']}.{mode}.wav", np.clip(audio, -1, 1), self.sr)
        secs = len(audio) / self.sr
        row = {"id": line["id"], "mode": mode, "text": text, "first_audio_s": round(first or total, 3), "total_s": round(total, 3),
               "audio_s": round(secs, 2), "realtime_x": round(secs / total, 2) if total else None}
        self.rows.append(row)
        print(f"[{self.name}] {line['id']:12} {mode:7} first {row['first_audio_s']:.2f} s, {row['audio_s']:.1f} s of audio in {total:.2f} s", flush=True)

    def chunked(self, line, text, synth_whole):
        """The app's way: each ~3-word piece synthesised as its own utterance, in order; first audio = the first piece."""
        def gen(_):
            for c in chunks(text):
                for a in synth_whole(c):
                    yield a
        self.run(line, "chunked", text, gen)

    def finish(self):
        extra = self.gpu.stop()
        res = {"model": self.name, "note": self.note, "load_s": getattr(self, "load_s", None), "gpu_mb": extra, "sample_rate": self.sr, "rows": self.rows}
        (OUT / f"{self.name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"[{self.name}] done: peak +{extra} MB GPU", flush=True)
