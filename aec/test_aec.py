"""Offline check of aec_server.py with simulated echo: the assistant's voice goes through a made-up room (delay, reverb,
a little speaker distortion) into the mic, a second voice talks over part of it (you cutting in), plus some noise.
Prints how much the echo drops where only the speaker plays, how much of your voice survives where both talk, and the
time per 100 ms frame. Runs aec_server.py exactly as the app does (stdin/stdout).

    uv run --project aec python aec/test_aec.py
"""
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VOICES = ROOT / "models" / "tts_bench" / "chatterbox-voices"
RATE = 16000


def load(path):
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
        if w.getnchannels() == 2:
            x = x.reshape(-1, 2).mean(1)
        sr = w.getframerate()
    n = int(len(x) * RATE / sr)  # linear resample to 16 kHz (fine for a test signal)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def room(rng, delay_ms, tail_ms=300, drr_db=3.0, rt60_s=0.25):
    """A speaker-to-mic path: `delay_ms` of travel and buffering, the direct sound, then reflections dying away
    (rt60_s) at drr_db below it in total, like a desk setup in a furnished room."""
    t = np.arange(1, int(RATE * tail_ms / 1000))
    tail = rng.standard_normal(len(t)) * np.exp(-6.9 * t / (RATE * rt60_s))
    tail *= np.sqrt(10 ** (-drr_db / 10) / np.dot(tail, tail))
    ir = np.concatenate([[1.0], tail])
    return np.concatenate([np.zeros(int(RATE * delay_ms / 1000)), ir]).astype(np.float32)


def run(mic, ref, delay_hint, frame=1600):
    p = subprocess.Popen([sys.executable, str(Path(__file__).with_name("aec_server.py")), "--rate", str(RATE), "--delay-ms", str(delay_hint)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out = []
    for i in range(0, len(mic) - frame + 1, frame):
        p.stdin.write(struct.pack("<II", 1, frame) + mic[i:i + frame].astype("<f4").tobytes() + ref[i:i + frame].astype("<f4").tobytes())
        p.stdin.flush()
        buf = b""
        while len(buf) < 4 * frame + 4:  # the samples, then their level
            buf += p.stdout.read(4 * frame + 4 - len(buf))
        out.append(np.frombuffer(buf[:4 * frame], dtype="<f4"))
    p.stdin.close()
    p.wait()
    return np.concatenate(out)


def db(a, b):
    return 10 * np.log10(np.dot(a, a) / max(np.dot(b, b), 1e-12))


def main():
    rng = np.random.default_rng(0)
    files = sorted(VOICES.glob("*.wav"))
    far = np.concatenate([load(f) for f in files[:3]])[: RATE * 24]  # the assistant talking, ~24 s
    you = load(files[3])[: RATE * 5] * 0.9                             # you, cutting in for 5 s
    for delay_ms, gain, label in ((60, 0.5, "speaker at normal volume"), (120, 0.9, "loud speaker, longer path")):
        echo = np.convolve(far, room(rng, delay_ms))[: len(far)] * gain
        echo = np.tanh(echo * 1.5) / 1.5  # a cheap speaker's distortion at volume
        near = np.zeros_like(far)
        at = RATE * 14
        near[at:at + len(you)] = you
        noise = rng.standard_normal(len(far)).astype(np.float32) * 0.003
        mic = (echo + near + noise).astype(np.float32)
        import time
        t0 = time.time()
        out = run(mic, far, delay_hint=40)
        took = time.time() - t0
        n = len(out)
        mic, echo, near = mic[:n], echo[:n], near[:n]
        settled = slice(RATE * 3, at)  # speaker only, after the first 3 s
        both = slice(at, at + len(you))
        print(f"{label} (echo {delay_ms} ms behind):")
        print(f"  echo removed where only the speaker plays: {db(mic[settled], out[settled]):.1f} dB")
        # your voice in the output, where both talk: the processing delays its output by a few ms, so line them up first
        o, y = out[both], near[both]
        lag = max(range(0, 401, 4), key=lambda k: float(np.dot(np.roll(o, -k), y)))
        corr = lambda a, b: float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))
        print(f"  while you cut in: your voice comes through at {corr(np.roll(o, -lag), y):.2f} correlation, {db(o, y):+.1f} dB "
              f"(the raw mic: {corr(mic[both], y):.2f}); output {1000 * lag / RATE:.0f} ms behind")
        print(f"  {took / (n / RATE) * 100:.1f}% of real time ({1000 * took / (n / 1600):.1f} ms per 100 ms frame, process start included)")


if __name__ == "__main__":
    main()
