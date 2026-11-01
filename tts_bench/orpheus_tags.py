"""Samples of Orpheus's emotion tags in a voice, to hear which earn a place in the partner's words (roadmap: the
partner would write "(laughs)" / "(sighs)" and tts.js would map them to these). Starts its own Orpheus servers on
ports 8190/8192 (so a running app or test is left alone; the GPU needs ~4.5 GB free), writes one WAV (and MP3) per line to
tts_bench/out/tags/<voice>/, and stops only what it started.
    uv run --project tune python tts_bench/orpheus_tags.py [--voices tara,leah,zoe] [--quant Q4_K_M]
"""
import os
import argparse
import subprocess
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
LLAMA = ROOT / "vendor" / "llama" / ("llama-server.exe" if os.name == "nt" else "llama-server")
MODELS = ROOT / "models" / "tts_bench" / "orpheus"
LINES = [
    ("plain", "Oh no, that sounds like a rough day. Did you at least get some sleep?"),
    ("laugh", "<laugh> Okay, that is the dumbest thing I've heard all week. I love it."),
    ("chuckle", "Yeah, that figures. <chuckle> You always find the one broken one."),
    ("sigh", "<sigh> I know. I've been meaning to call her for a month."),
    ("gasp", "<gasp> You did not. Tell me you did not say that to him."),
    ("groan", "<groan> Not the printer again. Every single time."),
    ("yawn", "<yawn> Sorry, it's late. Go on, I'm listening."),
    ("cough", "Right, so <cough> sorry, where was I. The thing with the car."),
    ("sniffle", "<sniffle> No, I'm fine. It's just been a lot this week."),
    ("mid_laugh", "So he opens the door and the cat is just sitting there <laugh> wearing the hat."),
    ("two_tags", "<sigh> Honestly. <chuckle> You'd think I'd learn by now."),
]


def up(url, s=240):
    for _ in range(s):
        try:
            if urllib.request.urlopen(url + "/health", timeout=2).status == 200:
                return True
        except Exception:
            time.sleep(1)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default="tara", help="comma-separated: tara, leah, jess, leo, dan, mia, zac, zoe")
    ap.add_argument("--quant", default="Q4_K_M", help="Q4_K_M (the app; Q8_0 and Q6_K were deleted 2026-10-28); samples go to out/tags/<voice>-<quant>/ unless Q8_0")
    a = ap.parse_args()
    procs = []
    try:
        procs.append(subprocess.Popen([str(LLAMA), "-m", str(MODELS / f"orpheus-3b-0.1-ft-{a.quant}.gguf"), "-ngl", "99", "-c", "8192", "--port", "8190", "--host", "127.0.0.1", "-np", "1"],
                                      cwd=str(LLAMA.parent), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        if not up("http://127.0.0.1:8190"):
            raise SystemExit("the Orpheus llama-server did not start")
        procs.append(subprocess.Popen(["uv", "run", "--project", str(ROOT / "tts_bench" / "orpheus"), "python", str(ROOT / "tts_bench" / "serve_orpheus.py"),
                                       "--port", "8192", "--llama-url", "http://127.0.0.1:8190", "--voice", "tara"],
                                      cwd=str(ROOT / "tts_bench" / "orpheus"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        if not up("http://127.0.0.1:8192"):
            raise SystemExit("serve_orpheus did not start")
        for voice in a.voices.split(","):
            out = ROOT / "tts_bench" / "out" / "tags" / (voice if a.quant == "Q8_0" else f"{voice}-{a.quant}")
            out.mkdir(parents=True, exist_ok=True)
            requests.post("http://127.0.0.1:8192/voice", json={"speaker": voice}, timeout=10)
            for name, text in LINES:
                r = requests.post("http://127.0.0.1:8192/tts", json={"text": text}, timeout=120)
                r.raise_for_status()
                pcm = np.frombuffer(r.content, "<f4")
                with wave.open(str(out / f"{name}.wav"), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(int(r.headers.get("X-Sample-Rate", 24000)))
                    w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
                print(f"{voice}/{name}.wav  {pcm.size / 24000:.1f} s  ({r.headers.get('X-Ms')} ms): {text}", flush=True)
                # an MP3 beside it: the WAV did not preview on a phone
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out / f"{name}.wav"), "-codec:a", "libmp3lame", "-q:a", "3", str(out / f"{name}.mp3")], check=False)
    finally:
        for p in procs:  # with the children: `uv run` starts the server as a child that outlives it
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)


if __name__ == "__main__":
    main()
