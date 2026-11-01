"""Orpheus 3B as an app voice (see serve_common.py): the language model runs in its own llama-server (the app starts
it, --llama-url) and the SNAC decoder here. By hand:
   cd tts_bench/orpheus && uv run python ../serve_orpheus.py --port 8092 --llama-url http://127.0.0.1:8090 [--voice tara]"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from serve_common import serve

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8092)
ap.add_argument("--llama-url", default="http://127.0.0.1:8090")
ap.add_argument("--voice", default="tara", help="tara, leah, jess, leo, dan, mia, zac, zoe")
ap.add_argument("--device", default="cuda")
ap.add_argument("--snac", default=str(HERE.parent / "models" / "tts_bench" / "snac_24khz"))
a = ap.parse_args()

import numpy as np
import requests
import torch
from snac import SNAC

snac = SNAC.from_pretrained(a.snac).eval().to(a.device)
START, END = [128259, 128000], [128009, 128260]  # start of human, BOS ... end of text, end of human (canopylabs' format)
CODE0 = 128266  # first audio code token; each of the 7 codes in a frame is offset by a further 4096 * position
import threading
_local = threading.local()


class _Sessions:  # one requests session per thread: the server makes two pieces at once
    def __getattr__(self, k):
        if not hasattr(_local, "s"):
            _local.s = requests.Session()
        return getattr(_local.s, k)


session = _Sessions()


def decode(codes):
    f = np.array(codes[: len(codes) // 7 * 7]).reshape(-1, 7)
    if not len(f):
        return np.zeros(0, np.float32)
    layers = (f[:, 0], f[:, [1, 4]].reshape(-1), f[:, [2, 3, 5, 6]].reshape(-1))
    with torch.inference_mode():
        out = snac.decode([torch.tensor(x, device=a.device).unsqueeze(0) for x in layers])
    return out.squeeze().float().cpu().numpy()


def synth(text):
    toks = session.post(a.llama_url + "/tokenize", json={"content": f"{a.voice}: {text}", "add_special": False}).json()["tokens"]
    body = {"prompt": START + toks + END, "n_predict": 1600, "temperature": 0.6, "top_p": 0.9, "repeat_penalty": 1.1,
            "return_tokens": True, "cache_prompt": False}
    out = session.post(a.llama_url + "/completion", json=body).json()
    codes = []
    for t in out.get("tokens", []):
        if t >= CODE0:
            codes.append((t - CODE0) - 4096 * (len(codes) % 7))
    return decode(codes)


FRAME = 2048  # samples per frame of 7 codes at 24 kHz
XF = 256  # ~10 ms: neighbouring windows decode the samples around a frame edge slightly differently; blend them


def stream_synth(text):
    """Yields audio as the codes arrive. Each new frame is decoded inside a window of four (the frame before it and
    two after), keeping only its own samples, so the seams between windows are smooth (the Orpheus streaming recipe):
    one frame (~85 ms) of lookahead."""
    toks = session.post(a.llama_url + "/tokenize", json={"content": f"{a.voice}: {text}", "add_special": False}).json()["tokens"]
    body = {"prompt": START + toks + END, "n_predict": 1600, "temperature": 0.6, "top_p": 0.9, "repeat_penalty": 1.1,
            "return_tokens": True, "cache_prompt": False, "stream": True}
    codes, done = [], 1  # frames before `done` have been sent (frame 0 is the attack, never sent)
    held = None  # the last XF samples sent so far are held back to be blended with the next window's version
    with session.post(a.llama_url + "/completion", json=body, stream=True) as r:
        for raw in r.iter_lines():
            if not raw.startswith(b"data: "):
                continue
            j = json.loads(raw[6:])
            for t in j.get("tokens", []):
                if t >= CODE0:
                    codes.append((t - CODE0) - 4096 * (len(codes) % 7))
            n = len(codes) // 7
            while n >= 4 and done <= n - 3:  # frame `done` sits second in the window done-1 .. done+2
                w = decode(codes[(done - 1) * 7:(done + 3) * 7])
                part = w[FRAME - XF:2 * FRAME]  # this frame, plus the XF samples before it as this window has them
                if held is not None:
                    ramp = np.linspace(0, 1, XF, dtype=np.float32)
                    part = part.copy()
                    part[:XF] = held * (1 - ramp) + part[:XF] * ramp
                else:
                    part = part[XF:]
                yield part[:-XF]
                held = part[-XF:].copy()
                done += 1
            if j.get("stop"):
                break
    n = len(codes) // 7
    if n >= 4:
        w = decode(codes[(n - 4) * 7:n * 7])
        rest = w[(done - (n - 4)) * FRAME - XF:].copy()  # the frames after the last window's second, blended in
        if held is not None:
            ramp = np.linspace(0, 1, XF, dtype=np.float32)
            rest[:XF] = held * (1 - ramp) + rest[:XF] * ramp
        else:
            rest = rest[XF:]
        yield rest
    elif n:
        yield decode(codes[: n * 7])


synth("Warming up.")
list(stream_synth("Warming up."))
def set_voice(req):
    """{"speaker": one of the eight Orpheus speakers}"""
    if req.get("speaker"):
        a.voice = req["speaker"]


serve("orpheus", 24000, synth, a.port, stream_synth=stream_synth, set_voice=set_voice)
