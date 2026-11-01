"""A tiny local HTTP voice server, shared by serve_chatterbox.py and serve_orpheus.py (the app's GPU voices).

  POST /tts   {"text": "...", "cont": bool, "stream": bool}  -> raw float32 little-endian mono PCM; header X-Sample-Rate
              stream (voices that can): the PCM comes in pieces as it is generated (X-Stream: 1), the first once
              PREROLL_S of sound is ready, then every CHUNK_S, so playback starts early and never runs dry on a hiccup
  POST /voice {"ref": path | null, "speaker": name}  -> switch voice without a restart (set_voice)
  GET /health                  -> {"status": "ok", "sample_rate": N}

One request at a time (the app sends its pieces in order). Every piece is trimmed to a natural breath at both ends:
these models pad each utterance with silence (Orpheus ~0.8 s before and ~0.5 s after, Chatterbox ~0.1 / ~0.3 s), and
the app plays pieces back to back, so untrimmed they sound like a pause between every few words.
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

import numpy as np

LEAD_S, TAIL_S, FADE_S = 0.03, 0.08, 0.01
TAIL_CONT_S = 0.02  # a piece that stops mid-sentence: the next one follows almost at once
PREROLL_S, CHUNK_S, SILENT = 0.45, 0.25, 0.01  # streaming: first send (at least), later sends, what counts as silence
WORD_S = 0.4  # a spoken word, for guessing how long a piece will play


def stream_trimmed(chunks, sr, tail_s, preroll_s=PREROLL_S):
    """trim() for audio that arrives in pieces: drops the silence before the first sound as it comes, holds back
    silence until more sound follows (so a pause inside the piece stays, the one at its end is cut to tail_s), and
    yields the sound in sends of preroll_s first and CHUNK_S after."""
    started, held, out, n, first = False, [], [], 0, True
    for a in chunks:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        loud = np.flatnonzero(np.abs(a) > SILENT)
        if not started:
            if not loud.size:
                continue
            a = a[max(0, loud[0] - int(LEAD_S * sr)):].copy()
            f = min(int(FADE_S * sr), a.size)
            a[:f] *= np.linspace(0, 1, f, dtype=np.float32)
            started, loud = True, np.flatnonzero(np.abs(a) > SILENT)
        if loud.size:
            out += held + [a[: loud[-1] + 1]]
            n += sum(x.size for x in held) + loud[-1] + 1
            held = [a[loud[-1] + 1:]]
        else:
            held.append(a)
        if n >= (preroll_s if first else CHUNK_S) * sr:
            yield np.concatenate(out)
            out, n, first = [], 0, False
    tail = np.concatenate(held)[: int(tail_s * sr)] if held else np.zeros(0, np.float32)
    if tail.size:
        f = min(int(FADE_S * sr), tail.size)
        tail = tail.copy()
        tail[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
    rest = np.concatenate(out + [tail]) if out or tail.size else np.zeros(0, np.float32)
    if rest.size:
        yield rest


def trim(audio, sr, rel=0.02, tail_s=TAIL_S):
    """Cuts the silence before the first and after the last sound, keeping a short breath at each end."""
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not a.size:
        return a
    loud = np.flatnonzero(np.abs(a) > max(0.005, rel * float(np.abs(a).max())))
    if not loud.size:
        return a[:0]
    s = max(0, loud[0] - int(LEAD_S * sr))
    e = min(a.size, loud[-1] + int(tail_s * sr))
    a = a[s:e].copy()
    f = min(int(FADE_S * sr), a.size // 2)
    if f:
        ramp = np.linspace(0, 1, f, dtype=np.float32)
        a[:f] *= ramp
        a[-f:] *= ramp[::-1]
    return a


def serve(name, sr, synth, port, stream_synth=None, set_voice=None):
    """synth(text) -> float32 audio at `sr`; stream_synth(text), if given, yields it in pieces as it is generated.
    Serves until killed."""

    # how fast the voice really generates (seconds of audio a second), from the pieces so far: when it is slower than
    # it speaks (a busy GPU), a piece must hold back more before it starts or playback runs dry and pops
    speed = {"rate": 2.0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json", headers=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, json.dumps({"status": "ok", "voice": name, "sample_rate": sr}).encode())
            else:
                self._send(404, b"{}")

        def do_POST(self):
            if self.path == "/voice":
                try:
                    req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                    if set_voice:
                        set_voice(req)
                    print(f"[{name}] voice: {req}", flush=True)
                    return self._send(200, b'{"ok": true}')
                except Exception as e:
                    print(f"[{name}] voice error: {e}", file=sys.stderr, flush=True)
                    return self._send(500, json.dumps({"error": str(e)}).encode())
            if self.path != "/tts":
                return self._send(404, b"{}")
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                text = str(req.get("text", "")).strip()
                cont = bool(req.get("cont")) and text and text[-1] not in ".!?"
                t0 = time.time()
                # mid-sentence: a trailing comma makes the voice end on a continuing tone instead of a full stop
                say = (text.rstrip(",;:") + ",") if cont else text
                if req.get("stream") and stream_synth and text:
                    # HTTP/1.0 without a length: the body is whatever comes until the connection closes
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("X-Sample-Rate", str(sr))
                    self.send_header("X-Stream", "1")
                    self.end_headers()
                    total, first_ms, raw = 0, None, [0]
                    est = len(text.split()) * WORD_S  # how long this piece will play
                    preroll = min(3.0, max(PREROLL_S, est * (1 - speed["rate"]) + 0.3))

                    def counted(gen):
                        for c in gen:
                            raw[0] += len(c)
                            yield c

                    for part in stream_trimmed(counted(stream_synth(say)), sr, TAIL_CONT_S if cont else TAIL_S, preroll):
                        if first_ms is None:
                            first_ms = int((time.time() - t0) * 1000)
                        self.wfile.write(part.astype("<f4").tobytes())
                        self.wfile.flush()
                        total += part.size
                    took = time.time() - t0
                    # speed over the part that is heard: from the first sound to the end (the leading silence the
                    # voice generates first is dropped, so it does not count)
                    sounding = took - (first_ms or 0) / 1000 + preroll
                    if total and sounding > 0.2:
                        speed["rate"] = 0.5 * speed["rate"] + 0.5 * min(3.0, total / sr / sounding)
                    print(f"[{name}] streamed, pre-roll {preroll:.2f} s, first sound {first_ms} ms, {total / sr:.2f} s in {int(took * 1000)} ms, "
                          f"speed {speed['rate']:.2f}x: {text}", flush=True)
                    return
                audio = trim(synth(say), sr, tail_s=TAIL_CONT_S if cont else TAIL_S) if text else np.zeros(0, np.float32)
                ms = int((time.time() - t0) * 1000)
                print(f"[{name}] {ms} ms, {audio.size / sr:.2f} s: {text}", flush=True)
                self._send(200, audio.astype("<f4").tobytes(), "application/octet-stream", {"X-Sample-Rate": str(sr), "X-Ms": str(ms)})
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return  # the app stopped listening (a cut-in or a voice switch dropped this piece): nothing to report
            except Exception as e:  # the app logs it and goes on; the call keeps running
                print(f"[{name}] error: {e}", file=sys.stderr, flush=True)
                try:
                    self._send(500, json.dumps({"error": str(e)}).encode())
                except OSError:
                    pass

    print(f"[{name}] ready on 127.0.0.1:{port} ({sr} Hz)", flush=True)
    # a voice that can make two pieces at once (Orpheus: stream_synth and a 2-slot llama-server) is served on threads
    (ThreadingHTTPServer if stream_synth else HTTPServer)(("127.0.0.1", port), H).serve_forever()
