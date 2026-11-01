"""talkme's echo canceller: WebRTC AEC3 (pywebrtc-audio) between the mic and the speech recogniser.

The app sends the mic and what its speaker played over the same stretch (both 16 kHz float32, captured in the same
audio callback of the app window) and gets the mic back with the voice removed. Over stdin/stdout, little-endian:

  request   uint32 kind, uint32 n, then for kind 1: n float32 mic samples followed by n float32 reference samples
            kind 1 = audio, kind 2 = reset (n unused), kind 3 = delay hint (n = milliseconds)
  reply     kind 1 only: n float32 cleaned samples, then 1 float32: their level (RMS) before gain control, which the
            app ducks on (gain control lifts the faint leftover echo, up to 50 dB, until it looks like the caller)

The stages run in WebRTC's own order: echo cancelling, then noise suppression, then gain control. Every ten seconds
it logs, for the stretches where the speaker was playing, how loud the echo was in the mic and how much the echo
canceller alone took off (gain control afterwards would hide it). Real echo cancelling takes 20 dB or more off.

    uv run --project aec python aec/aec_server.py --rate 16000 --delay-ms 40 --ns --agc
"""
import argparse
import struct
import sys
import time

import numpy as np
from pywebrtc_audio import EchoCanceller, GainController, NoiseSuppressor


def read_exact(f, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def dbfs(e, n):
    return 10 * np.log10(max(e / max(n, 1), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=int, default=16000)
    ap.add_argument("--delay-ms", type=int, default=40, help="hint: speaker buffer to mic; AEC3 estimates the rest")
    ap.add_argument("--ns", action="store_true", help="also suppress noise (the app's own setting)")
    ap.add_argument("--agc", action="store_true", help="also even out the mic level (the app's own setting)")
    a = ap.parse_args()
    aec = EchoCanceller(sample_rate=a.rate, num_channels=1, stream_delay_ms=a.delay_ms)
    ns = NoiseSuppressor(sample_rate=a.rate, num_channels=1) if a.ns else None
    agc = GainController(sample_rate=a.rate, num_channels=1) if a.agc else None
    fin, fout = sys.stdin.buffer, sys.stdout.buffer
    print(f"aec: ready ({a.rate} Hz, delay hint {a.delay_ms} ms, noise suppression {'on' if ns else 'off'}, gain {'on' if agc else 'off'})",
          file=sys.stderr, flush=True)
    # while the speaker plays: mic energy before and after the echo canceller, and samples counted
    e_in = e_out = 0.0
    n_play = 0
    frames = 0
    t_proc = 0.0
    last = time.time()
    while True:
        head = read_exact(fin, 8)
        if head is None:
            return
        kind, n = struct.unpack("<II", head)
        if kind == 2:
            aec.reset()
            continue
        if kind == 3:
            aec.stream_delay_ms = int(n)
            continue
        body = read_exact(fin, 8 * n)
        if body is None:
            return
        both = np.frombuffer(body, dtype="<f4")
        mic, ref = np.ascontiguousarray(both[:n]), np.ascontiguousarray(both[n:])
        t0 = time.perf_counter()
        clean = aec.process(mic, ref)
        out = clean
        if ns:
            out = ns.process(out)
        level = float(np.sqrt(np.mean(np.square(out, dtype=np.float64))))
        if agc:
            out = agc.process(out)
        t_proc += time.perf_counter() - t0
        fout.write(np.asarray(out, dtype="<f4").tobytes() + struct.pack("<f", level))
        fout.flush()
        frames += 1
        if float(np.abs(ref).max(initial=0.0)) > 0.01:  # the speaker was playing in this stretch
            e_in += float(np.dot(mic, mic))
            e_out += float(np.dot(clean, clean))
            n_play += n
        if time.time() - last > 10:
            if n_play:
                msg = (f"while the speaker played ({n_play / a.rate:.1f} s): mic {dbfs(e_in, n_play):.0f} dBFS, "
                       f"echo canceller took off {10 * np.log10(max(e_in, 1e-12) / max(e_out, 1e-12)):.1f} dB")
            else:
                msg = "the speaker did not play"
            print(f"aec: {frames} frames, {1000 * t_proc / frames:.2f} ms each; {msg}", file=sys.stderr, flush=True)
            e_in = e_out = 0.0
            n_play = frames = 0
            t_proc, last = 0.0, time.time()


if __name__ == "__main__":
    main()
