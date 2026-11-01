"""Checks that the one-token-first tick gives exactly the same replies as the plain tick, and how much faster it is.
    uv run python check_fast_tick.py --url http://127.0.0.1:8080 --adapter ../models/lora/microturn-duplex-lora.gguf --ticks 160
"""
import argparse
import time

from common import LISTEN, MicroTurnClient, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--calls", default="../models/lora/dataset/calls_heldout.jsonl")
    ap.add_argument("--call", type=int, default=1)
    ap.add_argument("--ticks", type=int, default=160)
    a = ap.parse_args()
    call = read_jsonl(a.calls)[a.call]
    ticks = call["ticks"][: a.ticks]
    res = {}
    for fast in (False, True):
        cl = MicroTurnClient(a.url, adapter=a.adapter, fast=fast)
        hist, said, lat = [], [], {"listen": [], "speak": []}
        for t in ticks:
            t0 = time.time()
            s, p, _ = cl.tick(hist, t["heard"])
            lat["listen" if s == LISTEN else "speak"].append(time.time() - t0)
            hist += [("user", t["heard"]), ("assistant", s)]
            said.append(s)
        res[fast] = said
        med = lambda x: sorted(x)[len(x) // 2] if x else float("nan")
        print(f"{'fast ' if fast else 'plain'}: listen ticks {len(lat['listen'])} median {med(lat['listen']) * 1000:.0f} ms | "
              f"speak ticks {len(lat['speak'])} median {med(lat['speak']) * 1000:.0f} ms | total {sum(lat['listen'] + lat['speak']):.1f} s")
    diff = [(i, x, y) for i, (x, y) in enumerate(zip(res[False], res[True])) if x != y]
    print("identical replies" if not diff else f"{len(diff)} ticks differ, first: {diff[:5]}")


if __name__ == "__main__":
    main()
