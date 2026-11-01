"""Times identical micro-turn requests with per-request adapter scales vs adapter scales set once on the server.
    uv run python check_lora_overhead.py --url http://127.0.0.1:8080 --adapter run2.gguf
"""
import argparse
import statistics
import time

import requests

from common import NOVOICE, render, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--adapter", required=True, help="file name of a loaded adapter")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    s = requests.Session()
    loaded = s.get(a.url + "/lora-adapters").json()
    scales = [{"id": x["id"], "scale": 1.0 if x["path"].endswith(a.adapter) else 0.0} for x in loaded]
    call = read_jsonl("../models/lora/dataset/calls_heldout.jsonl")[0]
    hist = []
    for t in call["ticks"][:60]:
        hist += [("user", t["heard"]), ("assistant", t["said"])]

    def run(per_request):
        times, h = [], list(hist)
        for i in range(a.n):
            body = {"prompt": render(h, NOVOICE), "n_predict": 1, "n_probs": 8, "temperature": 0, "cache_prompt": True}
            if per_request:
                body["lora"] = scales
            t0 = time.time()
            s.post(a.url + "/completion", json=body).raise_for_status()
            times.append(time.time() - t0)
            h += [("user", NOVOICE), ("assistant", "<listen>")]
        return statistics.median(times[3:]) * 1000

    s.post(a.url + "/lora-adapters", json=scales).raise_for_status()  # set globally once
    g = run(False)
    p = run(True)
    print(f"median per tick (1 token): adapter set on the server {g:.0f} ms | adapter in every request {p:.0f} ms")


if __name__ == "__main__":
    main()
