"""Times micro-turn ticks against a running llama-server in a model's chat format: a short scripted call, tick by tick,
with P(listen) and the reply, then the median and worst tick time. For a new base model (or GPU setting) before
training: the loop needs ticks well under 600 ms.
    uv run python check_ticks.py --format gemma4 --system partner"""
import argparse
import statistics
import time

from common import LISTEN, NOVOICE, SYSTEMS, MicroTurnClient

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8080")
ap.add_argument("--format", default="qwen3.6")
ap.add_argument("--system", default="partner")
ap.add_argument("--adapter", default=None)
ap.add_argument("--prefill", type=int, default=0, help="start from this many ticks of a real call (calls_run12.jsonl), to time a long call")
a = ap.parse_args()
cl = MicroTurnClient(a.url, listen_p=0.6, fast=True, adapter=a.adapter)
cl.format, cl.system = a.format, SYSTEMS[a.system]
script = ["Hey, how's", "it going?", NOVOICE, NOVOICE, NOVOICE, "So I was", "thinking about", "getting a bike", "for work.", NOVOICE, NOVOICE,
          NOVOICE, "What's twelve", "times twelve?", NOVOICE, NOVOICE, NOVOICE, NOVOICE]
history, times = [], []
if a.prefill:
    from common import read_jsonl
    call = read_jsonl("out/calls_run12.jsonl")[0]
    for t in call["ticks"][: a.prefill]:
        history += [("user", t["heard"]), ("assistant", t["said"])]
    print(f"prefilled {a.prefill} ticks")
for heard in script:
    t0 = time.time()
    said, p, _ = cl.tick(history, heard)
    times.append((time.time() - t0) * 1000)
    history += [("user", heard), ("assistant", said)]
    print(f"{times[-1]:6.0f} ms  P(listen) {p:.2f}  {heard!r:24} -> {said!r}")
print(f"median {statistics.median(times[1:]):.0f} ms, worst {max(times[1:]):.0f} ms (first tick {times[0]:.0f} ms: the prompt is new)")
