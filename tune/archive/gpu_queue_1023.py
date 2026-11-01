"""A GPU window (2026-10-23, ~4 h): tests only, no training, most useful first.
  1. g26 at listen thresholds 0.5 and 0.4: quiz and recorded calls (it jumps in early at 0.6; lower makes it hold back)
  2. the memory test on the varied set (data/heldout_memory.jsonl, v2): g26 at 0.6, g26m2 at 0.85, g26m at 0.6
  3. g26m2's live tests at 0.85
Log: out/gpuq.pipeline.log. Stop with tune/gpu_pause.sh; rerunning skips what is done.
    uv run python gpu_queue_1023.py
"""

import pipeline
from pipeline import OUT, ROOT, exists, kill_server, run, say, server, uv

pipeline.NAME, pipeline.LOG = "gpuq", OUT / "gpuq.pipeline.log"
base = pipeline.BASES["gemma4a4b"]
ad = {n: str(OUT / f"{n}.gguf") for n in ("g26", "g26m", "g26m2")}
common = ["--repeat-guard", "--system", "partner", "--format", "gemma4", "--history-ticks", str(base["history_ticks"])]


def up():
    server(base["gguf"], "-np", "1", *[x for n in ad.values() for x in ("--lora", n)], "--lora-init-without-apply", ctx=base["ctx"])


# 1. g26, lower thresholds
todo = [p for p in ("0.5", "0.4") if not exists(OUT / f"eval_yap_g26_p{p[2:]}.json")]
if todo:
    up()
for p in todo:
    tag = f"g26_p{p[2:]}"
    run(uv("probe_answers.py", "--adapter", ad["g26"], "--listen-p", p, *common, "--name", tag, "--out", str(OUT / f"probe_{tag}.json")), log=OUT / f"probe_{tag}.log")
    say(f"g26 at {p}: quiz " + open(OUT / f"probe_{tag}.log", encoding="utf-8").read().strip().splitlines()[-1])
    run(uv("evaluate.py", "--url", "http://127.0.0.1:8080", "--calls", "data/heldout_yap.jsonl", "--n", "24", "--parallel", "1", "--adapter", ad["g26"],
           "--listen-p", p, *common, "--out", str(OUT / f"eval_yap_{tag}.json")), log=OUT / f"eval_yap_{tag}.log")
    say(f"g26 at {p}: recorded calls done")
names = ["run12", "g26", "g26_p5", "g26_p4", "g26_p7", "g26_p8"]
run(uv("yap_metrics.py", "data/heldout_yap.jsonl", *[f"out/eval_yap_{n}.json" for n in names if exists(OUT / f"eval_yap_{n}.json")], "--first", "24",
       "--names", ",".join(n for n in names if exists(OUT / f"eval_yap_{n}.json")), "--md", "out/yap_g26_thresholds.md"), check=False)
say("g26 thresholds: out/yap_g26_thresholds.md")

# 2. the memory test, varied set
for name, p in (("g26", "0.6"), ("g26m2", "0.85"), ("g26m", "0.6")):
    tag = f"{name}_p{p[2:]}"
    if exists(OUT / f"mem2_{tag}.json"):
        continue
    up()
    run(uv("evaluate.py", "--url", "http://127.0.0.1:8080", "--calls", str(pipeline.MEMORY_CALLS), "--n", "0", "--parallel", "1", "--adapter", ad[name],
           "--listen-p", p, *common, "--out", str(OUT / f"eval_mem2_{tag}.json")), log=OUT / f"eval_mem2_{tag}.log")
    run(uv("memory_check.py", "--calls", str(pipeline.MEMORY_CALLS), "--eval", str(OUT / f"eval_mem2_{tag}.json"), "--out", str(OUT / f"mem2_{tag}.json")),
        log=OUT / f"mem2_{tag}.log")
    say(f"memory test v2, {name} at {p}: " + open(OUT / f"mem2_{tag}.log", encoding="utf-8").read().strip().splitlines()[-1])
kill_server()

# 3. g26m2 live at 0.85
if not (ROOT / "tests" / "out" / "report-bench-g26m2_p85-amy.json").exists():
    say("live tests: g26m2 at 0.85")
    run(["bash", "scripts/live_bench.sh", "g26m2_p85", "tune/out/g26m2.gguf", "0.85"], cwd=ROOT,
        env={"MODEL": base["gguf"], "PROMPT": "partner"}, log=pipeline.LOG, check=False)
    run(["node", "scripts/live_bench_report.js", "run12", "g26", "g26m2", "g26m2_p85"], cwd=ROOT, log=OUT / "live_g26m2_p85.log", check=False)
kill_server()
say("queue done; GPU free (live report: tests/out/live_bench.md)")
