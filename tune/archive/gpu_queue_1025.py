"""Overnight 2026-10-25 (11 h): finish g26m5, then measure the g26 runs again with one BOS (the fix changed how pushy
Gemma is, so the app's thresholds need new numbers). Log: out/gpuq3.pipeline.log. Stop with tune/gpu_pause.sh.
    uv run python gpu_queue_1025.py
"""
import pipeline
from pipeline import OUT, ROOT, TUNE, exists, kill_server, run, say, server, uv

pipeline.NAME, pipeline.LOG = "gpuq3", OUT / "gpuq3.pipeline.log"
base = pipeline.BASES["gemma4a4b"]
common = ["--repeat-guard", "--system", "partner", "--format", "gemma4", "--history-ticks", str(base["history_ticks"])]

# 1. g26m5, the whole pipeline (resumes training from its last save); its live tests are the last stage
if not (ROOT / "tests" / "out" / "report-bench-g26m5-amy.json").exists():
    run(uv("pipeline.py", "--recipe", str(TUNE / "recipes" / "g26m5.json")), check=False)
    say("g26m5 done")

# 2. one BOS: g26s1 and g26s2 at 0.6, g26 at 0.5 (quiz and recorded calls); g26 has 0.6 already (eval_yap_g26_1bos)
ad = {n: str(OUT / f"{n}.gguf") for n in ("g26", "g26s1", "g26s2")}
todo = [(n, p) for n, p in (("g26s1", "0.6"), ("g26s2", "0.6"), ("g26", "0.5")) if not exists(OUT / f"eval_yap_{n}_1bos_p{p[2:]}.json")]
if todo:
    server(base["gguf"], "-np", "1", *[x for f in ad.values() for x in ("--lora", f)], "--lora-init-without-apply", ctx=base["ctx"])
for n, p in todo:
    tag = f"{n}_1bos_p{p[2:]}"
    run(uv("probe_answers.py", "--adapter", ad[n], "--listen-p", p, *common, "--name", tag, "--out", str(OUT / f"probe_{tag}.json")), log=OUT / f"probe_{tag}.log")
    say(f"{tag}: quiz " + open(OUT / f"probe_{tag}.log", encoding="utf-8").read().strip().splitlines()[-1])
    run(uv("evaluate.py", "--url", "http://127.0.0.1:8080", "--calls", "data/heldout_yap.jsonl", "--n", "24", "--parallel", "1", "--adapter", ad[n],
           "--listen-p", p, *common, "--out", str(OUT / f"eval_yap_{tag}.json")), log=OUT / f"eval_yap_{tag}.log")
kill_server()
names = ["run12", "g26_1bos", "g26_1bos_p5", "g26s1_1bos_p6", "g26s2_1bos_p6", "g26m4", "g26m5"]
have = [n for n in names if exists(OUT / f"eval_yap_{n}.json")]
run(uv("yap_metrics.py", "data/heldout_yap.jsonl", *[f"out/eval_yap_{n}.json" for n in have], "--first", "24", "--names", ",".join(have),
       "--md", "out/yap_1bos.md"), check=False)
say("one-BOS recorded calls: out/yap_1bos.md")

# 3. g26 live at 0.6 with one BOS (the app's live behaviour with the fix)
if not (ROOT / "tests" / "out" / "report-bench-g26_1bos-amy.json").exists():
    say("live tests: g26 at 0.6, one BOS")
    run(["bash", "scripts/live_bench.sh", "g26_1bos", "tune/out/g26.gguf", "0.6"], cwd=ROOT,
        env={"MODEL": base["gguf"], "PROMPT": "partner"}, log=pipeline.LOG, check=False)
run(["node", "scripts/live_bench_report.js", "run12", "g26", "g26_1bos", "g26m4", "g26m5"], cwd=ROOT, log=OUT / "live_1bos.log", check=False)
kill_server()
say("queue done; GPU free")
