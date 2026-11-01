"""A GPU window (2026-10-24), most immediately useful first; stop with tune/gpu_pause.sh, rerun to resume.
  1. g26 at 0.5, its app default, in the live tests (only recorded calls had seen 0.5)
  2. g26s1 at 0.5: quiz, recorded calls, live
  3. g26s2: g26 with a third seed (quiz, recorded calls), for how much a run varies
  4. g26m4: persona calls trained only on their memory moments (the whole pipeline)
Log: out/gpuq2.pipeline.log.
    uv run python gpu_queue_1024.py
"""
import json

import pipeline
from pipeline import OUT, ROOT, TUNE, exists, kill_server, run, say, server, uv

pipeline.NAME, pipeline.LOG = "gpuq2", OUT / "gpuq2.pipeline.log"
base = pipeline.BASES["gemma4a4b"]
env = {"MODEL": base["gguf"], "PROMPT": "partner"}
common = ["--repeat-guard", "--system", "partner", "--format", "gemma4", "--history-ticks", str(base["history_ticks"])]


def live(tag, gguf, p):
    if (ROOT / "tests" / "out" / f"report-bench-{tag}-amy.json").exists():
        return
    kill_server()
    say(f"live tests: {tag}")
    run(["bash", "scripts/live_bench.sh", tag, gguf, p], cwd=ROOT, env=env, log=pipeline.LOG, check=False)


# 1. g26 at 0.5, live
live("g26_p5", "tune/out/g26.gguf", "0.5")

# 2. g26s1 at 0.5
tag = "g26s1_p5"
if not exists(OUT / f"eval_yap_{tag}.json"):
    server(base["gguf"], "-np", "1", "--lora", str(OUT / "g26s1.gguf"), "--lora-init-without-apply", ctx=base["ctx"])
    run(uv("probe_answers.py", "--adapter", str(OUT / "g26s1.gguf"), "--listen-p", "0.5", *common, "--name", tag, "--out", str(OUT / f"probe_{tag}.json")),
        log=OUT / f"probe_{tag}.log")
    say(f"{tag}: quiz " + open(OUT / f"probe_{tag}.log", encoding="utf-8").read().strip().splitlines()[-1])
    run(uv("evaluate.py", "--url", "http://127.0.0.1:8080", "--calls", "data/heldout_yap.jsonl", "--n", "24", "--parallel", "1",
           "--adapter", str(OUT / "g26s1.gguf"), "--listen-p", "0.5", *common, "--out", str(OUT / f"eval_yap_{tag}.json")), log=OUT / f"eval_yap_{tag}.log")
live(tag, "tune/out/g26s1.gguf", "0.5")
names = [n for n in ("run12", "g26", "g26_p5", "g26s1", "g26s1_p5") if exists(OUT / f"eval_yap_{n}.json")]
run(uv("yap_metrics.py", "data/heldout_yap.jsonl", *[f"out/eval_yap_{n}.json" for n in names], "--first", "24", "--names", ",".join(names),
       "--md", "out/yap_g26_at_05.md"), check=False)
run(["node", "scripts/live_bench_report.js", "run12", "g26", "g26_p5", "g26s1", "g26s1_p5"], cwd=ROOT, log=OUT / "live_at_05.log", check=False)
kill_server()
say("0.5 tests done (out/yap_g26_at_05.md, tests/out/live_bench.md)")

# 3. a third seed of g26
r = json.loads((TUNE / "recipes" / "g26s1.json").read_text(encoding="utf-8"))
r.update({"name": "g26s2", "export_as": "talkme-g26s2-partner-gemma4a4b.gguf",
          "_about": "g26 with a third training seed (2026-10-24): with g26 and g26s1, how much one run of the same recipe varies."})
r["train"]["seed"] = 2
r["tests"] = {"quiz": True, "yap": 24, "ours": False, "memory": False, "live": False, "listen_p": 0.6, "compare": ["run12", "g26", "g26s1"]}
(TUNE / "recipes" / "g26s2.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
if not exists(OUT / "calls_g26s2.jsonl"):
    (OUT / "calls_g26s2.jsonl").write_bytes((OUT / "calls_g26.jsonl").read_bytes())
run(uv("pipeline.py", "--recipe", str(TUNE / "recipes" / "g26s2.json")), check=False)
say("g26s2 done")

# 4. g26m4
run(uv("pipeline.py", "--recipe", str(TUNE / "recipes" / "g26m4.json")), check=False)
say("queue done; GPU free")
