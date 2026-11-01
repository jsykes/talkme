"""Single BOS vs double: until 2026-10-24 every request to a Gemma adapter started with two BOS tokens (ours and the one
llama-server adds), where training had one. g26 at 0.6 again with one: the quiz and the 24 recorded calls, on port 8095
with its own server (the app's server and voice servers are left alone). Compare with probe_g26 / eval_yap_g26.
    uv run python bos_check.py
"""
import os
import subprocess
import time
import urllib.request

import pipeline
from pipeline import OUT, ROOT, run, say, uv

pipeline.NAME, pipeline.LOG = "bos", OUT / "bos.pipeline.log"
base = pipeline.BASES["gemma4a4b"]
ad = str(OUT / "g26.gguf")
url = "http://127.0.0.1:8095"
srv = subprocess.Popen([str(ROOT / "vendor" / "llama" / ("llama-server.exe" if os.name == "nt" else "llama-server")), "-m", str(ROOT / base["gguf"]), "-np", "1", "--lora", ad,
                        "--lora-init-without-apply", "-ngl", "99", "--n-cpu-moe", "8", "-c", str(base["ctx"]), "-ub", "512", "--port", "8095",
                        "--host", "127.0.0.1", "--no-reasoning-preserve"], cwd=str(ROOT / "vendor" / "llama"),
                       stdout=open(OUT / "bos_server.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(200):
        try:
            if b'"status":"ok"' in urllib.request.urlopen(url + "/health", timeout=2).read():
                break
        except Exception:
            pass
        time.sleep(3)
    common = ["--url", url, "--listen-p", "0.6", "--repeat-guard", "--system", "partner", "--format", "gemma4", "--history-ticks", "400"]
    run(uv("probe_answers.py", "--adapter", ad, *common, "--name", "g26_1bos", "--out", str(OUT / "probe_g26_1bos.json")), log=OUT / "probe_g26_1bos.log")
    say("quiz with one BOS: " + open(OUT / "probe_g26_1bos.log", encoding="utf-8").read().strip().splitlines()[-1])
    run(uv("evaluate.py", *common, "--calls", "data/heldout_yap.jsonl", "--n", "24", "--parallel", "1", "--adapter", ad,
           "--out", str(OUT / "eval_yap_g26_1bos.json")), log=OUT / "eval_yap_g26_1bos.log")
    run(uv("yap_metrics.py", "data/heldout_yap.jsonl", "out/eval_yap_run12.json", "out/eval_yap_g26.json", "out/eval_yap_g26_1bos.json",
           "--first", "24", "--names", "run12,g26 (two BOS),g26 (one BOS)", "--md", "out/yap_bos.md"), check=False)
    say("recorded calls with one BOS: out/yap_bos.md")
finally:
    srv.kill()  # only this server
