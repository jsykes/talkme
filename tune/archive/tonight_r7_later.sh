#!/usr/bin/env bash
# The run 7 checks left for after trying it by hand (~1.5 h of GPU):
#   run 7 on our held-out set (threshold 0.6, repeat guard): did warnings and out-loud declines survive?
#   run 6 live tests at 0.6 (24 scenarios x 3 callers), to compare with run 7's
#   bash tune/tonight_r7_later.sh
set -u
cd "$(dirname "$0")"
LOG=out/run7.pipeline.log
say() { echo "[later] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 3
if [ ! -s out/eval_ours_run7.json ]; then
  ( cd ../vendor/llama && ./llama-server.exe -m "../../models/base/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" --lora ../../tune/out/run7.gguf --lora-init-without-apply \
      -ngl 99 --n-cpu-moe 8 -c 32768 -np 1 -ub 512 --port 8080 --host 127.0.0.1 --no-reasoning-preserve > ../llama-server.log 2>&1 ) &
  for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 3; done
  say "run 7 on our held-out set"
  uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls data/heldout_ours.jsonl --n 0 --parallel 1 --adapter out/run7.gguf \
    --listen-p 0.6 --repeat-guard --out out/eval_ours_run7.json > out/eval_ours_run7.log 2>&1
  taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 3
  say "ours: $(grep -A6 '== totals' out/eval_ours_run7.log | tail -6 | tr '\n' ' ')"
fi
cd ..
echo "0.6" > tests/out/live_mode_run6
say "run 6 live tests"
bash scripts/live_bench.sh run6 tune/out/run6.gguf 2>&1 | tee -a "tune/$LOG"
node scripts/live_bench_report.js run6 run7 > /dev/null
taskkill //F //IM llama-server.exe > /dev/null 2>&1
echo "[later] $(date +%H:%M:%S) done (tests/out/live_bench.md); GPU free" | tee -a "tune/$LOG"
