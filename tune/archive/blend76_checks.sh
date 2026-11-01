#!/usr/bin/env bash
# The run 7 + run 6 blend (half each), the likeliest partner to try by hand. One step after another:
#   1. 24 recorded calls at 0.6 with the repeat guard: does it keep run 7's human timing?
#   2. live tests, 24 scenarios x 3 callers, at 0.6 (recordings in tests/out/*-bench-blend76-*.wav)
#   3. our held-out set at 0.6 with the guard: warnings, declines, answering on time
# Waits for tonight_r7_checks2.sh to finish first (by its log line).
#   bash tune/blend76_checks.sh
set -u
cd "$(dirname "$0")"
LOG=out/run7.pipeline.log
BLEND="run7.gguf:0.5,run6.gguf:0.5"
APP_BLEND="blend:talkme-run7-partner.gguf=0.5,talkme-run6.gguf=0.5"
say() { echo "[blend] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
until grep -q "checks\] .* done; GPU free (out/yap_run7_guard.md)" "$LOG"; do sleep 20; done
server() {
  taskkill //F //IM llama-server.exe > /dev/null 2>&1
  for i in $(seq 1 30); do tasklist | grep -qi llama-server || break; sleep 1; done
  ( cd ../vendor/llama && ./llama-server.exe -m "../../models/base/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" --lora ../../tune/out/run7.gguf,../../tune/out/run6.gguf \
      --lora-init-without-apply -ngl 99 --n-cpu-moe 8 -c 32768 -np 1 -ub 512 --port 8080 --host 127.0.0.1 --no-reasoning-preserve > ../llama-server.log 2>&1 ) &
  for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q '"status":"ok"' && return 0; sleep 3; done; return 1
}

if [ ! -s out/eval_yap_blend76.json ]; then
  server || { say "server did not start"; exit 1; }
  say "24 recorded calls"
  uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls data/heldout_yap.jsonl --n 24 --parallel 1 --adapter "$BLEND" \
    --listen-p 0.6 --repeat-guard --out out/eval_yap_blend76.json > out/eval_yap_blend76.log 2>&1 || say "recorded calls failed: $(tail -2 out/eval_yap_blend76.log | tr '\n' ' ')"
  taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 3
  uv run --project . python yap_metrics.py data/heldout_yap.jsonl out/eval_yap_run6_p06.json out/eval_yap_run7_p06.json out/eval_yap_run7_guard2.json \
    out/eval_yap_blend76.json --first 24 --names "run6@0.6,run7@0.6,run7 +guard,run7+run6 +guard" --md out/yap_blend76.md > /dev/null 2>&1
  say "recorded calls: out/yap_blend76.md"
fi

say "live tests"
( cd .. && bash scripts/live_bench.sh blend76 "$APP_BLEND" 0.6 2>&1 | tee -a "tune/$LOG"; node scripts/live_bench_report.js run7 blend76 > /dev/null )
taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 3

if [ ! -s out/eval_ours_blend76.json ]; then
  server || { say "server did not start"; exit 1; }
  say "our held-out set"
  uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls data/heldout_ours.jsonl --n 0 --parallel 1 --adapter "$BLEND" \
    --listen-p 0.6 --repeat-guard --out out/eval_ours_blend76.json > out/eval_ours_blend76.log 2>&1 || say "held-out failed: $(tail -2 out/eval_ours_blend76.log | tr '\n' ' ')"
  taskkill //F //IM llama-server.exe > /dev/null 2>&1
  say "ours: $(grep -A6 '== totals' out/eval_ours_blend76.log | tail -6 | tr '\n' ' ')"
fi
say "done; GPU free"
