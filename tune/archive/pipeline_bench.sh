#!/usr/bin/env bash
# Benchmark closer to real use, run after run 5 (needs the whole GPU, ~4 h; stop any time with: taskkill //F //IM llama-server.exe
# and kill this script; finished parts keep their results).
#   1. replay our own held-out set (data/heldout_ours.jsonl: 26 calls, Nemotron timing, speaker profiles, answered cut-ins)
#      with run 5 and run 2
#   1b. replay real recorded conversations (data/heldout_yap.jsonl, Open Yap 1K sample through the app's recogniser)
#      with run 5 and run 2: does it take the turn where a real person did
#   2. live audio benchmark (24 scenarios x 3 callers) for run 5 and run 2, both with the current app code
#   3. the same replay for the published adapter and run 4 (lowest priority, last)
#   bash pipeline_bench.sh
set -u
cd "$(dirname "$0")"
LOG=out/bench.pipeline.log
say() { echo "[bench] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
server() { ( cd ../vendor/llama && ./llama-server.exe -m "../../models/base/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" "$@" -ngl 99 --n-cpu-moe 8 \
    -c 32768 -np 1 -ub 512 --port 8080 --host 127.0.0.1 --no-reasoning-preserve > ../llama-server.log 2>&1 ) &
  for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && return 0; sleep 3; done; return 1; }
stop_server() { taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 3; }
ADAPTERS=../../tune/out/run5.gguf,../../tune/out/run2.gguf,../../tune/out/run4.gguf,../../models/lora/microturn-duplex-lora.gguf
replay() { # name adapter
  say "replaying our held-out set with $1"
  uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls data/heldout_ours.jsonl --n 0 --parallel 1 \
    --adapter "$2" --out "out/eval_ours_$1.json" > "out/eval_ours_$1.log" 2>&1 || say "replay $1 failed: $(tail -2 out/eval_ours_$1.log | tr '\n' ' ')"
  say "$1: $(grep -E '^(start|overall|summary)' out/eval_ours_$1.log | tail -1)"
}

if curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok; then say "something is already serving on 8080; stop it first"; exit 1; fi
server --lora "$ADAPTERS" --lora-init-without-apply || { say "server did not start"; exit 1; }
replay run5 out/run5.gguf
replay run2 out/run2.gguf
for i in $(seq 1 240); do [ -s data/heldout_yap.jsonl ] && break; sleep 30; done
for r in "run5 out/run5.gguf" "run2 out/run2.gguf"; do
  set -- $r
  say "replaying the recorded conversations with $1"
  uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls data/heldout_yap.jsonl --n 0 --parallel 1     --adapter "$2" --out "out/eval_yap_$1.json" > "out/eval_yap_$1.log" 2>&1 || say "yap replay $1 failed: $(tail -2 out/eval_yap_$1.log | tr '\n' ' ')"
done
uv run --project . python yap_metrics.py data/heldout_yap.jsonl out/eval_yap_run2.json out/eval_yap_run5.json --md out/yap.md > /dev/null 2>&1
say "recorded conversations: out/yap.md"
stop_server
uv run --project . python compare.py out/eval_ours_run2.json out/eval_ours_run5.json --md out/compare_ours.md > /dev/null 2>&1
uv run --project . python extra_metrics.py out/eval_ours_run2.json out/eval_ours_run5.json > out/extra_ours.txt 2>&1
say "held-out (ours) comparison: out/compare_ours.md"

cd ..
bash scripts/live_bench.sh run5 tune/out/run5.gguf 2>&1 | tee -a "tune/$LOG"
bash scripts/live_bench.sh run2 models/lora/talkme-run2.gguf 2>&1 | tee -a "tune/$LOG"
node scripts/live_bench_report.js run2 run5 > /dev/null && echo "[bench] $(date +%H:%M:%S) live report: tests/out/live_bench.md" | tee -a "tune/$LOG"
cd tune

server --lora "$ADAPTERS" --lora-init-without-apply || { say "server did not start"; exit 1; }
replay published ../models/lora/microturn-duplex-lora.gguf
replay run4 out/run4.gguf
stop_server
uv run --project . python compare.py out/eval_ours_published.json out/eval_ours_run2.json out/eval_ours_run4.json out/eval_ours_run5.json --md out/compare_ours.md > /dev/null 2>&1
uv run --project . python extra_metrics.py out/eval_ours_published.json out/eval_ours_run2.json out/eval_ours_run4.json out/eval_ours_run5.json > out/extra_ours.txt 2>&1
say "done"
