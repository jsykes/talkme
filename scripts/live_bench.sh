#!/usr/bin/env bash
# Live audio benchmark: every scenario in tests/scenarios.json, three passes, a different caller each pass
# (US woman / UK man a bit fast / UK woman a bit slow). Needs the GPU (starts its own llama-server per pass).
#   bash scripts/live_bench.sh <name> <adapter.gguf> [threshold]     e.g. bash scripts/live_bench.sh g26b4 models/lora/talkme-partner-v1-gemma4a4b.gguf 0.6
# Then: node scripts/live_bench_report.js run2 run5 ...
set -u
cd "$(dirname "$0")/.."
name=$1; adapter=$2; thr=${3:-0.3}; smp=${4:+--sample}   # 4th argument "sample": human timing (listen decision sampled)
# tests/out/live_mode_<name> ("<threshold> [sample]") overrides both, for a pipeline that is already running
if [ -s "tests/out/live_mode_$name" ]; then read -r thr mode < "tests/out/live_mode_$name"; smp=${mode:+--sample}; echo "[bench] $name: threshold $thr${smp:+, human timing} (from tests/out/live_mode_$name)"; fi
callers=("amy piper:vits-piper-en_US-amy-medium 1.0" "george kokoro:26 1.1" "emma kokoro:21 0.9")
for c in "${callers[@]}"; do
  set -- $c
  # a pass that already finished (its report exists) is skipped, so a stopped benchmark picks up where it was
  if [ -s "tests/out/report-bench-$name-$1.json" ] && [ -z "${FORCE:-}" ]; then echo "[bench] $name, caller $1: already done"; continue; fi
  echo "[bench] $(date +%H:%M:%S) $name, caller $1"
  node scripts/audio_test.js --adapter "$adapter" --threshold "$thr" $smp ${MODEL:+--model "$MODEL"} ${PROMPT:+--prompt "$PROMPT"} --user-voice "$2" --user-speed "$3" --tag "bench-$name-$1" > "tests/out/bench-$name-$1.log" 2>&1 \
    || echo "[bench] $name $1 FAILED (see tests/out/bench-$name-$1.log)"
  tail -2 "tests/out/bench-$name-$1.log"
done
echo "[bench] $(date +%H:%M:%S) $name done"
