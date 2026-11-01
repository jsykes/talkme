#!/usr/bin/env bash
# Listen-threshold sweep for run 3 on the 14 screening calls, then full replays of the best setting and of the published
# adapter, then the final table. Uses the server already on 8080 (one slot, all adapters loaded).
set -u
cd "$(dirname "$0")"
LOG=out/sweep.log
say() { echo "[sweep] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
for thr in 0.45 0.6; do
  say "run3 at listen threshold $thr on 14 calls"
  uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls ../models/lora/dataset/calls_heldout.jsonl --stratified 7 \
    --parallel 1 --listen-p "$thr" --adapter out/run3.gguf --out "out/eval14_run3_t$thr.json" > "out/eval14_run3_t$thr.log" 2>&1
  say "done: $(uv run --project . python compare_logs.py out/eval_run2.log out/eval14_run3_t$thr.log | tail -7 | tr -s ' ' | tr '\n' ';')"
done
say "sweep finished"
