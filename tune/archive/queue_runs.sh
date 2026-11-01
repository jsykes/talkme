#!/usr/bin/env bash
# Waits for a training run to finish, exports it to GGUF, then starts the next run, so the GPU never idles.
#   bash queue_runs.sh <log of the running run> <adapter dir of the running run> <gguf name> -- <train_lora.py args of the next run>
set -u
cd "$(dirname "$0")"
LOG=$1; ADAPTER=$2; GGUF=$3; shift 3; [ "${1:-}" = "--" ] && shift
HF=../models/hf/Qwen3.6-35B-A3B
until grep -q "^exit=" "$LOG" 2>/dev/null; do sleep 30; done
if grep -q "^exit=0" "$LOG"; then
  echo "[queue] $(date +%H:%M) run finished; exporting $ADAPTER -> out/$GGUF"
  uv run --project . --extra train python export_gguf.py --adapter "$ADAPTER" --hf "$HF" --out "out/$GGUF" > "out/${GGUF%.gguf}.export.log" 2>&1 \
    && echo "[queue] exported out/$GGUF" || echo "[queue] export FAILED (see out/${GGUF%.gguf}.export.log)"
else
  echo "[queue] run failed: $(tail -3 "$LOG")"
fi
if [ $# -gt 0 ]; then
  NEXT_LOG=$(printf '%s\n' "$@" | grep -A1 -- '--out' | tail -1).log
  echo "[queue] $(date +%H:%M) starting next run, log $NEXT_LOG"
  uv run --project . --extra train python train_lora.py "$@" > "$NEXT_LOG" 2>&1; echo "exit=$?" >> "$NEXT_LOG"
  echo "[queue] $(date +%H:%M) next run ended: $(tail -1 "$NEXT_LOG")"
fi
