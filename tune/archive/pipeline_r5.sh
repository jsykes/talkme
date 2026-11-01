#!/usr/bin/env bash
# Run 5: every call built from the dialogues, with cut-ins answered.
#   - answer_cutins.py gives every cut-in in the dialogues an answer (the published calls never answered one, which
#     taught silence after an interruption and blocked resuming); the author's prebuilt calls are therefore left out
#   - keeps run 4's clean data, resume/decline dialogues and speaker profiles, with narrower pauses
# Stages: answer cut-ins (base model) -> build -> cache + train -> export -> held-out replay -> live tests.
#   bash pipeline_r5.sh            (needs the whole GPU for ~6 hours; close the app and other GPU jobs)
set -u
cd "$(dirname "$0")"
HF=../models/hf/Qwen3.6-35B-A3B
LOG=out/run5.pipeline.log
say() { echo "[run5] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
server() { ( cd ../vendor/llama && ./llama-server.exe -m "../../models/base/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" "$@" -ngl 99 --n-cpu-moe 8 \
    -c 32768 -np 1 -ub 512 --port 8080 --host 127.0.0.1 --no-reasoning-preserve > ../llama-server.log 2>&1 ) &
  for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && return 0; sleep 3; done; return 1; }

if curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok; then say "something is already serving on 8080; stop it first"; exit 1; fi

say "answering cut-ins with the base model"
server || { say "server did not start"; exit 1; }
uv run --project . python answer_cutins.py --url http://127.0.0.1:8080 --exclude-ids data/exclude_dialogues.txt \
  --dialogues ../models/lora/dataset/dialogues_short.jsonl ../models/lora/dataset/dialogues_long.jsonl --outdir out > out/answer_cutins.log 2>&1
taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 3
say "$(grep -E 'cut-ins answered' out/answer_cutins.log | tr '\n' ' ')"

say "building calls (all from dialogues)"
uv run --project . python build_calls.py --dialogues out/dialogues_short_answered.jsonl out/dialogues_long_answered.jsonl \
  --exclude-calls ../models/lora/dataset/calls_heldout.jsonl --exclude-ids data/exclude_dialogues.txt \
  --per 3 --hazard-boost 1 --timing mixed --max-ticks 460 --seed 57 --out out/calls_built_r5.jsonl > out/build_r5.log 2>&1
uv run --project . python build_calls.py --dialogues out/dial_special.jsonl --exclude-ids data/exclude_dialogues.txt \
  --per 2 --timing mixed --max-ticks 460 --seed 58 --out out/calls_special_r5.jsonl >> out/build_r5.log 2>&1
uv run --project . python mix_calls.py --calls out/calls_built_r5.jsonl out/calls_special_r5.jsonl --out out/calls_run5.jsonl > out/mix_r5.log 2>&1
say "$(grep '^wrote' out/mix_r5.log)"

say "training run 5 (cache, then 500 steps)"
uv run --project . --extra train python train_lora.py --hf "$HF" --calls out/calls_run5.jsonl --cache out/cache_r5 --max-tokens 8000 \
  --chunk 160 --steps 500 --batch 4 --rank 32 --alpha 64 --lr 2e-4 --out out/adapter_run5 > out/run5.log 2>&1; echo "exit=$?" >> out/run5.log
grep -q "^exit=0" out/run5.log || { say "training failed: $(tail -3 out/run5.log | tr '\n' ' ')"; exit 1; }
uv run --project . --extra train python export_gguf.py --adapter out/adapter_run5 --hf "$HF" --out out/run5.gguf > out/run5.export.log 2>&1 \
  || { say "export failed"; exit 1; }
say "exported out/run5.gguf"

say "replaying all 28 held-out calls with run 5"
server --lora ../../tune/out/run5.gguf,../../tune/out/run2.gguf,../../tune/out/run4.gguf --lora-init-without-apply || { say "server did not start"; exit 1; }
uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls ../models/lora/dataset/calls_heldout.jsonl --n 0 \
  --parallel 1 --adapter out/run5.gguf --out out/eval_run5.json > out/eval_run5.log 2>&1
uv run --project . python compare.py out/eval_published.json out/eval_run2.json out/eval_run4.json out/eval_run5.json --md out/compare_run5.md > /dev/null 2>&1
uv run --project . python extra_metrics.py out/eval_published.json out/eval_run2.json out/eval_run4.json out/eval_run5.json > out/extra_run5.txt 2>&1
say "replay done: out/compare_run5.md, out/extra_run5.txt"

say "live audio tests"
cd ..
for p in 1 2 3; do
  node scripts/audio_test.js --adapter tune/out/run5.gguf --scenario story-resume,decline --tag new-run5-$p > tests/out/new-run5-$p.log 2>&1
  echo "[run5] $(date +%H:%M:%S) resume/decline pass $p: $(grep '^checks ' tests/out/new-run5-$p.log)" | tee -a "tune/$LOG"
done
node scripts/audio_test.js --adapter tune/out/run5.gguf --scenario bleach,piano,car-story,barge-in,keep-going,quick-question --tag run5-old6 > tests/out/run5-old6.log 2>&1
echo "[run5] $(date +%H:%M:%S) original six: $(grep '^checks ' tests/out/run5-old6.log)" | tee -a "tune/$LOG"
taskkill //F //IM llama-server.exe > /dev/null 2>&1
echo "[run5] $(date +%H:%M:%S) done" | tee -a "tune/$LOG"
