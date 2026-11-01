#!/usr/bin/env bash
# Run 4: run 2's recipe from scratch, with cleaner data and two new behaviours.
#   - drops every dialogue the dataset card flags (data/exclude_dialogues.txt) and every published call containing one
#   - adds ~150 generated dialogues where an interrupted answer is resumed and a request is declined out loud
#   - long calls split with the fixed splitter (no piece opens with the assistant talking)
# Stages: generate (base model on llama-server) -> build -> cache + train 600 steps -> export -> held-out replay vs run 2.
#   bash pipeline_r4.sh            (needs the whole GPU for ~6 hours; the app and other GPU jobs must be closed)
set -u
cd "$(dirname "$0")"
HF=../models/hf/Qwen3.6-35B-A3B
BASE=../models/base/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
LOG=out/run4.pipeline.log
say() { echo "[run4] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
server() { ( cd ../vendor/llama && ./llama-server.exe -m "../../models/base/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" "$@" -ngl 99 --n-cpu-moe 8 \
    -c 32768 -np 1 -ub 512 --port 8080 --host 127.0.0.1 --no-reasoning-preserve > ../llama-server.log 2>&1 ) &
  for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && return 0; sleep 3; done; return 1; }

if curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok; then say "something is already serving on 8080; stop it first"; exit 1; fi

if [ ! -s out/dial_special.jsonl ] || [ "$(wc -l < out/dial_special.jsonl)" -lt 150 ]; then
  say "generating resume/decline dialogues with the base model"
  server || { say "server did not start"; exit 1; }
  uv run --project . python gen_dialogues.py --kind special --count $((150 - $( [ -f out/dial_special.jsonl ] && wc -l < out/dial_special.jsonl || echo 0 ))) \
    --seed 41 --temperature 0.9 --out out/dial_special.jsonl >> out/gen_special.log 2>&1
  taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 3
fi
say "special dialogues: $(wc -l < out/dial_special.jsonl)"

say "building calls"
uv run --project . python build_calls.py --dialogues ../models/lora/dataset/dialogues_short.jsonl ../models/lora/dataset/dialogues_long.jsonl \
  --exclude-calls ../models/lora/dataset/calls_heldout.jsonl --exclude-ids data/exclude_dialogues.txt \
  --per 2 --hazard-boost 1 --timing mixed --max-ticks 460 --seed 47 --out out/calls_built_r4.jsonl > out/build_r4.log 2>&1
uv run --project . python build_calls.py --dialogues out/dial_special.jsonl --exclude-ids data/exclude_dialogues.txt \
  --per 2 --timing mixed --max-ticks 460 --seed 48 --out out/calls_special_r4.jsonl >> out/build_r4.log 2>&1
uv run --project . python mix_calls.py --calls ../models/lora/dataset/calls_train.jsonl out/calls_built_r4.jsonl out/calls_special_r4.jsonl \
  --exclude-ids data/exclude_dialogues.txt --out out/calls_run4.jsonl > out/mix_r4.log 2>&1
say "$(grep '^wrote' out/mix_r4.log)"

say "training run 4 (cache, then 600 steps)"
uv run --project . --extra train python train_lora.py --hf "$HF" --calls out/calls_run4.jsonl --cache out/cache_r4 --max-tokens 8000 \
  --chunk 160 --steps 600 --batch 4 --rank 32 --alpha 64 --lr 2e-4 --out out/adapter_run4 > out/run4.log 2>&1; echo "exit=$?" >> out/run4.log
grep -q "^exit=0" out/run4.log || { say "training failed: $(tail -3 out/run4.log | tr '\n' ' ')"; exit 1; }
uv run --project . --extra train python export_gguf.py --adapter out/adapter_run4 --hf "$HF" --out out/run4.gguf > out/run4.export.log 2>&1 \
  || { say "export failed"; exit 1; }
say "exported out/run4.gguf"

say "replaying all 28 held-out calls with run 4"
server --lora ../../tune/out/run4.gguf,../../tune/out/run2.gguf --lora-init-without-apply || { say "server did not start"; exit 1; }
uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls ../models/lora/dataset/calls_heldout.jsonl --n 0 \
  --parallel 1 --adapter out/run4.gguf --out out/eval_run4.json > out/eval_run4.log 2>&1
uv run --project . python compare.py out/eval_published.json out/eval_run2.json out/eval_run4.json --md out/compare_run4.md > /dev/null 2>&1
uv run --project . python extra_metrics.py out/eval_published.json out/eval_run2.json out/eval_run4.json > out/extra_run4.txt 2>&1
taskkill //F //IM llama-server.exe > /dev/null 2>&1
say "done: out/compare_run4.md, out/extra_run4.txt"
