#!/usr/bin/env bash
# Warm and playful v2: same dialogues, hazards thinned to run 2's density, 250 run 2 anchors. Runs after the blend replays.
set -u
cd "$(dirname "$0")"
HF=../models/hf/Qwen3.6-35B-A3B
LOG=out/styles.log
say() { echo "[v2] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
until grep -q "blends finished" "$LOG"; do sleep 15; done
taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 3
for st in warm playful; do
  say "training ${st}v2"
  uv run --project . --extra train python train_lora.py --hf "$HF" --calls "out/calls_${st}v2.jsonl" --cache "out/cache_${st}v2" --chunk 160 \
    --init-adapter out/adapter_run2 --steps 120 --batch 4 --rank 32 --alpha 64 --lr 1e-4 --warmup 5 --seed 6 \
    --out "out/adapter_${st}v2" > "out/train_${st}v2.log" 2>&1; echo "exit=$?" >> "out/train_${st}v2.log"
  grep -q "^exit=0" "out/train_${st}v2.log" || { say "${st}v2 failed: $(tail -2 out/train_${st}v2.log | tr '\n' ' ')"; continue; }
  uv run --project . --extra train python export_gguf.py --adapter "out/adapter_${st}v2" --hf "$HF" --out "out/${st}v2.gguf" > "out/export_${st}v2.log" 2>&1 \
    && say "${st}v2 exported" || say "export ${st}v2 failed"
done
( cd ../vendor/llama && ./llama-server.exe -m ../../models/base/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
    --lora ../../tune/out/run2.gguf,../../tune/out/brisk.gguf,../../tune/out/warmv2.gguf,../../tune/out/playfulv2.gguf \
    --lora-init-without-apply -ngl 99 --n-cpu-moe 8 -c 32768 -np 1 -ub 512 --port 8080 --host 127.0.0.1 --no-reasoning-preserve \
    > ../llama-server.log 2>&1 ) &
for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 3; done
for st in warmv2 playfulv2; do
  [ -f "out/${st}.gguf" ] || continue
  say "replaying 8 held-out calls with $st"
  uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls ../models/lora/dataset/calls_heldout.jsonl --stratified 4 \
    --parallel 1 --adapter "out/${st}.gguf" --out "out/eval8_${st}.json" > "out/eval8_${st}.log" 2>&1
done
say "v2 finished; server left running on 8080"
