#!/usr/bin/env bash
# Run 1 -> export -> held-out replays (published, run 1) on the whole GPU -> run 2 (cache + train) -> export -> replay.
# Each stage logs to out/pipeline.log; a stage that fails stops the pipeline.
set -u
cd "$(dirname "$0")"
ROOT=$(cd .. && pwd)
HF=../models/hf/Qwen3.6-35B-A3B
PUB=../models/lora/microturn-duplex-lora.gguf
LOG=out/pipeline.log
say() { echo "[pipeline] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }

say "waiting for run 1"
until grep -q "^exit=" out/run1.log 2>/dev/null; do sleep 30; done
grep -q "^exit=0" out/run1.log || { say "run 1 failed: $(tail -3 out/run1.log | tr '\n' ' ')"; exit 1; }

say "exporting run 1"
uv run --project . --extra train python export_gguf.py --adapter out/adapter_run1 --hf "$HF" --out out/run1.gguf > out/run1.export.log 2>&1 \
  || { say "export failed: $(tail -3 out/run1.export.log | tr '\n' ' ')"; exit 1; }

say "starting llama-server for replays (whole GPU, 4 slots, both adapters)"
( cd ../vendor/llama && ./llama-server.exe -m ../../models/base/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --lora ../../models/lora/microturn-duplex-lora.gguf \
    --lora ../../tune/out/run1.gguf --lora-init-without-apply -ngl 99 --n-cpu-moe 10 -c 114688 -np 4 -ub 512 \
    --port 8080 --host 127.0.0.1 --no-reasoning-preserve > ../llama-server.log 2>&1 ) &
for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 3; done
curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok || { say "server did not start"; exit 1; }

for name in published run1; do
  ad=$PUB; [ "$name" = run1 ] && ad=out/run1.gguf
  say "replaying held-out calls with $name"
  uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls ../models/lora/dataset/calls_heldout.jsonl --n 0 \
    --parallel 4 --adapter "$ad" --out "out/eval_$name.json" > "out/eval_$name.log" 2>&1 || { say "replay $name failed"; }
  say "$name: $(grep -A8 '== totals ==' out/eval_$name.log | tr '\n' ' ')"
done
uv run --project . python compare.py out/eval_published.json out/eval_run1.json --md out/compare_run1.md > /dev/null 2>&1
say "comparison written to out/compare_run1.md"

say "stopping replay server"
taskkill //F //IM llama-server.exe > /dev/null 2>&1
sleep 3

say "run 2 (cache in chunks of 160, then 600 steps)"
uv run --project . --extra train python train_lora.py --hf "$HF" --calls out/calls_run2.jsonl --cache out/cache_r2 --max-tokens 8000   --chunk 160 --steps 600 --batch 4 --rank 32 --alpha 64 --lr 2e-4 --out out/adapter_run2 > out/run2.log 2>&1; echo "exit=$?" >> out/run2.log
grep -q "^exit=0" out/run2.log || { say "run 2 failed: $(tail -3 out/run2.log | tr '
' ' ')"; exit 1; }
say "exporting run 2"
uv run --project . --extra train python export_gguf.py --adapter out/adapter_run2 --hf "$HF" --out out/run2.gguf > out/run2.export.log 2>&1   || { say "export run 2 failed"; exit 1; }

say "starting llama-server for the run 2 replay"
( cd ../vendor/llama && ./llama-server.exe -m ../../models/base/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --lora ../../models/lora/microturn-duplex-lora.gguf,../../tune/out/run1.gguf,../../tune/out/run2.gguf --lora-init-without-apply -ngl 99 --n-cpu-moe 10 -c 114688 -np 4 -ub 512     --port 8080 --host 127.0.0.1 --no-reasoning-preserve > ../llama-server.log 2>&1 ) &
for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 3; done
say "replaying held-out calls with run2"
uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls ../models/lora/dataset/calls_heldout.jsonl --n 0   --parallel 4 --adapter out/run2.gguf --out out/eval_run2.json > out/eval_run2.log 2>&1 || say "replay run2 failed"
say "run2: $(grep -A8 '== totals ==' out/eval_run2.log | tr '
' ' ')"
uv run --project . python compare.py out/eval_published.json out/eval_run1.json out/eval_run2.json --md out/compare_run2.md > /dev/null 2>&1
say "comparison written to out/compare_run2.md; server left running on 8080 with all three adapters"
