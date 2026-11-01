#!/usr/bin/env bash
# After the run 2 replay: run 3 (continue from run 2, re-weighted), export, then full held-out replays of run 3 and of
# the published adapter on one server, one call at a time (see talkme GPU notes: one slot is fastest on this model).
set -u
cd "$(dirname "$0")"
HF=../models/hf/Qwen3.6-35B-A3B
LOG=out/pipeline3.log
say() { echo "[pipeline3] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }

say "waiting for the run 2 replay"
until grep -q "^exit=" out/eval_run2.log 2>/dev/null; do sleep 20; done
say "run 2 replay ended: $(tail -1 out/eval_run2.log)"
taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 3

say "run 3: continue from run 2, 150 steps, lr 1e-4, listen_speech 0.3->0.45, hazard_start 4->8"
uv run --project . --extra train python train_lora.py --hf "$HF" --calls out/calls_run2.jsonl --cache out/cache_r2 --skip-cache \
  --init-adapter out/adapter_run2 --weight-map "0.3:0.45,4.0:8.0" --steps 150 --batch 4 --rank 32 --alpha 64 --lr 1e-4 --warmup 5 \
  --seed 3 --out out/adapter_run3 > out/run3.log 2>&1; echo "exit=$?" >> out/run3.log
grep -q "^exit=0" out/run3.log || { say "run 3 failed: $(tail -3 out/run3.log | tr '\n' ' ')"; exit 1; }
uv run --project . --extra train python export_gguf.py --adapter out/adapter_run3 --hf "$HF" --out out/run3.gguf > out/run3.export.log 2>&1 \
  || { say "export run 3 failed"; exit 1; }
say "run 3 exported"

( cd ../vendor/llama && ./llama-server.exe -m ../../models/base/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
    --lora ../../models/lora/microturn-duplex-lora.gguf,../../tune/out/run1.gguf,../../tune/out/run2.gguf,../../tune/out/run3.gguf \
    --lora-init-without-apply -ngl 99 --n-cpu-moe 8 -c 32768 -np 1 -ub 512 --port 8080 --host 127.0.0.1 --no-reasoning-preserve \
    > ../llama-server.log 2>&1 ) &
for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 3; done
for pair in "run3:out/run3.gguf" "published:../models/lora/microturn-duplex-lora.gguf"; do
  name=${pair%%:*}; ad=${pair#*:}
  say "replaying all held-out calls with $name"
  uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls ../models/lora/dataset/calls_heldout.jsonl --n 0 \
    --parallel 1 --adapter "$ad" --out "out/eval_$name.json" > "out/eval_$name.log" 2>&1; echo "exit=$?" >> "out/eval_$name.log"
  say "$name replay ended: $(tail -1 out/eval_$name.log)"
done
uv run --project . python compare.py out/eval_published.json out/eval_run2.json out/eval_run3.json --md out/compare_final.md > out/compare_final.txt 2>&1
say "table written to out/compare_final.md; server left running on 8080 with all adapters"
