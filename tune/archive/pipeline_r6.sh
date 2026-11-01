#!/usr/bin/env bash
# Run 6: run 5's data (answered cut-ins, resume/decline, speaker profiles) + the author's prebuilt calls for
# back-channels and warnings, with the silence after their unanswered cut-in questions weighted zero
# (mask_cutins.py) + real conversation timing from otoSpeech (timing_only: loss on the first reply token only).
# Stages: build data (CPU) -> cache + train -> export -> three held-out replays -> live benchmark (runs 6, 5, 2).
#   bash pipeline_r6.sh      (needs the whole GPU for ~9 hours; uses whatever otoSpeech conversations are transcribed so far,
#   ~80 are enough for the 350 calls it samples)
set -u
cd "$(dirname "$0")"
HF=../models/hf/Qwen3.6-35B-A3B
LOG=out/run6.pipeline.log
say() { echo "[run6] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
server() { ( cd ../vendor/llama && ./llama-server.exe -m "../../models/base/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" "$@" -ngl 99 --n-cpu-moe 8 \
    -c 32768 -np 1 -ub 512 --port 8080 --host 127.0.0.1 --no-reasoning-preserve > ../llama-server.log 2>&1 ) &
  for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && return 0; sleep 3; done; return 1; }
stop_server() { taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 3; }

if curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok; then say "something is already serving on 8080; stop it first"; exit 1; fi

say "building data: $(ls ../data/otospeech/asr | wc -l) transcribed otoSpeech conversations"
node ../scripts/oto_to_calls.js --out tune/out/calls_oto.jsonl --dev-out tune/out/calls_oto_dev.jsonl > out/oto_calls.log 2>&1 \
  || { say "oto_to_calls failed: $(tail -2 out/oto_calls.log | tr '\n' ' ')"; exit 1; }
say "$(head -1 out/oto_calls.log)"
uv run --project . python mask_cutins.py --calls ../models/lora/dataset/calls_train.jsonl --out out/calls_train_masked.jsonl > out/mask_r6.log 2>&1
say "$(tail -1 out/mask_r6.log)"
uv run --project . python mix_calls.py --calls out/calls_run5.jsonl out/calls_train_masked.jsonl:260 out/calls_oto.jsonl:350 \
  --exclude-ids data/exclude_dialogues.txt --seed 66 --out out/calls_run6.jsonl > out/mix_r6.log 2>&1
say "$(grep '^wrote' out/mix_r6.log)"

say "training run 6 (cache, then 600 steps)"
uv run --project . --extra train python train_lora.py --hf "$HF" --calls out/calls_run6.jsonl --cache out/cache_r6 --max-tokens 8000 \
  --chunk 160 --steps 600 --batch 4 --rank 32 --alpha 64 --lr 2e-4 --out out/adapter_run6 > out/run6.log 2>&1; echo "exit=$?" >> out/run6.log
grep -q "^exit=0" out/run6.log || { say "training failed: $(tail -3 out/run6.log | tr '\n' ' ')"; exit 1; }
uv run --project . --extra train python export_gguf.py --adapter out/adapter_run6 --hf "$HF" --out out/run6.gguf > out/run6.export.log 2>&1 \
  || { say "export failed"; exit 1; }
cp out/run6.gguf ../models/lora/talkme-run6.gguf
say "exported out/run6.gguf (in the app as talkme-run6.gguf)"

server --lora ../../tune/out/run6.gguf,../../tune/out/run5.gguf,../../tune/out/run2.gguf,../../tune/out/run4.gguf,../../models/lora/microturn-duplex-lora.gguf \
  --lora-init-without-apply || { say "server did not start"; exit 1; }
for set in "published ../models/lora/dataset/calls_heldout.jsonl eval_run6" "ours data/heldout_ours.jsonl eval_ours_run6" "recorded data/heldout_yap.jsonl eval_yap_run6"; do
  set -- $set
  say "replaying the $1 held-out set with run 6"
  uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls "$2" --n 0 --parallel 1 --adapter out/run6.gguf \
    --out "out/$3.json" > "out/$3.log" 2>&1 || say "replay $1 failed: $(tail -2 out/$3.log | tr '\n' ' ')"
done
stop_server
uv run --project . python compare.py out/eval_published.json out/eval_run2.json out/eval_run4.json out/eval_run5.json out/eval_run6.json --md out/compare_run6.md > /dev/null 2>&1
uv run --project . python extra_metrics.py out/eval_published.json out/eval_run2.json out/eval_run5.json out/eval_run6.json \
  out/eval_ours_run2.json out/eval_ours_run5.json out/eval_ours_run6.json > out/extra_run6.txt 2>&1
uv run --project . python yap_metrics.py data/heldout_yap.jsonl out/eval_yap_run2.json out/eval_yap_run5.json out/eval_yap_run6.json --md out/yap_run6.md > /dev/null 2>&1
say "held-out replays done: out/compare_run6.md, out/extra_run6.txt, out/yap_run6.md"

cd ..
bash scripts/live_bench.sh run6 tune/out/run6.gguf 2>&1 | tee -a "tune/$LOG"
bash scripts/live_bench.sh run5 tune/out/run5.gguf 2>&1 | tee -a "tune/$LOG"
bash scripts/live_bench.sh run2 models/lora/talkme-run2.gguf 2>&1 | tee -a "tune/$LOG"
node scripts/live_bench_report.js run2 run5 run6 > /dev/null
taskkill //F //IM llama-server.exe > /dev/null 2>&1
echo "[run6] $(date +%H:%M:%S) done (live report: tests/out/live_bench.md)" | tee -a "tune/$LOG"
