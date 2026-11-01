#!/usr/bin/env bash
# Run 7 "partner": a conversation partner rather than an assistant. Mostly real two-person calls (otoSpeech) with the
# person's words kept as targets (timing AND phrasing), plus enough of our data to keep the safety behaviours:
# answered cut-ins, resume and out-loud declines (run 5's special calls, all of them), hazard warnings and
# back-channels (run 5 built calls, masked prebuilt calls).
# Judged first on the real recorded conversations (yap_metrics.py); our held-out set checks warnings did not get lost.
# The same command starts and resumes it (after tune/gpu_pause.sh): finished stages are skipped.
#   bash pipeline_r7_partner.sh      (needs the whole GPU, ~8 hours)
set -u
cd "$(dirname "$0")"
HF=../models/hf/Qwen3.6-35B-A3B
LOG=out/run7.pipeline.log
NAME=run7
say() { echo "[run7] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
server() { ( cd ../vendor/llama && ./llama-server.exe -m "../../models/base/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" "$@" -ngl 99 --n-cpu-moe 8 \
    -c 32768 -np 1 -ub 512 --port 8080 --host 127.0.0.1 --no-reasoning-preserve > ../llama-server.log 2>&1 ) &
  for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && return 0; sleep 3; done; return 1; }
stop_server() { taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 3; }

if curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok; then say "something is already serving on 8080; stop it first"; exit 1; fi

# data: built once; the cache is indexed by call number, so a resume must reuse exactly this file
if [ ! -s out/calls_run7.jsonl ]; then
  say "building data: $(ls ../data/otospeech/asr | wc -l) transcribed otoSpeech conversations (words kept)"
  node ../scripts/oto_to_calls.js --words --out tune/out/calls_oto_words.jsonl --dev-out tune/out/calls_oto_words_dev.jsonl > out/oto_words_calls.log 2>&1 \
    || { say "oto_to_calls failed: $(tail -2 out/oto_words_calls.log | tr '\n' ' ')"; exit 1; }
  say "$(head -1 out/oto_words_calls.log)"
  [ -s out/calls_train_masked.jsonl ] || uv run --project . python mask_cutins.py --calls ../models/lora/dataset/calls_train.jsonl --out out/calls_train_masked.jsonl > out/mask_r7.log 2>&1
  uv run --project . python mix_calls.py --calls out/calls_oto_words.jsonl:700 out/calls_special_r5.jsonl out/calls_built_r5.jsonl:250 \
    out/calls_train_masked.jsonl:120 --exclude-ids data/exclude_dialogues.txt --seed 77 --out out/calls_run7.jsonl > out/mix_r7.log 2>&1
  say "$(grep '^wrote' out/mix_r7.log)"
else
  say "resuming with the existing out/calls_run7.jsonl"
fi

if [ ! -s out/$NAME.gguf ]; then
  # a pause can cut off the cache file being written: test the newest ones and delete any that do not load
  [ -d out/cache_r7 ] && uv run --project . --extra train python -c "
import torch, pathlib
fs = sorted(pathlib.Path('out/cache_r7').glob('call*.pt*'), key=lambda f: f.stat().st_mtime)[-40:]
bad = []
for f in fs:
    try:
        assert f.suffix == '.pt'; torch.load(f)
    except Exception:
        bad.append(f.name); f.unlink()
print('cache check: removed', bad if bad else 'nothing')
" 2>&1 | tail -1 | tee -a "$LOG"
  say "training run 7 partner (cache, then 600 steps; resumes from saved steps)"
  uv run --project . --extra train python train_lora.py --hf "$HF" --calls out/calls_run7.jsonl --cache out/cache_r7 --max-tokens 8000 \
    --chunk 160 --steps 600 --batch 4 --rank 32 --alpha 64 --lr 2e-4 --out out/adapter_run7 --resume >> out/run7.log 2>&1; echo "exit=$?" >> out/run7.log
  tail -1 out/run7.log | grep -q "^exit=0" || { say "training failed: $(tail -3 out/run7.log | tr '\n' ' ')"; exit 1; }
  uv run --project . --extra train python export_gguf.py --adapter out/adapter_run7 --hf "$HF" --out out/$NAME.gguf > out/run7.export.log 2>&1 \
    || { say "export failed"; exit 1; }
  cp out/$NAME.gguf ../models/lora/talkme-run7-partner.gguf
  say "exported out/$NAME.gguf (in the app as talkme-run7-partner.gguf)"
fi

# replays: the recorded conversations first (the partner's yardstick), then our held-out set (warnings, declines)
todo=""
# a human-timed adapter is judged with human timing (sampled listen decision) and a looser 0.6 threshold, not 0.3
for set in "recorded data/heldout_yap.jsonl eval_yap_run7 --sample" "recorded data/heldout_yap.jsonl eval_yap_run7_p06 --listen-p 0.6" "ours data/heldout_ours.jsonl eval_ours_run7 --sample"; do
  set -- $set; [ -s "out/$3.json" ] || todo="$todo|$set"
done
if [ -n "$todo" ]; then
  server --lora ../../tune/out/$NAME.gguf,../../tune/out/run6.gguf,../../tune/out/run5.gguf,../../tune/out/run2.gguf --lora-init-without-apply \
    || { say "server did not start"; exit 1; }
  IFS='|' read -ra sets <<< "${todo#|}"
  for set in "${sets[@]}"; do  # each: kind calls output [evaluate options]
    set -- $set; kind=$1; calls=$2; out=$3; shift 3
    say "replaying the $kind held-out set with run 7 ($*)"
    uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls "$calls" --n 0 --parallel 1 --adapter out/$NAME.gguf "$@" \
      --out "out/$out.json" > "out/$out.log" 2>&1 || say "replay $kind failed: $(tail -2 out/$out.log | tr '\n' ' ')"
    if [ "$kind" = recorded ]; then
      p06=$( [ -s out/eval_yap_run7_p06.json ] && echo out/eval_yap_run7_p06.json )
      uv run --project . python yap_metrics.py data/heldout_yap.jsonl out/eval_yap_run2.json out/eval_yap_run5.json out/eval_yap_run7.json $p06 \
        --names "run2@0.3,run5@0.3,run7 sampled${p06:+,run7@0.6}" --md out/yap_run7.md > /dev/null 2>&1
      uv run --project . python yap_metrics.py data/heldout_yap.jsonl out/eval_yap_run5.json out/eval_yap_run6_p06.json out/eval_yap_run7.json $p06 --first 24 \
        --names "run5@0.3,run6@0.6,run7 sampled${p06:+,run7@0.6}" --md out/yap_run7_vs_run6.md > /dev/null 2>&1
      say "recorded conversations: out/yap_run7.md, out/yap_run7_vs_run6.md"
    fi
  done
  stop_server
fi
uv run --project . python extra_metrics.py out/eval_ours_run2.json out/eval_ours_run5.json out/eval_ours_run6.json out/eval_ours_run7.json > out/extra_run7.txt 2>&1
say "held-out replays done: out/yap_run7.md, out/extra_run7.txt"

cd ..
bash scripts/live_bench.sh run7 tune/out/$NAME.gguf 0.3 sample 2>&1 | tee -a "tune/$LOG"
node scripts/live_bench_report.js run6 run7 > /dev/null
taskkill //F //IM llama-server.exe > /dev/null 2>&1
echo "[run7] $(date +%H:%M:%S) done (live report: tests/out/live_bench.md)" | tee -a "tune/$LOG"
