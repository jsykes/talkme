#!/usr/bin/env bash
# Run 8 "partner": a human-feeling conversation partner that still knows things.
#   - words: ~520 friend-style dialogues written by the base model (gen_dialogues.py --kind partner): reacts, asks back,
#     nudges after a silence, tells stories through the user's mm-hms, answers facts correctly
#   - timing: all 538 otoSpeech conversations, timing only (when to talk, never what strangers said)
#   - more back-channels: denser ones in the partner calls (every ~3.6 s of user speech), continuer weight 2.5
#   - kept: run 5's resume and decline calls (all), answered cut-ins and warnings, masked prebuilt back-channels
# Judged on the recorded conversations and the live tests (incl. quiet-nudge, long-talk, story-hold), at 0.6 + guard.
# The same command starts and resumes it (after tune/gpu_pause.sh): finished stages are skipped.
#   bash pipeline_r8_partner.sh
set -u
cd "$(dirname "$0")"
# base: HauhauCS's abliterated Qwen3.6-35B-A3B (full precision rebuilt by hauhau_rebuild.py; its Q4_K_M GGUF for
# replays, live tests and the app). The adapter only fits that base: talkme-run8-partner-hauhau.gguf.
HF=../models/hf/Qwen3.6-35B-A3B-hauhau
BASE_GGUF=models/base/hauhau/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
LOG=out/run8.pipeline.log
NAME=run8
say() { echo "[run8] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
server() {
  taskkill //F //IM llama-server.exe > /dev/null 2>&1
  for i in $(seq 1 30); do tasklist | grep -qi llama-server || break; sleep 1; done
  ( cd ../vendor/llama && ./llama-server.exe -m "../../$BASE_GGUF" "$@" -ngl 99 --n-cpu-moe 8 \
      -c 32768 -np 1 -ub 512 --port 8080 --host 127.0.0.1 --no-reasoning-preserve > ../llama-server.log 2>&1 ) &
  for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q '"status":"ok"' && return 0; sleep 3; done; return 1
}

# the partner dialogues are written by the generator run (4 workers); wait for its combined file
until [ -s out/dial_partner.jsonl ]; do sleep 30; done
taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 3   # the generator's server

if [ ! -s out/calls_run8.jsonl ]; then
  say "building data: $(wc -l < out/dial_partner.jsonl) partner dialogues, $(ls ../data/otospeech/asr | wc -l) otoSpeech conversations"
  uv run --project . python build_calls.py --dialogues out/dial_partner.jsonl --exclude-ids data/exclude_dialogues.txt --per 2 --timing mixed \
    --filler-min-talk 2.8 --filler-gap 3.6 --filler-words "Mm-hm.|Yeah.|Right.|Uh-huh.|Oh yeah?|Oh wow.|Totally.|Yeah, yeah." \
    --max-ticks 460 --seed 88 --out out/calls_partner_r8.jsonl > out/build_r8.log 2>&1 || { say "build failed: $(tail -2 out/build_r8.log | tr '\n' ' ')"; exit 1; }
  say "$(grep '^wrote' out/build_r8.log)"
  node ../scripts/oto_to_calls.js --out tune/out/calls_oto_all.jsonl --dev-out tune/out/calls_oto_all_dev.jsonl > out/oto_all_calls.log 2>&1
  say "$(head -1 out/oto_all_calls.log)"
  # more back-channels: the back-channel ticks of the partner and real-conversation calls weigh 2.5 (an unused weight
  # value, so --weight-map can still single them out)
  uv run --project . python -c "
import json
for f in ('out/calls_partner_r8.jsonl', 'out/calls_oto_all.jsonl'):
    rows = [json.loads(l) for l in open(f, encoding='utf-8')]
    for c in rows:
        for t in c['ticks']:
            if t['tag'] == 'continuer':
                t['w'] = 2.5
    open(f, 'w', encoding='utf-8').write(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in rows))
print('back-channel ticks reweighted to 2.5')"
  [ -s out/calls_train_masked.jsonl ] || uv run --project . python mask_cutins.py --calls ../models/lora/dataset/calls_train.jsonl --out out/calls_train_masked.jsonl > /dev/null 2>&1
  # no decline examples: the resume calls rebuilt without their decline turns, and the few calls with a refusal-style
  # hedge left out; danger warnings stay
  uv run --project . python mix_calls.py --calls out/calls_partner_r8.jsonl out/calls_oto_all.jsonl:400 out/calls_special_r8_nodecline.jsonl \
    out/calls_built_r5_nodecline.jsonl:200 out/calls_train_masked_nodecline.jsonl:80 --exclude-ids data/exclude_dialogues.txt --seed 88 --out out/calls_run8.jsonl > out/mix_r8.log 2>&1
  say "$(grep '^wrote' out/mix_r8.log)"
else
  say "resuming with the existing out/calls_run8.jsonl"
fi

if [ ! -s out/$NAME.gguf ]; then
  [ -d out/cache_r8 ] && uv run --project . --extra train python -c "
import torch, pathlib
fs = sorted(pathlib.Path('out/cache_r8').glob('call*.pt*'), key=lambda f: f.stat().st_mtime)[-40:]
bad = []
for f in fs:
    try:
        assert f.suffix == '.pt'; torch.load(f)
    except Exception:
        bad.append(f.name); f.unlink()
print('cache check: removed', bad if bad else 'nothing')
" 2>&1 | tail -1 | tee -a "$LOG"
  say "training run 8 partner (cache, then 600 steps; resumes from saved steps)"
  uv run --project . --extra train python train_lora.py --hf "$HF" --calls out/calls_run8.jsonl --cache out/cache_r8 --max-tokens 8000 \
    --chunk 160 --steps 600 --batch 4 --rank 32 --alpha 64 --lr 2e-4 --out out/adapter_run8 --resume >> out/run8.log 2>&1; echo "exit=$?" >> out/run8.log
  tail -1 out/run8.log | grep -q "^exit=0" || { say "training failed: $(tail -3 out/run8.log | tr '\n' ' ')"; exit 1; }
  uv run --project . --extra train python export_gguf.py --adapter out/adapter_run8 --hf "$HF" --out out/$NAME.gguf > out/run8.export.log 2>&1 \
    || { say "export failed"; exit 1; }
  cp out/$NAME.gguf ../models/lora/talkme-run8-partner-hauhau.gguf
  say "exported out/$NAME.gguf (in the app as talkme-run8-partner-hauhau.gguf; needs the HauhauCS base)"
fi

# replays at 0.6 with the repeat guard (the app's settings for human-timed adapters)
need=0; for f in eval_yap_run8 eval_ours_run8 probe_run8; do [ -s out/$f.json ] || need=1; done
if [ $need = 1 ]; then
  server --lora ../../tune/out/$NAME.gguf --lora-init-without-apply || { say "server did not start"; exit 1; }
  if [ ! -s out/probe_run8.json ]; then
    uv run --project . python probe_answers.py --adapter out/$NAME.gguf --listen-p 0.6 --repeat-guard --name run8 --out out/probe_run8.json > out/probe_run8.log 2>&1
    say "quiz: $(tail -1 out/probe_run8.log)"
  fi
  if [ ! -s out/eval_yap_run8.json ]; then
    say "recorded conversations (24 calls)"
    uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls data/heldout_yap.jsonl --n 24 --parallel 1 --adapter out/$NAME.gguf \
      --listen-p 0.6 --repeat-guard --out out/eval_yap_run8.json > out/eval_yap_run8.log 2>&1
    uv run --project . python yap_metrics.py data/heldout_yap.jsonl out/eval_yap_run7_guard2.json out/eval_yap_blend76.json out/eval_yap_run8.json --first 24 \
      --names "run7 +guard,run7+run6 blend,run8 partner" --md out/yap_run8.md > /dev/null 2>&1
    say "recorded conversations: out/yap_run8.md"
  fi
  if [ ! -s out/eval_ours_run8.json ]; then
    say "our held-out set"
    uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls data/heldout_ours.jsonl --n 0 --parallel 1 --adapter out/$NAME.gguf \
      --listen-p 0.6 --repeat-guard --out out/eval_ours_run8.json > out/eval_ours_run8.log 2>&1
    say "ours: $(grep -A6 '== totals' out/eval_ours_run8.log | tail -6 | tr '\n' ' ')"
  fi
  taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 3
fi

cd ..
MODEL="$BASE_GGUF" bash scripts/live_bench.sh run8 tune/out/$NAME.gguf 0.6 2>&1 | tee -a "tune/$LOG"
# the run 7 + run 6 blend on the three partner scenarios it has not had yet, so run 8 has something to beat
for c in "amy piper:vits-piper-en_US-amy-medium 1.0" "george kokoro:26 1.1" "emma kokoro:21 0.9"; do
  set -- $c
  [ -s "tests/out/report-partner3-blend76-$1.json" ] && continue
  node scripts/audio_test.js --adapter "blend:talkme-run7-partner.gguf=0.5,talkme-run6.gguf=0.5" --threshold 0.6 --user-voice "$2" --user-speed "$3" \
    --scenario quiet-nudge,long-talk,story-hold --tag "partner3-blend76-$1" > "tests/out/partner3-blend76-$1.log" 2>&1
  echo "[run8] blend on the partner scenarios, caller $1: $(grep '^checks ' tests/out/partner3-blend76-$1.log | cut -c1-20)" | tee -a "tune/$LOG"
done
node scripts/live_bench_report.js blend76 run8 > /dev/null
taskkill //F //IM llama-server.exe > /dev/null 2>&1
echo "[run8] $(date +%H:%M:%S) done; GPU free (live report: tests/out/live_bench.md)" | tee -a "tune/$LOG"
