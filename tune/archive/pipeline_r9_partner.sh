#!/usr/bin/env bash
# Run 9 "partner": run 8's partner, fixed where it fell short (docs/RESULTS.md, runs 7 and 8):
#   - danger warnings at full strength: half the new partner dialogues have a warning moment (gen_dialogues --kind
#     partner2 --focus hazard), plus 400 warning-bearing calls; warning ticks weigh more (hazard_start 5.0, hazard 2.2)
#   - stopping when cut in on: the other half have an interrupted story the partner stops, answers and resumes;
#     long-story reactions are only short sounds; cut-in stops weigh more (barge_stop 4.5)
#   - fewer back-channels and "Yeah." openers: weight 1.8 (run 8: 2.5), every ~5 s of user speech (run 8: 3.6 s),
#     250 real-conversation calls (run 8: 400)
#   - no musing like the caller: the real-conversation calls teach when to start, back-channel and wait, but no
#     mid-reply words (speak / self_resume / overlap_start ticks weigh 0)
#   - no invented shared life (validated in the generator); nudges after 2.6-5.5 s of silence, from real people
#     (scripts/silence_stats.js: median 3.1 s, p90 5.5 s)
#   - its own system prompt ("You are a friend on a live phone call...", common.PARTNER_SYSTEM), sent by the app
# Base: HauhauCS (abliterated), no decline examples, as run 8. Adapter: talkme-run9-partner-hauhau.gguf.
# The same command starts and resumes it (after tune/gpu_pause.sh): finished stages are skipped.
#   bash pipeline_r9_partner.sh
set -u
cd "$(dirname "$0")"
HF=../models/hf/Qwen3.6-35B-A3B-hauhau
BASE_GGUF=models/base/hauhau/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
WRITER_GGUF=models/base/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf   # the dialogues are written by the original base model, as for run 8
LOG=out/run9.pipeline.log
NAME=run9
say() { echo "[run9] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
server() { # gguf, then llama-server options
  local model=$1; shift
  taskkill //F //IM llama-server.exe > /dev/null 2>&1
  for i in $(seq 1 30); do tasklist | grep -qi llama-server || break; sleep 1; done
  ( cd ../vendor/llama && ./llama-server.exe -m "../../$model" "$@" -ngl 99 --n-cpu-moe 8 \
      -c 32768 -ub 512 --port 8080 --host 127.0.0.1 --no-reasoning-preserve > ../llama-server.log 2>&1 ) &
  for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q '"status":"ok"' && return 0; sleep 3; done; return 1
}

# 1. new partner dialogues (4 writers in parallel)
if [ ! -s out/dial_partner2.jsonl ]; then
  say "writing partner dialogues with a warning moment or an interrupted story"
  server "$WRITER_GGUF" -np 4 || { say "server did not start"; exit 1; }
  for w in 0 1 2 3; do
    uv run --project . python gen_dialogues.py --url http://127.0.0.1:8080 --kind partner2 --focus $([ $w -lt 2 ] && echo hazard || echo resume) --count 65 --seed $((910 + w)) \
      --start-id $((90000 + w * 1000)) --temperature 0.9 --out out/dial_partner2_$w.part.jsonl > out/gen_partner2_$w.log 2>&1 &
    pids="${pids:-} $!"
  done
  wait $pids   # not a bare wait: the server runs in the background too
  cat out/dial_partner2_?.part.jsonl > out/dial_partner2.jsonl
  taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 3
  say "$(wc -l < out/dial_partner2.jsonl) new partner dialogues"
fi

# 2. data
if [ ! -s out/calls_run9.jsonl ]; then
  FILLERS="Mm-hm.|Yeah.|Right.|Uh-huh.|Oh yeah?|Oh wow.|Totally.|Oh nice.|Huh.|Ha, yeah.|Oh no.|Really?"
  BC="--filler-min-talk 3.0 --filler-gap 5.0 --nudge-quiet 2.6,5.5"
  uv run --project . python build_calls.py --dialogues out/dial_partner2.jsonl --exclude-ids data/exclude_dialogues.txt --per 2 --timing mixed \
    $BC --filler-words "$FILLERS" --max-ticks 460 --seed 99 --out out/calls_partner2_r9.jsonl > out/build_r9.log 2>&1 || { say "build failed"; exit 1; }
  uv run --project . python build_calls.py --dialogues out/dial_partner.jsonl --exclude-ids data/exclude_dialogues.txt --per 1 --timing mixed \
    $BC --filler-words "$FILLERS" --max-ticks 460 --seed 98 --out out/calls_partner1_r9.jsonl >> out/build_r9.log 2>&1 || { say "build failed"; exit 1; }
  say "$(grep '^wrote' out/build_r9.log | tr '\n' ' ')"
  [ -s out/calls_oto_all.jsonl ] || node ../scripts/oto_to_calls.js --out tune/out/calls_oto_all.jsonl --dev-out tune/out/calls_oto_all_dev.jsonl > out/oto_all_calls.log 2>&1
  # warning-bearing calls only, from the answered-cut-in and the masked prebuilt calls (no declines, as run 8)
  uv run --project . python -c "
import json
for src, dst in (('out/calls_built_r5_nodecline.jsonl', 'out/calls_built_r5_hazard.jsonl'), ('out/calls_train_masked_nodecline.jsonl', 'out/calls_masked_hazard.jsonl')):
    rows = [json.loads(l) for l in open(src, encoding='utf-8')]
    keep = [c for c in rows if any(t['tag'] == 'hazard_start' for t in c['ticks'])]
    open(dst, 'w', encoding='utf-8').write(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in keep))
    print(dst, len(keep), 'of', len(rows))
" | tee -a "$LOG"
  uv run --project . python mix_calls.py --calls out/calls_partner2_r9.jsonl out/calls_partner1_r9.jsonl out/calls_oto_all.jsonl:250 \
    out/calls_built_r5_hazard.jsonl:300 out/calls_masked_hazard.jsonl:100 out/calls_special_r8_nodecline.jsonl \
    --exclude-ids data/exclude_dialogues.txt --seed 99 --out out/calls_run9.jsonl > out/mix_r9.log 2>&1
  # loss weights (the values double as tag ids for --weight-map, so each new one is a value no other tag uses)
  uv run --project . python -c "
import json
from collections import Counter
rows = [json.loads(l) for l in open('out/calls_run9.jsonl', encoding='utf-8')]
n = Counter()
for c in rows:
    real = bool(c.get('timing_only'))
    for t in c['ticks']:
        tag = t['tag']
        if real and tag in ('speak', 'self_resume', 'overlap_start'):
            t['w'] = 0.0; n['real mid-reply words off'] += 1      # when to talk, not what a stranger said next
        elif tag == 'continuer':
            t['w'] = 1.8; n['back-channels 1.8'] += 1
        elif tag == 'hazard_start':
            t['w'] = 5.0; n['warning starts 5.0'] += 1
        elif tag == 'hazard':
            t['w'] = 2.2; n['warning words 2.2'] += 1
        elif tag == 'barge_stop':
            t['w'] = 4.5; n['cut-in stops 4.5'] += 1
open('out/calls_run9.jsonl', 'w', encoding='utf-8').write(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in rows))
print('reweighted:', dict(n))
" | tee -a "$LOG"
  say "$(grep '^wrote' out/mix_r9.log)"
else
  say "resuming with the existing out/calls_run9.jsonl"
fi

# 3. train on the HauhauCS base with the partner system prompt
if [ ! -s out/$NAME.gguf ]; then
  [ -d out/cache_r9 ] && uv run --project . --extra train python -c "
import torch, pathlib
fs = sorted(pathlib.Path('out/cache_r9').glob('call*.pt*'), key=lambda f: f.stat().st_mtime)[-40:]
bad = []
for f in fs:
    try:
        assert f.suffix == '.pt'; torch.load(f)
    except Exception:
        bad.append(f.name); f.unlink()
print('cache check: removed', bad if bad else 'nothing')
" 2>&1 | tail -1 | tee -a "$LOG"
  say "training run 9 partner (cache, then 600 steps; resumes from saved steps)"
  uv run --project . --extra train python train_lora.py --hf "$HF" --calls out/calls_run9.jsonl --cache out/cache_r9 --max-tokens 8000 \
    --chunk 160 --steps 600 --batch 4 --rank 32 --alpha 64 --lr 2e-4 --system partner --out out/adapter_run9 --resume >> out/run9.log 2>&1; echo "exit=$?" >> out/run9.log
  tail -1 out/run9.log | grep -q "^exit=0" || { say "training failed: $(tail -3 out/run9.log | tr '\n' ' ')"; exit 1; }
  uv run --project . --extra train python export_gguf.py --adapter out/adapter_run9 --hf "$HF" --out out/$NAME.gguf > out/run9.export.log 2>&1 \
    || { say "export failed"; exit 1; }
  cp out/$NAME.gguf ../models/lora/talkme-run9-partner-hauhau.gguf
  say "exported out/$NAME.gguf (in the app as talkme-run9-partner-hauhau.gguf: HauhauCS base, partner prompt)"
fi

# 4. replays at 0.6 with the repeat guard and the partner system prompt
need=0; for f in eval_yap_run9 eval_ours_run9 probe_run9; do [ -s out/$f.json ] || need=1; done
if [ $need = 1 ]; then
  server "$BASE_GGUF" -np 1 --lora ../../tune/out/$NAME.gguf --lora-init-without-apply || { say "server did not start"; exit 1; }
  if [ ! -s out/probe_run9.json ]; then
    uv run --project . python probe_answers.py --adapter out/$NAME.gguf --listen-p 0.6 --repeat-guard --system partner --name run9 --out out/probe_run9.json > out/probe_run9.log 2>&1
    say "quiz: $(tail -1 out/probe_run9.log)"
  fi
  if [ ! -s out/eval_yap_run9.json ]; then
    say "recorded conversations (24 calls)"
    uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls data/heldout_yap.jsonl --n 24 --parallel 1 --adapter out/$NAME.gguf \
      --listen-p 0.6 --repeat-guard --system partner --out out/eval_yap_run9.json > out/eval_yap_run9.log 2>&1
    uv run --project . python yap_metrics.py data/heldout_yap.jsonl out/eval_yap_blend76.json out/eval_yap_run8.json out/eval_yap_run9.json --first 24 \
      --names "run7+run6 blend,run8 partner,run9 partner" --md out/yap_run9.md > /dev/null 2>&1
    say "recorded conversations: out/yap_run9.md"
  fi
  if [ ! -s out/eval_ours_run9.json ]; then
    say "our held-out set"
    uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls data/heldout_ours.jsonl --n 0 --parallel 1 --adapter out/$NAME.gguf \
      --listen-p 0.6 --repeat-guard --system partner --out out/eval_ours_run9.json > out/eval_ours_run9.log 2>&1
    say "ours: $(grep -A6 '== totals' out/eval_ours_run9.log | tail -6 | tr '\n' ' ')"
  fi
  taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 3
fi

# 5. live tests
cd ..
MODEL="$BASE_GGUF" PROMPT=partner bash scripts/live_bench.sh run9 tune/out/$NAME.gguf 0.6 2>&1 | tee -a "tune/$LOG"
node scripts/live_bench_report.js run8 run9 > /dev/null
taskkill //F //IM llama-server.exe > /dev/null 2>&1
echo "[run9] $(date +%H:%M:%S) done; GPU free (live report: tests/out/live_bench.md)" | tee -a "tune/$LOG"
