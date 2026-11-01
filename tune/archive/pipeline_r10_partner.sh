#!/usr/bin/env bash
# Run 10 "partner": run 9, with what the user's own test and the new live checks showed (docs/RESULTS.md, run 9):
#   - says "no idea" like a friend: run 9 made up a neighbor's address ("142 Maple Street") and fell back to "I cannot
#     provide..." / "no memory of past conversations"; new dialogues where the user asks what the friend can't know
#     (gen_dialogues --kind partner2 --focus unknown)
#   - goodbyes: 40% of the new dialogues end with the user wrapping up and a quick "talk soon!" (run 9 stayed silent)
#   - no invented plans: run 8's dialogues are filtered for shared plans and a shared life (87 of 490 dropped)
#   - answers, then stops: the listens right after a finished reply weigh a little more (listen_done 1.9, was 1.5)
#   - back-channels a little up: weight 2.05 (run 9 1.8, run 8 2.5), still every ~5 s of user speech
#   - more warnings and cut-ins: 130 more of each (live cut-in stops 33%, held-out warnings 50%)
#   - the size test: the same data trained twice, at rank 32 (run10) and rank 64 (run10r64, in the app as
#     talkme-run10-partner-r64-hauhau.gguf)
# Base: HauhauCS (abliterated), no decline examples, partner system prompt, as run 9.
# The same command starts and resumes it (after tune/gpu_pause.sh): finished stages are skipped.
#   bash pipeline_r10_partner.sh
set -u
cd "$(dirname "$0")"
HF=../models/hf/Qwen3.6-35B-A3B-hauhau
BASE_GGUF=models/base/hauhau/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
WRITER_GGUF=models/base/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf   # the dialogues are written by the original base model
LOG=out/run10.pipeline.log
say() { echo "[run10] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
server() { # gguf, then llama-server options
  local model=$1; shift
  taskkill //F //IM llama-server.exe > /dev/null 2>&1
  for i in $(seq 1 30); do tasklist | grep -qi llama-server || break; sleep 1; done
  ( cd ../vendor/llama && ./llama-server.exe -m "../../$model" "$@" -ngl 99 --n-cpu-moe 8 \
      -c 32768 -ub 512 --port 8080 --host 127.0.0.1 --no-reasoning-preserve > ../llama-server.log 2>&1 ) &
  for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q '"status":"ok"' && return 0; sleep 3; done; return 1
}
gen() { # worker, focus, count, seed, first id
  uv run --project . python gen_dialogues.py --url http://127.0.0.1:8080 --kind partner2 --focus "$2" --goodbye 0.4 --count "$3" \
    --seed "$4" --start-id "$5" --temperature 0.9 --out out/dial_partner3_$1.part.jsonl >> out/gen_partner3_$1.log 2>&1
}

# 1. new partner dialogues: 130 warnings, 130 cut-in stories, 120 "no idea"s, 40% with a goodbye (4 writers)
if [ ! -s out/dial_partner3.jsonl ]; then
  say "writing partner dialogues: warnings, cut-in stories, things the friend can't know, goodbyes"
  server "$WRITER_GGUF" -np 4 || { say "server did not start"; exit 1; }
  rm -f out/dial_partner3_?.part.jsonl out/gen_partner3_?.log
  pids=""
  gen 0 hazard 95 1010 100000 & pids="$pids $!"
  gen 1 resume 95 1011 101000 & pids="$pids $!"
  gen 2 unknown 95 1012 102000 & pids="$pids $!"
  ( gen 3 hazard 35 1013 103000; gen 3 resume 35 1014 103500; gen 3 unknown 25 1015 103800 ) & pids="$pids $!"
  wait $pids   # not a bare wait: the server runs in the background too
  cat out/dial_partner3_?.part.jsonl > out/dial_partner3.jsonl
  taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 3
  say "$(wc -l < out/dial_partner3.jsonl) new partner dialogues"
fi

# 2. data
if [ ! -s out/calls_run10.jsonl ]; then
  # run 8's dialogues without shared plans or an invented shared life (the same checks the new ones pass)
  uv run --project . python -c "
import json, re, gen_dialogues as g
rows = [json.loads(l) for l in open('out/dial_partner.jsonl', encoding='utf-8')]
LIFE = r'\b(i saw (you|your)|when we (went|were|met)|remember when we|last time we|i was (there|at your))\b'
keep = [d for d in rows if not any(re.search(rx, ' '.join(t.get('user', '') + ' ' + t.get('assistant', '') for t in d['turns']), re.I) for rx in (g.SHARED_PLANS, LIFE))]
open('out/dial_partner_clean.jsonl', 'w', encoding='utf-8').write(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in keep))
print('run 8 dialogues kept:', len(keep), 'of', len(rows))
" | tee -a "$LOG"
  cat out/dial_partner2.jsonl out/dial_partner3.jsonl > out/dial_partner23.jsonl
  FILLERS="Mm-hm.|Yeah.|Right.|Uh-huh.|Oh yeah?|Oh wow.|Totally.|Oh nice.|Huh.|Ha, yeah.|Oh no.|Really?"
  BC="--filler-min-talk 3.0 --filler-gap 5.0 --nudge-quiet 2.6,5.5"
  uv run --project . python build_calls.py --dialogues out/dial_partner23.jsonl --exclude-ids data/exclude_dialogues.txt --per 2 --timing mixed \
    $BC --filler-words "$FILLERS" --max-ticks 460 --seed 109 --out out/calls_partner23_r10.jsonl > out/build_r10.log 2>&1 || { say "build failed"; exit 1; }
  uv run --project . python build_calls.py --dialogues out/dial_partner_clean.jsonl --exclude-ids data/exclude_dialogues.txt --per 1 --timing mixed \
    $BC --filler-words "$FILLERS" --max-ticks 460 --seed 108 --out out/calls_partner1_r10.jsonl >> out/build_r10.log 2>&1 || { say "build failed"; exit 1; }
  say "$(grep '^wrote' out/build_r10.log | tr '\n' ' ')"
  uv run --project . python mix_calls.py --calls out/calls_partner23_r10.jsonl out/calls_partner1_r10.jsonl out/calls_oto_all.jsonl:250 \
    out/calls_built_r5_hazard.jsonl:300 out/calls_masked_hazard.jsonl:100 out/calls_special_r8_nodecline.jsonl \
    --exclude-ids data/exclude_dialogues.txt --seed 109 --out out/calls_run10.jsonl > out/mix_r10.log 2>&1 || { say "mix failed"; exit 1; }
  # loss weights (the values double as tag ids for --weight-map, so each changed one is a value no other tag uses)
  uv run --project . python -c "
import json
from collections import Counter
rows = [json.loads(l) for l in open('out/calls_run10.jsonl', encoding='utf-8')]
n = Counter()
for c in rows:
    real = bool(c.get('timing_only'))
    for t in c['ticks']:
        tag = t['tag']
        if real and tag in ('speak', 'self_resume', 'overlap_start'):
            t['w'] = 0.0; n['real mid-reply words off'] += 1      # when to talk, not what a stranger said next
        elif tag == 'continuer':
            t['w'] = 2.05; n['back-channels 2.05'] += 1
        elif tag == 'hazard_start':
            t['w'] = 5.0; n['warning starts 5.0'] += 1
        elif tag == 'hazard':
            t['w'] = 2.2; n['warning words 2.2'] += 1
        elif tag == 'barge_stop':
            t['w'] = 4.5; n['cut-in stops 4.5'] += 1
        elif tag == 'listen_done' and not real:
            t['w'] = 1.9; n['listens after a finished reply 1.9'] += 1
open('out/calls_run10.jsonl', 'w', encoding='utf-8').write(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in rows))
print('reweighted:', dict(n))
" | tee -a "$LOG"
  say "$(grep '^wrote' out/mix_r10.log)"
else
  say "resuming with the existing out/calls_run10.jsonl"
fi

# 3. train (rank and alpha given), export, copy into the app
train() { # name, rank, alpha
  local name=$1 rank=$2 alpha=$3
  [ -s out/$name.gguf ] && return 0
  [ -d out/cache_r10 ] && uv run --project . --extra train python -c "
import torch, pathlib
fs = sorted(pathlib.Path('out/cache_r10').glob('call*.pt*'), key=lambda f: f.stat().st_mtime)[-40:]
bad = []
for f in fs:
    try:
        assert f.suffix == '.pt'; torch.load(f)
    except Exception:
        bad.append(f.name); f.unlink()
print('cache check: removed', bad if bad else 'nothing')
" 2>&1 | tail -1 | tee -a "$LOG"
  say "training $name (rank $rank; cache, then 600 steps; resumes from saved steps)"
  uv run --project . --extra train python train_lora.py --hf "$HF" --calls out/calls_run10.jsonl --cache out/cache_r10 --max-tokens 8000 \
    --chunk 160 --steps 600 --batch 4 --rank $rank --alpha $alpha --lr 2e-4 --system partner --out out/adapter_$name --resume >> out/$name.log 2>&1; echo "exit=$?" >> out/$name.log
  tail -1 out/$name.log | grep -q "^exit=0" || { say "training $name failed: $(tail -3 out/$name.log | tr '\n' ' ')"; exit 1; }
  uv run --project . --extra train python export_gguf.py --adapter out/adapter_$name --hf "$HF" --out out/$name.gguf > out/$name.export.log 2>&1 \
    || { say "export $name failed"; exit 1; }
  local app=talkme-run10-partner-hauhau.gguf   # the app reads "runN-partner" (prompt) and "-hauhau.gguf" (base)
  [ $name = run10r64 ] && app=talkme-run10-partner-r64-hauhau.gguf
  cp out/$name.gguf ../models/lora/$app
  say "exported out/$name.gguf (in the app as $app)"
}

# 4. replays at 0.6 with the repeat guard and the partner system prompt
replays() { # name
  local name=$1
  local need=0; for f in eval_yap_$name eval_ours_$name probe_$name; do [ -s out/$f.json ] || need=1; done
  [ $need = 0 ] && return 0
  server "$BASE_GGUF" -np 1 --lora ../../tune/out/$name.gguf --lora-init-without-apply || { say "server did not start"; exit 1; }
  if [ ! -s out/probe_$name.json ]; then
    uv run --project . python probe_answers.py --adapter out/$name.gguf --listen-p 0.6 --repeat-guard --system partner --name $name --out out/probe_$name.json > out/probe_$name.log 2>&1
    say "quiz: $(tail -1 out/probe_$name.log)"
  fi
  if [ ! -s out/eval_yap_$name.json ]; then
    uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls data/heldout_yap.jsonl --n 24 --parallel 1 --adapter out/$name.gguf \
      --listen-p 0.6 --repeat-guard --system partner --out out/eval_yap_$name.json > out/eval_yap_$name.log 2>&1
    say "recorded conversations done"
  fi
  if [ ! -s out/eval_ours_$name.json ]; then
    uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls data/heldout_ours.jsonl --n 0 --parallel 1 --adapter out/$name.gguf \
      --listen-p 0.6 --repeat-guard --system partner --out out/eval_ours_$name.json > out/eval_ours_$name.log 2>&1
    say "ours ($name): $(grep -A6 '== totals' out/eval_ours_$name.log | tail -6 | tr '\n' ' ')"
  fi
  taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 3
}
yap_table() {
  local names=$1 files=$2
  uv run --project . python yap_metrics.py data/heldout_yap.jsonl $files --first 24 --names "$names" --md out/yap_run10.md > /dev/null 2>&1
}
live() { # name, adapter
  ( cd .. && MODEL="$BASE_GGUF" PROMPT=partner bash scripts/live_bench.sh $1 $2 0.6 2>&1 | tee -a "tune/$LOG" )
}

# run 10 first, all the way to its live tests, so it can be tried while the size test trains
train run10 32 64
replays run10
yap_table "run8 partner,run9 partner,run10 partner" "out/eval_yap_run8.json out/eval_yap_run9.json out/eval_yap_run10.json"
live run10 tune/out/run10.gguf
# run 9 again with the new checks (the judge, friend-call), so the live table compares like with like
[ -s ../tests/out/report-bench-run9-amy.json ] && grep -q "friend-call" ../tests/out/report-bench-run9-amy.json || ( cd .. && FORCE=1 MODEL="$BASE_GGUF" PROMPT=partner bash scripts/live_bench.sh run9 tune/out/run9.gguf 0.6 2>&1 | tee -a "tune/$LOG" )
( cd .. && node scripts/live_bench_report.js run9 run10 > /dev/null )
say "run 10 ready to try (talkme-run10-partner-hauhau.gguf); live report: tests/out/live_bench.md. Now the size test (rank 64)"

# the size test: the same data at rank 64 (alpha 128 keeps the same scale)
train run10r64 64 128
replays run10r64
yap_table "run9 partner,run10 partner,run10 rank 64" "out/eval_yap_run9.json out/eval_yap_run10.json out/eval_yap_run10r64.json"
live run10r64 tune/out/run10r64.gguf
( cd .. && node scripts/live_bench_report.js run9 run10 run10r64 > /dev/null )
taskkill //F //IM llama-server.exe > /dev/null 2>&1
say "done; GPU free (live report: tests/out/live_bench.md, recorded calls: out/yap_run10.md)"
