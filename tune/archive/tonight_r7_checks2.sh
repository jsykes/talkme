#!/usr/bin/env bash
# Answer quiz for runs 2, 5, 6, 7 and the run 7 blends, then run 7 + the two-tick repeat guard on 24 recorded calls.
# (The repetition-penalty / DRY comparison was dropped: regenerating every spoken reply took ~1.1 s a tick, past the
# 600 ms budget, so the app could not use it anyway.)
#   bash tune/tonight_r7_checks2.sh
set -u
cd "$(dirname "$0")"
LOG=out/run7.pipeline.log
say() { echo "[checks] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 3
( cd ../vendor/llama && ./llama-server.exe -m "../../models/base/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" \
    --lora ../../tune/out/run7.gguf,../../tune/out/run6.gguf,../../tune/out/run5.gguf,../../tune/out/run2.gguf --lora-init-without-apply \
    -ngl 99 --n-cpu-moe 8 -c 32768 -np 1 -ub 512 --port 8080 --host 127.0.0.1 --no-reasoning-preserve > ../llama-server.log 2>&1 ) &
for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 3; done

say "answer quiz"
for s in 'run2|run2.gguf|0.3' 'run5|run5.gguf|0.3' 'run6|run6.gguf|0.6' 'run7|run7.gguf|0.6' \
         'run7 half strength|run7.gguf:0.5|0.6' 'run7 + run5|run7.gguf:0.5,run5.gguf:0.5|0.6' 'run7 + run6|run7.gguf:0.5,run6.gguf:0.5|0.6'; do
  IFS='|' read -r n ad lp <<< "$s"
  f="out/probe_$(echo "$n" | tr ' +' '__').json"
  uv run --project . python probe_answers.py --adapter "$ad" --listen-p "$lp" --repeat-guard --name "$n" --out "$f" > "${f%.json}.log" 2>&1
  say "$(tail -1 "${f%.json}.log")"
done

say "run 7 + two-tick repeat guard on 24 recorded calls"
uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls data/heldout_yap.jsonl --n 24 --parallel 1 --adapter out/run7.gguf \
  --listen-p 0.6 --repeat-guard --out out/eval_yap_run7_guard2.json > out/eval_yap_run7_guard2.log 2>&1
taskkill //F //IM llama-server.exe > /dev/null 2>&1
uv run --project . python yap_metrics.py data/heldout_yap.jsonl out/eval_yap_run6_p06.json out/eval_yap_run7_p06.json out/eval_yap_run7_guard.json \
  out/eval_yap_run7_guard2.json --first 24 --names "run6@0.6,run7@0.6,run7 +guard(any distance),run7 +guard(2 ticks)" --md out/yap_run7_guard.md > /dev/null 2>&1
say "done; GPU free (out/yap_run7_guard.md)"
