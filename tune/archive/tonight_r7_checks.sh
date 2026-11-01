#!/usr/bin/env bash
# Run 7 checks before trying it by hand, one step after another (no waiting on other jobs):
#   1. repetition-loop fixes on 24 recorded calls at threshold 0.6: repeat guard alone, + repetition penalty, + DRY
#   2. answer quiz: does it still answer? runs 2, 5, 6, 7, run 7 at half strength, run 7 + run 5, run 7 + run 6
# Then the GPU is free. Left for later (bash tune/tonight_r7_later.sh): run 7 on our held-out set, run 6 live tests.
#   bash tune/tonight_r7_checks.sh
set -u
cd "$(dirname "$0")"
LOG=out/run7.pipeline.log
say() { echo "[checks] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
server() { ( cd ../vendor/llama && ./llama-server.exe -m "../../models/base/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" "$@" -ngl 99 --n-cpu-moe 8 \
    -c 32768 -np 1 -ub 512 --port 8080 --host 127.0.0.1 --no-reasoning-preserve > ../llama-server.log 2>&1 ) &
  for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && return 0; sleep 3; done; return 1; }
taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 3

server --lora ../../tune/out/run7.gguf,../../tune/out/run6.gguf,../../tune/out/run5.gguf,../../tune/out/run2.gguf --lora-init-without-apply \
  || { say "server did not start"; exit 1; }

say "loop fixes on 24 recorded calls (run 7 at 0.6)"
for s in 'guard|' \
         'penalty|{"temperature":0.7,"top_p":0.9,"repeat_penalty":1.3,"repeat_last_n":64}' \
         'dry|{"temperature":0.7,"top_p":0.9,"dry_multiplier":0.8,"dry_base":1.75,"dry_allowed_length":2}'; do
  n=${s%%|*}; o=${s#*|}
  [ -s out/eval_yap_run7_$n.json ] && { say "$n already done"; continue; }
  uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls data/heldout_yap.jsonl --n 24 --parallel 1 \
    --adapter out/run7.gguf --listen-p 0.6 --repeat-guard ${o:+--speak-opts "$o"} --out out/eval_yap_run7_$n.json > out/eval_yap_run7_$n.log 2>&1 \
    || say "$n failed: $(tail -2 out/eval_yap_run7_$n.log | tr '\n' ' ')"
  say "done $n"
done
uv run --project . python yap_metrics.py data/heldout_yap.jsonl out/eval_yap_run6_p06.json out/eval_yap_run7_p06.json out/eval_yap_run7_guard.json \
  out/eval_yap_run7_penalty.json out/eval_yap_run7_dry.json --first 24 \
  --names "run6@0.6,run7@0.6,run7 +guard,run7 +guard+penalty,run7 +guard+DRY" --md out/yap_run7_decode.md > /dev/null 2>&1
say "loop table: out/yap_run7_decode.md"

say "answer quiz"
for s in 'run2|run2.gguf|0.3' 'run5|run5.gguf|0.3' 'run6|run6.gguf|0.6' 'run7|run7.gguf|0.6' \
         'run7 half strength|run7.gguf:0.5|0.6' 'run7 + run5|run7.gguf:0.5,run5.gguf:0.5|0.6' 'run7 + run6|run7.gguf:0.5,run6.gguf:0.5|0.6'; do
  IFS='|' read -r n ad lp <<< "$s"
  f="out/probe_$(echo "$n" | tr ' +' '__').json"
  uv run --project . python probe_answers.py --adapter "$ad" --listen-p "$lp" --repeat-guard --name "$n" --out "$f" > "${f%.json}.log" 2>&1
  say "$(tail -1 "${f%.json}.log")"
done

taskkill //F //IM llama-server.exe > /dev/null 2>&1
say "done; GPU free"
