#!/usr/bin/env bash
# Held-out replays of several adapters on one shared llama-server, then the comparison table.
#   bash replay_pair.sh <n-cpu-moe> <tag> <adapter.gguf> [<adapter.gguf> ...]
# Eval files are out/eval_<adapter file name without .gguf>.json; the table goes to out/compare_<tag>.md.
set -u
cd "$(dirname "$0")"
NCPU=$1; TAG=$2; shift 2
LOG=out/replay_$TAG.log
say() { echo "[replay] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
list=$(printf '%s,' "$@" | sed 's/,$//' | sed 's#\.\./#../../#g; s#out/#../../tune/out/#g')
say "server: n-cpu-moe $NCPU, adapters $list"
( cd ../vendor/llama && ./llama-server.exe -m ../../models/base/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --lora "$list" --lora-init-without-apply \
    -ngl 99 --n-cpu-moe "$NCPU" -c 114688 -np 4 -ub 512 --port 8080 --host 127.0.0.1 --no-reasoning-preserve > ../llama-server.log 2>&1 ) &
for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 3; done
curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok || { say "server did not start"; exit 1; }
say "loaded: $(curl -s http://127.0.0.1:8080/lora-adapters)"
evals=()
for ad in "$@"; do
  name=$(basename "$ad" .gguf); [ "$name" = microturn-duplex-lora ] && name=published
  say "replaying $name"
  uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls ../models/lora/dataset/calls_heldout.jsonl --n 0 \
    --parallel 4 --adapter "$ad" --out "out/eval_$name.json" > "out/eval_$name.log" 2>&1 || say "replay $name failed: $(tail -2 out/eval_$name.log | tr '\n' ' ')"
  say "$name: $(grep -A8 '== totals ==' out/eval_$name.log | tail -8 | tr '\n' ' ')"
  evals+=("out/eval_$name.json")
done
uv run --project . python compare.py "${evals[@]}" --md "out/compare_$TAG.md" > /dev/null 2>&1 && say "table: out/compare_$TAG.md"
taskkill //F //IM llama-server.exe > /dev/null 2>&1
say "done"
