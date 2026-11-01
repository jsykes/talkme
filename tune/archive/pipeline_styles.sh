#!/usr/bin/env bash
# Personality adapters on top of run 2: warm, brisk, playful.
# 1. generate styled dialogues with the base model (server on 8080 must be running, base model, no adapter)
# 2. per style: build calls (style back-channels), add 150 run 2 pieces as timing anchors (their cache is reused),
#    cache the new calls, continue from run 2 for 120 steps, export
# 3. replay each style and run 2 on 8 held-out calls (4 short, 4 long) for a timing check
set -u
cd "$(dirname "$0")"
HF=../models/hf/Qwen3.6-35B-A3B
LOG=out/styles.log
say() { echo "[styles] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
STYLES="warm brisk playful"
declare -A BASEID=([warm]=40000 [brisk]=50000 [playful]=60000)
declare -A FILL=([warm]="Mm-hm.|Oh, I see.|Aw.|Right.|Oh, wow." [brisk]="Mm-hm.|Okay." [playful]="Ha.|Oh nice.|Mm-hm.|Ooh.|No way.")
declare -A KEEP=([warm]=1.0 [brisk]=0.3 [playful]=0.8)

for st in $STYLES; do
  b=${BASEID[$st]}
  say "generating $st dialogues"
  uv run --project . python gen_dialogues.py --count 45 --kind short --style "$st" --start-id "$b" --seed 11 --out "out/dial_${st}_short.jsonl" >> "out/gen_$st.log" 2>&1
  uv run --project . python gen_dialogues.py --count 15 --kind long --style "$st" --start-id "$((b + 5000))" --seed 12 --out "out/dial_${st}_long.jsonl" >> "out/gen_$st.log" 2>&1
  say "$st: $(cat out/dial_${st}_short.jsonl out/dial_${st}_long.jsonl 2>/dev/null | wc -l) dialogues"
done
taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 3

for st in $STYLES; do
  say "building $st calls"
  uv run --project . python build_calls.py --dialogues "out/dial_${st}_short.jsonl" "out/dial_${st}_long.jsonl" \
    --exclude-calls ../models/lora/dataset/calls_heldout.jsonl --per 3 --hazard-boost 1 --timing mixed --max-ticks 460 --seed 21 \
    --filler-words "${FILL[$st]}" --filler-keep "${KEEP[$st]}" --out "out/calls_${st}_only.jsonl" >> "out/build_$st.log" 2>&1
  # style calls first, then 150 run 2 pieces whose cached activations are copied instead of recomputed
  uv run --project . python - "$st" <<'PYEOF' >> "out/build_$st.log" 2>&1
import json, shutil, sys
from pathlib import Path
st = sys.argv[1]
style = [l for l in open(f"out/calls_{st}_only.jsonl", encoding="utf-8") if l.strip()]
anchors = [l for l in open("out/calls_run2.jsonl", encoding="utf-8") if l.strip()][:150]
with open(f"out/calls_{st}.jsonl", "w", encoding="utf-8") as f:
    f.writelines(style + anchors)
cache = Path(f"out/cache_{st}")
cache.mkdir(exist_ok=True)
for j in range(len(anchors)):
    src, dst = Path(f"out/cache_r2/call{j:05d}.pt"), cache / f"call{len(style) + j:05d}.pt"
    if not dst.exists():
        shutil.copyfile(src, dst)
print(f"{st}: {len(style)} style calls + {len(anchors)} anchors")
PYEOF
  say "$st: $(tail -1 out/build_$st.log)"
  say "training $st (continue from run 2, 120 steps)"
  uv run --project . --extra train python train_lora.py --hf "$HF" --calls "out/calls_${st}.jsonl" --cache "out/cache_${st}" --chunk 160 \
    --init-adapter out/adapter_run2 --steps 120 --batch 4 --rank 32 --alpha 64 --lr 1e-4 --warmup 5 --seed 5 \
    --out "out/adapter_${st}" > "out/train_${st}.log" 2>&1; echo "exit=$?" >> "out/train_${st}.log"
  grep -q "^exit=0" "out/train_${st}.log" || { say "training $st failed: $(tail -3 out/train_${st}.log | tr '\n' ' ')"; continue; }
  uv run --project . --extra train python export_gguf.py --adapter "out/adapter_${st}" --hf "$HF" --out "out/${st}.gguf" > "out/export_${st}.log" 2>&1 \
    && say "$st exported" || say "export $st failed"
done

say "replay server with run 2 and the styles"
( cd ../vendor/llama && ./llama-server.exe -m ../../models/base/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
    --lora ../../tune/out/run2.gguf,../../tune/out/warm.gguf,../../tune/out/brisk.gguf,../../tune/out/playful.gguf \
    --lora-init-without-apply -ngl 99 --n-cpu-moe 8 -c 32768 -np 1 -ub 512 --port 8080 --host 127.0.0.1 --no-reasoning-preserve \
    > ../llama-server.log 2>&1 ) &
for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 3; done
for st in $STYLES; do
  [ -f "out/${st}.gguf" ] || continue
  say "replaying 8 held-out calls with $st"
  uv run --project . python evaluate.py --url http://127.0.0.1:8080 --calls ../models/lora/dataset/calls_heldout.jsonl --stratified 4 \
    --parallel 1 --adapter "out/${st}.gguf" --out "out/eval8_${st}.json" > "out/eval8_${st}.log" 2>&1
done
say "styles finished; server left running on 8080"
