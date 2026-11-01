#!/usr/bin/env bash
# The voice bake-off on the GPU (Windows): runs the
# four new voices one at a time with the GPU to themselves, transcribes everything back, and writes out/report.md and
# out/listen.html. About 15 minutes.   bash tts_bench/run_all.sh
set -u
cd "$(dirname "$0")"
LOG=out/bakeoff.log
mkdir -p out
say() { echo "[voices] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
stop_orpheus() {
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | Where-Object { \$_.CommandLine -match '--port 8090' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" > /dev/null 2>&1
}
for m in chatterbox kyutai cosyvoice; do
  say "$m"
  ( cd $m && uv run python ../run_$m.py >> ../$LOG 2>&1 ) || say "$m failed (see $LOG)"
done
say "orpheus"
( cd ../vendor/llama && ./llama-server.exe -m ../../models/tts_bench/orpheus/orpheus-3b-0.1-ft-Q8_0.gguf -ngl 99 -c 4096 --port 8090 --host 127.0.0.1 > ../../tts_bench/out/orpheus_server.log 2>&1 ) &
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8090/health | grep -q '"status":"ok"' && break; sleep 2; done
( cd orpheus && uv run python ../run_orpheus.py >> ../$LOG 2>&1 ) || say "orpheus failed (see $LOG)"
stop_orpheus
( cd .. && node tts_bench/run_kokoro.js >> tts_bench/$LOG 2>&1 )
( cd .. && node tts_bench/hear.js >> tts_bench/$LOG 2>&1 )
uv run --project ../tune python report.py >> "$LOG" 2>&1
say "done: tts_bench/out/report.md, tts_bench/out/listen.html"
