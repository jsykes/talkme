#!/usr/bin/env bash
# Stops the GPU work of a training pipeline (pipeline.py recipes, the older pipeline_r*.sh scripts, training, replays,
# live tests, voice servers, llama-server) so the GPU is free. Continue later by rerunning the same pipeline command,
# e.g. cd tune && uv run python pipeline.py --recipe recipes/g26.json : finished stages and cached calls are skipped and
# training resumes from its last saved step (every 50 steps); a replay or live-test stage that was cut restarts.
#   bash tune/gpu_pause.sh
cd "$(dirname "$0")"
pat='pipeline_r[0-9]|pipeline\.py|train_lora\.py|export_gguf\.py|evaluate\.py|probe_answers\.py|gen_dialogues\.py|memory_check\.py|live_bench|audio_test\.js|llama-server|serve_orpheus|serve_chatterbox'
if command -v powershell.exe >/dev/null; then
powershell.exe -NoProfile -Command "
\$pat = '$pat'
\$p = Get-CimInstance Win32_Process | Where-Object { \$_.ProcessId -ne \$PID -and \$_.CommandLine -match \$pat -and \$_.CommandLine -notmatch 'gpu_pause' }
\$p | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }
'stopped ' + @(\$p).Count + ' processes'"
else
  pids=$(pgrep -f "$pat" | grep -vx "$$")
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null
  echo "stopped $(echo $pids | wc -w) processes"
fi
log=$(ls -t out/*.pipeline.log 2>/dev/null | head -1); echo "[pause] $(date +%H:%M:%S) paused (GPU free); see the comment at the top of tune/gpu_pause.sh to continue" | tee -a "${log:-out/gpu_pause.log}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/GPU memory in use: /'
