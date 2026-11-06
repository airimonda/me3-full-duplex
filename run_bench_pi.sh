#!/bin/bash
# Pi side of a remote fd-benchmark run: keeps launch_assistant.py connected
# to the harness on the laptop (FDBENCH_URL in fd-benchmark/.env), restarting
# it after each round (multi-round remote runs wedge otherwise).
# Same backends as live mode (run_live.sh), bench mode for the assistant.
cd ~/me3/fd-benchmark
export FDBENCH_ENTRY=~/me3/bench_entry.py FDBENCH_PROJECT=~/me3 ME3_MODE=bench \
  ME3_ASR=deepgram ME3_TTS=deepgram ME3_DG_VOICE=aura-2-jupiter-en \
  ME3_DG_KEYTERMS="Watson,halo-halo,adobo,sinigang,Pan de Manila" \
  ME3_TASKS_MCP=http://127.0.0.1:8320/mcp ME3_SERVER=http://192.168.1.65:8300
while true; do
  ~/me3-venv/bin/python launch_assistant.py >> ~/me3/bench_pi.log 2>&1
  echo "$(date) launch_assistant exited rc=$?" >> ~/me3/bench_pi.log
  sleep 2
done
