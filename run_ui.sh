#!/bin/bash
# Pi supervisor for the Watson UI (ui_server.py, port 8301), viewed from the
# Mac at http://<pi>:8301. Reads the live voice-loop log and ~/sandbox.
cd ~/me3
while true; do
  ME3_SHOW_LOG=~/me3/runs/live/show.jsonl \
    ~/mcp-venv/bin/python ui_server.py >> ~/me3/ui_server.log 2>&1
  echo "$(date) ui_server exited rc=$?" >> ~/me3/ui_server.log
  sleep 3
done
