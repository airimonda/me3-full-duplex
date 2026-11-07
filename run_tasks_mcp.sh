#!/bin/bash
# Pi supervisor for the Watson task MCP server (tasks_mcp.py), like
# run_live.sh for the voice loop: restarts it if it dies, logs exits.
# Serves http://127.0.0.1:8320/mcp; jobs run pi.dev (ME3_HARNESS=pi) under
# bubblewrap in ~/sandbox; finished pages and searches are logged to
# show.jsonl, which ui_server.py serves and mac_show.py opens on the Mac.
cd ~/me3
export PATH="$HOME/.local/bin:$PATH"
while true; do
  # finished pages also go to GitHub Pages (deploy key ~/.ssh/watson_sites)
  ME3_HARNESS=${ME3_HARNESS:-pi} ME3_SHOW_LOG=~/me3/runs/live/show.jsonl \
  ME3_PAGES_REPO=~/watson-sites ME3_PAGES_URL=https://airimonda.github.io/watson-sites \
    ~/mcp-venv/bin/python tasks_mcp.py >> ~/me3/tasks_mcp.log 2>&1
  echo "$(date) tasks_mcp exited rc=$?" >> ~/me3/tasks_mcp.log
  sleep 3
done
