# ME3 environment notes (updated Oct 26)

## Pi ("watsonpi")
- Tailscale IP: <pi-tailscale-ip>, tailnet <tailnet>. Ping from Mac: ~105 ms (DERP relay,
  no direct path yet — may improve once both are on the same LAN).
- SSH: port 22 open, publickey-only. NO working key on this Mac. Unresolved — must be
  sorted before deploying ME3 code to the Pi (deadline: before Nov 1). Options: key holder's
  machine, or SD-card mount (no reflash needed).
- Currently runs the legacy "Watson" runtime: web UI on :8080 (now/timers/activity/
  diagnostics), backend runtime/ui_server.py, WebSocket /ws, sends {type:"command"}.
- ME3 decision: Watson is legacy; only its dashboard is retained. ME3 is a new pipeline.
  When deploying ME3, the Watson runtime stops (mic access conflict) or the Pi is reflashed.
- The Pi holds a model (user statement) — reflash acceptable if needed.

## Mac (this machine, M5 16GB)
- ~/me3/venv (Python 3.11): mlx-whisper, kokoro, soundfile, en_core_web_sm 3.8.0 (direct
  wheel; never `spacy download` on this machine — it targets ~/.venv, the wrong interpreter).
- Timing results in step1-results.md; Hermes spike in step2-results.md.
- Hermes: one-shot confirmed; mcp extra installed; webfetch MCP added (non-TTY add needs
  `printf 'y\n' |`); sandbox caveat: always pass --in DIR.
