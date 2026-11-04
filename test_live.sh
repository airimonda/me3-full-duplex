#!/usr/bin/env bash
# ME3 live test script — run on the Mac while talking to the Pi (watsonpi).
# Usage: bash test_live.sh [--watch N]
#   --watch N : keep printing new turns every N seconds (default 5)

PI=watsonpi
LOG=/home/abnunez/me3/fd-benchmark/runs/_assistant/watson_log.jsonl

say() { echo -e "\n== $1"; }

case "$1" in
  --watch)
    echo "Watching live turns (ctrl-c to stop). Speak to Watson now."
    ssh -o BatchMode=yes $PI "tail -f -n 0 $LOG" | while read -r line; do
      echo "$line" | python3 -c '
import json, sys
try: e = json.load(sys.stdin)
except Exception: sys.exit()
ev = e.get("event")
if ev == "turn":
    print(f"[turn] you: {e[\"user\"][:70]!r}")
    print(f"       watson: {e[\"reply\"][:90]!r}  (asr {e[\"asr_s\"]}s, 1st tok {e[\"first_token_s\"]}s, total {e[\"total_s\"]}s)")
elif ev == "barge_in":
    print("[barge-in] you cut her off")
elif ev == "voice_refreshed":
    print("[voice] profile refreshed")
elif ev == "enrolled":
    print("[voice] NEW speaker enrolled")
elif ev == "stranger_ignored":
    print(f"[voice] stranger ignored (sim {e.get(\"sim\")})")
'
    done
    exit 0
    ;;
esac

echo "ME3 live checklist"
echo "1. First, say: 'Watson' alone -> she should greet: 'Yes? I'm listening.'"
echo "2. Say: 'Watson, what time is it right now?' -> answer within ~3s"
echo "3. Say WITHOUT wake word: 'How many miles is a 10K?' -> should still answer"
echo "4. While she is answering, interrupt mid-sentence with a new question"
echo "   -> she must stop immediately (<0.5s) and answer the new one"
echo "5. Say: 'mm-hmm' / 'yeah' while she talks -> she should keep talking (backchannel)"
echo "6. Say: 'Watson, set a timer for 10 seconds' -> confirms, then announces"
echo "7. Say: 'Watson, have the agent research the best pizza near campus'"
echo "   -> NOTE: tasks currently cannot run from the Pi (needs Mac harness wiring);"
echo "   she should acknowledge gracefully, not crash"
echo "8. Have someone ELSE say: 'Watson, what time is it?' -> stranger_ignored expected"
echo "   (only after YOUR voice is enrolled — first long utterance from you enrolls it)"
echo "9. Leave TV/radio audio playing nearby -> she should stay silent"
echo

read -rp "Run the watcher now? [y/N] " a
[[ "$a" == "y" ]] && exec "$0" --watch 5
