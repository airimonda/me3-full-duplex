#!/bin/bash
cd ~/me3
while true; do
  # USB PnP mic: 100% gain clips; reboots and PipeWire restarts reset it
  amixer -c 3 sset Mic 81% >/dev/null 2>&1
  amixer -c 2 sset PCM 100% unmute >/dev/null 2>&1
  # speaker at -6 dB (wpctl is cubic: 0.8 = 0.51 linear): the owner must be louder than the echo at the mic
  # for keyword barge-in (Whisper transcribes the dominant voice)
  wpctl set-volume @DEFAULT_AUDIO_SINK@ 0.8 >/dev/null 2>&1
  ME3_ASR=deepgram ME3_TTS=deepgram ME3_DG_VOICE=aura-2-jupiter-en ME3_DG_KEYTERMS="Watson,halo-halo,adobo,sinigang,Pan de Manila" ME3_SERVER=http://192.168.1.65:8300 ME3_TASKS_MCP=http://127.0.0.1:8320/mcp ME3_TALKER_MODEL=z-ai/glm-5.3-flash \
    ~/me3-venv/bin/python pipecat_assistant.py --mic >> ~/me3/live.log 2>&1
  echo "$(date) exited rc=$?" >> ~/me3/live.log
  sleep 3
done
