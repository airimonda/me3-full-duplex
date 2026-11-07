#!/bin/bash
# Pi: Spotify Connect device "Watson" on the Pi speaker (librespot from the
# raspotify package; Premium account). Runs as the user, through PipeWire's
# ALSA plugin, so music shares the speaker with Watson's voice and the agent
# can duck it (pw-dump + wpctl). The system raspotify service stays disabled.
# First run: in any Spotify app on the same network, pick the device
# "Watson" once; librespot caches the login in ~/.cache/librespot.
#   setsid nohup bash ~/me3/run_spotify.sh >/dev/null 2>&1 </dev/null &
while true; do
  librespot --name Watson --device-type speaker \
    --backend alsa --device default \
    --bitrate 160 --initial-volume 70 --enable-volume-normalisation \
    --cache ~/.cache/librespot >> ~/me3/spotify.log 2>&1
  echo "$(date) librespot exited rc=$?" >> ~/me3/spotify.log
  sleep 3
done
