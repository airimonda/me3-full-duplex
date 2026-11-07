#!/bin/bash
# Pi: Spotify Connect device "Watson" on the Pi speaker (Premium account).
# librespot comes out of the raspotify .deb into ~/.local/bin, no sudo
# (setup_spotify_pi.sh). It plays through PipeWire's PulseAudio server, so
# music shares the speaker with Watson's voice and the agent can duck it.
# Ported from vcm-me2 deploy/librespot.service, including its power-cut fix:
# after an unclean shutdown Spotify keeps a stale "Watson" session and
# answers play/pause with 500/502; a throw-away player ("WatsonReset")
# signing in for ~12 s takes over the account's session and clears it.
#   setsid nohup bash ~/me3/run_spotify.sh >/dev/null 2>&1 </dev/null &
LS=~/.local/bin/librespot
while true; do
  for i in $(seq 30); do getent hosts apresolve.spotify.com >/dev/null && break; sleep 2; done
  timeout 12 "$LS" --name WatsonReset --device-type speaker --backend pulseaudio \
    --cache ~/.cache/librespot --disable-audio-cache --disable-discovery >> ~/me3/spotify.log 2>&1
  sleep 5
  "$LS" --name Watson --device-type speaker --backend pulseaudio \
    --bitrate 160 --initial-volume 100 --enable-volume-normalisation \
    --cache ~/.cache/librespot --disable-audio-cache >> ~/me3/spotify.log 2>&1
  echo "$(date) librespot exited rc=$?" >> ~/me3/spotify.log
  sleep 5
done
