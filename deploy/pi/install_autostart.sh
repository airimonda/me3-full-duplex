#!/bin/bash
# Start Watson by itself when the Pi is plugged in (owner request, Nov 7).
# The Pi auto-logs in to the desktop as abnunez, which starts the user's
# systemd manager and PipeWire; these user units then start the four
# launchers. No sudo needed. Run on the Pi:  bash ~/me3/deploy/pi/install_autostart.sh
# Undo:  systemctl --user disable --now watson-{live,tasks,ui,spotify}
set -e
UNITS="watson-live watson-tasks watson-ui watson-spotify"
mkdir -p ~/.config/systemd/user
for u in $UNITS; do cp ~/me3/deploy/pi/$u.service ~/.config/systemd/user/; done
systemctl --user daemon-reload
systemctl --user enable $UNITS
echo "enabled: $UNITS (start now with: systemctl --user start $UNITS)"
