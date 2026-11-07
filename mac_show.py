"""Mac viewer for Watson on the Pi: whatever Watson shows (a finished page,
a web search) opens in the Mac's browser within about a second.

Polls the Pi's ui_server.py (/api/show) and runs macOS `open` on each new
event. Starts from "now", so old results are never re-opened.

Run on the Mac:  ./venv/bin/python mac_show.py [--pi http://192.168.1.80:8301]
Also open the Watson UI itself:  open http://192.168.1.80:8301
"""
import argparse
import json
import re
import subprocess
import time
import urllib.request

# Spotify commands from the Pi (home.py queues them). Only these fixed
# AppleScripts run; the Pi never sends script text.
_SPOTIFY_URI = re.compile(r"^spotify:(track|album|playlist|artist):[A-Za-z0-9]{10,40}$")


def spotify(e):
    act = e.get("action")
    if act == "play":
        uri = e.get("uri") or ""
        script = (f'tell application "Spotify" to play track "{uri}"'
                  if _SPOTIFY_URI.match(uri) else 'tell application "Spotify" to play')
    elif act == "pause":
        script = 'tell application "Spotify" to pause'
    elif act == "next":
        script = 'tell application "Spotify" to next track'
    elif act == "prev":
        script = 'tell application "Spotify" to previous track'
    elif act == "volume":
        pct = max(0, min(100, int(e.get("percent", 50))))
        script = f'tell application "Spotify" to set sound volume to {pct}'
    else:
        return f"ignored spotify action {act!r}"
    # 30 s: a cold Spotify launch took >15 s on the first play (Nov 7)
    p = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=30)
    return f"spotify {act}: " + ("ok" if p.returncode == 0 else (p.stderr or "failed").strip()[:80])

ap = argparse.ArgumentParser()
ap.add_argument("--pi", default="http://watsonpi.local:8301")
ap.add_argument("--every", type=float, default=1.0)
args = ap.parse_args()
base = args.pi.rstrip("/")


def fetch(since):
    with urllib.request.urlopen(f"{base}/api/show?since={since}", timeout=5) as r:
        return json.load(r)


since = None
while since is None:          # anchor at the Pi's clock, not the Mac's
    try:
        since = fetch(time.time() + 1e9)["now"]
    except Exception as e:
        print(f"[mac_show] waiting for {base}: {e}", flush=True)
        time.sleep(3)
print(f"[mac_show] watching {base}", flush=True)

while True:
    try:
        d = fetch(since)
        for e in d["events"]:
            if e.get("kind") == "spotify":
                print(f"[mac_show] {spotify(e)}", flush=True)
                since = max(since, float(e["t"]))
                continue
            target = base + e["path"] if e.get("kind") == "deliverable" else e.get("url")
            if target:
                subprocess.Popen(["open", target])
                print(f"[mac_show] opened {target}", flush=True)
            since = max(since, float(e["t"]))
    except Exception as e:
        print(f"[mac_show] poll failed: {e}", flush=True)
        time.sleep(2)
    time.sleep(args.every)
