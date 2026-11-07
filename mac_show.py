"""Mac viewer for Watson on the Pi: whatever Watson shows (a finished page,
a web search) opens in the Mac's browser within about a second.

Polls the Pi's ui_server.py (/api/show) and runs macOS `open` on each new
event. Starts from "now", so old results are never re-opened.

Run on the Mac:  ./venv/bin/python mac_show.py [--pi http://192.168.1.80:8301]
Also open the Watson UI itself:  open http://192.168.1.80:8301
"""
import argparse
import json
import subprocess
import time
import urllib.request

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
            target = base + e["path"] if e.get("kind") == "deliverable" else e.get("url")
            if target:
                subprocess.Popen(["open", target])
                print(f"[mac_show] opened {target}", flush=True)
            since = max(since, float(e["t"]))
    except Exception as e:
        print(f"[mac_show] poll failed: {e}", flush=True)
        time.sleep(2)
    time.sleep(args.every)
