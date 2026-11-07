"""ME3 home + music tools: Spotify integration and the virtual smart home.

All state lives in the SANDBOX (~/sandbox/home_state.json) — the same folder the
pi agents may control. Spotify: real API if credentials are present (client id +
secret; refresh token enables actual playback control on a Premium account),
otherwise a virtual player in the sandbox.

Spotify creds (any of these locations):
  SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET / SPOTIFY_REFRESH_TOKEN
  in the environment, ~/.hermes/.env or ~/me3/.env (the Pi).

Playback drives the Spotify desktop app with AppleScript. On the Pi (not
macOS) there is no Spotify app: each command is queued as a structured
{"kind": "spotify", ...} event in the show log, and mac_show.py on the Mac
plays it with a fixed AppleScript (never a script sent from the Pi).
"""
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse

import httpx

HOME_STATE = os.path.expanduser("~/sandbox/home_state.json")
SPOTIFY_API = "https://api.spotify.com/v1"


def _env_key(name):
    v = os.environ.get(name)
    if v:
        return v
    for envpath in (os.path.expanduser("~/.hermes/.env"), os.path.expanduser("~/me3/.env")):
        if os.path.exists(envpath):
            for line in open(envpath):
                line = line.strip()
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip().strip('"')
    return None


def spotify_configured():
    """True when the keys for search + playback control are present."""
    return bool(_env_key("SPOTIFY_CLIENT_ID") and _env_key("SPOTIFY_REFRESH_TOKEN"))


# On the Pi: Spotify commands go to the Mac through the show log (mac_show.py)
_ON_MAC = sys.platform == "darwin"
_SHOW_LOG = os.path.expanduser(os.environ.get("ME3_SHOW_LOG") or "~/me3/runs/live/show.jsonl")


def _to_mac(action, **kw):
    """Queue a Spotify command for mac_show.py. Returns True when queued."""
    try:
        os.makedirs(os.path.dirname(_SHOW_LOG), exist_ok=True)
        with open(_SHOW_LOG, "a") as f:
            f.write(json.dumps({"t": round(time.time(), 3), "kind": "spotify",
                                "action": action, **kw}) + "\n")
        return True
    except OSError:
        return False


def _player(**kw):
    st = _load_state()
    st.setdefault("player", {}).update(kw)
    _save_state(st)


def _load_state():
    try:
        return json.load(open(HOME_STATE))
    except Exception:
        return {"lights": {}, "thermostat": {"target": 22}, "reminders": [],
                "alarms": [], "messages": [], "player": {"state": "stopped"}}


def _save_state(st):
    os.makedirs(os.path.dirname(HOME_STATE), exist_ok=True)
    json.dump(st, open(HOME_STATE, "w"), indent=2)


# ---------------- Spotify ----------------

_sp_token = {"value": None, "expires": 0}


def _spotify_token():
    """App token via client credentials; user token via refresh token if present."""
    global _sp_token
    now = time.time()
    if _sp_token["value"] and now < _sp_token["expires"]:
        return _sp_token["value"]
    cid, sec = _env_key("SPOTIFY_CLIENT_ID"), _env_key("SPOTIFY_CLIENT_SECRET")
    refresh = _env_key("SPOTIFY_REFRESH_TOKEN")
    if not cid or not sec:
        return None
    if refresh:
        r = httpx.post("https://accounts.spotify.com/api/token",
                       data={"grant_type": "refresh_token", "refresh_token": refresh},
                       headers={"Authorization": "Basic " + base64.b64encode(
                           f"{cid}:{sec}".encode()).decode()}, timeout=15)
        tok = r.json().get("access_token")
    else:
        r = httpx.post("https://accounts.spotify.com/api/token",
                       data={"grant_type": "client_credentials"},
                       headers={"Authorization": "Basic " + base64.b64encode(
                           f"{cid}:{sec}".encode()).decode()}, timeout=15)
        tok = r.json().get("access_token")
    if not tok:
        return None
    _sp_token = {"value": tok, "expires": now + 3500}
    return tok


def _sp_headers():
    tok = _spotify_token()
    return {"Authorization": f"Bearer {tok}"} if tok else None


def spotify_search(query, kind="track"):
    h = _sp_headers()
    if not h:
        return None
    r = httpx.get(f"{SPOTIFY_API}/search",
                  params={"q": query, "type": kind, "limit": 10}, headers=h, timeout=15)
    items = [i for i in r.json().get(f"{kind}s", {}).get("items", []) if i]
    if not items:
        return None
    # Spotify's own top hit is often unrelated ("Love Story Taylor Swift" ->
    # Katy Perry, Nov 7): pick the hit whose title + artist share the most
    # words with the request; ties keep Spotify's order.
    def words(t):
        return set(re.findall(r"[a-z0-9]+", (t or "").lower())) - {"by", "the", "a", "song", "play"}
    want = words(query)

    def score(it):
        names = [it.get("name", "")] + [a.get("name", "") for a in it.get("artists") or []]
        return len(want & words(" ".join(names)))
    return max(items, key=lambda it: (score(it), -items.index(it)))


def spotify_status():
    """Current playback from the local app: {'state','track','artist','volume'} or None."""
    ok, out = _as(
        'tell application "Spotify"\n'
        '  set pstate to (player state as text)\n'
        '  set tname to "-"\n'
        '  set tartist to "-"\n'
        '  set tvol to "50"\n'
        '  try\n'
        '    set tname to name of current track\n'
        '    set tartist to artist of current track\n'
        '  end try\n'
        '  try\n'
        '    set tvol to (sound volume as text)\n'
        '  end try\n'
        '  return pstate & "|" & tname & "|" & tartist & "|" & tvol\n'
        'end tell')
    if not ok or not out.strip():
        return None
    parts = (out.strip().split("\n")[-1]).split("|")
    if len(parts) < 4:
        return None
    try:
        vol = int(float(parts[3]))
    except ValueError:
        vol = None
    return {"state": parts[0], "track": parts[1], "artist": parts[2], "volume": vol}


def _as(cmd):
    """Run an AppleScript against the local Spotify app. Returns (ok, err)."""
    try:
        p = subprocess.run(["osascript", "-e", cmd],
                           capture_output=True, text=True, timeout=15)
        return p.returncode == 0, (p.stdout or "").strip()
    except Exception as e:
        return False, str(e)


def _virtual_note(query):
    st = _load_state()
    st["player"] = {"state": "playing", "query": query, "since": time.time()}
    _save_state(st)


def spotify_play(query):
    item = spotify_search(query)
    if not _ON_MAC:
        if item:
            name = item.get("name", query)
            artist = (item.get("artists") or [{}])[0].get("name", "")
            _to_mac("play", uri=item.get("uri"))
            _player(state="playing", query=f"{name} - {artist}" if artist else name,
                    since=time.time())
            return f"Playing {name} by {artist} on Spotify."
        _to_mac("play")   # nothing found (or no keys): resume whatever was playing
        _player(state="playing", query=query or "Spotify", since=time.time())
        return "Playing music on Spotify."
    if not item:
        # no API search (no creds at all) — still try direct AppleScript play
        ok, err = _as(f'tell application "Spotify" to play')
        if ok:
            return "Playing music on Spotify."
        _virtual_note(query)
        return f"(virtual player) Couldn't find {query!r} on Spotify."
    uri = item.get("uri")
    name = item.get("name", query)
    artist = (item.get("artists") or [{}])[0].get("name", "")
    ok, err = _as(f'tell application "Spotify" to play track "{uri}"')
    if ok:
        return f"Playing {name} by {artist} on Spotify."
    _virtual_note(query)
    return (f"Found {name} by {artist}, but Spotify isn't responding on this Mac "
            f"({err[:60]}). Open the Spotify app and try again.")


def spotify_pause():
    if not _ON_MAC:
        _to_mac("pause")
        _player(state="paused")
        return "Paused."
    ok, _ = _as('tell application "Spotify" to pause')
    if ok:
        return "Paused."
    st = _load_state()
    st["player"]["state"] = "paused"
    _save_state(st)
    return "Paused (virtual player)."


def spotify_next():
    if not _ON_MAC:
        _to_mac("next")
        return "Skipped to the next track."
    ok, _ = _as('tell application "Spotify" to next track')
    return "Skipped to the next track." if ok else "Skip failed — is Spotify open?"


def spotify_prev():
    if not _ON_MAC:
        _to_mac("prev")
        return "Previous track."
    ok, _ = _as('tell application "Spotify" to previous track')
    return "Previous track." if ok else "Skip-back failed — is Spotify open?"


def spotify_volume(pct):
    pct = max(0, min(100, int(pct)))
    if not _ON_MAC:
        _to_mac("volume", percent=pct)
        _player(volume=pct)
        return f"Volume set to {pct}%."
    ok, _ = _as(f'tell application "Spotify" to set sound volume to {pct}')
    if ok:
        return f"Volume set to {pct}%."
    st = _load_state()
    st["player"]["volume"] = pct
    _save_state(st)
    return f"Volume set to {pct}% (virtual player)."


# ---------------- Virtual home ----------------

def _light(name=None):
    st = _load_state()
    lights = st.setdefault("lights", {})
    name = (name or "").lower().strip()
    if not name or name in ("the", "all", "lights"):
        name = "default"
    return st, lights, name


def lights_set(on=None, brightness=None, color=None, name=None):
    st, lights, name = _light(name)
    L = lights.setdefault(name, {"on": False, "brightness": 100, "color": "warm white"})
    if on is not None:
        L["on"] = bool(on)
    if brightness is not None:
        L["brightness"] = max(1, min(100, int(brightness)))
        L["on"] = True
    if color:
        L["color"] = color
        L["on"] = True
    st["lights"] = lights
    _save_state(st)
    return (f"{name.capitalize()} lights "
            f"{'on' if L['on'] else 'off'}"
            + (f", {L['brightness']}% brightness" if L.get("on") else "")
            + (f", {L['color']}" if L.get("on") and color else "") + ".")


def lights_query():
    st = _load_state()
    return json.dumps(st.get("lights", {}))


def thermostat_set(temp):
    temp = int(temp)
    st = _load_state()
    st["thermostat"] = {"target": temp, "since": time.time()}
    _save_state(st)
    return f"Thermostat set to {temp} degrees."


# ---------------- Reminders / alarms / messaging ----------------

def add_reminder(text, when=None):
    st = _load_state()
    st.setdefault("reminders", []).append({"text": text, "when": when or "",
                                           "added": time.strftime("%Y-%m-%d %H:%M")})
    _save_state(st)
    return f"Reminder saved: {text}" + (f" for {when}" if when else "") + "."


def list_items(kind):
    st = _load_state()
    items = st.get(kind, [])
    if not items:
        return f"No {kind} yet."
    if kind == "reminders":
        return "Your reminders: " + "; ".join(
            r["text"] + (f" ({r['when']})" if r.get("when") else "") for r in items)
    return "Your alarms: " + "; ".join(a.get("time", "?") + (" " + a.get("label", "")).strip()
                                       for a in items)


def set_alarm(hhmm, label=""):
    st = _load_state()
    st.setdefault("alarms", []).append({"time": hhmm, "label": label})
    _save_state(st)
    return f"Alarm set for {hhmm}" + (f" — {label}" if label else "") + "."


def send_message(kind, who, text=""):
    st = _load_state()
    st.setdefault("messages", []).append({"kind": kind, "to": who, "text": text,
                                          "at": time.strftime("%Y-%m-%d %H:%M")})
    _save_state(st)
    return (f"{kind.capitalize()} to {who} logged in the sandbox. No real messaging "
            f"is connected in this build.")


# ---------------- Unified tool dispatch ----------------

def do_tool(name, args):
    a = args or {}
    if name == "spotify_play":
        return spotify_play(a.get("query", "music"))
    if name == "spotify_pause":
        return spotify_pause()
    if name == "spotify_next":
        return spotify_next()
    if name == "spotify_prev":
        return spotify_prev()
    if name == "spotify_volume":
        return spotify_volume(a.get("percent", 50))
    if name == "lights_control":
        on = a.get("on")
        return lights_set(on=on if on is None else str(on).lower() in ("true", "1", "on"),
                          brightness=a.get("brightness_percent"),
                          color=a.get("color"), name=a.get("light"))
    if name == "thermostat_set":
        return thermostat_set(a.get("degrees"))
    if name == "set_alarm":
        return set_alarm(a.get("time", ""), a.get("label", ""))
    if name == "add_reminder":
        return add_reminder(a.get("text", ""), a.get("when", ""))
    if name == "list_items":
        return list_items(a.get("kind", "reminders"))
    if name == "contact_message":
        return send_message(a.get("kind", "message"), a.get("who", ""), a.get("text", ""))
    return f"Unknown tool {name}"