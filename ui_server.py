"""Watson dashboard v2 — serves the retained vcm-me2 dashboard UI, wired to ME3.

  ~/me3/venv/bin/python ui_server.py     (port 8301)

Bridges ME3 to the old dashboard's WebSocket contract (vcm-me2/runtime/ui_server.py):
  server -> client: {"type": "state", "state": {...}}   (mode, music, light, thermostat,
                                                        timers, alarms, reminders, ...)
                    {"type": "turn", id, t, heard:{...}, reply:{...}|null}
                    {"type": "answer"|"log", ...}
  client -> server: mark/dismiss/command  -> accepted and ignored (no test slots in ME3)

Data sources: ~/me3/chat_log.jsonl (conversation) and ~/sandbox/home_state.json
(music / lights / thermostat / reminders / alarms written by home.py).
"""
import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.expanduser("~/me3"))
import home  # noqa: E402

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

ME3 = os.path.expanduser("~/me3")
LOG_PATH = os.path.join(ME3, "chat_log.jsonl")
HOME_STATE = os.path.expanduser("~/sandbox/home_state.json")
UI_DIR = os.path.expanduser("~/me3/vcm-me2/runtime/ui")
SANDBOX = os.path.expanduser("~/sandbox")

app = FastAPI()


def _spotify_connected():
    p = os.path.expanduser("~/.hermes/.env")
    try:
        return "SPOTIFY_REFRESH_TOKEN" in open(p).read()
    except OSError:
        return False


def _home_state():
    try:
        return json.load(open(HOME_STATE))
    except Exception:
        return {}


def dashboard_state():
    """ME3 state in the old dashboard's shape (runtime/state.py DEFAULT_STATE)."""
    st = _home_state()
    player = st.get("player", {})
    lights = st.get("lights", {})
    # aggregate: show the most recently turned-on light (any on = on)
    L = {}
    on_lights = [l for l in lights.values() if l.get("on")]
    if on_lights:
        L = on_lights[-1]
    thermo = st.get("thermostat", {})
    mode = "idle"
    try:
        with open(LOG_PATH, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 700))
            tail = f.read().decode(errors="replace").strip().splitlines()
        if tail:
            last = json.loads(tail[-1])
            if time.time() - float(last.get("t", 0)) < 4:
                mode = "speaking"
    except Exception:
        pass
    return {
        "mode": mode,
        "music": {"connected": _spotify_connected(),
                  "playing": player.get("state") == "playing",
                  "paused": player.get("state") == "paused",
                  "track": player.get("query"), "volume": player.get("volume"),
                  "device": "macbook-air", "provider": "spotify", "error": None},
        "light": {"connected": True, "on": bool(L.get("on")),
                  "brightness": L.get("brightness", 60),
                  "color": "#FFFFFF", "color_name": L.get("color", "White")},
        "thermostat": {"on": True, "setpoint": thermo.get("target", 22),
                       "room_temp": float(thermo.get("target", 22)) + 1.5},
        "timers": [],
        "alarms": [{"id": f"a{i+1}", "time": a.get("time", ""), "ampm": "",
                    "label": a.get("label", ""), "ringing": False}
                   for i, a in enumerate(st.get("alarms", []))],
        "reminders": [{"id": f"r{i+1}", "topic": r.get("text", ""), "created": time.time()}
                      for i, r in enumerate(st.get("reminders", []))],
        "ringing": None, "call": None, "weather": None,
    }


def log_entries():
    """All chat_log.jsonl entries as (t, dict)."""
    out = []
    try:
        with open(LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    out.append((float(e.get("t", 0)), e))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return out


def entries_to_messages(e):
    """One chat_log entry -> list of dashboard messages (old contract)."""
    t = float(e.get("t", time.time()))
    if "event" in e:
        ev = e["event"]
        if ev == "idle_ignored":
            return [{"type": "turn", "id": f"i{t}", "t": t,
                     "heard": {"text": e.get("user", ""), "accepted": False,
                               "reason": "wake word missing"}, "reply": None}]
        if ev == "task_submitted":
            return [{"type": "log", "level": "info",
                     "msg": f"Task started: {e.get('prompt', '')}"}]
        if ev == "announce":
            return [{"type": "log", "level": "info", "msg": f"Done: {e.get('line', '')}"}]
        if ev == "auto_search":
            return [{"type": "log", "level": "info", "msg": "Searched the web"}]
        return [{"type": "log", "level": "info", "msg": str(ev)}]
    heard = {"text": e.get("user", ""), "accepted": True, "prob": None, "slot": ""}
    reply = {"text": e.get("reply", "")} if e.get("reply") else None
    return [{"type": "turn", "id": f"t{t}", "t": t, "heard": heard, "reply": reply}]


def _log_offset():
    try:
        return os.path.getsize(LOG_PATH)
    except OSError:
        return 0


# ---------------- routes ----------------

@app.get("/")
async def index(request=None):
    return FileResponse(os.path.join(UI_DIR, "index.html"))


def tasks_scan():
    """Background agent tasks in ~/sandbox/task*/ -> dashboard rows."""
    out = []
    try:
        for name in sorted(os.listdir(SANDBOX), reverse=True):
            d = os.path.join(SANDBOX, name)
            if not name.startswith("task") or not os.path.isdir(d):
                continue
            logp = os.path.join(d, "run.log")
            age = time.time() - os.path.getmtime(logp) if os.path.exists(logp) else 1e9
            status = "working" if age < 30 else "finished"
            prompt = ""
            try:
                raw = open(os.path.join(d, "prompt.txt")).read().strip()
                m = re.search(r"Task:\s*(.+)", raw)
                if m:
                    prompt = m.group(1).split("\n")[0][:70]
            except OSError:
                pass
            # 'Done' announcements arrive as log events; show finished only
            out.append({"id": name, "status": status, "age": int(age),
                        "prompt": prompt})
    except OSError:
        pass
    return out[:6]


@app.get("/api/state")
async def api_state():
    msgs = []
    for _, e in log_entries():
        msgs.extend(entries_to_messages(e))
    try:
        music_real = home.spotify_status()
    except Exception:
        music_real = None
    return JSONResponse({"messages": msgs, "state": dashboard_state(),
                         "tasks": tasks_scan(), "home": _home_state(),
                         "music": music_real})


# dashboard may only drive the instant smart-home tools — never delegation
TOOL_WHITELIST = {"spotify_pause", "spotify_next", "spotify_prev", "spotify_volume",
                  "lights_control", "thermostat_set", "spotify_play"}


@app.post("/api/tool")
async def api_tool(req: dict):
    name = req.get("name", "")
    args = req.get("args") or {}
    if name not in TOOL_WHITELIST:
        return JSONResponse({"ok": False, "error": "tool not allowed"}, status_code=400)
    try:
        out = home.do_tool(name, args)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    try:
        music_real = home.spotify_status()
    except Exception:
        music_real = None
    return JSONResponse({"ok": True, "reply": out, "state": dashboard_state(),
                         "home": _home_state(), "music": music_real})


@app.get("/files/{task_id}/{name}")
async def task_file(task_id: str, name: str):
    if not task_id.startswith("task") or "/" in name or name.startswith("."):
        return HTMLResponse("not found", status_code=404)
    path = os.path.join(SANDBOX, task_id, name)
    if not os.path.isfile(path):
        return HTMLResponse("not found", status_code=404)
    return FileResponse(path)


@app.websocket("/ws")
async def ws_dashboard(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "state", "state": dashboard_state()})
    for _, e in log_entries():
        for m in entries_to_messages(e):
            await ws.send_json(m)
    offset = _log_offset()
    last_state = json.dumps(dashboard_state(), sort_keys=True)
    try:
        while True:
            await asyncio.sleep(1.0)
            new_offset = _log_offset()
            if new_offset > offset:
                try:
                    with open(LOG_PATH) as f:
                        f.seek(offset)
                        new_data = f.read()
                    offset = new_offset
                    for line in new_data.strip().splitlines():
                        try:
                            e = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        for m in entries_to_messages(e):
                            await ws.send_json(m)
                except FileNotFoundError:
                    pass
            st_json = json.dumps(dashboard_state(), sort_keys=True)
            if st_json != last_state:
                last_state = st_json
                await ws.send_json({"type": "state", "state": json.loads(st_json)})
    except (WebSocketDisconnect, RuntimeError):
        return


app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8301, log_level="warning")