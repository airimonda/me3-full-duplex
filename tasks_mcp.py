"""Watson task MCP server (runs on the Raspberry Pi, next to the voice loop).

The Pi voice loop hands big jobs to this server and keeps talking while
they run. Every job runs through tasks.py: a pi.dev agent under
sandbox-exec, writing ONLY inside ~/sandbox/taskNN. When a job finishes, a
watcher thread opens its deliverable on the Pi screen (Chromium, :0) and queues a
spoken announcement (no file paths, no URLs) for the Pi to collect.

Tools
  submit_task(request)  start a background job; returns {"id", "ack"}
  list_tasks()          running jobs with elapsed seconds
  finished()            announcements not yet collected (each returned once)
  browse(query)         web search for facts the talker does not know: opens
                        the results in the Mac browser and returns snippets

Run (Pi):  ~/mcp-venv/bin/python ~/me3/tasks_mcp.py   # http://127.0.0.1:8320/mcp
Env: ME3_TASKS_MCP_PORT (8320), ME3_BROWSE_OPEN=1 to also show searches on screen.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks  # noqa: E402
from brain import speakable  # noqa: E402

PORT = int(os.environ.get("ME3_TASKS_MCP_PORT", "8320"))
HOST = os.environ.get("ME3_TASKS_MCP_HOST", "127.0.0.1")   # same box as Watson
# owner, Nov 6: searches are spoken only; nothing opens on screen
BROWSE_OPEN = os.environ.get("ME3_BROWSE_OPEN", "0") == "1"

_lock = threading.Lock()
_announce: list[dict] = []   # finished jobs waiting for the Pi to speak them


def log(*a):
    print("[tasks_mcp]", *a, file=sys.stderr, flush=True)


def _watch(poll_s: float = 1.0):
    """Open each deliverable the moment its job ends; queue the line."""
    while True:
        try:
            with _lock:
                done = tasks.check_finished()
                for tid in done:
                    line = speakable(tasks.announcement(tid) or "")
                    opened = tasks.open_deliverable(tid)
                    f = tasks._finished.get(tid, {})
                    _announce.append({"id": tid, "line": line,
                                      "status": f.get("status", "?"),
                                      "opened": bool(opened),
                                      "t": time.time()})
                    log(f"finished {tid} status={f.get('status')} "
                        f"opened={bool(opened)} line={line!r}")
        except Exception as e:  # never let the watcher die
            log("watch error:", repr(e))
        time.sleep(poll_s)


def submit(request: str) -> dict:
    request = (request or "").strip()
    if not request:
        raise ValueError("request is empty")
    with _lock:
        if len(tasks._running) >= tasks.MAX_CONCURRENT:
            return {"id": None, "ack": "My hands are full right now. "
                    "I'll take that one as soon as a slot frees up."}
        tid = tasks.submit_task(request)
    log(f"submitted {tid}: {request[:80]!r}")
    if not tid:
        return {"id": None, "ack": "Sorry, I couldn't start that task."}
    return {"id": tid, "ack": "On it. I'll let you know when it's done."}


def do_browse(query: str) -> str:
    query = (query or "").strip()
    if not query:
        raise ValueError("query is empty")
    if BROWSE_OPEN:
        url = "https://duckduckgo.com/?q=" + urllib.parse.quote_plus(query)
        try:
            tasks.open_url(url)
        except Exception as e:
            log("browser open failed:", repr(e))
    import brain
    return brain.local_web_search(query)


def _build_server():
    from mcp.server.fastmcp import FastMCP
    app = FastMCP("me3-watson-tasks", host=HOST, port=PORT,
                  stateless_http=True, json_response=True)

    @app.tool()
    def submit_task(request: str) -> str:
        """Start a big background job (build a site, research, write files).
        The agent works only inside ~/sandbox. Returns JSON {"id", "ack"}."""
        return json.dumps(submit(request))

    @app.tool()
    def list_tasks() -> str:
        """Running background jobs with elapsed seconds, as JSON."""
        with _lock:
            now = time.time()
            return json.dumps({tid: {"prompt": t["prompt"][:80],
                                     "elapsed_s": round(now - t["t0"])}
                               for tid, t in tasks._running.items()})

    @app.tool()
    def finished() -> str:
        """Announcements for finished jobs, each returned once, as a JSON
        list of {"id", "line", "status", "opened"}. Lines are speakable."""
        with _lock:
            out = list(_announce)
            _announce.clear()
        return json.dumps(out)

    @app.tool()
    def browse(query: str) -> str:
        """Look up facts the talker does not know: returns the top web
        results as text for Watson to answer from (nothing opens on screen
        unless ME3_BROWSE_OPEN=1)."""
        return do_browse(query)

    return app


if __name__ == "__main__":
    threading.Thread(target=_watch, daemon=True).start()
    log(f"serving http://{HOST}:{PORT}/mcp  sandbox={tasks.TASKS_DIR}")
    _build_server().run(transport="streamable-http")
