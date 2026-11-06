"""Minimal MCP client for the Pi: tools/call over streamable HTTP with plain
httpx (the Pi venv has no `mcp` package, and the night before the demo is
not the time to install one). Talks to tasks_mcp.py, which runs stateless
with JSON responses, so one POST per call is a complete MCP exchange.

ME3_TASKS_MCP=http://<mac-ip>:8320/mcp enables it; unset = disabled.
"""
from __future__ import annotations

import itertools
import json
import os

import httpx

URL = os.environ.get("ME3_TASKS_MCP", "").strip()
_ids = itertools.count(1)
_HEADERS = {"Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": "2025-06-18"}


class MCPError(RuntimeError):
    pass


def enabled() -> bool:
    return bool(URL)


def _rpc(method: str, params: dict, timeout: float) -> dict:
    body = {"jsonrpc": "2.0", "id": next(_ids), "method": method, "params": params}
    r = httpx.post(URL, json=body, headers=_HEADERS, timeout=timeout)
    r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    if "text/event-stream" in ctype:  # tolerate an SSE-framed reply
        data = [ln[5:].strip() for ln in r.text.splitlines() if ln.startswith("data:")]
        msg = json.loads(data[-1]) if data else {}
    else:
        msg = r.json()
    if "error" in msg:
        raise MCPError(str(msg["error"])[:200])
    return msg.get("result", {})


def call(tool: str, timeout: float = 20.0, **args) -> str:
    """Call one tool and return its text content."""
    res = _rpc("tools/call", {"name": tool, "arguments": args}, timeout)
    text = "".join(c.get("text", "") for c in res.get("content", [])
                   if c.get("type") == "text")
    if res.get("isError"):
        raise MCPError(text[:200] or "tool error")
    return text


def list_tools(timeout: float = 10.0) -> list[str]:
    return [t["name"] for t in _rpc("tools/list", {}, timeout).get("tools", [])]
