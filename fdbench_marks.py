"""Stage marks: tell fdbench when each stage of a turn happens inside your assistant. Standard library only.

    from fdbench_marks import mark
    mark("vad_end")                       # your VAD decided the user finished speaking
    mark("asr_final", text=transcript)    # final transcript is ready (text is used for WER)
    mark("llm_request", prompt_tokens=n)  # the LLM call starts (prompt_tokens or messages=..., for context size)
    mark("llm_first_token")               # first token of the reply arrived
    mark("tts_request")                   # text sent to TTS
    mark("tts_first_audio")               # first audio chunk back from TTS
    mark("audio_out")                     # first reply audio handed to the speaker / transport

Each mark carries this machine's time.monotonic(), so all stage times are measured on one clock (the Raspberry Pi's).
When fdbench is not running, mark() does nothing and costs about a microsecond, so the calls can stay in your code.

Where marks go:
  * inside launch_assistant.py (direct feed): over the benchmark socket, nothing to set up;
  * anywhere else (an MCP server in its own process, a non-Pipecat harness, virtual-cable runs): UDP to
    FDBENCH_MARKS=udp://FDBENCH-HOST:8792. launch_assistant.py sets this for the processes it starts.

Pipecat assistants get all of these automatically from pipecat_marks.FdbenchObserver; see docs/STAGES.md.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from typing import Callable, Optional
from urllib.parse import urlparse

STAGES = {
    "vad_start": "your VAD heard the user start speaking",
    "vad_end": "your VAD decided the user stopped speaking (start of the E2E clock)",
    "user_stopped": "the turn was handed on (Pipecat UserStoppedSpeakingFrame); used when vad_end is missing",
    "asr_partial": "a partial transcript arrived (text=...)",
    "asr_final": "the final transcript is ready (text=...)",
    "llm_request": "the LLM call starts (prompt_tokens=..., messages=..., chars=...; speculative=True if it starts "
                   "before the turn has ended)",
    "llm_cancel": "an early (speculative) LLM call was discarded because the turn had not really ended",
    "llm_first_token": "the first token of the LLM reply arrived",
    "llm_usage": "token usage reported by the LLM (prompt_tokens=..., completion_tokens=...)",
    "tts_request": "text was sent to TTS",
    "tts_first_audio": "the first audio chunk came back from TTS",
    "audio_out": "the first reply audio went to the speaker / transport (end of the E2E clock)",
    "interrupted": "barge-in: playback and the LLM stream were cancelled",
    "bot_stopped": "the assistant stopped speaking",
    "tool_start": "a tool / MCP call started (name=...)",
    "tool_end": "a tool / MCP call returned (name=...)",
}

_sink: Optional[Callable[[dict], None]] = None
_udp: Optional[tuple] = None          # (socket, (host, port))
_udp_checked = False
_lock = threading.Lock()
MAX_TEXT = 500


def set_sink(fn: Optional[Callable[[dict], None]]) -> None:
    """Used by launch_assistant.py: send marks over the benchmark socket instead of UDP."""
    global _sink
    _sink = fn


def _udp_target():
    global _udp, _udp_checked
    if _udp_checked:
        return _udp
    with _lock:
        if not _udp_checked:
            url = os.environ.get("FDBENCH_MARKS", "").strip()
            if url:
                try:
                    u = urlparse(url)
                    if u.scheme == "udp" and u.hostname:
                        _udp = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (u.hostname, u.port or 8792))
                except (OSError, ValueError):
                    _udp = None
            _udp_checked = True
    return _udp


def enabled() -> bool:
    return _sink is not None or _udp_target() is not None


def mark(stage: str, **fields) -> None:
    """Record that `stage` happened now. Never raises."""
    t = time.monotonic()
    sink = _sink
    udp = None if sink is not None else _udp_target()
    if sink is None and udp is None:
        return
    d = {"event": "mark", "stage": str(stage), "t": t, "pid": os.getpid()}
    for k, v in fields.items():
        if isinstance(v, str) and len(v) > MAX_TEXT:
            v = v[:MAX_TEXT]
        d[k] = v
    try:
        if sink is not None:
            sink(d)
        else:
            udp[0].sendto(json.dumps(d, default=str).encode("utf-8"), udp[1])
    except Exception:  # noqa: BLE001  (marks never break the assistant)
        pass
