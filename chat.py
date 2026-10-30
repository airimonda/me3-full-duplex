"""ME3 full-duplex conversation mode — always listening, barge-in via echo correlation.

  ~/me3/venv/bin/python chat.py

Full duplex: the mic runs during playback. Speech detected during playback is
tested against the audio the agent is playing (normalized cross-correlation over
a delay search): high correlation = self-echo, ignore; low correlation = the
user barging in -> stop playback immediately and handle the interruption.

Tunables (env): ME3_SILENCE_MS (endpointing, default 400),
ME3_BARGE_MS (min sustained speech for a barge-in candidate, default 150),
ME3_ECHO_THRESH (correlation above which speech is judged self-echo, default 0.60).
"""
import io
import json
import os
import queue
import random
import re
import threading
import time
import wave

import httpx
import numpy as np
import sounddevice as sd
from silero_vad import load_silero_vad, VADIterator

from loop import (load_openrouter_key, transcribe_http, transcribe_full, speak_http,
                  wav_to_bytes, HTTP)  # noqa: E402
from duplex import DuplexGate, DuplexParams  # noqa: E402
import speakerid  # noqa: E402
import home  # noqa: E402
import tasks as taskmgr  # noqa: E402

WAKE_WORD = os.environ.get("ME3_WAKE_WORD", "watson").lower()
ACTIVE_WINDOW_S = int(os.environ.get("ME3_ACTIVE_WINDOW", "20"))  # legacy
speaker_emb = None        # enrolled voice of the first person who addresses Watson
awake = False             # True from the wake word until "goodbye"
pending_enroll = False    # waiting for a long-enough utterance to enroll the voice
stranger_rejects = []     # timestamps of rejected wake attempts (owner-lockout escape)


def _lev(a, b):
    """Levenshtein distance for fuzzy wake-word matching (ASR mishearings)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wake_match(text):
    """Returns (matched: bool, chars_consumed: int). Fuzzy-matches the wake word
    against the first 1-2 tokens — ASR renders 'Watson' as 'Orton', 'Watch on',
    'button' etc. Levenshtein <= 3 over the concatenated tokens."""
    toks = list(re.finditer(r"[A-Za-z']+", text))
    if not toks:
        return False, 0
    cand = toks[0].group().lower()
    consumed_end = toks[0].end()
    if len(toks) > 1 and toks[1].start() - toks[0].end() <= 1:
        cand += toks[1].group().lower()
        consumed_end = toks[1].end()
    cand = cand.replace("'", "")[:8]
    d = _lev(cand, WAKE_WORD)
    # d<=2: clear match. d==3 only for watson-shaped mishearings (two-word splits
    # like "watch on"/"ort on", or w/o onsets) — rejects random words like "button".
    ok = len(cand) >= 4 and (d <= 2 or (d == 3 and cand[0] in "wov"))
    if not ok:
        return False, 0
    # consume trailing punctuation/space after the matched tokens
    while consumed_end < len(text) and text[consumed_end] in ".,!:' \t":
        consumed_end += 1
    return True, consumed_end

SERVER = "http://127.0.0.1:8300"
SR = 16000
TTS_SR = 24000
BLOCK = 512
SILENCE_END_MS = int(os.environ.get("ME3_SILENCE_MS", "600"))
BARGE_MS = int(os.environ.get("ME3_BARGE_MS", "250"))       # legacy (superseded by duplex.py params)
ECHO_THRESH = float(os.environ.get("ME3_ECHO_THRESH", "0.40"))
PROBE_MS = int(os.environ.get("ME3_PROBE_MS", "600"))       # pause-probe window
VOICE_FLOOR = float(os.environ.get("ME3_VOICE_FLOOR", "0.012"))  # rms that counts as live speech
FILLERS = {"um", "uh", "mhm", "hmm", "uh-huh", "huh", "mm", "yeah", "okay", "ok"}
# Whisper's classic hallucinations on short quiet bursts (no wake word needed)
WHISPER_GHOSTS = {
    "thank you", "thank you very much", "thanks", "thanks for watching",
    "thanks for watching my video", "see you next time",
    "hello", "hi", "hey", "you", "so", "and", "the", "a", "i", "it",
    "mmhmm", "amen", "and you", "you know", "i know", "my pleasure",
}
MAX_UTT_S = 15
HISTORY_MAX_TURNS = 12
TALKER_MODEL = os.environ.get("ME3_TALKER_MODEL", "z-ai/glm-5.3")
LOG_PATH = os.path.expanduser("~/me3/chat_log.jsonl")

SYSTEM_PROMPT = (
    "You are a voice assistant in a live spoken conversation. Reply in ONE or TWO short "
    "sentences, plain text, no markdown, no lists, no emoji. Be conversational and brief. "
    "You do NOT do tasks yourself — workers handle everything in the background, and "
    "several can run at once. "
    "You have tools: get_time (local), get_time_in (exact time in other cities), web_search, "
    "and smart-home tools: spotify_play/pause/next/prev/volume (music + media control), "
    "lights_control (on/off, brightness, color), thermostat_set, set_alarm, add_reminder, "
    "list_items, contact_message (calls/messages are logged in the sandbox only). Handle "
    "music, lights, thermostat, timers, alarms, reminders and messages YOURSELF with these "
    "tools — never delegate them. get_time reports the machine's LOCAL time only; "
    "get_time_in is exact for other cities (never web_search or guess times). Use "
    "web_search for facts you don't know. NEVER reply with a bare offer like 'would you "
    "like me to...' — always act. Never mention tools. "
    "Song lyrics, poems, jokes, quotes and creative text: answer from your OWN "
    "knowledge — never web_search them and never DELEGATE them (search finds no lyrics; "
    "if asked for 'the next line', recite it directly). "
    "EVERYTHING else that requires any action — building anything, timers, reminders, "
    "notes, lookups, research, files, calculations — delegate with a reply that starts "
    "with the tag 'DELEGATE:' followed by a one-line restatement of the request, "
    "e.g. 'DELEGATE: a one-page website about Tokyo trips'. The tag is machine-read "
    "(never spoken aloud), so always start delegation replies with it exactly. "
    "If the request is incomplete or ambiguous — no topic given, key details missing, "
    "or you are unsure what they want — do NOT use the DELEGATE tag; instead ask ONE "
    "short, natural clarifying question (e.g. 'Sure — a website about what, exactly?'). "
    "Remember details the user tells you within this conversation."
)

PLAYING = False       # true while agent audio is on the speaker
DISCARD_UNTIL = 0.0   # monotonic time: drop mic input until then (post-playback echo tail)
q = queue.Queue()

PRECACHE_FALLBACK = {
    "ack_0": "On it — I'm handing that to my agent.",
    "ack_1": "My agent is taking that one now.",
    "ack_2": "I've passed that to my agent — give it a moment.",
    "ack_3": "Handing that off to my agent; I'll tell you when it's done.",
    "ack_4": "Okay, my agent is starting that for you.",
    "clarify": "I can start on that, but it's a bit vague. Could you tell me a little more about what you'd like?",
    "apology_2": "Sorry, I'm already working on 2 things in the background. Let me finish those before I take on more. I'll let you know as each one lands.",
    "apology_busy": "Sorry, my hands are full in the background right now. I'll come back to this one as soon as a slot frees up.",
    "go_ahead": "Go ahead — I'm listening.",
}


def speak_cached(key):
    """Instant playback of a pre-rendered phrase (falls back to live synthesis)."""
    try:
        r = HTTP.get(f"{SERVER}/speak_cache/{key}")
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    wav, _, _ = speak_http(PRECACHE_FALLBACK.get(key, "One moment."))
    return wav


class Barged(Exception):
    pass


DEFLECT_RE = re.compile(
    r"(i only know|would you like me|shall i|i can'?t|i cannot|i don'?t know|"
    r"i'?m not sure|unable to|specify a location|i apologize for the confusion|"
    r"check a (reliable|trusted) source|exact local time)", re.I)


def log_turn(entry):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def mic_callback(indata, frames, t, status):
    if time.monotonic() < DISCARD_UNTIL:
        return                          # post-playback echo-tail guard (300 ms)
    q.put(indata[:, 0].copy())          # full duplex: never drop mic input


def resample(pcm, sr_in, sr_out):
    if sr_in == sr_out:
        return pcm
    dur = len(pcm) / sr_in
    n_out = int(dur * sr_out)
    return np.interp(np.linspace(0, dur, n_out, endpoint=False),
                     np.arange(len(pcm)) / sr_in, pcm).astype(np.float32)


def xcorr_echo_score(candidate, reference):
    """Max normalized cross-correlation of candidate vs reference over a delay search."""
    n = min(len(candidate), len(reference))
    if n < SR // 10:
        return 0.0
    c = candidate[:n] - candidate[:n].mean()
    best = 0.0
    limit = min(len(reference) - n, int(SR * 0.25))
    for d in range(0, max(limit, 0) + 1, 80):
        r = reference[d:d + n]
        r = r - r.mean()
        denom = (np.linalg.norm(c) * np.linalg.norm(r)) + 1e-9
        best = max(best, float(np.dot(c, r) / denom))
    return best


def _token_set(s):
    return set(re.findall(r"[a-z0-9']+", s.lower()))


class BargeDetector:
    """VAD during playback. On speech onset: fast correlation check vs the played
    audio; if it looks like echo, ignore. Otherwise report a PAUSE candidate —
    play_with_barge then runs the pause-probe (does sound continue once the
    agent's own audio stops?) to separate echo tails from a real user."""

    def __init__(self, spoken_text=""):
        self.vad = VADIterator(load_silero_vad(), sampling_rate=SR,
                               threshold=0.60, min_silence_duration_ms=120)
        self.in_speech = False
        self.burst = []
        self.spoken_tokens = _token_set(spoken_text)

    def feed(self, chunk, played_ref_16k):
        """chunk: 16k mono float32. Returns ('pause', burst, corr) IMMEDIATELY on
        speech onset during playback — the pause-probe, not the detector, decides
        whether it was echo or a live user."""
        event = self.vad(chunk, return_seconds=False)
        if not self.in_speech and event and "start" in event:
            self.in_speech = True
            self.burst = [chunk]
            return ("pause", np.concatenate(self.burst), 0.0)
        if self.in_speech:
            self.burst.append(chunk)
            if event and "end" in event:
                self.in_speech = False
                self.burst = []
        return None


def play_with_barge(wav_bytes, watch=True):
    """Stream reply audio to the speaker in blocks; watch mic for real barge-ins.
    Uses the bench-validated DuplexGate (duplex.py) — pause-and-resume with the
    swept parameters. Returns (played_ref_16k, interrupted_bool, barge_pcm_or_None)."""
    global PLAYING
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32767.0
    ref16 = resample(pcm, sr, SR)
    # drain stale mic accumulated while the LLM was generating (nobody was reading q)
    while not q.empty():
        q.get()
    params = DuplexParams.from_env()
    gate = DuplexGate(params)
    # calibrate the ambient noise floor from a quiet 250ms sample before playback
    ambient = []
    calib_until = time.monotonic() + 0.25
    while time.monotonic() < calib_until:
        try:
            ambient.append(q.get(timeout=0.1))
        except queue.Empty:
            continue
    gate.calibrate(ambient)
    stream = sd.OutputStream(samplerate=TTS_SR, channels=1, dtype="float32")
    stream.start()
    PLAYING = True
    interrupted = False
    barge_pcm = None
    cooldown_until = 0.0

    def mic_iter():
        """Yield mic chunks during the probe; silence when the queue is empty."""
        while True:
            try:
                yield q.get(timeout=0.05)
            except queue.Empty:
                yield np.zeros(BLOCK, np.float32)

    try:
        pos = 0
        chunk_out = int(TTS_SR * 0.05)   # 50 ms blocks
        while pos < len(pcm):
            stream.write(pcm[pos:pos + chunk_out])
            pos += chunk_out
            ref_sent = min(len(ref16), int(pos * SR / TTS_SR))
            if not watch:
                while not q.empty():
                    q.get()   # ack playback: mic flows to nowhere
                continue
            now = time.monotonic()
            if now < cooldown_until:
                while not q.empty():
                    q.get()
                if now >= cooldown_until - 0.06:
                    gate.reset()
                continue
            act = None
            while not q.empty():
                mic = q.get()
                a = gate.feed(mic, ref16[:ref_sent])
                if a:
                    act = a
                    break
            if act and act[0] == "echo":
                gate.reset()
                continue
            if act and act[0] == "pause":
                # PAUSE-PROBE (bench-validated): agent audio stops; echo decays,
                # a live user persists in the persistence window.
                barge, tail_rms, collected = gate.probe(mic_iter(), act[1])
                if barge:
                    # backchannel gate: brief same-voice sounds ("yeah", "mhm")
                    # must NOT stop the agent — FDB backchannel expectation
                    speech_ms = sum(len(c) / SR * 1000 for c in collected
                                    if float(np.sqrt(np.mean(c ** 2))) > gate.floor)
                    if speech_ms < 450:
                        print(f"  [backchannel] {speech_ms:.0f}ms voiced — continuing",
                              flush=True)
                        barge = False
                    elif speaker_emb is not None:
                        cand = np.concatenate([c for c in collected if c is not None])
                        sim = speakerid.similarity(cand, speaker_emb)
                        if sim < speakerid.near_threshold():
                            print(f"  [voice] unknown voice interrupting (sim {sim:.2f} < "
                                  f"{speakerid.near_threshold()}) — resuming", flush=True)
                            barge = False
                if barge:
                    interrupted = True
                    barge_pcm = np.concatenate([c for c in collected if c is not None])
                    print(f"  [barge-in] user confirmed (tail rms {tail_rms:.4f} > "
                          f"floor {gate.floor:.4f})", flush=True)
                    break
                cooldown_until = time.monotonic() + params.resume_cooldown_ms / 1000
                gate.note_echo(tail_rms)   # learn our echo level for adaptive suppression
                print(f"  [echo] pause-probe: faded (tail rms {tail_rms:.4f}) — resuming",
                      flush=True)
    finally:
        global DISCARD_UNTIL
        PLAYING = False
        DISCARD_UNTIL = time.monotonic() + 0.3
        stream.stop()
        stream.close()
        # drain residual mic so echo tails don't leak into the next turn
        while not q.empty():
            q.get()
    return ref16, interrupted, barge_pcm


QUICK_TOOLS = [
    {"type": "function", "function": {"name": "get_time", "description": "Get the current date and local time (instant, not a task).",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_time_in", "description": "Get the exact current time in a specific city or country. ALWAYS use this for times in other places — never web_search and never guess.",
                                      "parameters": {"type": "object", "properties": {
                                          "city": {"type": "string", "description": "City or country name, e.g. Mumbai, Tokyo, London"}}},
                                      "required": ["city"]}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the web for facts, news, prices, or anything you are unsure about. Returns titles and snippets.",
                                      "parameters": {"type": "object", "properties": {
                                          "query": {"type": "string", "description": "Search query"}}},
                                      "required": ["query"]}},
    {"type": "function", "function": {"name": "set_timer", "description": "Start a countdown timer that announces itself when it expires.",
                                      "parameters": {"type": "object", "properties": {
                                          "seconds": {"type": "integer"},
                                          "label": {"type": "string"}}, "required": ["seconds"]}}},
    {"type": "function", "function": {"name": "spotify_play", "description": "Play music on Spotify. Pass a song, artist, genre or mood as the query.",
                                      "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "spotify_pause", "description": "Pause the music.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "spotify_next", "description": "Skip to the next track.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "spotify_prev", "description": "Go back to the previous track.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "spotify_volume", "description": "Set the music volume (0-100).",
                                      "parameters": {"type": "object", "properties": {"percent": {"type": "integer"}}, "required": ["percent"]}}},
    {"type": "function", "function": {"name": "lights_control", "description": "Control the smart lights: turn on/off, set brightness percent or color.",
                                      "parameters": {"type": "object", "properties": {
                                          "light": {"type": "string", "description": "Light name, e.g. living room"},
                                          "on": {"type": "string", "description": "true to turn on, false to turn off"},
                                          "brightness_percent": {"type": "integer"},
                                          "color": {"type": "string", "description": "e.g. warm white, blue, red"}}}}},
    {"type": "function", "function": {"name": "thermostat_set", "description": "Set the thermostat temperature in degrees.",
                                      "parameters": {"type": "object", "properties": {"degrees": {"type": "integer"}}, "required": ["degrees"]}}},
    {"type": "function", "function": {"name": "set_alarm", "description": "Set an alarm clock for a time like 7:00am.",
                                      "parameters": {"type": "object", "properties": {"time": {"type": "string"}, "label": {"type": "string"}}, "required": ["time"]}}},
    {"type": "function", "function": {"name": "add_reminder", "description": "Save a reminder for the user.",
                                      "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "when": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "list_items", "description": "Read back reminders or alarms.",
                                      "parameters": {"type": "object", "properties": {"kind": {"type": "string", "enum": ["reminders", "alarms"]}}, "required": ["kind"]}}},
    {"type": "function", "function": {"name": "contact_message", "description": "Log a call or message to a contact (sandbox only, no real calls).",
                                      "parameters": {"type": "object", "properties": {"kind": {"type": "string", "enum": ["call", "message"]}, "who": {"type": "string"}, "text": {"type": "string"}}, "required": ["who"]}}},
]


def execute_tool(name, args):
    import datetime
    if name == "get_time":
        now = datetime.datetime.now()
        local_tz = time.tzname
        return (f"The machine's local time is {now.strftime('%H:%M on %A, %B %d, %Y')} "
                f"({local_tz}). You cannot know the time in other time zones.")
    if name == "get_time_in":
        from datetime import datetime
        from zoneinfo import ZoneInfo
        CITY_TZ = {
            "india": "Asia/Kolkata", "mumbai": "Asia/Kolkata", "delhi": "Asia/Kolkata",
            "jakarta": "Asia/Jakarta", "indonesia": "Asia/Jakarta",
            "tokyo": "Asia/Tokyo", "japan": "Asia/Tokyo",
            "manila": "Asia/Manila", "philippines": "Asia/Manila",
            "london": "Europe/London", "new york": "America/New_York",
            "los angeles": "America/Los_Angeles", "dubai": "Asia/Dubai",
            "singapore": "Asia/Singapore", "china": "Asia/Shanghai",
            "beijing": "Asia/Shanghai", "seoul": "Asia/Seoul", "korea": "Asia/Seoul",
            "paris": "Europe/Paris", "berlin": "Europe/Berlin",
            "sydney": "Australia/Sydney", "sao paulo": "America/Sao_Paulo",
        }
        city = (args.get("city") or "").lower().strip()
        tzname = CITY_TZ.get(city)
        if not tzname:
            return (f"No built-in timezone for {city!r}. Known cities: "
                    + ", ".join(sorted(CITY_TZ)) + ".")
        now = datetime.now(ZoneInfo(tzname))
        return (f"The current time in {city.title()} is "
                f"{now.strftime('%H:%M on %A, %B %d, %Y')} (zone: {tzname}).")
    if name == "web_search":
        try:
            from ddgs import DDGS
            query = args.get("query", "")
            results = DDGS().text(query, max_results=5, region="wt-wt")
            if not results:
                return "No results found."
            lines = [f"- {r.get('title','')}: {r.get('body','')[:200]}" for r in results]
            return "Web results:\n" + "\n".join(lines)
        except Exception as e:
            return f"Search failed: {e}"
    if name == "set_timer":
        secs = int(args.get("seconds", 0))
        label = args.get("label", "your timer")

        def _fire():
            time.sleep(secs)
            try:
                wav, _, _ = speak_http(f"Time's up — that's your {label}.")
                play_with_barge(wav, watch=False)
            except Exception:
                pass
        threading.Thread(target=_fire, daemon=True).start()
        nice = f"{secs // 60} minute{'s' if secs // 60 != 1 else ''}" if secs >= 60 else f"{secs} seconds"
        return f"Timer set for {nice} — {label}."
    if name in ("spotify_play", "spotify_pause", "spotify_next", "spotify_prev",
                "spotify_volume", "lights_control", "thermostat_set", "set_alarm",
                "add_reminder", "list_items", "contact_message"):
        return home.do_tool(name, args)
    return f"Unknown tool {name}"


def talker_stream(messages, key, on_first_sentence, tools=None):
    """Streaming talker with optional tool support. If the model emits tool calls,
    execute them and do a second streaming pass for the spoken reply.
    Returns (reply, first_token_dt)."""
    t0 = time.perf_counter()
    first_dt = None
    chunks = []
    calls = {}   # index -> {"id","name","args"}

    def _one_pass(msgs, use_tools):
        nonlocal first_dt, chunks, calls
        body = {"model": TALKER_MODEL, "messages": msgs, "max_tokens": 200, "stream": True}
        if use_tools:
            body["tools"] = tools
        with HTTP.stream(
                "POST", "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"}, json=body) as resp:
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    d = json.loads(data)["choices"][0]["delta"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if d.get("tool_calls"):
                    for tc in d["tool_calls"]:
                        i = tc.get("index", 0)
                        slot = calls.setdefault(i, {"id": "", "name": "", "args": ""})
                        slot["id"] = slot["id"] or tc.get("id") or ""
                        fn = tc.get("function") or {}
                        slot["name"] = slot["name"] or fn.get("name") or ""
                        slot["args"] += fn.get("arguments") or ""
                if d.get("content"):
                    if first_dt is None:
                        first_dt = time.perf_counter() - t0
                    chunks.append(d["content"])
                    on_first_sentence("".join(chunks))

    _one_pass(messages, tools is not None)
    if calls and tools:
        messages = messages + [{"role": "assistant",
                                "content": None,
                                "tool_calls": [{"id": s["id"], "type": "function",
                                                "function": {"name": s["name"],
                                                             "arguments": s["args"]}}
                                               for s in calls.values()]}]
        for s in calls.values():
            try:
                args = json.loads(s["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            out = execute_tool(s["name"], args)
            print(f"  [tool] {s['name']}({args}) -> {out[:60]!r}", flush=True)
            messages.append({"role": "tool", "tool_call_id": s["id"], "content": out})
        chunks = []
        _one_pass(messages, False)
    return "".join(chunks).strip(), first_dt


def handle_utterance(pcm, history, key, allow_extend=True):
    t0 = time.perf_counter()
    text, asr_dt = transcribe_http(wav_to_bytes(pcm, SR))
    print(f"  [asr {asr_dt:.3f}s] you: {text!r}", flush=True)
    if not text or len(text.strip()) < 2:
        return None
    if _token_set(text) <= FILLERS:
        print("  [filler] dropped", flush=True)
        return None

    # Whisper hallucination filter: short bursts of noise/silence produce
    # stock phrases ("Thank you.", "Thanks for watching!") — drop them before
    # they count as speech or pollute the wake-word gate.
    norm = re.sub(r"[^\w\s]", "", text.lower()).strip()
    if norm in WHISPER_GHOSTS and not wake_match(text)[0]:
        print(f"  [ghost] Whisper hallucination dropped: {text!r}", flush=True)
        log_turn({"t": round(time.time(), 3), "event": "ghost_ignored", "user": text})
        return None

    # --- wake word + speaker identity gate ---
    global speaker_emb, awake, pending_enroll
    woke, woke_end = wake_match(text)
    if not awake:
        if not woke:
            print(f"  [idle] not addressed (no '{WAKE_WORD}') — ignored", flush=True)
            log_turn({"t": round(time.time(), 3), "event": "idle_ignored", "user": text})
            return None
        if speaker_emb is None:
            emb = speakerid.embed(pcm)
            # only trust enrollment from a substantial clip: a weak profile
            # permanently locks out the real owner
            if emb is None or speakerid.voiced_seconds(pcm) < 1.0:
                pending_enroll = True
                print("  [voice] wake heard — will enroll from your next utterance", flush=True)
            else:
                speaker_emb = emb
                speakerid.save_profile(emb)
                pending_enroll = False
                print("  [voice] new voice enrolled and saved — I'll recognize you from now on", flush=True)
                log_turn({"t": round(time.time(), 3), "event": "enrolled"})
                wav, _, _ = speak_http("Got it — I'll recognize your voice from now on.")
                play_with_barge(wav, watch=False)
        else:
            if pending_enroll:
                emb = speakerid.embed(pcm)
                if emb is not None and speakerid.voiced_seconds(pcm) >= 1.0:
                    speaker_emb = emb
                    speakerid.save_profile(emb)
                    pending_enroll = False
                    print("  [voice] enrollment completed from follow-up utterance", flush=True)
                    log_turn({"t": round(time.time(), 3), "event": "enrolled"})
                else:
                    print("  [voice] still too short to enroll — will keep trying", flush=True)
            else:
                sim = speakerid.similarity(pcm, speaker_emb)
                if sim < speakerid.threshold():
                    if sim >= speakerid.near_threshold():
                        # near-miss: almost certainly the owner (mic moved,
                        # different room) — accept and refresh the profile
                        speaker_emb = speakerid.embed(pcm)
                        if speaker_emb is not None and speakerid.voiced_seconds(pcm) >= 1.0:
                            speakerid.save_profile(speaker_emb)
                            print(f"  [voice] near-miss (sim {sim:.2f}) — accepted, "
                                  "profile refreshed", flush=True)
                            log_turn({"t": round(time.time(), 3),
                                      "event": "voice_refreshed", "sim": round(sim, 3)})
                        else:
                            print(f"  [voice] near-miss (sim {sim:.2f}) — accepted "
                                  "(too short to refresh)", flush=True)
                    else:
                        stranger_rejects.append(time.time())
                        recent = [t for t in stranger_rejects if time.time() - t < 120]
                        stranger_rejects[:] = recent
                        print(f"  [voice] stranger at idle (sim {sim:.2f}) — ignoring "
                              f"({len(recent)} recent)", flush=True)
                        log_turn({"t": round(time.time(), 3), "event": "stranger_ignored",
                                  "user": text, "sim": round(sim, 3)})
                        if len(recent) >= 3:
                            stranger_rejects.clear()
                            wav, _, _ = speak_http(
                                "I don't recognize this voice. If you are my owner, say: "
                                "Watson, reset my voice.")
                            play_with_barge(wav, watch=False)
                        return None
        awake = True
        text = text[woke_end:].strip()
        if not text:
            greeting = "Yes? I'm listening."
            wav, _, _ = speak_http(greeting)
            play_with_barge(wav, watch=False)
            history.append({"role": "assistant", "content": greeting})
            return None
    elif woke:
        text = text[woke_end:].strip() or text
    # owner escape hatch: re-enroll the voice from scratch
    if re.search(r"reset (my|your|the) voice", text.lower()):
        speaker_emb = None
        pending_enroll = True
        if os.path.exists(speakerid.PROFILE_PATH):
            os.remove(speakerid.PROFILE_PATH)
        print("  [voice] profile cleared — re-enrolling from your next utterance", flush=True)
        log_turn({"t": round(time.time(), 3), "event": "voice_reset"})
        wav, _, _ = speak_http("Voice reset. Tell me something and I'll learn your voice.")
        play_with_barge(wav, watch=False)
        return None

    if any(w in text.lower() for w in ("goodbye", "good bye", "that's all")):
        awake = False   # back to idle: wake word required for the next conversation
        return "quit"

    # trail-off detection: user was still formulating when endpointing fired.
    # Don't speak — silently extend the listening window and merge what follows.
    if len(text.split()) <= 3 and re.search(r"\b(and|but|or|to|for|with|so|then|of|about|flights?)\s*$",
                                            text.strip(), flags=re.I):
        if allow_extend:
            print("  [trail-off] extending listening window...", flush=True)
            return "extend"
        prompt_continue = "Go ahead — I'm listening."
        print("  [trail-off] prompting to continue", flush=True)
        wav = speak_cached("go_ahead")
        play_with_barge(wav, watch=False)
        return None

    # correction: "I meant X" / "actually..." cancels running tasks, then the
    # normal flow re-delegates the corrected request (steer-by-cancel-resubmit)
    if re.search(r"\b(i meant|actually|no wait|scratch that|forget that)\b", text.lower()) \
            and taskmgr._running:
        taskmgr.reset_all()
        print("  [tasks] cancelled running tasks — correction received", flush=True)
        log_turn({"t": round(time.time(), 3), "event": "tasks_cancelled", "user": text})

    # task status is answered locally, not by the LLM (cheap + always accurate)
    if re.search(r"\b(status|how('| i)s (it |the (task|build|website|game))|any progress)", text.lower()):
        running = taskmgr.list_running()
        if running:
            parts = []
            for tid, s in running.items():
                thing = taskmgr._thing(taskmgr._running[tid]["prompt"]) if tid in taskmgr._running else "it"
                secs = int(taskmgr._running[tid]["t0"] and time.time() - taskmgr._running[tid]["t0"])
                dur = f", about {secs//60} min in" if secs >= 90 else ""
                parts.append(f"{thing}{dur}")
            if len(parts) == 1:
                reply_text = f"Still working on {parts[0]} — I'll tell you the moment it's done."
            else:
                reply_text = ("Still working on " + " and ".join(parts) +
                              " — I'll announce each one as it lands.")
        else:
            taskmgr.check_finished()
            reply_text = "Nothing's running right now — everything's finished. What's next?"
        wav, _, _ = speak_http(reply_text)
        play_only = play_with_barge(wav)
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": reply_text})
        print(f"  [status] {reply_text!r}", flush=True)
        log_turn({"t": round(time.time(), 3), "user": text, "reply": reply_text,
                  "event": "status_query"})
        return None

    history.append({"role": "user", "content": text})
    msgs = ([{"role": "system", "content": SYSTEM_PROMPT}] +
            history[-2 * HISTORY_MAX_TURNS:])

    first_audio = {"done": False, "dt": 0.0, "prefix": ""}

    def speak_and_watch(audio_text):
        """Synthesize + play with barge watching; raise Barged if interrupted."""
        wav, _, _ = speak_http(audio_text)
        _, interrupted, barge_pcm = play_with_barge(wav)
        if interrupted:
            print("  [barge-in] stopped talking — user interrupted", flush=True)
            log_turn({"t": round(time.time(), 3), "event": "barge_in",
                      "user": text, "partial_reply": audio_text})
            raise Barged(barge_pcm)

    def maybe_speak(partial, suppress_deflect=False):
        if first_audio["done"]:
            return
        if not suppress_deflect and DEFLECT_RE.search(partial):
            # deflection-speak detected: hold back, the caller will search & re-ask
            first_audio["deflected"] = True
            return
        if any(c in partial for c in ".!?"):
            first_audio["done"] = True
            first_audio["dt"] = time.perf_counter() - t0
            first_audio["prefix"] = partial
            if partial.lower().startswith("delegate:"):
                handle_delegation(partial)
                return
            speak_and_watch(partial)

    def handle_delegation(tagged):
        """Parse the DELEGATE tag, submit-or-apologize, speak a natural ack.
        The tag itself is never spoken."""
        restatement = re.sub(r"^delegate:\s*", "", tagged, flags=re.I).strip()
        first_audio["suppress_rest"] = True
        if len(restatement.split()) < 4 or restatement.endswith("?"):
            clarify = ("I can start on that, but it's a bit vague. "
                       "Could you tell me a little more about what you'd like?")
            print("  [route] vague restatement — asking instead of delegating", flush=True)
            wav = speak_cached("clarify")
            play_with_barge(wav, watch=False)
            history.append({"role": "assistant", "content": clarify})
            return
        n_running = len(taskmgr.list_running())
        if n_running >= taskmgr.MAX_CONCURRENT:
            apology = (f"Sorry, I'm already working on {n_running} things in the "
                       f"background. Let me finish those before I take on more. "
                       f"I'll let you know as each one lands.")
            print("  [tasks] at capacity — apologizing", flush=True)
            wav = speak_cached(f"apology_{n_running}") if n_running == 2 else speak_http(apology)[0]
            play_with_barge(wav, watch=False)
            history.append({"role": "assistant", "content": apology})
            return
        tid = taskmgr.submit_task(restatement)
        if not tid:
            apology = ("Sorry, my hands are full in the background right now. "
                       "I'll come back to this one as soon as a slot frees up.")
            wav = speak_cached("apology_busy")
            play_with_barge(wav, watch=False)
            history.append({"role": "assistant", "content": apology})
            return
        ack_idx = random.randrange(5)
        ack = PRECACHE_FALLBACK[f"ack_{ack_idx}"]
        print(f"  [task] submitted {tid}: {restatement[:80]!r}  ack: {ack!r}", flush=True)
        log_turn({"t": round(time.time(), 3), "event": "task_submitted",
                  "id": tid, "prompt": restatement})
        first_audio["delegated"] = tid
        first_audio["restatement"] = restatement
        wav = speak_cached(f"ack_{ack_idx}")
        play_with_barge(wav, watch=False)   # ack is short; protect the submission
        history.append({"role": "assistant", "content": ack})

    try:
        reply, first_tok = talker_stream(msgs, key, maybe_speak, tools=QUICK_TOOLS)
    except Barged as b:
        return ("interrupted", b.args[0])   # interrupting speech becomes next utterance

    first_tok = first_tok or 0.0
    if not reply:
        return None
    if first_audio.get("deflected"):
        # deterministic handoff: the model deflected, so search the web and re-ask
        print("  [search] deflection detected — searching the web", flush=True)
        log_turn({"t": round(time.time(), 3), "event": "auto_search", "user": text})
        sres = execute_tool("web_search", {"query": text})
        msgs2 = msgs + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content":
                f"Web search results for the user's question:\n{sres}\n\n"
                f"Answer the user's original question ({text.strip()!r}) in one or two "
                f"short sentences from these results. If they truly don't answer it, "
                f"say so briefly and reply 'DELEGATE:' plus a restatement."},
        ]
        try:
            reply, first_tok = talker_stream(msgs2, key,
                                             lambda p: maybe_speak(p, suppress_deflect=True),
                                             tools=QUICK_TOOLS)
        except Barged as b:
            return ("interrupted", b.args[0])
        first_tok = first_tok or 0.0
        if not reply:
            return None
    # fallback: delegation marker arrived but maybe_speak never fired (e.g. the
    # whole reply came without a sentence break)
    if (reply.lower().startswith("delegate:")
            and not first_audio.get("delegated")):
        handle_delegation(reply)
        return None

    try:
        if not first_audio["done"]:
            first_audio["dt"] = time.perf_counter() - t0
            speak_and_watch(reply)
        elif first_audio.get("suppress_rest"):
            pass   # apology already spoken; nothing more to say
        else:
            rest = reply[len(first_audio.get("prefix", "")):].strip()
            if rest:
                speak_and_watch(rest)
    except Barged as b:
        return ("interrupted", b.args[0])

    if first_audio.get("delegated"):
        history.append({"role": "assistant",
                        "content": f"(delegated: {first_audio.get('restatement', '')})"})
    else:
        history.append({"role": "assistant", "content": reply})

    entry = {"t": round(time.time(), 3), "user": text, "reply": reply,
             "asr_s": round(asr_dt, 3), "first_token_s": round(first_tok, 3),
             "first_audio_s": round(first_audio["dt"], 3),
             "total_s": round(time.perf_counter() - t0, 3)}
    print(f"  [llm {first_tok:.3f}s | audio {first_audio['dt']:.3f}s] agent: {reply!r}", flush=True)
    log_turn(entry)
    return None


def main():
    key = load_openrouter_key()
    global speaker_emb
    speaker_emb = speakerid.load_profile()
    if speaker_emb is not None:
        print("  [voice] saved voice profile loaded — enrollment already done", flush=True)
    vad = VADIterator(load_silero_vad(), sampling_rate=SR, threshold=0.5,
                      min_silence_duration_ms=SILENCE_END_MS)
    history = []
    in_speech = False
    utt = []
    print(f"[chat] full-duplex always listening (endpointing {SILENCE_END_MS} ms; "
          f"barge gate = bench-tuned DuplexGate: probe {DuplexParams.from_env().probe_ms} ms, "
          f"persist {DuplexParams.from_env().persistence_ms} ms, corr {DuplexParams.from_env().corr_echo}). "
          f"Say 'goodbye' to quit.", flush=True)

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=BLOCK, callback=mic_callback):
        interrupt_pcm = None
        while True:
            if interrupt_pcm is not None:
                pcm, interrupt_pcm = interrupt_pcm, None
            else:
                try:
                    chunk = q.get(timeout=0.5)
                except queue.Empty:
                    # idle gap: also a natural pause for announcements
                    for tid in taskmgr.check_finished():
                        line = taskmgr.announcement(tid)
                        if line:
                            print(f"  [announce] {line}", flush=True)
                            wav, _, _ = speak_http(line)
                            _, _, bpcm = play_with_barge(wav)
                            opened = taskmgr.open_deliverable(tid)
                            if opened:
                                print(f"  [open] {opened}", flush=True)
                            log_turn({"t": round(time.time(), 3), "event": "announce",
                                      "id": tid, "line": line, "opened": opened})
                    continue
                event = vad(chunk, return_seconds=False)
                if in_speech:
                    utt.append(chunk)
                    if event and "end" in event:
                        pcm = np.concatenate(utt)
                        utt, in_speech = [], False
                    elif len(utt) * BLOCK / SR > MAX_UTT_S:
                        pcm = np.concatenate(utt)
                        utt, in_speech = [], False
                    else:
                        continue
                elif event and "start" in event:
                    in_speech = True
                    utt = [chunk]
                    continue
                else:
                    continue
            if len(pcm) < SR * 0.35:
                continue
            try:
                res = handle_utterance(pcm, history, key)
            except Exception as e:
                print(f"[chat] turn failed: {e}", flush=True)
                continue
            if res == "extend":
                # grace window: keep listening; merge the continuation into the
                # same utterance instead of replying to a half-sentence
                grace = []
                collecting = False
                deadline = time.time() + 3.0
                got_speech = False
                while time.time() < deadline:
                    try:
                        chunk = q.get(timeout=0.3)
                    except queue.Empty:
                        if collecting:
                            break   # they stopped again -> enough
                        continue
                    ev = vad(chunk, return_seconds=False)
                    if not collecting and ev and "start" in ev:
                        collecting = True
                    if collecting:
                        grace.append(chunk)
                        if ev and "end" in ev:
                            break
                if grace:
                    merged = np.concatenate([pcm] + grace)
                    try:
                        res = handle_utterance(merged, history, key, allow_extend=False)
                    except Exception as e:
                        print(f"[chat] turn failed: {e}", flush=True)
                        continue
                else:
                    try:
                        res = handle_utterance(pcm, history, key, allow_extend=False)
                    except Exception as e:
                        print(f"[chat] turn failed: {e}", flush=True)
                        continue
            if res == "quit":
                print("[chat] goodbye!", flush=True)
                break
            if isinstance(res, tuple) and res[0] == "interrupted":
                interrupt_pcm = res[1]   # user's interrupting speech: handle next
                continue
            # natural pause: announce any background tasks that just finished
            for tid in taskmgr.check_finished():
                line = taskmgr.announcement(tid)
                if line:
                    print(f"  [announce] {line}", flush=True)
                    wav, _, _ = speak_http(line)
                    _, _, bpcm = play_with_barge(wav)
                    opened = taskmgr.open_deliverable(tid)
                    if opened:
                        print(f"  [open] {opened}", flush=True)
                    log_turn({"t": round(time.time(), 3), "event": "announce",
                              "id": tid, "line": line, "opened": opened})


if __name__ == "__main__":
    main()