"""Watson's brain: system prompt, tool schemas and the tool dispatcher.

Moved out of chat.py (deprecated) on 2026-11-05. pipecat_assistant.py
imports this module; it has no audio or loop code, so it loads on the Pi
without sounddevice or torch.
"""
import re
import time

import home  # noqa: E402  (Spotify + virtual smart home)


# v4 (Nov 6, owner-approved): short version of the owner's v3 prompt, same
# rules in the same order (~230 tokens vs ~540). Measured: prompt size moves
# glm-5.3-flash TTFT by only ~0.2-0.5 s; it was trimmed for context hygiene.
_PROMPT_OWNER = """You are Watson, a voice assistant in a live spoken conversation. Reply in one or two short spoken sentences, plain text, no markdown or emoji; give more detail only when asked.
When a request is clear, act with the right tool without offering first. If a needed detail is missing, ask one short question.
Handle music, lights, thermostat, timers, alarms, reminders and messages yourself with their tools.
Say an action is done only after the tool confirms it; say clearly if it is pending or failed.
For news, facts you are unsure of, and the time in other cities, use web_search; never guess. If the search doesn't settle it, say so briefly.
Never mention tools or internal details.
After an interruption, follow the user's latest request; an action already submitted still stands unless they cancel it."""

# Runtime protocol: pipecat_assistant parses a reply that starts with
# "DELEGATE:" and hands the job to the background agent (task MCP server).
_PROMPT_DELEGATE = """
Big jobs (sites, apps, research, files) go to the background agent: reply DELEGATE: <one-line request>. Never speak the tag. If the job has no topic yet ("build me a website"), ask what it should be about first."""

SYSTEM_PROMPT = _PROMPT_OWNER + _PROMPT_DELEGATE


QUICK_TOOLS = [
    {"type": "function", "function": {"name": "get_time", "description": "Get today's date and the LOCAL time here. Only for 'what time or date is it here'. Never for news, results, events or other cities.",
                                      "parameters": {"type": "object", "properties": {}}}},
    # get_time_in removed from the talker's tools (owner, Nov 6): city times
    # go to web_search. Its handler in execute_tool stays for old callers.
    {"type": "function", "function": {"name": "web_search", "description": "Search the web for facts, news, sports and election results, prices, the current time in another city, or anything you are unsure about. Returns titles and snippets.",
                                      "parameters": {"type": "object", "properties": {
                                          "query": {"type": "string", "description": "Search query"}}},
                                      "required": ["query"]}},
    {"type": "function", "function": {"name": "set_timer", "description": "Start a countdown timer that announces itself when it expires.",
                                      "parameters": {"type": "object", "properties": {
                                          "seconds": {"type": "integer"},
                                          "label": {"type": "string"}}, "required": ["seconds"]}}},
    # one schema instead of five spotify_* (tool schemas go out on every call)
    {"type": "function", "function": {"name": "music", "description": "Control music: play a song, artist, genre or mood; pause; next; previous; or set volume.",
                                      "parameters": {"type": "object", "properties": {
                                          "action": {"type": "string", "enum": ["play", "pause", "next", "previous", "volume"]},
                                          "query": {"type": "string", "description": "what to play (action=play)"},
                                          "percent": {"type": "integer", "description": "0-100 (action=volume)"}},
                                      "required": ["action"]}}},
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


_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
# "~/anything" or an absolute path with 2+ segments ("24/7", "and/or" survive)
_PATH_RE = re.compile(r"~/[\w.\-/]*|(?<!\w)/[\w.\-]+(?:/[\w.\-]+)+/?")
_FILE_RE = re.compile(r"\b[\w\-]+\.(?:html?|css|js|py|txt|md|json|pdf|png|jpe?g|csv|sb|log)\b", re.I)


def speakable(text: str) -> str:
    """Strip what must never be read aloud: URLs, file paths, file names.
    ('Saved to ~/sandbox/task07/index.html' -> 'Saved to')"""
    t = _URL_RE.sub("", text or "")
    t = _PATH_RE.sub("", t)
    t = _FILE_RE.sub("", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"\(\s*\)|\[\s*\]", "", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def local_web_search(query: str) -> str:
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=5, region="wt-wt")
        if not results:
            return "No results found."
        lines = [f"- {r.get('title','')}: {r.get('body','')[:200]}" for r in results]
        return "Web results:\n" + "\n".join(lines)
    except Exception as e:
        return f"Search failed: {e}"


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
        query = args.get("query", "")
        # Pi: hand unknown facts to the Mac's browser tool (tasks_mcp.py),
        # which also shows the search on the demo screen. The Pi venv has
        # no ddgs, so local search is only the Mac/fallback path.
        import mcp_client
        if mcp_client.enabled():
            try:
                return mcp_client.call("browse", timeout=25.0, query=query)
            except Exception as e:
                local = local_web_search(query)
                return local if not local.startswith("Search failed") else f"Search failed: {e}"
        return local_web_search(query)
    if name == "set_timer":
        # timers must speak through the caller's own audio path:
        # pipecat_assistant intercepts set_timer (_set_timer_local)
        return "set_timer must be handled by the voice pipeline."
    if name == "music":
        act = (args.get("action") or "").lower()
        target = {"play": "spotify_play", "pause": "spotify_pause", "next": "spotify_next",
                  "previous": "spotify_prev", "prev": "spotify_prev",
                  "volume": "spotify_volume"}.get(act)
        if not target:
            return f"Unknown music action {act!r}."
        sub = {"query": args.get("query", "")} if target == "spotify_play" else (
            {"percent": args.get("percent", 50)} if target == "spotify_volume" else {})
        return home.do_tool(target, sub)
    # spotify_* stay callable directly (the dashboard's buttons use them)
    if name in ("spotify_play", "spotify_pause", "spotify_next", "spotify_prev",
                "spotify_volume", "lights_control", "thermostat_set", "set_alarm",
                "add_reminder", "list_items", "contact_message"):
        return home.do_tool(name, args)
    return f"Unknown tool {name}"
