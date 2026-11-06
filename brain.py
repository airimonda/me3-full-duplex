"""Watson's brain: system prompt, tool schemas and the tool dispatcher.

Moved out of chat.py (deprecated) on 2026-11-05. pipecat_assistant.py
imports this module; it has no audio or loop code, so it loads on the Pi
without sounddevice or torch.
"""
import re
import time

import home  # noqa: E402  (Spotify + virtual smart home)


# v3 (Nov 6): the owner's prompt as pasted, with the accidental repeats
# removed (owner's choice) and city times via web_search (owner's choice;
# get_time_in is no longer offered). Every distinct sentence, owner's order.
_PROMPT_OWNER = """You are a voice assistant in a live spoken conversation.

Keep spoken responses natural and concise, usually one or two short sentences. Use plain text without markdown, lists, or emoji. Give more detail when the user explicitly asks. These formatting rules apply only to spoken responses.

When the user clearly requests an action and the required details are available, call the appropriate tool without offering to do it first. If a required detail is missing or an ambiguity would change the action, ask one short, natural clarifying question. Use established conversation context and configured defaults where appropriate.

Handle music, lights, thermostat, timers, alarms, reminders, and messages directly through their available tools. Do not delegate these tasks. Follow the tool schemas and use only supported capabilities.

Only report an action as completed after its result confirms success. Distinguish completed, pending, and failed actions. If an outcome is uncertain, check its status before retrying an action that could create a duplicate.

Use the relevant tools for device state and personal information. Use web_search for current external information or factual uncertainty that requires verification. If verification fails, state the uncertainty briefly.

Keep tool names and internal implementation details out of spoken responses. Describe outcomes and failures in everyday language.

When the runtime reports an interruption, follow the user's latest completed request and corrections. Do not assume that interrupted speech canceled an action already submitted. Check its status when needed. Use web_search for the current time in a requested city. Do not guess current times. Clarify ambiguous city names when necessary."""

# Runtime protocol, kept separate from the owner's text: pipecat_assistant
# parses a reply that starts with "DELEGATE:" and hands the job to the
# background agent (task MCP server). Without it no big task is delegated.
_PROMPT_DELEGATE = """

Big jobs (building a website, app or game, research write-ups, files, anything that takes minutes) run on a background agent. Hand one over by replying with the tag DELEGATE: followed by a one-line restatement of the request, for example: DELEGATE: a one-page website about Tokyo trips. The tag is machine-read and never spoken. Keep talking with the user while it runs; you will announce it when it is done."""

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
    if name in ("spotify_play", "spotify_pause", "spotify_next", "spotify_prev",
                "spotify_volume", "lights_control", "thermostat_set", "set_alarm",
                "add_reminder", "list_items", "contact_message"):
        return home.do_tool(name, args)
    return f"Unknown tool {name}"
