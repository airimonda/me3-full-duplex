"""Watson's brain: system prompt, tool schemas and the tool dispatcher.

Moved out of chat.py (deprecated) on 2026-11-05. pipecat_assistant.py
imports this module; it has no audio or loop code, so it loads on the Pi
without sounddevice or torch.
"""
import time

import home  # noqa: E402  (Spotify + virtual smart home)


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
        # timers must speak through the caller's own audio path:
        # pipecat_assistant intercepts set_timer (_set_timer_local)
        return "set_timer must be handled by the voice pipeline."
    if name in ("spotify_play", "spotify_pause", "spotify_next", "spotify_prev",
                "spotify_volume", "lights_control", "thermostat_set", "set_alarm",
                "add_reminder", "list_items", "contact_message"):
        return home.do_tool(name, args)
    return f"Unknown tool {name}"
