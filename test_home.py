import sys
sys.path.insert(0, str(Path.home() / "me3") + "")
import chat

tests = [
    ({"name": "lights_control", "args": {"on": "true", "light": "living room"}},
     "turn on the living room lights"),
    ({"name": "lights_control", "args": {"brightness_percent": 30, "light": "bedroom"}},
     "dim the bedroom lights to 30%"),
    ({"name": "thermostat_set", "args": {"degrees": 24}}, "set the temperature to 24"),
    ({"name": "set_alarm", "args": {"time": "7:00am", "label": "wake up"}}, "set an alarm for 7am"),
    ({"name": "add_reminder", "args": {"text": "check the oven"}}, "remind me to check the oven"),
    ({"name": "list_items", "args": {"kind": "reminders"}}, "what are my reminders"),
    ({"name": "contact_message", "args": {"kind": "call", "who": "mom"}}, "call mom"),
    ({"name": "spotify_play", "args": {"query": "lofi beats"}}, "play some lofi beats"),
    ({"name": "spotify_volume", "args": {"percent": 60}}, "volume 60"),
    ({"name": "spotify_pause", "args": {}}, "pause"),
]
for tool, spoken in tests:
    out = chat.execute_tool(tool["name"], tool["args"])
    print(f"  {spoken!r:55} -> {out[:80]}")
