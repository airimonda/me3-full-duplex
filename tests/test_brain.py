"""brain.py must match chat.py's prompt/tools exactly, and its dispatcher
must run the tools the assistant calls. Run: ./venv/bin/python tests/test_brain.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import brain
fails = 0
def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
try:
    import chat  # deprecated; only for the equality check while it still exists
    check("SYSTEM_PROMPT equals chat.py", brain.SYSTEM_PROMPT == chat.SYSTEM_PROMPT)
    check("QUICK_TOOLS equals chat.py", brain.QUICK_TOOLS == chat.QUICK_TOOLS)
except ImportError:
    print("SKIP  chat.py not importable (deprecated)")
names = {t["function"]["name"] for t in brain.QUICK_TOOLS}
check("15 tools", len(names) == 15, sorted(names))
out = brain.execute_tool("get_time", {});            check("get_time", "local time" in out, out[:60])
out = brain.execute_tool("get_time_in", {"city": "Tokyo"}); check("get_time_in Tokyo", "Tokyo" in out and "Asia/Tokyo" in out, out[:60])
out = brain.execute_tool("web_search", {"query": "capital of Australia"}); check("web_search", out.startswith("Web results"), out[:60])
out = brain.execute_tool("lights_control", {"light": "kitchen", "on": "true"}); check("lights_control", "kitchen" in out.lower(), out[:60])
out = brain.execute_tool("list_items", {"kind": "reminders"}); check("list_items", isinstance(out, str) and out, out[:60])
out = brain.execute_tool("set_timer", {"seconds": 5}); check("set_timer deferred to pipeline", "pipeline" in out, out[:60])
out = brain.execute_tool("nope", {});                 check("unknown tool", out.startswith("Unknown tool"), out)
sys.exit(1 if fails else 0)
