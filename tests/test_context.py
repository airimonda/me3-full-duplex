"""Context hygiene (Nov 6): after a turn has its spoken reply, its tool-call
and tool-result messages are dropped from history; other turns are untouched.
Also: the merged `music` tool dispatches to the spotify_* handlers.
No server needed. Run: ./venv/bin/python tests/test_context.py"""
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ME3_MODE"] = "mic"
import brain
import pipecat_assistant as pa

fails = 0
agent = pa.WatsonAgent(Path(tempfile.mkdtemp()))
sysmsg = agent.history[0]
u1 = {"role": "user", "content": "lights off"}
call = {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
        "function": {"name": "lights_control", "arguments": "{}"}}]}
res = {"role": "tool", "tool_call_id": "c1", "content": "x" * 1000}
u2 = {"role": "user", "content": "and the time?"}
call2 = dict(call)
res2 = {"role": "tool", "tool_call_id": "c2", "content": "12:00"}
agent.history = [sysmsg, u1, call, res, u2, call2, res2]
agent._collapse_tools(u1)
roles = [m["role"] + ("+calls" if m.get("tool_calls") else "") for m in agent.history]
want = ["system", "user", "user", "assistant+calls", "tool"]
if roles != want:
    fails += 1
    print("FAIL collapse:", roles, "want", want)
chars = sum(len(str(m.get("content") or "")) for m in agent.history)
print(f"history after collapse: {roles}, {chars} chars")

names = [t["function"]["name"] for t in brain.QUICK_TOOLS]
if "music" not in names or any(n.startswith("spotify_") for n in names):
    fails += 1
    print("FAIL tool list:", names)
out = brain.execute_tool("music", {"action": "nope"})
if "Unknown music action" not in out:
    fails += 1
    print("FAIL music dispatch:", out)
print("PASS" if fails == 0 else f"FAIL ({fails})")
sys.stdout.flush()
os._exit(0 if fails == 0 else 1)
