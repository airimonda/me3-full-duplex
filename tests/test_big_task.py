"""Big task end to end through WatsonAgent (mic mode), the way the Pi runs it.

1. _delegate() hands a build job to the task MCP server and speaks the ack.
2. While it runs, a small task (get_time_in Tokyo) answers immediately.
3. _announce_finished() collects the finished job over MCP and speaks a
   line with no path/URL; the server reports it opened the deliverable,
   and the deliverable sits inside ~/sandbox.
Needs server.py (:8300) and tasks_mcp.py (ME3_TASKS_MCP) running.
Run: ME3_TASKS_MCP=http://127.0.0.1:8320/mcp ./venv/bin/python tests/test_big_task.py"""
import asyncio, json, os, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ME3_MODE"] = "mic"
import pipecat_assistant as pa


async def main():
    fails = 0
    d = Path(tempfile.mkdtemp())
    agent = pa.WatsonAgent(d)
    spoken = []
    async def fake_push(frame, direction=None):
        pass
    agent.push_frame = fake_push
    real_speak = agent._speak
    async def spy_speak(text):
        spoken.append(text)
        return await real_speak(text)
    agent._speak = spy_speak

    t0 = time.time()
    ack = await agent._delegate("a one-page website that says Hello from Watson",
                                agent.gen, user_text="build me a page")
    print(f"ack ({time.time() - t0:.1f} s): {ack!r}")

    t1 = time.time()
    small = agent.chatbrain.execute_tool("get_time_in", {"city": "Tokyo"})
    print(f"small task while the big one runs ({time.time() - t1:.2f} s): {small}")

    my_tid = [json.loads(l) for l in open(d / "watson_log.jsonl")
              if '"delegated"' in l][-1]["tid"]
    deadline = time.time() + 600
    announced = None
    while time.time() < deadline and announced is None:
        await asyncio.sleep(1.0)
        await agent._announce_finished()   # older leftovers are spoken first
        ev = [json.loads(l) for l in open(d / "watson_log.jsonl")]
        hits = [e for e in ev if e["event"] == "announce" and e["id"] == my_tid]
        if hits:
            announced = hits[-1]
    print(f"announce after {time.time() - t0:.0f} s: {announced}")
    ev = [json.loads(l) for l in open(d / "watson_log.jsonl")]
    deleg = [e for e in ev if e["event"] == "delegated"]
    if not deleg or deleg[-1].get("via") != "mcp" or deleg[-1].get("tid") in ("None", ""):
        fails += 1
        print("FAIL not delegated over MCP:", deleg)
    if not announced:
        fails += 1
        print("FAIL no announcement")
    else:
        line = announced["line"]
        if "/" in line or "sandbox" in line.lower() or ".html" in line.lower():
            fails += 1
            print("FAIL announcement reads a path:", line)
        if not announced.get("opened"):
            fails += 1
            print("FAIL deliverable not opened")
        tid = announced["id"]
        tdir = os.path.expanduser(f"~/sandbox/{tid}")
        files = [f for f in os.listdir(tdir) if f not in ("prompt.txt", "run.log")]
        print(f"sandbox deliverables ({tid}):", files, "| delegated tid:", deleg[-1]["tid"])
        if not files:
            fails += 1
            print("FAIL nothing in the sandbox folder")
    if "Tokyo" not in small:
        fails += 1
        print("FAIL small task did not answer")
    print("spoken:", spoken)
    print("PASS" if fails == 0 else f"FAIL ({fails})")
    sys.stdout.flush()
    os._exit(0 if fails == 0 else 1)

asyncio.run(main())
