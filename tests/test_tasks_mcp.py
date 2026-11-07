"""Task MCP end to end (needs tasks_mcp.py running; ME3_TASKS_MCP points at it).

1. speakable(): paths, URLs and file names are never spoken.
2. MCP: tools/list shows submit_task, list_tasks, finished, browse.
3. With --live: submit a real small build job, poll finished() until it
   lands, check the line has no path, the deliverable sits inside
   ~/sandbox, and the server reports it opened it.
Run: ME3_TASKS_MCP=http://127.0.0.1:8320/mcp ./venv/bin/python tests/test_tasks_mcp.py [--live]"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain import speakable
import mcp_client

fails = 0
CASES = [
    ("Done! Saved to ~/sandbox/task07/index.html — take a look.", "Done! Saved to — take a look."),
    ("See /Users/ailene/sandbox/task03 for it.", "See for it."),
    ("Open https://example.com/x?y=1 now.", "Open now."),
    ("I wrote index.html and style.css.", "I wrote and."),
    ("Open 24/7 and/or on weekends.", "Open 24/7 and/or on weekends."),
    ("It's 5:19 PM.", "It's 5:19 PM."),
]
for raw, want in CASES:
    got = speakable(raw)
    if got != want:
        fails += 1
        print(f"FAIL speakable {raw!r}: got {got!r}, want {want!r}")
print(f"speakable: {len(CASES) - fails}/{len(CASES)}")

if not mcp_client.enabled():
    print("FAIL ME3_TASKS_MCP not set")
    sys.exit(1)
tools = mcp_client.list_tools()
print("tools:", tools)
if not {"submit_task", "list_tasks", "finished", "browse"} <= set(tools):
    fails += 1
    print("FAIL missing tools")

if "--live" in sys.argv:
    t0 = time.time()
    sub = json.loads(mcp_client.call("submit_task",
                     request="a one-page website that says Hello from Watson in big letters"))
    print("submit:", sub)
    if not sub.get("id"):
        fails += 1
        print("FAIL submit returned no id")
    else:
        got = None
        while time.time() - t0 < 600 and got is None:
            time.sleep(5)
            for a in json.loads(mcp_client.call("finished")):
                if a["id"] == sub["id"]:
                    got = a
        print(f"finished after {time.time() - t0:.0f} s:", got)
        tdir = os.path.expanduser(f"~/sandbox/{sub['id']}")
        files = [f for f in os.listdir(tdir) if f not in ("prompt.txt", "run.log")]
        print("deliverables in sandbox:", files)
        if not got or got["status"] != "done":
            fails += 1
            print("FAIL job did not finish as done")
        elif speakable(got["line"]) != got["line"] or "/" in got["line"]:
            fails += 1
            print("FAIL announcement contains a path")
        if got and not got.get("opened"):
            fails += 1
            print("FAIL deliverable was not opened")
        if not files:
            fails += 1
            print("FAIL no deliverable inside the sandbox folder")
print("PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(0 if fails == 0 else 1)
