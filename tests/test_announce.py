"""Finished-task announcement: a stub task manager reports one finished task;
the agent must speak its line (audio frames pushed), open the deliverable and
log 'announce'. Needs server.py on :8300. Run: ./venv/bin/python tests/test_announce.py"""
import asyncio, json, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ME3_MODE"] = "mic"
import pipecat_assistant as pa

class StubTasks:
    opened = []
    def check_finished(self): return ["task99"]
    def announcement(self, tid): return "Done! Your Tokyo trip page is ready — take a look."
    def open_deliverable(self, tid): self.opened.append(tid); return "/tmp/task99/index.html"

async def main():
    d = Path(tempfile.mkdtemp())
    agent = pa.WatsonAgent(d)
    agent.taskmgr = StubTasks()
    frames = []
    async def fake_push(frame, direction=None): frames.append(frame)
    agent.push_frame = fake_push
    await agent._announce_finished()
    events = [json.loads(l) for l in open(d / "watson_log.jsonl")]
    ann = [e for e in events if e["event"] == "announce"]
    audio = [f for f in frames if type(f).__name__ == "OutputAudioRawFrame"]
    secs = sum(len(f.audio) for f in audio) / 2 / pa._OUT_SR
    ok = bool(ann) and secs > 1.0 and StubTasks.opened == ["task99"] and not agent.playing
    print(f"announce events {len(ann)}, audio {secs:.1f} s in {len(audio)} frames, "
          f"opened {StubTasks.opened}, playing after={agent.playing}  ->  {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)

asyncio.run(main())
