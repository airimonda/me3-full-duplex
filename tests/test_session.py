"""Session ownership (owner's rule, Nov 6): only the session holder is
answered; anyone takes the session by calling Watson; Watson's own echo
cannot. Runs the real _respond with a stub speaker-ID.
No server needed. Run: ./venv/bin/python tests/test_session.py"""
import asyncio, json, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ME3_MODE"] = "mic"
import numpy as np
import pipecat_assistant as pa

rng = np.random.default_rng(1)


def unit():
    v = rng.normal(size=256).astype(np.float32)
    return v / np.linalg.norm(v)


OWNER, STRANGER, WATSON = unit(), unit(), unit()


class StubSpk:
    def __init__(self):
        self.next = None
        self.saved = []
    def embed(self, pcm): return self.next
    def voiced_seconds(self, pcm): return 3.0
    def near_threshold(self): return 0.60
    def threshold(self): return 0.72
    def save_profile(self, e): self.saved.append(e)


class NoNet:
    def __getattr__(self, name):
        raise RuntimeError("accepted")


async def say(agent, text, voice):
    agent.spk.next = voice
    n0 = len(agent.history)
    async def asr(pcm): return text
    agent._asr = asr
    try:
        await agent._respond(np.full(48000, 3000, dtype=np.int16).tobytes())
    except Exception:
        pass
    return any(m["role"] == "user" for m in agent.history[n0:])


async def main():
    fails = 0
    d = Path(tempfile.mkdtemp())
    agent = pa.WatsonAgent(d)
    agent.spk = StubSpk()
    agent.speaker = OWNER.copy()
    agent.watson_emb = WATSON
    agent.http = NoNet()
    async def no_tts(t): return None
    agent._speak = no_tts

    steps = [  # (text, voice, answered?, who holds the session after)
        ("what is the weather in Manila today", OWNER, True, "owner"),
        ("what is the weather in Manila today", STRANGER, False, "owner"),
        ("Watson, what is the weather in Manila today", STRANGER, True, "stranger"),
        ("what is the capital of Spain please", STRANGER, True, "stranger"),
        ("what is the capital of Spain please", OWNER, False, "stranger"),
        ("Watson, I am back, what time is it", OWNER, True, "owner"),
        ("Watson, here is some echo of my own voice", WATSON, False, "owner"),
    ]
    for text, voice, want_ans, want_holder in steps:
        got = await say(agent, text, voice)
        holder = "owner" if pa._cos(agent.speaker, OWNER) > 0.99 else (
            "stranger" if pa._cos(agent.speaker, STRANGER) > 0.99 else "other")
        ok = got == want_ans and holder == want_holder
        fails += not ok
        who = "owner" if voice is OWNER else "stranger" if voice is STRANGER else "watson-echo"
        print(f"{'ok  ' if ok else 'FAIL'} {who:12} {text[:44]:44} answered={got} holder={holder}")
    ev = [json.loads(l)["event"] for l in open(d / "watson_log.jsonl")]
    for need in ("session_claimed", "stranger_ignored", "echo_claim_ignored"):
        if need not in ev:
            fails += 1
            print("FAIL missing event", need)
    print("PASS" if fails == 0 else f"FAIL ({fails})")
    sys.stdout.flush()
    os._exit(0 if fails == 0 else 1)

asyncio.run(main())
