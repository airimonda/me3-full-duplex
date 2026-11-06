"""Wake word recall + idle gate (mic mode).

1. wake_split: "Watson" and its common Whisper mishearings ("button",
   "Watch on.", "What's on?") count as the wake word; ordinary openers
   ("What's the time", "Wait", "Watch this") do not.
2. _respond (real code, stubbed ASR/TTS/LLM): after ME3_IDLE_S with no
   activity, plain speech is ignored (ignored_idle); a wake-prefixed command
   wakes Watson (woke) and is answered with the wake word stripped; once
   awake, plain speech is answered again.
3. Wake-prefixed 2-word commands ("Watson, lights off") are not dropped as
   fragments.
No server needed. Run: ./venv/bin/python tests/test_wake.py"""
import asyncio, json, os, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ME3_MODE"] = "mic"
import numpy as np
import pipecat_assistant as pa

WAKE = [
    ("Watson", ""), ("Watson.", ""), ("Watson?", ""), ("watson!", ""),
    ("button.", ""), ("button", ""), ("Watch on.", ""), ("What's on?", ""),
    ("Wharton", ""), ("Whatson", ""), ("Wattson, what time is it?", "what time is it?"),
    ("Hey Watson, play some jazz", "play some jazz"),
    ("Okay Watson turn off the lights", "turn off the lights"),
    ("Watson, what is the temperature in Metro Manila today?",
     "what is the temperature in Metro Manila today?"),
    ("Watson is it going to rain", "is it going to rain"),
    ("Watsun, lights off", "lights off"),
    ("Wason, set a timer", "set a timer"),
    ("What son, stop", "stop"),
    ("That's on.", ""), ("Thats on, what time is it", "what time is it"),
    ("that sun.", ""), ("That sun, lights off", "lights off"),   # live Pi, Nov 6
    ("Hatson", ""), ("Whatsun?", ""),
]
NOT_WAKE = [
    "What's the time?", "What is the weather like", "Wait", "Wait a second",
    "Watch this", "Was it raining yesterday", "Washing machine is done",
    "Weather in Manila", "Wilson called me", "Water the plants",
    "I asked Watson earlier", "Turn off the lights", "Reason with me",
    "That's not right", "That's all", "That's the one", "What's up",
    "Hats off to you", "Person of interest", "Lesson one", "Boston weather",
    "Thanks a lot", "Lights off", "first.", "What's your name?",
]

fails = 0
for text, rest in WAKE:
    ok, got = pa.wake_split(text)
    if not ok or got != rest:
        fails += 1
        print(f"FAIL wake     {text!r}: got ({ok}, {got!r}) want (True, {rest!r})")
for text in NOT_WAKE:
    ok, got = pa.wake_split(text)
    if ok:
        fails += 1
        print(f"FAIL not-wake {text!r}: matched, rest {got!r}")
print(f"wake_split: {len(WAKE) + len(NOT_WAKE) - fails}/{len(WAKE) + len(NOT_WAKE)} cases")


class NoNet:
    """Stops _respond right after the gates (where the LLM call starts)."""
    def __getattr__(self, name):
        raise RuntimeError("accepted")


async def run_case(agent, text, idle):
    agent.last_active = time.monotonic() - (pa._IDLE_S + 1 if idle else 0)
    n0 = len(agent.history)
    async def fake_asr(pcm): return text
    agent._asr = fake_asr
    try:
        await agent._respond(np.full(8000, 3000, dtype=np.int16).tobytes())  # 0.5 s
    except Exception:
        pass
    user = [m["content"] for m in agent.history[n0:] if m["role"] == "user"]
    return user[0] if user else None


async def main():
    global fails
    d = Path(tempfile.mkdtemp())
    agent = pa.WatsonAgent(d)
    agent.speaker = None          # no voice gate; short pcm -> no enrollment
    agent.http = NoNet()
    async def no_tts(text): return None
    agent._speak = no_tts
    async def fake_push(frame, direction=None): pass
    agent.push_frame = fake_push

    cases = [  # (text, idle, expected answered text or None)
        ("What is the weather in Manila today", True, None),
        ("Watson, what is the weather in Manila today", True,
         "what is the weather in Manila today"),
        ("Button, what time is it in Tokyo", True, "what time is it in Tokyo"),
        ("What is the weather in Manila today", False,
         "What is the weather in Manila today"),
        ("Watson, lights off", False, "lights off"),
        ("lights off", False, None),               # plain 2 words: still a fragment
    ]
    for text, idle, want in cases:
        got = await run_case(agent, text, idle)
        if got != want:
            fails += 1
            print(f"FAIL respond  {text!r} idle={idle}: answered {got!r}, want {want!r}")
    # bare wake while idle: greets and wakes (activity timer reset)
    agent.last_active = time.monotonic() - pa._IDLE_S - 1
    async def asr_bare(pcm): return "What's on?"
    agent._asr = asr_bare
    await agent._respond(np.full(8000, 3000, dtype=np.int16).tobytes())
    if time.monotonic() - agent.last_active > 5:
        fails += 1
        print("FAIL bare wake did not reset the idle timer")
    # live Pi, Nov 6: "Watson <pause> lights off" arrives as two utterances.
    # After a bare wake, the next real utterance is addressed even if short;
    # a ghost or our own greeting echo in between doesn't use the window up.
    # (awake, a 3-word greeting echo is answered like any sentence — the point
    # is that it leaves the follow-up window armed)
    for text, want in [("Thank you.", None), ("Yes, I'm listening.", "Yes, I'm listening."),
                       ("lights off.", "lights off."), ("lights off.", None)]:
        got = await run_case(agent, text, False)
        if got != want:
            fails += 1
            print(f"FAIL follow-up {text!r}: answered {got!r}, want {want!r}")

    ev = [json.loads(l)["event"] for l in open(d / "watson_log.jsonl")]
    for need in ("ignored_idle", "woke", "greeted"):
        if need not in ev:
            fails += 1
            print(f"FAIL no {need} event; events: {ev}")
    print(f"respond events: {[e for e in ev if e != 'asr']}")
    print("PASS" if fails == 0 else f"FAIL ({fails})")
    sys.stdout.flush()
    os._exit(0 if fails == 0 else 1)

asyncio.run(main())
