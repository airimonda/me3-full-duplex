"""_SpeechOut (streaming TTS): plays every piece in order, never blocks the
caller, and on a barge stashes all unplayed audio for a backchannel resume.
Run: ./venv/bin/python -m pytest tests/test_speech_out.py -q"""
import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipecat_assistant as pa  # noqa: E402


class FakeAgent:
    """The attributes _SpeechOut touches, plus a fake streaming TTS: each
    sentence becomes 3 pieces of 0.1 s PCM whose samples encode the
    sentence index (so order can be checked)."""

    def __init__(self):
        self.gen = 1
        self.playing = False
        self.resume_pcm = None
        self.played16 = np.zeros(0, dtype=np.float32)
        self.last_active = 0
        self.pushed = []
        self.events = []
        self.n_said = 0

    async def _speak_chunks(self, text):
        idx = int(text.split()[-1])
        for _ in range(3):
            await asyncio.sleep(0.01)
            yield np.full(2400, idx, dtype=np.int16).tobytes()  # 0.1 s at 24 kHz

    async def push_frame(self, frame):
        self.pushed.append(frame.audio)

    def log_event(self, ev, **kw):
        self.events.append((ev, kw))


def _samples(chunks):
    return np.frombuffer(b"".join(chunks), dtype=np.int16)


def test_plays_all_sentences_in_order():
    async def run():
        a = FakeAgent()
        out = pa._SpeechOut(a, 1)
        for i in (1, 2, 3):
            out.say(f"sentence {i}")
        await out.finish()
        return a, out

    a, out = asyncio.run(run())
    s = _samples(a.pushed)
    assert len(s) == 3 * 3 * 2400
    assert list(dict.fromkeys(s.tolist())) == [1, 2, 3]   # order kept
    assert out.first_audio is not None and a.playing
    assert not a.events


def test_say_never_blocks():
    async def run():
        a = FakeAgent()
        out = pa._SpeechOut(a, 1)
        t0 = asyncio.get_running_loop().time()
        for i in range(5):
            out.say(f"s {i}")
        dt = asyncio.get_running_loop().time() - t0
        await out.finish()
        return dt

    assert asyncio.run(run()) < 0.01


def test_barge_stashes_unplayed_audio():
    async def run():
        a = FakeAgent()
        out = pa._SpeechOut(a, 1)
        for i in (1, 2, 3):
            out.say(f"sentence {i}")
        await asyncio.sleep(0.4)        # some frames played, synth ran ahead
        a.gen = 2                       # barge
        await out.finish()
        return a

    a = asyncio.run(run())
    played = _samples(a.pushed)
    rest = np.frombuffer(a.resume_pcm, dtype=np.int16)
    assert 0 < len(played) < 3 * 3 * 2400
    assert len(rest) > 0
    # played + stashed is a prefix of the full reply, in order, no gaps
    both = np.concatenate([played, rest])
    full = np.repeat([1, 2, 3], 3 * 2400)
    assert np.array_equal(both, full[:len(both)])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
