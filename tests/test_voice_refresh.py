"""Voice-profile refresh must BLEND, never replace (Nov 6: one short
"Watson lights off" overwrote the profile and locked the owner out).

1. blend_profile: one refresh toward an unrelated voice keeps the profile
   >= 0.99 cosine to the old one; result is unit length.
2. _respond (real code, stubbed speakerid): a near-miss wake-prefixed turn
   saves the blended profile, updates the in-memory profile too, and logs
   voice_refreshed.
No server needed. Run: ./venv/bin/python tests/test_voice_refresh.py"""
import asyncio, json, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ME3_MODE"] = "mic"
import numpy as np
import pipecat_assistant as pa

fails = 0
rng = np.random.default_rng(0)
owner = rng.normal(size=256).astype(np.float32)
owner /= np.linalg.norm(owner)
other = rng.normal(size=256).astype(np.float32)
other /= np.linalg.norm(other)

b = pa.blend_profile(owner, other)
cos = float(np.dot(b, owner))
print(f"blend: cos(new, old) {cos:.3f}, norm {np.linalg.norm(b):.3f}")
if cos < 0.99 or abs(np.linalg.norm(b) - 1) > 1e-4:
    fails += 1
    print("FAIL blend moved the profile too far or is not unit length")


# near-miss voice: cosine 0.65 to the owner (the gate now embeds once and
# computes the cosine itself, off the event loop)
_orth = other - np.dot(other, owner) * owner
_orth /= np.linalg.norm(_orth)
near = (0.65 * owner + np.sqrt(1 - 0.65 ** 2) * _orth).astype(np.float32)


class StubSpk:
    saved = []
    def near_threshold(self): return 0.60
    def threshold(self): return 0.72
    def embed(self, pcm): return near
    def voiced_seconds(self, pcm): return 3.0
    def save_profile(self, emb): self.saved.append(np.array(emb))


class NoNet:
    def __getattr__(self, name):
        raise RuntimeError("accepted")


async def main():
    global fails
    d = Path(tempfile.mkdtemp())
    agent = pa.WatsonAgent(d)
    agent.spk = StubSpk()
    agent.speaker = owner.copy()
    agent.http = NoNet()
    async def asr(pcm): return "Watson, turn the kitchen lights off please"
    agent._asr = asr
    try:
        await agent._respond(np.full(48000, 3000, dtype=np.int16).tobytes())
    except Exception:
        pass
    saved = StubSpk.saved
    if len(saved) != 1:
        fails += 1
        print(f"FAIL expected 1 saved profile, got {len(saved)}")
    else:
        c_saved = float(np.dot(saved[0], owner))
        c_mem = float(np.dot(agent.speaker, owner))
        print(f"respond: saved cos to owner {c_saved:.3f}, in-memory {c_mem:.3f}")
        if c_saved < 0.99 or not np.allclose(saved[0], agent.speaker):
            fails += 1
            print("FAIL saved profile is not the blend, or memory not updated")
    ev = [json.loads(l)["event"] for l in open(d / "watson_log.jsonl")]
    if "voice_refreshed" not in ev:
        fails += 1
        print(f"FAIL no voice_refreshed event: {ev}")
    print("PASS" if fails == 0 else f"FAIL ({fails})")
    sys.stdout.flush()
    os._exit(0 if fails == 0 else 1)

asyncio.run(main())
