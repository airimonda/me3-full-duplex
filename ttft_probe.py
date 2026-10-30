"""TTFT probe: 4 consecutive talker calls in ONE process to show warm-connection gain."""
import time
from loop import load_openrouter_key, talker

key = load_openrouter_key()
for i in range(4):
    t0 = time.perf_counter()
    reply, first = talker("What time is it right now?", key)
    print(f"call {i}: ttft {first:.3f}s  total {time.perf_counter()-t0:.3f}s  {reply[:50]!r}", flush=True)