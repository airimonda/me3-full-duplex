# Step 4 (Oct 26) — streaming and tuning results

## Warm connections
- All HTTP now goes through one shared httpx.Client (loop.py `HTTP`), reused by chat.py.
- In-process probe (ttft_probe.py, GLM-5.3-flash via OpenRouter): TTFT 1.19-1.45 s steady.
- Cold-process runs measure 3.5-3.9 s — do not benchmark per-process; chat.py sessions are warm.
- GLM floor is provider-side (~1.2 s TTFT). Known accepted trade-off (user decision).

## Routing bug found
- GLM classified "What time is it right now?" as delegate-worthy ("Let me check that...")
  instead of answering. Fix in Step 6 routing prompt (time = quick tool, never delegate).
- Marker inconsistency fixed: loop.py prompt now uses "Let me check that." like chat.py.

## Endpointing sweep (sweep_endpoint.py, Silero VADIterator, 16 kHz)
| budget | endpoint lag | cuts pauses of |
|---|---|---|
| 300 ms | 0.39 s | >= ~0.4 s |
| 400 ms | 0.48 s | >= ~0.5 s |
| 500 ms | 0.58 s | >= ~0.6 s |
| 700 ms | 0.77 s | >= ~0.8 s |
Lag = budget + ~90 ms VAD overhead, linear. Default stays 400 ms (ME3_SILENCE_MS).
The lag-vs-pause-cut trade-off is the planned Step 10 headline finding for v3
pause/hesitation samples.

## Misc
- Short/empty transcripts already dropped (<2 chars).
- TTS streams clause-by-clause (first sentence plays while the LLM is still generating).
