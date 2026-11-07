# ME3 — Full-Duplex Voice Agent ("Watson")

A full-duplex voice assistant on a Raspberry Pi. You can talk to it naturally, interrupt it
mid-sentence, and ask it to do real work in the background. Built for AI 231 (demo: November 7, 2026).

On the Pi, Watson starts answering a median **2.56 s** after you stop talking (p95 3.93 s, target 4.0 s),
measured with the course harness (fdbench) on Nov 7. Speech recognition, the language model and Watson's
voice all stream.

## What it does

- **Full-duplex conversation**: Watson listens while it talks. Interrupt with "stop", "wait" or
  "Watson" and it stops; a "yeah" backchannel lets it resume. TV, side conversations and noise are ignored.
- **Session ownership**: Watson answers the session holder only (resemblyzer voice embedding). Anyone
  claims the session by saying "Watson". It knows its own voice, so its echo is not mistaken for you.
- **Instant tools**: 10 function-calling tools (time, web search, music, lights, thermostat, timers,
  alarms, reminders, messages). On a tool turn Watson says "One sec." right away.
- **Background tasks**: big jobs ("build me a website about …") go to a task MCP server on the Pi that runs
  the pi.dev coding agent in a bubblewrap sandbox (`~/sandbox`). Watson keeps talking and announces the
  result when it is done; the page opens on the Mac.

## Architecture

```
 Raspberry Pi 4 (4 GB)                                         cloud
 ┌──────────────────────────────────────────────┐
 │ USB mic → Silero VAD (16 kHz, 0.6 s endpoint) │── live websocket ──► Deepgram ASR nova-3
 │ WatsonAgent (one Pipecat FrameProcessor)      │◄─ transcript ~0.25 s after you stop
 │   barge-in, voice gates, tools, history (12)  │── SSE stream ──────► OpenRouter z-ai/glm-5.3-flash
 │   _SpeechOut: plays audio as it streams in    │                      (fallback openai/gpt-4o-mini)
 │ speaker ← 24 kHz PCM                          │◄─ streamed PCM ───── Deepgram TTS aura-2-jupiter-en
 │ tasks_mcp.py :8320 → pi.dev agent (bwrap)     │
 │ ui_server.py :8301 (Watson UI)                │
 └──────────────────────────────────────────────┘
 Mac: fdbench harness, mac_show.py (opens results), fallback Whisper + Kokoro server (server.py :8300)
```

The same `WatsonAgent` runs behind the Pi's mic (`--mic`) and behind the benchmark socket, with no code changes.

### Latency path (Nov 7)

| Stage | How | Median |
|---|---|---|
| ASR | Deepgram live websocket fed while the user talks, from a pre-connected spare socket; Finalize at the VAD endpoint | 0.24 s |
| LLM | OpenRouter SSE read as it arrives; sentences go to TTS as soon as they end | 1.99 s to first token |
| TTS | Deepgram raw PCM played as bytes arrive; next sentence synthesized during playback | 0.29 s |

Feature flags (all default on): `ME3_TTS_STREAM`, `ME3_ASR_STREAM`, `ME3_TOOL_ACK`, `ME3_LLM_FAST`.
Set one to `0` to get the earlier behaviour for an A/B run.

## Run it

On the Pi (`ssh watsonpi`), each script restarts its process if it exits:

```bash
setsid nohup bash ~/me3/run_live.sh      >/dev/null 2>&1 </dev/null &   # Watson on mic + speaker
setsid nohup bash ~/me3/run_tasks_mcp.sh >/dev/null 2>&1 </dev/null &   # task MCP + pi.dev agent
setsid nohup bash ~/me3/run_ui.sh        >/dev/null 2>&1 </dev/null &   # Watson UI on :8301
setsid nohup bash ~/me3/run_spotify.sh   >/dev/null 2>&1 </dev/null &   # Spotify Connect "Watson" (Premium)
```

Spotify, one time: `bash ~/me3/setup_spotify_pi.sh install`, then `... pair` and approve the spotify.com/pair link.

On the Mac, open finished tasks as they land: `./venv/bin/python mac_show.py --pi http://<pi-ip>:8301`.

Keys are read at runtime from the environment or `~/.hermes/.env`: `OPENROUTER_API_KEY`, `DEEPGRAM_API_KEY`
(the Deepgram key may also live in `~/me3/.env`).

Mac-only mode (local Whisper + Kokoro, no Pi): `make setup`, then `make run`.

## Benchmark

Harness: [ai231-me3-full-duplex-benchmarking](https://github.com/trueshotjames/ai231-me3-full-duplex-benchmarking)
(cloned as `fd-benchmark/`). Watson runs on the Pi; the harness runs on the Mac:

```bash
# Pi: keep the benchmark client connected (stop run_live.sh first)
setsid nohup bash ~/me3/run_bench_pi.sh >/dev/null 2>&1 </dev/null &
# Mac: full run (2 rounds + 30 timed turns, gpt-4o judge, ~30 min)
cd fd-benchmark && ./.venv/bin/python -m fdbench run --student me3 --rounds 2 --seed 922546 \
  --connect direct --where remote --judge-provider openrouter --yes --no-live --recordings k
# quick latency-only check (~8 min): --rounds 0 --phrases 10 --runs 2 --judge-provider none
```

Latest full run: Nov 7, `results/run-2026-11-07-pi/` (fdbench run `2026-11-07_10-10-53`):

| Metric | Nov 6 (before streaming) | Nov 7 |
|---|---|---|
| E2E median / p95 | 4.72 / 7.49 s | **2.56 / 3.93 s** |
| ASR median | 0.92 s | 0.24 s |
| LLM first token median / p95 | 2.10 / 4.28 s | 1.99 / 3.26 s |
| TTS first audio median | 1.45 s | 0.29 s |
| WER | 11.1 % | 12.1 % |
| Pause take-overs | 1/4 | 0/4 |
| Interruptions handled | 0/1 | 2/2 (judge 5.0/5) |
| Answered speech not meant for it | 1/4 | 0/4 |
| Context per LLM call | not reported | 1,670 prompt tokens median, 12 messages |

Full-Duplex-Bench groups have n = 2, so single trials flip between runs. History, earlier runs and the
gate trade-off: [RESULTS.md](RESULTS.md) and [docs/paper.md](docs/paper.md).

## Files

| File | Role |
|---|---|
| `pipecat_assistant.py` | the agent: VAD, barge-in, gates, streaming ASR/LLM/TTS, tools, delegation |
| `brain.py` | system prompt (owner's text, v4), tool schemas, tool dispatcher |
| `tasks_mcp.py`, `mcp_client.py` | task MCP server (pi.dev in bwrap) and the agent's client |
| `speakerid.py` | voice enrollment and similarity |
| `home.py` | music + virtual smart-home state |
| `ui_server.py`, `ui/index.html`, `mac_show.py` | Watson UI on the Pi (:8301); Mac viewer that opens finished tasks |
| `server.py` | fallback Whisper + Kokoro server on the Mac (:8300) |
| `bench_entry.py`, `fdbench_marks.py` | fdbench entry point and stage marks |
| `run_live.sh`, `run_bench_pi.sh`, `run_tasks_mcp.sh`, `run_ui.sh` | Pi launchers |
| `docs/demo-spiel.html`, `docs/machine_exercise_2page.html` | demo run sheet and the 2-page deck |
| `tests/` | unit tests (`./venv/bin/python tests/test_speech_out.py`, etc.) |
| `legacy/` | deprecated: `chat.py`, `duplex.py` and their scripts |

## Notes

- `voice_profile.npy` is biometric and deliberately git-ignored.
- After a PipeWire restart the USB mic resets to 100 % gain (clips); `run_live.sh` sets it back to 81 %.
