# ME3 — Full-Duplex Voice Agent ("Watson")

A full-duplex spoken AI assistant you can talk to naturally, interrupt mid-sentence, and ask to do real background work. Built for AI 231 (demo: November 7).

One command to run it, one command to reproduce the benchmark:

```bash
make setup   # once: create the venv and install everything
make run     # live: talk to Watson on mic + speaker
make bench   # reproduce the frozen benchmark (28 trials, ~9 min)
```

Requires: Apple-Silicon macOS (MLX), an OpenRouter API key in `~/.hermes/.env` (`OPENROUTER_API_KEY=...`), and for background tasks the Hermes Agent CLI.

## What it does

- **Full-duplex conversation** — Silero VAD + an echo-aware barge-in policy means Watson hears you while speaking, stops within ~0.3–1.6 s of real interruptions, keeps talking through backchannels ("mm-hm"), and ignores TV, side conversations and room noise.
- **Speaker enrollment** — one clear utterance is enough; Watson only answers your voice afterwards (resemblyzer embedding, 0.72 threshold, self-healing refresh). Stranger → politely ignored.
- **Instant tools** — 15 quick tools (time, time-in-city, web search, Spotify playback, lights, thermostat, alarms, reminders, media control) executed directly by the talker via function-calling.
- **Background tasks** — say *"have the agent do ___"* and the request is handed, with no LLM appraisal, to a sandboxed Hermes Agent confined by a Seatbelt profile to `~/sandbox`. Watson acknowledges and keeps talking; the agent announces results when done.
- **Honest voice** — never claims inability, never reads file paths aloud, handles Taglish, asks one clarifying question when your request is incomplete.

## Architecture

```
 mic ──► LocalAudioTransport ──► WatsonAgent (one FrameProcessor) ──► SpeakerOutput
            (Pipecat 1.14)        │ Silero VAD (16 kHz), 600 ms endpointing
                                  │ echo gate: pause-probe + PCM correlation
                                  │ speaker gate: resemblyzer profile
                                  ▼
                     ASR   Whisper large-v3-turbo (MLX, local)   :8300
                     LLM   gpt-4o-mini via OpenRouter (streamed)
                     TTS   Kokoro-82M (local, 24 kHz)             :8300
                     tools 15 quick tools (home.py + local)
                     tasks sandboxed Hermes one-shot (~/sandbox)
```

The brain (VAD policy, gates, tool use, delegation) lives in one Pipecat `FrameProcessor`, so the same agent runs behind a mic (`make run`) or behind the benchmark's WebSocket transport (`make bench`) with zero code changes.

## Files

| File | Role |
|---|---|
| `pipecat_assistant.py` | the Pipecat agent — pipeline, gates, tools, delegation |
| `server.py` | Whisper + Kokoro FastAPI server (ASR/TTS on :8300) |
| `chat.py` | the original live full-duplex loop (wake word, DuplexGate) |
| `tasks.py` | sandboxed Hermes task manager (MAX_CONCURRENT = 2) |
| `home.py` | Spotify (AppleScript; free account) + smart-home state |
| `speakerid.py` | voice enrollment/similarity |
| `duplex.py` / `bench_duplex.py` | echo-aware pause/barge decision core + FDB-style sim |
| `ui_server.py` | dashboard on :8301 |
| `RESULTS.md` | frozen benchmark (run 16) + full progression |
| `docs/paper.md` | research-style write-up: architecture, evaluation, findings |

## Benchmarks

Evaluated with [ai231-me3-full-duplex-benchmarking](https://github.com/trueshotjames/ai231-me3-full-duplex-benchmarking) (Full-Duplex-Bench v1 sampled + reduced v3, 28 trials, GPT-4o judge). Frozen result — `make bench` reproduces it:

- pause handling 0% TOR (both synthetic and Candor) — never interrupts a speaker's pause
- interruption: 100% yield, judge score 1.00
- false turns: side-talk / TV / noise all rejected (1 background miss on n=1)
- v1.5 overlap: Resume 100% (pauses and comes back); backchannel kept talking 100%
- response latency 1.25–2.8 s

Full table, progression across five runs, and the addressed-speech-gate trade-off analysis: **[RESULTS.md](RESULTS.md)** and **[docs/paper.md](docs/paper.md)**.

## Notes

- Spotify playback is driven through the desktop app via AppleScript (Web API play is Premium-only).
- The voice profile (`voice_profile.npy`) is biometric and deliberately git-ignored.
- The Pi (voice front-end) half of the project deploys separately; this repo is the Mac executor.
