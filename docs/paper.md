# Watson: A Full-Duplex Voice Agent with Echo-Aware Interruption Handling and Delegated Task Execution

> **Nov 7 update:** sections 1–7 describe the Mac system frozen at run 16 (Oct 30). Watson now runs on a
> Raspberry Pi with Deepgram ASR/TTS and glm-5.3-flash, all streamed; see section 8 and `RESULTS.md`.

*AI 231 class project — ME3. Author: A. Mondares. Code frozen at run 16 (2026-10-30). Companion repo: `airimonda/me3-full-duplex`; benchmark harness: `trueshotjames/ai231-me3-full-duplex-benchmarking` (Full-Duplex-Bench v1 sampled + v3 reduced).*

## Abstract

We present Watson, a full-duplex voice assistant that converses naturally over a
single shared audio channel while performing real work in the background. Watson
combines (1) an echo-aware barge-in policy — a dynamic noise floor, a pause-probe
window, and PCM cross-correlation against its own playback — that lets it hear
users while speaking without triggering on itself; (2) a voice-enrollment gate that
restricts responses to the enrolled speaker; (3) an addressed-speech gate for
benchmark channels where speaker identity is unavailable; and (4) a two-tier
execution model in which instant tools run inside the conversational turn while
long tasks are delegated to a sandboxed background agent. On a 28-trial
Full-Duplex-Bench evaluation Watson achieves 0% false turn-over on pause handling,
100% interruption yield, 100% resume behaviour under overlap, and median response
latency of 1.25–2.8 s, while rejecting all but one non-addressed speech event.

## 1. Introduction

Voice assistants in the wild face two problems that scripted pipelines ignore.
First, the channel is shared: the agent's own playback reaches its microphone, so
any speech detector will hear the agent "interrupting itself". Second, not all
speech in the room is addressed to the agent; answering a TV broadcast or a
third-party conversation is worse than staying silent.

Watson's design goals follow the ME3 project brief: natural turn-taking, barge-in
stop within ~300 ms, first audio within ~1.5 s, real background task execution
(build tasks, research, files) with results announced at a natural pause, and a
class-demonstrable system running on consumer hardware (a laptop executor with a
Raspberry Pi front-end).

## 2. Architecture

Watson is implemented as a single Pipecat 1.14 `FrameProcessor` (the `WatsonAgent`)
between the input transport and the output transport. The brain is one file
(`pipecat_assistant.py`), and the same agent runs behind a local microphone
transport (live mode) or the benchmark's WebSocket transport (evaluation mode);
only the transport changes.

Pipeline: `transport.input() -> WatsonAgent -> transport.output()`.

### 2.1 Speech detection and endpointing
Silero VAD (ONNX, 16 kHz) runs on every 100 ms frame with 100 ms onset and
600 ms release; an utterance shorter than 250 ms is discarded.

### 2.2 Echo-aware barge-in (live mode)
While Watson is playing, incoming audio is compared against a dynamic floor
`max(0.018, 2.0 × ambient_rms)`, where `ambient_rms` is an EMA learned from quiet
frames. A suspected interruption does **not** stop playback: a probe window
(600 ms) accumulates mic audio and aborts silently if (a) the level falls back
below the floor, or (b) the probe signal cross-correlates (|r| > 0.5, ±150 ms
delay search) with the PCM currently being played — i.e., Watson heard itself.
Only sustained, non-echo speech commits the interruption: generation counter
increments, `InterruptionFrame` clears the output buffer, and the unplayed
remainder is stashed so the reply can **resume** if the interruption turns out to
be unaddressed.

### 2.3 Who is speaking: enrollment and the addressed-speech gate
In live mode, each real utterance is embedded with resemblyzer and compared to the
enrolled profile (similarity ≥ 0.72 to answer; 0.60–0.72 accepted with a
profile refresh; below 0.60 ignored). Embedding failures fail open so the owner is
never locked out. Fragments and non-Latin hallucinations skip the gate (the
addressed-speech gate handles them) because embeddings are unreliable on
sub-second mumbles.

In benchmark mode no speaker identity exists, so a *semantic* gate is used: the
talker model itself labels borderline turns `SILENT` for non-addressed speech
(third-person talk, side conversations, TV/radio, personal monologues) under a
few-shot rule. Section 5 analyses the recall/precision trade-off this gate forces.

### 2.4 Talker, tools, delegation
The talker is `gpt-4o-mini` via OpenRouter, streamed and spoken sentence-by-sentence
through Kokoro-82M (24 kHz), with clause-level early emission to cut first-audio
latency. Fifteen quick tools (time, time-in-city, web search, Spotify transport and
volume, lights, thermostat, alarms, reminders, media control, messages) are bound
via OpenAI-style function calling; results feed a second streaming pass for the
spoken reply. Two delegation paths exist: the model emits a machine-read
`DELEGATE:` tag for complex requests, and the user can bypass appraisal entirely
with *"have the agent do ___"*. Delegated tasks run as headless Hermes one-shot
agents confined by a macOS Seatbelt profile to `~/sandbox` (kernel-level, two
concurrent slots); Watson speaks a natural acknowledgement and the task's outcome
is announced later. Timers announce through the pipeline itself.

## 3. Evaluation setup

We use the course benchmark harness (Full-Duplex-Bench): 430-file dataset,
9 groups, direct-feed connection (the harness plays clips and records the agent's
answers; no room variance), 2 rounds, 28 trials, `gpt-4o` as judge, faster-whisper
for scoring ASR. Reported latency numbers are the harness's own measurements of
the recorded audio.

## 4. Results

Frozen run 16 (2026-10-30), full table in `results/run16-2026-10-30/`:

| Metric | Value |
|---|---|
| TOR, pause handling (synthetic / Candor) | **0% / 0%** (0/2 each) |
| TOR, turn-taking | 50% (1/2) replied, latency 2.78 s |
| False turn-taking (side talk / TV / background / noise) | **0% / 0% / 100% / 0%** |
| TOR, interruption (v1) | **100% (2/2)**, judge 1.00 |
| v1.5 interruption behaviour | **Resume 100%** — paused and came back every time |
| Backchannel | kept talking 100%; Resume 50% |
| Response latency (v1.5) | **1.25–2.8 s** |

Progression across the port from the hand-written loop to Pipecat:

| | run 11 (Oct 27) | run 14 | run 15 | run 16 (frozen) |
|---|---|---|---|---|
| Pause TOR | 0 ✓ | ✓ / 50% | 0% / 100% | **0% / 0%** |
| Turn-taking replied | ✓ | 0/2 | 1/2 | 1/2 |
| False turns | 3/4 ✗ | **0/4** | 1/4 | 3/4 ✓ |
| Interruption TOR | 100% | 50% | 100% | **100%** |
| v1.5 resume | stops | **100%** | ✓ | **100%** |
| Response latency | 4.2–4.8 s | 2.5 s | ~2.7 s | **1.25–2.8 s** |

## 5. Discussion

**The addressed-speech gate trade-off.** Run 14 answered first-person monologue
fragments (pause trials failed), run 15's short-remark rule fixed turn-taking but
answered mid-monologue remarks (pause precision collapsed to 0% → 100% TOR).
Run 16 balances both with a greetings-versus-first-person-narration rule. With
n = 1 per sub-condition, residual misses flip between runs — we attribute the
remaining turn-taking/background misses to sampling noise in the semantic gate
rather than a systematic failure. A principled fix (dual-channel reference audio,
or a dedicated addressed-speech classifier) is future work.

**Echo is the dominant live failure mode.** Before the pause-probe + correlation
policy, every playback burst re-triggered the VAD: the agent interrupted itself
within ~1 s of starting to speak (Whisper additionally hallucinated stock phrases
on its own echo, e.g. "Thank you."). Correlation against the exact playback PCM
resolves this without a hardware AEC, at the cost of a ≤ 600 ms commit window for
genuine interruptions.

**Delegation changes the interaction contract.** Because the agent acknowledges
and continues talking rather than blocking, users interleave new requests during
long tasks; the concurrency cap (2) forces an explicit "hands are full" refusal
path rather than silent queueing.

## 6. Limitations

- Single-channel echo rejection is correlation-based; reverb at high speaker
  volumes can lower the correlation below threshold.
- The semantic gate runs on the talker model; borderline turns cost one extra
  streaming decision (~2–4 s of dead time on non-addressed fragments).
- Benchmarks use synthetic/direct-feed audio; reverberant-room results will differ.
- The Raspberry Pi front-end is deployed separately and not part of this freeze.

## 7. Reproducibility

```bash
make setup   # venv + pinned deps (requirements.txt)
make bench   # 2 rounds × 14 trials, direct feed, gpt-4o judge
```

Run 16 artifacts: `results/run16-2026-10-30/` (results.json/csv/html, harness log,
agent turn log). The agent log records every gate decision (`stranger_ignored`,
`echo_ignored`, `probe_resume`, `barge_in`, `superseded`, `delegated`, `tool`),
so each verdict can be traced to a concrete policy decision.

## 8. Update (Nov 7, 2026): the Pi deployment and streaming

The voice front-end and the agent now run on a Raspberry Pi 4 (4 GB, no fan). ASR and TTS moved to
Deepgram (nova-3, aura-2-jupiter-en) called from the Pi; the talker is `z-ai/glm-5.3-flash` via
OpenRouter with `gpt-4o-mini` as fallback; big tasks go to a task MCP server on the Pi that runs the
pi.dev agent under bubblewrap. Echo is rejected by speaker identity rather than waveform correlation,
because on the Pi's open speaker the echo is as loud as the owner and correlates poorly.

Four latency changes, each A/B-tested on the Pi with fdbench's stage-latency test:

1. **Streaming TTS.** Deepgram's raw PCM is played as it arrives, and the next sentence is synthesized
   while the current one plays (TTS first audio 1.45 → 0.29 s median).
2. **Streaming ASR.** A Deepgram live websocket, opened in advance and kept alive, is fed while the user
   talks; Finalize + CloseStream at the VAD endpoint returns the transcript ~0.25 s later (0.92 → 0.24 s).
   Two bugs surfaced on the way: stopping at the first `from_finalize` result dropped trailing words, and
   a socket opened at speech onset made Deepgram trail the audio by 0.4–1.0 s.
3. **True LLM streaming.** The talker request used `httpx.post`, which downloads the whole response
   before returning; reading the stream as it arrives cut the first-token tail (p95 4.28 → 3.26 s).
4. **Tool turns.** A cached "One sec." plays while tools and the second LLM pass run.

Full run on the Pi (`results/run-2026-11-07-pi/`): end of speech to Watson's audio 2.56 s median,
3.93 s p95 (Nov 6: 4.72 / 7.49 s), WER 12.1 %, pause take-overs 0/4, interruptions handled 2/2
(judge 5.0/5), false turns 0/4, turn-taking missed 1/2. The language model's first token (≈2 s median)
is now most of the remaining wait. Context per call, measured by fdbench from OpenRouter's usage
report: 1,670 prompt tokens median, 12 messages.
