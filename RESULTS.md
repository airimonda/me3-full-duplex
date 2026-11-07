# ME3 Results

## Latest — Nov 7, 2026: Watson on the Pi, streaming ASR/LLM/TTS

Presented run (the deck's numbers): fdbench run `2026-11-07_11-22-33` (copied to
`results/run-2026-11-07-pi-3rounds/`): Watson on the Raspberry Pi 4, harness on the Mac
(`--where remote`), 3 rounds of Full-Duplex-Bench (30 trials) + 30 timed turns (10 phrases x 3),
judge openrouter/gpt-4o. Code: `main` after PR #5.

The earlier 2-round run `2026-11-07_10-10-53` (code after PR #3, copied to
`results/run-2026-11-07-pi/`) is kept for comparison.

| Metric | Nov 6 (run 23-27-10) | Nov 7 (run 10-10-53, 2 rounds) | Nov 7 (run 11-22-33, 3 rounds) |
|---|---|---|---|
| E2E, end of speech → audio out (median / mean / p95) | 4.72 / 5.22 / 7.49 s | 2.56 / 2.67 / 3.93 s | **2.57 / 3.20 / 6.27 s** |
| ASR final (median / p95) | 0.92 / 1.67 s | 0.24 / 0.30 s | 0.28 / 0.42 s |
| LLM first token (median / p95) | 2.10 / 4.28 s | 1.99 / 3.26 s | 2.02 / 4.78 s |
| TTS first audio (median / p95) | 1.45 / 3.25 s | 0.29 / 0.40 s | 0.30 / 1.45 s |
| Barge-in, playback stopped (mean / p95) | not measured | 0.75 / 0.76 s | 0.77 / 0.79 s |
| WER | 11.1 % | 12.1 % | 10.6 % |
| Pause take-overs | 1/4 | 0/4 | 1/6 |
| Turn-taking missed | 1/2 | 1/2 | 1/3 |
| Answered speech not meant for it | 1/4 | 0/4 | 2/6 (side talk; TV, noise, background 0) |
| Interruptions handled | 0/1 | 2/2, judge 5.0/5 | 3/3, judge 5.0/5 |
| Backchannel resume | 1/2 | 1/2 | 2/3 |
| Context per LLM call | not reported | 1,670 prompt tokens median (max 1,994), 12 messages | 1,665 prompt tokens median (max 2,107), 12 messages (max 15) |
| Pi peak temperature | 68 °C | 73 °C, never throttled | 74 °C, never throttled |

The two Nov 7 runs have the same E2E median. The 3-round run has a longer tail: a few turns had
slow LLM first tokens (TTFT p95 4.78 s), which moves the E2E p95 from 3.93 s to 6.27 s.

What changed, each step checked first with a stage-latency-only run (10 phrases × 2) on the Pi:

| Step | Cheap-run E2E median |
|---|---|
| Nov 7 baseline (whole-WAV Deepgram TTS) | 5.41 s |
| + streaming TTS (play PCM as it arrives) | 4.29 s |
| + streaming ASR (live websocket), tool ack, empty-stream fallback | 2.79 s |
| + Deepgram HTTPS warm-up | 2.67 s |
| + pre-connected spare ASR socket (fixes late/truncated transcripts) | 2.14 s |
| + 400 ms pre-roll on barge-in turns (WER fix; LLM was slow this run, TTFT median 2.30 s) | 2.82 s |
| + true LLM streaming (`http.post` had buffered the whole reply) | 2.00 s |

LLM first-token time varies a lot between runs (cheap-run medians 1.49–2.30 s), so read single steps
with care; the full-run table above is the reference.

Bugs the benchmarks caught on the way: live ASR dropped trailing words (stopped at the first
`from_finalize`; now waits for `Metadata`), lost first words on barge-in turns (bench pre-roll), and
stale Deepgram keep-alive connections (warm-up + one retry). n = 2 per Full-Duplex-Bench group: single
trials flip between runs.

---

## Run 16 (2026-10-30, Mac, fd-benchmark) — earlier freeze

Frozen benchmark of the Pipecat pipeline (`pipecat_assistant.py`) with the full
chat.py brain (15 quick tools, DELEGATE→Hermes sandbox, addressed-speech gate).

Config: 2 rounds · 28 trials · FDB v1 sampled + reduced v3 · direct feed · local ·
judge = openrouter/gpt-4o. Run folder: `fd-benchmark/runs/me3/2026-10-30_10-49-51`
(copied here as `results/run16-2026-10-30/`).

### Headline

| Metric | Value |
|---|---|
| TOR, pause handling (synthetic) | **0% (0/2)** — stays silent through pauses |
| TOR, pause handling (Candor) | **0% (0/2)** |
| TOR, turn-taking | 50% (1/2) replied |
| Missed turn-taking rate | 50% (1/2) |
| False turn-taking rate | **25% (1/4)** — side talk ✓, TV ✓, noise ✓; background ✗ |
| TOR, interruption | **100% (2/2)**, judge score 1.00 |
| v1.5 interruption behaviour | **Resume 100%** (paused & came back 100%) |
| Backchannel | kept talking 100%, Resume 50% |
| Response latency (v1.5) | **1.25–2.8 s** (was 4.2–4.8 s pre-port) |

### Progression (all runs, same config)

| | run 11 (Oct 27) | run 14 | run 15 | **run 16 (frozen)** |
|---|---|---|---|---|
| Pause TOR | 0/0 ✓ | ✓/50% | 0%/100% | **0%/0%** |
| Turn-taking | ✓ | 0/2 | 1/2 | 1/2 |
| False turns | 3/4 ✗ | 0/4 ✓ | 1/4 | 3/4 ✓ |
| Interruption TOR | 100% | 50% | 100% | **100%** |
| v1.5 resume | stops | 100% | ✓ | **100%** |
| Response latency | 4.2–4.8 s | 2.5 s | ~2.7 s | **1.25–2.8 s** |

### Known trade-off (write-up finding)

A single-channel LLM addressed-speech gate trades turn-taking recall against
pause precision: run 14 answered monologue fragments (pause ✗, turn-taking ✗),
run 15's short-remark rule swung the other way. Run 16 balances both with a
greetings-vs-first-person-monologue rule; residual misses on n=1 samples flip
between runs (sampling noise, not a systematic bug).
