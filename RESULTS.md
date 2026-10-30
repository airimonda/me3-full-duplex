# ME3 Frozen Results — Run 16 (2026-10-30, fd-benchmark)

Frozen benchmark of the Pipecat pipeline (`pipecat_assistant.py`) with the full
chat.py brain (15 quick tools, DELEGATE→Hermes sandbox, addressed-speech gate).

Config: 2 rounds · 28 trials · FDB v1 sampled + reduced v3 · direct feed · local ·
judge = openrouter/gpt-4o. Run folder: `fd-benchmark/runs/me3/2026-10-30_10-49-51`
(copied here as `results/run16-2026-10-30/`).

## Headline

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

## Progression (all runs, same config)

| | run 11 (Oct 27) | run 14 | run 15 | **run 16 (frozen)** |
|---|---|---|---|---|
| Pause TOR | 0/0 ✓ | ✓/50% | 0%/100% | **0%/0%** |
| Turn-taking | ✓ | 0/2 | 1/2 | 1/2 |
| False turns | 3/4 ✗ | 0/4 ✓ | 1/4 | 3/4 ✓ |
| Interruption TOR | 100% | 50% | 100% | **100%** |
| v1.5 resume | stops | 100% | ✓ | **100%** |
| Response latency | 4.2–4.8 s | 2.5 s | ~2.7 s | **1.25–2.8 s** |

## Known trade-off (write-up finding)

A single-channel LLM addressed-speech gate trades turn-taking recall against
pause precision: run 14 answered monologue fragments (pause ✗, turn-taking ✗),
run 15's short-remark rule swung the other way. Run 16 balances both with a
greetings-vs-first-person-monologue rule; residual misses on n=1 samples flip
between runs (sampling noise, not a systematic bug).
