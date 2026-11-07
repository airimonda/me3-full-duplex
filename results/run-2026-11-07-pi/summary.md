# Full-Duplex Voice Agent on a Raspberry Pi

Student me3 · run 2026-11-07_10-10-53 · 2026-11-07

## 5 · Evaluation & Benchmark

| Metric | Definition | Target | Mean | p95 | n |
|---|---|---|---|---|---|
| E2E latency | end of speech → audio out | < 4.00 s | 2.67 s | 3.93 s | 30 |
| ASR final | end of speech → transcript | < 0.80 s | 0.26 s | 0.30 s | 30 |
| LLM TTFT | transcript → first token | < 1.50 s | 2.11 s | 3.26 s | 30 |
| TTS TTFB | text → first audio chunk | < 0.60 s | 0.31 s | 0.40 s | 18 |
| Barge-in | speech → playback stopped | < 0.80 s | 0.73 s | 0.76 s | 3 |
| WER | ASR accuracy on test set | < 15 % | 12.1 % | — | 30 |

- One clock: the assistant machine's monotonic clock; every stage timestamped (VAD end → ASR final → LLM 1st token → TTS 1st audio → speaker)
- E2E = speaker − VAD end; barge-in = stop − speech start
- 10 phrases × 3 runs; mean + p95; JSON log per run: stage_latency.json (per turn: trials/<id>/stages.jsonl)

## Results (deck 2)

Latency chart: latency_chart.svg

- RPi model / RAM: Raspberry Pi 4 Model B Rev 1.1 · 4 GB
- Cooling: not stated · peak 73 °C · throttled never
- Network: cloud: Wi-Fi · benchmark link: Wi-Fi, 5 ms round trip
- Date: 2026-11-07
- Model versions: harness pipecat 1.15.0, ASR Deepgram nova-3 (cloud), LLM z-ai/glm-5.3-flash (OpenRouter); fallback openai/gpt-4o-mini, TTS Deepgram aura-2-jupiter-en (cloud), VAD Silero VAD (Pipecat SileroVADAnalyzer), Harness Pipecat custom FrameProcessor (WatsonAgent)

## 3 · Context engineering: context size per run

| Context size | Median | Max | First → last call |
|---|---|---|---|
| Prompt tokens per LLM call (estimate, usage) | 1670 | 1994 | 1442 → 1712 |
| Messages in context | 12 | 15 | 2 → 13 |
