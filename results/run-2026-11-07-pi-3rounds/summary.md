# Full-Duplex Voice Agent on a Raspberry Pi

Student me3 · run 2026-11-07_11-22-33 · 2026-11-07

## 5 · Evaluation & Benchmark

| Metric | Definition | Target | Mean | p95 | n |
|---|---|---|---|---|---|
| E2E latency | end of speech → audio out | < 4.00 s | 3.20 s | 6.27 s | 30 |
| ASR final | end of speech → transcript | < 0.80 s | 0.34 s | 0.42 s | 30 |
| LLM TTFT | transcript → first token | < 1.50 s | 2.48 s | 4.78 s | 30 |
| TTS TTFB | text → first audio chunk | < 0.60 s | 0.43 s | 1.45 s | 19 |
| Barge-in | speech → playback stopped | < 0.80 s | 0.77 s | 0.79 s | 3 |
| WER | ASR accuracy on test set | < 15 % | 10.6 % | — | 30 |

- One clock: the assistant machine's monotonic clock; every stage timestamped (VAD end → ASR final → LLM 1st token → TTS 1st audio → speaker)
- E2E = speaker − VAD end; barge-in = stop − speech start
- 10 phrases × 3 runs; mean + p95; JSON log per run: stage_latency.json (per turn: trials/<id>/stages.jsonl)

## Results (deck 2)

Latency chart: latency_chart.svg

- RPi model / RAM: Raspberry Pi 4 Model B Rev 1.1 · 4 GB
- Cooling: not stated · peak 74 °C · throttled never
- Network: cloud: Wi-Fi · benchmark link: Wi-Fi, 16 ms round trip
- Date: 2026-11-07
- Model versions: harness pipecat 1.15.0, ASR Deepgram nova-3 (cloud), LLM z-ai/glm-5.3-flash (OpenRouter); fallback openai/gpt-4o-mini, TTS Deepgram aura-2-jupiter-en (cloud), VAD Silero VAD (Pipecat SileroVADAnalyzer), Harness Pipecat custom FrameProcessor (WatsonAgent)

## 3 · Context engineering: context size per run

| Context size | Median | Max | First → last call |
|---|---|---|---|
| Prompt tokens per LLM call (estimate, usage) | 1665 | 2107 | 1463 → 1696 |
| Messages in context | 12 | 15 | 2 → 12 |
