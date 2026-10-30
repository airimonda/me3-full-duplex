# Step 1 (Oct 26) — Mac-side results

Machine: MacBook, Apple M5, 16GB RAM, macOS 27.2.
Project dir: ~/me3 (venv at ~/me3/venv, Python 3.11).

## Whisper (mlx-whisper, model mlx-community/whisper-large-v3-turbo)

- Model load + warmup: 102 s (first run only; includes 800MB HF download; server keeps it resident)
- Per-utterance latency (cold, right after load): 0.390 / 0.403 / 0.432 s
- Per-utterance latency (warm): 0.382 / 0.406 / 0.434 s
- Transcripts all correct (3 short test utterances, 16 kHz mono wav).
- Verdict: within the 250-400 ms budget at the upper edge. Note: these are `say`-generated
  clips (clean TTS audio); real mic audio may differ slightly. Server must pre-load the model
  at startup, never per request.

## Kokoro (kokoro pypi package, voice af_heart, 24 kHz)

- Model load: 1.74 s (warm; weights cached from first run)
- First audio chunk (steady state): 0.50 s / 0.48 s on the two longer texts (~5 s of audio
  generated in the same wall time — synthesis is faster than real-time)
- First utterance after a fresh process start was 2.9 s (includes voice-model download/warmup) —
  warm the pipeline at server startup.
- Verdict: ~0.5 s to first chunk is above the 150-300 ms target. Tuning knobs (Step 4): split
  text into shorter clauses before synthesis, and/or generate the first clause while the LLM
  is still streaming the rest.
- Output samples: ~/me3/tts_out_0.wav, tts_out_1.wav, tts_out_2.wav (verify by ear).

## Environment notes

- Installed via uv: mlx-whisper, kokoro, soundfile, espeak-ng (brew), en_core_web_sm 3.8.0
  (direct wheel — `spacy download` targets the wrong interpreter on this machine; always use
  `uv pip install --python ~/me3/venv/bin/python <wheel>`).
- Timing scripts: ~/me3/time_whisper.py, ~/me3/time_kokoro.py; raw data in
  whisper_timing.json, kokoro_timing.json.

## Still open on the Pi (your hands, not mine)

- arecord mic test, aplay speaker test
- Silero VAD (ONNX runtime) on live mic, log speech start/end
- 10-15 min test recordings (quiet / noisy / speaker playing)
- Static IP or phone-hotspot setup for the Pi→Mac link
