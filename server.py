"""ME3 Mac-side model server: Whisper ASR + Kokoro TTS, both warm.

Endpoints (HTTP, for the Pi's asr-mcp / tts-mcp proxies):
  GET  /health          -> {"status": "ok", "asr": bool, "tts": bool}
  POST /transcribe      -> raw WAV bytes in body (16k mono recommended)
                           returns {"text": "...", "latency_s": ...}
  POST /speak           -> JSON {"text": "..."} (optionally "voice")
                           returns audio/wav bytes (24 kHz mono)
Run:  ~/me3/venv/bin/python ~/me3/server.py   (listens on 0.0.0.0:8300)
"""
import io
import os
import json
import time
import wave

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response
from kokoro import KPipeline

# Watson's voice: af_nicole (user choice, Nov 6). The speaker check tells
# Watson's own echo from the owner by voice, so keep this far from the owner.
# 2026-11-06 scan against the Pi profile (resemblyzer sim): am_michael 0.387,
# am_onyx 0.388, am_eric 0.402, af_nicole 0.427 (safest female), af_heart
# 0.556, af_jessica 0.626. Must match pipecat_assistant.py's TTS_VOICE default.
TTS_VOICE = os.environ.get("ME3_TTS_VOICE", "af_nicole")
from mlx_whisper import transcribe

ASR_MODEL = "mlx-community/whisper-large-v3-turbo"
SAMPLE_RATE = 24000

app = FastAPI()
asr_ready = False
tts_ready = False


@app.on_event("startup")
def load_models():
    global pipe
    t0 = time.perf_counter()
    # warm ASR with a silent 0.5s buffer
    silence = np.zeros((8000,), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, silence, 16000, format="WAV")
    transcribe(buf.getvalue(), path_or_hf_repo=ASR_MODEL, language="en")
    t1 = time.perf_counter()
    pipe = KPipeline(lang_code="a")
    # warm TTS with a short utterance
    for _ in pipe("Hello.", voice=TTS_VOICE):
        pass
    print(f"[server] ASR warm in {t1-t0:.1f}s, TTS warm in {time.perf_counter()-t1:.1f}s", flush=True)
    global asr_ready, tts_ready
    asr_ready = tts_ready = True


@app.get("/health")
def health():
    return {"status": "ok", "asr": asr_ready, "tts": tts_ready}


@app.post("/transcribe")
async def transcribe_ep(request: Request):
    body = await request.body()
    t0 = time.perf_counter()
    audio, sr = sf.read(io.BytesIO(body), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    out = transcribe(audio, path_or_hf_repo=ASR_MODEL, language="en")
    dt = time.perf_counter() - t0
    segs = out.get("segments") or []
    conf = None
    if segs:
        no_speech = sum(s.get("no_speech_prob", 0) for s in segs) / len(segs)
        avg_logprob = sum(s.get("avg_logprob", 0) for s in segs) / len(segs)
        conf = round(1.0 - no_speech, 3)
        return {"text": out["text"].strip(), "latency_s": round(dt, 3),
                "no_speech": round(no_speech, 3), "avg_logprob": round(avg_logprob, 3),
                "conf": conf}
    return {"text": out["text"].strip(), "latency_s": round(dt, 3)}


PRECACHE = {
    "ack_0": "On it — I'm handing that to my agent.",
    "ack_1": "My agent is taking that one now.",
    "ack_2": "I've passed that to my agent — give it a moment.",
    "ack_3": "Handing that off to my agent; I'll tell you when it's done.",
    "ack_4": "Okay, my agent is starting that for you.",
    "clarify": "I can start on that, but it's a bit vague. Could you tell me a little more about what you'd like?",
    "apology_2": "Sorry, I'm already working on 2 things in the background. Let me finish those before I take on more. I'll let you know as each one lands.",
    "apology_busy": "Sorry, my hands are full in the background right now. I'll come back to this one as soon as a slot frees up.",
    "go_ahead": "Go ahead — I'm listening.",
    "hello": "Hi there! How's your day going?",
}


def _wav_bytes(audio_all):
    buf = io.BytesIO()
    with wave.open(buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((audio_all * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


_precache_wavs = {}


@app.on_event("startup")
def warm_precache():
    global _precache_wavs
    for key, text in PRECACHE.items():
        try:
            frames = [a for _, _, a in pipe(text, voice=TTS_VOICE)]
            _precache_wavs[key] = _wav_bytes(np.concatenate(frames))
        except Exception as e:
            print(f"[precache] {key} failed: {e}", flush=True)
    print(f"[precache] {len(_precache_wavs)}/{len(PRECACHE)} phrases ready", flush=True)


@app.get("/speak_cache/{key}")
def speak_cache(key: str):
    if key not in _precache_wavs:
        return Response(json.dumps({"error": "unknown key"}), status_code=404,
                        media_type="application/json")
    return Response(_precache_wavs[key], media_type="audio/wav",
                    headers={"X-Synthesis-Latency": "cached"})


@app.post("/speak")
def speak_ep(payload: dict):
    text = payload.get("text", "").strip()
    voice = payload.get("voice", TTS_VOICE)
    if not text:
        return Response(json.dumps({"error": "empty text"}), status_code=400,
                        media_type="application/json")
    t0 = time.perf_counter()
    frames = []
    for _, _, audio in pipe(text, voice=voice):
        frames.append(audio)
    audio_all = np.concatenate(frames) if frames else np.zeros((1,), dtype=np.float32)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((audio_all * 32767).astype(np.int16).tobytes())
    dt = time.perf_counter() - t0
    return Response(
        wav_buf.getvalue(),
        media_type="audio/wav",
        headers={"X-Synthesis-Latency": f"{dt:.3f}", "X-Audio-Seconds": f"{len(audio_all)/SAMPLE_RATE:.2f}"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8300, log_level="warning")