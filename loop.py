"""ME3 voice loop (Mac dev version) — VAD -> ASR -> talker -> TTS -> playback.

Two modes:
  python loop.py --file audio.wav          # deterministic file test, saves reply wav
  python loop.py --mic                     # live mic (silero VAD segments speech)

Per-stage latency is printed and saved to latency_log.json:
  speech_end -> transcript_ready -> first_token -> first_audio

The talker is a small fast OpenRouter model with a short-reply system prompt.
"""
import argparse
import io
import json
import os
import sys
import time
import wave

import httpx
import numpy as np

SERVER = "http://127.0.0.1:8300"
TALKER_MODEL = os.environ.get("ME3_TALKER_MODEL", "openai/gpt-4o-mini")

SYSTEM_PROMPT = (
    "You are a voice assistant. Reply in ONE or TWO short sentences, plain text, "
    "no markdown, no lists. If the user asks for something that needs real work "
    "(building things, research, files), reply exactly 'Let me check that.' plus a "
    "one-line restatement, and nothing else."
)


def load_openrouter_key():
    path = os.path.expanduser("~/.hermes/.env")
    with open(path) as f:
        for line in f:
            if line.startswith("OPENROUTER_API_KEY="):
                return line.strip().split("=", 1)[1]
    sys.exit("OPENROUTER_API_KEY not found in ~/.hermes/.env")


# one shared client for the whole app: keeps TCP/TLS connections warm
HTTP = httpx.Client(timeout=httpx.Timeout(60, connect=5))


def talker(text, key, stream_cb=None):
    """One LLM call. If stream_cb, calls it per token and returns (text, first_token_dt)."""
    t0 = time.perf_counter()
    first_dt = None
    chunks = []
    if True:
        with HTTP.stream(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": TALKER_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "max_tokens": 120,
                "stream": True,
            },
        ) as resp:
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    if first_dt is None:
                        first_dt = time.perf_counter() - t0
                    chunks.append(delta)
                    if stream_cb:
                        stream_cb(delta)
    return "".join(chunks).strip(), first_dt


def speak_http(text):
    t0 = time.perf_counter()
    r = HTTP.post(f"{SERVER}/speak", json={"text": text})
    first = time.perf_counter() - t0
    wav_bytes = r.content
    return wav_bytes, first, time.perf_counter() - t0


def transcribe_full(wav_bytes):
    """Returns the full /transcribe payload: text + latency + confidence fields."""
    t0 = time.perf_counter()
    r = HTTP.post(f"{SERVER}/transcribe", content=wav_bytes,
                  headers={"Content-Type": "audio/wav"})
    return r.json(), time.perf_counter() - t0


def transcribe_http(wav_bytes):
    data, dt = transcribe_full(wav_bytes)
    return data.get("text", ""), dt


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32767.0
        if w.getnchannels() == 2:
            pcm = pcm.reshape(-1, 2).mean(axis=1)
    return pcm, sr


def resample_linear(pcm, sr_in, sr_out=16000):
    if sr_in == sr_out:
        return pcm
    dur = len(pcm) / sr_in
    n_out = int(dur * sr_out)
    return np.interp(np.linspace(0, dur, n_out, endpoint=False),
                     np.arange(len(pcm)) / sr_in, pcm).astype(np.float32)


def wav_to_bytes(pcm, sr=16000):
    buf = io.BytesIO()
    with wave.open(buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((pcm * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def run_pipeline(user_audio_pcm, sr_in, key, save_reply=None):
    log = {}
    pcm16 = resample_linear(user_audio_pcm, sr_in)
    wav_bytes = wav_to_bytes(pcm16, 16000)

    t_speech_end = time.perf_counter()
    text, asr_dt = transcribe_http(wav_bytes)
    log["transcript_ready_s"] = round(asr_dt, 3)
    log["transcript"] = text
    print(f"[asr] {asr_dt:.3f}s  transcript: {text!r}", flush=True)
    if not text:
        print("[loop] empty transcript, dropping", flush=True)
        return None

    reply_parts, first_audio_dt = [], None

    def on_token(tok):
        nonlocal first_audio_dt
        # clause-chunked TTS: fire on first sentence end
        if first_audio_dt is None and any(c in tok for c in ".!?,"):
            reply_parts.append(tok)
            wav, first, _ = speak_http("".join(reply_parts))
            first_audio_dt = time.perf_counter() - t_speech_end - asr_dt
            log["first_audio_s"] = round(first_audio_dt, 3)
            if save_reply:
                with open(save_reply, "wb") as f:
                    f.write(wav)
            print(f"[tts] first chunk after {first_audio_dt:.3f}s from speech end", flush=True)

    reply, first_tok_dt = talker(text, key, stream_cb=on_token)
    log["first_token_s"] = round(first_tok_dt, 3)
    log["reply"] = reply
    print(f"[llm] first token {first_tok_dt:.3f}s  reply: {reply!r}", flush=True)

    if first_audio_dt is None:
        wav, _, total = speak_http(reply)
        first_audio_dt = time.perf_counter() - t_speech_end
        log["first_audio_s"] = round(first_audio_dt, 3)
        if save_reply:
            with open(save_reply, "wb") as f:
                f.write(wav)
    total = time.perf_counter() - t_speech_end
    log["total_s"] = round(total, 3)
    print(f"[loop] total {total:.3f}s", flush=True)
    return log


def mic_mode(key):
    import sounddevice as sd
    from silero_vad import load_silero_vad, get_speech_timestamps

    sr = 16000
    model = load_silero_vad()
    print("[mic] listening (ctrl-c to stop)...", flush=True)
    buffer = []
    collected = None

    def callback(indata, frames, t, status):
        nonlocal buffer
        buffer.append(indata[:, 0].copy())

    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", callback=callback,
                        blocksize=512):
        while True:
            buf = np.concatenate(buffer) if buffer else np.zeros(0, dtype=np.float32)
            if len(buf) >= sr * 1.0:  # evaluate VAD once per second of audio
                ts = get_speech_timestamps(buf, model, sampling_rate=sr,
                                           min_speech_duration_ms=250,
                                           min_silence_duration_ms=400)
                if ts:
                    speech = np.concatenate([buf[s["start"]:s["end"]] for s in ts])
                    if len(speech) > sr * 0.4:
                        collected = speech
                        buffer = []
                        break
                if len(buf) > sr * 30:
                    buffer = buf[-sr * 5:]
                else:
                    buffer = [buf]  # flatten next loop
            time.sleep(0.2)
    log = run_pipeline(collected, sr, key, save_reply="reply.wav")
    if log:
        import sounddevice as sd2
        wav = read_wav("reply.wav")
        sd2.play(wav[0], wav[1])
        sd2.wait()
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--mic", action="store_true")
    args = ap.parse_args()
    key = load_openrouter_key()
    if args.file:
        pcm, sr = read_wav(args.file)
        log = run_pipeline(pcm, sr, key, save_reply="reply.wav")
    elif args.mic:
        log = mic_mode(key)
    else:
        ap.error("need --file or --mic")
    with open("latency_log.json", "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()