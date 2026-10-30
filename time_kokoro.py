import time, wave, json
import numpy as np
from kokoro import KPipeline

t0 = time.perf_counter()
pipe = KPipeline(lang_code="a")  # american english
load_s = time.perf_counter() - t0
print(f"model load: {load_s:.2f}s", flush=True)

texts = [
    "What time is it right now?",
    "Let me check that for you. I will start building the site and let you know when it is ready.",
    "Set a timer for twenty seconds, and remind me to check the oven when it goes off.",
]
results = []
for i, text in enumerate(texts):
    t = time.perf_counter()
    first_chunk_s = None
    frames = []
    sr = 24000
    for gs, ps, audio in pipe(text, voice="af_heart"):
        if first_chunk_s is None:
            first_chunk_s = time.perf_counter() - t
        frames.append(audio)
    total_s = time.perf_counter() - t
    audio_all = np.concatenate(frames)
    wav_path = f"${HOME}/me3/tts_out_{i}.wav"
    with wave.open(wav_path, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((audio_all * 32767).astype(np.int16).tobytes())
    results.append({"i": i, "first_chunk_s": round(first_chunk_s, 3), "total_s": round(total_s, 3),
                    "audio_len_s": round(len(audio_all)/sr, 2), "text": text})
    print(f"utt{i}: first chunk {first_chunk_s:.3f}s, total {total_s:.3f}s, audio {len(audio_all)/sr:.2f}s", flush=True)

json.dump(results, open("${HOME}/me3/kokoro_timing.json", "w"), indent=2)
