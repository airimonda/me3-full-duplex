import time, json, sys
from mlx_whisper import transcribe

MODEL = "mlx-community/whisper-large-v3-turbo"
t0 = time.perf_counter()
# warm-up load
transcribe("${HOME}/me3/test1.wav", path_or_hf_repo=MODEL, language="en")
load_s = time.perf_counter() - t0
print(f"model load+warmup: {load_s:.2f}s", flush=True)

results = []
for f in ["test1", "test2", "test3"]:
    path = f"${HOME}/me3/{f}.wav"
    t = time.perf_counter()
    out = transcribe(path, path_or_hf_repo=MODEL, language="en", word_timestamps=False)
    dt = time.perf_counter() - t
    results.append({"file": f, "latency_s": round(dt, 3), "text": out["text"].strip()})
    print(f"{f}: {dt:.3f}s  {out['text'].strip()!r}", flush=True)

# repeat run to show warm steady-state
for f in ["test1", "test2", "test3"]:
    path = f"${HOME}/me3/{f}.wav"
    t = time.perf_counter()
    out = transcribe(path, path_or_hf_repo=MODEL, language="en")
    dt = time.perf_counter() - t
    results.append({"file": f + "_warm", "latency_s": round(dt, 3), "text": out["text"].strip()})
    print(f"{f} warm: {dt:.3f}s", flush=True)

json.dump(results, open("${HOME}/me3/whisper_timing.json", "w"), indent=2)
