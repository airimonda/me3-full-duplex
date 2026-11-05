"""Echo check: Watson's own voice (delayed, quieter, noisy, at 16 kHz) must
score as echo; a different sentence must not. Uses the same 50 ms
playback-segment resampling path as WatsonAgent._play_pcm.
Needs server.py on :8300 (Kokoro). Run: ./venv/bin/python tests/test_echo.py"""
import io, os, sys, wave
import httpx, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipecat_assistant as pa

def tts(text):
    r = httpx.post(pa.SERVER + "/speak", json={"text": text}, timeout=60)
    r.raise_for_status()
    with wave.open(io.BytesIO(r.content)) as w:
        assert w.getframerate() == pa._OUT_SR, w.getframerate()
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

def played_reference(pcm24):
    """What _play_pcm accumulates: 50 ms segments resampled to 16 kHz."""
    ref = np.zeros(0, dtype=np.float32)
    for i in range(0, len(pcm24), 1200):
        seg = pa._resample(pcm24[i:i + 1200], pa._OUT_SR, pa._IN_SR)
        ref = np.concatenate([ref, seg])[-int(pa._ECHO_HIST_S * pa._IN_SR):]
    return ref

rng = np.random.default_rng(0)
reply = tts("Sure. The capital of Australia is Canberra, and it was chosen as a compromise between Sydney and Melbourne.")
other = tts("Hey Watson, can you turn off the kitchen lights please?")
ref = played_reference(reply[: int(2.5 * pa._OUT_SR)])        # 2.5 s played so far
reply16 = pa._resample(reply, pa._OUT_SR, pa._IN_SR)
other16 = pa._resample(other, pa._OUT_SR, pa._IN_SR)
fails = 0
for delay_ms in (50, 250, 600):
    end = int(2.5 * pa._IN_SR) - int(delay_ms / 1000 * pa._IN_SR)   # mic hears audio from delay_ms ago
    mic = reply16[end - 6400:end] * 0.3 + rng.normal(0, 200, 6400)  # 400 ms probe, -10 dB, noise
    s = pa.xcorr_echo_score(mic.astype(np.float32), ref)
    ok = s >= pa._ECHO_CORR
    fails += not ok
    print(f"echo  delay {delay_ms:4d} ms  score {s:.2f}  {'PASS' if ok else 'FAIL'} (want >= {pa._ECHO_CORR})")
mic = other16[3200:9600] * 0.8 + rng.normal(0, 200, 6400)
s = pa.xcorr_echo_score(mic.astype(np.float32), ref)
ok = s < pa._ECHO_CORR
fails += not ok
print(f"user  other sentence     score {s:.2f}  {'PASS' if ok else 'FAIL'} (want < {pa._ECHO_CORR})")
sys.exit(1 if fails else 0)
