"""Endpointing sweep: for each silence budget, measure when VAD closes the turn.

Method: take a sample utterance, append known trailing silence, run VADIterator
exactly like chat.py does, and record the time from true speech end to the
'end' event. Repeat with a mid-utterance pause sample to check false turn-takes.

Run: ~/me3/venv/bin/python sweep_endpoint.py
"""
import io
import os
import wave

import numpy as np
from silero_vad import load_silero_vad, VADIterator

SR = 16000
BLOCK = 512
BUDGETS = [300, 400, 500, 700]


def load_wav(path):
    with wave.open(path, "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32767.0
        if w.getnchannels() == 2:
            pcm = pcm.reshape(-1, 2).mean(axis=1)
    assert sr == SR, f"{path} is {sr}Hz, need {SR}"
    return pcm


def run_vad(pcm, min_silence_ms):
    vad = VADIterator(load_silero_vad(), sampling_rate=SR, threshold=0.5,
                      min_silence_duration_ms=min_silence_ms)
    events = []
    for i in range(0, len(pcm) - BLOCK, BLOCK):
        ev = vad(pcm[i:i + BLOCK], return_seconds=False)
        if ev:
            events.append((i / SR, ev))
    return events


def speech_end(pcm, thresh=0.02):
    """Last sample index above an energy floor (true speech end estimate)."""
    win = SR // 100
    energies = [np.abs(pcm[i:i + win]).mean() for i in range(0, len(pcm) - win, win)]
    idx = [i for i, e in enumerate(energies) if e > thresh]
    return (idx[-1] * win / SR) if idx else 0.0


def sweep(name, pcm):
    true_end = speech_end(pcm)
    print(f"\n== {name} (speech ends at {true_end:.2f}s, file {len(pcm)/SR:.2f}s)")
    rows = []
    for b in BUDGETS:
        events = run_vad(pcm, b)
        ends = [t for t, ev in events if "end" in ev]
        starts = [t for t, ev in events if "start" in ev]
        if not ends:
            rows.append((b, None, len(starts)))
            continue
        # first end-event AFTER true speech end = the turn-close
        close = next((t for t in ends if t >= true_end), None)
        rows.append((b, close, len(ends)))
        lag = (close - true_end) if close else None
        print(f"  budget {b}ms: turn closed at {f'{close:.2f}s' if close else 'NEVER'} "
              f"-> endpoint lag {f'{lag:.2f}s' if lag else 'n/a'} "
              f"({len(ends)} end events total)")
    return rows


def main():
    # sample 1: plain utterance + natural trailing silence
    pcm1 = load_wav(os.path.expanduser("~/me3/test1.wav"))
    pcm1 = np.concatenate([pcm1, np.zeros(int(SR * 1.5), dtype=np.float32)])
    sweep("plain utterance + 1.5s silence", pcm1)

    # sample 2: two utterances with a mid pause (false turn-take test)
    pcm2a = load_wav(os.path.expanduser("~/me3/test1.wav"))
    pcm2b = load_wav(os.path.expanduser("~/me3/test2.wav"))
    for pause in (0.5, 0.8):
        gap = np.zeros(int(SR * pause), dtype=np.float32)
        combined = np.concatenate([pcm2a, gap, pcm2b])
        sweep(f"two utterances with {pause:.1f}s pause", combined)


if __name__ == "__main__":
    main()