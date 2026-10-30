"""Speaker identity for ME3 — enroll the first voice, then only recognize it.

resemblyzer's GE2E encoder: embed_utterance returns a 256-d embedding;
same-speaker cosine ~0.75-0.95, different-speaker ~0.2-0.5 on clean audio.
Threshold default 0.65 (env ME3_VOICE_SIM).
"""
import os

import numpy as np

_SIM_THRESHOLD = float(os.environ.get("ME3_VOICE_SIM", "0.72"))
# near-miss zone: likely the owner under different mic conditions — accept and
# refresh the profile (self-healing) instead of locking them out.
_SIM_NEAR = float(os.environ.get("ME3_VOICE_NEAR", "0.60"))
_enc = None
PROFILE_PATH = os.path.expanduser("~/me3/voice_profile.npy")


def save_profile(emb):
    np.save(PROFILE_PATH, emb)


def load_profile():
    """Returns the persisted embedding, or None."""
    if os.path.exists(PROFILE_PATH):
        try:
            emb = np.load(PROFILE_PATH)
            if emb.shape == (256,):
                return emb
        except Exception:
            pass
    return None


def _encoder():
    global _enc
    if _enc is None:
        from resemblyzer import VoiceEncoder
        _enc = VoiceEncoder()   # CPU, ~small model, loads once
    return _enc


def voiced_seconds(pcm16k):
    """Seconds of actual voice energy (for judging enrollment-clip quality)."""
    if pcm16k is None or not len(pcm16k):
        return 0.0
    amp = np.abs(pcm16k)
    idx = np.where(amp > 0.012)[0]
    if not len(idx):
        return 0.0
    return (idx[-1] - idx[0]) / 16000.0


def embed(pcm16k):
    """pcm16k: float32 mono 16 kHz. Returns 256-d unit embedding, or None if too short."""
    if pcm16k is None or len(pcm16k) < 16000 * 0.3:   # <0.3s is unusable
        return None
    try:
        # trim leading/trailing near-silence so embeddings focus on the voice
        amp = np.abs(pcm16k)
        idx = np.where(amp > 0.012)[0]
        if len(idx) < 4800:      # <0.3s of actual voice
            return None
        seg = pcm16k[max(0, idx[0] - 1600):min(len(pcm16k), idx[-1] + 1600)]
        seg = seg / (np.abs(seg).max() + 1e-9)
        return _encoder().embed_utterance(seg.astype(np.float32))
    except Exception as e:
        print(f"[voice] embed failed: {e}", flush=True)
        return None


def similarity(pcm16k, enrolled):
    e = embed(pcm16k)
    if e is None or enrolled is None:
        return -1.0
    return float(np.dot(e, enrolled) / (np.linalg.norm(e) * np.linalg.norm(enrolled) + 1e-9))


def is_same_voice(pcm16k, enrolled):
    return similarity(pcm16k, enrolled) >= _SIM_THRESHOLD


def threshold():
    return _SIM_THRESHOLD


def near_threshold():
    return _SIM_NEAR