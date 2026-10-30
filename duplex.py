"""Full-duplex decision core — single source of truth for live loop AND bench.

The gate's job while the agent speaks: distinguish USER speech from ECHO of the
agent's own playback, and decide pause/resume/stop.

Mechanism (pause-and-resume):
  - VAD onset during playback -> pause the agent's audio immediately
  - echo decays once its source stops; a live user persists
  - after probe_ms, decide by persistence: mean RMS of the trailing
    persistence_ms window > calibrated floor -> BARGE (stop for real);
    otherwise it was echo -> RESUME playback from the same position.

All tunables live in DuplexParams; the bench sweeps them, chat.py uses the
winning values via env vars.
"""
import os
from dataclasses import dataclass

import numpy as np
from silero_vad import load_silero_vad, VADIterator


@dataclass
class DuplexParams:
    vad_threshold: float = 0.70      # VAD sensitivity during playback (bench winner)
    probe_ms: int = 600              # pause-probe length
    persistence_ms: int = 150        # tail window that decides barge vs echo
    floor_mult: float = 2.5          # floor = max(floor_mult * ambient, min_floor)
    min_floor: float = 0.006
    max_floor: float = 0.05
    min_silence_ms: int = 120        # VAD end threshold inside detector
    corr_echo: float = 0.50          # burst correlation above which it is pure echo (reverb-tolerant)
    early_ms: int = 200              # burst length before the early correlation check
    barge_min_rms: float = 0.05      # absolute loudness required to confirm a barge
    pause_min_rms: float = 0.035     # bursts quieter than this never pause playback (echo)
    resume_cooldown_ms: int = 900    # after a resume, ignore pauses this long (replay echo)
    echo_over_factor: float = 1.6    # bursts must exceed learned echo level by this to pause

    @classmethod
    def from_env(cls):
        return cls(
            vad_threshold=float(os.environ.get("ME3_VAD_THRESH_PLAY", "0.70")),
            probe_ms=int(os.environ.get("ME3_PROBE_MS", "600")),
            persistence_ms=int(os.environ.get("ME3_PERSIST_MS", "150")),
            floor_mult=float(os.environ.get("ME3_FLOOR_MULT", "2.5")),
            corr_echo=float(os.environ.get("ME3_CORR_ECHO", "0.50")),
            barge_min_rms=float(os.environ.get("ME3_BARGE_MIN_RMS", "0.05")),
            pause_min_rms=float(os.environ.get("ME3_PAUSE_MIN_RMS", "0.035")),
            resume_cooldown_ms=int(os.environ.get("ME3_RESUME_COOLDOWN", "900")),
        )


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2))) if len(a) else 0.0


_MODEL = None


def _vad_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_silero_vad()
    return _MODEL


class DuplexGate:
    """State machine fed 16 kHz mono chunks. Live loop and bench share this."""

    def __init__(self, params: DuplexParams, sr=16000):
        self.p = params
        self.sr = sr
        self.echo_rms = None   # learned loudness of our own echo (survives resets)
        self._new_vad()
        self.floor = params.min_floor

    def _new_vad(self):
        self.vad = VADIterator(_vad_model(), sampling_rate=self.sr,
                               threshold=self.p.vad_threshold,
                               min_silence_duration_ms=self.p.min_silence_ms)
        self.in_speech = False
        self.burst = []
        self.early_done = False

    def note_echo(self, tail_rms):
        """Record the measured level of confirmed self-echo (from a probe that
        resumed). Later bursts quieter than echo_rms * echo_over_factor are echo."""
        self.echo_rms = max(self.echo_rms or 0.0, tail_rms)

    def reset(self):
        """Hard-reset VAD state after every pause-probe. CRITICAL: without this,
        the detector latches on the echo's first 'start' event and every later
        user interruption is swallowed (no new 'start' ever fires)."""
        self._new_vad()

    def calibrate(self, ambient_chunks):
        """Set the noise floor from a quiet sample taken just before playback."""
        if ambient_chunks:
            a = rms(np.concatenate(ambient_chunks))
            self.floor = min(max(self.p.floor_mult * a, self.p.min_floor), self.p.max_floor)
        return self.floor

    def feed(self, chunk, ref_played=None):
        """Feed one mic chunk while the agent is speaking.
        ref_played: agent audio up to the current playback position (16 kHz).

        Returns:
          ('echo', corr)        -> burst is pure echo of our own voice; caller resets
          ('pause', burst, corr)-> burst may contain user speech; caller pauses and
                                   runs probe() immediately (mid-speech, not at end)
          None                  -> keep feeding

        Early check: after ~200ms of burst, correlate what we have so far. Pure
        echo is discarded WITHOUT pausing (no stutter); anything else pauses at
        ~200ms into the speech onset — early enough to matter."""
        ev = self.vad(chunk, return_seconds=False)
        if not self.in_speech and ev and "start" in ev:
            self.in_speech = True
            self.burst = [chunk]
            self.early_done = False
            return None
        if self.in_speech:
            self.burst.append(chunk)
            burst = np.concatenate(self.burst)
            if not self.early_done and len(burst) >= int(self.sr * self.p.early_ms / 1000):
                self.early_done = True
                corr = xcorr_echo_score(burst, ref_played, self.sr) if ref_played is not None else 0.0
                burst_rms = rms(burst)
                # echo if: correlated with our own audio, quieter than the absolute
                # echo floor, or quieter than our LEARNED echo level (live-adaptive)
                echo_ceiling = self.p.pause_min_rms
                if self.echo_rms is not None:
                    echo_ceiling = max(echo_ceiling, self.echo_rms * self.p.echo_over_factor)
                if corr >= self.p.corr_echo or burst_rms < echo_ceiling:
                    self.in_speech = False
                    self.burst = []
                    return ("echo", corr)
                return ("pause", burst, corr)
            if ev and "end" in ev:
                self.in_speech = False
                self.burst = []
                if not self.early_done:
                    corr = xcorr_echo_score(burst, ref_played, self.sr) if ref_played is not None else 0.0
                    if corr >= self.p.corr_echo:
                        return ("echo", corr)
                    return ("pause", burst, corr)
        return None

    def probe(self, mic_iter, burst):
        """Run the pause-probe. mic_iter yields mic chunks (16 kHz) during the probe.
        Returns (barge: bool, tail_rms: float, collected: list[chunk])."""
        collected = [burst]
        tail = [burst]
        tail_len = int(self.p.persistence_ms / 1000 * self.sr)
        deadline = self.p.probe_ms / 1000
        t = 0.0
        while t < deadline:
            try:
                chunk = next(mic_iter)
            except StopIteration:
                break
            dur = len(chunk) / self.sr
            t += dur
            collected.append(chunk)
            tail.append(chunk)
            while sum(len(c) for c in tail) > tail_len + len(tail[0]):
                tail.pop(0)
        tail_audio = np.concatenate(tail)
        tail_rms = rms(tail_audio)
        barge = tail_rms > self.floor and tail_rms > self.p.barge_min_rms
        return (barge, tail_rms, collected)


def xcorr_echo_score(candidate, reference, sr=16000):
    """Max normalized cross-correlation of candidate vs reference over a delay search.
    The search covers the WHOLE reference: the true alignment sits at the burst's
    absolute position in the playback, which can be seconds in."""
    n = min(len(candidate), len(reference), sr)   # cap candidate at 1s
    if n < sr // 10:
        return 0.0
    c = candidate[:n] - candidate[:n].mean()
    best = 0.0
    limit = len(reference) - n
    for d in range(0, max(limit, 0) + 1, 80):
        r = reference[d:d + n]
        r = r - r.mean()
        denom = (np.linalg.norm(c) * np.linalg.norm(r)) + 1e-9
        best = max(best, float(np.dot(c, r) / denom))
    return best