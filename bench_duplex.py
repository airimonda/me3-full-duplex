"""Full-duplex bench — FDB-style scenarios against the DuplexGate decision core.

Simulated channel (deterministic, no room variance):
  mic(t) = ECHO_GAIN * agent_playback(t - ECHO_DELAY) + USER_GAIN * user(t)

Scenarios (per Full-Duplex-Bench v1 dimensions):
  interruption   : user barges mid-reply       -> agent MUST stop   (t_stop measured)
  backchannel    : brief "uh-huh" mid-reply    -> agent MUST continue
  background     : quiet 3rd-party speech      -> agent MUST continue
  silence        : no user at all              -> agent MUST NOT pause repeatedly

Sweeps DuplexParams and reports detection / false-barge / false-pause / t_stop.
Run: ~/me3/venv/bin/python bench_duplex.py
"""
import itertools
import io
import json
import os
import wave

import numpy as np
from duplex import DuplexGate, DuplexParams

SR = 16000
BLOCK = 512
ECHO_GAIN = 0.18
ECHO_DELAY_S = 0.024
BUFFER_MS = 120          # residual echo after pause (playback buffer)
USER_GAIN = 1.0


def load_wav(path):
    with wave.open(path, "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32767.0
        if w.getnchannels() == 2:
            pcm = pcm.reshape(-1, 2).mean(axis=1)
    return resample(pcm, sr)


def resample(pcm, sr_in, sr_out=SR):
    if sr_in == sr_out:
        return pcm
    dur = len(pcm) / sr_in
    return np.interp(np.linspace(0, dur, int(dur * sr_out), endpoint=False),
                     np.arange(len(pcm)) / sr_in, pcm).astype(np.float32)


def sim_scenario(agent_wav, user_wav=None, user_start_s=None, params=None, user_gain=1.0):
    """Clean event-driven simulation of one full-duplex exchange.

    Model:
      played[k]  = what the speaker actually output in block k (drains buffer_ms
                   after a pause, then silent; resumes at the frozen position)
      mic[k]     = ECHO_GAIN * played[k - delay_blocks] + user_gain * user[k]
      The gate sees only mic[] plus the reference agent[:pos] — exactly like live.

    Returns dict: interrupted, t_stop_ms, n_pauses.
    """
    p = params or DuplexParams()
    agent = resample(agent_wav, SR)
    user = resample(user_wav, SR) if user_wav is not None else np.zeros(0, np.float32)
    total_blocks = (len(agent) // BLOCK) + int(1.5 * SR / BLOCK)
    delay_blocks = int(round(ECHO_DELAY_S * SR / BLOCK))
    buffer_blocks = int(round(BUFFER_MS / 1000 * SR / BLOCK))

    gate = DuplexGate(p)
    gate.calibrate([np.zeros(int(SR * 0.25), np.float32)])

    played = [np.zeros(BLOCK, np.float32)] * delay_blocks   # pre-roll silence
    pos = 0
    paused = False
    drain = 0
    n_pauses = 0
    interrupted = False
    t_stop = None
    user_onset = None
    cooldown_until_block = -1
    probe = None           # (blocks_collected, deadline_block)

    for k in range(total_blocks):
        # 1. speaker output this block
        out = np.zeros(BLOCK, np.float32)
        if not paused:
            chunk = agent[pos:pos + BLOCK]
            out[:len(chunk)] = chunk
            pos += len(chunk)
        elif drain < buffer_blocks:
            chunk = agent[pos:pos + BLOCK]      # device buffer drains after pause
            out[:len(chunk)] = chunk
            pos += len(chunk)
            drain += 1
        played.append(out)

        # 2. mic = echo + user
        mic = ECHO_GAIN * played[k]
        blk = np.zeros(BLOCK, np.float32)
        if user_wav is not None:
            j = k * BLOCK - int((user_start_s or 0) * SR)
            if 0 <= j < len(user):
                seg = user[j:j + BLOCK]
                blk[:len(seg)] = user_gain * seg
                if np.abs(seg).max() > 0.02 and user_onset is None:
                    user_onset = k * BLOCK / SR
        mic_blk = mic + blk

        # 3. active probe: collect mic, decide at deadline
        if probe is not None:
            collected, deadline = probe
            collected.append(mic_blk)
            if k >= deadline:
                tail_n = max(1, int(p.persistence_ms / 1000 * SR / BLOCK))
                tail = collected[-tail_n:]
                tail_rms = float(np.sqrt(np.mean(np.concatenate(tail).astype(np.float64) ** 2)))
                gate.reset()
                probe = None
                n_pauses += 1
                if os.environ.get("BENCH_DEBUG"):
                    print(f"[decision@{k * BLOCK / SR:.2f}s] tail_rms={tail_rms:.4f} "
                          f"floor={gate.floor:.4f} bmin={p.barge_min_rms} "
                          f"user_onset={user_onset} user_gain={user_gain}")
                if tail_rms > max(gate.floor, p.barge_min_rms):
                    interrupted = True
                    t_stop = (k * BLOCK / SR - user_onset) * 1000 if user_onset is not None else None
                    break
                else:
                    paused = False
                    drain = 0
                    cooldown_until_block = k + int(p.resume_cooldown_ms / 1000 * SR / BLOCK)
            continue

        # 4. gate feeding (suppressed during resume cooldown)
        if k < cooldown_until_block:
            if k >= cooldown_until_block - 2:
                gate.reset()
            continue
        act = gate.feed(mic_blk, agent[:pos])
        if act and act[0] == "echo":
            gate.reset()
            continue
        if act and act[0] == "pause":
            paused = True
            drain = 0
            probe = ([act[1]], k + int(p.probe_ms / 1000 * SR / BLOCK))

    return {"interrupted": interrupted, "n_pauses": n_pauses,
            "t_stop_ms": round(t_stop) if t_stop else None}


def main():
    agent = load_wav(str(Path.home() / "me3") + "/tts_out_1.wav")   # ~5s agent reply
    barge = load_wav(str(Path.home() / "me3") + "/user_barge.wav")
    back = load_wav(str(Path.home() / "me3") + "/user_backchannel.wav")
    bg = load_wav(str(Path.home() / "me3") + "/user_bg.wav")

    scenarios = {
        "interruption": dict(user_wav=barge, user_start_s=1.0, expect=True),
        "backchannel": dict(user_wav=back, user_start_s=1.0, expect=False),
        "background": dict(user_wav=bg, user_start_s=1.0, expect=False),
        "silence": dict(user_wav=None, expect=False),
    }
    BG_GAIN = 0.25   # far-field third party — quieter than the user

    grid = list(itertools.product([2.5], [150], [0.7],
                                  [0.6, 0.7], [400, 600], [0.05, 0.07], [200], [0.02, 0.03]))
    rows = []
    for fm, pm, vt, ce, probe, bmr, em, pmr in grid:
        params = DuplexParams(floor_mult=fm, persistence_ms=pm, vad_threshold=vt,
                              corr_echo=ce, probe_ms=probe, barge_min_rms=bmr,
                              early_ms=em, pause_min_rms=pmr)
        r = {"floor_mult": fm, "persist_ms": pm, "vad": vt, "corr_echo": ce,
             "probe": probe, "barge_min_rms": bmr, "early_ms": em, "pause_min_rms": pmr}
        ok = True
        for name, kw in scenarios.items():
            gain = BG_GAIN if name == "background" else USER_GAIN
            out = sim_scenario(agent, params=params, user_gain=gain,
                               **{k: v for k, v in kw.items() if k != "expect"})
            expect = kw["expect"]
            if name == "interruption":
                r["interrupt"] = out["interrupted"]
                r["t_stop"] = out["t_stop_ms"]
                ok = ok and (out["interrupted"] == expect)
            else:
                false_barge = out["interrupted"]
                r[f"false_{name}"] = false_barge
                r[f"pauses_{name}"] = out["n_pauses"]
                ok = ok and (false_barge == expect)
        r["all_correct"] = ok
        rows.append(r)

    good = [x for x in rows if x["all_correct"]]
    print(f"{'floor':>5} {'persist':>7} {'vad':>4} {'ce':>4} {'probe':>5} {'bmin':>5} {'early':>5} | "
          f"{'intr?':>5} {'t_stop':>6} | {'bc-false':>8} {'bg-false':>8} {'sil-pauses':>10} | ok")
    for x in rows:
        print(f"{x['floor_mult']:>5} {x['persist_ms']:>7} {x['vad']:>4} {x['corr_echo']:>4} "
              f"{x['probe']:>5} {x['barge_min_rms']:>5} {x['early_ms']:>5} | "
              f"{str(x['interrupt']):>5} {str(x['t_stop']):>6} | "
              f"{str(x['false_backchannel']):>8} {str(x['false_background']):>8} "
              f"{x['pauses_silence']:>10} | {'YES' if x['all_correct'] else 'no'}")
    if good:
        best = min(good, key=lambda x: (x["t_stop"] or 9999, x["pauses_silence"]))
        print("\nBEST (all scenarios correct, lowest t_stop):", json.dumps(best))
        json.dump(rows, open(str(Path.home() / "me3") + "/bench_duplex_results.json", "w"), indent=2)
    else:
        print("\nNo config passed all scenarios — widen the grid.")


if __name__ == "__main__":
    main()