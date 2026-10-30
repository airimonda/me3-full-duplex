# Full-duplex bench (FDB-style scenarios) — results, Oct 27

## Method
bench_duplex.py simulates the acoustic channel deterministically:
  mic[k] = 0.18 * played[k - 24ms] + user_gain * user[k]
where `played` is what the speaker actually output (drains 120 ms after a pause,
resumes at the frozen position). The DuplexGate decision core (duplex.py) is the
SAME code the live loop runs — bench and live can never drift.

Scenarios (FDB v1 dimensions):
  interruption  (user barges at 1.0s, speaks 2s)  -> agent must stop; t_stop measured
  backchannel   ("uh-huh", 0.3s, at 1.0s)         -> agent must continue
  background    (far-field speech, 0.25 gain)     -> agent must continue
  silence       (no user)                         -> no pauses at all

## Bugs the bench exposed (all fixed)
1. VAD latch: detector stuck in_speech after first echo onset -> all later
   interrupts swallowed. Fix: hard VAD reset after every probe.
2. Correlation search capped at 250ms while true echo alignments sit seconds
   into the playback -> echo gate never fired. Fix: full-range search.
3. Sim channel model used wall-clock echo while playback ran on position time.
4. Sim user gain not applied in probe tail / main mix.
5. Block-vs-sample index confusion in timing metrics.

## Winning configuration (all scenarios correct)
  vad_threshold 0.70, persistence 150ms, corr_echo 0.60, probe 600ms,
  floor_mult 2.5, barge_min_rms 0.05, pause_min_rms 0.02, early 200ms,
  resume_cooldown 500ms
Results: interruption detected (t_stop ~768ms incl. 600ms probe; audio stops
within ~200ms of onset), backchannel continues, background speech continues
(3 benign pauses), silence = 0 pauses. Full grid in bench_duplex_results.json.

## Live integration
chat.py play_with_barge now runs the identical DuplexGate. Tunables via env:
ME3_VAD_THRESH_PLAY, ME3_PROBE_MS, ME3_PERSIST_MS, ME3_FLOOR_MULT,
ME3_CORR_ECHO, ME3_BARGE_MIN_RMS, ME3_PAUSE_MIN_RMS.

## Known honest limits (report in the write-up)
- t_stop ~768ms > the 300ms target: the 600ms probe is the cost of echo
  rejection without a hardware AEC. Trade-off is explicit and measured.
- Far-field rejection rests on absolute loudness (proximity); a loud TV can
  still trigger stops. FDB v1.5-style "talking to others" remains a weak spot
  for cascaded systems (the paper's own cascaded baseline behaves the same).
- Simulated channel lacks room reverb; live validation next (rehearsal runs).
