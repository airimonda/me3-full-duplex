import os
os.environ["BENCH_DEBUG"] = "1"
import bench_duplex as B
from duplex import DuplexParams

agent = B.load_wav(str(Path.home() / "me3") + "/tts_out_1.wav")
bg = B.load_wav(str(Path.home() / "me3") + "/user_bg.wav")
p = DuplexParams(floor_mult=2.5, persistence_ms=150, vad_threshold=0.7,
                 corr_echo=0.6, probe_ms=600, barge_min_rms=0.05, early_ms=200,
                 pause_min_rms=0.02)
out = B.sim_scenario(agent, user_wav=bg, user_start_s=1.0, params=p, user_gain=0.25)
print(out)
