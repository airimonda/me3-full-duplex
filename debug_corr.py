import numpy as np
from duplex import xcorr_echo_score
import bench_duplex as B

agent = B.load_wav("${HOME}/me3/tts_out_1.wav")
SR = 16000
d = int(0.024 * SR)

echo = 0.18 * agent              # mic[k] = 0.18 * agent[k - d]
mic_start = int(0.51 * SR)       # mic index where burst starts
burst = echo[mic_start:mic_start + 3200]
content_start = mic_start - d    # agent index of burst content
ref = agent[:mic_start + 3200 + d]   # what the sim passes: agent[:pos+d]
print("corr vs sim-ref:", xcorr_echo_score(burst, ref))
print("corr vs aligned :", xcorr_echo_score(burst, agent[content_start:content_start+3200]))
print("content_start:", content_start, "ref len:", len(ref))
