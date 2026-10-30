import numpy as np
import bench_duplex as B

bg = B.load_wav("${HOME}/me3/user_bg.wav")
barge = B.load_wav("${HOME}/me3/user_barge.wav")
agent = B.load_wav("${HOME}/me3/tts_out_1.wav")
print("bg rms:", np.sqrt(np.mean(bg**2)).round(4), "max:", np.abs(bg).max().round(3))
print("barge rms:", np.sqrt(np.mean(barge**2)).round(4))
print("agent rms:", np.sqrt(np.mean(agent**2)).round(4))
print("bg*0.25 rms:", (0.25*np.sqrt(np.mean(bg**2))).round(4))
