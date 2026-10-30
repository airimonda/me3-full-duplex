"""Sanity: same-voice vs different-voice similarity with our clips."""
import speakerid
import bench_duplex as B

agent = B.load_wav("${HOME}/me3/tts_out_1.wav")     # agent's voice
barge = B.load_wav("${HOME}/me3/user_barge.wav")    # user voice
back = B.load_wav("${HOME}/me3/user_backchannel.wav")

emb = speakerid.embed(barge)
print("enroll (user_barge):", "ok" if emb is not None else "FAILED")
print("same clip again:", round(speakerid.similarity(barge, emb), 3))
print("other user clip (backchannel):", round(speakerid.similarity(back, emb), 3))
print("agent voice (TTS):", round(speakerid.similarity(agent, emb), 3))
print("threshold:", speakerid.threshold())