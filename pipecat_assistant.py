"""ME3 Watson as a Pipecat pipeline (docs/GUIDE.md step 4B contract).

build(transport, log_dir) -> (run_coroutine, cancel)

Pipeline:  transport.input() -> WatsonAgent -> transport.output()

WatsonAgent (single FrameProcessor, keeps ME3's architecture and timings):
  - Silero VAD (onnx, 16 kHz) on every incoming frame; endpointing 600 ms
  - utterance -> Whisper on the Mac  (POST http://127.0.0.1:8300/transcribe)
  - talker: gpt-4o-mini via OpenRouter, streamed; sentences split for TTS
  - TTS: Kokoro on the Mac (POST /speak) -> OutputAudioRawFrame 24 kHz
  - barge-in: user speech while playing -> InterruptionFrame downstream
    (pipecat_fdbench turns that into {"event": "clear"} for fdbench)
  - turn/latency log -> log_dir/watson_log.jsonl

Server-side services are unchanged (server.py on 127.0.0.1:8300 must be up).
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import sys
import time
import wave
from pathlib import Path

import httpx
import numpy as np
from pipecat.processors.frame_processor import FrameProcessor

ME3 = Path(__file__).resolve().parent
VAD_MODEL = ME3 / "fd-benchmark" / "models" / "silero_vad.onnx"
SERVER = os.environ.get("ME3_SERVER", "http://127.0.0.1:8300")
TALKER_MODEL = os.environ.get("ME3_TALKER_MODEL", "z-ai/glm-5.3")

# ASR / TTS backends: "local" (server.py on the Mac) or "dgx3" (class vLLM GPU server)
ME3_ASR = os.environ.get("ME3_ASR", "local")   # local | dgx3
ME3_TTS = os.environ.get("ME3_TTS", "local")   # local | dgx3
DGX3 = os.environ.get("DGX3_HOST", "")         # e.g. http://10.x.x.x; empty = local backends only
DGX3_ASR_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
DGX3_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def _dgx3_key() -> str | None:
    v = os.environ.get("DGX3_API_KEY")
    if v:
        return v
    p = Path.home() / ".hermes" / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith("DGX3_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    return None

_PROMPT_BASE = (
    "You are Watson, a voice assistant in a live spoken conversation. Reply in TWO or "
    "THREE full sentences (casual acknowledgements may be one sentence), plain text, no "
    "markdown, no lists, no emoji. Never reply with a single word or a bare question. "
    "If asked about song lyrics, poems, jokes or quotes, answer from your own knowledge — "
    "never say you cannot. Never mention tools or systems."
)
_PROMPT_GATE = (
    " ADDRESSED-SPEECH GATE (most important rule): you hear everything in the room, "
    "including TV, radio and other people's conversations. Reply with exactly the single "
    "word SILENT and nothing else ONLY when you have positive evidence the speech is "
    "not directed at you. SILENT ONLY for: TV or radio content; a line addressed to "
    "someone else by name ('Gianna, this coffee is legit amazing' -> SILENT; 'Officer, "
    "is this the right form to fill' -> SILENT; 'Rita, pass the soy sauce' -> SILENT); "
    "clear two-person side talk where one person tells another to do something; talk "
    "about you in the third person; phone calls or dictation not meant for the room. "
    "DEFAULT TO ANSWER: when one person speaks in a natural one-on-one cadence — a "
    "direct question, a request, a command, a greeting, a direct social remark, or a "
    "statement or story told as if to a listener — treat it as ADDRESSED TO YOU even "
    "without a wake word or name. ('What time is it' -> ANSWER; 'How long should I "
    "steep green tea' -> ANSWER; 'Remind me to call the dentist' -> ANSWER; 'I'm bored "
    "with my exercise routine' -> ANSWER; 'I've been thinking about changing my career "
    "path' -> ANSWER; 'Nice to meet you' -> ANSWER). A solo speaker sharing a thought "
    "or story with a listener is still addressed to you — do NOT call it a monologue "
    "and do NOT say SILENT for it. When unsure between ANSWER and SILENT, choose "
    "ANSWER."
)
# mic mode: voice enrollment already rejects strangers/TV, so the SILENT gate
# is redundant there and kept eating real commands ("build me a Mario game").
# Benchmark mode keeps the gate (overheard clips, no voice check possible).
SYSTEM_PROMPT = _PROMPT_BASE + _PROMPT_GATE
SYSTEM_PROMPT_MIC = _PROMPT_BASE

# filler-only utterances and Whisper's classic hallucinations on short quiet
# bursts / TV audio (ported from chat.py)
FILLERS = {"um", "uh", "mhm", "hmm", "uh-huh", "huh", "mm", "yeah", "okay", "ok"}
WHISPER_GHOSTS = {
    "thank you", "thank you very much", "thanks", "thanks for watching",
    "thanks for watching my video", "see you next time",
    "hello", "hi", "hey", "you", "so", "and", "the", "a", "i", "it",
    "mmhmm", "amen", "and you", "you know", "i know", "my pleasure",
    "go to see you in the next video", "see you in the next video",
    "wharton", "watch on", "button",
}

def _token_set(s):
    return set(re.findall(r"[a-z0-9']+", s.lower()))

_IN_SR = 16000
_OUT_SR = 24000
# USB PnP mics only accept 48k directly via PortAudio; capture at 48k and
# downsample to 16k before VAD/ASR (no-op when the mic supports 16k).
_MIC_SR = int(os.environ.get("ME3_MIC_SR", "48000"))


def _to_in_sr(samples: np.ndarray) -> np.ndarray:
    if _MIC_SR == _IN_SR:
        return samples
    n = len(samples) * _IN_SR // _MIC_SR
    return np.interp(np.linspace(0, len(samples) - 1, n, endpoint=False),
                     np.arange(len(samples)), samples).astype(np.int16)
_ENDPOINT_MS = 600
_MIN_UTT_MS = 250
_BARGE_MS = 150
_SPEAK_THRESH = 0.5


def _openrouter_key() -> str | None:
    v = os.environ.get("OPENROUTER_API_KEY")
    if v:
        return v
    p = Path.home() / ".hermes" / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    return None


def _wav_bytes(pcm16: bytes, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16)
    return buf.getvalue()


class Silero:
    """Silero VAD over onnxruntime, 512-sample chunks at 16 kHz."""

    def __init__(self, model_path: Path):
        import onnxruntime
        self.sess = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)

    def prob(self, chunk: np.ndarray) -> float:
        x = chunk.astype(np.float32) / 32768.0
        out, self.state = self.sess.run(
            None, {"input": x[None, :], "state": self.state, "sr": np.array(_IN_SR, dtype=np.int64)})
        return float(out[0][0])


class WatsonAgent(FrameProcessor):
    """The whole ME3 loop as one Pipecat FrameProcessor."""

    def __init__(self, log_dir: Path):
        FrameProcessor.__init__(self)
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.vad = SileroVADAnalyzer(sample_rate=_IN_SR, params=VADParams(
            confidence=0.5, start_secs=0.1, stop_secs=0.6, min_volume=0.004))
        self.vad.set_sample_rate(_IN_SR)
        self.buf = bytearray()
        self.in_speech = False
        self.silence_ms = 0.0
        self.speech_ms = 0.0
        self.playing = False
        self.barge_ms = 0.0
        self.play_task = None
        self.respond_task = None   # in-flight _respond coroutine (for supersede)
        self.key = _openrouter_key()
        self.dgx_key = _dgx3_key()
        self._bad_logged = 0
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=20,
                                                            write=30, pool=10))
        self.gen = 0  # generation counter: invalidate stale playback after barge-in
        self.resume_pcm = None   # unplayed remainder of an interrupted reply (raw PCM)
        self.barged = False
        self.ambient_rms = 0.01  # dynamic noise floor (EMA of quiet frames)
        self.probing = False     # suspected barge: confirming before committing
        self.probe_ms = 0.0
        self.probe_rms_peak = 0.0
        self.probe_pcm = bytearray()   # mic audio captured during the probe
        self.play_ref = None           # pcm currently being played (echo reference)
        self.play_pos = 0              # samples of play_ref already pushed
        # speaker enrollment: same profile file as the live loop (chat.py)
        sys.path.insert(0, str(Path.home() / "me3"))
        import speakerid
        import chat as chatbrain      # QUICK_TOOLS + execute_tool (single source of truth)
        import tasks as taskmgr       # sandboxed Hermes delegation
        self.chatbrain = chatbrain
        self.taskmgr = taskmgr
        self.tools = chatbrain.QUICK_TOOLS
        self.mode = os.environ.get("ME3_MODE", "bench")
        # chat.py's full prompt (tools + delegation rules) in both modes; the
        # addressed-speech gate paragraph is bench-only (mic uses voice enrollment)
        sysprompt = chatbrain.SYSTEM_PROMPT + (_PROMPT_GATE if self.mode == "bench" else "")
        self.history = [{"role": "system", "content": sysprompt}]
        self.spk = speakerid
        self.speaker = speakerid.load_profile()
        # the resemblyzer encoder takes ~100 s to load — warm it in the
        # background so the first gated turn doesn't freeze
        import threading
        threading.Thread(target=self.spk._encoder, daemon=True).start()

    # ---- audio path ------------------------------------------------------
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        from pipecat.frames.frames import InputAudioRawFrame
        from pipecat.processors.frame_processor import FrameDirection
        if isinstance(frame, InputAudioRawFrame) and frame.sample_rate == _MIC_SR:
            await self._on_audio(_to_in_sr(np.frombuffer(frame.audio, dtype=np.int16)))
        else:
            await self.push_frame(frame, direction)

    async def _on_audio(self, samples: np.ndarray):
        from pipecat.audio.vad.vad_analyzer import VADState
        from pipecat.frames.frames import InterruptionFrame
        from pipecat.processors.frame_processor import FrameDirection
        pcm = samples.tobytes()
        state = await self.vad.analyze_audio(pcm)
        ms = len(samples) / _IN_SR * 1000
        rms = float(np.sqrt(np.mean((samples.astype(np.float32) / 32768.0) ** 2)) + 1e-9)
        floor = max(0.018, self.ambient_rms * 2.0)

        if self.playing:
            # probes do NOT pause playback — audio keeps flowing; only a
            # confirmed (non-echo, sustained) interruption stops it
            if self.probing:
                self.probe_pcm.extend(pcm)
                self.probe_ms += ms
                self.probe_rms_peak = max(self.probe_rms_peak, rms)
                echo = self._echo_score(np.frombuffer(self.probe_pcm, dtype=np.int16))
                if echo > 0.5:
                    # Watson's own voice coming back — abort silently
                    self.probing = False
                    self.probe_pcm = bytearray()
                    self.log_event("echo_ignored", echo=round(echo, 2))
                elif state == VADState.QUIET and rms < floor:
                    self.probing = False
                    self.probe_pcm = bytearray()
                    self.log_event("probe_resume", probe_ms=round(self.probe_ms, 1))
                elif self.probe_ms >= 600:
                    # sustained speech that is NOT echo: commit the interruption
                    self.probing = False
                    self.probe_pcm = bytearray()
                    self.gen += 1
                    self.playing = False
                    self.buf = bytearray()
                    self.in_speech = True
                    self.barged = True
                    await self.push_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
                    self.log_event("barge_in", rms=round(self.probe_rms_peak, 3))
            elif state != VADState.QUIET and rms > floor:
                # suspect a barge: confirm over the probe window first
                self.probing = True
                self.probe_ms = 0.0
                self.probe_rms_peak = rms
                self.probe_pcm = bytearray(pcm)  # start the probe buffer with this frame
            else:
                if state == VADState.QUIET:
                    self.ambient_rms = 0.9 * self.ambient_rms + 0.1 * rms
                self.barge_ms = max(0.0, self.barge_ms - ms)
            return

        if state != VADState.QUIET:
            self.buf.extend(pcm)
            self.in_speech = True
            self.speech_ms += ms
            self.silence_ms = 0.0
        elif self.in_speech:
            # STOPPING -> QUIET: the analyzer already consumed the 600 ms tail
            utt = bytes(self.buf)
            self.buf = bytearray()
            self.in_speech = False
            self.silence_ms = 0.0
            speech = self.speech_ms
            self.speech_ms = 0.0
            if speech >= _MIN_UTT_MS and len(utt) > _IN_SR * 2 * _MIN_UTT_MS / 1000:
                asyncio.create_task(self._respond(utt))
            self.dbg_max = 0.0
        else:
            # idle: keep learning the ambient noise floor
            self.ambient_rms = 0.9 * self.ambient_rms + 0.1 * rms

    # ---- brain -----------------------------------------------------------
    def _set_timer_local(self, secs: int, label: str) -> str:
        """Timer that announces itself through THIS pipeline (chat.py's version
        speaks through its own player, which we don't have here)."""
        async def fire():
            await asyncio.sleep(max(1, secs))
            try:
                wav = await self._speak(f"Time's up — that's your {label}.")
                if wav and self.gen == self.gen:  # gen check inside _play
                    self.playing = True
                    await self._play(wav, self.gen)
                    self.playing = False
            except Exception:
                pass
        asyncio.create_task(fire())
        nice = (f"{secs // 60} minute{'s' if secs // 60 != 1 else ''}"
                if secs >= 60 else f"{secs} seconds")
        return f"Timer set for {nice} — {label}."

    async def _delegate(self, rest: str, gen: int, user_text: str = "") -> str:
        """Submit the task to the sandboxed Hermes agent; speak a natural ack."""
        try:
            running = len(self.taskmgr.list_running())
        except Exception:
            running = 0
        if running >= self.taskmgr.MAX_CONCURRENT:
            ack = ("Sorry, my hands are full in the background right now. "
                   "I'll come back to this one as soon as a slot frees up.")
            self.log_event("delegate_busy", task=rest[:80])
        else:
            try:
                tid = await asyncio.to_thread(self.taskmgr.submit_task, rest)
            except Exception as e:
                # task backend unavailable (e.g. no sandbox-exec on the Pi):
                # never crash the turn — acknowledge honestly instead
                self.log_event("delegate_failed", err=repr(e)[:120],
                               task=rest[:80])
                tid = None
            if tid:
                ack = ("Handing that to my agent now. I'll let you know "
                       "when it's done.")
            else:
                ack = "Sorry, I couldn't start that task."
            self.log_event("delegated", user=user_text, task=rest[:80],
                           tid=str(tid)[:40])
        wav = await self._speak(ack)
        if wav and self.gen == gen:
            self.playing = True
            await self._play_pcm(_wav_pcm(wav), gen)
            self.playing = False
        return ack

    async def _respond(self, utt_pcm: bytes):
        t0 = time.perf_counter()
        gen = self.gen
        # volume gate: TV/radio clips sit far below live speech level
        peak = float(np.max(np.abs(np.frombuffer(utt_pcm, dtype=np.int16)))) / 32768.0
        if peak < 0.10:
            self.log_event("ignored_quiet", peak=round(peak, 3))
            return
        from pipecat.frames.frames import OutputAudioRawFrame
        text = await self._asr(utt_pcm)
        t_asr = time.perf_counter()
        if not text or len(text.strip()) < 2:
            return
        self.log_event("asr", text=text, asr_s=round(t_asr - t0, 3))
        # ghost filter: Whisper's classic hallucinations on short bursts / TV
        # audio — drop before they count as speech or touch the voice profile
        norm = re.sub(r"[^\w\s]", "", text.lower()).strip()
        no_wake = not norm.startswith("watson")
        if no_wake and norm in WHISPER_GHOSTS:
            self.log_event("ghost_ignored", user=text)
            return
        if no_wake and _token_set(text) <= FILLERS:
            self.log_event("filler_ignored", user=text)
            return
        # wake-word handling: bare "Watson" -> greet; "Watson, ..." -> strip the
        # wake word and ALWAYS answer (never let the SILENT gate eat a command)
        low = text.strip().lower().strip(".,!? ")
        if low in ("watson", "watson?"):
            wav = await self._speak("Yes? I'm listening.")
            if wav:
                self.playing = True
                await self._play(wav, gen)
            self.playing = False
            self.log_event("greeted", user=text)
            return
        # speaker gate: same voice profile the live loop uses.
        # Normalize first — the raw mic level here is low and embed()'s
        # silence-trim would otherwise discard the whole utterance.
        no_gate = self.mode == "bench" or low.startswith("watson")
        if no_gate and low.startswith("watson") and len(low) > 6:
            text = text.strip()[7:].strip(" ,.!?") or text.strip()
        norm = np.frombuffer(utt_pcm, dtype=np.int16).astype(np.float32)
        norm /= (np.abs(norm).max() + 1e-9)
        if self.mode == "mic":
            # fragments/gibberish embed unreliably — don't voice-gate them (the
            # SILENT gate handles them); only gate real sentences
            words = text.strip().split()
            fragment = len(words) <= 2 or not all(ord(c) < 0x2E80 for c in text)
            if fragment:
                self.log_event("fragment_skipped_gate", user=text)
                # mic mode: a fragment ("we", "okay") must not spawn a response —
                # it would supersede the in-flight reply and kill it
                return
            elif self.speaker is not None:
                sim = self.spk.similarity(norm, self.speaker)
                if sim < 0.0:
                    # embed failed (too quiet/short): fail OPEN, never lock out
                    self.log_event("voice_gate_unavailable", user=text)
                elif sim < self.spk.near_threshold():
                    self.log_event("stranger_ignored", user=text, sim=round(sim, 3))
                    return
                elif sim < self.spk.threshold():
                    # near-miss: accept, but only refresh the profile from
                    # wake-prefixed commands (definitely the owner speaking)
                    if no_gate:
                        emb = self.spk.embed(norm)
                        if emb is not None and self.spk.voiced_seconds(norm) >= 2.0:
                            self.spk.save_profile(emb)
                            self.log_event("voice_refreshed", sim=round(sim, 3))
            elif (low.startswith("watson") and len(utt_pcm) > _IN_SR * 2):
                # no profile yet: enroll ONLY a wake-prefixed speaker, so a
                # stranger (or the TV) can never claim the profile by talking
                emb = self.spk.embed(norm)
                if emb is not None and self.spk.voiced_seconds(norm) >= 2.0:
                    self.spk.save_profile(emb)
                    self.speaker = emb
                    self.log_event("enrolled")
        # bench mode: no voice gate — benchmark clips are not enrolled speakers
        # a NEW accepted utterance supersedes an in-flight reply (fragments and
        # ghosts never got here — they can't kill the real reply)
        prev = self.respond_task
        if prev is not None and not prev.done():
            self.gen += 1
            self.playing = False
            from pipecat.frames.frames import InterruptionFrame
            from pipecat.processors.frame_processor import FrameDirection
            await self.push_frame(InterruptionFrame(), direction=FrameDirection.DOWNSTREAM)
            self.log_event("superseded")
        self.respond_task = asyncio.current_task()
        # --- direct handoff: "have the agent do X" skips the LLM entirely ---
        m = re.search(r"have the agent\s+(?:do\s+|build\s+|make\s+|create\s+|write\s+|fix\s+|research\s+|find\s+|set\s+up\s+)?(.+)", text, re.I)
        if not m:
            m2 = re.search(r"\bagent[,;:]\s+(.{4,})", text, re.I)
            if m2:
                m = m2
        if m and (m.group(1) or "").strip():
            rest = (m.group(1) or "").strip().rstrip(".!")
            verbs = {"do", "build", "make", "create", "write", "fix",
                     "research", "find", "set up"}
            if rest.lower() not in verbs and len(rest.split()) >= 2:
                ack = await self._delegate(rest, gen, user_text=text)
                self.log_event("turn", user=text, reply=f"[delegated] {rest[:60]}",
                               asr_s=round(t_asr - t0, 3),
                               first_token_s=round(t_asr - t0, 3),
                               first_audio_s=round(t_asr - t0, 3),
                               total_s=round(time.perf_counter() - t0, 3))
                return
        self.history.append({"role": "user", "content": text})
        reply = ""
        ack = None
        delegate_rest = False
        first_audio = None
        # bench mode: hard-cap the reply so playback never bleeds into the
        # next trial (36 s replies caused false TORs at trial boundaries)
        max_words = 60 if self.mode == "bench" else 400
        first_token = None

        async def one_pass(msgs, use_tools):
            """One streaming LLM pass. Accumulates tool calls; emits TTS
            sentence-by-sentence; returns the tool calls (if any)."""
            nonlocal first_token, first_audio, reply, delegate_rest
            calls = {}
            buf = ""
            body = {"model": TALKER_MODEL, "messages": msgs, "stream": True,
                    "max_tokens": 400}
            if use_tools:
                body["tools"] = self.tools
            stream = await self.http.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.key}"}, json=body)
            self.log_event("llm_http", status=stream.status_code,
                           model=TALKER_MODEL, n_msgs=len(msgs),
                           n_tools=len(body.get("tools", [])))
            async for line in stream.aiter_lines():
                if self.gen != gen:
                    return calls
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    d = json.loads(data)["choices"][0]["delta"]
                except Exception:
                    if "error" in data or self._bad_logged < 3:
                        self.log_event("llm_bad_line", line=data[:200])
                        self._bad_logged += 1
                    continue
                for tc in d.get("tool_calls") or []:
                    i = tc.get("index", 0)
                    slot = calls.setdefault(i, {"id": "", "name": "", "args": ""})
                    slot["id"] = slot["id"] or tc.get("id") or ""
                    fn = tc.get("function") or {}
                    slot["name"] = slot["name"] or fn.get("name") or ""
                    slot["args"] += fn.get("arguments") or ""
                delta = d.get("content")
                if not delta:
                    continue
                if first_token is None:
                    first_token = time.perf_counter()
                buf += delta
                reply += delta
                # addressed-speech gate: the model stays silent for overheard talk
                # (never applied to wake-prefixed utterances — those are for us)
                if (self.mode == "bench" and not no_gate and len(buf) < 40
                        and buf.lstrip().upper().startswith("SILENT")):
                    self.log_event("ignored_unaddressed", user=text, raw=buf[:60])
                    self.history.pop()  # drop the overheard turn from history
                    if self.barged and self.resume_pcm:
                        # interrupted by overheard speech, not a real request:
                        # resume the reply we were in the middle of
                        pcm, self.resume_pcm = self.resume_pcm, None
                        self.barged = False
                        self.playing = True
                        self.log_event("resume_reply")
                        await self._play_pcm(pcm, self.gen)
                    return calls
                if buf.lstrip().upper().startswith("DELEGATE"):
                    delegate_rest = True  # tag is machine-read; collect, never speak
                    continue
                if delegate_rest:
                    continue
                m = _emit_point(buf)
                if m:
                    chunk_text, buf = buf[:m + 1], buf[m + 1:]
                    if chunk_text.strip():
                        wav = await self._speak(chunk_text.strip())
                        if wav and self.gen == gen:
                            if first_audio is None:
                                first_audio = time.perf_counter()
                            self.playing = True
                            await self._play(wav, gen)
                    if len(reply.split()) >= max_words:
                        break  # bench cap: don't let playback bleed into the next trial
            if buf.strip() and not delegate_rest and self.gen == gen:
                wav = await self._speak(buf.strip())
                if wav:
                    if first_audio is None:
                        first_audio = time.perf_counter()
                    self.playing = True
                    await self._play(wav, gen)
            return calls

        try:
            calls = None
            for attempt in (1, 2):
                try:
                    calls = await one_pass(self.history, True)
                except Exception as e:
                    self.log_event("llm_stream_error", attempt=attempt,
                                   err=repr(e)[:120])
                    calls = None
                    if attempt == 2:
                        raise
                if (reply or calls or first_audio is not None
                        or self.gen != gen):
                    break  # got something (or superseded): don't retry
                self.log_event("llm_empty_retry", attempt=attempt, user=text)
            if calls and self.gen == gen:
                self.history.append({"role": "assistant", "content": None, "tool_calls": [
                    {"id": s["id"], "type": "function",
                     "function": {"name": s["name"], "arguments": s["args"]}}
                    for s in calls.values()]})
                for s in calls.values():
                    try:
                        args = json.loads(s["args"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    if s["name"] == "set_timer":
                        out = self._set_timer_local(int(args.get("seconds", 0)),
                                                    args.get("label", "your timer"))
                    else:
                        out = await asyncio.to_thread(self.chatbrain.execute_tool,
                                                      s["name"], args)
                    self.log_event("tool", name=s["name"], args=args, out=str(out)[:80])
                    self.history.append({"role": "tool", "tool_call_id": s["id"],
                                         "content": out})
                reply = ""
                delegate_rest = False
                await one_pass(self.history, False)
            if delegate_rest:
                rest = reply.split("DELEGATE:", 1)[1].strip().rstrip(".!")
                if rest:
                    ack = await self._delegate(rest, gen, user_text=text)
                    reply = ack
        except Exception:
            import traceback
            self.log_event("respond_error", user=text, tb=traceback.format_exc()[-800:])
            print("[watson] respond_error:", traceback.format_exc()[-300:], flush=True)
        finally:
            self.playing = False
            self.barged = False
            total = time.perf_counter() - t0
            if reply.strip():
                self.history.append({"role": "assistant", "content": reply})
            ft = max(0.0, (first_token or t_asr) - t0)
            fa = max(0.0, (first_audio if first_audio is not None else total) - t0)
            self.log_event("turn", user=text, reply=reply,
                           asr_s=round(t_asr - t0, 3),
                           first_token_s=round(ft, 3),
                           first_audio_s=round(fa, 3),
                           total_s=round(total, 3))

    async def _asr(self, pcm16: bytes) -> str:
        if ME3_ASR == "dgx3":
            b64 = base64.b64encode(_wav_bytes(pcm16, _IN_SR)).decode()
            r = await self.http.post(
                f"{DGX3}:8000/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.dgx_key}"},
                json={"model": DGX3_ASR_MODEL, "max_tokens": 200, "messages": [{
                    "role": "user", "content": [
                        {"type": "text",
                         "text": "Transcribe this audio to text. Output only the transcript."},
                        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                    ]}]})
            if r.status_code != 200:
                return ""
            return (r.json()["choices"][0]["message"]["content"] or "").strip()
        r = await self.http.post(f"{SERVER}/transcribe",
                                 content=_wav_bytes(pcm16, _IN_SR),
                                 headers={"Content-Type": "audio/wav"})
        return (r.json() or {}).get("text", "")

    async def _speak(self, text: str):
        if ME3_TTS == "dgx3":
            r = await self.http.post(
                f"{DGX3}:8001/v1/audio/speech",
                headers={"Authorization": f"Bearer {self.dgx_key}"},
                json={"model": DGX3_TTS_MODEL, "input": text, "response_format": "wav"})
            if r.status_code != 200:
                return None
            return r.content
        r = await self.http.post(f"{SERVER}/speak",
                                 json={"text": text, "voice": "af_heart"})
        if r.status_code != 200:
            return None
        return r.content

    async def _play(self, wav: bytes, gen: int):
        await self._play_pcm(_wav_pcm(wav), gen)

    def _echo_score(self, mic: np.ndarray) -> float:
        """Normalized correlation of the probe mic audio against the audio
        Watson is currently playing (searching a small delay window).
        High score = the mic is hearing Watson's own voice."""
        if self.play_ref is None or len(mic) < 400:
            return 0.0
        ref = np.frombuffer(self.play_ref, dtype=np.int16).astype(np.float32)
        m = mic.astype(np.float32)
        if np.std(m) < 1e-6:
            return 0.0
        best = 0.0
        # the mic hears what was played ~0-100 ms ago; play_pos is in samples
        base = max(0, self.play_pos - len(m))
        for off in range(-800, 2401, 160):
            s = ref[max(0, base + off):base + off + len(m)]
            if len(s) != len(m) or np.std(s) < 1e-6:
                continue
            c = abs(float(np.corrcoef(m, s)[0, 1]))
            best = max(best, c)
        return best

    async def _play_pcm(self, pcm: bytes, gen: int):
        from pipecat.frames.frames import OutputAudioRawFrame
        chunk = 1200  # 50 ms at 24 kHz
        played = 0
        self.play_ref = pcm
        self.play_pos = 0
        while played < len(pcm):
            if self.gen != gen:
                # interrupted: stash the unplayed PCM so we can resume
                self.resume_pcm = pcm[played:]
                return
            seg = pcm[played:played + chunk * 2]
            played += len(seg)
            self.play_pos = played // 2
            await self.push_frame(OutputAudioRawFrame(seg, _OUT_SR, 1))
            await asyncio.sleep(chunk / _OUT_SR * 0.92)  # slightly faster than real time

    def log_event(self, event: str, **kw):
        rec = {"t": round(time.time(), 3), "event": event, **kw}
        with open(self.log_dir / "watson_log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


def _emit_point(buf: str) -> int:
    """Split TTS at the first sentence end, or at an early phrase break
    (comma/dash after >= 40 chars) so first audio arrives sooner."""
    for i, ch in enumerate(buf):
        if ch in ".!?\u2026":
            return i
    if len(buf) >= 40:
        for i, ch in enumerate(buf):
            if ch in ",;:—\u2014":
                return i
    return -1


def _wav_pcm(wav: bytes) -> bytes:
    with wave.open(io.BytesIO(wav), "rb") as w:
        return w.readframes(w.getnframes())


async def build(transport, log_dir: Path):
    """fd-benchmark entry: build(transport, log_dir) -> (run_coro, cancel)."""
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import WorkerRunner
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker

    agent = WatsonAgent(log_dir)

    worker = PipelineWorker(
        Pipeline([transport.input(), agent, transport.output()]),
        params=PipelineParams(audio_in_sample_rate=_MIC_SR, audio_out_sample_rate=_OUT_SR),
        enable_tracing=False,
        cancel_on_idle_timeout=False,  # quiet stretches between trials must not kill us
    )
    runner = WorkerRunner(handle_sigint=False)

    async def run():
        await runner.add_workers(worker)
        await runner.run()

    async def cancel():
        await worker.cancel()

    return run(), cancel


# --------------------------------------------------------------------------- mic mode
# Same pipeline, local mic + speaker instead of the benchmark socket.
# sounddevice-based transport (no PyAudio/portaudio dependency).

class MicInput(FrameProcessor):
    """Microphone capture: 16 kHz mono int16 -> InputAudioRawFrame."""

    def __init__(self):
        super().__init__()
        self._queue = None
        self._stream = None
        self._task = None

    async def setup(self, setup):
        import sounddevice as sd
        print("[mic] setup begin", flush=True)
        await super().setup(setup)
        print("[mic] super.setup done", flush=True)
        self._queue = asyncio.Queue(maxsize=256)
        loop = asyncio.get_running_loop()
        q = self._queue

        def callback(indata, frames, time_info, status):
            loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

        self._stream = sd.InputStream(samplerate=_IN_SR, channels=1, dtype="int16",
                                      blocksize=800, callback=callback)
        print("[mic] stream opened", flush=True)

    async def start(self, frame):
        print("[mic] start called", flush=True)
        self._stream.start()
        self._task = asyncio.create_task(self._pump())

    async def _pump(self):
        from pipecat.frames.frames import InputAudioRawFrame
        while True:
            data = await self._queue.get()
            await self.push_frame(InputAudioRawFrame(data, _IN_SR, 1))

    async def cleanup(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        await super().cleanup()


class SpeakerOutput(FrameProcessor):
    """Speaker playback: OutputAudioRawFrame -> sounddevice 24 kHz output."""

    def __init__(self):
        super().__init__()
        self._stream = None
        self._loop = None

    async def setup(self, setup):
        print("[spk] setup", flush=True)
        await super().setup(setup)

    async def start(self, frame):
        print("[spk] start called", flush=True)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        from pipecat.frames.frames import OutputAudioRawFrame
        if isinstance(frame, OutputAudioRawFrame) and frame.sample_rate == _OUT_SR:
            if self._stream is None:
                import sounddevice as sd
                self._loop = asyncio.get_running_loop()
                self._stream = sd.OutputStream(samplerate=_OUT_SR, channels=1,
                                               dtype="int16", blocksize=1200)
                self._stream.start()
            data = frame.audio
            await self._loop.run_in_executor(None, self._stream.write, data)
        else:
            await self.push_frame(frame, direction)

    async def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def cancel(self):
        await self.stop()


class MicTransport:
    """Duck-typed like BaseTransport: .input() / .output()."""

    def __init__(self):
        self._input = MicInput()
        self._output = SpeakerOutput()

    def input(self):
        return self._input

    def output(self):
        return self._output


async def mic_main():
    os.environ["ME3_MODE"] = "mic"   # voice gate on, SILENT gate off
    # mic mode runs without the benchmark launcher, so load the project .env
    # (OPENROUTER_API_KEY, ME3_SERVER, ...) — without it the talker gets a
    # None key and every reply is a silent 401.
    env_file = ME3 / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))
    from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
    log_dir = Path.home() / "me3" / "runs" / "live"
    transport = LocalAudioTransport(LocalAudioTransportParams(
        audio_in_enabled=True, audio_out_enabled=True,
        audio_in_sample_rate=_MIC_SR, audio_out_sample_rate=_OUT_SR,
        audio_in_channels=1, audio_out_channels=1))
    run_coro, cancel = await build(transport, log_dir)
    try:
        await run_coro
    except asyncio.CancelledError:
        pass
    finally:
        await cancel()


if __name__ == "__main__":
    import sys
    if "--mic" in sys.argv:
        print("[watson] Pipecat mic mode — say 'Watson' or just talk; Ctrl+C to quit", flush=True)
        asyncio.run(mic_main())
    else:
        print("usage: pipecat_assistant.py --mic   (benchmark mode runs via fd-benchmark/launch_assistant.py)")