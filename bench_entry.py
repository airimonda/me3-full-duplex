"""fd-benchmark entry for Watson, loading the code that sits next to this file.

fd-benchmark/assistant_entry.py (gitignored, in the clone) always imports
from the folder that contains fd-benchmark. This entry imports from its
own folder instead, so a checkout elsewhere (a worktree) can be benchmarked:

  FDBENCH_ENTRY=<this file> FDBENCH_PROJECT=<this folder> python -m fdbench run ...
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

PROJECT_MARKER = "pipecat_assistant.py"


def audio_rates() -> tuple[int, int]:
    return 16000, 24000


def describe() -> dict:
    """Model versions for the results card."""
    import pipecat_assistant as pa
    return {"ASR": "Whisper large-v3-turbo (mlx-whisper, Mac)",
            "LLM": f"{pa.TALKER_MODEL} (OpenRouter); fallback {pa.FALLBACK_MODEL}",
            "TTS": f"Kokoro-82M {pa.TTS_VOICE} (Mac)",
            "VAD": "Silero VAD (Pipecat SileroVADAnalyzer)",
            "Harness": "Pipecat custom FrameProcessor (WatsonAgent)"}


async def build(transport, log_dir: Path):
    os.environ.setdefault("ME3_MODE", "bench")
    from pipecat_assistant import build as me3_build
    return await me3_build(transport, log_dir)
