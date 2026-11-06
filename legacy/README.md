# legacy/ (deprecated 2026-11-05)

`pipecat_assistant.py` is the only voice loop. These files are kept for
history and are not imported by anything outside this folder.

- `chat.py`: the original full-duplex loop. Its prompt, tool schemas and
  tool dispatcher moved to `../brain.py`.
- `loop.py`: HTTP helpers used only by `chat.py` and `ttft_probe.py`.
- `duplex.py`, `bench_duplex.py`, `bench_duplex_results.json`: the
  pause-and-probe barge-in gate used by `chat.py`, and its bench. Its echo
  check (`xcorr_echo_score`, learned echo level) was copied into
  `pipecat_assistant.py`; the pause-on-barge policy was not.
- `test_*.py`, `ttft_probe.py`, `debug_*.py`: scripts that import the files
  above. Tool checks now live in `../tests/test_brain.py`, echo checks in
  `../tests/test_echo.py`.

To run any of them, run from this folder so the imports resolve.
