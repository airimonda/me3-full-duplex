# ME3 — one-command operation
#   make setup   install the Python environment (uv, ~5 min)
#   make run     live assistant: local mic + speaker   (needs the ASR/TTS server)
#   make server  start/replace the Whisper+Kokoro server on :8300
#   make bench   run the Full-Duplex-Bench suite (28 trials, ~9 min)
#   make stop    stop everything
PY      := $(HOME)/me3/venv/bin/python
FDB     := $(HOME)/me3/fd-benchmark/.venv/bin/python
ENV     := VIRTUAL_ENV= PYTHONPATH=

.PHONY: help setup server run bench stop

help:
	@echo "make setup   - create venv + install deps"
	@echo "make server  - start ASR/TTS server (127.0.0.1:8300)"
	@echo "make run     - live mic assistant (starts server first if needed)"
	@echo "make bench   - Full-Duplex-Bench: 2 rounds, 28 trials"
	@echo "make stop    - kill server and any running agent"

setup:
	bash -c 'test -x $(PY) || uv venv $(HOME)/me3/venv --python 3.11'
	uv pip install --python $(PY) -r requirements.txt

server:
	@bash -c 'curl -sf -m 2 http://127.0.0.1:8300/health >/dev/null && echo "server already up on :8300" || (echo "starting server..."; nohup $(PY) $(HOME)/me3/server.py > $(HOME)/me3/server.log 2>&1 & sleep 8; curl -sf -m 2 http://127.0.0.1:8300/health && echo " server ready")'

run: server
	$(ENV) $(PY) $(HOME)/me3/pipecat_assistant.py --mic

bench:
	$(ENV) $(FDB) -m fdbench run --student me3 --rounds 2 --connect direct \
		--where local --judge-provider openrouter --yes --recordings k \
		2>&1 | tee $(HOME)/me3/fd-benchmark/run.log

stop:
	-pkill -f server.py; -pkill -f pipecat_assistant
