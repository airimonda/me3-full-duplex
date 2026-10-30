"""Verify two Hermes tasks run in parallel."""
import subprocess
import sys
import time

sys.path.insert(0, "${HOME}/me3")
import tasks as T

a = T.submit_task("write a file hello1.txt containing the text hello one")
b = T.submit_task("write a file hello2.txt containing the text hello two")
print("submitted:", a, b, flush=True)

time.sleep(10)
running = T.list_running()
print("running after 10s:", running, flush=True)
for tid in (a, b):
    if tid is None:
        continue
    ps = subprocess.run(["pgrep", "-f", f"--in ${HOME}/me3/tasks/{tid}"],
                        capture_output=True, text=True)
    print(f"{tid} hermes pid(s): {ps.stdout.strip() or 'none'}", flush=True)

# wait for both to finish, then report
for _ in range(60):
    done = T.check_finished()
    if not T._running:
        break
    time.sleep(5)
for tid in (a, b):
    f = T._finished.get(tid)
    print(f"{tid}: {f['status'] if f else 'still running'} files={f['files'] if f else []}", flush=True)