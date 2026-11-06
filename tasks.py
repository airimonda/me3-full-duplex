"""ME3 background task manager — parallel Hermes runs with status + result announcements.

Each delegated task is a headless Hermes one-shot run in its own sandbox folder:
  hermes chat --oneshot --query-file prompt.txt --in <sandbox> --max-turns N --run-budget S

The talker keeps conversing while tasks run. After each conversational turn,
check_finished() returns newly completed tasks. Only legacy/chat.py announced them; pipecat_assistant.py does not call it yet.
"""
import os
import re
import shutil
import subprocess
import sys
import time

TASKS_DIR = os.path.expanduser("~/sandbox")   # the ONE folder agents may control
# the profile that ships next to this file (not a hard-coded ~/me3 copy)
SANDBOX_PROFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "hermes-sandbox.sb")
MAX_TURNS = int(os.environ.get("ME3_TASK_MAX_TURNS", "40"))
RUN_BUDGET_S = int(os.environ.get("ME3_TASK_BUDGET", "900"))  # 15 min per task
MAX_CONCURRENT = int(os.environ.get("ME3_MAX_TASKS", "2"))    # background slots

_running = {}   # id -> {"proc": Popen, "dir": str, "log": str, "prompt": str, "t0": float}
_finished = {}  # id -> {"status": "done"/"failed", "result": str, "files": [str], "elapsed_s": float}


def _task_prompt(restatement):
    # timers/reminders are their own kind of task: wait, then report
    m = re.search(r"timer.*?(\d+)\s*(second|minute|hour)", restatement, flags=re.I)
    if m or re.search(r"\btimer\b", restatement, flags=re.I):
        secs = int(m.group(1)) * {"second": 1, "minute": 60, "hour": 3600}[m.group(2).lower()] if m else 20
        label = re.sub(r".*?timer[^a-z0-9]*(for|that)?\s*", "", restatement, flags=re.I) or "your timer"
        return (f"Run `sleep {secs}` in the terminal, then print exactly one line: "
                f"RESULT: Time's up — that's {label.strip() or 'your timer'}.")
    return (
        "You are working in a sandbox folder that you must never write outside of. Task: "
        + restatement.strip() + "\n"
        "Create the deliverable in the CURRENT folder only (for a website: a single "
        "index.html with inline CSS, self-contained, no external requests). Keep it "
        "simple and working. Never create, modify or delete files anywhere else. "
        "When done, print one final line starting with RESULT: summarizing what you built."
    )


def _which(name):
    """Agent CLIs install to ~/.local/bin, which a service PATH may lack."""
    found = shutil.which(name)
    if found:
        return found
    for d in ("~/.local/bin", "~/.pi/agent/bin"):   # pi.dev installer: ~/.pi/agent/bin
        local = os.path.expanduser(f"{d}/{name}")
        if os.path.exists(local):
            return local
    return name


def _sandboxed(cmd, tdir):
    """Confine the agent: it may write only in ~/sandbox, its own runtime
    dirs and temp. Mac: sandbox-exec + hermes-sandbox.sb. Linux (the Pi):
    bubblewrap, with the whole filesystem read-only except those binds."""
    home = os.path.expanduser("~")
    if sys.platform == "darwin" and shutil.which("sandbox-exec") and os.path.exists(SANDBOX_PROFILE):
        return ["sandbox-exec", "-D", "HOME=" + home, "-f", SANDBOX_PROFILE] + cmd
    if shutil.which("bwrap"):
        writable = [TASKS_DIR] + [os.path.join(home, d) for d in
                                  (".pi", ".hermes", ".npm", ".cache", ".local/share/pi-node")]
        args = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
                "--tmpfs", "/tmp", "--unshare-pid", "--die-with-parent"]
        for w in writable:
            if os.path.isdir(w):
                args += ["--bind", w, w]
        return args + ["--chdir", tdir, "--"] + cmd
    print("[tasks] WARNING: no sandbox available, running unconfined", flush=True)
    return cmd


def open_url(target):
    """Show a file or URL on the demo screen: Mac `open`; on the Pi, Chromium
    on the HDMI desktop (DISPLAY=:0)."""
    if sys.platform == "darwin":
        subprocess.Popen(["open", target])
        return True
    env = dict(os.environ, DISPLAY=os.environ.get("DISPLAY") or ":0")
    browser = shutil.which("chromium") or shutil.which("chromium-browser")
    cmd = [browser, "--new-window", target] if browser else ["xdg-open", target]
    subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    return True


def submit_task(restatement):
    """Start a background Hermes run. Returns task id, or None if at capacity."""
    if len(_running) >= MAX_CONCURRENT:
        return None
    os.makedirs(TASKS_DIR, exist_ok=True)
    nums = [int(d[4:]) for d in os.listdir(TASKS_DIR)
            if d.startswith("task") and d[4:].isdigit()]
    tid = f"task{(max(nums) + 1 if nums else 1):02d}"
    tdir = os.path.join(TASKS_DIR, tid)
    os.makedirs(tdir, exist_ok=True)
    prompt_path = os.path.join(tdir, "prompt.txt")
    log_path = os.path.join(tdir, "run.log")
    with open(prompt_path, "w") as f:
        f.write(_task_prompt(restatement))
    logf = open(log_path, "w")
    harness = os.environ.get("ME3_HARNESS", "pi")
    if harness == "hermes":
        cmd = [_which("hermes"), "chat", "--oneshot", "--query-file", prompt_path,
               "--in", tdir, "--max-turns", str(MAX_TURNS),
               "--run-budget", str(RUN_BUDGET_S)]
    else:
        agent_model = os.environ.get("ME3_AGENT_MODEL", "z-ai/glm-5.3-flash")
        prompt_text = open(prompt_path).read().strip()
        cmd = [_which("pi"), "-p", "--provider", "openrouter", "--model", agent_model,
               "--no-session", "--", prompt_text]
    cmd = _sandboxed(cmd, tdir)
    env = dict(os.environ)
    if "OPENROUTER_API_KEY" not in env:
        # Mac: ~/.hermes/.env; Pi: ~/me3/.env (never copied between machines)
        for keyfile in ("~/.hermes/.env", "~/me3/.env"):
            keyfile = os.path.expanduser(keyfile)
            if not os.path.exists(keyfile):
                continue
            for line in open(keyfile):
                if line.startswith("OPENROUTER_API_KEY="):
                    env["OPENROUTER_API_KEY"] = line.strip().split("=", 1)[1].strip('"')
                    break
            if "OPENROUTER_API_KEY" in env:
                break
    proc = subprocess.Popen(
        cmd,
        stdout=logf, stderr=subprocess.STDOUT,
        cwd=tdir, env=env,
    )
    _running[tid] = {"proc": proc, "dir": tdir, "log": log_path,
                     "prompt": restatement, "t0": time.time()}
    return tid


def _tail(path, nbytes=1200):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            return f.read().decode(errors="replace")
    except OSError:
        return ""


def status_summary(tid):
    """One-sentence status for the talker."""
    t = _running.get(tid)
    if not t:
        f = _finished.get(tid)
        if f:
            return f"{tid}: {f['status']} in {f['elapsed_s']:.0f}s"
        return f"{tid}: unknown"
    proc = t["proc"]
    if proc.poll() is None:
        tail = _tail(t["log"], 300).replace("\n", " ")[:140]
        return f"{tid} running ({time.time()-t['t0']:.0f}s): {tail}"
    return f"{tid} just finished"


def check_finished():
    """Move finished processes into _finished. Returns list of newly finished ids."""
    newly = []
    for tid, t in list(_running.items()):
        proc = t["proc"]
        if proc.poll() is None:
            continue
        elapsed = time.time() - t["t0"]
        tail = _tail(t["log"], 4000)
        m = re.findall(r"RESULT:\s*(.+)", tail)
        result_line = m[-1].strip() if m else tail.strip().splitlines()[-1] if tail.strip() else ""
        ok = proc.returncode == 0 and bool(result_line)
        files = [f for f in os.listdir(t["dir"])
                 if os.path.isfile(os.path.join(t["dir"], f))
                 and f not in ("prompt.txt", "run.log")]
        _finished[tid] = {"status": "done" if ok else "failed",
                          "result": result_line[:300], "files": files,
                          "prompt": t.get("prompt", ""),
                          "elapsed_s": elapsed, "dir": t["dir"]}
        newly.append(tid)
        del _running[tid]
    return newly


def _shorten(s, n=110):
    if len(s) <= n:
        return s
    cut = s[:n].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + "…"


def _thing(prompt):
    """A short, human way to refer to the task's subject, e.g. 'the Tokyo itinerary site'."""
    s = re.sub(r"^you want\s*", "", prompt.strip(), flags=re.I)
    # "research the history of X" -> "the history of X"
    s = re.sub(r"^(please\s+)?(research|build|make|create|write|draft|design|"
               r"plan|find|look up|put together|prepare|summari[sz]e)\s+(me\s+)?",
               "", s, flags=re.I)
    s = re.sub(r"^(a|an|the)\s+", "", s, flags=re.I)
    # cut after the thing's noun; prefer website/site/game/app over "page"
    # so "a one-page website that says hi" -> "the one-page website"
    if re.search(r"\b(website|site|game|app)\b", s, flags=re.I):
        s = re.sub(r"\b(website|site|game|app)\b.*$", r"\1", s, flags=re.I)
    else:
        s = re.sub(r"(?<!-)\bpage\b.*$", "page", s, flags=re.I)
    words = s.split()
    return "the " + " ".join(words[:8]) if words else "your request"


def announcement(tid):
    """Spoken line for a finished task — human phrasing, no ids or file paths."""
    f = _finished.get(tid)
    if not f:
        return None
    thing = _thing(f.get("prompt", ""))
    if f["status"] == "done" and f["files"]:
        return f"Done! {thing.capitalize()} is ready — take a look."
    if f["status"] == "done":
        return f"Done with that. {_shorten(f['result'], 90)}"
    return f"Hmm, {thing} didn't work out — I hit a snag. Want me to give it another go?"


def deliverable_path(tid):
    """Absolute path of the task's primary deliverable, or None."""
    f = _finished.get(tid)
    if not f or not f["files"]:
        return None
    return os.path.join(f["dir"], f["files"][0])


def open_deliverable(tid):
    """Open the finished task's deliverable on the Mac (macOS `open`)."""
    path = deliverable_path(tid)
    if not path:
        return None
    try:
        open_url(path)
        return path
    except Exception as e:
        print(f"[tasks] open failed for {path}: {e}", flush=True)
        return None


def list_running():
    return {tid: status_summary(tid) for tid in _running}


def sandbox_url(tid):
    """Path the demo screen can open (served over HTTP later, Step 7)."""
    f = _finished.get(tid)
    return f["dir"] if f else None


def reset_all():
    """Kill all running tasks (demo reset)."""
    for tid, t in list(_running.items()):
        t["proc"].kill()
    _running.clear()
    _finished.clear()