# Step 2 (Oct 26) — Hermes spike results

Machine: this Mac (Apple M5). Hermes was already installed — no install script needed.
Default model: z-ai/glm-5.3-flash via OpenRouter (provider auto). hermes doctor: clean
(security advisories none, env healthy, API key configured).

## One-shot mode — CONFIRMED (open question resolved)

- `hermes chat -q "..." --oneshot` answers and exits on a non-TTY. Also `-Q` works.
- Website task: "build a one-page coffee-history website, single index.html, inline CSS,
  under 300 lines" -> completed in 34 s wall (6 messages, 4 tool calls, 174-line page).
- Useful one-shot flags found in `hermes chat --help`:
  - `--in DIR` — change into DIR before starting (scopes workspace + resume lookups)
  - `--max-turns N` and `--run-budget SECONDS` — per-task caps (use for cost caps)
  - `--safe-mode` and `--worktree/-w` — isolation options
  - `--query-file PATH` — reads prompt from a file, shell-safe (good for harness-mcp
    submit_task, which must pass arbitrary spoken text safely)

## Sandbox caveat — REAL, confirmed live

- First one-shot run with cwd=~/me3/sandbox wrote `~/index.html` to the
  HOME directory; the sandbox folder stayed empty. Hermes one-shot does not pin to cwd.
- Mitigation for harness-mcp: always pass `--in <sandbox-dir>` (verified working in the fetch
  test), and still treat the sandbox as advisory, not a hard wall. Re-test file writes with
  --in before the demo; if writes still escape, look at --safe-mode.

## MCP — CONFIRMED working

- The `mcp` extra is already installed in Hermes's runtime venv (discovery will not be skipped).
- Added a free server: `printf 'y\n' | hermes mcp add webfetch --command uvx --args mcp-server-fetch`
  (non-TTY: pipe "y" to the enable-all prompt; no --yes flag exists).
- Verified end to end: one-shot fetch of https://example.com ran 6 tool calls in 15 s, and the
  session transcript in ~/.hermes/state.db shows `mcp__webfetch__fetch` registered and executed.

## harness-mcp design implications

- submit_task -> `hermes chat --query-file <prompt.txt> --in <sandbox> --oneshot
  --max-turns N --run-budget S` as a background process, log to file.
- get_status -> tail the log file.
- get_result -> final message from log + `ls` of the sandbox.
- steer_task -> still cancel-and-resubmit (no mid-run messaging found in one-shot mode).

## Still open on the Pi

- Pi→Mac round-trip control latency (needs the Pi online). Everything else in Step 2 is done.
