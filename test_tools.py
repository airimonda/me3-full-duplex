"""Test talker tool-calling without the mic."""
import sys
import time

sys.path.insert(0, "${HOME}/me3")
import chat  # noqa: E402  (imports loop, HTTP, tools)

key = chat.load_openrouter_key()

for q in ["What time is it right now?",
          "Set a timer for 10 seconds for the pasta.",
          "Remember that I prefer dark themes.",
          "What do you remember?"]:
    msgs = [{"role": "system", "content": chat.SYSTEM_PROMPT},
            {"role": "user", "content": q}]
    t0 = time.perf_counter()
    reply, first = chat.talker_stream(msgs, key, lambda p: None, tools=chat.QUICK_TOOLS)
    print(f"Q: {q!r}\n  -> ({(first or 0):.2f}s/{time.perf_counter()-t0:.2f}s) {reply!r}\n", flush=True)
