import sys, time
sys.path.insert(0, str(Path.home() / "me3") + "")
import chat

key = chat.load_openrouter_key()
for q in ["What time is it in Jakarta right now?",
          "What is the capital of Spain?"]:
    msgs = [{"role": "system", "content": chat.SYSTEM_PROMPT}, {"role": "user", "content": q}]
    t0 = time.perf_counter()
    reply, _ = chat.talker_stream(msgs, key, lambda p: None, tools=chat.QUICK_TOOLS)
    print(f"Q: {q}\n  pass1 ({time.perf_counter()-t0:.1f}s): {reply!r}", flush=True)
    if chat.DEFLECT_RE.search(reply):
        print("  -> deflected, auto-search would fire here", flush=True)
