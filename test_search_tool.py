import sys, time
sys.path.insert(0, "${HOME}/me3")
import chat

key = chat.load_openrouter_key()
for q in ["What is the capital of India?",
          "Who won the most recent F1 race?"]:
    msgs = [{"role": "system", "content": chat.SYSTEM_PROMPT}, {"role": "user", "content": q}]
    t0 = time.perf_counter()
    reply, first = chat.talker_stream(msgs, key, lambda p: None, tools=chat.QUICK_TOOLS)
    print(f"Q: {q}\n  -> ({time.perf_counter()-t0:.1f}s) {reply!r}\n", flush=True)
