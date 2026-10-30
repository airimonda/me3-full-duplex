import sys, time
sys.path.insert(0, str(Path.home() / "me3") + "")
import chat

key = chat.load_openrouter_key()
for q in ["Play some jazz music.", "Turn off the living room lights.",
          "Set a timer for 20 seconds.", "Remind me to call the bank tomorrow."]:
    msgs = [{"role": "system", "content": chat.SYSTEM_PROMPT}, {"role": "user", "content": q}]
    t0 = time.perf_counter()
    reply, _ = chat.talker_stream(msgs, key, lambda p: None, tools=chat.QUICK_TOOLS)
    print(f"{q!r}\n  -> ({time.perf_counter()-t0:.1f}s) {reply!r}\n", flush=True)
