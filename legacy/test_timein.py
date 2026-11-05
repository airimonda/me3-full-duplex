import sys, time
sys.path.insert(0, str(Path.home() / "me3") + "")
import chat

print(chat.execute_tool("get_time_in", {"city": "Mumbai"}))
print(chat.execute_tool("get_time_in", {"city": "Jakarta"}))

key = chat.load_openrouter_key()
msgs = [{"role": "system", "content": chat.SYSTEM_PROMPT},
        {"role": "user", "content": "What time is it in Mumbai, India right now?"}]
t0 = time.perf_counter()
reply, _ = chat.talker_stream(msgs, key, lambda p: None, tools=chat.QUICK_TOOLS)
print(f"({time.perf_counter()-t0:.1f}s) {reply!r}")
