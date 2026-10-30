import sys, time
sys.path.insert(0, "${HOME}/me3")
import chat

print(chat.execute_tool("set_timer", {"seconds": 3, "label": "test"}))
time.sleep(4.5)
print("timer should have fired by now")
