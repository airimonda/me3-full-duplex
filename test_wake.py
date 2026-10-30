import chat

cases = ["Watson", "Orton", "Watch on.", "What's on TV?", "Watson, build me a website",
         "button.", "Watsun my friend", "what is the capital"]
for c in cases:
    matched, end = chat.wake_match(c)
    rest = c[end:] if matched else "-"
    print(f"{c!r:45} -> {'WAKE' if matched else 'no  '}  rest={rest!r}")
