"""One-time Spotify authorization for ME3/Watson.

Prerequisites (you do once at developer.spotify.com):
  1. Create an app: https://developer.spotify.com/dashboard -> Create app
  2. In the app settings, add this Redirect URI exactly:
         http://127.0.0.1:8888/callback
  3. Put your credentials in ~/.hermes/.env:
         SPOTIFY_CLIENT_ID=...your client id...
         SPOTIFY_CLIENT_SECRET=***your client secret***

Then run:  ~/me3/venv/bin/python ~/me3/spotify_auth.py
  - prints an authorization link, open it in your browser, log in and approve
  - the local server catches the redirect, exchanges the code, and appends
    SPOTIFY_REFRESH_TOKEN to ~/.hermes/.env
  - Watson then controls real playback (needs Spotify Premium)
"""
import base64
import http.server
import os
import threading
import urllib.parse
import webbrowser

import httpx

ENV_PATH = os.path.expanduser("~/.hermes/.env")
PORT = 8888
REDIRECT_URI = f"http://127.0.0.1:{PORT}/callback"
SCOPES = " ".join([
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
    "streaming",
])


def env_get(name):
    if os.environ.get(name):
        return os.environ[name]
    if os.path.exists(ENV_PATH):
        for line in open(ENV_PATH):
            if line.strip().startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"')
    return None


def env_append(name, value):
    lines = open(ENV_PATH).read().rstrip("\n") if os.path.exists(ENV_PATH) else ""
    body = [ln for ln in lines.split("\n") if ln and not ln.startswith(name + "=")]
    body.append(f"{name}={value}")
    with open(ENV_PATH, "w") as f:
        f.write("\n".join(body) + "\n")


def main():
    cid, secret = env_get("SPOTIFY_CLIENT_ID"), env_get("SPOTIFY_CLIENT_SECRET")
    if not cid or not secret:
        print("Missing SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET in ~/.hermes/.env")
        print("1. https://developer.spotify.com/dashboard -> Create app")
        print(f"2. Redirect URI: {REDIRECT_URI}")
        print("3. Add the two lines to ~/.hermes/.env, then run this again.")
        return

    auth_url = ("https://accounts.spotify.com/authorize?" +
                urllib.parse.urlencode({
                    "client_id": cid, "response_type": "code",
                    "redirect_uri": REDIRECT_URI, "scope": SCOPES,
                    "show_dialog": "true"}))
    print("\n1. Open this link in your browser and approve access:\n")
    print(auth_url)
    print(f"\n2. Waiting for the redirect on port {PORT}...\n")
    webbrowser.open(auth_url)

    result = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            code = (qs.get("code") or [None])[0]
            error = (qs.get("error") or [None])[0]
            if not code and not error:
                # stray hit (stale tab/reload) — keep waiting for the real redirect
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"<h2>Waiting for Spotify authorization&hellip;</h2>")
                return
            result["code"], result["error"] = code, error
            self.send_response(200)
            self.end_headers()
            msg = "Spotify authorized! You can close this tab." if code \
                else f"Authorization failed: {error}"
            self.wfile.write(f"<h2>{msg}</h2>".encode())

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    while "code" not in result and "error" not in result:
        time.sleep(0.2)
    server.shutdown()
    if not result.get("code"):
        print("Authorization failed:", result.get("error"))
        return

    r = httpx.post("https://accounts.spotify.com/api/token",
                   data={"grant_type": "authorization_code", "code": result["code"],
                         "redirect_uri": REDIRECT_URI},
                   headers={"Authorization": "Basic " + base64.b64encode(
                       f"{cid}:{secret}".encode()).decode()}, timeout=20)
    tok = r.json()
    if "refresh_token" not in tok:
        print("Token exchange failed:", tok)
        return
    env_append("SPOTIFY_REFRESH_TOKEN", tok["refresh_token"])
    print("SUCCESS — SPOTIFY_REFRESH_TOKEN saved to ~/.hermes/.env")
    print("Watson can now control real playback. Try: 'Watson, play some jazz.'")


if __name__ == "__main__":
    import time
    main()