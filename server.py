#!/usr/bin/env python3
"""Chat sahifasi va API. Faqat shu kompyuterdan ochiladi (127.0.0.1)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from chat import Bot

HOST, PORT = "127.0.0.1", 8000
ROOT = Path(__file__).resolve().parent
PAGE = (ROOT / "static" / "index.html").read_bytes()
ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}
MAX_BODY = 8192
MAX_MESSAGE = 2000
bot = Bot()
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "SinovOybek"
    sys_version = ""

    def log_message(self, *args):  # yozishmalar logga tushmasin
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode())

    def _host_ok(self):
        return self.headers.get("Host") in ALLOWED_HOSTS

    def do_GET(self):
        if not self._host_ok():
            return self._json(403, {"error": "ruxsat yo‘q"})
        if self.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        self._json(404, {"error": "topilmadi"})

    def do_POST(self):
        if not self._host_ok():
            return self._json(403, {"error": "ruxsat yo‘q"})
        origin = self.headers.get("Origin")
        if origin and origin.split("//")[-1] not in ALLOWED_HOSTS:
            return self._json(403, {"error": "ruxsat yo‘q"})
        if self.path != "/api/chat":
            return self._json(404, {"error": "topilmadi"})
        if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
            return self._json(415, {"error": "JSON kerak"})
        try:
            size = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return self._json(400, {"error": "noto‘g‘ri so‘rov"})
        if size <= 0 or size > MAX_BODY:
            return self._json(413, {"error": "xabar juda uzun"})
        try:
            msg = json.loads(self.rfile.read(size)).get("message", "")
        except (ValueError, AttributeError):
            return self._json(400, {"error": "noto‘g‘ri so‘rov"})
        if not isinstance(msg, str) or not msg.strip():
            return self._json(400, {"error": "xabar bo‘sh"})
        if len(msg) > MAX_MESSAGE:
            return self._json(413, {"error": "xabar juda uzun"})
        with lock:
            reply = bot.ask(msg.strip())
        self._json(200, {"reply": reply})


if __name__ == "__main__":
    print(f"Sinov-Oybek tayyor: http://{HOST}:{PORT}  (to‘xtatish: Ctrl+C)", flush=True)
    try:
        ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print()
