"""Local browser demo server for the LangGraph customer-support agent."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from graph_proposed import PROPOSED_GRAPH, llm
from tools import backend

ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
MAX_BODY_BYTES = 64 * 1024


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "CustomerSupportDemo/1.0"

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

    def _send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("Invalid Content-Length.")
        if size < 1 or size > MAX_BODY_BYTES:
            raise ValueError("Request body must be between 1 byte and 64 KB.")
        try:
            body = json.loads(self.rfile.read(size))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Request body must be valid JSON.") from exc
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object.")
        return body

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/api/health":
            return self._send_json({"ok": True, "provider": llm.provider_name})
        relative = "index.html" if path == "/" else path.removeprefix("/web/")
        if path != "/" and not path.startswith("/web/"):
            return self._send_json({"error": "Not found."}, 404)
        target = (WEB_ROOT / relative).resolve()
        if WEB_ROOT.resolve() not in target.parents or not target.is_file():
            return self._send_json({"error": "Not found."}, 404)
        content = target.read_bytes()
        mime, _ = mimetypes.guess_type(target.name)
        self.send_response(200)
        self.send_header("Content-Type", f"{mime or 'application/octet-stream'}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._read_json()
            if path == "/api/reset":
                backend.reset()
                return self._send_json({"ok": True})
            if path == "/api/chat":
                state = body.get("state", {})
                message = body.get("message", "")
                if not isinstance(state, dict) or not isinstance(message, str) or not message.strip():
                    raise ValueError("Provide a non-empty message and an object state.")
                state = dict(state)
                state.update({
                    "user_message": message.strip(),
                    "trace": [],
                    "approval_replay": False,
                    "user_approved": False,
                    "needs_approval": False,
                })
                result = PROPOSED_GRAPH.invoke(state)
                return self._send_json({"ok": True, "state": result})
            if path == "/api/approve":
                state = body.get("state")
                if not isinstance(state, dict) or not state.get("needs_approval"):
                    raise ValueError("There is no pending approval to confirm.")
                state = dict(state)
                state["user_approved"] = True
                state["approval_replay"] = True
                result = PROPOSED_GRAPH.invoke(state)
                return self._send_json({"ok": True, "state": result})
            if path == "/api/decline":
                state = body.get("state")
                if not isinstance(state, dict) or not state.get("needs_approval"):
                    raise ValueError("There is no pending approval to decline.")
                state = dict(state)
                state["needs_approval"] = False
                state["user_approved"] = False
                state["approval_replay"] = False
                state["resolved"] = False
                state["response"] = "Cancellation stopped. The booking was not changed."
                state["history"] = list(state.get("history", [])) + ["Customer declined cancellation."]
                state["trace"] = list(state.get("trace", [])) + ["Approval declined; no cancellation tool called"]
                return self._send_json({"ok": True, "state": state})
            return self._send_json({"error": "Not found."}, 404)
        except ValueError as exc:
            return self._send_json({"error": str(exc)}, 400)
        except Exception as exc:
            return self._send_json({"error": str(exc)}, 502)


def main():
    parser = argparse.ArgumentParser(description="Run the local customer-support agent demo.")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind (default: localhost).")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000).")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Customer Support Agent demo: http://{args.host}:{args.port}")
    print(f"Language model: {llm.provider_name}")
    print("Demo bookings are mock data. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
