#!/usr/bin/env python3
"""serve_showcase.py — 0xBakeer's arbiter showcase (MIT) served over OUR DeciServ gate.

His showcase pages (showcase/<game>/index.html) speak the Jev contract:
    GET  /readyz            -> {"status": "ready", "models": [...]}
    POST /v1/systemone      -> {"model", "answers", "usage", "routing", "latency_ms"}
Our gate speaks:
    POST /decide            -> {"answers", "latency_ms", ...}

Same question shapes ({type: choice|score|noul, instructions, criteria}), same
answer shapes (probabilities/confidence/score) — this server is a thin adapter:
serves his static pages from the same origin and forwards /v1/systemone to
DeciServ on Spark. The pages' own graphics, halos, panels and HUDs render
exactly as he built them; every decision comes from OUR model.

    python3 games/serve_showcase.py --port 8010 \
        --gate http://192.168.2.185:8710
Then open:
    http://localhost:8010/showcase/snake/   (etc. — six tabs, one per game)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---- OTA latency instrumentation (additive; see games/LATENCY_INSTRUMENTATION.md)
from latency_instrumentation import (
    health as ota_health,
    install as ota_install,
    js as ota_js,
    note_adapter,
    note_browser,
    parse_gate_questions,
)

REPO = Path(__file__).resolve().parent.parent
SHOWCASE = REPO / "vendor" / "arbiter" / "showcase"

TYPES = {
    ".html": "text/html", ".mjs": "text/javascript", ".js": "text/javascript",
    ".css": "text/css", ".json": "application/json", ".md": "text/markdown",
    ".svg": "image/svg+xml", ".png": "image/png", ".webp": "image/webp",
    ".ico": "image/x-icon", ".woff2": "font/woff2", ".txt": "text/plain",
}

GATE = "http://192.168.2.185:8710"

# ---- rebranding (vendor/ stays unmodified; we rewrite on the wire) --------
# His UI says "arbiter plays" because it IS his code (MIT, kept unmodified on
# disk). When WE serve it, the pages should say DeciServ — with his credit
# carried in the page footer and in NOTICE.md / README.md.
WORDMARK = "deciserv plays"
CREDIT_LINK = '<a href="https://github.com/0xBakeer/arbiter">showcase: arbiter (MIT)</a>'


def rebrand(body: bytes) -> bytes:
    html = body.decode("utf-8")
    html = html.replace("arbiter plays", "deciserv plays")
    html = html.replace("<title>deciserv plays", "<title>deciserv plays")  # title follows wordmark
    # index page hero copy names his server; say ours, credit his build
    html = html.replace("Live when the arbiter server is running on this origin; each page falls back to a recorded run otherwise.",
                        f"Live against DeciServ on this origin; each page falls back to a recorded run otherwise. Showcase by 0xBakeer: {CREDIT_LINK}")
    # model naming inside status/engine strings
    html = html.replace("'laya-' + res.routing.model", "'deciserv-' + res.routing.model")
    return html.encode("utf-8")


def proxy_decide(body: bytes) -> tuple[bytes, int]:
    """POST /v1/systemone -> DeciServ /decide. Shapes his response envelope."""
    payload = json.loads(body or b"{}")
    # strip Jev-only fields our gate doesn't accept
    gate_body = {"state": payload.get("state", ""),
                 "questions": payload.get("questions") or {}}
    t0 = time.time()
    req = urllib.request.Request(
        GATE.rstrip("/") + "/decide", data=json.dumps(gate_body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())
        _ms = out.get("latency_ms")
    except Exception as exc:
        note_adapter(parse_gate_questions(payload), t0, None, None, status=502, error=exc)
        raise
    gate_ms = _ms
    if gate_ms is None:
        gate_ms = (time.time() - t0) * 1000.0
    note_adapter(parse_gate_questions(payload), t0, time.time(), gate_ms)
    resp = {
        "model": "deciserv-laya",
        "answers": out.get("answers", {}),
        "usage": {"input_tokens": out.get("input_tokens", 0), "output_tokens": 0},
        "routing": {"model": "laya", "engine": "deciserv"},
        "latency_ms": round(float(gate_ms), 2),
    }
    return json.dumps(resp).encode(), 200


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: A003 — quiet, signature-compatible
        pass

    def _send(self, data: bytes, status=200, ctype="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/readyz", "/healthz"):
            self._send(json.dumps({"status": "ready", "models": ["deciserv"]}).encode())
            return
        # OTA instrumentation endpoints (additive; never touches game routes)
        if path == "/__ota__/latency_client.mjs":
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            game = (q.get("game") or ["unknown"])[0]
            tick = (q.get("tick") or [""])[0]
            self._send(ota_js("latency_client")
                       .replace("var game = q.game || 'unknown';",
                                f"var game = {json.dumps(game)};")
                       .replace("var tick = parseFloat(q.tick) || null;",
                                f"var tick = parseFloat({json.dumps(tick)}) || null;").encode(),
                       200, "text/javascript")
            return
        if path == "/__ota__/beacon":
            n = int(self.headers.get("Content-Length") or 0)
            try:
                note_browser(json.loads(self.rfile.read(n) or b"{}"))
            except Exception:  # noqa: BLE001 — never break serving on beacon junk
                pass
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "13")
            self.end_headers()
            self.wfile.write(b'{"ok":true}\n\n')
            return
        if path == "/__ota__/health.json":
            self._send(json.dumps(ota_health()).encode())
            return
        # /showcase (no trailing slash) → /showcase/ ; also 301 / → /showcase/
        if path == "/showcase":
            self.send_response(301)
            self.send_header("Location", "/showcase/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path == "/":
            self.send_response(301)
            self.send_header("Location", "/showcase/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        # static showcase files
        root = SHOWCASE.resolve()
        rel = path.lstrip("/")
        if rel.startswith("showcase/"):
            rel = rel[len("showcase/"):]
        target = (root / rel).resolve()
        if rel in ("", "/"):
            target = root / "index.html"
        elif target.is_dir() or not rel:
            target = target / "index.html"
        if target == root or not str(target).startswith(str(root)) or not target.is_file():
            self._send(b"not found", 404, "text/plain")
            return
        ctype = TYPES.get(target.suffix, "application/octet-stream")
        body = target.read_bytes()
        if ctype == "text/html":
            body = rebrand(body)
        self._send(body, 200, ctype)

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0]
        # OTA beacon endpoint (browser timing events; additive, never game traffic)
        if path == "/__ota__/beacon":
            n = int(self.headers.get("Content-Length") or 0)
            try:
                note_browser(json.loads(self.rfile.read(n) or b"{}"))
            except Exception:  # noqa: BLE001 — never break serving on beacon junk
                pass
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "13")
            self.end_headers()
            self.wfile.write(b'{"ok":true}\n\n')
            return
        if path not in ("/v1/systemone", "/v1/predict"):
            self._send(b"not found", 404, "text/plain")
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            data, status = proxy_decide(self.rfile.read(length))
        except urllib.error.URLError as exc:
            data = json.dumps({"error": {"kind": "gate_unreachable",
                                         "message": str(exc)}}).encode()
            status = 502
        except Exception as exc:  # noqa: BLE001
            data = json.dumps({"error": {"kind": "internal", "message": str(exc)}}).encode()
            status = 500
        self._send(data, status)


def main(argv=None):
    global GATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--gate", default=GATE)
    ap.add_argument("--bind", default="0.0.0.0",
                    help="bind address (0.0.0.0 = LAN-reachable, e.g. phone)")
    args = ap.parse_args(argv)
    GATE = args.gate
    ota_install()
    srv = ThreadingHTTPServer((args.bind, args.port), Handler)
    host = "127.0.0.1" if args.bind in ("127.0.0.1", "localhost") else args.bind
    print(f"showcase over DeciServ: http://{host}:{args.port}/showcase/  (gate {GATE})")
    print("open: /showcase/snake/  /showcase/hopper/  /showcase/crossing/")
    print("      /showcase/paddle/ /showcase/mines/  /showcase/dungeon/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())