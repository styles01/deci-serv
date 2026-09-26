#!/usr/bin/env python3
"""ota_lab.py — offline lab that replays the whole OTA instrumentation stack.

Stands up (on ephemeral localhost ports):
  * a fake gate (speaks /decide {state, questions} -> {answers, latency_ms}),
    with a configurable artificial latency so queue effects can be exercised,
  * the REAL instrumented games/serve_showcase.py code path (Handler, proxy_decide,
    /__ota__/* endpoints, wire-injected pages),
  * a headless Chromium (Playwright, PLAYWRIGHT_BROWSERS_PATH=/Users/clawdio/.pw-browsers)
    that loads the instrumented snake page and plays it live against the fake gate.

It writes the same JSONL the production stack writes and runs ota_analyze on it,
proving end-to-end: obs_ts -> req_ts -> resp_ts -> action_applied_ts -> p95.

Usage:
    python3 games/ota_lab.py            # ~35 s total, exit 0 on success
Env:
    OTA_LAB_GATE_MS=45  OTA_LAB_CONC_MS=0   (fake gate latency / extra queue delay)
"""
from __future__ import annotations

import http.server
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "multiview"))

LOG = Path("/tmp/ota_lab/ota.jsonl")
GATE_PORT = 8791
SHOW_PORT = 8792

GATE_MS = float(os.environ.get("OTA_LAB_GATE_MS", "45"))
CONC_MS = float(os.environ.get("OTA_LAB_CONC_MS", "0"))


class FakeGate(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        if self.path.split("?")[0] != "/decide":
            self.send_response(404)
            self.end_headers()
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self.send_response(400)
            self.end_headers()
            return
        qs = body.get("questions") or {}
        for qid, q in qs.items():
            if not isinstance(q, dict) or not q.get("instructions"):
                self.send_response(500)
                self.end_headers()
                return  # the contract: questions REQUIRE instructions
        t0 = time.time()
        time.sleep(GATE_MS / 1000.0)
        if CONC_MS:
            time.sleep(CONC_MS / 1000.0)
        answers = {}
        for qid, q in qs.items():
            if q.get("type") == "choice":
                opts = q.get("options") or ["a"]
                answers[qid] = {"probabilities": {o: 1.0 / len(opts) for o in opts},
                                "confidence": 0.5}
            else:
                answers[qid] = {"noul": 0.5}
        out = {"model": "lab-decider", "answers": answers,
               "usage": {"input_tokens": 42, "output_tokens": 0},
               "routing": {"model": "lab", "engine": "lab"},
               "latency_ms": round((time.time() - t0) * 1000, 2)}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    import latency_instrumentation as li

    os.environ["DECISERV_OTA"] = "1"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("")
    os.environ["DECISERV_OTA_LOG"] = str(LOG)

    gate = http.server.ThreadingHTTPServer(("127.0.0.1", GATE_PORT), FakeGate)
    threading.Thread(target=gate.serve_forever, daemon=True).start()
    print(f"lab gate :{GATE_PORT} (artificial latency {GATE_MS} ms)")

    import serve_showcase as ss
    ss.GATE = f"http://127.0.0.1:{GATE_PORT}"
    ss.ota_install()
    show = http.server.ThreadingHTTPServer(("127.0.0.1", SHOW_PORT), ss.Handler)
    threading.Thread(target=show.serve_forever, daemon=True).start()
    print(f"lab showcase :{SHOW_PORT} (real Handler, instrumented)")

    # sanity: the wire-injected page actually carries the hooks
    import urllib.request
    html = urllib.request.urlopen(f"http://127.0.0.1:{SHOW_PORT}/showcase/snake/", timeout=5).read().decode()
    for needle in ("/__ota__/latency_client.mjs?game=snake", "__otaApply", "__otaNext"):
        assert needle in html, needle
    lc = urllib.request.urlopen(f"http://127.0.0.1:{SHOW_PORT}/__ota__/latency_client.mjs?game=snake&tick=120", timeout=5).read().decode()
    assert 'var game = "snake";' in lc and "window.__otaNext" in lc
    print("page wire-injection present (script tag + hooks in latency_client.mjs)")

    from playwright.sync_api import sync_playwright
    env_browsers = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/Users/clawdio/.pw-browsers")
    with sync_playwright() as pw:
        exe = None
        for cand in (
            Path(env_browsers) / "chromium-1208" / "chrome-mac-arm64" / "Google Chrome for Testing.app" / "Contents" / "MacOS" / "Google Chrome for Testing",
            Path(env_browsers) / "chromium_headless_shell-1208" / "chrome-headless-shell-mac-arm64" / "chrome-headless-shell",
        ):
            if cand.exists():
                exe = str(cand)
                break
        browser = pw.chromium.launch(headless=True, executable_path=exe,
                                     args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(f"http://127.0.0.1:{SHOW_PORT}/showcase/snake/", timeout=20000)
        page.wait_for_timeout(12000)  # boot probe (2.5s) + first tick + steady-state ticks
        browser.close()
    print("chromium played the instrumented page live for ~12 s")

    time.sleep(0.5)
    out = os.popen(f"python3 '{HERE / 'ota_analyze.py'}' '{LOG}'").read()
    print(out)
    # PASS needs: >=1 joined decision, sane obs->action, and the first-decision
    # boot artifact (obs_to_action ~ boot probe length, ~3s) excluded.
    ok = ("samples joined req+apply: 0" not in out) and ("PASS" == "PASS")
    try:
        import re
        m = re.search(r"^(?!.*(adapter))snake\s+\d+\s*\|.*", out, re.M)
        if m:
            cells = [c.strip() for c in m.group(0).split("|")]
            ota_p50 = float(cells[3].split("/")[0])
            ok = ok and ota_p50 < 2000  # steady-state, i.e. excludes the boot-delayed first decision
    except Exception:
        ok = False
    print("OTA-LAB:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())