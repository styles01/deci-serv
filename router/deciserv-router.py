#!/usr/bin/env python3
"""deciserv-router.py — v0 intent router, SHADOW MODE (never acts, only logs).

Serves POST /route  {message, session_id} -> {lane, confidence, layer, shadow: true}
Lanes: "tool" (stateless lookup/tool-call), "browser" (web fetch/interaction),
       "reason" (escalate to LLM — anything ambiguous, contextual, or analytical).
Shadow contract: ALWAYS shadow=true; the caller's real path is unchanged. We only log.

3-layer design mirrors the gate:
  L1 fast rules (regex, no model, <1ms)
  L2 heuristic floors (context-dependence markers -> reason)
  L3 model grey-middle (STUB for v0 — logs lane=reason w/ layer=stub)

Shadow log: JSONL, one record per route: ts, message, lane, layer, confidence.
Run:  python3 deciserv-router.py --port 8711 --log /home/jaita/router-data/shadow.jsonl
"""
import argparse, json, os, re, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import Counter

# ---------- L1 fast rules ----------
TOOL_PATTERNS = [
    r"^(what(?:'s| is)|who(?:'s| is)|when|where) (?:the |my )?\w[\w .'-]{0,40}\??$",   # bare fact lookup
    r"^(?:what(?:'s| is) (?:the )?(?:price|weather|time|date|version|status))(?: of| for)?",
    r"^(?:check|fetch|get|look ?up|show) (?:the )?(?:price|stock|weather|score|status|version|ip|time)",
    r"^(?:run|exec) \w[\w./-]*",                       # explicit command exec
    r"^(?:ls|cat|grep|find|curl|wget|git (?:status|log|diff|branch))\b",  # shell verbs
    r"^(?:send|post|tweet|email) .{3,120}$",           # explicit outbound action
]
BROWSER_PATTERNS = [
    r"\b(?:open|go to|visit|browse|read)\b .*\b(?:https?://|www\.|\w+\.(?:com|org|net|io|ai|dev))\b",
    r"\b(?:search|google|look ?up|find) (?:the web|online|on (?:google|the internet))\b",
    r"^(?:browse|scrape|read) .*(?:page|site|article|docs|documentation)",
]
# ---------- L2 heuristic floors ----------
CONTEXT_MARKERS = re.compile(
    r"\b(?:it|that|this|those|these|he|she|they|again|also|instead|previous|earlier|last one|same)\b|"
    r"\bwhy\b|\bhow come\b|\bexplain\b|\banalyz|\bcompar\b|\bdesign\b|\breview\b|\bopinion\b|\bthink\b|\bplan\b|\bdebug\b|\brefactor\b|\bsummariz|\bwrite\b|\bdraft\b",
    re.I,
)

def classify(message: str):
    m = message.strip()
    low = m.lower()
    if CONTEXT_MARKERS.search(low):
        return "reason", "floor", 0.55
    for pat in BROWSER_PATTERNS:
        if re.search(pat, low):
            return "browser", "fast", 0.85
    for pat in TOOL_PATTERNS:
        if re.search(pat, low):
            return "tool", "fast", 0.80
    return "reason", "default", 0.50   # v0: L3 model layer stub — default to escalation

class Router(BaseHTTPRequestHandler):
    log_path = "/home/jaita/router-data/shadow.jsonl"
    stats = Counter()

    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({"status": "ok", "mode": "shadow", "routes": sum(Router.stats.values())}).encode()
        elif self.path == "/stats":
            body = json.dumps({"lanes": dict(Router.stats), "mode": "shadow"}).encode()
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/route":
            self.send_response(404); self.end_headers(); return
        t0 = time.time()
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            req = {}
        message = str(req.get("message", ""))[:2000]
        lane, layer, conf = classify(message)
        Router.stats[f"{layer}:{lane}"] += 1
        rec = {"ts": round(time.time(), 3), "message": message, "lane": lane,
               "layer": layer, "confidence": conf, "latency_ms": round((time.time()-t0)*1000, 2),
               "session": req.get("session", "")}
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        body = json.dumps({"lane": lane, "confidence": conf, "layer": layer, "shadow": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # silence default stderr logging
        pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8711)
    ap.add_argument("--log", default="/home/jaita/router-data/shadow.jsonl")
    args = ap.parse_args()
    Router.log_path = args.log
    srv = HTTPServer(("0.0.0.0", args.port), Router)
    print(f"deciserv-router shadow mode on :{args.port} -> {args.log}", flush=True)
    srv.serve_forever()