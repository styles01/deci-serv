#!/usr/bin/env python3
"""DeciServ — a PyTorch server for System-1 decision models (local serving).

Small, fast, non-autoregressive classifier models that return calibrated,
typed probabilities in a single forward pass. DeciServ loads one such model
resident on a CUDA device and serves it over localhost HTTP with a Jev-shaped
contract:

    POST /decide   {"state": "...", "questions": {q: {type, instructions, criteria}}}
                   -> {"answers": {q: {type, choice|score, probabilities, confidence}}}

    GET  /health   -> {"status": "ok", "model": ..., "precision": ..., "device": ...}
    GET  /metrics  -> {"calls", "p_avg_ms", "p_max_ms", "errors", "gpu_peak_gib"}

Design principles:
  * stdlib HTTP (ThreadingHTTPServer) — zero web-framework dependency
  * fp16/bf16 on CUDA by default (half the resident memory of fp32)
  * torch.inference_mode() on every pass
  * provider architecture: model backends are pluggable classes; Laya ships in-box
  * never generates text — decision gates cannot hallucinate

Usage:
    python -m deciserv.server --checkpoint /path/to/model --port 8710 --precision fp16
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, threading, uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .providers import get_provider

_stats = {"calls": 0, "lat_ms_sum": 0.0, "lat_max": 0.0, "err": 0}
_stats_lock = threading.Lock()

# Stratified metrics: per question-type x decision-class bins (R3 /metrics v2)
_strat = {}  # {question_name: {"n": int, "lat_ms_sum": float, "conf_sum": float}}
_strat_lock = threading.Lock()

class DecisionLog:
    """Persistent JSONL decision log (v2). Logs the decision, never the payload
    (muse-jev-playbook hard rule 4): state stored as SHA-256 hash + length only,
    raw probabilities kept — they are the training corpus in waiting."""
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.Lock()
    def record(self, entry: dict):
        with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
_LOG = None  # set at startup

def _state_hash(state) -> str:
    s = json.dumps(state, sort_keys=True, ensure_ascii=False) if not isinstance(state, str) else state
    return hashlib.sha256(s.encode()).hexdigest()[:16]


class DeciHandler(BaseHTTPRequestHandler):
    provider = None  # class attr set at startup

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", **self.provider.describe()})
        elif self.path == "/metrics":
            with _stats_lock:
                n = max(_stats["calls"], 1)
                m = {"calls": _stats["calls"],
                     "p_avg_ms": round(_stats["lat_ms_sum"] / n, 1),
                     "p_max_ms": round(_stats["lat_max"], 1),
                     "errors": _stats["err"]}
            with _strat_lock:
                m["by_question"] = {q: {"n": d["n"],
                                        "avg_ms": round(d["lat_ms_sum"] / max(d["n"], 1), 1),
                                        "avg_conf": round(d["conf_sum"] / max(d["n"], 3))}
                                    for q, d in sorted(_strat.items())}
            m.update(self.provider.memory_stats())
            self._send(200, m)
        elif self.path == "/metrics/reset":
            with _stats_lock:
                _stats.update({"calls": 0, "lat_ms_sum": 0.0, "lat_max": 0.0, "err": 0})
            with _strat_lock:
                _strat.clear()
            self._send(200, {"status": "reset", "log_preserved": bool(_LOG)})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/decide":
            self.send_response(404)
            self.end_headers()
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            t0 = time.time()
            res = self.provider.decide(req["state"], req.get("questions", {}))
            dt = (time.time() - t0) * 1000
            with _stats_lock:
                _stats["calls"] += 1
                _stats["lat_ms_sum"] += dt
                _stats["lat_max"] = max(_stats["lat_max"], dt)
            with _strat_lock:
                for qname, ans in (res.get("answers") or {}).items():
                    d = _strat.setdefault(qname, {"n": 0, "lat_ms_sum": 0.0, "conf_sum": 0.0})
                    d["n"] += 1
                    d["lat_ms_sum"] += dt
                    d["conf_sum"] += float(ans.get("confidence") or 0.0)
            # v2 envelope: provenance + identity on every response (OpenClaw
            # DecisionProvider contract discipline).
            decision_id = f"d-{int(time.time()*1000):x}-{os.urandom(4).hex()}"
            policy = req.get("policy", "safety")
            res_v2 = {**res,
                      "decision_id": decision_id,
                      "provider": self.provider.describe().get("model", "unknown"),
                      "policy": policy,
                      "latency_ms": round(dt, 1)}
            if _LOG is not None:
                _LOG.record({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "decision_id": decision_id,
                    "provider": res_v2["provider"],
                    "policy": policy,
                    "state_hash": _state_hash(req.get("state")),
                    "questions": list((req.get("questions") or {}).keys()),
                    "answers": res.get("answers", {}),
                    "confidence": res.get("confidence"),
                    "latency_ms": round(dt, 1),
                })
            self._send(200, res_v2)
        except Exception as e:  # noqa: BLE001 — gate must answer, not crash
            with _stats_lock:
                _stats["err"] += 1
            self._send(500, {"error": str(e)})


def main(argv=None):
    ap = argparse.ArgumentParser(prog="deciserv",
                                 description="PyTorch server for System-1 decision models")
    ap.add_argument("--provider", default="laya", help="decision-model provider (default: laya)")
    ap.add_argument("--checkpoint", required=True, help="model checkpoint path or HF id")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8710)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-path", default=os.environ.get("DECISERV_LOG",
                    os.path.expanduser("~/deciserv-data/decisions/decisions.jsonl")),
                    help="persistent JSONL decision log path (state stored hashed)")
    args = ap.parse_args(argv)

    global _LOG
    _LOG = DecisionLog(args.log_path)

    from .providers import load_provider
    provider = load_provider(args.provider, args.checkpoint, precision=args.precision, device=args.device)
    DeciHandler.provider = provider
    desc = provider.describe()
    print(f"deciserv [{args.provider}] {desc.get('model')} precision={desc.get('precision')} "
          f"on :{args.port} (load {desc.get('load_s', '?')}s)", flush=True)
    ThreadingHTTPServer((args.host, args.port), DeciHandler).serve_forever()


if __name__ == "__main__":
    main()