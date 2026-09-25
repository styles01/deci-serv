#!/usr/bin/env python3
"""deciserv/server_decider.py — DeciServ-compatible /decide endpoint backed by decider-2b.

Speaks the same wire contract as deciserv.server (POST /decide {state, questions} →
{answers, ...}) so the battery harness and serve_showcase adapter work unchanged.
Decider's native /v1/systemone contract is served as-is at /v1/systemone too.
"""
import json
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_PATH = "/home/jaita/models/hf/Mapika/decider-2b"
DECIDER = None  # set in main()
LOAD_S = 0.0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: A003
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
        if path in ("/readyz", "/healthz", "/health"):
            import torch
            free_b, total = torch.cuda.mem_get_info()
            self._send(json.dumps({
                "status": "ok", "model": "decider-2b (Mapika v11)",
                "precision": "bf16", "device": "cuda",
                "load_s": round(LOAD_S, 1),
                "pool_used_gb": round((total - free_b) / 1e9, 2),
            }).encode())
            return
        self._send(b'{"error": "not found"}', 404)

    def do_POST(self):  # noqa: N802
        global DECIDER
        path = self.path.split("?")[0]
        if path not in ("/decide", "/v1/systemone"):
            self._send(b'{"error": "not found"}', 404)
            return
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        try:
            req = json.loads(body or b"{}")
        except Exception:
            self._send(b'{"error": "bad json"}', 400)
            return
        t0 = time.time()
        try:
            if path == "/v1/systemone":
                # decider native: pass through, add latency
                out = DECIDER.system_one(req.get("state", ""), req.get("questions") or {})
                out["latency_ms"] = round((time.time() - t0) * 1000, 2)
                self._send(json.dumps(out, default=str).encode())
                return
            # /decide — DeciServ contract: questions {id: {type, instructions, options|criteria}}
            state = req.get("state", "")
            questions = req.get("questions") or {}
            # laya's criteria-only choices work on decider too: criteria dict → options
            norm = {}
            for qid, q in (questions or {}).items():
                q = dict(q)
                if not q.get("options") and isinstance(q.get("criteria"), dict):
                    q["options"] = list(q["criteria"].keys())
                norm[qid] = q
            out = DECIDER.system_one(state, norm)
            answers = out.get("answers", {})
            resp = {
                "model": "decider-2b",
                "answers": answers,
                "usage": {"input_tokens": (out.get("usage") or {}).get("input_tokens", 0),
                          "output_tokens": 0},
                "routing": {"model": "decider-2b", "engine": "deciserv"},
                "latency_ms": round((time.time() - t0) * 1000, 2),
            }
            self._send(json.dumps(resp, default=str).encode())
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self._send(json.dumps({"error": {"kind": "decider_error",
                                             "message": str(e)[:300]}}).encode(), 500)


def main():
    global DECIDER, LOAD_S
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8711
    t0 = time.time()
    sys.path.insert(0, MODEL_PATH)
    import torch
    # transformers 5.14 compat: causal_conv1d_fn is called with x= keyword
    import decider.engine as DE
    from decider.infer import Decider
    DECIDER = Decider(MODEL_PATH)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    def fused_compat(hidden_states=None, weight=None, bias=None, activation=None,
                     seq_idx=None, x=None, **kw):
        h = hidden_states if hidden_states is not None else x
        return DE.fused_causal_conv1d_fn(h, weight, bias, activation)

    mq.causal_conv1d_fn = fused_compat
    for mod in DECIDER.m.lm.modules():
        if type(mod).__name__ == "Qwen3_5GatedDeltaNet":
            mod.causal_conv1d_fn = fused_compat
    # warm
    DECIDER.system_one("warmup.", {"w": {"type": "choice", "instructions": "warm",
                                         "options": ["a", "b"]}})
    LOAD_S = time.time() - t0
    print(f"decider lane ready on :{port} (load {LOAD_S:.1f}s)", flush=True)
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.serve_forever()


if __name__ == "__main__":
    main()