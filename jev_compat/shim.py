#!/usr/bin/env python3
"""Jev-compat shim (PROOF OF CONCEPT — /tmp only, not a production path).

Fronts DeciServ POST /decide on the DGX Spark with the hosted Jev API surface:
    POST /v1/systemone  {state, model, questions} -> {model, usage, answers}
    GET  /v1/models     -> catalog style response
so skills that hardcode POST {base}/v1/systemone can run against Laya via a
base-URL env override. Auth: any non-empty Bearer accepted (dummy keys fine).

Response normalization (hosted shape, per semdecide's strict validator and
the skill-side parsers read on 2026-09-24):
  * answers filtered to exactly the requested keys (hybrid gate may add
    policy-set extras like destroys_data/what_is_lost — semdecide errors on
    unexpected keys)
  * noul -> float (DeciServ hybrid synth emits "true"/"false" strings)
  * choice -> option-name string + probabilities dict (synth may emit an
    index int and "side: text" keys — remapped positionally to criteria keys)
  * score -> number + probabilities ARRAY by default (hosted shape; readable
    both as list and via get(str(i))); env DECISERV_SHIM_SCORE_PROBS=dict
    switches to {"0": p, ...} for map-shaped consumers
  * top-level model/usage/decision_id/latency_ms/provider added
"""
import json
import os
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DECIDESERV = os.environ.get("DECISERV_SHIM_UPSTREAM", "http://192.168.2.185:8710/decide")
MODEL_NAME = os.environ.get("DECISERV_SHIM_MODEL", "laya-gate-v1-merged (deciserv)")
SCORE_PROBS = os.environ.get("DECISERV_SHIM_SCORE_PROBS", "list").strip().lower()


def _stringify_state(state):
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def _norm_answer(ans, qspec):
    if not isinstance(ans, dict):
        return None
    qtype = qspec.get("type", "noul")
    out = {"type": qtype}
    if qtype == "noul":
        v = ans.get("noul", ans.get("choice"))
        if isinstance(v, str):
            v = 1.0 if v.strip().lower() == "true" else 0.0
        try:
            out["noul"] = float(v)
        except (TypeError, ValueError):
            return None
        return out
    if qtype == "choice":
        crit = qspec.get("criteria")
        opt_names = list(crit.keys()) if isinstance(crit, dict) else list(qspec.get("options") or [])
        choice = ans.get("choice")
        if isinstance(choice, int) and opt_names:
            choice = opt_names[choice] if 0 <= choice < len(opt_names) else str(choice)
        probs = ans.get("probabilities") or {}
        if isinstance(probs, list):
            probs = {opt_names[i] if i < len(opt_names) else str(i): p for i, p in enumerate(probs)}
        # synth path keys look like "side: text" — reduce to option names
        if crit and isinstance(probs, dict) and not all(k in crit for k in probs):
            probs = {str(k).split(":", 1)[0].strip(): v for k, v in probs.items()}
        try:
            conf = float(ans.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        out.update({"choice": choice, "probabilities": probs, "confidence": conf})
        return out
    if qtype == "score":
        crit = qspec.get("criteria") or []
        n = max(len(crit), 1)
        probs = ans.get("probabilities") or []
        if isinstance(probs, dict):
            probs = [probs.get(str(i), 0.0) for i in range(n)]
        probs = [float(p) for p in list(probs)[:n]]
        while len(probs) < n:
            probs.append(0.0)
        try:
            score = float(ans.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        try:
            conf = float(ans.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        out["score"] = score
        out["probabilities"] = {str(i): p for i, p in enumerate(probs)}
        if crit:
            out["legend"] = {str(i): str(lbl) for i, lbl in enumerate(crit)}
        out["confidence"] = conf
        return out
    return None


class Handler(BaseHTTPRequestHandler):
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
        if self.path in ("/v1/models", "/models"):
            self._send(200, {"object": "list",
                             "data": [{"id": "laya-gate-v1-merged", "object": "model",
                                       "owned_by": "deciserv"}]})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/systemone", "/systemone"):
            self.send_response(404)
            self.end_headers()
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            state = _stringify_state(req.get("state"))
            questions = req.get("questions") or {}
            t0 = time.time()
            upstream = urllib.request.Request(
                DECIDESERV,
                data=json.dumps({"state": state, "questions": questions}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(upstream, timeout=25) as resp:
                raw = json.loads(resp.read())
            answers = {}
            full = raw.get("answers") or {}
            for qid, qspec in questions.items():
                norm = _norm_answer(full.get(qid), qspec if isinstance(qspec, dict) else {})
                if norm is not None:
                    answers[qid] = norm
            dt = (time.time() - t0) * 1000
            est_in = max(len(state) // 4, 1)
            est_out = 8 * max(len(answers), 1)
            self._send(200, {
                "model": MODEL_NAME,
                "usage": {"input_tokens": est_in, "output_tokens": est_out},
                "answers": answers,
                "decision_id": raw.get("decision_id"),
                "provider": raw.get("provider"),
                "latency_ms": raw.get("latency_ms", round(dt, 1)),
            })
        except Exception as e:  # noqa: BLE001
            self._send(502, {"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("DECISERV_SHIM_PORT", "8711"))
    print(f"jev-shim :{port} -> {DECIDESERV} (score_probs={SCORE_PROBS})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()