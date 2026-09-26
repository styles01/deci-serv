#!/usr/bin/env python3
"""games/latency_instrumentation.py — observation-to-action (OTA) latency instrumentation.

Additive, stdlib-only measurement layer for the six-game decider setup
(0xBakeer's arbiter showcase, MIT, served over the DeciServ gate — see
games/LATENCY_INSTRUMENTATION.md for the full design).

What it measures, per decision:
    obs_ts            observation captured (game state final, decision requested)
    req_ts            request sent (browser -> :8010 adapter, and adapter -> gate)
    resp_ts           response received (pure-ish inference latency = gate latency_ms)
    action_applied_ts action applied in the game loop (next tick boundary)
Derivations: gate_inference_ms, queue decomposition (prequeue / postqueue / wait),
obs_to_action_ms, in_time flag vs the game's tick period, co_resident_load_hint
(EXL3 vLLM :8000 running/waiting requests + gate pool_used_gb, polled cheaply).

Design rules:
- Hooks are tiny and additive; all logic lives here.
- Never raises into the serving path: every public hook swallows its own errors.
- No service is restarted or reloaded by this module; it only runs when the
  (already instrumented) servers are (re)started by the operator.
- JSONL output, one event per line, merged into per-decision records by
  games/ota_analyze.py.

Environment:
    DECISERV_OTA=1            enable (default 1; set 0 to disable cleanly)
    DECISERV_OTA_DIR          output dir (default games/ota_latency)
    DECISERV_OTA_LOG          explicit JSONL path (overrides the dated default)
    DECISERV_OTA_SLO_MS       default obs->action SLO per game tick fraction (default 150)
    DECISERV_OTA_CO_RES       co-residency probe target (default http://192.168.2.185:8000)
    DECISERV_OTA_GATE_READYZ  gate readyz URL for pool_used_gb (default http://192.168.2.185:8712)

NOTE: probe URLs must use literal IPs, NOT *.local mDNS names — from this Mac a
fresh connection to larryspark.local costs ~3.0 s (link-local IPv6 connect
fallback) vs ~8 ms via 192.168.2.185. Measured 2026-09-26.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from pathlib import Path

GAMES_DIR = Path(__file__).resolve().parent
CELL_ORDER = ["snake", "paddle", "hopper", "crossing", "mines", "dungeon"]  # multiview grid order

_STATE = {"installed": False, "enabled": False}
_LOCK = threading.Lock()
_POLL = {"hint": None, "at": 0.0}


# ---------------------------------------------------------------- output ----

def _out_path() -> Path:
    log = os.environ.get("DECISERV_OTA_LOG")
    if log:
        return Path(log)
    d = Path(os.environ.get("DECISERV_OTA_DIR") or (GAMES_DIR / "ota_latency"))
    day = time.strftime("%Y%m%d")
    return d / day / "ota.jsonl"


def _append(record: dict) -> None:
    import fcntl
    path = _out_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, separators=(",", ":"), default=str)
        with open(path, "a", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.write(line + "\n")
            fh.flush()
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except Exception:  # noqa: BLE001 — instrumentation must never break serving
        pass


def _now_ms() -> float:
    return round(time.time() * 1000.0, 3)


def _slo_default() -> float:
    try:
        return float(os.environ.get("DECISERV_OTA_SLO_MS", "150"))
    except ValueError:
        return 150.0


# ------------------------------------------------- co-residency hint poll ----

def _poll_hint() -> dict:
    """Cheap cached probe: EXL3 vLLM :8000 /metrics + decider gate /readyz.

    Runs at most every 2 s in whichever thread hits a logging hook; never
    blocks serving beyond a 0.75 s socket timeout on the miss path.
    """
    now = time.time()
    with _LOCK:
        if _POLL["hint"] is not None and now - _POLL["at"] < 2.0:
            return _POLL["hint"]
    hint = {}
    url = os.environ.get("DECISERV_OTA_CO_RES", "http://192.168.2.185:8000")
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=0.75) as r:
            txt = r.read().decode("utf-8", "replace")
        for line in txt.splitlines():
            if line.startswith("vllm:num_requests_running ") or line.startswith("vllm:num_requests_running{"):
                hint["exl3_running"] = float(line.rsplit(" ", 1)[1])
            elif line.startswith("vllm:num_requests_waiting ") or line.startswith("vllm:num_requests_waiting{"):
                hint["exl3_waiting"] = float(line.rsplit(" ", 1)[1])
        hint["exl3"] = "up" if hint else "idle"  # reachable but no gauges parsed
    except Exception:  # noqa: BLE001
        hint["exl3"] = "down"
    rurl = os.environ.get("DECISERV_OTA_GATE_READYZ", "http://192.168.2.185:8712")
    try:
        with urllib.request.urlopen(rurl.rstrip("/") + "/readyz", timeout=0.75) as r:
            j = json.loads(r.read().decode("utf-8", "replace"))
        if "pool_used_gb" in j:
            hint["gate_pool_used_gb"] = j["pool_used_gb"]
    except Exception:  # noqa: BLE001
        pass
    with _LOCK:
        _POLL["hint"] = hint
        _POLL["at"] = now
    return hint


# ------------------------------------------------------------- lifecycle ----

def install() -> None:
    """Idempotent enable. Safe no-op when DECISERV_OTA=0. Never raises."""
    with _LOCK:
        if _STATE["installed"]:
            return
        _STATE["installed"] = True
        _STATE["enabled"] = os.environ.get("DECISERV_OTA", "1") != "0"
    if _STATE["enabled"]:
        _append({"ts": _now_ms(), "type": "session", "pid": os.getpid()})


def enabled() -> bool:
    return _STATE["enabled"]


# ------------------------------------------- adapter-side (serve_showcase) ----

def parse_gate_questions(payload: dict) -> dict:
    """Extract the client-side OTA context the instrumented pages attach.

    Returns a dict describing the request for the 'adapter' event; also
    returns the questions dict unchanged (serving code keeps using it).
    The gate contract requires instructions on every question — the pages
    already do this; we only COUNT them here.
    """
    try:
        questions = payload.get("questions") or {}
        n_q = len(questions)
        n_opts = 0
        for q in questions.values():
            opts = q.get("options") if isinstance(q, dict) else None
            if isinstance(opts, list):
                n_opts += len(opts)
        ctx = payload.get("_ota") or {}
        return {
            "game": str(ctx.get("game") or "unknown"),
            "page": str(ctx.get("page") or ""),
            "seq": ctx.get("seq"),
            "obs_ts": ctx.get("obs_ts"),
            "req_ts": ctx.get("req_ts"),
            "n_questions": n_q,
            "n_options": n_opts,
        }
    except Exception:  # noqa: BLE001
        return {"game": "unknown", "page": "", "seq": None, "obs_ts": None,
                "req_ts": None, "n_questions": 0, "n_options": 0}


def note_adapter(ctx: dict, t0: float, t_resp, gate_ms, status=200, error=None) -> None:
    """Log the adapter hop (browser->:8010->gate) for one decision."""
    if not _STATE["enabled"]:
        return
    try:
        rec = {
            "ts": _now_ms(), "type": "adapter",
            "game": ctx.get("game") or "unknown", "page": ctx.get("page") or "",
            "seq": ctx.get("seq"),
            "adapter_req_ts": round(t0 * 1000.0, 3),
            "n_questions": ctx.get("n_questions"), "n_options": ctx.get("n_options"),
            "status": status, "co_resident_load_hint": _poll_hint(),
        }
        if t_resp is not None:
            rec["adapter_resp_ts"] = round(t_resp * 1000.0, 3)
            rec["adapter_ms"] = round((t_resp - t0) * 1000.0, 3)
        if gate_ms is not None:
            rec["gate_inference_ms"] = float(gate_ms)
        if error:
            rec["error"] = str(error)[:160]
        _append(rec)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------- browser-side beacons ----

def note_browser(rec: dict) -> None:
    """Log one browser beacon ('req' from client.mjs, 'apply' from the page)."""
    if not _STATE["enabled"]:
        return
    try:
        if not isinstance(rec, dict):
            return
        out = {
            "ts": _now_ms(),
            "type": str(rec.get("type") or "unknown"),
            "game": str(rec.get("game") or "unknown"),
            "page": str(rec.get("page") or ""),
            "seq": rec.get("seq"),
        }
        for k in ("tick_ms", "obs_ts", "req_ts", "resp_ts", "gate_inference_ms",
                  "rt_ms", "action_applied_ts", "obs_to_action_ms",
                  "postqueue_ms", "wait_ms"):
            if rec.get(k) is not None:
                out[k] = rec[k]
        out["co_resident_load_hint"] = _poll_hint()
        _append(out)
    except Exception:  # noqa: BLE001
        pass


def health() -> dict:
    """Snapshot for GET /__ota__/health.json (no secrets, cheap)."""
    try:
        path = _out_path()
        n = 0
        try:
            with open(path, "rb") as f:
                for _ in f:
                    n += 1
        except OSError:
            pass
        return {"ok": True, "enabled": _STATE["enabled"], "log": str(path), "lines": n,
                "hint": _poll_hint()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


# ------------------------------------------------- injected browser module ----

_LATENCY_CLIENT = """/* latency_client.mjs — OTA instrumentation (DeciServ). Served by
   games/serve_showcase.py at /__ota__/latency_client.mjs; classic script so it
   defines window hooks before the page's module scripts run. See
   games/LATENCY_INSTRUMENTATION.md. */
(function () {
  'use strict';
  var src = (document.currentScript && document.currentScript.src) || '';
  var q = {};
  try { src.split('?')[1].split('&').forEach(function (kv) {
    var p = kv.split('='); q[p[0]] = decodeURIComponent(p[1] || ''); }); } catch (e) {}
  var game = q.game || 'unknown';
  var tick = parseFloat(q.tick) || null;
  var page = Math.random().toString(36).slice(2, 8);
  var origin = (window.performance && performance.timeOrigin) || 0;
  window.__ota = { game: game, page: page, tick: tick, seq: 0, last_seq: null,
                   obs_epoch: null, resp_epoch: null };
  window.__otaBeacon = function (obj) {
    try {
      obj.game = obj.game || game; obj.page = obj.page || page;
      fetch('/__ota__/beacon', { method: 'POST', keepalive: true,
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(obj) }).catch(function () {});
    } catch (e) {}
  };
  /* requestNext(state) hook: marks observation captured for a new decision */
  window.__otaNext = function () {
    var o = window.__ota;
    o.seq += 1; o.last_seq = o.seq;
    o.obs_epoch = Math.round(origin + (performance.now()));
    o.resp_epoch = null;
    return o.seq;
  };
  /* apply(res, now, tickStart) hook: marks the decision applied in the loop */
  window.__otaApply = function (now, tickStart) {
    var o = window.__ota;
    try {
      var applied = Math.round(origin + (now || performance.now()));
      var rec = { type: 'apply', ts: Date.now(), game: game, page: page,
                  seq: o.last_seq, action_applied_ts: applied };
      if (o.obs_epoch != null) rec.obs_to_action_ms = applied - o.obs_epoch;
      if (o.resp_epoch != null) rec.postqueue_ms = applied - o.resp_epoch;
      if (tickStart != null) rec.wait_ms = applied - Math.round(origin + tickStart);
      window.__otaBeacon(rec);
    } catch (e) {}
  };
})();
"""


def js(name: str) -> str:
    if name == "latency_client":
        return _LATENCY_CLIENT
    return "// unknown module\n"