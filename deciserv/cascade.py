"""L0 cascade — CPU fast-path benign gate for DeciServ (velto pattern).

Runs BEFORE the model layer in the /decide path. The existing 3-layer gate
(deciserv/policy.py) already owns the destructive side (fast path + text
floors); the cascade owns the BENIGN bulk: states the cheap rules can allow
outright, so they never reach the GPU model.

Contract with server.py (drop-in, fail-safe):
    gate = _cascade.decide(state, questions, call_model=call_model) \\
           or _policy.evaluate(state, questions, call_model=call_model)

decide() returns None for ANY state it will not allow — the request then flows
through the unchanged 3-layer gate (identical behavior to today). When it DOES
decide, it returns a policy-evaluate-shaped dict:

    {"verdict": "allow", "final_risk": 0.0, "layer": "cascade",
     "text_floor": None, "rule": "read_only_benign", "reason": "..."}

layer == "cascade" is the battery marker: server.py's DecisionLog already logs
gate["layer"] verbatim, so cascade decisions separate cleanly in the decision
log and the harness can split them from fast_path/text_floor/model rows.

Rules, in order:
  R1 read_only_benign (pure CPU, the benign bulk):
       policy.is_read_only(text) is True
       AND policy.floor_for(text) raises no floor (== 0.0)
       AND text length is within the configured bound
       -> allow, final_risk 0.0. No model call, no GPU.
  R2 margin_gap (OPT-IN, default OFF): for grey-middle states (not read-only,
     no floor) the cascade may consult the model ONCE via call_model and allow
     when EVERY answer clears the configured confidence floor
     (cascade_conf_floor) AND top1-top2 probability margin (cascade_gap_floor)
     AND leans benign (nouls lean false/yes-low, score stays low) AND the
     policy composite stays under policy.ASK_AT. When R2 declines, decide()
     returns None and the full gate re-runs (which re-calls the model — tune
     the floors on decision-log data before enabling on hot paths).

Thresholds (never hard-coded in logic): JSON file, path from env
DECISERV_THRESHOLDS, default thresholds.json beside server.py. Missing file,
invalid JSON, or cascade_enabled != true => cascade disabled (fail-safe to
current behavior). Recognized keys (all optional when the file exists):

    cascade_enabled       bool   default false   (must be explicitly true)
    cascade_conf_floor    float  default 0.60
    cascade_gap_floor     float  default 0.15
    cascade_margin_gap    bool   default false   (enables R2)
    cascade_max_text_len  int    default 400     (chars; longer -> full gate)

This module never raises through decide(): any internal error counts as a
decline and the request falls through to the gate.
"""
from __future__ import annotations
import json
import os
import threading

from . import policy as _policy

# --------------------------------------------------------------------------
# thresholds file (mtime-cached; edits apply without a restart)
_thr_lock = threading.Lock()
_thr_cache = {"key": None, "mtime": None, "thr": {}}

DEFAULTS = {
    "cascade_enabled": False,
    "cascade_conf_floor": 0.60,
    "cascade_gap_floor": 0.15,
    "cascade_margin_gap": False,
    "cascade_max_text_len": 400,
}


def thresholds_path() -> str:
    """DECISERV_THRESHOLDS env override, else thresholds.json beside server.py
    (this module ships in the same directory as server.py)."""
    env = os.environ.get("DECISERV_THRESHOLDS", "").strip()
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "thresholds.json")


def _as_bool(v, default):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "on")
    return default


def _as_float(v, default, lo=0.0, hi=1.0):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    return min(max(f, lo), hi)


def _as_int(v, default, lo=1):
    try:
        n = int(v)
    except (TypeError, ValueError):
        return default
    return max(n, lo)


def thresholds() -> dict:
    """Raw (validated) thresholds dict; {} when the file is missing/invalid —
    the fail-safe 'cascade disabled' state."""
    path = thresholds_path()
    try:
        mtime = os.stat(path).st_mtime_ns
    except OSError:
        return {}  # no file -> cascade disabled (never hard-coded defaults)
    with _thr_lock:
        if _thr_cache["key"] == path and _thr_cache["mtime"] == mtime:
            return _thr_cache["thr"]
    raw = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            raw = loaded
    except Exception:  # noqa: BLE001 — invalid file is 'disabled', never a crash
        raw = {}
    with _thr_lock:
        _thr_cache.update({"key": path, "mtime": mtime, "thr": raw})
    return raw


def config() -> dict:
    """Effective cascade config (defaults fill only keys absent from an
    EXISTING file; a missing file disables the cascade outright)."""
    thr = thresholds()
    if not thr:
        return {**DEFAULTS, "enabled": False}
    return {
        "enabled": _as_bool(thr.get("cascade_enabled"), False),
        "conf_floor": _as_float(thr.get("cascade_conf_floor"), DEFAULTS["cascade_conf_floor"]),
        "gap_floor": _as_float(thr.get("cascade_gap_floor"), DEFAULTS["cascade_gap_floor"]),
        "margin_gap": _as_bool(thr.get("cascade_margin_gap"), DEFAULTS["cascade_margin_gap"]),
        "max_text_len": _as_int(thr.get("cascade_max_text_len"),
                                DEFAULTS["cascade_max_text_len"]),
    }


# --------------------------------------------------------------------------
# counters (module-level; safe to expose later on /metrics)
_stats_lock = threading.Lock()
_stats = {"requests": 0, "rule_allow": 0, "margin_allow": 0, "decline": 0, "errors": 0}


def stats() -> dict:
    with _stats_lock:
        return dict(_stats)


# --------------------------------------------------------------------------
def _answers_of(res):
    """Provider results come back as {'answers': {...}} (/decide envelope) or a
    bare answers dict — same tolerance as policy.evaluate."""
    if not isinstance(res, dict):
        return None
    if "answers" in res and isinstance(res["answers"], dict):
        return res["answers"]
    return res


def margin_gap_accepts(answers: dict, cfg: dict) -> bool:
    """True when EVERY typed answer clears the parameterized margin-gap rule:
    confidence >= cascade_conf_floor, top1-top2 probability margin >=
    cascade_gap_floor, and the answer leans benign (noul p(yes) < 0.5, score
    below the top criterion). Pure function over synthetic or live dicts."""
    conf_floor = float(cfg.get("conf_floor", DEFAULTS["cascade_conf_floor"]))
    gap_floor = float(cfg.get("gap_floor", DEFAULTS["cascade_gap_floor"]))
    if not isinstance(answers, dict) or not answers:
        return False
    for _qid, ans in answers.items():
        if not isinstance(ans, dict):
            return False
        conf = ans.get("confidence")
        if isinstance(conf, bool) or not isinstance(conf, (int, float)):
            return False
        if not (0.0 <= float(conf) <= 1.0) or float(conf) < conf_floor:
            return False
        probs = ans.get("probabilities")
        if isinstance(probs, dict):
            vals = [float(v) for v in probs.values()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
        elif isinstance(probs, list):
            vals = [float(v) for v in probs
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
        else:
            return False
        if len(vals) < 2:            # margin needs a runner-up
            return False
        vals.sort(reverse=True)
        if (vals[0] - vals[1]) < gap_floor:
            return False
        qtype = ans.get("type", "noul")
        if qtype == "noul":
            p_yes = None
            if isinstance(probs, dict):
                for k in ("yes", "true"):
                    if k in probs:
                        p_yes = float(probs[k])
                        break
            if p_yes is None:
                v = ans.get("noul")
                if isinstance(v, str):
                    p_yes = 1.0 if v.strip().lower() in ("true", "yes") else 0.0
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    p_yes = float(v)
            if p_yes is not None and p_yes >= 0.5:
                return False
        elif qtype == "score":
            s = ans.get("score")
            if isinstance(s, (int, float)) and not isinstance(s, bool) and float(s) >= 2.0:
                return False
    return True


def decide(state, questions=None, call_model=None, mode=None):
    """L0 cascade verdict for one /decide request.

    Returns a policy-evaluate-shaped dict with layer "cascade" when a rule
    allows, else None (fall through to the 3-layer gate — never blocks, never
    escalates, never raises).
    """
    try:
        cfg = config()
        if not cfg.get("enabled"):
            return None
        mode = (mode or _policy.POLICY_MODE or "hybrid").strip().lower()
        if mode != "hybrid":
            return None
        text = state if isinstance(state, str) else str(state)
        if not text.strip() or len(text) > int(cfg.get("max_text_len", 400)):
            return None
        floor, _why = _policy.floor_for(text)
        if floor > 0.0:
            return None                      # floors belong to the gate's layer 2
        with _stats_lock:
            _stats["requests"] += 1
        # R1 — the benign bulk: read-only verbs, no floor, length in bounds.
        if _policy.is_read_only(text):
            with _stats_lock:
                _stats["rule_allow"] += 1
            return {"verdict": "allow", "final_risk": 0.0, "layer": "cascade",
                    "text_floor": None, "rule": "read_only_benign",
                    "reason": "cascade: read-only verbs only, no text floor, length in bounds"}
        # R2 — margin-gap (opt-in): one model pass, allow only on a wide,
        # confident, benign-leaning margin below the ask band.
        if cfg.get("margin_gap") and call_model is not None:
            merged = dict(_policy.QUESTIONS)
            for q, spec in (questions or {}).items():
                merged.setdefault(q, spec)
            answers = _answers_of(call_model(text, merged))
            if answers and margin_gap_accepts(answers, cfg):
                risk = _policy.model_weighted_risk(answers)
                if risk < _policy.ASK_AT:
                    with _stats_lock:
                        _stats["margin_allow"] += 1
                    return {"verdict": "allow", "final_risk": round(float(risk), 4),
                            "layer": "cascade", "text_floor": None, "rule": "margin_gap",
                            "answers": answers,
                            "reason": (f"cascade: margin-gap allow (conf>={cfg['conf_floor']}, "
                                       f"gap>={cfg['gap_floor']}, risk {risk:.2f} < ask "
                                       f"{_policy.ASK_AT:.2f})")}
        with _stats_lock:
            _stats["decline"] += 1
        return None
    except Exception:  # noqa: BLE001 — fail-safe: cascade never breaks the gate
        with _stats_lock:
            _stats["errors"] += 1
        return None