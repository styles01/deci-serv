"""dcache — thread-safe LRU+TTL decision cache for DeciServ's /decide.

Memoizes whole /decide RESPONSES (the gate envelope, minus per-request fields)
keyed by sha256(state_text + question + options_json + model_revision). The
key is canonical JSON of the full request questions dict — question names,
types, criteria/options and all — so any question-shape change is a new key,
never a stale hit.

Integration (drop-in, default OFF until DECISERV_CACHE=1):
    cached = _dcache.get(state, questions, model_revision)
    if cached is not None: return cached            # HIT
    ... existing gate/model path ...
    _dcache.put(state, questions, model_revision, res_v2)

  * HIT responses get latency_ms preserved from the ORIGINAL call plus
    cache: "hit" — every other field identical to the stored response.
  * put() strips decision_id (per-request identity) and latency_ms before
    storing; get() re-stamps the ORIGINAL latency_ms and adds cache:"hit".
    On a hit the caller must NOT re-record the decision into the JSONL log
    (skip the _LOG.record under the hit branch) — the original decision is
    already logged.
  * Bypass: any request whose JSON questions blob contains
    "nondeterministic" (marker inside a question spec, e.g. a
    "nondeterministic": true key) is never cached and never served —
    non-deterministic states must hit the model every time. Also bypassed:
    DECISERV_CACHE off, provider/model_revision unknown ("unknown"), and
    empty state.

Config (env only, read at first use; DECISERV_CACHE_TTL_S / _MAX default
3600 s / 10 000 entries). Counters are exposed for the existing stats shape:

    with _dcache._lock: hits = _dcache.HITS ...

or the snapshot helper dcache.snapshot() -> {"cache_hits", "cache_misses",
"cache_hit_rate"} for the /stats (== /metrics) endpoint.
"""
from __future__ import annotations
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict

_lock = threading.Lock()          # guards MAX/TTL + counters + the OrderedDict
_MAX = int(os.environ.get("DECISERV_CACHE_MAX", "10000") or 10000)
_TTL = float(os.environ.get("DECISERV_CACHE_TTL_S", "3600") or 3600)
ENABLED_ENV = "DECISERV_CACHE"

_store: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
counters = {"cache_hits": 0, "cache_misses": 0}
# requests that were never eligible (cache off / nondeterministic / bad key)
counters["cache_bypasses"] = 0


def enabled() -> bool:
    """DECISERV_CACHE=0/1; default OFF until explicitly enabled."""
    v = (os.environ.get(ENABLED_ENV, "0") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _boolish(obj) -> bool:
    """True when the object graph mentions non-determinism anywhere in its
    JSON (covers question specs carrying "nondeterministic": true)."""
    try:
        blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).lower()
    except (TypeError, ValueError):
        return True  # unserializable -> treat as nondeterministic (bypass)
    return "nondeterministic" in blob or "non-deterministic" in blob


def key(state, questions, model_revision: str) -> str | None:
    """sha256 over state_text + canonical questions JSON + model_revision.
    None when the request is not cacheable (nondeterministic marker, empty
    state, unknown model_revision)."""
    if not isinstance(model_revision, str) or not model_revision or model_revision == "unknown":
        return None
    if isinstance(state, str):
        state_text = state
    else:
        try:
            state_text = json.dumps(state, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return None
    if not state_text.strip():
        return None
    if _boolish(questions) or _boolish(state):
        return None
    try:
        q_blob = json.dumps(questions, sort_keys=True, ensure_ascii=False,
                            separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    h = hashlib.sha256()
    h.update(state_text.encode("utf-8", "replace"))
    h.update(b"\x1e")
    h.update(q_blob.encode("utf-8", "replace"))
    h.update(b"\x1e")
    h.update(model_revision.encode("utf-8", "replace"))
    return h.hexdigest()


def get(state, questions, model_revision: str):
    """Stored envelope for this exact request, or None (miss/bypass/disabled).
    Never raises; on any internal error returns None (cache is best-effort)."""
    if not enabled():
        return None
    k = None
    try:
        k = key(state, questions, model_revision)
    except Exception:  # noqa: BLE001
        k = None
    if k is None:
        with _lock:
            counters["cache_bypasses"] += 1
        return None
    now = time.monotonic()
    with _lock:
        item = _store.get(k)
        if item is None:
            counters["cache_misses"] += 1
            return None
        expires, payload = item
        if now >= expires:
            _store.pop(k, None)          # TTL-expired: counts as a miss
            counters["cache_misses"] += 1
            return None
        _store.move_to_end(k)            # LRU refresh on hit
        counters["cache_hits"] += 1
    out = dict(payload)                  # shallow copy; caller may mutate
    orig = out.pop("orig_latency_ms", None)   # restore the ORIGINAL latency:
    if orig is not None:                      # a hit must still carry latency_ms
        out["latency_ms"] = orig              # (contract requires the field)
    out["cache"] = "hit"
    return out


def put(state, questions, model_revision: str, response: dict) -> bool:
    """Store a /decide response envelope. Strips decision_id + latency_ms
    (per-request fields); get() restores latency_ms from the stored 'orig'
    latency. No-op when disabled/bypassed. Never raises."""
    if not enabled() or not isinstance(response, dict):
        return False
    k = None
    try:
        k = key(state, questions, model_revision)
    except Exception:  # noqa: BLE001
        k = None
    if k is None:
        return False
    stored = {k2: v for k2, v in response.items() if k2 not in ("decision_id", "latency_ms", "cache")}
    if "answers" not in stored:          # only memoize decision responses
        return False
    try:
        orig_lat = float(response.get("latency_ms", 0.0) or 0.0)
    except (TypeError, ValueError):
        orig_lat = 0.0
    stored["orig_latency_ms"] = orig_lat
    with _lock:
        if k in _store:
            _store.move_to_end(k)
        _store[k] = (time.monotonic() + _TTL, stored)
        while len(_store) > max(_MAX, 1):
            _store.popitem(last=False)   # evict LRU
    return True


def snapshot() -> dict:
    """Counters for the /stats (== /metrics) response — the exact keys the
    task spec names: cache_hits, cache_misses, cache_hit_rate."""
    with _lock:
        hits = counters["cache_hits"]
        misses = counters["cache_misses"]
        total = hits + misses
        return {
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_hit_rate": round(hits / total, 4) if total else 0.0,
            "cache_size": len(_store),
            "cache_bypasses": counters["cache_bypasses"],
            "cache_enabled": enabled(),
            "cache_ttl_s": _TTL,
            "cache_max_entries": _MAX,
        }


def reset_stats() -> None:
    """Counter reset for /metrics/reset (the store itself is kept)."""
    with _lock:
        counters["cache_hits"] = 0
        counters["cache_misses"] = 0
        counters["cache_bypasses"] = 0


def clear() -> None:
    """Drop every entry (used by tests / model_revision changes)."""
    with _lock:
        _store.clear()


class _Disable:
    """Context manager to run with the cache force-disabled (tests)."""

    def __enter__(self):
        self._old = os.environ.get(ENABLED_ENV)
        os.environ[ENABLED_ENV] = "0"
        return self

    def __exit__(self, *a):
        if self._old is None:
            os.environ.pop(ENABLED_ENV, None)
        else:
            os.environ[ENABLED_ENV] = self._old
        return False