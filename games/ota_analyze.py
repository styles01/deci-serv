#!/usr/bin/env python3
"""ota_analyze.py — aggregate games/ota_latency JSONL into the A/B report table.

Reads one or more ota.jsonl files (browser 'req' + 'apply' events, adapter
events), joins req<->apply per (game, page, seq), and prints per-game and
overall p50/p95/p99 for:

  gate_inference_ms   pure gate time (gate latency_ms; server-side)
  rt_ms               browser round trip (request sent -> response received)
  obs_to_action_ms    observation captured -> action applied (the OTA metric)
  postqueue_ms        response received -> action applied (post-queue wait)
  wait_ms             response landed before the tick boundary -> action
                      applied late (the visible 'decisions arrive late' gap)

Usage:
  python3 games/ota_analyze.py games/ota_latency/<day>/ota.jsonl
  python3 games/ota_analyze.py games/ota_latency/idle_a/ota.jsonl games/ota_latency/decode_b/ota.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

CELL_ORDER = ["snake", "paddle", "hopper", "crossing", "mines", "dungeon"]


def pctl(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def load(paths):
    reqs = {}      # (game, page, seq) -> req event
    applies = defaultdict(list)
    adapter = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"  (missing: {path})", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("type") == "session":
                    continue
                key = (r.get("game"), r.get("page"), r.get("seq"))
                if r.get("type") == "req":
                    reqs[key] = r
                elif r.get("type") == "apply":
                    applies[key].append(r)
                elif r.get("type") == "adapter":
                    adapter.append(r)
    return reqs, applies, adapter


def hint_str(h):
    if not isinstance(h, dict) or not h:
        return "-"
    exl3 = h.get("exl3", "?")
    run = h.get("exl3_running")
    wait = h.get("exl3_waiting")
    bits = [str(exl3)]
    if run is not None:
        bits.append(f"run={int(run)}")
    if wait is not None:
        bits.append(f"wait={int(wait)}")
    if h.get("gate_pool_used_gb") is not None:
        bits.append(f"pool={h['gate_pool_used_gb']}")
    return " ".join(bits)


def summarize(paths):
    reqs, applies, adapter = load(paths)
    rows = defaultdict(lambda: defaultdict(list))   # game -> metric -> [ms]
    joined = 0
    for key, req in reqs.items():
        if key[2] is None:
            continue
        app = applies.get(key)
        if not app:
            continue
        # take the first apply AFTER the req (dedupe double-applies)
        app = sorted(app, key=lambda r: r.get("ts") or 0)
        app = app[0]
        joined += 1
        g = key[0] or "unknown"
        if req.get("gate_inference_ms") is not None:
            rows[g]["gate_inference_ms"].append(float(req["gate_inference_ms"]))
        if req.get("rt_ms") is not None:
            rows[g]["rt_ms"].append(float(req["rt_ms"]))
        if req.get("obs_to_action_ms") is not None:
            rows[g]["obs_to_action_ms"].append(float(req["obs_to_action_ms"]))
        if app.get("obs_to_action_ms") is not None:
            rows[g]["obs_to_action_ms"].append(float(app["obs_to_action_ms"]))
        if app.get("postqueue_ms") is not None:
            rows[g]["postqueue_ms"].append(float(app["postqueue_ms"]))
        if app.get("wait_ms") is not None:
            rows[g]["wait_ms"].append(float(app["wait_ms"]))
    # adapter-side pure inference across ALL calls (not only joined ones)
    for a in adapter:
        if a.get("gate_inference_ms") is not None:
            rows["(adapter:all-calls)"]["gate_inference_ms"].append(float(a["gate_inference_ms"]))

    order = [g for g in CELL_ORDER if g in rows] + \
            [g for g in rows if g not in CELL_ORDER]
    out = []
    for g in order:
        m = rows[g]
        n = len(m.get("obs_to_action_ms", []))
        if not n and m.get("gate_inference_ms"):
            n = len(m["gate_inference_ms"])
        r = {"game": g, "n": n}
        for metric in ("gate_inference_ms", "rt_ms", "obs_to_action_ms",
                       "postqueue_ms", "wait_ms"):
            xs = m.get(metric, [])
            r[metric] = {p: pctl(xs, p) for p in (50, 95, 99)}
        out.append(r)
    return out, joined, len(adapter)


def main(argv):
    paths = argv[1:]
    if not paths:
        print(__doc__)
        return 2
    rows, joined, n_adapter = summarize(paths)
    print(f"samples joined req+apply: {joined}   adapter events: {n_adapter}\n")
    hdr = (f"{'game':<22} {'n':>5} | {'gate p50/p95/p99':>22} | {'rt p50/p95/p99':>20} | "
           f"{'ota p50/p95/p99':>22} | {'postq p95':>9} | {'wait p95':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def f(d):
            if d.get(50) is None:
                return "-"
            return f"{d[50]:.1f}/{d[95]:.1f}/{d[99]:.1f}"
        postq95 = r["postqueue_ms"].get(95)
        wait95 = r["wait_ms"].get(95)
        print(f"{r['game']:<22} {r['n']:>5} | {f(r['gate_inference_ms']):>22} | "
              f"{f(r['rt_ms']):>20} | {f(r['obs_to_action_ms']):>22} | "
              f"{('-' if postq95 is None else f'{postq95:.1f}'):>9} | "
              f"{('-' if wait95 is None else f'{wait95:.1f}'):>9}")
    print("\nmetric rows marked (adapter:all-calls) cover every gate call, not only decisions with a matching apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))