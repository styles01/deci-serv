#!/usr/bin/env python3
"""tev1_to_deciserv.py — convert togethercomputer/tev1 new-v1 records into
DeciServ typed-decisions training rows (finetune_lib format).

tev1 record shape (data/new-v1/records/*.jsonl):
  {id, source, split, kind, group_id, state (str|dict), question (str),
   options: [{label, key, description}], answer (e.g. "C ineligible"),
   answer_key (e.g. "ineligible"), provenance}

Our training row shape (consumed by finetune/build_dataset.py --rows):
  {"input": {"state": str,
             "question": {"type": "choice"|"noul"|"score",
                          "instructions": str,
                          "options": ["key: description", ...],
                          "criteria": [...] (score only)},
             "question_meta": {"source", "group_id", "kind"}},
   "target": {"probabilities": [..], "label": int},
   "class": str}

Mapping policy:
  * kind "choice"  -> type "choice", options kept, target = onehot on answer_key
                       (2-24 options; Laya handles arbitrary cardinality)
  * kind "noul"    -> type "noul", target ordered [false, true]  (semantic order)
  * kind "score"   -> type "score", target over level indices "0".."K-1"
  * state dicts (policy/routing records) are flattened to readable text:
      context + policy + facts lines — Laya's state is a plain string.
  * class: policy_v2/policy/routing_v2 keep their source as class; public
    classification sources map to class "benign" (they only teach calibration
    over generic states, never gate semantics).

Gate-relevance filter (default ON via --gate-only): keep policy_v2, policy,
routing_v2 (19,500 rows of executable-rule decision supervision) and drop the
pure topic/sentiment sets unless --all is passed.

Usage:
  python3 tev1_to_deciserv.py --tev1 /path/tev1/data/new-v1/records/train.jsonl \
      --out /path/train_tev1.jsonl [--gate-only] [--limit N] [--val 0.05]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys


def flatten_state(state) -> str:
    """tev1 policy/routing states arrive as dicts; render as readable text."""
    if isinstance(state, str):
        return state
    parts = []
    for key in ("context", "policy", "interpretation"):
        v = state.get(key)
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, dict):
            if "rules" in v:  # priority routing
                rules = v["rules"]
                rendered = "; ".join(
                    f"(priority {r.get('priority')}) when {r.get('when')} -> {r.get('route')}"
                    for r in rules)
                parts.append("Priority routing rules: " + rendered)
            elif isinstance(v, dict):  # generic nested policy
                parts.append(json.dumps(v, ensure_ascii=False))
    # the varying payload: target record(s) facts — REQUIRED, they are what
    # differentiates states within a policy group
    rec = state.get("record")
    if rec is not None:
        parts.append("Target record: " + json.dumps(rec, sort_keys=True, ensure_ascii=False))
    recs = state.get("records")
    if recs:
        rendered = " | ".join(json.dumps(x, sort_keys=True, ensure_ascii=False) for x in recs)
        parts.append("Possible completions: " + rendered)
    tgt = state.get("target_id")
    if tgt:
        parts.append(f"Target id: {tgt}")
    if "untrusted_comment" in state:
        parts.append("Untrusted comment: " + str(state["untrusted_comment"]))
    if not parts:  # fall back to full dump
        parts.append(json.dumps(state, sort_keys=True, ensure_ascii=False))
    return "\n".join(parts)


def answer_to_distribution(options, answer_key, kind):
    """Onehot over options ordered by their label (A..). noul keeps [false,true]."""
    if kind == "noul":
        idx = 1 if str(answer_key).lower() in ("true", "yes") else 0
        return [0.0, 1.0] if idx == 1 else [1.0, 0.0]
    keys = [o["key"] for o in options]
    if answer_key not in keys:
        return None
    p = [0.0] * len(keys)
    p[keys.index(answer_key)] = 1.0
    return p


def convert_record(r: dict) -> dict | None:
    """Convert one tev1 record into a canonical finetune_lib.build_example row."""
    import finetune_lib as flib
    kind = r.get("kind", "choice")
    options = r.get("options") or []
    source = r.get("source", "tev1")
    if kind == "noul":
        qtype = "noul"
        # semantic order is always [false, true]
        keys = ["false", "true"]
        opts = ["false: No / not satisfied.", "true: Yes."]
        crit = {}
    elif kind == "score":
        qtype = "score"
        levels = sorted({o["key"] for o in options})
        keys = levels
        opts = [f"{lv}: level {lv}" for lv in levels]
        crit = {lv: (next((o.get("description", "") for o in options if o["key"] == lv), "")) for lv in levels}
    else:
        qtype = "choice"
        keys = [o["key"] for o in options]
        opts = [f"{o['key']}: {o.get('description', o['key'])}" for o in options]
        if not (2 <= len(opts) <= 24):
            return None
        crit = {o["key"]: o.get("description", "") for o in options}
    probs = answer_to_distribution(options, r.get("answer_key"), kind)
    if probs is None:
        return None
    q = {"id": f"tev1:{r.get('source', 'x')}", "type": qtype,
         "instructions": r.get("question", "Decide using the supplied state."),
         "criteria": crit, "options": opts}
    label = probs.index(max(probs))
    cls = source if source in GATE_SOURCES else "benign"
    state_text = flatten_state(r.get("state"))
    return flib.build_example(state_text, f"tev1:{source}", q,
                              probs, label, "corpus", cls)


GATE_SOURCES = {"policy_v2", "policy", "routing_v2"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tev1", required=True, help="path to tev1 records/*.jsonl")
    ap.add_argument("--out", required=True, help="output jsonl (DeciServ rows)")
    ap.add_argument("--gate-only", action="store_true", default=True)
    ap.add_argument("--all", action="store_true", help="keep every source incl. topic/sentiment")
    ap.add_argument("--limit", type=int, default=0, help="cap rows (0 = no cap)")
    ap.add_argument("--val", type=float, default=0.05, help="held-out fraction")
    args = ap.parse_args(argv)

    rng = random.Random(1337)
    kept, skipped = 0, 0
    by_source: dict[str, int] = {}
    out_tmp = args.out + ".tmp"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.tev1) as src, open(out_tmp, "w") as dst:
        for line in src:
            r = json.loads(line)
            if args.gate_only and not args.all and r.get("source") not in GATE_SOURCES:
                skipped += 1
                continue
            row = convert_record(r)
            if row is None:
                skipped += 1
                continue
            dst.write(json.dumps(row) + "\n")
            kept += 1
            if args.limit and kept >= args.limit:
                break
    os.replace(out_tmp, args.out)
    print(f"kept {kept} rows -> {args.out} (skipped {skipped})")
    return 0


if __name__ == "__main__":
    sys.exit(main())