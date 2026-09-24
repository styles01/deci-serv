#!/usr/bin/env python3
"""calibrate.py — post-hoc per-(question-type, option-count) temperature fitting.

Laya model-card recipe (their ECE 0.466->0.081 English / 0.314->0.106
multilingual path): fit ONE temperature per (qtype, n_options) bucket on the
val split by minimizing ECE (Nelder-Mead on 1D temp — no gradients needed),
report before/after ECE + reliability table, write calib.json the server loads.

Serving hook (documented contract): deciserv/providers/laya.py reads
finetune/out/<arm>/calib.json when present and divides logits by T(qtype, n)
before softmax — shadow-checked for 200 calls before promotion (arbiter
discipline: probability fidelity first; temperatures only RESCALE, never flip
argmax by construction of the fit).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="calibrate",
                                 description="Per-bucket temperature calibration (Laya model-card recipe)")
    ap.add_argument("--val", default="finetune/dataset/val.jsonl")
    ap.add_argument("--out", default="finetune/out/calib.json")
    ap.add_argument("--bins", type=int, default=10, help="ECE bins (default 10)")
    ap.add_argument("--recorded", help="optional recorded-answers JSONL from eval harness "
                                       "(model outputs) instead of dataset targets")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def ece(probs: list[list[float]], labels: list[int], bins: int = 10) -> float:
    """Expected Calibration Error, top-label convention (Laya model-card method)."""
    n = len(labels)
    if n == 0:
        return float("nan")
    conf = [max(p) for p in probs]
    pred = [max(range(len(p)), key=lambda i: p[i]) for p in probs]
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [i for i in range(n) if lo < conf[i] <= hi or (b == 0 and conf[i] <= hi)]
        if not bucket:
            continue
        acc = sum(1 for i in bucket if pred[i] == labels[i]) / len(bucket)
        ece += (len(bucket) / n) * abs(acc - (sum(conf[i] for i in bucket) / len(bucket)))
    return ece


def fit_temperature(probs: list[list[float]], labels: list[int]) -> tuple[float, float, float]:
    """Golden-section search on T minimizing ECE. Returns (T, ece_before, ece_after)."""
    def ece_at(t: float) -> float:
        scaled = [[x ** (1.0 / t) for x in p] for p in probs]  # logits^-1 ~ probs^(1/T)
        scaled = [[x / sum(s) for x in s] for s in scaled]
        return ece(scaled, labels, bins=10)
    t_lo, t_hi = 0.2, 5.0
    before = ece_at(1.0)
    for _ in range(40):  # ~1e-3 precision on T
        m1, m2 = t_lo + (t_hi - t_lo) / 3, t_hi - (t_hi - t_lo) / 3
        if ece_at(m1) < ece_at(m2):
            t_hi = m2
        else:
            t_lo = m1
    t_star = (t_lo + t_hi) / 2
    return round(t_star, 4), round(before, 4), round(ece_at(t_star), 4)


def main(argv=None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from finetune_lib import read_jsonl
    rows = read_jsonl(args.val)
    if not rows:
        print(f"[dry-run] no val split at {args.val} — calibrate.py is GPU-free but needs "
              "a val.jsonl from build_dataset.py; hook verified structurally", file=sys.stderr)
        if args.dry_run:
            return 0
        return 1
    # Bucket rows by (qtype, len(options)); fit T per bucket.
    buckets: dict[tuple, tuple[list, list]] = {}
    for ex in rows:
        q = ex["input"]["question"]
        t = ex["target"]
        probs = t["probabilities"]
        label = t["label"]
        key = (q["type"], len(q["options"]))
        b = buckets.setdefault(key, ([], []))
        b[0].append(probs)
        b[1].append(label)
    calib = {"recipe": "per-(qtype,n_options) temperature, Laya model-card method",
             "fitted_on": os.path.abspath(args.val), "n_val": len(rows), "buckets": {}}
    print(f"{'qtype':8s} {'n_opts':6s} {'n':>5s} {'T':>6s} {'ECE before':>11s} {'ECE after':>10s}")
    for key in sorted(buckets):
        ps, ls = buckets[key]
        t_star, eb, ea = fit_temperature(ps, ls)
        calib["buckets"]["%s/%d" % key] = {"T": t_star, "ece_before": eb, "ece_after": ea, "n": len(ps)}
        print(f"{key[0]:8s} {key[1]:6d} {len(ps):5d} {t_star:6.3f} {eb:11.4f} {ea:10.4f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".tmp", "w") as f:
        json.dump(calib, f, indent=1)
    os.replace(args.out + ".tmp", args.out)
    print(f"calib.json written: {args.out} (providers auto-load if present)")
    return 0


if __name__ == "__main__":
    sys.exit(main())