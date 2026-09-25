#!/usr/bin/env python3
"""build_v5_corpus.py — assemble the DeciServ v5 training corpus.

Assembles:
  * v4 corpus rows (dataset-v4-mixed train/val/test — kept in their original
    split assignment so prior eval baselines stay comparable), and
  * pngwn/typed-decisions-v2 converted rows (pngwn2_train.jsonl from
    pngwn_to_deciserv.py — CC-BY-SA-4.0, verified 31,109 raw / 45,932 rows),
optionally tasksource rows IF a license verdict of USABLE is supplied (the
2026-09-24 verdict is RESEARCH-ONLY-ABORT, so tasksource is NOT included by
default), and emits train_v5.jsonl / val_v5.jsonl / test_v5.jsonl plus a
manifest JSON (per-source counts, licenses, floor-check stats, SHA-256s).

Floor check (shreyanbr pattern): per question id, constant-predictor floor =
majority-gold-label share. Questions whose majority share is >= the threshold
(default 0.95) are flagged and ALL their rows are dropped (a constant
predictor already matches that question; it carries ~no decision signal).

Eval-only corpora (SargeDev/jev-distill-corpus-v3 ood split, JonesLin/
next-jev-tetris-decisions-10k) are NEVER loaded into the training pool —
hard-coded exclusion, documented in the manifest.

Split policy:
  * v4 rows keep the split they came from (train/val/test of dataset-v4-mixed).
  * pngwn2 rows are split 85/10/5 with finetune_lib.stratified_split
    (stratified by source x question id x label, seed 1337).
  * val/test therefore contain only rows that passed the floor check.

Usage:
  python3 build_v5_corpus.py --v4-dir v4_corpus \
      --pngwn2 converted/pngwn2_train.jsonl --out-dir v5_corpus \
      [--floor-threshold 0.95] [--seed 1337]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finetune_lib as flib  # noqa: E402

LICENSES = {
    "v4-mixed": ("internal mix: tev1 gate (train_tev1_gate.jsonl) + injecagent + "
                 "arcadia harvest + synthetic corpus; upstream licenses not "
                 "re-audited for v5", "included"),
    "pngwn2": ("CC-BY-SA-4.0 (HF card + README of pngwn/typed-decisions-v2, "
               "verified by direct download 2026-09-24)", "included"),
    "tasksource-jev": ("other — card quote: \"Tasksource harmonizes datasets from "
                       "many publishers; their original licenses and terms still "
                       "apply, hence `license: other`.\" No LICENSE file in repo "
                       "(LICENSE/LICENSE.md/LICENSE.txt all 404); 625 upstream "
                       "sources, per-source licenses not audited. VERDICT: "
                       "RESEARCH-ONLY-ABORT (model is served, so research-only/"
                       "unclear provenance is not cleared for training).",
                       "excluded-license"),
    "sargedev-ood": ("Apache-2.0 (SargeDev/jev-distill-corpus-v3 ood split, "
                     "13,058 rows) — EVAL-ONLY by project rule", "excluded-eval-only"),
    "next-jev-tetris-10k": ("JonesLin/next-jev-tetris-decisions-10k — EVAL-ONLY "
                            "by project rule", "excluded-eval-only"),
    "zefancai-open-jev": ("CC0-1.0 (520,199 rows) — fallback corpus; NOT NEEDED "
                          "this run (pngwn verified non-empty: 31,109 train rows)",
                          "fallback-unused"),
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def floor_check(examples: list[dict], threshold: float) -> tuple[dict, set]:
    """Per-question constant-predictor floor -> stats + ids to drop."""
    per_q: dict[str, Counter] = defaultdict(Counter)
    for ex in examples:
        per_q[ex["input"]["question"]["id"]][ex["target"]["label"]] += 1
    stats: dict[str, dict] = {}
    dropped: set[str] = set()
    for qid, c in sorted(per_q.items()):
        total = sum(c.values())
        mlabel, mcount = c.most_common(1)[0]
        share = mcount / total
        stats[qid] = {"n": total, "majority_label": mlabel,
                      "majority_share": round(share, 4),
                      "floor_breach": share >= threshold}
        if share >= threshold:
            dropped.add(qid)
    return stats, dropped


def group_split_by_state(examples: list[dict], ratios=(0.85, 0.10, 0.05),
                         seed: int = 1337) -> tuple[list, list, list]:
    """85/10/5 split by state_hash GROUPS: all rows of one upstream state stay
    in the same split (workflow4 quads never straddle). Groups are stratified
    by their (qid -> majority label) signature so label balance is preserved;
    deterministic under seed. Signature bins with 1 group -> train, 2 groups
    -> train+val (mirrors flib.stratified_split's small-stratum rule)."""
    import random
    groups: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        groups[ex["state_hash"]].append(ex)
    sig_bins: dict[tuple, list[str]] = defaultdict(list)
    for h, rows in groups.items():
        sig = tuple(sorted(
            (r["input"]["question"]["id"], r["target"]["label"]) for r in rows))
        sig_bins[sig].append(h)
    rng = random.Random(seed)
    tr: list[dict] = []
    va: list[dict] = []
    te: list[dict] = []
    for sig in sorted(sig_bins):
        hs = sig_bins[sig]
        rng.shuffle(hs)
        n = len(hs)
        n_te = min(int(round(n * ratios[2])), max(0, n - 1))
        n_va = min(int(round(n * ratios[1])), max(0, n - n_te))
        te += [r for h in hs[:n_te] for r in groups[h]]
        va += [r for h in hs[n_te:n_te + n_va] for r in groups[h]]
        tr += [r for h in hs[n_te + n_va:] for r in groups[h]]
    return tr, va, te


def _check_no_group_leak(train, val, test) -> None:
    """Assert every state_hash appears in exactly one of the three splits."""
    seen: dict[str, set] = defaultdict(set)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        for e in rows:
            seen[e["state_hash"]].add(name)
    leak = [h for h, s in seen.items() if len(s) > 1]
    if leak:
        raise SystemExit(f"group-leak check FAILED: {len(leak)} state groups straddle splits")
    print(f"group-leak check: OK ({len(seen)} state groups, each in exactly one split)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4-dir", required=True,
                    help="dir with v4 train/val/test.jsonl (original split kept)")
    ap.add_argument("--pngwn2", required=True, help="converted pngwn2 train rows")
    ap.add_argument("--pngwn2-snap", default=None,
                    help="pngwn snapshot dir (enables upstream cal/test eval-state "
                         "quarantine before splitting)")
    ap.add_argument("--tasksource", default=None,
                    help="converted tasksource rows (only used with "
                         "--tasksource-verdict USABLE)")
    ap.add_argument("--tasksource-verdict", default="RESEARCH-ONLY-ABORT",
                    choices=["USABLE", "RESEARCH-ONLY-ABORT",
                             "SHAREALIKE-CONSTRAINED"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--floor-threshold", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)

    pool: list[tuple[dict, str]] = []          # (example, origin)
    rows_in: dict[str, int] = defaultdict(int)
    for split in ("train", "val", "test"):
        rows = flib.read_jsonl(os.path.join(args.v4_dir, f"{split}.jsonl"))
        rows_in["v4-mixed"] += len(rows)
        pool += [(r, ("v4-mixed", split)) for r in rows]
    png = flib.read_jsonl(args.pngwn2)
    rows_in["pngwn2"] += len(png)
    pool += [(r, ("pngwn2", None)) for r in png]
    n_quarantined = 0
    ts_rows = 0
    if args.tasksource:
        if args.tasksource_verdict == "USABLE":
            raise SystemExit("tasksource inclusion path is intentionally closed: "
                             "no converter was built because the 2026-09-24 "
                             "license verdict is RESEARCH-ONLY-ABORT. Extend "
                             "this script deliberately if that ever changes.")
        print(f"tasksource rows at {args.tasksource} EXCLUDED: verdict "
              f"{args.tasksource_verdict} != USABLE")

    # ---- floor check: per corpus (for the report) and combined (enforced) ----
    floor_stats: dict[str, dict] = {}
    dropped_qids: set[str] = set()
    for name, subset in (("v4-mixed", [e for e, o in pool if o[0] == "v4-mixed"]),
                         ("pngwn2", [e for e, o in pool if o[0] == "pngwn2"]),
                         ("combined", [e for e, _ in pool])):
        stats, dropped = floor_check(subset, args.floor_threshold)
        floor_stats[name] = {"threshold": args.floor_threshold,
                             "questions": len(stats),
                             "rows": sum(s["n"] for s in stats.values()),
                             "per_question": stats,
                             "dropped_question_ids": sorted(dropped)}
        dropped_qids |= dropped

    n_pool_before_floor = len(pool)
    n_floor_dropped = sum(1 for e, _ in pool
                          if e["input"]["question"]["id"] in dropped_qids)
    pool = [(e, o) for e, o in pool
            if e["input"]["question"]["id"] not in dropped_qids]

    # ---- dedupe (state_hash x question id, source-rank priority) ----
    examples = [e for e, _ in pool]
    n_pre_dedupe = len(examples)
    examples = flib.dedupe_by_state(examples)
    n_post_dedupe = len(examples)

    # ---- splits: v4 keeps its original split; pngwn2 -> GROUPED 85/10/5 ----
    # R3 leak fix (adversarial review deleg_3dafc747): pngwn2 expands 31,109
    # upstream rows to 45,932 canonical rows (workflow4 = 4 slot-rows per
    # state), so a row-level stratified split scatters sibling rows of the
    # SAME state across train/val/test and inflates val agreement by
    # memorization. Split by state_hash groups instead; also quarantine any
    # pngwn2 group whose state matches an upstream cal/test state (used for
    # temperature fitting) so training never sees eval states.
    v4 = [e for e in examples if e.get("class") not in
          ("pngwn2:synthetic", "pngwn2:noul", "pngwn2:choice")]
    pn2 = [e for e in examples if e.get("class") in
           ("pngwn2:synthetic", "pngwn2:noul", "pngwn2:choice")]
    if args.pngwn2_snap:
        import pngwn_to_deciserv as P
        up_eval = set()
        for f in ("cal_raw.jsonl", "test_raw.jsonl"):
            p = os.path.join(args.pngwn2_snap, f)
            if os.path.exists(p):
                for line in open(p):
                    up_eval.add(flib.state_hash(P.split_state(json.loads(line)["prompt"])))
        n_q = len(pn2)
        pn2 = [e for e in pn2 if e["state_hash"] not in up_eval]
        n_quarantined = n_q - len(pn2)
        print(f"pngwn2 upstream eval-state quarantine: dropped {n_quarantined} rows "
              f"(state collides with pngwn2 cal/test)")
    pn2_train, pn2_val, pn2_test = group_split_by_state(pn2, seed=args.seed)
    _check_no_group_leak(pn2_train, pn2_val, pn2_test)
    # v4 rows carry their original split via a temp field
    for e in examples:
        e["_v5_split"] = None
    origin_map = {id(e): o for e, o in pool}
    for e in examples:
        o = origin_map.get(id(e))
        if o and o[0] == "v4-mixed":
            e["_v5_split"] = o[1]
    train = [e for e in v4 if e["_v5_split"] == "train"] + pn2_train
    val = [e for e in v4 if e["_v5_split"] == "val"] + pn2_val
    test = [e for e in v4 if e["_v5_split"] == "test"] + pn2_test
    for e in examples:
        e.pop("_v5_split")

    # ---- write outputs + manifest ----
    os.makedirs(args.out_dir, exist_ok=True)
    files = {}
    for name, rows in (("train_v5.jsonl", train), ("val_v5.jsonl", val),
                       ("test_v5.jsonl", test)):
        p = os.path.join(args.out_dir, name)
        flib.write_jsonl(p, rows)
        files[name] = {"rows": len(rows), "sha256": sha256_file(p)}

    def source_counts(rows: list[dict]) -> dict:
        c = Counter()
        for e in rows:
            cls = e.get("class") or "none"
            if cls.startswith("pngwn2"):
                c[cls] += 1
            else:  # v4 rows: distinguish by source AND class (e.g. tev1 policy_v2
                c[f"v4:{e.get('source', '?')}:{cls}"] += 1
        return dict(sorted(c.items()))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "inputs": {
            "v4-mixed": {"dir": os.path.abspath(args.v4_dir),
                         "rows_in": rows_in["v4-mixed"]},
            "pngwn2": {"path": os.path.abspath(args.pngwn2),
                       "rows_in": rows_in["pngwn2"]},
        },
        "licenses": {k: {"license": v[0], "decision": v[1]}
                     for k, v in LICENSES.items()},
        "floor_check": {
            "rule": (f"drop every question whose majority-gold-label share "
                     f">= {args.floor_threshold} (constant-predictor floor)"),
            "rows_in": n_pool_before_floor,
            "rows_dropped": n_floor_dropped,
            "questions_dropped": {qid: floor_stats["combined"]["per_question"][qid]
                                  for qid in sorted(dropped_qids)},
            "per_corpus": floor_stats,
        },
        "dedupe": {"before": n_pre_dedupe, "after": n_post_dedupe,
                   "removed": n_pre_dedupe - n_post_dedupe},
        "pngwn2_eval_state_quarantine": {
            "n_rows_dropped": n_quarantined if args.pngwn2_snap else 0,
            "note": ("canonical rows whose state matches pngwn2's upstream cal/test "
                     "states are dropped so training never sees eval states")},
        "splits": {k: v["rows"] for k, v in files.items()},
        "train_by_source": source_counts(train),
        "sha256": {k: v["sha256"] for k, v in files.items()},
        "excluded": ["tasksource-jev (license)", "sargedev-ood (eval-only)",
                     "next-jev-tetris-10k (eval-only)"],
    }
    man_path = os.path.join(args.out_dir, "manifest_v5.json")
    with open(man_path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(man_path + ".tmp", man_path)

    print(json.dumps({k: manifest[k] for k in
                      ("rows_in", "floor_check", "dedupe", "splits")
                      if k in manifest}, indent=1)[:4000])
    print(f"manifest: {man_path}")
    print(f"train {len(train)} / val {len(val)} / test {len(test)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())