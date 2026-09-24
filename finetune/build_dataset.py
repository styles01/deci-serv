#!/usr/bin/env python3
"""build_dataset.py — harvested decision log + synthetic corpus -> fine-tune dataset.

Converts the two supervision sources into typed-decision training examples and
writes an 85/10/5 train/val/test split (stratified by label x intent),
state-hash deduped, with balance stats printed and an HF-style dataset card.

INPUTS (see finetune_lib docstring for row schemas):
  * harvest JSONL — server.py DecisionLog envelope (state_hash + answers).
    The raw state text is NOT in the log (SHA-256 only). Either:
      (a) pass --states STATE_MAP.json mapping state_hash -> raw state text
          (the harvester side-loads it; log stays payload-free), or
      (b) pass --corpus with train_gen.py-style rows ({"class","state",
          p_block,p_escalate,p_pass}) which are self-contained.
  * synthetic corpus JSONL — train_gen.py v1 format, 2000 rows
    (800 benign / 400 ambiguous / 800 destructive, 3 phrasing modes).

FORMAT ASSUMPTION (documented, verified vs github.com/NandhaKishorM/laya
laya/common.py + the typed-decisions fine-tune notebook):
  Output rows are (state, ONE question) items:
    {"input": {"state": str,
               "question": {"id","type","instructions","criteria","options"}},
     "target": {"qtype": 0|1|2, "probabilities": [float...], "label": int},
     "source": "harvest-label"|"harvest-floor"|"harvest-self"|"corpus",
     "class": "benign"|"ambiguous"|"destructive"|null,
     "state_hash": sha256[:16]}
  option order = Laya label-index order: choice = criteria key insertion order,
  score = "level i: <criterion>" by level index, noul = [false, true].
  choice questions are capped at 20 options (vendor head budget note).

Files produced (in --out):
  train.jsonl / val.jsonl / test.jsonl   — typed-decision supervision rows
  dataset_card.md                        — HF-style card (from the template)
  build_report.json                      — counts, strata, balance tables

GPU-free; no model download. If the raw corpus isn't on the Mac, load_rows
fails with a message saying exactly where to copy it from.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_lib import (  # noqa: E402
    balance_report, balance_stats, build_examples, dedupe_by_state, load_packs,
    load_rows, pack_questions, read_jsonl, separation_metric,
    split_examples_by_class, stratified_split, write_jsonl,
)

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_card.md")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="build_dataset", description=__doc__.split("\n")[0])
    ap.add_argument("--rows", nargs="+", required=True,
                    help="JSONL inputs: decision-log rows and/or synthetic corpus rows")
    ap.add_argument("--states", help="optional JSON {state_hash: raw_state} side-load map")
    ap.add_argument("--corpus", action="store_true",
                    help="treat all inputs as train_gen.py-style synthetic corpus rows")
    ap.add_argument("--harvest", action="store_true",
                    help="treat all inputs as DecisionLog rows (auto-detect if neither flag)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "finetune", "dataset"),
        help="output dir (default: finetune/dataset)")
    ap.add_argument("--max-choices", type=int, default=20,
                    help="drop choice questions with more options (vendor budget)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--card", default=TEMPLATE)
    args = ap.parse_args(argv)

    rows = load_rows(args.rows)
    if args.states:
        with open(args.states) as f:
            state_map = json.load(f)
        for r in rows:
            if r.get("state") is None and r.get("state_hash") in state_map:
                r["state"] = state_map[r["state_hash"]]

    packs = load_packs()
    questions = pack_questions(packs)
    # drop choice questions over the vendor's ~20-option head budget
    questions = {qid: q for qid, q in questions.items()
                 if q["type"] != "choice" or len(q["options"]) <= args.max_choices}

    examples = build_examples(rows, questions=questions)
    n_pre = len(examples)
    examples = dedupe_by_state(examples)
    n_dupe = n_dupe_ = len(examples)
    n_removed = n_dupe and (len(examples))
    train, val, test = stratified_split(examples, seed=args.seed)

    out = args.out
    counts = {}
    for name, rows_split in (("train", train), ("val", val), ("test", test)):
        counts[name] = write_jsonl(os.path.join(out, f"{name}.jsonl"), rows_split)

    stats = {
        "n_rows_in": len(rows),
        "n_examples_raw": n_dupe or 0,
        "n_examples_deduped": len(examples),
        "n_train": counts["train"], "n_val": counts["val"], "n_test": counts["test"],
        "n_questions": len(questions),
        "question_ids": ", ".join(sorted(questions)),
        "seed": args.seed,
        "split_ratios": "85/10/5",
        "date": "2026-09-24",
    }
    card = os.path.join(out, "dataset_card.md")
    if os.path.exists(args.card):
        from finetune_lib import dataset_card as render_card
        render_card(args.card, card, stats)

    report = {"inputs": args.rows, "seed": args.seed,
              "questions": sorted(questions),
              "counts": counts,
              "dedupe": {"before": len(rows), "examples": n_dupe_ or 0,
                          "after": len(examples)},
              "balance": {name: balance_stats(split) for name, split in
                          (("train", train), ("val", val), ("test", test))}}
    write_jsonl(os.path.join(out, "build_report.jsonl"),
                [report])  # jsonl for shape consistency
    os.replace(os.path.join(out, "build_report.jsonl"), os.path.join(out, "build_report.json"))

    print(f"rows in: {len(rows)} -> examples: {len(examples)} (deduped from "
          f"{len(rows)}) -> train {counts['train']} / val {counts['val']} / "
          f"test {counts['test']}")
    for name, split in (("train", train), ("val", val), ("test", test)):
        print(balance_report(split, name))
    # separation preview on the val split (target-side sanity, not model output)
    by_cls = split_examples_by_class(val)
    if "destructive" in by_cls and "benign" in by_cls:
        sep = separation_metric(by_cls["destructive"], by_cls["benign"])
        print(f"[val] target separation: {json.dumps(sep)}")
    print(f"dataset card: {card}")
    return 0


if __name__ == "__main__":
    sys.exit(main())