#!/usr/bin/env python3
"""convert_tasksource_to_deciserv.py — tasksource/tasksource-jev-typed-decisions
parquet shards -> DeciServ canonical rows (streaming, row-group at a time).

Input: one or more tasksource parquet files with columns
  state, id, kind (choice|score|noul), options (list<str>), target (list<f64>),
  question, source, variant, split, group_id, question_id

Mapping policy (verified against the card + shard-0 rows):
  choice -> type "choice", id "tasksource:{source}/{question_id}",
            letter keys A.. over options in row order, target as given
  score  -> type "score",  id "tasksource:{source}/{question_id}",
            options "level i: <text>" over the row's own option order
  noul   -> type "noul",   id "tasksource:{source}/{question_id}",
            options ["false: no", "true: yes"], target [1-p, p] with p=target[0]
Floor filter: rows whose (source, question_id) appears in --floor-groups json
(per-question constant-predictor share >= 0.999, n >= 20) are dropped.
Choice questions with more than --max-choices options are dropped (vendor budget).

License: card `license: other` — upstream per-source licenses apply (research
use only unless the per-source license permits; recorded in the manifest).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finetune_lib as flib  # noqa: E402

ORIGIN = "tasksource-jev-typed-decisions"


def convert_batch(t) -> tuple[list[dict], dict]:
    """One pyarrow Table -> (canonical rows, counters)."""
    out = []
    stats = {"in": t.num_rows, "floor_dropped": 0, "too_many_options": 0,
             "bad_target": 0, "kept": 0}
    cols = {c: t.column(c).to_pylist() for c in
            ["state", "id", "kind", "options", "target", "question", "source",
             "question_id", "split"]}
    for i in range(t.num_rows):
        kind = cols["kind"][i]
        opts = cols["options"][i]
        tgt = cols["target"][i]
        src = cols["source"][i]
        qid_s = cols["question_id"][i]
        if kind == "noul":
            # noul target is [p_true]; [0.0] is a VALID "definitely false" row,
            # only None/empty targets are unusable
            if not tgt:
                stats["bad_target"] += 1
                continue
        elif not tgt or sum(tgt) <= 0:
            stats["bad_target"] += 1
            continue
        if kind == "noul":
            qtype = "noul"
            qid = f"tasksource:{src}/{qid_s}"
            p = float(tgt[0])
            target = [1.0 - p, p]
            label = target.index(max(target))
            criteria = {"false": "no", "true": "yes"}
            options = ["false: no", "true: yes"]
            qtext = cols["question"][i] or "Is the statement true?"
        elif kind == "score":
            if not opts:
                stats["bad_target"] += 1
                continue
            qtype = "score"
            qid = f"tasksource:{src}/{qid_s}"
            target = [float(x) for x in tgt]
            if len(target) != len(opts):
                stats["bad_target"] += 1
                continue
            criteria = [str(o) for o in opts]
            options = [f"level {j}: {o}" for j, o in enumerate(criteria)]
            label = target.index(max(target))
            qtext = cols["question"][i] or "Rate the state on the supplied scale."
        elif kind == "choice":
            if not opts or len(opts) > MAX_CHOICES:
                stats["too_many_options"] += 1
                continue
            qtype = "choice"
            qid = f"tasksource:{src}/{qid_s}"
            target = [float(x) for x in tgt]
            if len(target) != len(opts):
                stats["bad_target"] += 1
                continue
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            criteria = {letters[j]: o for j, o in enumerate(opts)}
            options = [f"{letters[j]}: {o}" for j, o in enumerate(opts)]
            label = target.index(max(target))
            qtext = cols["question"][i] or "Choose the criterion that best answers the question."
        else:
            stats["bad_target"] += 1
            continue
        key = (src, qid_s)
        if key in FLOOR_GROUPS:
            stats["floor_dropped"] += 1
            continue
        q = {"id": qid, "type": qtype, "instructions": qtext,
             "criteria": criteria, "options": options}
        state = cols["state"][i]
        row = flib.build_example(state, qid, q, target, label, "corpus", None)
        row["origin"] = ORIGIN
        row["origin_uid"] = cols["id"][i]
        out.append(row)
        stats["kept"] += 1
    return out, stats


MAX_CHOICES = 20
FLOOR_GROUPS = set()


def main(argv=None) -> int:
    global MAX_CHOICES, FLOOR_GROUPS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--floor-groups", help="json list of {source,question_id} groups to drop")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-choices", type=int, default=20)
    args = ap.parse_args(argv)
    MAX_CHOICES = args.max_choices
    if args.floor_groups:
        FLOOR_GROUPS = {(g["source"], g["question_id"]) for g in json.load(open(args.floor_groups))}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total = {k: 0 for k in ["in", "kept", "floor_dropped", "too_many_options", "bad_target"]}
    with open(args.out + ".tmp", "w") as f:
        for path in args.parquet:
            pf = pq.ParquetFile(path)
            for rg in range(pf.metadata.num_row_groups):
                t = pf.read_row_group(rg)
                rows, stats = convert_batch(t)
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                for k in total:
                    total[k] += stats[k]
                del t, rows
    os.replace(args.out + ".tmp", args.out)
    print(f"tasksource: in={total['in']} kept={total['kept']} "
          f"floor_dropped={total['floor_dropped']} too_many_options={total['too_many_options']} "
          f"bad_target={total['bad_target']} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())