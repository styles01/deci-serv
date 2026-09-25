#!/usr/bin/env python3
"""convert_pngwn_to_deciserv.py — pngwn/typed-decisions-v2 -> DeciServ canonical rows.

Input (downloaded from HF dataset pngwn/typed-decisions-v2):
  {split}_meta.jsonl  per-example metadata: kind, options, gold, gold_probs,
                      n_options, uid, domain, fmt ("single"|"multi"), ...
  {split}_raw.jsonl   {"uid","domain","prompt","response","answer_offsets"}

Output rows are finetune_lib.build_example canonical rows (same shape as
tev1_to_deciserv.py output; build_examples() passes them through untouched):
  {"input": {"state", "question": {"id","type","instructions","criteria","options"}},
   "target": {"qtype", "probabilities", "label"},
   "source": "corpus", "class": null, "state_hash", "origin", "origin_uid"}

Mapping policy (verified against the v2 card + REPRESENTATION.md):
  kind severity -> type "score", id "pngwn:severity", levels 1..5 in order
  kind team     -> type "choice", id "pngwn:team", letter keys A.. over 4 teams
  kind review / escalate / noul -> type "noul" (semantic order [false,true]);
        pngwn options ["yes","no"] index0=yes -> target [p_no, p_yes]
  kind choice (MMLU-Pro) -> type "choice", id "pngwn:choice", letter keys
  fmt "multi" (workflow4) -> four examples, one per slot
    (slot order asserted: severity, review, team, escalate)

License: CC BY-SA 4.0 (dataset card) — share-alike applies to derived corpora.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finetune_lib as flib  # noqa: E402

ORIGIN = "pngwn-typed-decisions-v2"
SEV_LEVELS = ["1", "2", "3", "4", "5"]
MULTI_SLOT_ORDER = ["severity", "review", "team", "escalate"]


def _parse_single(prompt: str):
    """'### State: ... ### Question: q ### Options: A) x ... ### Answer:' -> (state, q)."""
    assert prompt.startswith("### State:\n"), "unexpected prompt head"
    m = re.search(r"\n### Question:\n", prompt)
    n = re.search(r"\n### Options:\n", prompt)
    tail = prompt.rfind("\n### Answer:")
    assert m and n and tail > n.end(), "missing sections"
    state = prompt[len("### State:\n"):m.start()]
    q = prompt[m.end():n.start()].strip()
    return state, q


def _parse_multi(prompt: str):
    """'### State: ... ### Questions: 1) q [A) ..] ...' -> (state, [(idx, q), ...])."""
    assert prompt.startswith("### State:\n"), "unexpected prompt head"
    m = re.search(r"\n### Questions:\n", prompt)
    tail = prompt.rfind("\n### Answer:")
    assert m and tail > m.end(), "missing sections"
    state = prompt[len("### State:\n"):m.start()]
    qs = []
    for line in prompt[m.end():tail].strip().splitlines():
        line = line.strip()
        assert line.endswith("]"), f"bad multi question line: {line[:120]!r}"
        head = line[:-1].rsplit(" [", 1)[0]     # options bracket is always last
        mm = re.match(r"(\d+)\) (.+)", head)
        assert mm, f"bad multi question line: {line[:120]!r}"
        qs.append((int(mm.group(1)), mm.group(2).strip()))
    return state, qs


def _yn(probs_or_none, gold):
    """pngwn yes/no slot -> noul target [p_no, p_yes] (semantic order [false,true])."""
    if isinstance(probs_or_none, list) and len(probs_or_none) == 2:
        p_yes, p_no = float(probs_or_none[0]), float(probs_or_none[1])
    else:
        g = int(gold[0] if isinstance(gold, (list, tuple)) else gold)
        p_yes = 1.0 if g == 0 else 0.0   # pngwn index 0 = "yes"
        p_no = 1.0 - p_yes
    target = [p_no, p_yes]
    label = target.index(max(target))
    return target, label


def convert_row(meta: dict, raw: dict) -> list[dict]:
    """One meta+raw pair -> list of canonical rows (1 for single, 4 for multi)."""
    uid = meta["uid"]
    fmt = meta["fmt"]
    if fmt == "single":
        state, qtext = _parse_single(raw["prompt"])
        gold, probs = meta["gold"], meta["gold_probs"]
        if isinstance(gold, (list, tuple)) and gold and isinstance(gold[0], (int, float)):
            gold = gold[0]
        if isinstance(probs, (list, tuple)) and probs and (probs[0] is None or isinstance(probs[0], (list, tuple))):
            probs = probs[0]
        slots = [{"kind": meta["kind"], "options": meta["options"][0],
                  "gold": gold, "probs": probs, "qtext": qtext}]
    elif fmt == "multi":
        state, qs = _parse_multi(raw["prompt"])
        kinds = MULTI_SLOT_ORDER
        assert len(qs) == 4 and len(meta["gold"]) == 4 and len(meta["options"]) == 4, uid
        slots = []
        for i, (idx, qtext) in enumerate(qs):
            assert idx == i + 1, f"{uid}: slot order {idx} != {i+1}"
            slots.append({"kind": kinds[i], "options": meta["options"][i],
                          "gold": meta["gold"][i], "probs": meta["gold_probs"][i],
                          "qtext": qtext})
    else:
        raise ValueError(f"{uid}: unknown fmt {fmt!r}")

    out = []
    for i, slot in enumerate(slots):
        kind = slot["kind"]
        opts = slot["options"]
        if kind == "severity":
            assert opts == SEV_LEVELS, f"{uid}: severity options {opts}"
            qtype = "score"
            qid = "pngwn:severity"
            criteria = list(SEV_LEVELS)
            options = [f"level {j}: {lv}" for j, lv in enumerate(SEV_LEVELS)]
            target = [float(x) for x in slot["probs"]] if isinstance(slot["probs"], list) \
                else [1.0 if j == int(slot["gold"]) else 0.0 for j in range(5)]
            label = target.index(max(target))
        elif kind == "team":
            qtype = "choice"
            qid = "pngwn:team"
            letters = "ABCDEFGH"
            criteria = {letters[j]: o for j, o in enumerate(opts)}
            options = [f"{letters[j]}: {o}" for j, o in enumerate(opts)]
            target = [1.0 if j == int(slot["gold"]) else 0.0 for j in range(len(opts))]
            label = int(slot["gold"])
        elif kind in ("review", "escalate", "noul"):
            qtype = "noul"
            qid = f"pngwn:{kind}"
            criteria = {"false": "no", "true": "yes"}
            options = ["false: no", "true: yes"]
            target, label = _yn(slot["probs"], slot["gold"])
        elif kind == "choice":
            qtype = "choice"
            qid = "pngwn:choice"
            letters = "ABCDEFGHIJ"
            assert 2 <= len(opts) <= len(letters), f"{uid}: {len(opts)} options"
            criteria = {letters[j]: o for j, o in enumerate(opts)}
            options = [f"{letters[j]}: {o}" for j, o in enumerate(opts)]
            target = [float(x) for x in slot["probs"]]
            label = target.index(max(target))
        else:
            raise ValueError(f"{uid}: unknown kind {kind!r}")
        q = {"id": qid, "type": qtype, "instructions": slot["qtext"],
             "criteria": criteria, "options": options}
        row = flib.build_example(state, qid, q, target, label, "corpus", None)
        row["origin"] = ORIGIN
        row["origin_uid"] = f"{uid}:slot{i}" if fmt == "multi" else uid
        out.append(row)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--meta", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    raw_by_uid = {}
    with open(args.raw) as f:
        for line in f:
            r = json.loads(line)
            raw_by_uid[r["uid"]] = r
    out_rows, skipped, n_in = [], 0, 0
    with open(args.meta) as f:
        for line in f:
            m = json.loads(line)
            n_in += 1
            raw = raw_by_uid.get(m["uid"])
            if raw is None:
                skipped += 1
                continue
            try:
                rows = convert_row(m, raw)
            except (AssertionError, ValueError) as e:
                skipped += 1
                print(f"SKIP {m['uid']}: {e}", file=sys.stderr)
                continue
            out_rows.extend(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".tmp", "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(args.out + ".tmp", args.out)
    print(f"converted {len(out_rows)} canonical rows from {n_in} meta rows -> {args.out} (skipped {skipped})")
    return 0


if __name__ == "__main__":
    sys.exit(main())