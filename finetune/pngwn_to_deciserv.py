#!/usr/bin/env python3
"""pngwn_to_deciserv.py — convert pngwn/typed-decisions-v2 into DeciServ
canonical typed-decision rows (finetune_lib build_example shape).

pngwn/typed-decisions-v2 layout (verified 2026-09-24, direct HF download;
datasets-server shows 0 rows because the repo ships .npz + raw/meta jsonl,
not parquet):
  {split}_raw.jsonl   {"uid", "domain", "prompt", "response", "answer_offsets"}
  {split}_meta.jsonl  {"kind", "options": [[opt,...] per question],
                       "gold": [int per question], "gold_probs": [[...] or null],
                       "n_options", "difficulty", "uid", "domain", "fmt",
                       "answer_positions", "prompt_tokens"}

Row semantics (README.md):
  * domain "synthetic": ticket-triage generator. fmt "multi" = workflow4 with
    4 sub-questions (severity 1-5, review yes/no, team, escalate yes/no) whose
    closed-form oracle lives in gold_probs (null for the team slot -> onehot
    gold). fmt "single" = one of those questions alone.
  * domain "noul": options ["yes","no"], gold_probs = [p_yes, p_no]
    (README: "option index 0 is 'yes'").
  * domain "choice": MMLU-Pro-style MCQ, gold_probs one-hot over options.

DeciServ canonical row (same shape tev1_to_deciserv.py emits; consumed by
finetune_lib.build_examples passthrough and build_dataset.py --rows):
  {"input": {"state": str, "question": {"id","type","instructions","criteria","options"}},
   "target": {"qtype": 0|1|2, "probabilities": [...], "label": int},
   "source": "corpus", "class": "pngwn2:<domain>", "state_hash": sha256[:16]}

Mapping policy:
  * state      = raw prompt with the trailing "### Answer:" marker stripped
                 (state/questions/options framing kept verbatim — it is the
                 exact text the gold answers refer to).
  * noul       -> qtype "noul", options ["false: No / not satisfied.",
                 "true: Yes."], target = [p_no, p_yes] (semantic order).
  * choice     -> qtype "choice", options ["<LETTER>: <text>", ...], criteria
                 {letter: text}; 2..24 options enforced.
  * synthetic  -> one row per sub-question; question id derived from the
                 sub-question text (templates repeat across rows, so the id
                 is stable and floor-checkable); soft target = gold_probs
                 when present, else onehot(gold).
  * cal/test splits are NOT converted here (kept as upstream eval/cal sets —
     train_v5 must never absorb them); pass --split train.

License: CC-BY-SA-4.0 (card + README). Attribution + share-alike: usable for
internal training; do not publicly redistribute derived rows.

Usage:
  python3 pngwn_to_deciserv.py --snap <snapshot_dir> --split train \
      --out /path/pngwn2_train.jsonl [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finetune_lib as flib  # noqa: E402

ANSWER_MARKER = re.compile(r"###\s*Answer:\s*$")
Q_SECTION = re.compile(r"^### Questions?:", re.M)   # '### Question:' or '### Questions:'
Q_BLOCK = re.compile(r"^### Questions:\n(.*?)(?=\n### Answer:)", re.M | re.S)
Q_LINE = re.compile(r"^\s*\d+\)\s*(.+?)\s*\[[A-Z]\).*\]\s*$", re.M)
QUESTION_BLOCK = re.compile(r"^### Question:\n(.+?)\n", re.M)
LETTERS = "ABCDEFGHIJKL"


def slug(text: str, words: int = 6) -> str:
    ws = re.findall(r"[a-z0-9]+", text.lower())[:words]
    return "_".join(ws) if ws else "q"


def split_state(prompt: str) -> str:
    """State text = prompt up to the first Question(s) block, minus the
    '### State:' header. Drops sibling questions (multi rows) and the trailing
    '### Answer:' marker — neither belongs to the state."""
    m = Q_SECTION.search(prompt)
    text = prompt[:m.start()] if m else ANSWER_MARKER.sub("", prompt)
    if text.startswith("### State:\n"):
        text = text[len("### State:\n"):]
    return text.rstrip("\n")


def question_texts(prompt: str, k: int) -> list[str]:
    """Sub-question texts: numbered Questions block (multi), or the single
    '### Question:' line (single-format rows), with a stable fallback."""
    m = Q_BLOCK.search(prompt)
    if m:
        texts = [t.strip() for t in Q_LINE.findall(m.group(1))]
        if len(texts) >= k:
            return texts[:k]
    qm = QUESTION_BLOCK.search(prompt)
    if qm:
        t = qm.group(1).strip()
        return [t] * k
    return [f"question {i+1}" for i in range(k)]


def subquestion_ids(prompt: str, k: int) -> list[str]:
    """Stable per-sub-question ids from the sub-question texts."""
    texts = question_texts(prompt, k)
    return ["pngwn2:synthetic:" + slug(t) for t in texts]


def _yn_target(probs, gold) -> tuple[list[float], int]:
    """pngwn yes/no slot (options ['yes','no'], index 0 = 'yes') -> DeciServ
    noul target [p_false, p_yes] (semantic order [false, true])."""
    if probs is not None:
        p_yes, p_no = float(probs[0]), float(probs[1])
    else:
        p_yes = 1.0 if int(gold) == 0 else 0.0
        p_no = 1.0 - p_yes
    return [p_no, p_yes], ([p_no, p_yes]).index(max(p_no, p_yes))


def convert_raw_meta(raw: dict, meta: dict, source_tag: str = "corpus") -> list[dict]:
    """One (raw, meta) pair -> list of canonical rows (one per sub-question)."""
    domain = meta["domain"]
    fmt = meta.get("fmt", "single")
    options_per_q = meta["options"]
    golds = meta["gold"]
    probs_per_q = meta["gold_probs"]
    state = split_state(raw["prompt"])
    qids: list[str] = []
    if domain == "synthetic":
        qids = subquestion_ids(raw["prompt"], len(options_per_q))
        texts = question_texts(raw["prompt"], len(options_per_q))
    rows: list[dict] = []
    for i, opts in enumerate(options_per_q):
        probs = probs_per_q[i] if probs_per_q and i < len(probs_per_q) else None
        gold = golds[i]
        if domain == "noul":
            qm = QUESTION_BLOCK.search(raw["prompt"])
            instr = qm.group(1).strip() if qm else "Is the statement true?"
            # pngwn order [p_yes, p_no] -> DeciServ semantic [false, true]
            p_false = float(probs[1]) if probs else 0.0
            p_true = float(probs[0]) if probs else 0.0
            if probs is None:
                p_false, p_true = (1.0, 0.0) if gold == 0 else (0.0, 1.0)
            target = [p_false, p_true]
            label = target.index(max(target))
            q = {"id": "pngwn2:noul", "type": "noul",
                 "instructions": instr,
                 "criteria": {},
                 "options": ["false: No / not satisfied.", "true: Yes."]}
        elif domain == "choice":
            if not (2 <= len(opts) <= 24):
                return rows  # caller counts skips
            keys = [LETTERS[j] for j in range(len(opts))]
            target = [float(x) for x in probs] if probs else None
            if target is None:
                target = [0.0] * len(opts)
                target[gold] = 1.0
            label = target.index(max(target))
            q = {"id": "pngwn2:choice", "type": "choice",
                 "instructions": "Which option is correct?",
                 "criteria": {k: str(v) for k, v in zip(keys, opts)},
                 "options": [f"{k}: {v}" for k, v in zip(keys, opts)]}
        else:  # synthetic
            kind = subq_kind(i, fmt)
            text = texts[i]
            if kind == "severity":
                # ordered 1..5 scale -> qtype "score", level-indexed options
                qid = "pngwn2:severity"
                target = [float(x) for x in probs] if probs is not None else None
                if target is None:
                    target = [0.0] * len(opts)
                    target[int(gold)] = 1.0
                label = target.index(max(target))
                q = {"id": qid, "type": "score",
                     "instructions": _subq_text(raw["prompt"], i, text=texts[i]),
                     "criteria": [str(o) for o in opts],
                     "options": [f"level {j}: {o}" for j, o in enumerate(opts)]}
            elif kind in ("review", "escalate"):
                # yes/no with p(yes) semantics -> qtype "noul" [false, true]
                qid = f"pngwn2:{kind}"
                target, label = _yn_target(probs, gold)
                q = {"id": qid, "type": "noul",
                     "instructions": _subq_text(raw["prompt"], i, text=texts[i]),
                     "criteria": {},
                     "options": ["false: No / not satisfied.", "true: Yes."]}
            else:  # team -> plain choice, onehot gold
                qid = "pngwn2:team"
                keys = [LETTERS[j] for j in range(len(opts))]
                target = [0.0] * len(opts)
                target[int(gold)] = 1.0
                label = int(gold)
                q = {"id": qid, "type": "choice",
                     "instructions": _subq_text(raw["prompt"], i, text=texts[i]),
                     "criteria": {k: str(v) for k, v in zip(keys, opts)},
                     "options": [f"{k}: {v}" for k, v in zip(keys, opts)]}
        rows.append(flib.build_example(state, q["id"], q, target, label,
                                       source_tag, f"pngwn2:{domain}"))
    return rows


MULTI_KINDS = ["severity", "review", "team", "escalate"]


def subq_kind(slot: int, fmt: str) -> str:
    """pngwn kind name for a synthetic slot (meta['kind'] for single format)."""
    return fmt if fmt in MULTI_KINDS else MULTI_KINDS[slot]


def _subq_text(prompt: str, i: int, text: str | None = None) -> str:
    if text is not None:
        return text
    m = Q_BLOCK.search(prompt)
    if m:
        texts = Q_LINE.findall(m.group(1))
        if i < len(texts):
            return texts[i].strip()
    m1 = QUESTION_BLOCK.search(prompt)
    if m1:
        return m1.group(1).strip()
    return "Answer sub-question %d." % (i + 1)


def convert_file(snap: str, split: str, out: str, limit: int = 0,
                 source_tag: str = "corpus") -> dict:
    raw_path = os.path.join(snap, f"{split}_raw.jsonl")
    meta_path = os.path.join(snap, f"{split}_meta.jsonl")
    n_raw = n_rows = n_skip = 0
    by_class: dict[str, int] = {}
    out_tmp = out + ".tmp"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(raw_path) as fr, open(meta_path) as fm, open(out_tmp, "w") as dst:
        for rline, mline in zip(fr, fm):
            if limit and n_raw >= limit:
                break
            raw, meta = json.loads(rline), json.loads(mline)
            assert raw["uid"] == meta["uid"], f"uid mismatch {raw['uid']}"
            n_raw += 1
            try:
                rows = convert_raw_meta(raw, meta, source_tag)
            except Exception:
                n_skip += 1
                continue
            for row in rows:
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
                by_class[row["class"]] = by_class.get(row["class"], 0) + 1
    os.replace(out_tmp, out)
    return {"split": split, "raw_in": n_raw, "rows_out": n_rows,
            "skipped_raw": n_skip, "by_class": by_class, "out": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True, help="pngwn typed-decisions-v2 snapshot dir")
    ap.add_argument("--split", default="train", choices=["train", "cal", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    stats = convert_file(args.snap, args.split, args.out, args.limit)
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())