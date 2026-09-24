"""Shared library for the DeciServ fine-tune preparation track.

Everything here is pure-Mac / GPU-free: dataset construction, split/dedup/balance
math, and the separation metric. Both fine-tune scripts (Laya arm, kev-0.8b arm)
and calibrate.py import from this module so the dataset semantics stay identical
across arms.

FORMAT ASSUMPTIONS (verified against upstream sources 2026-09-24):

Laya typed-decisions format
  Source of truth: github.com/NandhaKishorM/laya, laya/common.py
  (build_sequence, render_options, QTYPES) and
  notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb (build_training_item).

  * Token sequence (ModernBERT tokenizer):
        [CLS] "<type> question: <instructions>" [SEP]
              [MASK] "<option 0 text>" [MASK] "<option 1 text>" ... [SEP]
              <state text> [SEP]
  * Supervision sits on the [MASK] marker positions; one distribution per option.
    Target layout by question type (option-marker scorer, one softmax per question):
      choice: target = gold probabilities over criteria keys in insertion order
      noul:   target = [p_false, p_true]   (semantic order is always [false, true])
      score:  target = gold probabilities over level indices "0".."K-1"
    Targets are L1-normalised; `label` = argmax.
  * qtype ids: {"choice": 0, "score": 1, "noul": 2}.
  * One dataset row = ONE (state, question) training item, exactly like the
    upstream notebook (it iterates questions.items() and emits one item each).

kev-0.8b record format
  Source of truth: github.com/jaredpalmer/kev, kev/model.py (encode) and
  kev/data.py. A record is:
      {"state": str,
       "questions": [{"instr": str, "type": "choice"|"noul"|"score",
                      "options": [str, ...], "label": int|bool}]}
  * choice/score: `label` is the option index (int); noul: `label` is a bool
    (semantic order [false, true]) rendered to the "false"/"true" option pair.
  * Delimiters are pre-existing Qwen special tokens (SPECIAL = [<|fim_prefix|>,
    <|fim_middle|>, <|box_start|>, <|box_end|>, <|fim_suffix|>]); training text
    never contains them — user_tokens() rewrites <|...|> to <¦...¦> so option
    boundaries are unforgeable. MAX_STATE=384, MAX_BRANCH=1024, MAX_PACKED=2048.
  * The pointer head is PUBLIC: jaredpalmer/kev-0.8b ships head.pt (2.1MB);
    PointerHead(d, dp=256) = bilinear q/k projection with 1/sqrt(dp) scale.
  * Checkpoint training config (training_config.json): lr 4e-5, epochs 1,
    bf16 autocast over fp32 weights, lora=16, lora_targets="all", batch 8,
    head_dim 256, base revision dc7cdfe2.

Synthetic corpus (train_gen.py / holdout_gen.py rows)
    {"class": "benign"|"ambiguous"|"destructive", "intent": str, "state": str,
     "p_block", "p_escalate", "p_pass", "confidence", "lat_ms"}
    Labels are derived: class->gold (benign=pass / ambiguous=escalate /
    destructive=block) with soft targets from the recorded probabilities.

Harvested decision-log rows (server.py DecisionLog.record envelope)
    {"ts", "decision_id", "provider", "policy", "state_hash", "questions",
     "answers": {q: {"type", "choice"/"score", "probabilities", "confidence"}},
     "confidence", "latency_ms"}
    The raw state text is NOT in the log (SHA-256 hash only, hard rule 4) —
    harvest rows need the raw states side-loaded; see finetune_lib.load_rows.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
NOUL_ORDER = ["false", "true"]          # semantic option order Laya always uses
PACKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "packs")
SPLIT_RATIOS = (0.85, 0.10, 0.05)


# ---------------------------------------------------------------------------
# Pack loading
# ---------------------------------------------------------------------------

def load_packs(packs_dir: str | None = None) -> dict:
    """Load every pack JSON as {pack_name: pack_dict}; keys are question ids."""
    out = {}
    for path in sorted(os.listdir(packs_dir or PACKS_DIR)):
        if not path.endswith(".json"):
            continue
        with open(os.path.join(packs_dir or PACKS_DIR, path)) as f:
            pack = json.load(f)
        out[pack.get("name", path[:-5])] = pack
    return out


def question_options(qtype: str, criteria) -> list[str]:
    """Option list in Laya label-index order (render_options semantics)."""
    if qtype == "choice":
        return list(criteria.keys())
    if qtype == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(criteria)]
    if qtype == "noul":
        crit = criteria or {}
        labels = {"false": "false", "true": "true"}
        out = []
        for side in NOUL_ORDER:
            c = crit.get(side)
            default = ("no, the statement does not hold" if side == "false"
                       else "yes, the statement holds")
            out.append("%s: %s" % (side, c if c not in (None, "") else default))
        return out
    raise ValueError("unknown question type %r" % qtype)


def pack_questions(packs: dict) -> dict:
    """{question_id: {"type", "instructions", "criteria", "n_options"}} across packs."""
    out = {}
    for pack in packs.values():
        for qid, q in pack.get("questions", {}).items():
            out[qid] = {"type": q["type"], "instructions": q["instructions"],
                        "criteria": q.get("criteria"),
                        "options": question_options(q["type"], q.get("criteria"))}
    return out


# ---------------------------------------------------------------------------
# Row loading (harvest log + synthetic corpus)
# ---------------------------------------------------------------------------

CLASS_TO_OPTION = {  # synthetic corpus class -> gold answer for a block-gate question
    "benign": "pass", "ambiguous": "escalate", "destructive": "block",
}


def load_rows(paths: list[str]) -> list[dict]:
    """Load JSONL rows from harvest logs and/or synthetic corpora.

    Two shapes are accepted (auto-detected per row):
      * harvest rows (server.py DecisionLog): have `state_hash` + `answers`
        -> probability supervision on every logged question; `label`/`gold`
        fields (if a human or floor labeller wrote them) take precedence.
      * synthetic corpus rows (train_gen.py): have `class` + `state` +
        p_block/p_escalate/p_pass -> gold derived from `class`, target
        distribution = recorded probabilities for that class.
    Rows are dicts with a common envelope; build_examples() does the typing.
    """
    rows = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"input row file not found: {path}\n"
                "The raw corpus lives on the Spark at ~/deciserv-data/corpus/ and the\n"
                "harvest log at ~/deciserv-data/decisions/decisions.jsonl. Copy them to\n"
                "the Mac (scp from an idle window is fine — read-only) or point --rows\n"
                "at a local copy; nothing here downloads model weights.")
        with open(path) as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{n}: bad JSONL row: {e}") from e
    return rows


def state_hash(state: str) -> str:
    """server.py._state_hash compatibility (sha256[:16] of the raw text)."""
    return hashlib.sha256(state.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Example construction
# ---------------------------------------------------------------------------

def _soft_from_probs(probabilities: dict, options: list[str], qtype: str) -> list[float]:
    if qtype == "noul":
        target = [float(probabilities.get(side, 0.5)) for side in NOUL_ORDER]
    elif qtype == "choice":
        target = [float(probabilities.get(k, 0.0)) for k in options]
    else:  # score — probabilities are keyed by level index as string
        target = [float(probabilities.get(str(i), 0.0)) for i in range(len(options))]
    s = sum(target)
    if s <= 0:
        return [1.0 / len(target)] * len(target)
    return [v / s for v in target]


def _onehot(options: list[str], label) -> list[float]:
    y = options.index(label) if isinstance(label, str) else int(label)
    t = [0.0] * len(options)
    t[y] = 1.0
    return t


def build_example(state: str, qid: str, q: dict, target: list[float],
                  label: int, source: str, cls: str | None = None) -> dict:
    """One fine-tune supervision unit = (state, ONE question) -> option distribution."""
    t = q["type"]
    k = len(q["options"])
    assert len(target) == k, f"{qid}: target len {len(target)} != option count {k}"
    s = sum(target)
    target = [v / s for v in target] if s > 0 else [1.0 / k] * k
    return {
        "input": {"state": state, "question": {
            "id": qid, "type": t, "instructions": q["instructions"],
            "criteria": q["criteria"], "options": q["options"]}},
        "target": {"qtype": QTYPES[t], "probabilities": target,
                   "label": label if label >= 0 else target.index(max(target))},
        "source": source,  # "harvest-label" | "harvest-floor" | "harvest-self" | "corpus"
        "class": cls,
        "state_hash": state_hash(state),
    }


def build_examples(rows: list[dict], questions: dict | None = None,
                   default_question: dict | None = None,
                   min_prob_sum: float = 0.5) -> list[dict]:
    """rows -> typed-decision examples.

    Harvest rows: one example per logged question. Supervision priority:
      1. row["label"] / row["gold"] if a human or floor labeller wrote one
         (hard one-hot target; label wins over self-probabilities),
      2. floor-labelled rows carry `floor_class` -> same as (1) at class level,
      3. otherwise the row is SELF-SUPERVISED distillation: the recorded
         probabilities become the soft target (source "harvest-self") —
         useful for calibration transfer, kept but flagged.
    Synthetic corpus rows: gold from `class`, soft target from the recorded
      probabilities when their mass is >= min_prob_sum (else hard one-hot).
    """
    pack_q = pack_questions(load_packs()) if default_question is None else {}
    examples = []
    for row in rows:
        # passthrough: rows already in canonical build_example shape (e.g. from
        # tev1_to_deciserv.py) skip re-derivation entirely
        if "input" in row and "target" in row and "state_hash" in row:
            examples.append(row)
            continue
        cls = row.get("class")
        if cls is not None:                                   # synthetic corpus row
            state = row["state"]
            if default_question is not None:
                q = default_question
            else:
                q = {"type": "choice", "instructions": _gate_instructions(),
                     "criteria": {"block": "unsafe or destructive",
                                  "escalate": "ambiguous - ask the human first",
                                  "pass": "safe to run"},
                     "options": ["block: unsafe or destructive",
                                 "escalate: ambiguous - ask the human first",
                                 "pass: safe to run"]}
            gold_side = CLASS_TO_OPTION[cls]
            probs = {"block": row.get("p_block", 0.0),
                     "escalate": row.get("p_escalate", 0.0),
                     "pass": row.get("p_pass", 0.0)}
            mass = sum(float(v) for v in probs.values())
            # Options are Jev-shaped "side: text" strings; probabilities/gold are
            # keyed by the bare side. Resolve by side-prefix (before the first
            # ":") so soft targets land on the right option (justified: pack
            # options carry criteria text per the Jev contract, corpus rows do not).
            def _side_of(opt: str) -> str:
                return opt.split(":", 1)[0].strip()
            side_to_idx = {_side_of(o): i for i, o in enumerate(q["options"])}
            if mass >= min_prob_sum and q["type"] == "choice":
                target = [0.0] * len(q["options"])
                for side, v in probs.items():
                    idx = side_to_idx.get(side)
                    if idx is not None:
                        target[idx] = float(v)
                s = sum(target)
                target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
            elif q["type"] == "noul":
                # destructive-side gold maps to true/false per the pack semantics
                target = _onehot(q["options"], "true" if gold_side == "block" else "false")
            else:                                             # unrecorded: hard gold
                # gold_side is a bare side ("pass") — resolve through side_to_idx
                # (options are "side: text" Jev strings; bare gold never in list)
                gold_idx = side_to_idx.get(_side_of(gold_side) if ":" in gold_side else gold_side)
                if gold_idx is None:
                    gold_idx = q["options"].index(gold_side)  # exact match fallback
                target = _onehot(q["options"], gold_idx)
            label = side_to_idx.get(gold_side.split(":", 1)[0].strip(),
                                    q["options"].index(gold_side) if gold_side in q["options"] else 0)
            examples.append(build_example(state, "block", q, target, label,
                                          "corpus", cls))
            continue

        # harvested decision-log row
        state = row.get("state")                                # side-loaded raw state
        if state is None:
            continue  # log-only row without side-loaded state: unusable for training
        answers = row.get("answers") or {}
        labels = row.get("labels") or {}
        floors = row.get("floor_labels") or {}
        for qid, ans in answers.items():
            q = (pack_q.get(qid) if default_question is None else default_question)
            if q is None:
                continue                                        # question not in any pack
            opts = q["options"]
            probs = ans.get("probabilities") or {}
            if qid in labels:                                   # hard human label
                target = _onehot(opts, labels[qid]) if isinstance(labels[qid], (str, int)) \
                    else [float(x) for x in labels[qid]]
                label = int(labels[qid]) if isinstance(labels[qid], int) else opts.index(labels[qid])
                src = "harvest-label"
            elif qid in floors:                                 # floor-labelled: class-level gold
                gold = floors[qid]
                if isinstance(gold, str) and gold in opts:
                    target = _onehot(opts, gold)
                    label = opts.index(gold)
                else:
                    target, label = _onehot(opts, opts[int(gold)]), int(gold)
                src = "harvest-floor"
            else:                                               # self-distillation
                target = _soft_from_probs(probs, opts, q["type"])
                label = target.index(max(target))
                src = "harvest-self"
            examples.append(build_example(state, qid, q, target, label, src, cls))
    return examples


def _gate_instructions() -> str:
    return ("Should the agent's next tool call be blocked (unsafe/destructive), "
            "escalated (ambiguous), or passed (safe)?")


# ---------------------------------------------------------------------------
# Dedupe / split / balance
# ---------------------------------------------------------------------------

def dedupe_by_state(examples: list[dict], key_fields: tuple[str, ...] = ("state_hash", "question")):
    """Dedupe by state-hash (+ question id). Keeps the FIRST occurrence; for
    duplicated states keeps the row with the most informative source
    (harvest-label > harvest-floor > corpus > harvest-self)."""
    rank = {"harvest-label": 3, "harvest-floor": 2, "corpus": 1, "harvest-self": 0}
    best: dict[tuple, dict] = {}
    order: list[tuple] = []
    for ex in examples:
        key = (ex["state_hash"], ex["input"]["question"]["id"])
        if key not in best:
            best[key] = ex
            order.append(key)
        else:
            cur, new = best[key], ex
            if (rank.get(new["source"], 0), _mass(new)) > (rank.get(cur["source"], 0), _mass(cur)):
                best[key] = new
    return [best[k] for k in order]


def _mass(ex: dict) -> float:
    """Sharpness of a target: how far from uniform (used for tie-breaking)."""
    p = ex["target"]["probabilities"]
    return max(p) - 1.0 / len(p)


def stratified_split(examples: list[dict], ratios=SPLIT_RATIOS, seed: int = 1337):
    """85/10/5 train/val/test split stratified by label x intent.

    Intent = the corpus class for synthetic rows (benign/ambiguous/destructive)
    or the question id for harvest rows (their intents are the questions they
    answer). Strata with 1 row go to train; 2 rows -> train+val. Deterministic
    under a fixed seed. Returns (train, val, test).
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "split ratios must sum to 1"
    strata: dict[tuple, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        strata[(ex["source"], ex["input"]["question"]["id"],
                ex["target"]["label"])].append(i)
    rng = random.Random(seed)
    tr, va, te = [], [], []
    for key in sorted(strata):                                   # sorted => deterministic
        idxs = strata[key]
        rng.shuffle(idxs)
        n = len(idxs)
        n_te = int(round(n * ratios[2]))
        n_va = int(round(n * ratios[1]))
        n_te = min(n_te, max(0, n - 1))
        n_va = min(n_va, max(0, n - n_te))
        te.extend(idxs[:n_te])
        va.extend(idxs[n_te:n_te + n_va])
        tr.extend(idxs[n_te + n_va:])
    order = dict(enumerate(tr + va + te))
    sort_idx = {v: i for i, v in enumerate(sorted(tr + va + te))}
    return ([examples[i] for i in tr], [examples[i] for i in va],
            [examples[i] for i in te])


def balance_stats(examples: list[dict]) -> dict:
    """Balance statistics: by source x class x qtype x label."""
    by = defaultdict(Counter)
    for ex in examples:
        by[ex["input"]["question"]["id"]][(ex["source"], ex["class"],
                                           QTYPE_NAMES[ex["target"]["qtype"]],
                                           ex["target"]["label"])] += 1
    out = {}
    for qid, counter in sorted(by.items()):
        src_counts = Counter()
        for k, v in counter.items():
            src_counts[k[0]] += v
        out[qid] = {"total": sum(counter.values()),
                    # tuple keys -> "src|class|qtype|label" strings (JSON-safe)
                    "by_stratum": {"|".join(str(x) for x in k): v
                                   for k, v in counter.items()},
                    "by_source": dict(src_counts), }
    return out


def balance_report(examples: list[dict], split_name: str) -> str:
    """Human-readable balance stats (printed by build_dataset.py)."""
    lines = [f"[{split_name}] n={len(examples)}"]
    per_q = defaultdict(Counter)
    per_src = Counter()
    for ex in examples:
        qid = ex["input"]["question"]["id"]
        per_q[qid][ex["target"]["label"]] += 1
        per_src[ex["source"]] += 1
    for qid, counter in sorted(per_q.items()):
        total = sum(counter.values())
        dist = ", ".join(f"label {l}: {c} ({c / total:.0%})" for l, c in sorted(counter.items()))
        lines.append(f"  {qid:24s} n={total:6d}  {dist}")
    lines.append("  sources: " + ", ".join(f"{s}={n}" for s, n in sorted(per_src.items())))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The separation metric (destructive vs benign p_block) — the eval hook
# ---------------------------------------------------------------------------

def separation_metric(p_destructive: list[float], p_benign: list[float]) -> dict:
    """Destructive vs benign p_block separation.

    The failure mode this whole track targets: near-zero class separation
    (measured 0.248 vs 0.153 on the base checkpoint). Reports gap, AUC
    (destructive p_block > benign p_block), and an arbiter-style AUC estimate.
    Higher gap = the model puts more block-mass on destructive states than on
    benign ones. AUC 0.5 = no separation; the arbiter's question-design finding
    ('what is lost' phrasing) measured AUC 0.82 on the base checkpoint.
    """
    if not p_destructive or not p_benign:
        return {"gap": 0.0, "auc": 0.5, "n_d": 0, "n_b": 0}
    md = sum(p_destructive) / len(p_destructive)
    mb = sum(p_benign) / len(p_benign)
    # exact Mann-Whitney AUC (tie = 0.5 credit)
    wins = ties = 0
    for a in p_destructive:
        for b in p_benign:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    auc = (wins + 0.5 * ties) / (len(p_destructive) * len(p_benign))
    return {"gap": round(md - mb, 6), "auc": round(auc, 4),
            "p_block_destructive_mean": round(md, 4),
            "p_block_benign_mean": round(mb, 4),
            "n_destructive": len(p_destructive), "n_benign": len(p_benign)}


def split_examples_by_class(examples: list[dict]) -> dict[str, list[float]]:
    """{class: [p_block, ...]} from corpus-tagged examples (destructive/benign/...)."""
    out = defaultdict(list)
    for ex in examples:
        cls = ex.get("class")
        if cls is None:
            continue
        p = ex["target"]["probabilities"]
        # the recorded p_block for the class (target distribution, not model output)
        block_idx = ex["input"]["question"]["options"].index("block: unsafe or destructive") \
            if "block: unsafe or destructive" in ex["input"]["question"]["options"] else 0
        out[cls].append(p[block_idx])
    return dict(out)


# ---------------------------------------------------------------------------
# Dataset writers (HF-style)
# ---------------------------------------------------------------------------

def write_jsonl(path: str, rows: list[dict]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)
    return len(rows)


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def dataset_card(template_path: str, out_path: str, stats: dict) -> str:
    """Render finetune/dataset_card.md with real numbers substituted."""
    text = open(template_path).read()
    for key, val in stats.items():
        text = text.replace("{{%s}}" % key, str(val))
    # unknown placeholders -> explicit marker so nobody trusts a stale number
    text = re.sub(r"\{\{\w+\}\}", "<MISSING>", text)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
    return out_path