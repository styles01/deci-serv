"""GPU-free unit tests for the fine-tune prep track (build_dataset/finetune_lib).

Run: cd /Users/clawdio/deci-serv && python3 -m unittest discover -s finetune/tests -v
No GPU, no model downloads, no network.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # finetune/

from finetune_lib import (  # noqa: E402
    balance_stats, build_examples, dedupe_by_state, separation_metric,
    split_examples_by_class, state_hash, stratified_split,
)

GATE_Q = {"type": "choice", "instructions": "Should this tool call be gated?",
          "criteria": {"block": "unsafe", "escalate": "ambiguous", "pass": "safe"},
          "options": ["block: unsafe", "escalate: ambiguous", "pass: safe"]}


def corpus_row(cls, state, pb, pe=0.1, pp=0.2):
    return {"class": cls, "state": state, "p_block": pb, "p_escalate": pe, "p_pass": pp}


class TestStateHash(unittest.TestCase):
    def test_deterministic_sha256(self):
        self.assertEqual(state_hash("rm -rf /"), state_hash("rm -rf /"))

    def test_differs_by_content(self):
        self.assertNotEqual(state_hash("rm -rf /"), state_hash("ls"))


class TestBuildExamples(unittest.TestCase):
    def test_side_prefix_mapping(self):
        """Corpus probs keyed by bare side must land on 'side: text' options."""
        ex = build_examples([corpus_row("destructive", "rm -rf /", 0.8, 0.1, 0.1)],
                            default_question=GATE_QUESTION)
        self.assertEqual(len(ex), 1)
        # p_block 0.8 -> target mass on index 0 (block)
        self.assertAlmostEqual(ex[0]["target"]["probabilities"][0], 0.8, places=6)
        self.assertEqual(ex[0]["target"]["label"], 0)

    def test_gold_label_per_class(self):
        rows = [corpus_row("destructive", "a", 0.5), corpus_row("ambiguous", "b", 0.4),
                corpus_row("benign", "c", 0.2)]
        ex = build_examples(rows, default_question=GATE_QUESTION)
        labels = sorted(e["target"]["label"] for e in ex)
        self.assertEqual(labels, [0, 1, 2])  # destructive->0, ambiguous->1, benign->2

    def test_low_mass_falls_back_to_onehot(self):
        ex = build_examples([corpus_row("benign", "ls", 0.1, 0.0, 0.0)],
                            default_question=GATE_QUESTION, min_prob_sum=0.5)
        self.assertEqual(ex[0]["target"]["probabilities"], [0.0, 0.0, 1.0])


GATE_QUESTION = {"id": "block", "type": "choice", "instructions": "gate this call",
                 "criteria": {"block": "unsafe", "escalate": "ambiguous", "pass": "safe"},
                 "options": ["block: unsafe", "escalate: ambiguous", "pass: safe"]}


class TestDedupe(unittest.TestCase):
    def test_dedupes_by_state_hash_and_question(self):
        rows = [corpus_row("benign", "git status", 0.1),
                corpus_row("benign", "git status", 0.2),   # same state -> dup
                corpus_row("destructive", "rm -rf /", 0.9)]
        ex = build_examples(rows, default_question=GATE_QUESTION)
        deduped = dedupe_by_state(ex)
        self.assertEqual(len(deduped), 2)


class TestStratifiedSplit(unittest.TestCase):
    def test_all_classes_present_in_train(self):
        rows = ([corpus_row("benign", f"ls {i}", 0.1) for i in range(10)] +
                [corpus_row("destructive", f"rm -rf d{i}", 0.9) for i in range(10)] +
                [corpus_row("ambiguous", f"rm -rf cache{i}", 0.4) for i in range(10)])
        ex = dedupe_by_state(build_examples(rows, default_question=GATE_QUESTION))
        tr, va, te = stratified_split(ex)
        self.assertGreater(len(tr), 0)
        self.assertGreater(len(va) + len(te), 0)
        # every class survives somewhere in train
        classes = {e.get("class") for e in tr}
        self.assertEqual(classes, {"benign", "destructive", "ambiguous"})

    def test_deterministic_with_seed(self):
        rows = [corpus_row("benign", f"s{i}", 0.1) for i in range(20)]
        ex = dedupe_by_state(build_examples(rows, default_question=GATE_QUESTION))
        tr1, _, _ = stratified_split(ex, seed=1337)
        tr2, _, _ = stratified_split(ex, seed=1337)
        self.assertEqual([e["input"]["state"] for e in tr1],
                         [e["input"]["state"] for e in tr2])


class TestBalanceStats(unittest.TestCase):
    def test_json_safe(self):
        rows = [corpus_row("benign", "s1", 0.1), corpus_row("benign", "s2", 0.1),
                corpus_row("destructive", "d1", 0.9)]
        ex = build_examples(rows, default_question=GATE_QUESTION)
        stats = balance_stats(ex)
        json.dumps(stats)  # must not raise (tuple keys are the historical bug)
        self.assertEqual(stats["block"]["total"], len(ex))
        self.assertEqual(stats["block"]["by_source"], {"corpus": len(ex)})


class TestSeparationMetric(unittest.TestCase):
    def test_perfect_separation(self):
        ex = build_examples([corpus_row("destructive", "d1", 1.0),
                             corpus_row("destructive", "d2", 0.9),
                             corpus_row("benign", "b1", 0.05),
                             corpus_row("benign", "b2", 0.1)],
                            default_question=GATE_QUESTION)
        by = split_examples_by_class(ex)
        m = separation_metric(by["destructive"], by["benign"])
        self.assertEqual(m["auc"], 1.0)
        # destructive p_block mean (0.95 post-normalization) vs benign (~0.047):
        self.assertGreater(m["gap"], 0.75)

    def test_zero_separation(self):
        rows = [corpus_row("destructive", f"d{i}", 0.2) for i in range(5)] + \
               [corpus_row("benign", f"b{i}", 0.2) for i in range(5)]
        ex = build_examples(rows, default_question=GATE_QUESTION)
        by = split_examples_by_class(ex)
        m = separation_metric(by["destructive"], by["benign"])
        self.assertLessEqual(abs(m["gap"]), 0.05)


if __name__ == "__main__":
    unittest.main()