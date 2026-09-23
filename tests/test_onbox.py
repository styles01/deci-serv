"""On-box integration test — runs only when a real Laya checkpoint is present.

Set DECISERV_LAYA_CKPT to the checkpoint dir to enable:
    DECISERV_LAYA_CKPT=~/models/hf/convaiinnovations/laya python -m pytest tests/test_onbox.py
Skips cleanly otherwise.
"""
import os
import unittest

CKPT = os.environ.get("DECISERV_LAYA_CKPT", "").strip()

@unittest.skipIf(not CKPT, "DECISERV_LAYA_CKPT not set — on-box test skipped")
class TestLayaOnBox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, CKPT)
        from deciserv.providers.laya import LayaProvider
        cls.p = LayaProvider(CKPT, precision="fp16", device="cuda")

    def _decide(self, state, q="proceed", criteria=None, instructions=None):
        return self.p.decide(state, {q: {"type": "choice",
                                         "instructions": instructions or "Proceed?",
                                         "criteria": criteria or {"proceed": "run it",
                                                                  "escalate": "ask first",
                                                                  "refuse": "refuse"}}})

    def test_health_shape(self):
        d = self.p.describe()
        self.assertEqual(d["precision"], "fp16")
        self.assertGreater(d["load_s"], 0)

    def test_destructive_state_returns_probabilities(self):
        r = self._decide("User asks to rm -rf the production database folder right now.")
        a = r["answers"]["proceed"]
        self.assertIn("probabilities", a)
        self.assertAlmostEqual(sum(a["probabilities"].values()), 1.0, places=2)

    def test_benign_state(self):
        self._decide("User asks to list the files in the project docs folder.")

    def test_batched_equals_sequential(self):
        """One /decide with 3 questions ≡ 3 single-question calls (per-question
        temperature heads must not cross-contaminate under batching)."""
        qs = {
            "q1": {"type": "choice", "instructions": "Destructive?",
                   "criteria": {"proceed": "run it", "escalate": "ask first", "refuse": "refuse"}},
            "q2": {"type": "choice", "instructions": "Read-only safe?",
                   "criteria": {"proceed": "run it", "escalate": "ask first", "refuse": "refuse"}},
            "q3": {"type": "choice", "instructions": "Network egress?",
                   "criteria": {"proceed": "run it", "escalate": "ask first", "refuse": "refuse"}},
        }
        state = "User asks to install an unsigned pip package globally and email the log externally."
        batched = self.p.decide(state, qs)
        for q in qs:
            single = self.p.decide(state, {q: qs[q]})
            b, s = batched["answers"][q], single["answers"][q]
            for k in b["probabilities"]:
                self.assertAlmostEqual(b["probabilities"][k], s["probabilities"][k], places=2,
                                       msg=f"batched vs sequential mismatch on {q}/{k}")

    def test_latency_budget(self):
        import time
        self._decide("warmup call with a benign state about listing files")
        t0 = time.time()
        self._decide("User asks to list files in docs folder.")
        dt_ms = (time.time() - t0) * 1000
        self.assertLess(dt_ms, 200, f"gate decision took {dt_ms:.0f}ms (lane-idle budget 120ms p95)")


if __name__ == "__main__":
    unittest.main()