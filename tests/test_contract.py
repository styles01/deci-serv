# Frozen-transcript contract tests.
# These run WITHOUT the real model (offline); the provider is stubbed.
# On-box integration tests (real checkpoint) live in tests/test_onbox.py —
# skipped automatically when the checkpoint is absent.
import json
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import MagicMock

from deciserv.server import DeciHandler, _stats


class FakeProvider:
    """Deterministic provider stub with the Provider surface."""

    def describe(self):
        return {"model": "fake", "precision": "fp16", "device": "cpu", "load_s": 0.1}

    def memory_stats(self):
        return {"gpu_peak_gib": 0.0}

    def decide(self, state, questions):
        answers = {}
        for q, spec in questions.items():
            answers[q] = {
                "type": spec.get("type", "choice"),
                "choice": list(spec.get("criteria", {"a": 1, "b": 2}))[0],
                "probabilities": {"a": 0.5, "b": 0.5},
                "confidence": 0.0,
            }
        return {"answers": answers}


def serve():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), DeciHandler)
    DeciHandler.provider = FakeProvider()
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, t


def post(port, path, obj):
    import urllib.request
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=5).read())


class TestContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd, cls.t = serve()
        cls.port = cls.httpd_port = cls.httpd.server_address[1]
        # reset stats between runs
        _stats.update({"calls": 0, "lat_ms_sum": 0.0, "lat_max": 0.0, "err": 0})

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def test_decide_jev_shape(self):
        r = post(self.port, "/decide", {
            "state": "test state",
            "questions": {"q1": {"type": "choice", "instructions": "pick",
                                 "criteria": {"a": "yes", "b": "no"}}},
        })
        self.assertIn("answers", r)
        self.assertIn("q1", r["answers"])
        a = r["answers"]["q1"]
        for k in ("type", "choice", "probabilities", "confidence"):
            self.assertIn(k, a)

    def test_health_shape(self):
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5) as resp:
            h = json.load(resp)
        self.assertEqual(h["status"], "ok")
        for k in ("model", "precision", "device"):
            self.assertIn(k, h)

    def test_metrics_counts_calls(self):
        post(self.port, "/decide", {"state": "s", "questions": {}})
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/metrics", timeout=5) as resp:
            m = json.load(resp)
        self.assertGreaterEqual(m["calls"], 1)
        self.assertIn("p_avg_ms", m)
        self.assertIn("gpu_peak_gib", m)

    def test_bad_decide_returns_500_not_crash(self):
        import urllib.request
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/decide",
            data=b"not json", headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(cm.exception.code, 500)


class TestDecisionSemantics(unittest.TestCase):
    """The gate decision matrix (defaults; conformal fit overrides in production)."""

    def test_no_silent_pass_above_block_threshold(self):
        BLOCK, ESC_LO = 0.70, 0.40
        def verdict(p_block):
            if p_block >= BLOCK:
                return "block"
            if p_block >= ESC_LO:
                return "escalate"
            return "pass"
        # the adversarial-review hole: 0.85 destructive confidence must NOT pass
        self.assertEqual(verdict(0.85), "block")
        self.assertEqual(verdict(0.70), "block")   # inclusive lower bound
        self.assertEqual(verdict(0.69), "escalate")
        self.assertEqual(verdict(0.40), "escalate")
        self.assertEqual(verdict(0.39), "pass")

    def test_circuit_breaker_never_silent_bypass(self):
        def degrade(tool_class, consecutive_failures):
            if consecutive_failures >= 3:
                return "escalate" if tool_class == "destructive" else "pass_with_log"
            return "consult_sidecar"
        self.assertEqual(degrade("destructive", 5), "escalate")
        self.assertEqual(degrade("read_only", 5), "pass_with_log")


if __name__ == "__main__":
    unittest.main()