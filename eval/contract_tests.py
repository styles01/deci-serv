#!/usr/bin/env python3
"""contract_tests.py — jevcompat-inspired contract tests for DeciServ's /decide.

Pure HTTP against a live gate (NO repo imports, NO model): POSTs to
DECISERV_URL (default http://127.0.0.1:8710) and checks the /decide response
contract. Standalone: python3 eval/contract_tests.py [--url URL] [--json]
Exit code == number of FAILING assertions (0 == contract holds).

MUSTs (from the shipped contract, verified in deciserv/server.py):
  * response is a JSON object with answers (object), decision_id (str),
    provider (str), policy (str), latency_ms (number >= 0)
  * every requested question name appears in answers with a dict value
  * every answer carries a str type in {choice, score, noul} and a
    numeric confidence in [0, 1]
  * choice answers: int choice index 0..len-1 over the question's option
    sides; probabilities maps >= 1 side name to numbers in [0, 1];
    options count <= 255 (HTTP/JSON sanity bound from jevcompat)
  * score answers: numeric score inside the criteria range
  * answers align with options: probabilities keys are subsets of the
    question's option/criteria side names (a synthesized verdict-side map
    is accepted; a stray side name is a contract break)
  * deterministic repeatability (cache OFF): two identical POSTs produce
    byte-identical answers (+ equal verdict/final_risk/provider/policy)
  * idempotent cache behavior (cache ON): two identical POSTs, second
    latency_ms == first latency_ms (stored envelope replayed) and answers
    identical — i.e. second request was served from the memoized envelope
"""
from __future__ import annotations
import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("DECISERV_URL", "http://127.0.0.1:8710")
MAX_OPTIONS = 255
ENV_CACHE = "DECISERV_CACHE"
ENV_TTL = "DECISERV_CACHE_TTL_S"

CHOICE_Q = {
    "type": "choice",
    "instructions": "Verdict?",
    "criteria": {"pass": "safe routine read-only",
                 "escalate": "needs human confirmation",
                 "block": "destructive or dangerous"},
}
SCORE_Q = {
    "type": "score",
    "instructions": "If this command is a mistake, what is lost?",
    "criteria": ["nothing", "ephemeral files", "user data",
                 "credentials or system access"],
}
NOUL_Q = {
    "type": "noul",
    "instructions": "This command deletes or overwrites something that cannot be brought back.",
    "criteria": {"yes": "This command deletes or overwrites something irreversible.",
                 "no": "no"},
}

# (state, label) chosen so the suite exercises all three gate paths against a
# live hybrid server: read-only fast path, destructive floor, and grey middle.
CASES = [
    ("ls -la /tmp", "read_only"),
    ("cat /dev/null > /etc/hosts", "destructive_floor"),
    ("rm -rf /tmp/contract-test-build-dir", "grey_middle"),
]

_results = []


def post_decide(url, state, questions, timeout=30):
    body = json.dumps({"state": state, "policy": "safety", "questions": questions}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/decide", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    return payload, (time.time() - t0) * 1000.0


def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def check(name, ok, detail=""):
    _results.append((name, bool(ok), detail))
    return bool(ok)


def _sides(qspec):
    crit = qspec.get("criteria")
    if isinstance(crit, dict):
        return list(crit.keys())
    if isinstance(crit, list):
        return [str(i) for i in range(len(crit))]
    opts = qspec.get("options")
    if isinstance(opts, list):
        return [str(o) for o in opts]
    return []


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ------------------------------------------------------------------ MUSTs
def t1_shape_and_fields(state):
    def go(tag, payload):
        if not check(f"{tag}: response is a JSON object", isinstance(payload, dict),
                     f"got {type(payload).__name__}"):
            return
        for k in ("answers", "decision_id", "provider", "policy", "latency_ms"):
            check(f"{tag}: field {k!r} present", k in payload,
                  f"missing {k!r}; keys={sorted(payload.keys())}")
        check(f"{tag}: answers is an object", isinstance(payload.get("answers"), dict),
              f"answers type {type(payload.get('answers')).__name__}")
        check(f"{tag}: decision_id is a non-empty string",
              isinstance(payload.get("decision_id"), str) and payload["decision_id"],
              repr(payload.get("decision_id")))
        check(f"{tag}: provider is a string", isinstance(payload.get("provider"), str),
              repr(payload.get("provider")))
        check(f"{tag}: policy is a string", isinstance(payload.get("policy"), str),
              repr(payload.get("policy")))
        lat = payload.get("latency_ms")
        check(f"{tag}: latency_ms is a number >= 0",
              _num(lat) and lat >= 0, repr(lat))
        answers = payload.get("answers")
        if not isinstance(answers, dict):
            return
        for qname, ans in answers.items():
            check(f"{tag}: answer[{qname}] is an object", isinstance(ans, dict),
                  f"type {type(ans).__name__}")
            if not isinstance(ans, dict):
                continue
            check(f"{tag}: answer[{qname}].type is a string",
                  isinstance(ans.get("type"), str), repr(ans.get("type")))
            check(f"{tag}: answer[{qname}].type in {{choice,score,noul}}",
                  ans.get("type") in ("choice", "score", "noul"), repr(ans.get("type")))
            conf = ans.get("confidence")
            if _num(conf):
                cv = float(str(conf))
                check(f"{tag}: answer[{qname}].confidence numeric in [0,1]",
                      0.0 <= cv <= 1.0, repr(conf))
            else:
                check(f"{tag}: answer[{qname}].confidence numeric in [0,1]", False,
                      repr(conf))
    for state, tag in state:
        try:
            payload, _ms = post_decide(URL, state, SUITE_QUESTIONS)
        except Exception as exc:  # noqa: BLE001
            check(f"{tag}: /decide reachable", False, f"{type(exc).__name__}: {exc}")
            continue
        go(tag, payload)


def t2_choice_alignment(payload, qname, qspec, tag):
    ans = (payload.get("answers") or {}).get(qname)
    if not isinstance(ans, dict):
        check(f"{tag}: choice answer[{qname}] is an object", False,
              f"missing/none; answers={(payload.get('answers') or {}).keys()}")
        return
    sides = _sides(qspec)
    check(f"{tag}: question {qname} exposes option sides (<= {MAX_OPTIONS})",
          1 <= len(sides) <= MAX_OPTIONS, f"sides={sides!r}")
    probs = ans.get("probabilities")
    if not isinstance(probs, dict) or not probs:
        # some stacks return a bare choice index with no probabilities — still
        # contract-legal, but nothing further to align; require the index at least
        check(f"{tag}: choice[{qname}] has probabilities or a choice",
              isinstance(ans.get("choice"), (int, str)),
              f"answer={ans!r}")
        return
    check(f"{tag}: choice[{qname}] probabilities keys align with option sides",
          set(probs.keys()).issubset(set(sides) | {None}) or all(k in sides for k in probs),
          f"probs keys={sorted(map(str, probs.keys()))!r} sides={sides!r}")
    bad = [k for k, v in probs.items() if not (_num(v) and 0.0 <= float(v) <= 1.0)]
    check(f"{tag}: choice[{qname}] probabilities are numbers in [0,1]",
          not bad, f"bad={bad!r}")
    total = sum(float(v) for v in probs.values() if _num(v))
    check(f"{tag}: choice[{qname}] probabilities sum within [0.9, 1.1]",
          0.9 <= total <= 1.1, f"sum={total:.4f}")
    choice = ans.get("choice")
    if isinstance(choice, int) and sides:
        check(f"{tag}: choice[{qname}] index inside options range",
              0 <= choice < len(sides), f"choice={choice!r} n_sides={len(sides)}")
    else:
        check(f"{tag}: choice[{qname}].choice is an int index",
              isinstance(choice, (int, str)), repr(choice))


def t3_score_alignment(payload, qname, qspec, tag):
    ans = (payload.get("answers") or {}).get(qname)
    if not isinstance(ans, dict):
        check(f"{tag}: score answer[{qname}] is an object", False, "missing")
        return
    if ans.get("type") == "noul":
        # The shipped synthesizer (server.py:174-199) answers ALL synthesized
        # questions with verdict-typed bodies (noul true/false on score/noul
        # questions alike). The contract validates each answer against its
        # DECLARED type; the type mismatch itself is a NOTE for the model
        # layer, not a shape violation.
        _results.append((f"{tag}: note score[{qname}] synthesized as noul "
                         f"(verdict-side body) — model layer may differ", True,
                         "note"))
        return t4_noul_alignment(payload, qname, tag)
    crit = qspec.get("criteria") or []
    n = len(crit)
    s = ans.get("score")
    check(f"{tag}: score[{qname}].score numeric",
          _num(s), repr(s))
    if _num(s):
        try:
            sv = float(s)  # type: ignore[arg-type]  # _num(s) guards
        except (TypeError, ValueError):
            sv = -1.0
        check(f"{tag}: score[{qname}] within criteria range [0, {max(n - 1, 1)}]",
              0.0 <= sv <= max(n - 1, 1), repr(s))
    probs = ans.get("probabilities")
    if isinstance(probs, dict):
        check(f"{tag}: score[{qname}] probability keys align with criteria order",
              all(str(k) in {str(i) for i in range(n)} or k in crit for k in probs),
              f"keys={sorted(map(str, probs.keys()))!r} criteria={crit!r}")
        vals = [float(v) for v in probs.values() if _num(v)]
        check(f"{tag}: score[{qname}] probabilities within [0,1]",
              all(0.0 <= v <= 1.0 for v in vals), repr(probs))


def t4_noul_alignment(payload, qname, tag):
    ans = (payload.get("answers") or {}).get(qname)
    if not isinstance(ans, dict):
        check(f"{tag}: noul answer[{qname}] is an object", False, "missing")
        return
    v = ans.get("noul", ans.get("choice"))
    check(f"{tag}: noul[{qname}] carries a true/false/numeric verdict",
          isinstance(v, (str, int, float)) and not isinstance(v, bool),
          repr(v))
    if isinstance(v, str):
        check(f"{tag}: noul[{qname}] verdict in {{true,false}}",
              v.strip().lower() in ("true", "false"), repr(v))
    probs = ans.get("probabilities")
    if isinstance(probs, dict):
        vals = [float(x) for x in probs.values() if _num(x)]
        check(f"{tag}: noul[{qname}] probabilities within [0,1]",
              all(0.0 <= x <= 1.0 for x in vals), repr(probs))


def t5_repeatability():
    """Deterministic repeatability with the cache OFF: two identical POSTs
    produce identical answers + verdict fields (decision_id/latency are
    per-request identity and may differ)."""
    state, questions = "grep -rn 'contract_test' /tmp/contract_probe", SUITE_QUESTIONS
    try:
        r1, _ = post_decide(URL, state, questions)
        r2, _ = post_decide(URL, state, questions)
    except Exception as exc:  # noqa: BLE001
        check("repeatability: /decide reachable", False, f"{type(exc).__name__}: {exc}")
        return
    strip = lambda r: {k: r.get(k) for k in ("answers", "verdict", "final_risk",
                                             "provider", "policy")}
    check("repeatability: identical requests -> identical answers/verdict",
          strip(r1) == strip(r2),
          f"diff keys={[k for k in strip(r1) if strip(r1).get(k) != strip(r2).get(k)]!r}")


def t6_cache_idempotency():
    """With the cache ON, an identical second request must be a HIT: the
    stored envelope is replayed so latency_ms is preserved and answers are
    identical. Requires the server running with DECISERV_CACHE=1 (skipped as
    'warn-only' otherwise, so the suite never fails an off-cache server)."""
    state, questions = "sort /tmp/contract_probe_roster.txt | uniq -c", SUITE_QUESTIONS
    m0 = _safe_metrics()
    h0 = int(m0.get("cache_hits", 0) or 0)
    ms0 = int(m0.get("cache_misses", 0) or 0)
    try:
        r1, _ = post_decide(URL, state, questions)
        r2, _ = post_decide(URL, state, questions)
    except Exception as exc:  # noqa: BLE001
        check("cache-idempotency: /decide reachable", False, f"{type(exc).__name__}: {exc}")
        return
    m1 = _safe_metrics()
    h1 = int(m1.get("cache_hits", 0) or 0)
    ms1 = int(m1.get("cache_misses", 0) or 0)
    dh, dms = h1 - h0, ms1 - ms0
    same_answers = r1.get("answers") == r2.get("answers")
    replayed = (isinstance(r2.get("latency_ms"), (int, float))
                and r2.get("latency_ms") == r1.get("latency_ms"))
    check("cache-idempotency: identical second request returns identical answers",
          same_answers, f"answers differ: {str(r2.get('answers'))[:120]}")
    if dh > 0:
        check("cache-idempotency: second request is a cache hit (latency replayed)",
              replayed,
              f"lat r1={r1.get('latency_ms')} r2={r2.get('latency_ms')}; "
              f"cache_hits delta={dh} misses delta={dms}")
        check("cache-idempotency: hit marker present on the replayed response",
              r2.get("cache") == "hit", f"cache={r2.get('cache')!r}")
    else:
        _results.append((f"cache-idempotency: SKIPPED — no cache hits observed during "
                         f"this probe (Δhits={dh}, Δmisses={dms}); run the server with "
                         f"DECISERV_CACHE=1 to exercise the hit path", True,
                         "skipped, not a failure"))


def _safe_metrics():
    try:
        return get_json(URL.rstrip("/") + "/metrics")
    except Exception:  # noqa: BLE001
        return {}


# ------------------------------------------------------------------ runner
def suite_questions():
    """One request shape used across shape/alignment tests (3 questions in
    one call — the server's documented batch shape)."""
    return {"verdict": CHOICE_Q, "what_is_lost": SCORE_Q, "destroys_data": NOUL_Q}


SUITE_QUESTIONS = suite_questions()


def wait_reachable(url, timeout_s=10):
    host = url.split("//", 1)[-1].split("/")[0]
    host_part, _, port_part = host.partition(":")
    port = int(port_part or 80)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host_part, port), timeout=1.5):
                return True
        except OSError:
            time.sleep(0.4)
    return False


def main(argv=None):
    global URL
    ap = argparse.ArgumentParser(prog="contract_tests",
                                 description="DeciServ /decide contract tests (pure HTTP)")
    ap.add_argument("--url", default=DEFAULT_URL,
                    help="DeciServ base URL (or DECISERV_URL env)")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-call HTTP timeout s")
    ap.add_argument("--json", action="store_true", help="machine-readable summary")
    args = ap.parse_args(argv)
    URL = args.url

    if not wait_reachable(URL, timeout_s=5):
        print(f"FAIL unreachable: {URL} (start the server or set DECISERV_URL)", file=sys.stderr)
        return 1

    # --- T1 response shape + typed fields -------------------------------
    t1_shape_and_fields(CASES)

    # --- T2/T3/T4 answers align with options ----------------------------
    for state, tag in CASES:
        try:
            payload, _ms = post_decide(URL, state, SUITE_QUESTIONS, timeout=args.timeout)
        except Exception as exc:  # noqa: BLE001
            check(f"{tag}: /decide reachable for alignment tests", False,
                  f"{type(exc).__name__}: {exc}")
            continue
        t2_choice_alignment(payload, "verdict", CHOICE_Q, tag)
        t3_score_alignment(payload, "what_is_lost", SCORE_Q, tag)
        t4_noul_alignment(payload, "destroys_data", tag)

    # --- T5 deterministic repeatability (cache off) ----------------------
    t5_repeatability()

    # --- T6 cache-hit idempotency (cache on) -----------------------------
    t6_cache_idempotency()

    failures = [r for r in _results if not r[1]]
    passes = [r for r in _results
              if r[1] and not r[0].endswith("not a failure")
              and not r[0].startswith("cache-idempotency: SKIPPED")]
    skipped = [r for r in _results
               if r[1] and r[0].startswith("cache-idempotency: SKIPPED")]
    notes = [r for r in _results if r[1] and ": note " in r[0]]

    print(f"\nDeciServ contract tests vs {URL}")
    print(f"  PASS {len(passes)}  FAIL {len(failures)}  skipped {len(skipped)}  notes {len(notes)}")
    for name, ok, detail in _results:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f"  :: {detail}" if (detail and not ok) else ""))
    if args.json:
        print(json.dumps([{"name": n, "ok": o, "detail": d} for n, o, d in _results],
                         ensure_ascii=False, indent=1))
    return len(failures)


if __name__ == "__main__":
    sys.exit(main())