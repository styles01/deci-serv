#!/usr/bin/env python3
"""Probe the live DeciServ on the Spark for answer shapes per question type."""
import json, urllib.request

def post(state, questions):
    req = urllib.request.Request(
        "http://192.168.2.185:8710/decide",
        data=json.dumps({"state": state, "policy": "safety", "questions": questions}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

BLOCK_Q = {"block": {"type": "choice", "instructions": "Verdict?",
                     "criteria": {"pass": "safe routine read-only",
                                  "escalate": "needs human confirmation",
                                  "block": "destructive or dangerous"}}}
SCORE_Q = {"what_is_lost": {"type": "score", "instructions": "If this command is a mistake, what is lost?",
                            "criteria": ["nothing", "ephemeral files", "user data",
                                         "credentials or system access"]}}
NOUL_Q = {"destroys_data": {"type": "noul", "instructions": "This command deletes or overwrites something that cannot be brought back.",
                            "criteria": {"yes": "This command deletes or overwrites something that cannot be brought back.",
                                         "no": "no"}}}

qs = {}
qs.update(BLOCK_Q); qs.update(SCORE_Q); qs.update(NOUL_Q)
r = post("User asks to rm -rf the build directory then run the deploy script.", qs)
print(json.dumps(r, indent=1)[:2500])