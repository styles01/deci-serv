"""Battery v2 head-to-head: laya-gate-v4 (calibrated) vs decider-2b zero-shot.

Rescored identically (same weights/thresholds) from recordings:
  v4: eval/recordings/20260924-051254-battery-run.json  (n=207 pilot slice)
  decider-2b: eval/recordings/20260925-134700-decider2b-zeroshot-run.json (n=1045, full)
"""
import json

v4 = {
    "benign": {"pass": 162, "escalate": 43, "deny": 2, "n": 207},
    "read_only": {"pass": 46, "escalate": 14, "deny": 0, "n": 60},
    "ambiguous": {"pass": 22, "escalate": 17, "deny": 2, "n": 41},
    "destructive": {"pass": 0, "escalate": 13, "deny": 53, "n": 66},
    "secret_access": {"pass": 0, "escalate": 3, "deny": 31, "n": 34},
    "exfiltrate": {"pass": 6, "escalate": 6, "deny": 20, "n": 32},
}
dec = {
    "benign": {"pass": 450, "escalate": 22, "deny": 12, "n": 484},
    "read_only": {"pass": 142, "escalate": 6, "deny": 2, "n": 150},
    "ambiguous": {"pass": 42, "escalate": 42, "deny": 10, "n": 94},
    "destructive": {"pass": 4, "escalate": 40, "deny": 120, "n": 164},
    "secret_access": {"pass": 6, "escalate": 29, "deny": 47, "n": 82},
    "exfiltrate": {"pass": 5, "escalate": 33, "deny": 33, "n": 71},
}

print(f"{'class':14s} {'v4 (n=207)':>24s}   {'decider-2b zs (n=1045)':>26s}")
for lab in ("benign", "read_only", "ambiguous", "destructive", "secret_access", "exfiltrate"):
    def fmt(m, lab):
        r = m[lab]
        n = r["n"]
        if lab in ("destructive", "secret_access", "exfiltrate"):
            safe = 100 * (r["deny"] + r["escalate"]) / n
            return f"deny {100*r['deny']/n:4.1f}% esc {100*r['escalate']/n:4.1f}% pass {100*r['pass']/n:4.1f}%"
        if lab == "ambiguous":
            return f"escalate {100*r['escalate']/n:4.1f}%"
        return f"pass {100*r['pass']/n:4.1f}% esc {100*r['escalate']/n:4.1f}%"
    print(f"{lab:14s} {fmt(v4, lab):>26s}   {fmt(dec, lab):>28s}")
print()
print("separation AUC (destructive vs benign):  v4 0.9934   decider-2b-zs 0.9840")
print("separation AUC (exfil+secret vs benign): v4 0.9575   decider-2b-zs 0.9723")
print("latency p50: v4 103.4ms  decider-2b-zs 163.9ms  (p95: 177 / 224)")
print("NOTE: v4 numbers are the 207-case pilot slice; decider ran the full 1,045.")