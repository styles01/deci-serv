import json, sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "finetune"))
from calibrate import fit_temperature

RUN = "eval/recordings/20260925-043625-battery_v2_v4-base-run.json"
d = json.load(open(RUN))
cases = d.get("cases") or []


def kind(cid):
    if "__mirror" in cid:
        return "mirror"
    if "__rename" in cid:
        return "rename"
    if "__prose" in cid:
        return "prose"
    if cid.startswith("ood-"):
        return "ood"
    return "original"


def p_yes(a):
    """noul answer -> probability of 'true' side. Float noul = p(no);
    string noul ('true'/'false', fast-path/floor envelope) uses the
    recorded probabilities dict; fall back to the hard word."""
    v = a.get("noul")
    if isinstance(v, (int, float)):
        return 1.0 - float(v)
    pr = a.get("probabilities") or {}
    if "true" in pr:
        try:
            return float(pr["true"])
        except (TypeError, ValueError):
            pass
    s = str(v).strip().lower()
    return {"true": 1.0, "false": 0.0}.get(s)


NOUL_YES = {"destructive", "secret_access", "exfiltrate"}
B = {}
n_fit = 0
skipped = 0
for c in cases:
    cid = c["id"]
    if kind(cid) != "original":
        continue
    lab = c["label"]
    if lab not in NOUL_YES:
        continue  # calibration fit uses unambiguous golds only (R2: no ambiguous)
    for qid, a in (c.get("raw", {}).get("answers") or {}).items():
        if a.get("type") != "noul":
            continue
        py = p_yes(a)
        if py is None:
            skipped += 1
            continue
        B.setdefault(qid, ([], []))
        B[qid][0].append([1.0 - py, py])
        B[qid][1].append(1)
        n_fit += 1

print("fit rows:", n_fit, "| skipped unparseable:", skipped)
out = {}
for qid, (ps, ls) in sorted(B.items()):
    t, eb, ea = fit_temperature(ps, ls)
    out[qid] = {"T": round(t, 4), "ece_before": round(eb, 4),
                "ece_after": round(ea, 4), "n": len(ps)}
    print("%-22s n=%4d  T=%6.3f  ECE %.4f -> %.4f" % (qid, len(ps), t, eb, ea))

json.dump({"recipe": "per-qid temperature, battery-v2 recorded answers (v4 base), destructive family only",
           "fitted_on": "battery_v2 originals (destructive/secret_access/exfiltrate gold=1)",
           "run": os.path.basename(RUN), "n": n_fit, "buckets": out},
          open(os.path.join(_HERE, "recordings", "v4_base_calib_battery.json"), "w"), indent=1)
print("saved eval/recordings/v4_base_calib_battery.json")