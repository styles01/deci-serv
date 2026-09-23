# Client examples for DeciServ

## Plain requests (Python)

```python
import requests

GATE = "http://localhost:8710"

def ask(state, questions):
    r = requests.post(f"{GATE}/decide", json={"state": state, "questions": questions}, timeout=5)
    r.raise_for_status()
    return r.json()["answers"]

answers = ask(
    "User asks the agent to rm -rf the database folder during a benchmark run.",
    {"proceed": {
        "type": "choice",
        "instructions": "Should this destructive command proceed without asking the human?",
        "criteria": {"proceed": "run it", "escalate": "ask first", "refuse": "refuse"},
    }},
)
p = answers["proceed"]["probabilities"]
```

## Hermes Agent pre_tool_call hook (plugin)

```python
# ~/.hermes/plugins/laya-gate/hook.py (sketch — thresholds are fitted on YOUR traffic)
DECISERV = "http://localhost:8710"
BLOCK_AT = 0.70   # inclusive-lower; conformal fit overrides in production
ESCALATE_AT = 0.40

def on_pre_tool_call(tool_name, args, **kw):
    try:
        r = requests.post(f"{DECISERV}/decide", json={
            "state": f"tool={tool_name} args={json.dumps(args)[:2000]}",
            "questions": {"block": {"type": "choice",
                                    "instructions": "Should this tool call be blocked?",
                                    "criteria": {"block": "unsafe/destructive",
                                                 "escalate": "ambiguous - ask human",
                                                 "pass": "safe to run"}}},
        }, timeout=2)
        p_block = r.json()["answers"]["block"]["probabilities"]["block"]
    except Exception:
        # fail-closed for destructive classes, pass-with-log for read-only
        return {"action": "block"} if tool_name in DESTRUCTIVE else None
    if p_block >= BLOCK_AT:
        return {"action": "block", "message": f"[laya-gate] blocked (p={p_block:.2f})"}
    if p_block >= ESCALATE_AT:
        return {"action": "approve"}   # escalate to the human approval gate
    return None
```

## curl one-liner

```bash
curl -s localhost:8710/decide -H 'Content-Type: application/json' -d '{
  "state": "Should the agent email this draft to the whole company list right now?",
  "questions": {"send": {"type": "choice",
    "instructions": "Send immediately or hold for review?",
    "criteria": {"proceed": "send now", "escalate": "hold for review", "refuse": "do not send"}}}}'
```