import time, json, torch, sys
sys.path.insert(0, "/home/jaita/models/hf/Mapika/decider-2b")
import decider.engine as DE
from decider.infer import Decider

d = Decider("/home/jaita/models/hf/Mapika/decider-2b")

from transformers.models.qwen3_5 import modeling_qwen3_5 as mq


def fused_compat(hidden_states=None, weight=None, bias=None, activation=None,
                 seq_idx=None, x=None, **kw):
    h = hidden_states if hidden_states is not None else x
    return DE.fused_causal_conv1d_fn(h, weight, bias, activation)


mq.causal_conv1d_fn = fused_compat
n = 0
for mod in d.m.lm.modules():
    if type(mod).__name__ == "Qwen3_5GatedDeltaNet":
        mod.causal_conv1d_fn = fused_compat
        n += 1
print("rebound", n)

state = ("A paddle must be moved under a falling ball. The ball is right of the paddle "
         "and moving down. Options: move left, stay, move right.")
questions = {"move": {"type": "choice", "options": ["left", "stay", "right"],
                      "instructions": "Which move keeps the rally alive?"}}

d.system_one(state, questions)
torch.cuda.synchronize()
t0 = time.time()
N = 50
for _ in range(N):
    r = d.system_one(state, questions)
torch.cuda.synchronize()
print("latency %.2f ms over %d calls" % ((time.time() - t0) / N * 1000, N))
a = r["answers"]["move"]
print(json.dumps({k: a[k] for k in a if k in ("type", "choice", "probabilities", "confidence")},
                 default=str)[:320])