"""RAM/VRAM footprint of decider-2b on the GB10 (unified memory accounting)."""
import time
import sys

sys.path.insert(0, "/home/jaita/models/hf/Mapika/decider-2b")


def host_avail_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) * 1024 / 1e9


def gpu_used_gb():
    import torch
    free_b, total_b = torch.cuda.mem_get_info()
    return (total_b - free_b) / 1e9, total_b / 1e9


t0 = time.time()
from decider.infer import Decider

d = Decider("/home/jaita/models/hf/Mapika/decider-2b")

# transformers 5.14.1 calls causal_conv1d_fn with x= keyword; decider's fused shim
# expects positional hidden_states. Rebind module + live instances with a compat fn.
import decider.engine as DE
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq


def fused_compat(hidden_states=None, weight=None, bias=None, activation=None,
                 seq_idx=None, x=None, **kw):
    h = hidden_states if hidden_states is not None else x
    return DE.fused_causal_conv1d_fn(h, weight, bias, activation)


mq.causal_conv1d_fn = fused_compat
for mod in d.m.lm.modules():
    if type(mod).__name__ == "Qwen3_5GatedDeltaNet":
        mod.causal_conv1d_fn = fused_compat

before = host_avail_gb()
g0, total = gpu_used_gb()
print("load %.1fs" % (time.time() - t0))
a1 = host_avail_gb()
g1, _ = gpu_used_gb()

state = ("A paddle must be moved under a falling ball. The ball is right of the paddle "
         "and moving down. Options: left, stay, right.")
q = {"move": {"type": "choice", "options": ["left", "stay", "right"],
              "instructions": "keep rally alive"}}
d.system_one(state, q)
g2, _ = gpu_used_gb()
a2 = host_avail_gb()

print("GPU before load: %.2f GB | after load: %.2f GB | after inference: %.2f GB (total %.1f)" %
      (g0, g1, g2, total))
print("host MemAvailable: before %.2f GB | after load %.2f GB | after inference %.2f GB" %
      (before, a1, a2))
print("host drop across load+infer: %.2f GB" % (before - a2))