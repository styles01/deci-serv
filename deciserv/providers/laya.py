"""Laya provider — convaiinnovations/laya (System-1 decision model, RLCD-trained).

Checkpoints (HF convaiinnovations/*):
  * laya               ModernBERT-large 421M, 512 tok/question, English
  * laya-multilingual  mmBERT-base 322M, 1024 ctx, 100+ languages, ~2.2x faster
  * laya-typed-decisions  ModernBERT-large 421M, 1024 ctx, Choice/Score/Noul

All questions are scored in ONE forward pass (option-marker scorer); the vendor
ships per-cardinality calibration temperatures. Never generates text.

Expected on-box (DGX Spark GB10, fp16, measured 2026-09-23):
  load ~20s, resident ~2.3GB, gate decision p50 ~13ms (lane-idle).
"""
from __future__ import annotations
import json
import os
import sys
import time

from .base import Provider


class LayaProvider(Provider):
    """Wraps rl_agent_api.RLAgent from the laya repo (trust_remote_code layout)."""

    model_name = "laya"

    def load(self):
        self._t0 = time.time()
        import torch  # noqa: F401 — imported here so server starts clean without CUDA
        self.ckpt = os.path.abspath(self.checkpoint)
        sys.path.insert(0, self.ckpt)
        os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", self.ckpt))
        from rl_agent_api import RLAgent  # ships inside the laya checkpoint dir
        self.agent = RLAgent(self.ckpt, device=self.device)
        if self.precision in ("fp16", "bf16"):
            dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
            for attr in ("model", "net", "encoder"):  # tolerate naming drift
                m = getattr(self.agent, attr, None)
                if m is not None and hasattr(m, "half"):
                    setattr(self.agent, attr, m.to(dtype))
                    break
            else:
                # fall back: convert every direct torch module attribute
                for attr in dir(self.agent):
                    m = getattr(self.agent, attr, None)
                    if hasattr(m, "half") and not attr.startswith("_"):
                        try:
                            setattr(self.agent, attr, m.half())
                            break
                        except Exception:
                            continue
        self.load_s = time.time() - self._t0

    def decide(self, state: str, questions: dict) -> dict:
        import torch
        with torch.inference_mode():
            res = self.agent.system_one(state, questions)
        return res

    def describe(self) -> dict:
        return {"model": f"laya ({os.path.basename(self.ckpt)})",
                "precision": self.precision,
                "device": self.device,
                "load_s": round(self.load_s, 1)}