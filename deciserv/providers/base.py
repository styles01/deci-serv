"""Provider contract: one System-1 decision model, resident, behind /decide."""
from __future__ import annotations


class Provider:
    """Base class for decision-model backends.

    Subclasses implement load() + decide(). The HTTP layer owns nothing
    model-specific — providers are the only model-coupled code.
    """

    def __init__(self, checkpoint: str, precision: str = "fp16", device: str = "cuda"):
        self.checkpoint = checkpoint
        self.precision = precision
        self.device = device
        self._t0 = None
        self.load()

    # -- lifecycle -----------------------------------------------------------
    def load(self) -> float:
        """Load model; returns load seconds. Must set self._t0."""
        raise NotImplementedError

    # -- Jev-shaped contract -------------------------------------------------
    def decide(self, state: str, questions: dict) -> dict:
        """state: text; questions: {q: {"type": "choice"|"score", ...}}.

        Returns {"answers": {q: {"type", "choice"/"score", "probabilities", "confidence"}}}.
        """
        raise NotImplementedError

    # -- introspection -------------------------------------------------------
    def describe(self) -> dict:
        return {"model": getattr(self, "model_name", "unknown"),
                "precision": self.precision,
                "device": self.device,
                "load_s": round(getattr(self, "load_s", -1), 1)}

    def memory_stats(self) -> dict:
        out = {}
        try:
            import torch
            if self.device.startswith("cuda") and torch.cuda.is_available():
                out["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        except Exception:
            pass
        return out