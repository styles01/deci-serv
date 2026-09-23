"""Decision-model providers for DeciServ.

A provider wraps one System-1 decision model behind the Jev-shaped contract:
    decide(state, questions) -> {"answers": {q: {type, choice|score, probabilities, confidence}}}
"""
from .base import Provider
from .laya import LayaProvider

REGISTRY = {
    "laya": LayaProvider,
}


def load_provider(name: str, checkpoint: str, precision: str = "fp16", device: str = "cuda") -> Provider:
    cls = REGISTRY.get(name)
    if cls is None:
        raise SystemExit(f"unknown provider '{name}'. Available: {', '.join(REGISTRY)}")
    return cls(checkpoint, precision=precision, device=device)


def get_provider() -> dict:
    """Registry metadata for docs/help."""
    return {k: (v.__doc__ or "").strip().split("\n")[0] for k, v in REGISTRY.items()}