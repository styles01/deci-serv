"""Load question packs (adapted from muse-jev-playbook, MIT)."""
import json, os

def load_pack(name: str) -> dict:
    path = os.path.join(os.path.dirname(__file__), f"{name}.json")
    with open(path) as f:
        return json.load(f)

def available() -> list[str]:
    return sorted(f[:-5] for f in os.listdir(os.path.dirname(__file__)) if f.endswith(".json"))
