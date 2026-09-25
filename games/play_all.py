"""Play all four DeciServ games in one pass and write a combined move log.

Wraps games/laya_arcadia.py's play() for each game, merging per-game move
logs into one JSONL for games/replay.py --png (the 2x3 grid).

Usage (server on):
  python3 games/play_all.py --server http://192.168.2.185:8710 --episodes 3
Offline (shield-only, no gate):
  python3 games/play_all.py --offline --episodes 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import laya_arcadia as LA  # noqa: E402

GAMES = ["snake", "mines", "crossing", "hopper"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--server", default="http://192.168.2.185:8710")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", default="/tmp/arcadia_moves.jsonl")
    ap.add_argument("--full", action="store_true", default=True)
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args(argv)

    games = [g for g in args.games.split(",") if g]
    log = open(args.out, "w")
    results = []
    server = None if args.offline else args.server
    for name in games:
        if name not in LA.GAMES:
            print(f"skip {name} (not in harness)")
            continue
        t0 = time.time()
        res = LA.play(LA.GAMES[name], name, server, args.episodes, log,
                      full=True, seed0=13, timeout=args.timeout)
        res["wall_s"] = round(time.time() - t0, 1)
        results.append(res)
        print(f"{name:9s} mean {res['mean']:6.2f}  max {res['max']:3d}  "
              f"shield {res['shield_interventions']:3d}  "
              f"p50 {res['latency_ms_p50']}ms  wall {res['wall_s']}s")
    log.close()
    out = Path(args.out)
    results_path = str(out.with_suffix(".summary.json"))
    with open(results_path, "w") as f:
        json.dump(results, f, indent=1)
    print("moves:", args.out, "| summary:", results_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())