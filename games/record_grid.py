#!/usr/bin/env python3
"""record_grid.py — play OUR six games against OUR gate and record a replay.

One pass, six games (4 ours + paddle/dungeon ported from arbiter, MIT),
every tick logged with everything the renderer needs: board snapshot, action,
model probabilities, shield interventions, events, score, caption state.
Output replay files are DROP-IN for games/grid_replay.py — same ticks schema
as arbiter's showcase replays (action/model_action/intervened/reason/events/
summary), so the same renderer draws both his runs and ours.

Usage:
  python3 games/record_grid.py --server http://192.168.2.185:8710 \
      --out replays/laya-v4 [--episodes 1] [--tag v4-base]
  python3 games/record_grid.py --offline --out replays/shield-only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import laya_arcadia as LA          # noqa: E402
import arcadia_extra as AX         # noqa: E402

# grid order: matches the tweet layout (2 rows x 3 cols)
GAMES = ["snake", "hopper", "crossing", "paddle", "mines", "dungeon"]
GAME_CLASSES = dict(LA.GAMES, **AX.GAMES_EXTRA)


# ------------------------------------------------------------- painters ----
# Board snapshots are rebuilt from the live game objects at log time (truth,
# not reconstruction): grids + glyph sets compatible with grid_replay's painter.
def board_snake(g):
    W, H = g.W, g.H
    grid = [["."] * W for _ in range(H)]
    for i, (x, y) in enumerate(g.snake):
        grid[y][x] = "O" if i == 0 else "#"
    fx, fy = g.food
    grid[fy][fx] = "*"
    return ["".join(r) for r in grid]


def board_hopper(g):
    W, H = 9, 12
    grid = [[" "] * W for _ in range(H)]
    # platform heights ladder (g.y unused by physics; paint platforms as rows)
    for px, py in g.platforms:
        if 0 <= py < H:
            for dx in (-1, 0, 1):
                if 0 <= px + dx < W:
                    grid[H - 1 - py][px + dx] = "="
    if 0 <= g.x < W:
        row = (H - 1 - (g.score * 2)) % H  # height proxy: platforms climbed
        grid[row][g.x] = "^"
    return ["".join(r) for r in grid]


def board_crossing(g):
    L, W = g.LANES, g.W
    grid = []
    for lane in range(L + 1):
        r = []
        for x in range(W):
            if lane == 0 or lane == L:
                r.append("=")
            elif lane in g.cars and x in g.cars[lane]["pos"]:
                r.append("X")
            elif lane == g.row and x == g.col:
                r.append("O")
            else:
                r.append(" ")
        grid.append("".join(r))
    return grid


def board_paddle(g):
    W, H = g.W, g.H
    grid = [[" "] * W for _ in range(H)]
    grid[g.by][g.bx] = "O"
    for dx in (-1, 0, 1):
        if 0 <= g.px + dx < W:
            grid[H - 1][g.px + dx] = "="
    return ["".join(r) for r in grid]


def board_mines(g):
    N = g.N
    grid = [["."] * N for _ in range(N)]
    for (x, y) in g.opened:
        n = g.count((x, y))
        grid[y][x] = str(n) if n else "+"
    return ["".join(r) for r in grid]


def board_dungeon(g):
    W, H = 24, 7
    grid = [["."] * W for _ in range(H)]
    for x in range(W):
        grid[0][x] = "="
        grid[H - 1][x] = "="
    marker = 1 + (int(g.depth) * 3) % (W - 2)
    grid[H // 2][marker] = "@"
    return ["".join(r) for r in grid]


BOARDS = {"snake": board_snake, "hopper": board_hopper, "crossing": board_crossing,
          "paddle": board_paddle, "mines": board_mines, "dungeon": board_dungeon}


def caption_for(name, g):
    if name == "dungeon":
        return f"depth {g.depth}  hp {g.hp}  gold {g.gold}"
    if name == "mines":
        return f"opened {len(g.opened)}/22"
    if name == "crossing":
        return f"lane {g.row}/8"
    if name == "hopper":
        return f"climbs {g.score}"
    return ""


# ---------------------------------------------------------------- play ----
def record_game(name, game_cls, server, episodes, log_path, seed0=13,
                timeout=30, shield_all=True):
    """Play one game against the gate, writing arbiter-schema ticks."""
    ticks_by_ep = []
    lat = []
    all_interventions = 0
    for ep in range(episodes):
        g = game_cls(seed=seed0 + ep * 7919)
        ticks = []
        while g.alive and g.steps < 250:
            state_text = g.render()
            questions = g.questions()
            answers = LA.decide_once(server, state_text, questions, lat,
                                     timeout=timeout) if server else {}
            act, probs = g.decide(answers)
            model_action = act
            why = None
            if shield_all and hasattr(g, "shield"):
                act2, why = g.shield(act, probs)
                if why:
                    all_interventions += 1
                act = act2
            pre_steps = g.steps
            ev = g.step(act)
            ticks.append({
                "step": g.steps,
                "action": str(act),
                "model_action": str(model_action),
                "intervened": bool(why),
                "reason": why or "",
                "events": [ev] if ev else [],
                "state": state_text,
                "probabilities": _clean_probs(probs),
                "answers": answers,
                "summary": _summary(name, g),
                "board": BOARDS[name](g),
            })
            if not g.alive:
                break
        ticks_by_ep.append(ticks)
        final = ticks[-1]["summary"] if ticks else {}
        print(f"  {name} ep{ep}: steps {len(ticks)}  score {final.get('score', 0)}"
              + (f"  ({final.get('alive', True) and 'alive' or 'dead'})" if ticks else ""))
    # longest episode wins (grid shows one run per game)
    best = max(ticks_by_ep, key=len) if ticks_by_ep else []
    return best, all_interventions


def _clean_probs(probs):
    out = {}
    for k, v in (probs or {}).items():
        try:
            out[str(k)] = round(float(v), 4)
        except (TypeError, ValueError):
            continue
    return out


def _summary(name, g):
    if name == "snake":
        return {"score": g.score, "steps": g.steps, "length": len(g.snake),
                "alive": g.alive}
    if name == "hopper":
        return {"score": g.score, "steps": g.steps, "alive": g.alive}
    if name == "crossing":
        return {"score": g.score, "steps": g.steps, "row": g.row, "alive": g.alive}
    if name == "paddle":
        return {"score": g.score, "steps": g.steps, "alive": g.alive,
                "gap": abs(g.bx - g.px)}
    if name == "mines":
        return {"score": g.score, "steps": g.steps, "alive": g.alive,
                "unknown": g.N * g.N - g.K - len(g.opened),
                "cleared": not g.alive and g.score > 10}
    if name == "dungeon":
        return {"score": g.score, "steps": g.steps, "depth": g.depth,
                "hp": g.hp, "gold": g.gold, "potions": g.potions,
                "alive": g.alive, "escaped": g.escaped}
    return {}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://192.168.2.185:8710")
    ap.add_argument("--offline", action="store_true",
                    help="no gate: decide() falls back to first legal action + shield")
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--out", default=None, help="output dir (default replays/<tag>)")
    ap.add_argument("--tag", default=None, help="run tag, e.g. v4-base (default timestamp)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--meta-note", default="")
    args = ap.parse_args(argv)

    tag = args.tag or datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
    outdir = Path(args.out) if args.out else Path("games/replays") / tag
    outdir.mkdir(parents=True, exist_ok=True)
    server = None if args.offline else args.server
    games = [g for g in args.games.split(",") if g in GAME_CLASSES]

    manifest = {"recorded_at": datetime.now(timezone.utc).isoformat(),
                "server": server or "offline", "tag": tag,
                "games": {}, "note": args.meta_note}
    for name in games:
        t0 = time.time()
        ticks, interventions = record_game(name, GAME_CLASSES[name], server,
                                           args.episodes, None, seed0=args.seed,
                                           timeout=args.timeout)
        path = outdir / f"{name}.json"
        payload = {
            "game": name, "policy": "deciserv", "seed": args.seed,
            "model": manifest["server"], "tick_ms": 0,
            "recorded_at": manifest["recorded_at"],
            "meta": {"interventions": interventions,
                     "episodes": args.episodes},
            "ticks": ticks,
            "summary": ticks[-1]["summary"] if ticks else {},
            "stats": {"wall_s": round(time.time() - t0, 1),
                      "shield_interventions": interventions},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        s = payload["summary"]
        manifest["games"][name] = {"ticks": len(ticks), "score": s.get("score"),
                                   "interventions": interventions,
                                   "file": str(path)}
        print(f"{name:9s} ticks {len(ticks):3d}  score {s.get('score')}  "
              f"shield {interventions:3d}  wall {payload['stats']['wall_s']}s")
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("replays in", outdir)
    print("render:  python3 games/grid_replay.py --replays", outdir,
          "--png /tmp/grid.png --gif /tmp/grid.gif")
    return 0


if __name__ == "__main__":
    sys.exit(main())