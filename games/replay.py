#!/usr/bin/env python3
"""replay.py — re-simulate logged game actions and render the game grid.

The move log (/tmp/arcadia_moves.jsonl from games/laya_arcadia.py) records
{game, ep, step, action, event, score} per move. Coordinates aren't logged,
so this tool REPLAYS each logged action through the same deterministic game
classes (same seeds: seed0 + ep*7919, exactly like play()) and paints each
tick as ASCII — a frame-accurate "video" of what the model played, no browser.

Modes:
  --png PATH   render the grid image (Pillow, 2x3 cells) — one panel per game
  --txt        ASCII frames to stdout (terminal replay)

Usage:
  python3 games/replay.py --moves /tmp/arcadia_moves.jsonl --png /tmp/grid.png
  python3 games/replay.py --moves /tmp/arcadia_moves.jsonl --txt --fps 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import laya_arcadia as LA  # noqa: E402

OUR_GAMES = ["snake", "mines", "crossing", "hopper"]
GRID_ORDER = OUR_GAMES + ["paddle", "dungeon"]  # 2x3; bakeer's other two marked absent


def load_moves(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def moves_by_ep(rows, game):
    eps = defaultdict(list)
    for r in rows:
        if r.get("game") == game:
            eps[r["ep"]].append(r)
    return {ep: sorted(v, key=lambda r: r["step"]) for ep, ep_moves in eps.items() for ep, v in [(ep, ep_moves)]}


# -------------------------------------------------------------- painters --
def board_snake(g):
    W = H = 12
    grid = [["·" for _ in range(W)] for _ in range(H)]
    for i, (x, y) in enumerate(g.snake):
        grid[y][x] = "◉" if i == 0 else "█"
    fx, fy = g.food
    grid[fy][fx] = "★"
    return ["".join(r) for r in grid]


def board_mines(g):
    N = 5
    grid = []
    for y in range(N):
        row = []
        for x in range(N):
            c = (x, y)
            if c in g.opened:
                n = g.count(c)
                row.append(str(n) if n else "○")
            else:
                row.append("▪")
        grid.append("".join(row))
    return grid


def board_crossing(g):
    L, W = g.LANES, g.W
    grid = []
    for lane in range(L + 1):
        row = []
        for x in range(W):
            if lane == 0 or lane == L:
                row.append("═")
            elif lane in g.cars and x in g.cars[lane]["pos"]:
                row.append("▤")
            elif lane == g.row and x == g.col:
                row.append("☺")
            else:
                row.append(" ")
        grid.append("".join(row))
    return grid


def board_hopper(g):
    W, H = g.W, g.H
    grid = [[" " for _ in range(W)] for _ in range(H)]
    for px, py in g.platforms:
        if 0 <= py < H:
            for dx in (-1, 0, 1):
                if 0 <= px + dx < W:
                    grid[H - 1 - py][px + dx] = "▬"
    if 0 <= g.x < W and 0 <= g.y < H:
        grid[H - 1 - g.y][g.x] = "▲"
    return ["".join(r) for r in grid]


BOARDS = {"snake": board_snake, "mines": board_mines,
          "crossing": board_crossing, "hopper": board_hopper}


def replay_game(game, rows, seed0=13, max_frames=400):
    """Replay the logged actions of one game through fresh instances.

    Returns {"frames": [...], "finals": [...]} — frames sample the FIRST
    episode (visual), finals carry every episode's last state for scores.
    """
    cls = LA.GAMES[game]
    eps = defaultdict(list)
    for r in rows:
        if r.get("game") == game:
            eps[r["ep"]].append(r)
    frames, finals = [], []
    for ep in sorted(eps):
        moves = sorted(eps[ep], key=lambda r: r["step"])
        g = cls(seed=seed0 + ep * 7919)
        shown = 0
        frame = None
        for r in moves:
            act = r["action"]
            ev = g.step(act)
            frame = {"ep": ep, "step": r["step"], "action": r["action"],
                     "event": ev, "score": g.score, "alive": g.alive,
                     "board": BOARDS[game](g) if game in BOARDS else None}
            if ep == 0 and shown < max_frames:
                frames.append(frame)
                shown += 1
            if not g.alive:
                break
        finals.append(frame)
    return {"frames": frames, "finals": finals}


# ------------------------------------------------------------------ png ----
def render_png(replays, out_path):
    from PIL import Image, ImageDraw

    CELL_W, CELL_H = 460, 320
    PAD = 14
    COLS, ROWS = 3, 2
    W = COLS * CELL_W + (COLS + 1) * PAD
    H = ROWS * CELL_H + (ROWS + 1) * PAD + 40
    img = Image.new("RGB", (W, H), (16, 17, 21))
    d = ImageDraw.Draw(img)
    d.text((PAD, 10), "DeciServ gate (laya) plays the arcade — live grid", fill=(225, 225, 230))

    def cell_rect(idx):
        col, row = idx % COLS, idx // COLS
        x0 = PAD + col * (CELL_W + PAD)
        y0 = PAD + 34 + row * (CELL_H + PAD)
        return x0, y0, x0 + CELL_W, y0 + CELL_H

    for idx, name in enumerate(GRID_ORDER):
        x0, y0, x1, y1 = cell_rect(idx)
        d.rectangle([x0, y0, x1, y1], fill=(24, 25, 30), outline=(66, 68, 78))
        rep = replays.get(name)
        if not rep or not rep["frames"]:
            d.text((x0 + 14, y0 + CELL_H // 2 - 8),
                   f"{name}: not in DeciServ harness", fill=(110, 112, 124))
            continue
        last = rep["frames"][-1]
        # board area: fixed cell size, centered
        board = last["board"]
        ch = 10 if max(len(b) for b in board) <= 14 else 8
        bx = x0 + 18
        by = y0 + 34
        for r_i, line in enumerate(board):
            for c_i, cch in enumerate(line):
                color = (70, 74, 84)
                if cch in "█◉":
                    color = (86, 225, 125)
                elif cch == "★":
                    color = (245, 200, 60)
                elif cch == "▤":
                    color = (232, 160, 62)
                elif cch == "☺":
                    color = (245, 245, 250)
                elif cch == "▲":
                    color = (92, 200, 242)
                elif cch in "▬":
                    color = (150, 152, 168)
                elif cch in "12345":
                    color = (120, 185, 242)
                elif cch == "○":
                    color = (95, 165, 225)
                elif cch == "═":
                    color = (86, 88, 98)
                elif cch == "▪":
                    color = (56, 58, 68)
                d.text((bx + c_i * ch, by + r_i * (ch + 2)), cch, fill=color)
        d.text((x0 + 12, y0 + 8),
               f"{name}   ep{last['ep']}   score {last['score']}   step {last['step']} ({last['event']})",
               fill=(228, 228, 232) if last.get("alive", True) else (232, 96, 96))
    img.save(out_path)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", default="/tmp/arcadia_moves.jsonl")
    ap.add_argument("--games", default=",".join(OUR_GAMES))
    ap.add_argument("--png", default=None, help="render 2x3 grid PNG to PATH")
    ap.add_argument("--txt", action="store_true", help="ASCII replay to stdout")
    ap.add_argument("--fps", type=float, default=3.0)
    args = ap.parse_args(argv)

    rows = load_moves(args.moves)
    games = [g for g in args.games.split(",") if g]
    replays = {}
    for g in games:
        if g in LA.GAMES:
            replays[g] = replay_game(g, rows)
    for g, rep in replays.items():
        finals = rep["finals"]
        mean = sum(f["score"] for f in finals) / max(1, len(finals))
        print(f"{g:9s} episodes {len(finals):2d}  mean {mean:6.2f}  "
              f"best {max(f['score'] for f in finals)}")
    if args.png:
        path = render_png(replays, args.png)
        print("png:", path)
    if args.txt:
        fps = args.fps
        for g, rep in replays.items():
            for fr in rep["frames"]:
                if not fr["board"]:
                    continue
                print(f"--- {g} ep{fr['ep']} step {fr['step']} {fr['action']} -> {fr['event']} (score {fr['score']})")
                print("\n".join(fr["board"]))
                time.sleep(1.0 / fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())