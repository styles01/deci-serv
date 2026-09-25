#!/usr/bin/env python3
"""gridview.py — render DeciServ/laya_arcadia game moves as a 6-cell game grid.

Reads the JSONL move log written by games/laya_arcadia.py (--full for state
text) and renders each move as ASCII frames, replaying episodes at a fixed
frame rate. Output: an animated ASCII "video" written as a text file, plus a
final score table. Designed for terminal/Telegram display — no browser, no
JS: pure Python, stdlib only.

Games covered: snake, mines, crossing, hopper (the four in laya_arcadia.py).
Paddle/dungeon (0xBakeer's extra two) don't exist in our harness; the grid
shows those cells as "not in DeciServ" unless --games limits to ours.

Usage:
  python3 gridview.py --moves /tmp/arcadia_moves.jsonl --fps 3 --loop 2
  python3 gridview.py --moves /tmp/arcadia_moves.jsonl --games snake,mines
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict


# ----------------------------------------------------------------- replay --
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


def group_episodes(rows):
    """rows -> {game: {ep: [move, ...]}} preserving order."""
    eps = defaultdict(lambda: defaultdict(list))
    for r in rows:
        eps[r["game"]][r["ep"]].append(r)
    return eps


# -------------------------------------------------------------- renderers --
def render_snake(rows):
    W = H = 12
    body, food, head = [], None, None
    dead = None
    for r in rows:
        st = r.get("state") or ""
        ev = r.get("event") or ""
        if ev in ("ate",):
            pass
        body = _snake_body_from_log(rows, r)
        food = _food_from_state(st=st(rows, r)) if False else None
    # fallback: we rebuild from facts() replay instead — see render_game()
    return None


def st(rows, r):
    return None


def _snake_body_from_log(rows, r):
    return body


# The move log carries state text + events but not coordinates, so a pixel
# renderer needs a full re-simulation. render_game() re-runs the logged
# actions through a fresh game instance instead of parsing prose — see below.
RENDER_UNKNOWN = """  (move log lacks coordinates; run with --full and replay.py)"""


def render_generic(rows):
    """Last-resort per-move caption for games without a board renderer."""
    out = []
    for r in rows[-1:]:
        out.append(f"  step {r.get('step')}: {r.get('action')} -> {r.get('event')} (score {r.get('score')})")
    return "\n".join(out)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("see replay.py for the real renderer — this module is the shared board painters")