#!/usr/bin/env python3
"""grid_replay.py — the tweet's six-game grid, rebuilt from arbiter's own replays.

arbiter (MIT, github.com/0xBakeer/arbiter) ships recorded model runs in
showcase/replay/<game>.json: a `ticks` array per game carrying `action`,
`model_action`, `intervened`, `reason`, `events`, and a per-tick `summary`
(ground truth: score, row/height/gap/depth, alive). Interior coordinates are
not snapshotted, so boards are rebuilt as follows — honest fidelity notes:

  * snake: action-log replay; food placement is our RNG (his placement RNG
    isn't recoverable from the log); 'ate' events from HIS log drive growth.
  * hopper: altitude from HIS summary.height each tick; pipes schematic.
  * crossing: lane progress from HIS summary.row; car traffic is our RNG.
  * paddle: ball path schematic; paddle offset from HIS summary.gap.
  * mines: cells opened from HIS 'opened k cells at rNcM' event log.
  * dungeon: depth/hp/gold/potions straight from HIS summaries.
Captions always use HIS numbers (action, shield reason, score), so the grid
never claims anything the replay file doesn't.

Usage:
  python3 games/grid_replay.py --replays vendor/arbiter/showcase/replay \
      --png /tmp/arbiter_grid.png [--gif /tmp/arbiter_grid.gif] [--fps 6]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

GAMES = ["snake", "hopper", "crossing", "paddle", "mines", "dungeon"]


def load_replays(dirpath):
    d = Path(dirpath)
    return {g: json.load(open(d / f"{g}.json")) for g in GAMES if (d / f"{g}.json").exists()}


# ------------------------------------------------------------ game ports --
def replay_snake(data):
    """12x12 snake driven by the logged actions; growth driven by HIS events."""
    seed = data.get("seed", 11)
    rng = random.Random(seed * 977 + 5)
    W = H = 12
    g = {"snake": [(5, 6), (4, 6), (3, 6)], "alive": True, "score": 0}
    body = set(g["snake"])
    free = [(x, y) for x in range(W) for y in range(H) if (x, y) not in body]
    g["food"] = rng.choice(free) if free else g["snake"][0]
    DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
    frames = []
    for t in data.get("ticks", []):
        act = t.get("action")
        s = t.get("summary", {})
        dx, dy = DIRS.get(act, (0, 0))
        hx, hy = g["snake"][0]
        nxt = (hx + dx, hy + dy)
        events = t.get("events") or []
        if g["alive"]:
            if any("ate" in e for e in events):
                g["snake"].insert(0, nxt)
                g["score"] += 1
                body = set(g["snake"])
                free = [(x, y) for x in range(W) for y in range(H) if (x, y) not in body]
                g["food"] = rng.choice(free) if free else g["snake"][0]
            elif 0 <= nxt[0] < W and 0 <= nxt[1] < H and nxt not in body:
                g["snake"].insert(0, nxt)
                g["snake"].pop()
            else:
                g["alive"] = False
        grid = [["."] * W for _ in range(H)]
        for i, (x, y) in enumerate(g["snake"]):
            grid[y][x] = "O" if i == 0 else "#"
        if g["alive"]:
            fx, fy = g["food"]
            grid[fy][fx] = "*"
        frames.append({"step": t.get("step"), "action": act,
                       "intervened": bool(t.get("intervened")),
                       "reason": t.get("reason", ""),
                       "score": s.get("score", g["score"]),
                       "board": ["".join(r) for r in grid],
                       "alive": s.get("alive", g["alive"])})
    return frames


def replay_hopper(data):
    """Altitude from HIS summary.height; gap pipes schematic."""
    frames = []
    W, H = 26, 14
    height = 2.0
    rng = random.Random(data.get("seed", 5) * 131 + 7)
    gap_y = 6
    gaps = []
    for t in data.get("ticks", []):
        s = t.get("summary", {})
        height = float(s.get("height", height))
        if (t.get("step", 0) or 0) % 8 == 0:
            gap_y = 2 + int(rng.random() * (H - 6))
            gaps.append({"x": W - 1, "y": gap_y})
        for gp in gaps:
            gp["x"] -= 1
        gaps = [gp for gp in gaps if gp["x"] >= 0]
        grid = [[" "] * W for _ in range(H)]
        for gp in gaps:
            for yy in range(H):
                if abs(yy - gp["y"]) > 2:
                    grid[yy][gp["x"]] = "|"
        hy = int(min(H - 1, max(0, (H - 1) - height * (H - 1) / 20.0)))
        grid[hy][4] = "^"
        frames.append({"step": t.get("step"), "action": t.get("action"),
                       "intervened": bool(t.get("intervened")),
                       "reason": t.get("reason", ""), "score": s.get("score", 0),
                       "board": ["".join(r) for r in grid],
                       "alive": s.get("alive", True)})
    return frames


def replay_crossing(data):
    """Lane progress from HIS summary.row; car traffic our RNG."""
    seed = data.get("seed", 2)
    rng = random.Random(seed * 313 + 11)
    L, W = 8, 9
    cars = {lane: {"pos": sorted(rng.sample(range(W), 2)), "speed": 1 + (lane % 2),
                   "left": lane % 2 == 1} for lane in range(L)}
    row = 0
    col = W // 2
    frames = []
    for t in data.get("ticks", []):
        s = t.get("summary", {})
        row = s.get("row", row)
        for c in cars.values():
            c["pos"] = [((x - c["speed"]) % W if c["left"] else (x + c["speed"]) % W)
                        for x in c["pos"]]
        if t.get("action") == "left":
            col = max(0, col - 1)
        elif t.get("action") == "right":
            col = min(W - 1, col + 1)
        grid = []
        for lane in range(L + 1):
            r = []
            for x in range(W):
                if lane == 0 or lane == L:
                    r.append("=")
                elif x in cars[lane]["pos"]:
                    r.append("X")
                elif lane == row and x == col:
                    r.append("O")
                else:
                    r.append(" ")
            grid.append("".join(r))
        frames.append({"step": t.get("step"), "action": t.get("action"),
                       "intervened": bool(t.get("intervened")),
                       "reason": t.get("reason", ""), "score": s.get("score", 0),
                       "board": grid, "alive": s.get("alive", True)})
    return frames


def replay_paddle(data):
    """Ball schematic; paddle offset from HIS summary.gap; score HIS."""
    W, H = 26, 12
    frames = []
    gap = 3
    for t in data.get("ticks", []):
        s = t.get("summary", {})
        gap = s.get("gap", gap)
        st = t.get("step", 0) or 0
        ballx = (st * 3) % (W - 2) + 1
        bally = H - 2 - (st % (H - 4))
        pad = max(0, min(W - 2, W // 2 + int(gap) - 1))
        grid = [[" "] * W for _ in range(H)]
        grid[bally][ballx] = "O"
        for dx in (-1, 0, 1):
            if 0 <= pad + dx < W:
                grid[H - 1][pad + dx] = "="
        frames.append({"step": t.get("step"), "action": t.get("action"),
                       "intervened": bool(t.get("intervened")),
                       "reason": t.get("reason", ""), "score": s.get("score", 0),
                       "board": ["".join(r) for r in grid],
                       "alive": s.get("alive", True)})
    return frames


def replay_mines(data):
    """9x9 mines; opened cells parsed from HIS 'opened k cells at rNcM' events."""
    opened = []
    frames = []
    for t in data.get("ticks", []):
        s = t.get("summary", {})
        for e in t.get("events") or []:
            if isinstance(e, str) and e.startswith("opened") and "at " in e:
                try:
                    loc = e.split("at ")[-1].strip()
                    r = int(loc[1]); c = int(loc[3])
                    opened.append((r, c))
                except (IndexError, ValueError):
                    pass
        grid = [["."] * 9 for _ in range(9)]
        for (r, c) in opened:
            if 0 <= r < 9 and 0 <= c < 9:
                grid[r][c] = "+"
        frames.append({"step": t.get("step"), "action": t.get("action"),
                       "intervened": bool(t.get("intervened")),
                       "reason": t.get("reason", ""), "score": s.get("score", 0),
                       "board": ["".join(r) for r in grid],
                       "alive": s.get("alive", True)})
    return frames


def replay_dungeon(data):
    """Depth/hp/gold/potions straight from HIS summaries; corridor schematic."""
    frames = []
    depth, hp, gold = 1, 5, 0
    W, H = 24, 7
    for t in data.get("ticks", []):
        s = t.get("summary", {})
        depth = s.get("depth", depth)
        hp = s.get("hp", hp)
        gold = s.get("gold", gold)
        marker = 1 + (int(depth) * 3) % (W - 2)
        grid = [["."] * W for _ in range(H)]
        for x in range(W):
            grid[0][x] = "="
            grid[H - 1][x] = "="
        grid[H // 2][marker] = "@"
        frames.append({"step": t.get("step"), "action": t.get("action"),
                       "intervened": bool(t.get("intervened")),
                       "reason": t.get("reason", ""),
                       "score": s.get("score", gold),
                       "board": ["".join(r) for r in grid],
                       "alive": s.get("alive", True),
                       "caption": f"depth {depth}  hp {hp}  gold {gold}"})
    return frames


REPLAY = {"snake": replay_snake, "hopper": replay_hopper, "crossing": replay_crossing,
          "paddle": replay_paddle, "mines": replay_mines, "dungeon": replay_dungeon}


def replay_any(game, data):
    """Use the recorded boards when the replay carries them (our record_grid
    output — ground truth per tick); fall back to re-simulation for arbiter's
    replay files, which snapshot summaries only."""
    ticks = data.get("ticks") or []
    if ticks and all(isinstance(t.get("board"), list) and t["board"] for t in ticks):
        return [{"step": t.get("step"), "action": t.get("action"),
                 "intervened": bool(t.get("intervened")),
                 "reason": t.get("reason", ""),
                 "score": (t.get("summary") or {}).get("score", ""),
                 "board": t["board"],
                 "alive": (t.get("summary") or {}).get("alive", True)}
                for t in ticks]
    return REPLAY[game](data)


# ---------------------------------------------------------------- image ----
ACCENT = {
    "snake": (86, 225, 125), "hopper": (92, 200, 242), "crossing": (232, 160, 62),
    "paddle": (200, 120, 220), "mines": (120, 185, 242), "dungeon": (245, 200, 60),
}


def _font(sz):
    from PIL import ImageFont
    for path in ("/System/Library/Fonts/Menlo.ttc",
                 "/System/Library/Fonts/Monaco.dfont",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(path, sz)
        except Exception:
            continue
    return ImageFont.load_default()


# 1920x1080 layout — matches the tweet's video resolution
GRID_W, GRID_H = 1920, 1080
HEADER_H = 48
PAD = 14
COLS_N, ROWS_N = 3, 2
PANEL_W = (GRID_W - (COLS_N + 1) * PAD) // COLS_N
PANEL_H = (GRID_H - HEADER_H - (ROWS_N + 1) * PAD) // ROWS_N

COL_BG     = (7, 11, 20)
COL_PANEL  = (12, 17, 32)
COL_BORDER = (28, 38, 64)
COL_BOARD  = (10, 15, 28)
COL_GRID   = (18, 26, 45)
COL_TXT    = (225, 232, 248)
COL_DIM    = (108, 120, 150)
COL_BLUE   = (70, 115, 225)
COL_CHIP   = (26, 58, 130)
COL_RED    = (226, 88, 76)
COL_ORANGE = (240, 140, 40)

ENTITY = {
    "snake":    (141, 168, 235),   # periwinkle body, like his
    "hopper":   (120, 150, 220),   # platform bars
    "crossing": (232, 96, 80),     # salmon cars
    "paddle":   (70, 115, 225),    # paddle bar
    "mines":    (95, 150, 240),    # numbers
    "dungeon":  (200, 210, 235),
}

FOOTER = {
    "snake": "One forward pass per move; the shield owns the walls.",
    "hopper": "A choice of three per tick; the wall sinks as you climb.",
    "crossing": "A move and a lane-clear score, in the same call.",
    "paddle": "The tick is as fast as the gate answers.",
    "mines": "One choice over every closed cell, each tick.",
    "dungeon": "Prose in, typed actions and a danger score out.",
}
COUNTER = {  # top-right (label, key in tick summary)
    "snake": ("len", "length"), "hopper": ("climb", "score"),
    "crossing": ("lane", "row"), "paddle": ("gap", "gap"),
    "mines": ("closed", "unknown"), "dungeon": ("gold", "gold"),
}


def _font(sz, italic=False):
    from PIL import ImageFont
    if italic:
        for path in ("/System/Library/Fonts/Supplemental/Arial Italic.ttf",
                     "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf"):
            try:
                return ImageFont.truetype(path, sz)
            except Exception:
                continue
    for path in ("/System/Library/Fonts/Menlo.ttc",
                 "/System/Library/Fonts/Monaco.dfont",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(path, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap(s, n):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


def _draw_entities(d, game, board, ox, oy, c):
    """Vector-draw board glyphs as rounded shapes, in his style."""
    ent = ENTITY[game]
    for r_i, line in enumerate(board):
        for c_i, ch in enumerate(line):
            x, y = ox + c_i * c, oy + r_i * c
            if game == "snake":
                if ch == "#":
                    d.rounded_rectangle([x + 1, y + 1, x + c - 1, y + c - 1],
                                        radius=max(2, int(c * .3)), fill=ent)
                elif ch == "O":
                    d.rounded_rectangle([x, y, x + c, y + c],
                                        radius=max(2, int(c * .35)), fill=(250, 252, 255))
                    if c >= 12:
                        ex, ey = x + c // 4, y + c // 3
                        d.ellipse([ex, ey, ex + 3, ey + 3], fill=(20, 26, 44))
                        d.ellipse([ex + c // 2, ey, ex + c // 2 + 3, ey + 3], fill=(20, 26, 44))
                elif ch == "*":
                    d.rounded_rectangle([x + 2, y + 2, x + c - 2, y + c - 2],
                                        radius=max(2, int(c * .25)), fill=COL_ORANGE)
            elif game == "hopper":
                if ch == "=":
                    hh = max(3, int(c * .38))
                    d.rounded_rectangle([x, y + (c - hh) // 2, x + c, y + (c - hh) // 2 + hh],
                                        radius=2, fill=ent)
                elif ch == "^":
                    d.rounded_rectangle([x + 1, y - c // 3, x + c - 1, y + c],
                                        radius=max(2, int(c * .4)), fill=(250, 252, 255))
            elif game == "crossing":
                if ch == "X":
                    d.rounded_rectangle([x + 1, y + 1, x + c - 1, y + c - 1],
                                        radius=max(2, int(c * .3)), fill=ent)
                    if c >= 12:
                        d.rounded_rectangle([x + c // 4, y + c // 4,
                                             x + 3 * c // 4, y + c // 2],
                                            radius=2, fill=(150, 55, 45))
                elif ch == "O":
                    d.rounded_rectangle([x + 2, y, x + c - 2, y + c],
                                        radius=max(2, int(c * .45)), fill=(250, 252, 255))
                elif ch == "=":
                    d.rectangle([x, y + c // 2 - 1, x + c, y + c // 2 + 1], fill=(62, 74, 104))
            elif game == "paddle":
                if ch == "O":
                    d.ellipse([x + 1, y + 1, x + c - 1, y + c - 1], fill=COL_ORANGE)
                elif ch == "=":
                    d.rounded_rectangle([x, y + c // 4, x + c, y + 3 * c // 4],
                                        radius=3, fill=ent)
            elif game == "mines":
                if ch.isdigit():
                    col = COL_ORANGE if ch in "12" and False else ent
                    d.text((x + c // 6, y + c // 8), ch, fill=ent, font=_font(max(10, c)))
                elif ch == "+":
                    d.ellipse([x + c // 3, y + c // 3, x + 2 * c // 3, y + 2 * c // 3],
                              fill=ent)
            elif game == "dungeon":
                if ch == "=":
                    d.rectangle([x, y + c // 2, x + c, y + c // 2 + 2], fill=(62, 74, 104))
                elif ch == "@":
                    d.ellipse([x + 1, y + 1, x + c - 1, y + c - 1], fill=(250, 252, 255))


def _draw_probs_panel(d, x0, y0, w, h, fr, game):
    """Sidebar: the decision trace — option bars, like his telemetry column."""
    y = y0
    d.text((x0, y), "move", fill=COL_DIM, font=_font(13))
    d.text((x0 + w - 46, y), f"q {len(fr.get('probabilities') or {})}",
           fill=COL_DIM, font=_font(12))
    y += 22
    probs = fr.get("probabilities") or {}
    rows = sorted(probs.items(), key=lambda kv: -kv[1])[:8]
    act = str(fr.get("action"))
    if rows:
        row_h = 17
        for name, p in rows:
            chosen = (name == act)
            nm = name if len(name) <= 12 else name[:11] + "…"
            d.text((x0, y + 1), nm, fill=COL_TXT if chosen else COL_DIM, font=_font(12))
            bx, bw = x0 + 86, w - 86 - 40
            d.rectangle([bx, y + 3, bx + bw, y + 11], fill=(20, 28, 48))
            d.rectangle([bx, y + 3, bx + int(bw * min(1.0, p)), y + 11],
                        fill=COL_BLUE if chosen else (52, 76, 140))
            d.text((x0 + w - 34, y + 1), f"{p:.2f}",
                   fill=COL_TXT if chosen else COL_DIM, font=_font(12))
            y += row_h
    else:
        s = fr.get("summary") or {}
        for k, v in list(s.items())[:8]:
            d.text((x0, y), f"{k}", fill=COL_DIM, font=_font(12))
            d.text((x0 + w - 60, y), str(v), fill=COL_TXT, font=_font(12))
            y += 17
        d.text((x0, y + 6), "(re-simulated)", fill=(80, 88, 110), font=_font(12, italic=True))
        y += 24
    if fr.get("intervened"):
        yy = y0 + h - 44
        d.text((x0, yy), "[shield]", fill=COL_RED, font=_font(13))
        yy += 17
        for ln in _wrap(str(fr.get("reason", "")), 26)[:2]:
            d.text((x0, yy), ln, fill=COL_RED, font=_font(12))
            yy += 15
    return y


def _draw_dungeon_prose(d, fr, bx0, by0, bw, bh):
    """Dungeon panel: the room paragraph IS the board (text-native game)."""
    state = fr.get("state") or ""
    lines = _wrap(state, 44)[:14]
    y = by0 + 8
    for ln in lines:
        d.text((bx0 + 10, y), ln, fill=(212, 220, 240), font=_font(14))
        y += 19
    s = fr.get("summary") or {}
    hp = s.get("hp", 20)
    hw = int((bw - 20) * max(0, hp) / 20)
    d.rectangle([bx0 + 10, by0 + bh - 26, bx0 + 10 + (bw - 20), by0 + bh - 20],
                fill=(30, 20, 26))
    d.rectangle([bx0 + 10, by0 + bh - 26, bx0 + 10 + hw, by0 + bh - 20], fill=COL_RED)
    d.text((bx0 + 10, by0 + bh - 46),
           f"hp {hp}/20   potions {s.get('potions', 0)}   depth {s.get('depth', '?')}",
           fill=COL_DIM, font=_font(13))


def _draw_frame(img, replays, ti, title, subtitle=None, credit=None):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, GRID_W, HEADER_H], fill=(10, 14, 26))
    d.rectangle([24, 18, 36, 30], fill=COL_ORANGE)
    d.text((46, 12), title, fill=COL_TXT, font=_font(22))
    if subtitle:
        tw = d.textlength(subtitle, font=_font(18))
        d.text(((GRID_W - tw) // 2, 15), subtitle, fill=COL_DIM, font=_font(18))
    if credit:
        tw = d.textlength(credit, font=_font(16))
        d.text((GRID_W - tw - 24, 17), credit, fill=COL_BLUE, font=_font(16))

    for idx, game in enumerate(GAMES):
        col, row = idx % COLS_N, idx // COLS_N
        x0 = PAD + col * (PANEL_W + PAD)
        y0 = HEADER_H + PAD + row * (PANEL_H + PAD)
        d.rounded_rectangle([x0, y0, x0 + PANEL_W, y0 + PANEL_H],
                            radius=8, fill=COL_PANEL, outline=COL_BORDER, width=1)
        # label chip
        label = f"{idx + 1:02d} {game}"
        cw = d.textlength(label, font=_font(14)) + 16
        d.rounded_rectangle([x0 + 10, y0 + 9, x0 + 10 + cw, y0 + 31],
                            radius=5, fill=COL_CHIP)
        d.text((x0 + 18, y0 + 12), label, fill=(219, 231, 255), font=_font(14))
        rep = replays.get(game)
        if not rep:
            d.text((x0 + 16, y0 + PANEL_H // 2), "no replay",
                   fill=COL_DIM, font=_font(18))
            continue
        fr = rep[min(ti, len(rep) - 1)]
        board = fr["board"]
        gh, gw = len(board), max(len(b) for b in board)

        # board card (left ~64%) + sidebar (right)
        card_x, card_y = x0 + 10, y0 + 40
        card_w = int(PANEL_W * 0.63)
        card_h = PANEL_H - 40 - 34
        d.rounded_rectangle([card_x, card_y0 := card_y, card_x + card_w, card_y + card_h]
                            if False else [card_x, card_y, card_x + card_w, card_y + card_h],
                            radius=8, fill=COL_BOARD, outline=COL_BORDER, width=1)
        # counters over the card
        s = fr.get("summary") or {}
        d.text((card_x + 12, card_y + 8), str(s.get("score", fr.get("score", 0))),
               fill=COL_TXT, font=_font(26))
        d.text((card_x + 22 + d.textlength(str(s.get("score", 0)), font=_font(26)),
                card_y + 14), str(fr.get("step", "")), fill=COL_DIM, font=_font(15))
        lbl, key = COUNTER[game]
        v = s.get(key)
        if v is not None:
            t = f"{v} {lbl}"
            tw = d.textlength(t, font=_font(16))
            d.text((card_x + card_w - tw - 12, card_y + 12), t,
                   fill=COL_DIM, font=_font(16))
        # board tiles
        if game == "dungeon" and fr.get("state"):
            _draw_dungeon_prose(d, fr, card_x, card_y, card_w, card_h)
        else:
            in_x, in_y = card_x + 10, card_y + 42
            in_w, in_h = card_w - 20, card_h - 40
            c = int(min(in_w / gw, in_h / gh))
            c = max(c, 3)
            ox = card_x + 10 + (in_w - c * gw) // 2
            oy = card_y + 12 + (in_h - c * gh) // 2
            if game in ("mines", "crossing"):
                for gx in range(gw + 1):
                    d.line([ox + gx * c, oy, ox + gx * c, oy + gh * c], fill=COL_GRID)
                for gy in range(gh + 1):
                    d.line([ox, oy + gy * c, ox + gw * c, oy + gy * c], fill=COL_GRID)
            _draw_entities(d, game, board, ox, oy, c)
            if game == "mines" and fr.get("probabilities"):
                import re
                ranked = sorted(fr["probabilities"].items(), key=lambda kv: -kv[1])[:6]
                for key_s, p in ranked:
                    m = re.match(r"\((\d+),\s*(\d+)\)", key_s := key_s if False else key_s)
                    if not m:
                        m = re.match(r"\((\d+),\s*(\d+)\)", str(key_s))
                    if not m:
                        continue
                    cc, rr = int(m.group(1)), int(m.group(2))
                    if cc >= gw or rr >= gh:
                        continue
                    txt = f"{p:.2f}"
                    cw2 = d.textlength(txt, font=_font(11)) + 8
                    cx = ox + cc * c
                    cy = oy + rr * c - 6
                    d.rounded_rectangle([cx, cy, cx + cw2, cy + 15], radius=3,
                                        fill=(16, 22, 40), outline=COL_BORDER)
                    d.text((cx + 4, cy + 1), txt, fill=COL_TXT, font=_font(11))
        # sidebar
        sb_x = card_x + card_w + 10
        sb_w = x0 + PANEL_W - sb_x - 12
        _draw_probs_panel(d, sb_x, card_y + 4, sb_w, card_h - 8, fr, game)
        # footer caption
        cap = FOOTER[game]
        tw = d.textlength(cap, font=_font(15, italic=True))
        d.text((x0 + (PANEL_W - tw) // 2, y0 + PANEL_H - 27), cap,
               fill=(92, 102, 128), font=_font(15, italic=True))


def _default_title(replays_dir):
    p = Path(replays_dir)
    if "arbiter" in str(p):
        return "arbiter plays Laya — six games, one forward pass per move (replay)"
    return "DeciServ plays Laya Arcadia — six games, one forward pass per move"


def render_png(replays, out_path, tick_idx=10 ** 9, title=None):
    from PIL import Image
    W, H = GRID_W, GRID_H
    img = Image.new("RGB", (W, H), (15, 16, 20))
    _draw_frame(img, replays, tick_idx,
                title or "arbiter plays Laya — six games, one forward pass per move (replay)")
    img.save(out_path)
    return out_path


def render_gif(replays, out_path, fps=6, max_frames=300, title=None):
    from PIL import Image
    n = max(len(r) for r in replays.values()) if replays else 0
    n = min(n, max_frames)
    W, H = GRID_W, GRID_H
    imgs = []
    for ti in range(n):
        img = Image.new("RGB", (W, H), (15, 16, 20))
        _draw_frame(img, replays, ti,
                    title or "arbiter plays Laya — six games, one pass per move")
        imgs.append(img)
    if imgs:
        imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default="vendor/arbiter/showcase/replay")
    ap.add_argument("--png", default="/tmp/arbiter_grid.png")
    ap.add_argument("--gif", default=None)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--tick", type=int, default=10 ** 9)
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)

    raw = load_replays(args.replays)
    replays = {}
    for game, data in raw.items():
        frames = replay_any(game, data)
        replays[game] = frames
        last = frames[-1] if frames else {}
        print(f"{game:9s} ticks {len(frames):3d}  final score {last.get('score', '?')}")
    title = args.title or _default_title(args.replays)
    path = render_png(replays, args.png, tick_idx=args.tick, title=title)
    print("png:", path)
    if args.gif:
        gpath = render_gif(replays, args.gif, fps=args.fps, title=title)
        print("gif:", gpath)
    return 0


if __name__ == "__main__":
    main()