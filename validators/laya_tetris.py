#!/usr/bin/env python3
"""LayaTETRIS — decision-model validator.

Tetris as a structured decision benchmark for the Laya checkpoint served by
DeciServ (DGX Spark :8710). The agent never plays via raw LLM text: every piece
placement is ONE typed decision call — POST /decide {state, questions} — with:
  choice  placement   criteria = the legal (column, rotation) moves
  noul    danger      "Is the board one piece away from a tall stack?"
  noul    hold_beneficial (when hold is available)
Deterministic harness owns ALL mechanics (gravity, clears, scoring) — the
decision model only advises, same fail-open pattern as DeciServ gates and the
prism/jev-trader clients. Board state is rendered as compact ASCII.

Validator metrics (what makes this a VALIDATOR, not a toy):
  * decision latency p50/p95 per piece (Spark round-trip)
  * move quality vs the one-step greedy baseline (deterministic simulator)
  * calibration: Laya's `danger` noul vs ACTUAL measured board danger
  * game survival: pieces placed, lines cleared, game-over reason
Render-free headless mode (default): board printed every N pieces + final table.
"""
import json, sys, time, urllib.request
from statistics import median

DECIDESERV = "http://192.168.2.185:8710/decide"
COLS, ROWS = 10, 20

# (name, [(x,y) cells at rotation 0], # of distinct rotations)
PIECES = {
    "I": ([(0,0),(1,0),(2,0),(3,0)], 2),
    "O": ([(0,0),(1,0),(0,1),(1,1)], 1),
    "T": ([(0,0),(1,0),(2,0),(1,1)], 4),
    "S": ([(1,0),(2,0),(0,1),(1,1)], 2),
    "Z": ([(0,0),(1,0),(1,1),(2,1)], 2),
    "J": ([(0,0),(0,1),(1,1),(2,1)], 4),
    "L": ([(2,0),(0,1),(1,1),(2,1)], 4),
}

def rot(cells, r):
    out = cells
    for _ in range(r % 4):
        out = [(y, -x) for x, y in out]  # 90° CW in (col,row) space
    mx = min(c for c, _ in out); my = min(rr for _, rr in out)
    out = [(c - mx, rr - my) for c, rr in out]  # re-normalize to 0-based
    return out

def legal_moves(board, piece):
    base, maxr = PIECES[piece][0], PIECES[piece][1]
    moves = []
    for r in range(maxr):
        cells = rot(base, r)
        xs = [c for c, _ in cells]; ys = [y for _, y in cells]
        w = max(xs) - min(xs) + 1
        for col in range(COLS - w + 1):
            off = min(xs)
            placed = [(col + c - off, y - min(ys)) for c, y in cells]
            if any(y >= ROWS for _, y in placed):
                continue
            # drop
            drop = 0
            while True:
                nxt = [(x, y + drop + 1) for x, y in placed]
                if any(y >= ROWS or board[y][x] for x, y in nxt):
                    break
                drop += 1
            final = [(x, y + drop) for x, y in placed]
            moves.append((r, col, final))
    return moves

def apply(board, cells):
    b = [row[:] for row in board]
    for x, y in cells:
        if 0 <= y < ROWS:
            b[y][x] = 1
        else:
            return None, 0  # overflow = game over move
    nb = [row for row in b if not all(row)]  # keep rows that are NOT full
    cleared = ROWS - len(nb)
    while len(nb) < ROWS:
        nb.insert(0, [0]*COLS)
    return nb, cleared

def col_heights(board):
    return [max((y for y in range(ROWS) if board[y][x]), default=-1)+1 for x in range(COLS)]

def greedy_score(board, cells):
    nb, cleared = apply(board, cells)
    if nb is None:
        return -1e9, None
    h = col_heights(nb)
    agg = sum(h); holes = sum(1 for x in range(COLS)
                              for y in range(ROWS) if nb[y][x]==0 and any(nb[yy][x] for yy in range(y)))
    bump = sum(abs(h[i]-h[i+1]) for i in range(COLS-1))
    return -0.51*agg + 0.76*cleared*10 - 0.36*holes*10 - 0.18*bump, nb

def render(board):
    return "\n".join("".join("[]" if v else ".." for v in row) for row in board)

def ask_laya(board, piece, hold, moves):
    state = (
        f"Tetris. Board {ROWS}x{COLS}, bottom row printed last. Rows top to bottom:\n"
        + "\n".join("".join("#" if v else "." for v in row) for row in board)
        + f"\nCurrent piece: {piece}. Hold available: {hold}.\n"
        + f"Column heights (left to right): {col_heights(board)}\n"
        + "Pick the placement that clears lines if possible and otherwise keeps the skyline flat without making holes."
    )
    qid = "placement"
    questions = {
        qid: {
            "type": "choice",
            "instructions": "Choose the best (rotation, column) placement index.",
            "criteria": [f"rot={r},col={c}" for r, c, _ in [(m[0], m[1], m[2]) for m in moves]],
        },
        "danger": {
            "type": "noul",
            "instructions": "Probability the board is within 2 pieces of an unmanageable stack.",
        },
    }
    body = json.dumps({"state": state, "questions": questions}).encode()
    t0 = time.time()
    req = urllib.request.Request(DECIDESERV, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        out = json.loads(resp.read())
    dt = (time.time() - t0) * 1000
    a = out.get("answers", {})
    pick = a.get(qid)
    if isinstance(pick, dict):
        pick = pick.get("answer", pick.get("choice", pick.get("value", pick)))
    danger_a = a.get("danger", {})
    if isinstance(danger_a, dict):
        danger = danger_a.get("noul", danger_a.get("answer"))
    else:
        danger = danger_a
    return pick, danger, dt, out

def main():
    pieces_cycle = list(PIECES.keys())
    import random
    random.seed(42)
    board = [[0]*COLS for _ in range(ROWS)]
    score = lines = pieces = 0
    lats, calib = [], []
    hold_used = False
    hold = None
    bag = []
    while pieces < 60:
        if not bag:
            bag = pieces_cycle[:]; random.shuffle(bag)
        piece = bag.pop()
        moves = legal_moves(board, piece)
        if not moves:
            print(f"GAME OVER: no legal move for {piece} at piece #{pieces+1}")
            break
        try:
            pick, danger, dt, raw = ask_laya(board, piece, hold, moves)
        except Exception as e:
            print("DECISION CALL FAILED:", e); break
        lats.append(dt)
        # map Laya's answer to an index
        idx = None
        labels = [f"rot={m[0]},col={m[1]}" for m in moves]
        if isinstance(pick, str):
            for i, lab in enumerate(labels):
                if pick == lab or lab in pick:
                    idx = i; break
        elif isinstance(pick, (int, float)) and 0 <= int(pick) < len(moves):
            idx = int(pick)
        if idx is None:  # fallback: parse probability argmax, else random
            probs = None
            if isinstance(pick, dict):
                probs = pick.get("probabilities")
            if isinstance(probs, dict):
                try:
                    idx = max(range(len(labels)), key=lambda i: float(probs.get(labels[i], 0)))
                except Exception:
                    idx = None
            if idx is None:
                idx = random.randrange(len(moves))
        # baseline
        best, _ = max((greedy_score(board, m[2]) + (i,))[0:1] + (i,)
                      for i, m in enumerate(moves)) if False else (None, None)
        scores = [(greedy_score(board, m[2])[0], i) for i, m in enumerate(moves)]
        base_idx = max(scores)[1]
        agree = (idx == base_idx)
        # apply
        r_, c_, cells_ = moves[idx]
        nb, cleared = apply(board, cells_)
        game_over = nb is None
        if game_over:
            print(f"GAME OVER: Laya topped out at piece #{pieces+1} (move rot={r_},col={c_})")
            break
        board = nb
        lines += cleared
        score += [0, 40, 100, 300, 1200][min(cleared, 4)]
        pieces += 1
        h = col_heights(board)
        real_danger = max(h) >= 12
        calib.append((float(danger) if danger is not None else 0.5, real_danger))
        holes_now = sum(1 for x in range(COLS) for y in range(ROWS)
                        if board[y][x] == 0 and any(board[yy][x] for yy in range(y)))
        agree_n = getattr(main, "_agree_n", 0) + (1 if agree else 0)
        main._agree_n = agree_n
        if pieces % 10 == 0 or pieces == 1:
            print(f"[{pieces:3d}] piece={piece} rot={r_} col={c_} clr={cleared} "
                  f"lines={lines} avgH={sum(h)/COLS:.1f} holes={holes_now} "
                  f"laya_danger={danger} {'AGREE' if agree else 'differs'} {dt:.0f}ms")
    # validator table
    n = len(lats)
    if n:
        lats_sorted = sorted(lats)
        p50 = lats_sorted[n//2]; p95 = lats_sorted[min(n-1, int(n*0.95))]
        agree_rate = None
        ag = calib
        hi = [d for d, r in ag if r]; lo = [d for d, r in ag if not r]
        sep = (sum(hi)/len(hi) - sum(lo)/len(lo)) if hi and lo else float('nan')
        print("\n=== VALIDATOR TABLE ===")
        print(f"pieces placed: {pieces}   lines cleared: {lines}   score: {score}")
        print(f"greedy agreement: {main._agree_n}/{pieces} moves match the deterministic baseline")
        print(f"decision latency: p50 {p50:.0f}ms  p95 {p95:.0f}ms  (n={n})")
        print(f"danger calibration: mean(noul|actually-danger)={sum(hi)/max(len(hi),1):.3f} "
              f"mean(noul|safe)={sum(lo)/max(len(lo),1):.3f}  separation={sep:.3f}")
        print("(separation > 0.15 = model tracks real risk; ~0 = flat/OOD as expected pre-finetune)")

if __name__ == "__main__":
    main()