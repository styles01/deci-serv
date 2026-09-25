#!/usr/bin/env python3
"""laya_arcadia.py — Decision-model game harness: Laya plays games on our DeciServ
gate (port 8710). Same spirit as 0xBakeer's arbiter showcase (six games); our own impl.

Contract per tick (one forward pass on the gate server):
  - state:  short text board description
  - choice: "move"/"open" over legal actions; criteria = per-action consequence sentences
  - noul:   one auxiliary boolean with exact ground truth (calibration check)
Shield mirrors our DeciServ floors: code owns safety, model owns preference.

Usage:
  python3 laya_arcadia.py --games snake,mines --episodes 20 --server http://192.168.2.185:8710
  python3 laya_arcadia.py --offline   # shield-preference baseline, no server
"""
import argparse, json, time, urllib.request, random

def ask_server(server, state, questions, timeout=30):
    body = json.dumps({"state": state, "questions": questions}).encode()
    req = urllib.request.Request(server.rstrip("/") + "/decide", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

# ---------------------------------------------------------------- snake ----
class Snake:
    W = H = 12
    DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
    OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
    ORDER = ["up", "down", "left", "right"]

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.snake = [(5, 6), (4, 6), (3, 6)]
        self.dir = "right"
        self.alive, self.score, self.steps = True, 0, 0
        self.place_food()

    def place_food(self):
        body = set(self.snake)
        free = [(x, y) for x in range(self.W) for y in range(self.H) if (x, y) not in body]
        self.food = self.rng.choice(free) if free else self.snake[0]

    def legal(self):
        return [d for d in self.ORDER if len(self.snake) < 2 or d != self.OPPOSITE[self.dir]]

    def ahead(self, d):
        dx, dy = self.DIRS[d]; h = self.snake[0]
        return (h[0] + dx, h[1] + dy)

    def inside(self, c):
        return 0 <= c[0] < self.W and 0 <= c[1] < self.H

    def fatal(self, d):
        t = self.ahead(d)
        if not self.inside(t):
            return True
        eats = (t == self.food)
        body = self.snake[:-1] if eats else self.snake
        return t in body

    def space_after(self, d):
        t = self.ahead(d)
        if not self.inside(t):
            return 0
        body = set(self.snake)
        seen, stack = set(), [t]
        while stack:
            c = stack.pop()
            if c in seen or not self.inside(c) or c in body:
                continue
            seen.add(c)
            x, y = c
            stack += [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        return len(seen)

    def render(self):
        hx, hy = self.snake[0]
        dy, dx = self.food[1] - hy, self.food[0] - hx
        where = []
        if dy < 0: where.append(f"{-dy} up")
        if dy > 0: where.append(f"{dy} down")
        if dx < 0: where.append(f"{-dx} left")
        if dx > 0: where.append(f"{dx} right")
        return (f"Snake on a 12 by 12 grid, {len(self.snake)} segments long, moving {self.dir}. "
                f"Food is {' and '.join(where) if where else 'under the head'}.")

    def facts(self, d):
        t = self.ahead(d)
        if not self.inside(t):
            return "the wall is one step away, the snake dies"
        hx, hy = self.snake[0]
        body = set(self.snake)
        bd = None
        for i in range(1, max(self.W, self.H)):
            c = (hx + self.DIRS[d][0] * i, hy + self.DIRS[d][1] * i)
            if not self.inside(c):
                break
            if c in body:
                bd = i
                break
        fx, fy = self.food
        before = abs(hx - fx) + abs(hy - fy)
        after = abs(t[0] - fx) + abs(t[1] - fy)
        closer = "food gets closer" if after < before else "food gets farther"
        sp = self.space_after(d) if self.inside(t) else 0
        L = len(self.snake)
        if sp < L:
            space_s = f"a trap: only {sp} free cells, less than the snake is long"
        elif sp < L * 3:
            space_s = f"{sp} free cells, tight"
        else:
            space_s = "open space beyond"
        parts = [closer, space_s]
        if bd is not None:
            parts.append(f"own body {bd} steps ahead")
        return ", ".join(parts)

    def questions(self):
        criteria = {d: self.facts(d) for d in self.legal()}
        return {"move": {"type": "choice", "instructions":
                    "Pick the direction that eats the food without hitting a wall or the snake itself.",
                    "criteria": criteria},
                "food_reachable": {"type": "noul", "instructions":
                    "There is a clear path from the head to the food, around the snake's body.",
                    "criteria": {"true": "the food can be walked to",
                                 "false": "the body cuts the food off"}}}

    def decide(self, answers):
        probs = (answers.get("move") or {}).get("probabilities", {})
        best, act = -1.0, self.legal()[0]
        for d in self.legal():
            p = probs.get(d, 0) or 0
            if p > best:
                best, act = p, d
        return act, probs

    def shield(self, act, probs):
        if not self.fatal(act):
            return act, None
        safe = [d for d in self.legal() if not self.fatal(d)]
        if not safe:
            return act, "no safe move left"
        pick = max(safe, key=lambda d: probs.get(d, 0) or 0)
        why = "into the wall" if not self.inside(self.ahead(act)) else "into its own body"
        return pick, f"{act} goes {why}; took {pick}"

    def step(self, d):
        self.dir = d if d in self.legal() else self.dir
        t = self.ahead(self.dir)
        self.steps += 1
        if not self.inside(t):
            self.alive = False
            return "hit wall"
        if t == self.food:
            self.snake.insert(0, t)
            self.score += 1
            self.place_food()
            return "ate"
        if t in self.snake:
            self.alive = False
            return "hit body"
        self.snake.insert(0, t)
        self.snake.pop()
        return "move"

# ---------------------------------------------------------------- mines ----
class Mines:
    """5x5, 3 mines. Model picks the cell least likely to be a mine, numbers not shown —
    the criteria sentences carry the mine-count facts (text encoder weighs words)."""
    N, K = 5, 3

    def __init__(self, seed):
        self.rng = random.Random(seed)
        all_cells = [(x, y) for x in range(self.N) for y in range(self.N)]
        self.mines = set(self.rng.sample(all_cells, self.K))
        self.opened = set()
        self.alive, self.score, self.steps = True, 0, 0

    def count(self, c):
        x, y = c
        n = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (x + dx, y + dy) in self.mines:
                    n += 1
        return n

    def touch(self, c):
        x, y = c
        edge = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if not (0 <= x + dx < self.N and 0 <= y + dy < self.N):
                    edge += 1
        return edge

    def candidates(self):
        return [c for c in [(x, y) for x in range(self.N) for y in range(self.N)]
                if c not in self.opened]

    def legal(self):
        return self.candidates()

    def render(self):
        return (f"Mines board 5 by 5, {self.K} hidden mines. "
                f"{len(self.opened)} cells opened safely, {len(self.candidates())} closed.")

    def questions(self):
        criteria = {}
        for c in self.candidates()[:12]:
            n = self.count(c)
            t = (f"{n} mines around it" if n else "no mines around it")
            if self.touch(c) >= 3:
                t += ", on the safe edge"
            criteria[str(c)] = t
        return {"open": {"type": "choice", "instructions":
                    "Pick the closed cell least likely to hold a mine.",
                    "criteria": criteria}}

    def decide(self, answers):
        probs = (answers.get("open") or {}).get("probabilities", {})
        best, act = -1.0, None
        for k, p in probs.items():
            if p is not None and p > best:
                best, act = p, k
        if act is None:
            act = str(self.candidates()[0])
        return act, probs

    def step(self, cell_str):
        try:
            c = eval(str(cell_str)) if isinstance(cell_str, str) and "(" in str(cell_str) else cell_str
        except Exception:
            self.alive = False
            return "invalid cell"
        if not isinstance(c, tuple):
            self.alive = False
            return "invalid cell"
        if c in self.mines:
            self.alive = False
            return "boom"
        self.opened.add(c)
        self.score += 1
        self.steps += 1
        if len(self.opened) >= self.N * self.N - self.K:
            self.alive = False
            self.score += 10
            return "cleared"
        return "safe"

# ---------------------------------------------------------------- crossing ----
class Crossing:
    """Chicken-crossing: 8-lane road, cars move each tick. Model picks
    move/stay. Criteria sentences = which lanes will be occupied where you'd stand."""
    LANES, W = 8, 7

    def __init__(self, seed):
        self.rng = random.Random(seed)
        # lane -> list of car x positions (cars move right on even lanes, left odd)
        self.cars = {}
        for lane in range(self.LANES):
            speed = 1 + (lane % 2)
            cars = sorted(self.rng.sample(range(self.W), 2))
            self.cars[lane] = {"pos": cars, "speed": speed, "left": lane % 2 == 1}
        self.row = 0                      # 0 = start curb, LANES = safe curb
        self.col = self.W // 2
        self.alive, self.score, self.steps = True, 0, 0

    def car_at(self, lane, x):
        return x in self.cars[lane]["pos"]

    def advance_cars(self):
        for lane, c in self.cars.items():
            newpos = []
            for x in c["pos"]:
                if c["left"]:
                    newpos.append((x - c["speed"]) % self.W)
                else:
                    newpos.append((x + c["speed"]) % self.W)
            c["pos"] = newpos

    def render(self):
        return (f"Road with 8 lanes of traffic, crossing from bottom to top. "
                f"You stand on lane {self.row} of 8 at position {self.col} of 7. "
                f"Cars move horizontally, one lane each turn.")

    def facts(self, action):
        if action == "forward":
            nxt = self.row + 1
            if nxt > self.LANES:
                return "forward reaches the safe curb, you win"
            danger = [x for x in range(self.W) if self.car_at(nxt, x)]
            if not danger:
                return "forward: the next lane is clear of cars right now"
            near = min(abs(self.col - x) for x in danger)
            return f"forward: next lane has {len(danger)} car(s), closest is {near} step(s) from your position"
        if action == "back":
            if self.row <= 0:
                return "back is off the board, you stay put"
            return "back retreats to the previous lane, losing progress but usually safe"
        if action == "left":
            if self.col <= 0:
                return "left is the board edge, you stay"
            return "left shifts one column, same lane, dodging sideways"
        if action == "right":
            if self.col >= self.W - 1:
                return "right is the board edge, you stay"
            return "right shifts one column, dodging sideways"
        return "staying still while cars move"

    def legal(self):
        return ["forward", "back", "left", "right", "stay"]

    def questions(self):
        criteria = {a: self.facts(a) for a in self.legal()}
        # ground-truth noul: is forward immediately survivable?
        nxt = self.row + 1
        clear = nxt > self.LANES or not any(
            abs(self.col - x) <= 0 for x in range(self.W) if nxt <= self.LANES and self.car_at(nxt, x))
        return {"move": {"type": "choice", "instructions":
                    "Cross the road lane by lane without being hit by a car.",
                    "criteria": criteria},
                "lane_clear": {"type": "noul", "instructions":
                    "The lane directly ahead has no car at your position right now.",
                    "criteria": {"true": "no car would hit you if you step forward now",
                                 "false": "a car sits on the cell you would step into"}}}

    def decide(self, answers):
        probs = (answers.get("move") or {}).get("probabilities", {})
        best, act = -1.0, "stay"
        for k, p in probs.items():
            if p and p > best:
                best, act = p, k
        return act, probs

    def shield(self, act, probs):
        # Code owns survival: stepping into a car is death, veto like a floor.
        if act != "forward":
            return act, None
        nxt = self.row + 1
        if nxt > self.LANES:
            return act, None
        if self.car_at(nxt, self.col):
            alts = [a for a in ("left", "right", "stay") if a in self.legal()]
            pick = max(alts, key=lambda a: probs.get(a, 0) or 0)
            return pick, f"car on forward cell; took {pick}"
        return act, None

    def step(self, act):
        self.steps += 1
        if getattr(self, "stagnant", 0) > 60:
            self.alive = False
            return "ran out of time hiding on the curb"
        if act != "forward":
            self.stagnant = getattr(self, "stagnant", 0) + 1
        else:
            self.stagnant = 0
        ev = "waited"
        if act == "forward":
            if self.row + 1 > self.LANES:
                self.alive = False; self.score += 20; return "reached curb"
            if self.car_at(self.row + 1, self.col):
                self.alive = False; return "hit by car"
            self.row += 1; self.score += 1; ev = "crossed lane"
        elif act == "back" and self.row > 0:
            self.row -= 1; ev = "stepped back"
        elif act == "left" and self.col > 0:
            self.col -= 1; ev = "dodged left"
        elif act == "right" and self.col < self.W - 1:
            self.col += 1; ev = "dodged right"
        # Cars advance EVERY tick (like bakeer's games: world moves after each move).
        self.advance_cars()
        if self.row > 0 and self.car_at(self.row, self.col):
            self.alive = False
            return "car drove into you"
        return ev

# ---------------------------------------------------------------- hopper ----
class Hopper:
    """Doodle-jump style: model hops between platforms going up. Choice of
    jump left/right/stay; falling below screen ends the run."""
    W, H = 9, 12

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.x, self.y = 4, 2
        self.platforms = [(3, 4), (6, 6), (2, 7), (7, 9), (4, 10)]
        self.alive, self.score, self.steps = True, 0, 0

    def render(self):
        return (f"Hopper on a 9 wide by 12 tall wall. You sit at position {self.x}. "
                f"Platforms ahead at heights 4, 6, 7, 9, 10. "
                f"Jump reaches 3 tiles up and drifts 2 sideways.")

    def facts(self, a):
        drift = {"jump_left": -2, "jump_right": 2, "stay": 0}[a]
        nx = self.x + drift
        if not (0 <= nx < self.W):
            return "drifts off the wall edge, you fall"
        # nearest platform above
        best = None
        for px, py in self.platforms:
            if py > self.y and abs(px - nx) <= 2:
                if best is None or py < best[1]:
                    best = (px, py)
        if best is None:
            return "no platform within reach at that drift"
        return f"lands on the platform at position {best[0]}, height {best[1]}"

    def legal(self):
        return ["jump_left", "stay", "jump_right"]

    def questions(self):
        return {"move": {"type": "choice", "instructions":
                    "Pick the jump that lands on a platform higher up. Falling off ends the run.",
                    "criteria": {a: self.facts(a) for a in self.legal()}},
                "platform_near": {"type": "noul", "instructions":
                    "A platform sits within two tiles of your current position.",
                    "criteria": {"true": "a landing spot is close", "false": "nothing close"}}}

    def decide(self, answers):
        probs = (answers.get("move") or {}).get("probabilities", {})
        best, act = -1.0, "stay"
        for k, p in probs.items():
            if p and p > best:
                best, act = p, k
        return act, probs

    def shield(self, act, probs):
        # Floor: a jump off the wall is certain death; convert to stay.
        drift = {"jump_left": -2, "jump_right": 2, "stay": 0}.get(act, 0)
        if 0 <= self.x + drift < self.W:
            return act, None
        return "stay", f"{act} leaves the wall; took stay"

    def step(self, act):
        self.steps += 1
        drift = {"jump_left": -2, "jump_right": 2, "stay": 0}.get(act, 0)
        self.x = self.x + drift
        if not (0 <= self.x < self.W):
            self.alive = False
            return "fell off the wall"
        # gravity: drop to the highest platform below current height
        lands = [py for px, py in self.platforms if abs(px - self.x) <= 1]
        if lands:
            self.score += 1
            self.platforms = [(px, py - 2) for px, py in self.platforms if py > min(lands)][:5]
        else:
            self.score = max(0, self.score - 1)
        if self.score >= 15:
            self.alive = False
        return "hopped"

GAMES = {"snake": Snake, "mines": Mines, "crossing": Crossing, "hopper": Hopper}

# ---------------------------------------------------------------- runner ----
def play(game_cls, name, server, episodes, log, full=False, epsilon=0.0, seed0=13):
    scores, interventions, lat = [], 0, []
    for ep in range(episodes):
        g = game_cls(seed=seed0 + ep * 7919)
        while g.alive and g.steps < 250:
            t0 = time.time()
            answers, questions, state_text = {}, {}, ""
            if server:
                try:
                    state_text = g.render()
                    questions = g.questions()
                    resp = ask_server(server, state_text, questions)
                    answers = resp.get("answers", resp)
                    lat.append(resp.get("latency_ms", (time.time() - t0) * 1000))
                except Exception:
                    lat.append((time.time() - t0) * 1000)
            act, probs = g.decide(answers)
            if epsilon and len(g.legal()) > 1:
                import random as _r
                if _r.random() < epsilon:
                    act = _r.choice(g.legal())
            if name == "snake":
                act2, why = g.shield(act, probs)
                if why:
                    interventions += 1
                act = act2
            ev = g.step(act)
            rec = {"game": name, "ep": ep, "step": g.steps,
                   "action": str(act), "event": ev, "score": g.score}
            if full:
                rec.update({"state": state_text, "questions": questions,
                            "answers": answers})
            log.write(json.dumps(rec) + "\n")
        scores.append(g.score)
    lat_sorted = sorted(lat)
    return {"game": name, "episodes": episodes, "mean": round(sum(scores) / len(scores), 2),
            "max": max(scores), "shield_interventions": interventions,
            "latency_ms_p50": round(lat_sorted[len(lat_sorted) // 2], 1) if lat_sorted else None}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="snake,mines")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--server", default="http://192.168.2.185:8710")
    ap.add_argument("--offline", action="store_true",
                    help="no server: model probabilities absent, shield+fallback plays")
    ap.add_argument("--out", default="/tmp/arcadia_results.json")
    ap.add_argument("--epsilon", type=float, default=0.0,
                    help="exploration: take a random legal action this often")
    ap.add_argument("--full", action="store_true",
                    help="log state/questions/answers per move (training harvest)")
    args = ap.parse_args()
    log = open("/tmp/arcadia_moves.jsonl", "w")
    results = []
    for name in args.games.split(","):
        name = name.strip()
        if name not in GAMES:
            continue
        r = play(GAMES[name], name, None if args.offline else args.server, args.episodes, log, full=args.full, epsilon=args.epsilon)
        results.append(r)
        print(r, flush=True)
    log.close()
    json.dump(results, open(args.out, "w"), indent=1)
    print("saved", args.out)