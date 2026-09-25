#!/usr/bin/env python3
"""arcadia_extra.py — paddle + dungeon for the DeciServ game harness.

Ports of 0xBakeer's arbiter showcase games (MIT, github.com/0xBakeer/arbiter)
to the laya_arcadia.py contract: same class shape as Snake/Mines/Crossing/
Hopper (init/render/questions/decide/shield/step), posting to OUR gate
(POST /decide {"state", "questions"}), driven by play().

Paddle — the speed game. 20x14 box, 60ms clock in the original; here one
/three-way choice per tick: left/stay/right, criteria = "the paddle ends N
columns from the ball". Shield keeps the paddle on the board.

Dungeon — the prose game. Rooms are generated paragraphs (same generator
tables as arbiter's: places, monsters, treasures), three to five actions,
plus a danger score question. The shield vetoes a fight/loot/heal that one
hit could end (when any safe door exists).
"""
from __future__ import annotations

import random

# ---------------------------------------------------------------- paddle ----
class Paddle:
    """Breakout paddle. One three-way choice per tick, shortest state in the
    showcase: 'A paddle must be moved under a falling ball.' Everything the
    decision needs lives in the criteria (measured on his 20-sample set —
    longer states made his model play worse; we keep the finding)."""
    W, H = 20, 14
    HALF = 1  # paddle covers px-1 .. px+1

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.px = self.W // 2
        self.bx = 2 + int(self.rng.random() * (self.W - 4))
        self.by = 2
        self.vx = -1 if self.rng.random() < 0.5 else 1
        self.vy = 1
        self.score, self.steps = 0, 0
        self.alive = True

    def offset(self):
        return self.bx - self.px

    def legal(self):
        out = ["left", "stay", "right"]
        return [m for m in out
                if not (m == "left" and self.px - 1 < self.HALF)
                and not (m == "right" and self.px + 1 > self.W - 1 - self.HALF)]

    def render(self):
        return "A paddle must be moved under a falling ball."

    def facts(self, move):
        shift = {"left": -1, "right": 1}.get(move, 0)
        now = abs(self.offset())
        after = abs(self.offset() - shift)
        verdict = ("closer to the ball" if after < now
                   else "further from the ball" if after > now else "no change")
        return f"{verdict}: the paddle ends {after} columns from the ball"

    def questions(self):
        return {"move": {"type": "choice",
                         "instructions": "Move the paddle so that it is under the ball when the ball comes down.",
                         "criteria": {m: self.facts(m) for m in self.legal()}}}

    def decide(self, answers):
        probs = (answers.get("move") or {}).get("probabilities", {}) or {}
        best, act = -1.0, self.legal()[0]
        for m in self.legal():
            p = probs.get(m, 0) or 0
            if p > best:
                best, act = p, m
        return act, probs

    def shield(self, act, probs):
        if act in self.legal():
            return act, None
        return "stay", f"{act} would push the paddle off the board; took stay"

    def step(self, act):
        if act == "left":
            self.px = max(self.HALF, self.px - 1)
        if act == "right":
            self.px = min(self.W - 1 - self.HALF, self.px + 1)
        self.steps += 1
        self.bx += self.vx
        self.by += self.vy
        if self.bx <= 0:
            self.bx, self.vx = 0, 1
        if self.bx >= self.W - 1:
            self.bx, self.vx = self.W - 1, -1
        if self.by <= 0:
            self.by, self.vy = 0, 1
        if self.by >= self.H - 1:
            if abs(self.bx - self.px) <= self.HALF:
                self.by = self.H - 1
                self.vy = -1
                if self.bx != self.px:
                    self.vx = 1 if self.bx > self.px else -1
                self.score += 1
                return "returned the ball"
            self.alive = False
            return "missed the ball"
        return "rally"


# --------------------------------------------------------------- dungeon ----
MAX_HP = 20
ROOMS = 30
POTION_HEAL = 8

PLACES = [
    "a flooded undercroft, knee deep in black water",
    "a collapsed library, shelves fallen against each other",
    "a hall of broken statues facing the wrong way",
    "a narrow gallery above a drop with no bottom in sight",
    "a kitchen gone cold centuries ago, pots still hanging",
    "a chapel with the altar hacked apart",
    "a stable where something much larger than a horse was kept",
    "a corridor of doors, all of them nailed shut but one",
]
MONSTERS = [
    {"name": "a rat the size of a dog", "strength": 1},
    {"name": "a ghoul with a broken spear", "strength": 2},
    {"name": "a rusted iron sentry", "strength": 3},
    {"name": "a cave troll, half asleep", "strength": 4},
    {"name": "a wight in a mouldering crown", "strength": 5},
]
TREASURES = [
    "a chest sits half submerged against the wall",
    "coins are scattered where someone dropped a purse",
    "a strongbox has been pried open and abandoned",
    "a corpse still wears a heavy gold chain",
]


class Dungeon:
    """Rogue-lite in prose: the room is a paragraph, the options are sentences
    about that paragraph. Model picks the action + scores the danger; the
    shield vetoes a fight/loot/heal that one hit could end when any safe door
    exists. Same generator tables as arbiter's dungeon (MIT)."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.hp = MAX_HP
        self.max_hp = MAX_HP
        self.gold = 0
        self.potions = 2
        self.depth = 1
        self.score = 0
        self.alive = True
        self.escaped = False
        self.steps = 0
        self.room = self._make_room()

    # -- generator ----------------------------------------------------------
    def _make_room(self):
        place = self.rng.choice(PLACES)
        monster = None
        if self.rng.random() < 0.7:
            floor_i = min(len(MONSTERS) - 1, (self.depth - 1) // 8)
            pick = self.rng.randrange(floor_i, len(MONSTERS))
            base = MONSTERS[pick]
            monster = {"name": base["name"], "strength": base["strength"],
                       "hp": 2 + base["strength"] * 2,
                       "max_hit": base["strength"] + 2,
                       "gold": base["strength"] * 6 + self.rng.randrange(10)}
        treasure = None
        if self.rng.random() < 0.55:
            treasure = {"text": self.rng.choice(TREASURES),
                        "gold": 8 + self.rng.randrange(30),
                        "potion": self.rng.random() < 0.3}
        return {"place": place, "monster": monster, "treasure": treasure, "looted": False}

    def room_text(self):
        r = self.room
        parts = [f"Room {self.depth} of {ROOMS}: {r['place']}."]
        if r["monster"]:
            m = r["monster"]["name"]
            parts.append(m[0].upper() + m[1:] + " blocks the way.")
        if r["treasure"] and not r["looted"]:
            t = r["treasure"]["text"]
            parts.append(t[0].upper() + t[1:] + ".")
        if not r["monster"]:
            parts.append("A stair leads further down.")
        return " ".join(parts)

    # -- contract -----------------------------------------------------------
    def legal(self):
        r = self.room
        out = []
        if r["monster"]:
            out += ["fight", "flee"]
        if r["treasure"] and not r["looted"]:
            out.append("loot")
        if self.potions > 0 and self.hp < self.max_hp:
            out.append("heal")
        if not r["monster"]:
            out.append("descend")
        return out

    def render(self):
        return (f"{self.room_text()} You have {self.hp} of {self.max_hp} hit points, "
                f"{self.potions} potions and {self.gold} gold.")

    def facts(self, action):
        r = self.room
        m = r["monster"]
        if action == "fight":
            verdict = ("deadly" if m["max_hit"] >= self.hp
                       else "risky" if m["max_hit"] * 2 >= self.hp else "safe")
            return (f"{verdict}: it hits for up to {m['max_hit']} and you have {self.hp} "
                    f"hit points, it needs about {-(-m['hp'] // 3)} more rounds to go down")
        if action == "flee":
            return (f"safe: you run for the stair and leave the room behind, it may land "
                    f"one parting blow of up to {m['max_hit']}")
        if action == "loot":
            if m:
                return (f"risky: you reach for the treasure with {m['name']} still standing, "
                        f"it hits for up to {m['max_hit']}")
            return "safe: nothing is watching, the treasure is yours"
        if action == "heal":
            danger = (f", but {m['name']} hits for up to {m['max_hit']} while you drink"
                      if m else "")
            return (f"{'risky' if m else 'safe'}: a potion puts back {POTION_HEAL} hit points "
                    f"and you have {self.potions}{danger}")
        return f"safe: the stair goes down to room {self.depth + 1}, you leave with {self.hp} hit points"

    def questions(self):
        options = self.legal()
        out = {}
        if len(options) >= 2:
            out["move"] = {"type": "choice",
                           "instructions": ("You are working your way down thirty rooms. Pick the "
                                            "action that gets deeper and richer without getting killed."),
                           "criteria": {a: self.facts(a) for a in options}}
        out["danger"] = {"type": "score",
                         "instructions": "How dangerous is this room for you right now?",
                         "criteria": ["nothing here can hurt you badly",
                                      "a fight would cost real blood",
                                      "one more hit would kill you"]}
        return out

    def decide(self, answers):
        options = self.legal()
        probs = (answers.get("move") or {}).get("probabilities", {}) or {}
        if not answers.get("move"):
            # forced action (single-option room); danger still comes back
            self.last_danger = self._danger(answers)
            return options[0], probs
        best, act = -1.0, options[0]
        for o in options:
            p = probs.get(o, 0) or 0
            if p > best:
                best, act = p, o
        self.last_danger = self._danger(answers)
        return act, probs

    def _danger(self, answers):
        d = answers.get("danger") or {}
        return d.get("score")

    def shield(self, act, probs):
        m = self.room["monster"]
        risky = act in ("fight", "loot", "heal") and m and m["max_hit"] >= self.hp
        if not risky:
            return act, None
        safe = [a for a in self.legal()
                if a == "flee" or (a == "heal" and m["max_hit"] < self.hp)]
        if not safe:
            return act, None  # nothing safe left — let it ride
        ranked = sorted(safe, key=lambda a: (probs.get(a, 0) or 0), reverse=True)
        return ranked[0], (f"{m['name']} hits for up to {m['max_hit']} and you have "
                           f"{self.hp}; took {ranked[0]}")

    def step(self, act):
        chosen = act if act in self.legal() else self.legal()[0]
        self.steps += 1
        m = self.room["monster"]
        ev = []
        if chosen == "fight":
            dmg = 2 + self.rng.randrange(3)
            m["hp"] -= dmg
            ev.append(f"you hit {m['name']} for {dmg}")
            if m["hp"] <= 0:
                self.gold += m["gold"]
                ev.append(f"{m['name']} falls, {m['gold']} gold")
                self.room["monster"] = None
            else:
                ev += self._hurt(m)
        elif chosen == "flee":
            if self.rng.random() < 0.4:
                ev += self._hurt(m, "as you turn")
            if self.alive:
                ev.append("you slip past into the next room")
                self._descend()
        elif chosen == "loot":
            t = self.room["treasure"]
            self.gold += t["gold"]
            if t["potion"]:
                self.potions += 1
            self.room["looted"] = True
            ev.append(f"{t['gold']} gold" + (" and a potion" if t["potion"] else ""))
            if m:
                ev += self._hurt(m)
        elif chosen == "heal":
            self.potions -= 1
            self.hp = min(self.max_hp, self.hp + POTION_HEAL)
            ev.append(f"the potion puts you back to {self.hp}")
            if m:
                ev += self._hurt(m)
        else:
            self._descend()
            ev.append(f"down to room {self.depth}")
        return "; ".join(ev)

    def _hurt(self, m, when=""):
        dmg = 1 + self.rng.randrange(m["max_hit"])
        self.hp -= dmg
        out = [f"{m['name']} hits you for {dmg}" + (f" {when}" if when else "")]
        if self.hp <= 0:
            self.hp = 0
            self.alive = False
            out.append("you die here")
        return out

    def _descend(self):
        if self.depth >= ROOMS:
            self.escaped = True
            self.alive = False
            self.score = self.gold + 100
            return
        self.depth += 1
        self.score = self.gold + self.depth * 2
        self.room = self._make_room()


GAMES_EXTRA = {"paddle": Paddle, "dungeon": Dungeon}