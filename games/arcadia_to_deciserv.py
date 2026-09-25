#!/usr/bin/env python3
"""arcadia_to_deciserv.py — convert games harvest move-logs into CLM-style
reward-tagged typed-decisions rows (schema per Contrastive-LM/CLM
train/adapters.py read_transitions: state, action, task_id, step_idx,
trajectory_id, reward) + DeciServ typed-decisions supervision targets.

Reward shaping (potential-based, per CLM subagent rec):
  snake:   +1 food eaten, +0.1 food-distance-closing, -1 death (wall/body)
  mines:   +0.2 safe open, +2 board cleared, -1 boom
  crossing:+0.25 lane gained, +2 curb reached, -1 hit by car
  hopper:  +0.15 platform gained, -0.05 score decay
Reward-gated: keep rows with reward > 0 (positive steps only) OR
shield-intervened rows tagged reward=0 as SAFETY demos (never-negative teacher).

Usage:
  python3 arcadia_to_deciserv.py --moves /tmp/arcadia_harvest.jsonl \
      --out /Users/clawdio/deciserv-data/corpus/train_arcadia.jsonl
"""
import argparse, json, os
from collections import Counter

def reward_for(game, event, prev_score, score, action):
    """Potential-based shaping per event."""
    if game == "snake":
        if event == "ate": return 1.0
        if event in ("hit wall", "hit body"): return -1.0
        return 0.05 if event == "move" else 0.0
    if game == "mines":
        if event == "cleared": return 2.0
        if event == "boom": return -1.0
        if event == "safe": return 0.2
        return 0.0
    if game == "crossing":
        if event == "reached curb": return 2.0
        if event in ("hit by car", "car drove into you"): return -1.0
        if event == "crossed lane": return 0.25
        if event.startswith("ran out of time"): return -0.2
        return 0.0
    if game == "hopper":
        if event == "hopped" and score > prev_score: return 0.15
        if event == "hopped" and score < prev_score: return -0.05
        return 0.0
    return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", default="/tmp/arcadia_harvest.jsonl")
    ap.add_argument("--out", default="/Users/clawdio/deciserv-data/corpus/train_arcadia.jsonl")
    ap.add_argument("--min-reward", type=float, default=0.05)
    args = ap.parse_args()

    # group moves by (game, episode) to get trajectory ids + score context
    traj = {}
    order = []
    with open(args.moves) as f:
        for line in f:
            r = json.loads(line)
            key = (r["game"], r["ep"])
            if key not in traj:
                traj[key] = []
                order.append(key)
            traj[key].append(r)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_in, n_out = 0, 0
    rewards = Counter()
    per_game = Counter()
    with open(args.out, "w") as out:
        for key in order:
            game, ep = key
            moves = traj[key]
            for i, m in enumerate(moves):
                n_in += 1
                prev_score = moves[i-1]["score"] if i > 0 else 0
                # reward from the EVENT OF THIS MOVE
                rw = reward_for(game, m["event"], prev_score, m["score"], m["action"])
                # credit final score at end-of-trajectory for terminal shaping
                if i == len(moves) - 1 and game in ("snake", "crossing"):
                    rw += 0.05 * m["score"]
                if rw > args.min_reward or m["event"].startswith("shield"):
                    # typed-decisions row: the model's logged state text is not
                    # stored per-move (only action/event), so we train on the
                    # (state, action) pairs harvested WITH state text — require
                    # the harvest to include 'state' (play() logs it when given)
                    row = {
                        "task_id": f"arcadia:{game}",
                        "trajectory_id": f"{game}-ep{ep}",
                        "step_idx": m["step"],
                        "action": m["action"],
                        "event": m["event"],
                        "reward": round(rw, 3),
                    }
                    if "state" in m:
                        row["state"] = m["state"]
                        row["questions"] = m.get("questions")
                        row["answers"] = m.get("answers")
                    out.write(json.dumps(row) + "\n")
                    n_out += 1
                    per_game[game] += 1
                    rewards[("pos" if rw > 0 else "zero/neg")] += 1
    print(json.dumps({
        "moves_in": n_in, "rows_out": n_out,
        "per_game": dict(per_game),
        "reward_sign": dict(rewards),
        "out": args.out,
    }, indent=1))

if __name__ == "__main__":
    main()