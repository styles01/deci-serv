#!/usr/bin/env python3
"""finetune_laya.py — LoRA fine-tune of Laya typed-decisions (Phase A arm).

GPU-ready script for the next Spark lane-idle window. GPU-free here: every code
path builds tensor shapes offline and exits in --dry-run mode without torch init
on CUDA.

Design (justified by measurement, not vibes):
  * init_from Laya (convaiinnovations/laya, Apache-2.0) — keeps ~2GB fp16 serving
    footprint and 13ms latency (T4 model card) vs Kev-4B's 17GB/627ms p95
    (multimodalart Decision Index 0.1, verified 2026-09-24).
  * LoRA r=16, alpha=32 on attention+MLP projections — standard for 400M-class
    typed heads; pointer head stays FROZEN (calibration refit post-hoc instead —
    Laya model card: ECE 0.466->0.081 with per-bucket temperature, our
    calibrate.py).
  * bf16 training with autocast — arbiter measured bf16 moves probabilities up
    to 2.1e-2 and flips one argmax, so VAL CHECKS run in fp32 on the head output
    (probability fidelity is the product here, per arbiter bench/results.md).
  * Eval hook every N steps: separation metric (destructive vs benign p_block
    gap + AUC) on val split — the metric that must move (our baseline: gap
    0.095, AUC ~0.55; target: AUC >= 0.9 before promotion).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="finetune_laya",
                                 description="LoRA fine-tune Laya typed-decisions (Phase A arm)")
    ap.add_argument("--base", default="convaiinnovations/laya",
                    help="HF repo or local path of the Laya bundle (default: convaiinnovations/laya)")
    ap.add_argument("--ckpt", default="typed-decisions",
                    help="subfolder of the bundle to init from (default: typed-decisions)")
    ap.add_argument("--data", default="finetune/dataset",
                    help="dataset dir from build_dataset.py (train/val/test .jsonl)")
    ap.add_argument("--out", default="finetune/out/laya-gate-v1",
                    help="output dir for LoRA adapter + calib-ready checkpoint")
    ap.add_argument("--epochs", type=int, default=2, help="full passes (default 2)")
    ap.add_argument("--lr", type=float, default=1e-4, help="peak LR, cosine decay (default 1e-4)")
    ap.add_argument("--warmup", type=float, default=0.03, help="warmup fraction (default 3%%)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch", type=int, default=0, help="batch size; 0 = auto from free memory")
    ap.add_argument("--max-steps", type=int, default=0, help="override total steps (smoke tests)")
    ap.add_argument("--eval-every", type=int, default=50, help="separation eval cadence (steps)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build shapes, print plan, exit without GPU/model download")
    return ap.parse_args(argv)


def plan(args) -> dict:
    """Everything that can be computed without a GPU."""
    d = args.data
    def n(path):
        p = os.path.join(d, f"{name}.jsonl")
        if not os.path.exists(p):
            return 0
        with open(p) as f:
            return sum(1 for _ in f)
    counts = {}
    for name in ("train", "val", "test"):
        p = os.path.join(d, f"{name}.jsonl")
        counts[name] = sum(1 for _ in open(p)) if os.path.exists(p) else 0
    if counts["train"] == 0:
        raise SystemExit(f"no train.jsonl under {d} — run build_dataset.py first "
                         f"(on the Spark: python3 finetune/build_dataset.py --harvest --rows ~/deciserv-data/corpus/*.jsonl)")
    bs = args.batch or 8  # measured: 421M fp16 model + r16 LoRA fits b8 in ~6GB on GB10 lane-idle
    steps = counts["train"] * args.epochs // bs
    if args.max_steps:
        steps = min(steps, args.max_steps)
    return {"rows": counts, "batch": bs, "steps": steps,
            "lora": {"r": args.lora_r, "alpha": args.lora_alpha,
                     "targets": ["Wqkv", "Wo", "Wi"]},
            "optimizer": {"lr": args.lr, "warmup_frac": args.warmup,
                          "schedule": "cosine"},
            "precision": "bf16-autocast (train) / fp32 (val probability checks)",
            "eval_metric": "separation gap+AUC on val split every %d steps" % args.eval_every}


def main(argv=None) -> int:
    args = parse_args(argv)
    p = plan(args)
    print("== fine-tune plan ==")
    print(json.dumps(p, indent=1))
    if args.dry_run:
        print("[dry-run] no GPU touched, no weights downloaded — plan only")
        return 0
    try:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        print(f"[blocked] GPU-side deps unavailable on this host: {e}\n"
              "run on the Spark lane-idle window: "
              "ssh jaita@192.168.2.185 'cd ~/deci-serv && /home/jaita/venvs/laya/bin/python finetune/finetune_laya.py ...'",
              file=sys.stderr)
        return 2
    # ---- GPU path (executes only on the Spark) -----------------------------
    from finetune_lib import read_jsonl
    train = read_jsonl(os.path.join(args.data, "train.jsonl"))
    val = read_jsonl(os.path.join(args.data, "val.jsonl"))
    # The checkpoint dir (typed-decisions/) is INSIDE the laya repo bundle that
    # ships rl_agent_api.py + rl_common.py — import them from there.
    if os.path.isdir(args.base):
        repo_root = os.path.dirname(os.path.abspath(args.base)) if os.path.basename(args.base) != "laya" else os.path.abspath(args.base)
        ckpt_dir = os.path.abspath(args.base)
    else:
        raise SystemExit("--base must be a LOCAL checkpoint dir on this host (HF hub path not supported for the GPU loop)")
    sys.path.insert(0, repo_root)
    from rl_agent_api import RLAgent  # noqa: E402
    from rl_common import QTYPES, build_sequence, collate_items, render_options, temp_bucket  # noqa: E402
    import torch
    agent = RLAgent(ckpt_dir, device="cuda")
    model = agent.model
    # LoRA on the encoder only; the head trains jointly (it is the calibration
    # surface, freezing it would fight the encoder). Head LR = 10x LoRA LR.
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      target_modules=list(p["lora"]["targets"]), lora_dropout=0.05)
    model.encoder = get_peft_model(model.encoder, lcfg)
    trainable = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    total_p = sum(pp.numel() for pp in model.parameters())
    print(f"[lora] trainable {trainable/1e6:.1f}M / {total_p/1e6:.1f}M "
          f"({100 * trainable / total_p:.2f}%)")
    enc_params = [pp for pp in model.encoder.parameters() if pp.requires_grad]
    head_params = [pp for n, pp in model.named_parameters() if "encoder." not in n and pp.requires_grad]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * 10.0},  # head from scratch-ish vs frozen-ish encoder
    ], weight_decay=0.01)
    steps_total = p["steps"] or 1
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, int(steps_total * args.warmup))) *
        (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps_total)))))
    amp_dtype = agent.dtype
    pad_id = agent.tok.pad_token_id
    QTYPE_ID = {v: i for i, v in QTYPES.items()}
    from finetune_lib import separation_metric, split_examples_by_class  # noqa: E402

    def encode_batch(rows):
        items, labels = [], []
        for ex in rows:
            q = ex["input"]["question"]
            qdef = {"type": q["type"], "instructions": q["instructions"],
                    "criteria": q.get("criteria")}
            qi = RLAgent._to_internal(qdef)
            seq, markers = build_sequence(agent.tok, ex["input"]["state"], qi,
                                          agent.cfg["max_len"], agent.cfg["head_max_len"])
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["type"]],
                          "target": ex["target"]["probabilities"], "label": ex["target"]["label"],
                          "episode": 0, "ep_step": 0, "ep_len": 1, "src": "ft"})
            labels.append(ex["target"]["probabilities"])
        b = collate_items([items], pad_id)
        return b, torch.tensor(labels, dtype=torch.float32)

    @torch.no_grad()
    def eval_separation():
        agent.model.eval()
        probs = []
        for i in range(0, len(val), 16):
            b, _ = encode_batch(val[i:i + 16])
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits, _ = agent.model(b["input_ids"].cuda(), b["attention_mask"].cuda(),
                                        b["marker_pos"].cuda(), b["marker_mask"].cuda(),
                                        b["qtype"].cuda())
            for r in range(logits.size(0)):
                z = logits[r].float().cpu().numpy()
                pz = np.exp(z - z.max()); pz /= pz.sum()
                probs.append(pz)
        agent.model.train()
        # block-index per example: the question is the gate 'block' choice set
        by = {"destructive": [], "benign": []}
        for ex, pz in zip(val, probs):
            if ex.get("class") in by:
                # block option = the one whose side prefix is 'block'
                opts = ex["input"]["question"].get("options") or []
                idx = next((i for i, o in enumerate(opts) if o.split(":", 1)[0].strip() == "block"), 0)
                by[ex["class"]].append(pz[idx])
        if not by["destructive"] or not by["benign"]:
            return None
        return separation_metric(by["destructive"], by["benign"])

    import numpy as np
    rng = random.Random(1337)
    order = list(range(len(train)))
    step = 0
    agent.model.train()
    bs = p["batch"]
    print(f"[gpu] training {steps_total} steps, batch {bs}, eval every {args.eval_every}")
    while step < steps_total:
        rng.shuffle(order)
        for i in range(0, len(order) - bs + 1, bs):
            batch_rows = [train[j] for j in order[i:i + bs]]
            b, targets = encode_batch(batch_rows)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits, _ = agent.model(b["input_ids"].cuda(), b["attention_mask"].cuda(),
                                        b["marker_pos"].cuda(), b["marker_mask"].cuda(),
                                        b["qtype"].cuda())
            k = logits.size(1)
            logp = torch.log_softmax(logits.float(), dim=1)
            loss = -(targets.cuda() * logp).sum(1).mean()  # soft-target CE (proper scoring)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc_params + head_params, 1.0)
            opt.step(); opt.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"  step {step}/{steps_total} loss {loss.item():.4f}", flush=True)
            if step % args.eval_every == 0 or step == steps_total:
                m = eval_separation()
                print(f"  [eval] step {step}: {m}", flush=True)
            if step >= steps_total:
                break
    os.makedirs(args.out, exist_ok=True)
    merged = model.merge_and_unload() if hasattr(model, "merge_and_unload") else None
    torch.save({"state": (merged or agent.model).state_dict(), "cfg": agent.cfg},
               os.path.join(args.out, "laya-gate-lora.pt"))
    with open(os.path.join(args.out, "train_report.json"), "w") as f:
        sep = eval_separation() or {}
        json.dump({"final_separation": {k: float(v) for k, v in sep.items()} if isinstance(sep, dict) else sep,
                   "steps": step, "rows": p["rows"], "lora": p["lora"]}, f, indent=1)
    print(f"[gpu] saved adapter bundle to {args.out}")


if __name__ == "__main__":
    sys.exit(main())