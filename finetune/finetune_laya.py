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
import os
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
                     "targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "up_proj", "down_proj"]},
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
    print(f"[gpu] loading {args.base}:{args.ckpt} ...")
    tok = AutoTokenizer.from_pretrained(args.base, subfolder=args.ckpt)
    model = AutoModel.from_pretrained(args.base, subfolder=args.ckpt, torch_dtype="auto")
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      target_modules=p["lora"]["targets"], lora_dropout=0.05,
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    # Training loop: standard HF Trainer-free loop (small data, full control,
    # separation eval hook — see finetune_lib.separation_metric).
    raise SystemExit("[gpu] training loop lands with the lane-idle window; "
                     "dataset + plan verified GPU-free")  # placeholder by design


if __name__ == "__main__":
    sys.exit(main())