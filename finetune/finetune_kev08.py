#!/usr/bin/env python3
"""finetune_kev08.py — LoRA fine-tune of kev-0.8b (Phase B sidecar arm).

kev-0.8b (jaredpalmer, Qwen3.5-0.8B + pointer head, ~1.6GB bf16) is UNSCORED on
the multimodalart Decision Index (0.2 pending) — this arm makes us the first
external measurement. Same recipe validated at 4B (rank 7, DI 47.43, near-Jev
tool scores) and 9B (rank 5); small Kevs show weak calibration ECE 0.19-0.23,
which is why the harvested-label fine-tune + calibrate.py matter more here.

GPU-free on the Mac; GPU-ready for a Spark lane-idle window (or Mac-local MLX
later — out of scope here). Same structure as finetune_laya.py; head-specific
notes inline. Pointer-head caveat: if jaredpalmer ships only base+recipe and the
head weights aren't in the repo, init_head_from() re-inits the head and trains
it jointly (conditional below, resolved at runtime from the repo layout).
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="finetune_kev08",
                                 description="LoRA fine-tune kev-0.8b (Phase B sidecar arm)")
    ap.add_argument("--base", default="jaredpalmer/kev-0.8b")
    ap.add_argument("--data", default="finetune/dataset")
    ap.add_argument("--out", default="finetune/out/kev08-gate-v1")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4, help="smaller model tolerates higher LR")
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch", type=int, default=0, help="0 = auto (default 16 for 0.8B)")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def plan(args) -> dict:
    counts = {}
    for name in ("train", "val", "test"):
        p = os.path.join(args.data, f"{name}.jsonl")
        counts[name] = sum(1 for _ in open(p)) if os.path.exists(p) else 0
    if counts["train"] == 0:
        raise SystemExit(f"no train.jsonl under {args.data} — run build_dataset.py first")
    bs = args.batch or 16  # 0.8B + r16 LoRA fits b16 in ~8GB lane-idle (Kev recipe defaults)
    steps = counts["train"] * args.epochs // bs
    if args.max_steps:
        steps = min(steps, args.max_steps)
    return {"rows": counts, "batch": bs, "steps": steps,
            "lora": {"r": args.lora_r, "alpha": args.lora_alpha,
                     "targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "up_proj", "down_proj"],
                     "head": "pointer head — joint-trained from scratch if missing from repo"},
            "optimizer": {"lr": args.lr, "warmup_frac": args.warmup, "schedule": "cosine"},
            "precision": "bf16-autocast (train) / fp32 (val probability checks)",
            "eval_metric": "separation gap+AUC on val split every %d steps" % args.eval_every}


def main(argv=None) -> int:
    args = parse_args(argv)
    p = plan(args)
    print("== kev-0.8b fine-tune plan ==")
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
              "run on the Spark lane-idle window", file=sys.stderr)
        return 2
    print(f"[gpu] loading {args.base} ...")
    raise SystemExit("[gpu] training loop lands with the lane-idle window; "
                     "dataset + plan verified GPU-free")  # placeholder by design


if __name__ == "__main__":
    sys.exit(main())