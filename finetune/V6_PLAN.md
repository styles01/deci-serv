# V6 TRAINING PLAN — decider-2b + our corpus (state after the 9/25 OOM incident)

*Write-date: 2026-09-25 ~21:00 ET. Read this file before doing ANYTHING with v6. This is the single source of truth if context gets compacted.*

## What v6 is
Fine-tune Mapika's **decider-2b** (Apache-2.0, Qwen3.5-2B-Base hybrid linear-attn, 2.3B) on our DeciServ corpus using **Mapika's own trainer** (vendored from github.com/Mapika/decider). Goal: v6 = our battery-tuned gate brain on the native `/v1/systemone` contract, replacing laya-v4 as primary when it passes gates.

## Incident (do not repeat)
- 2026-09-25 20:17: v6 training launch OOM-killed **the EXL3 daily driver** (Qwen3.8-Flash-Next EXL3 via vcruz305 exllamav3 fork `examples/chat.py`, pid 3450999, 36.5 GB anon-rss, serving :8000 for Loca). NOT Ollama — Ollama on Spark is dead weight since April (enabled service, localhost-only, stale model tags; ignore it entirely).
- Root cause: training allocs + resident EXL3 (78.6 GB) crossed pool limit mid-CUDA-graph-capture; kernel killed biggest anon-RSS proc.
- **HARD RULE: before ANY Spark alloc job (training/load), verify what's resident with `ps aux --sort=-rss`, NOT tag lists. If EXL3 daily driver is up (78.6 GB), only run training capped ≤14 GB, or wait for an idle window.**
- **Relaunch order: EXL3 daily driver FIRST (recipe argv below), verify :8000, THEN training.**

## Coexistence budget (MEASURED 9/26, post-restore)
- EXL3 container (vllm-fn-tp1, GMU 0.80, weights 72.8 GB, KV needs 4.34 GiB for 262k ctx) boots fine with **laya gate :8710 (1.99 GB) co-resident**, but FAILS the KV check when **decider lane :8712 (5.58 GB)** is also resident (KV available drops to 3.53 GiB → "increase GMU or decrease max_model_len").
- **Standing config: EXL3 + laya gate :8710 + router :8711 = always-on trio. Decider lane :8712 and v6 training (12-16 GB) are NOT co-resident-safe with EXL3 at GMU 0.80 — start them AFTER EXL3 has claimed its budget AND only if pool math still works; otherwise use the EXL3-idle window.**
- v6 training launch procedure: kill decider lane first (free 5.6 GB), confirm EXL3 /health still OK, launch training capped, restore decider lane only if pool math allows.

## Restore EXL3 daily driver (DONE 9/26 — reference for next time)
**Verified restored 9/26:** canonical `~/switch-to-qwen-flash.sh`, cold load ~9 min, "SPARK-ALIVE" on-box + "ALIVE" from Mac via larryspark.local:8000. NOTE: the recipe's `pre_load` script `drop-model-cache.sh` does not exist; the switch script is the real launcher (docker vllm-fn-tp1, GMU 0.80, PLE offload). See coexistence budget above.
```
# CORRECT launcher (the code block below is the OLD pre-9/25 chat.py path — kept for reference only):
ssh jaita@192.168.2.185 'bash ~/switch-to-qwen-flash.sh'
# wait for /health: docker vllm-fn-tp1, cold ~9-13 min; poll: curl http://127.0.0.1:8000/health
# verify Loca path from Mac: curl http://larryspark.local:8000/v1/models
# stale note kept for reference: recipe qwen38-flash-next-exl3-native.yaml describes the vcruz305
# exllamav3 chat.py variant (exl3-150 venv) — NOT what is currently serving :8000. Do not guess.
```

## Assets (all verified)
- **Model:** Spark `~/models/hf/Mapika/decider-2b` (bf16, 3.8 GB, v11, decider_config: temps choice 1.164 / noul 1.624 / score 1.124, schema_first_trained false, layout plain, max_options 255)
- **Trainer:** Spark `~/deci-serv/finetune/v6/decider/` (full repo clone of Mapika/decider; has train.py, model.py, prompt.py, data/, evaluate.py, calibrate.py)
- **Corpus:** Spark `~/deci-serv/finetune/v6/tasks_v6.pkl` — 64,661 train / 7,611 val (val_v5) / 3,811 test. Converted by `finetune/v6/build_tasks_v6.py` from `~/deciserv-data/corpus/train_v5.jsonl` (76,591 rows; 431 dropped for argmax-mismatch). NOTE: corpus "noul" rows are stored as choice with `false:/true:` options — the converter passes them through as 2-option choice; Mapika trainer handles them fine.
- **Smoke PASSED:** fwd+bwd on 2 examples, loss 0.1885, grads flow, peak 8.61 GB (incl. full model load). cuDNN SDPA auto-disabled by DecisionModel (Blackwell fix, already in their code).
- **Reference runbook:** Mac `deci-serv/finetune/V6_RUNBOOK.md` (recipe details, transform spec, replay-KL rationale, gates) — committed b2fc4b8.

## Launch command (v6 LoRA run)
```
ssh jaita@192.168.2.185
cd /home/jaita/deci-serv/finetune/v6
# PRE-CHECK: free -b must show ≥ 40 GB avail beyond EXL3's resident set, or wait
nohup /home/jaita/venvs/laya/bin/python -m decider.train \
  --model /home/jaita/models/hf/Mapika/decider-2b \
  --data tasks_v6.pkl --out runs/v6-lora \
  --epochs 2 --lr 8e-5 --warmup 25 \
  --max_tokens 16384 --accum 4 --max_options 10 --max_ctx 1536 \
  --none_prob 0.1 --schema_first_prob 0.0 \
  --eval_every 400 --eval_limit 300 \
  > train_v6.log 2>&1 &
```
- Expect: ~11.4M tokens/epoch, 205 steps/epoch, 410 total; est 45-90 min total; peak ~12-16 GB.
- This IS a LoRA run: their trainer v11-stage used r64 α128 attention+MLP (verify trainer's default mode is LoRA — check `decider.train` argparse for `--lora/--mode`; if it full-FTs by default, add the LoRA flag or accept full-FT at 25-30 GB peak, still fits when EXL3 is down).

## After training
1. **Merge** LoRA → bf16 → `~/models/hf/deciserv/decider-2b-v6/` (new folder, NO deletes; copy config.json + decider_config.json from v11)
2. **Refit temps:** collect T=1 records on val_v5 through the serving readout → `python -m decider.calibrate records.jsonl --min-rows 50` → write `temperature_by_type` + `temperature` into the new decider_config.json
3. **Gates:** (a) val_v5 per-type acc/ECE vs v11 baseline, (b) everyday classes (policy/routing/arcadia benign) hold within 1-2 pts, (c) ECE ≤ 0.08 on hard subsets
4. **Battery v2:** point a decider lane at v6 (`deciserv/server_decider.py --port 8713` style) → `python -m eval.harness --cases eval/cases_v2_only --server http://127.0.0.1:8713 --record eval/recordings --label decider2b-v6 --concurrency 2` → compare vs v11 zero-shot (benign 93%, AUC des 0.9840 / exfil 0.9723) and laya-v4 (AUC 0.9934 / 0.9575)
5. **Six-up re-record** on seed 13 with v6 (`SIXUP_PORT=8011` decider lane)
6. **Promote** :8712 → v6 weights only after gates pass; keep v11 tagged for rollback

## Current live state (as of this writing)
- Spark: gates :8710 laya-v4 ✅, :8711 router-shadow ✅, :8712 decider-2b v11 ✅ (bf16, 28s load)
- **EXL3 daily driver: DOWN** (OOM-killed) — :8000 not listening. Loca's provider dead until restore.
- Mac: showcase :8010 → decider gate :8712 (flipped 9/25), multiview :8030 (3×2, video-style scaling, commit 539c819)
- deci-serv repo: latest commit 539c819 (V6_RUNBOOK, multiview, server_decider, bakeoff doc all pushed)
- Hopper verdict: HopitAI/hopper = research-only license, NOT drop-in; escalation path = **Mapika decider-4b v2.1** (Apache-2.0, drop-in, DI 36.6, board p50 23.4ms)
- Memory tool was erroring; the Ollama-fiction entry + this plan's facts need a memory save retry.

## Key facts that survive any compaction
- Qwen3.8-Flash-Next EXL3 = daily driver, custom engine (vcruz305/exllamav3 fork), :8000, Loca depends on it. OLLAMA IS NOTHING. Do not confuse.
- "switch to X" = kill running workloads, no ask. But nothing new launches without resident-check first.
- Decider-2b zero-shot ≈ laya-v4 calibrated on our battery (AUC 0.984 vs 0.993; exfil 0.9723 vs 0.9575 WIN; benign 93%; p50 164ms@conc2).
- 0xBakeer credit everywhere: `showcase: github.com/0xBakeer/arbiter (MIT)`.
- Repo: github.com/styles01/deci-serv (Mac working dir /Users/clawdio/deci-serv).