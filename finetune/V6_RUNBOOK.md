# v6 Fine-tune Runbook — decider-2b + our 76,591-row corpus

*Generated 2026-09-25 from Mapika's actual training stack (`decider/train.py`, `scripts/train.sh`, HISTORY/CHANGELOG) + our train_v5 corpus. Source of truth for the v6 run.*

## Their recipe (essentials)

- **Entrypoint:** `python -m decider.train --model <ckpt> --data data/mixture_<mode>.pkl --out runs/<name> --epochs 1 --lr 1e-5 --warmup 150 --max_tokens 16384 --accum 2 --max_options 255 --max_ctx 16384 --none_prob 0.1 --schema_first_prob 0.5`
- **Data:** pickle of (train Examples, {task: eval_examples}); Example = context + [Q(text, options[list], gold int)]
- **Objective:** CE on slot readout only — logits at each `Answer k: (` projected through lm_head rows of the 255 label tokens, masked to option count. No answer letters in input; all questions scored in ONE forward pass.
- **No packing** — token-budget batching (T padded to multiples of 64; micro-batch 16,384 tokens).
- **AdamW (0.9, 0.95), wd 0, clip 1.0**, warmup + cosine, seed 0.
- **Full-FT stages:** LR 1e-5 (from base) / 8e-6 (continue). **LoRA stages (v10→v11, v1→v2.1):** r64 α128 on attention+MLP, LR 1e-4, warmup 5%, 65,536 tok/step, 2 epochs, merged to bf16. **Replay rows** trained with KL(p_parent ‖ p_model) instead of labels — this is what kept their fitted temps near parent (1.145 vs 2.06 without).
- **Their pre-registered lesson:** hard-rows-only labels sharpen entropy globally, everyday skills regress (greedy bag-draw −10.9, TypeSafe −4.9), v11 shipped despite failing its own ECE gate (0.156 vs 0.08). Don't repeat: replay-KL + everyday-dominated calib pool.

## Our corpus → their format (transform spec)

76,591 rows (65,092 train / 7,664 val / 3,835 test; choice 38,371 / score 13,670 / noul 13,051):

1. `state` → `Example.context` verbatim; `instructions` → `Q.text`
2. **choice:** `options` already `'name: description'` — keep verbatim; `target.label` → `Q.gold` (drop probabilities)
3. **noul:** options → canonical `['no: …', 'yes: …']`, gold 0=false 1=true
4. **score:** split ~50% listwise (`'N: level'` with leading `N:` stripped) + ~50% isolated rows (`Proposed answer: <level>\nDoes the proposed answer fit?` options `['no','yes']`, gold=1 only on gold level). Never train all-isolated (v8 broke: fit sums 1.4–3.5, −20 acc).
5. Group by `state_hash` → ~10–20% multi-question examples (covers `Question 1:` format)
6. 168 choice rows >10 options → sub-sample to 10 keeping gold
7. Assert `probabilities.argmax == label` on load (floor-check existed at build)
8. Do NOT pre-shuffle options / add abstain — trainer does `none_augment` (p=0.1) + shuffle per epoch
9. `val_v5` doubles as the calibration record pool

## Budget on GB10

- **LoRA (recommended):** ~12–16 GB peak, ~45–90 min for 2 epochs (~30M tokens @ 6–12k tok/s; their v11 anchor: 12.2k tok/s on B300)
- **Full-FT:** ~25–30 GB (plain bf16 AdamW, no fp32 master — their 4B finding) or ~48–55 GB with fp32 master; 1.5–3h for 2 epochs
- LR for our step count: **5e-5–1e-4** (they ran 1,676 steps; we get ~460)

## Risks (from their HISTORY + our env)

1. FLA Triton kernels in backward on sm_121 untested → **2-row fwd+bwd smoke first**
2. cuDNN SDPA wrong-on-Blackwell → keep `enable_cudnn_sdp(False)` (their fix already in DecisionModel)
3. numpy<2 pin vs our numpy 2.5.1 → `pip install --no-deps decider-ai`, never let pip downgrade numpy
4. Calibration drift → replay-KL guard on 20–30% of steps; per-type ECE before/after; everyday splits must hold within ~1–2 pts
5. Temp refit MUST use serving readout (isolated score levels, T=1 collection); pool dominated by everyday rows
6. Tokenizer assert: 255 labels single-tokens after `(` — verify once

## Steps

1. **Env check (read-only):** 2-example fwd+bwd smoke of slot readout, cuDNN SDPA off; verify T=1 letter-logit probs == `Decider` on 3 rows
2. **Vendor trainer:** GitHub `decider/train.py, model.py, prompt.py, data/augment.py, evaluate.py` → `~/deci-serv/finetune/v6_trainer/` (or `pip install --no-deps decider-ai`)
3. **Convert corpus** per transform spec → `tasks_v6.pkl` with task tags; assert argmax==label 100%
4. **Mixture:** none_prob 0.1, max_options 10, `schema_first_prob 0` (v11 is state-first only), max_ctx 16384
5. **Train LoRA:** r64 α128 attention+MLP, LR 5e-5–1e-4, warmup 5% + cosine, 2 epochs, 16,384-token micro × accum 4, bf16 + grad-ckpt, eval every ~100 steps on val_v5 vs v11 baseline (acc + NLL + per-type ECE), log peak VRAM
6. **Replay guard:** 20–30% of steps sample toward frozen v11 distribution, KL loss
7. **Merge** → NEW folder `~/models/hf/deciserv/decider-2b-v6` (no deletes), config + `decider_config.json` (plain, max_options 255, isolated_levels true, schema_first false)
8. **Refit temps:** T=1 records on val_v5 through serving readout → `decider.calibrate.fit_by_type --min-rows 50` → write into decider_config; sanity: mean KL to v11 ≤ ~0.1 nats
9. **Gates:** val_v5 per-type acc/NLL/ECE vs v11 w/ bootstrap; everyday subsets hold within 1–2 pts; ECE ≤ ~0.08 hard subsets; latency via gate path
10. **Promote** in lane-idle window: swap :8712 config to v6 folder, keep v11 tagged for instant rollback

## Escalation

If v6 battery still shows exfil deny <90% or des unsafe-pass: **decider-4b (v2.1)** — Apache-2.0, 8.41 GB bf16, TRUE drop-in (same Decider class + shim, 24 GDN instances). NOT hopper (research-only license).