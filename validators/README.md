# LayaTETRIS — decision-model validator

Headless Tetris harness that validates the Laya checkpoint (DeciServ :8710).
Every piece placement is ONE typed decision: POST /decide {state, questions}
(choice over legal placements + danger noul). Deterministic harness owns all
mechanics; the model only advises (fail-open, same pattern as DeciServ gates).

Run: python3 laya_tetris.py  (60 pieces, seed 42)

Metrics: latency p50/p95, greedy-baseline agreement, danger calibration
(noul vs measured stack height), lines/score.

Run 2026-09-24 (laya-gate-v1-merged pre-finetune):
- latency p50 183ms / p95 236ms (n=60, Spark round-trip)
- 0 lines cleared in 60 pieces; degenerate leftward policy (13/14 picks col 0-1)
- baseline agreement 2/60; danger noul ~flat 0.27-0.33 (separation nan)
Interpretation: infra + latency + shape all correct; model quality is
OOD-flat as measured everywhere else. Re-run after each fine-tune checkpoint
to track separation/baseline-agreement trends.
