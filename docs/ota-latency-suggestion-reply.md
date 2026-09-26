# Reply to the observation→action latency suggestion

*Draft, 2–4 sentences, technical, no hype, no attribution (suggestion received uncredited; do not attach a handle). Ready to paste.*

---

Good call — we're wiring exactly that: per-decision timestamps for observation captured, request sent, response received, and action applied, so observation→action p95 (queue time included) gets logged alongside the gate's plain inference latency. The A/B is decider-only vs decider while our ~103 GB EXL3 driver decodes on the same GB10 pool, judged against each game's tick period with margin — the "arrives in time" test, not just "computes fast". Will report the p50/p95/p99 table once the two 240 s windows are done.

---

## Alternates (same content, different emphasis)

**Shorter, 2 sentences:**
Agreed — we now log observation→action p95 (queue included) next to raw inference latency per decision, across all six showcase games. Next step is the A/B: decider alone vs decider sharing the GPU with our EXL3 daily driver mid-decode, with each game's tick period as the in-time bound.

**Longer, 4 sentences:**
That's the right metric, and it's now instrumented end-to-end: every decision gets four timestamps (observation captured, request sent, response received, action applied) on a six-game live setup, one 4GB decision model per move on a DGX Spark. From those we compute inference latency separately from observation→action p95, including any wait between response and the next tick boundary where a late decision actually applies. The experiment is a clean A/B: decider alone, then decider while a ~103 GB EXL3 inference server decodes continuously on the same 130 GB unified-memory GPU. The pass/fail line is per game: observation→action p95 must stay under half the game's tick period with ≥99% of decisions in time — i.e., fast decisions still arrive in time while both models share the memory bus.