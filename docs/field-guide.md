# System-1 Decision Models on a Memory-Constrained Box: A Practitioner's Field Guide (2026)

**Scope.** Fast, small, non-generative classifier models used as gates/routers/guardrails in agentic LLM workflows; how to instantiate and serve them on an NVIDIA DGX Spark (GB10, 121GB unified LPDDR5x memory, aarch64, CUDA 13) alongside a large EXL3 LLM lane.

**Verified vs inferred.** Every claim below is tagged **[V]** (verified: read directly from a primary source or from live execution on the Spark) or **[I]** (inferred: my estimate/reasoning from verified facts). Confidence per claim: high / medium / low.

---

## 1. The model landscape beyond Laya

### 1.1 TypeSafe Jev (the commercial System-1 archetype) — [V, high]
- Hosted API only. `POST https://api.typesafe.ai/v1/systemone`, model id `jev-latest` / `typesafe/jev-1.13` on OpenRouter. Launch 2026-09-15, $0.042 per 1M input tokens, free output. Sources: https://github.com/Nedomas/awesome-jev-2 (488-entry directory), https://openrouter.ai/docs/guides/community/jev-tutorial, https://www.kie.ai/blog/what-is-jev
- Contract: **state + typed questions (Choice / Score / Noul) → typed answers with calibrated probabilities**; parallel evaluation, no token stream; TypeSafe quotes 70–500 ms. Sources: same + https://www.requesty.ai/blog/typesafe-jev-explained
- Trained with RLCD (Reinforcement Learning for Calibrated Decisions), by Diogo Almeida (ex-OpenAI RL). TypeSafe publishes a "Jev 1.13 jaggedness" doc of known failure modes — [V, high] (awesome-jev-2 "Official" section).
- **No downloadable weights, no open license** [V, high — kie.ai explicitly confirms; multiple HF "openjev" repos are independent replicas, not the vendor]. Arize (Laurie Voss) cautions the "can't hallucinate" claim means only *can't leave the schema* — within-schema wrong answers remain possible. https://arize.com/blog/typesafe-jev-llm-judge/ [V, high]

### 1.2 Laya (the open, self-hostable counterpart) — [V, high]
https://huggingface.co/convaiinnovations/laya
- Three checkpoints: **laya** (English, ModernBERT-large backbone 395M + decision head → **421M total**, 512-token budget/question), **laya-multilingual** (mmBERT-base, 322M, 1024-token default, up to 8k, 100+ languages, ~2.2× faster), **laya-typed-decisions** (ModernBERT-large 421M, 1024 ctx, for typed-decisions workflows, 0.766 acc on those workflows).
- Mechanism [V]: every option scored at its own  marker token; softmax per question; **all questions in one forward pass**; measured by vendor: **38.4 ms p50 (p95 42.1 ms) for 1 question, 156 ms for 10 batched, 721 ms for 50** on GPU.
- Training [V]: RLCD with **strictly proper scoring rules** (log score + spherical score; ranked probability score for ordinal questions), TD(λ=1.0) over dialogue prefix slices, 100% human-annotated data; **fitted per-cardinality calibration temperatures [1.637, 1.251, 1.983]**; vendor-reported 99.1% routing accuracy with **ECE 0.009**.
- Vendor-published Laya-vs-Jev table (Laya: 83.8% in-task macro acc vs Jev's published 67.8% across 4 workflows; ~10.4× faster p50) — **[V that the table exists on the model card; I] the numbers themselves are vendor claims, medium confidence, not independently reproduced**.
- Laya is Jev-*shaped* (same state+questions contract) but not weight-compatible with the Jev API; our existing Jev-compatible `/decide` shape maps onto it directly.

### 1.3 Open replicas of Jev — [V, medium confidence on quality]
- **AlexWortega/openjev** (MIT): Qwen3.5-4B as an NLI cross-encoder (`Qwen3_5ForSequenceClassification`, 3 labels, last-token pooling) + frozen-backbone per-task MLP heads on a Qwen3.5-35B-A3B backbone. https://huggingface.co/AlexWortega/openjev [V, high that it exists and how it works]
- **NicolaiLassen/open-bonzi-jev**: 27B-param model, 5.95 GB file, **~1.5 s per decision** on a laptop GPU. https://github.com/NicolaiLassen/open-bonzi-jev [V, high] — note the latency: decoder-based replicas are 1–2 orders of magnitude off the encoder-sidecar target. Wrong shape for a hot gate.
- **heman10x/openJev-verdict-2.0**: gated (HTTP 401 without HF auth), described as "vendor baseline cataloged in Laya's published evaluation suite." https://huggingface.co/heman10x/openJev-verdict-2.0 [V, high that it's gated; low confidence on contents]
- Ecosystem: **Nedomas/awesome-jev-2** (488 entries, refreshed 2026-09-18) — includes **pi-jev**: "TypeSafe Jev as a decision layer for the Pi coding agent: a **measured tool-call gate** plus jev_ask for typed calibrated answers" and **jev-router**: route to the cheapest model in Claude Code by task. This is the Hermes-adjacent community pattern: decision model consulted *before* tool calls. [V, high]

### 1.4 Guardrail classifier models (encoder-class, cheap gates)
- **meta-llama/Llama-Prompt-Guard-2-86M** (Purple Llama; multilingual injection/jailbreak detection, 86M) and **Llama-Prompt-Guard-2-22M** (English-only, for resource-constrained settings; Meta's own model card recommends 22M for constrained environments). DeBERTa-class encoders. https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M , https://developer.meta.com/ai/docs/model-cards-and-prompt-formats/prompt-guard/ [V, high]
- **google/shieldgemma-2b** (2.6B, decoder-based, text-to-text safety classifier; open weights, fine-tunable). https://huggingface.co/google/shieldgemma-2b , https://deepmind.google/models/gemma/shieldgemma-2/ [V, high]
- **allenai/wildguard** (7B Mistral-0.3 fine-tune, multi-task: prompt harm / response harm / refusal detection, Apache-2.0). https://huggingface.co/allenai/wildguard [V, high]
- **Llama Guard 3 8B / Llama Guard 4 12B**: the 8–12B decoder-based moderation family; a 2026 benchmark (arXiv 2608.21775) found specialized CM models (Llama-Guard-3-8B, LG4-12B, GPT-4.1) at ~67% on complex adversarial cases — decoder guards are heavier but still not perfect. https://arxiv.org/html/2608.21775 [V, medium]
- Framing: Prompt-Guard-class *encoders* are the right size class for a resident gate; ShieldGemma/WildGuard/Llama-Guard *decoder* guards are a different weight class (2.6–12B), usable but 10–100× the memory/latency of a 400M encoder.

### 1.5 Semantic routers and classifier-as-gate infrastructure — [V, high]
- **aurelio-labs/semantic-router**: utterance-embedding route matching ("superfast AI decision making"), encoders pluggable (Cohere, OpenAI, HF, FastEmbed), multi-modal; the original embedding-space router. https://github.com/aurelio-labs/semantic-router . Limits: threshold-on-cosine similarity, no calibrated probabilities, no learned decision head — fine for intent routing, not for calibrated gating. [V, high; characterization medium]
- **LLM Guard** (scanner pipeline: toxicity/bias/PII) — one 2026 comparison reports it **archived**, prefer maintained alternatives. https://nomadx.ae/blog/ai-agent-guardrails-nemo-guardrails-ai-llama-guard-2026/ [V, single source, low-medium confidence]
- Framework-level: **Guardrails AI** (RAIL validator pipeline), **NeMo Guardrails** (Colang DSL) — both integrate Llama-Guard-style sidecars as validators; comparisons: https://kanopylabs.com/blog/guardrails-ai-vs-nemo-guardrails-vs-llm-guard , https://aipromptshub.co/safety/nvidia-nemo-guardrails-vs-guardrails-ai [V, medium]

### 1.6 Frontier-platform precedent
- **OpenAI**: moderation is a **separate hosted classifier endpoint** (`omni-moderation-latest`, free) called by products on every request — the canonical "classifier as an out-of-band gate" pattern. https://platform.openai.com/docs/guides/moderation/overview [V, high]
- **Anthropic/Claude Code**: tool-call gating via (a) **deterministic hooks** (PreToolUse/Stop hooks as script verdicts — the Tirith-style pattern), (b) **Auto Mode's two-stage classifier** that evaluates each tool call before execution (auto-approve safe ops, block destructive patterns), and (c) verification **subagents** as second opinions. Sources: https://www.anthropic.com/engineering/claude-code-best-practices [V, high], https://www.agentpatterns.ai/tools/claude/auto-mode/ [V, medium], and an academic teardown mentioning an ML-based classifier in Claude Code's permission path: https://arxiv.org/html/2604.14228v1 [V, medium]

---

## 2. Runtime comparison for a ~421M encoder on GB10 (DGX Spark)

**Platform ground truth (live-checked on the Spark, 2026-09):** Linux 6.17.0-1026-nvidia, **aarch64**, CUDA toolkit **13.0** at /usr/local/cuda-13.0, GB10 (sm_121) reported by nvidia-smi, **121 GiB total RAM, ~23 GiB free under the current working load (98 GiB used — consistent with a ~60 GB-class resident LLM lane)**, Python 3.12.3, system `transformers 5.14.1`. **PyTorch aarch64+cu130 wheels are proven on this box**: torch 2.10.0+cu130, 2.12.1+cu130 (×2 venvs), 2.13.0+cu130 all report `torch.cuda.is_available() == True`. [V, high]

### 2.1 PyTorch (CUDA, fp16) — **recommended**
- Weights: 421M × 2 B ≈ **0.84 GB** (fp16) / 1.68 GB (fp32). [V arithmetic, high]
- Resident process: weights + CUDA context (~300–500 MB) + activations for batch-8×512-token inference (<~150 MB) + allocator slack ⇒ **~1.5–2.2 GB fp16, ~2.5–3.3 GB fp32**. [I, medium-high — consistent with the observed "3.3 GB" PyTorch figure, which is what fp32 + allocator cache looks like]
- Latency: Laya's own GPU numbers (38.4 ms p50 single question, 156 ms/10 questions) were measured on an unspecified GPU; GB10's compute is more than sufficient for a 0.43-GFLOP-class pass, so **~20–50 ms p50 in-lane** is a fair expectation. [I, medium]
- Risk: none material — wheels exist and run on this exact machine. [V, high]

### 2.2 ONNX Runtime — viable but wheel-hostile on this platform
- Official `onnxruntime-gpu` PyPI wheels exist **only for x86_64/Windows; there is no official aarch64 build at any version** (also absent from pypi.nvidia.com). Community builds for CUDA 13 / sm_121 / aarch64 exist: https://github.com/seitzbg/onnxruntime-gpu-sm121-aarch64 , https://huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121 ("As of March 2026, there is no official onnxruntime-gpu wheel for aarch64 + CUDA 13 on PyPI"), https://github.com/Albatross1382/onnxruntime-aarch64-cuda-blackwell (prebuilt ORT 1.24.4 shared libs, sm_121). [V, high]
- CPU-only `onnxruntime` installs fine on aarch64; a 421M encoder on GB10's 20-core Grace CPU is on the order of **~100–300 ms/pass**. [I, medium-low]
- Payoff if used: lower steady-state memory (~1.0–1.3 GB) and often 15–35% lower latency vs eager PyTorch for small classifiers (third-party benchmark; not GB10-specific). https://gigagpu.com/onnx-runtime-vs-pytorch-inference-gpu/ [V that the claim is published; I, low-medium that it transfers]
- Verdict: correct tool when wheel friction is solved by someone else; **not worth depending on a community wheel** for the first deployment. [I, high]

### 2.3 ExecuTorch — wrong tool here
ExecuTorch targets mobile/embedded/desktop via XNNPACK etc., with memory planning and quantization; deployment requires `torch.export` capture plus operator/backend coverage, "validate numerical accuracy, memory use, and performance on the target." https://github.com/pytorch/executorch , https://docs.pytorch.org/executorch/stable/intro-how-it-works , https://arxiv.org/pdf/2605.08195 [V, high]. **No CUDA 13 / sm_121 desktop-GPU backend**; you'd be running the encoder on CPU with an export toolchain you don't need. [I, high]

### 2.4 llama.cpp / GGUF — non-starter for this model class
BERT-family support in llama.cpp has existed only in limited embedding form and required dedicated arch plumbing; ModernBERT was explicitly flagged as needing llama.cpp-side changes (closed HF discussion pointing to llama.cpp), and GGUF BERT remains niche. https://huggingface.co/nomic-ai/modernbert-embed-base/discussions/9 , https://github.com/ggml-org/llama.cpp/discussions/7712 , https://github.com/ggml-org/llama.cpp/issues/2872 [V, high]. Decisive: **Laya's custom decision head (option-marker scorer, per-cardinality temperature calibration, act/escalate head) cannot be expressed as a GGUF arch** — you'd lose the calibrated head entirely. [I, high]

### 2.5 vLLM classify/pooling — real, but the wrong engine to add to this box
vLLM supports classification via pooling models (`LLM.encode` with `pooling_task="classify"`, `--runner pooling`, softmax activation; recent work generalizes multi-task/multi-label pooling). https://docs.vllm.ai/en/stable/models/pooling_models/classify/ , https://github.com/vllm-project/vllm/pull/30315 [V, high]. But: it's a second engine process (own CUDA context ~0.5 GB + scheduler), its continuous-batching strength is irrelevant for single-pass classifiers, ModernBERT-family support is secondary-tier, and the main LLM lane here is **EXL3/exllamav3, not vLLM** — you'd be adding a whole runtime to save milliseconds a FastAPI wrapper already gives you. [I, high]

### 2.6 Summary table (421M encoder, GB10, alongside ~60 GB EXL3 lane)

| Runtime | Resident mem | p50 latency/pass | Status on GB10 | Verdict |
|---|---|---|---|---|
| PyTorch fp16 CUDA | ~1.5–2.2 GB | ~20–50 ms (I) | **wheels proven on this box** [V] | **use this** |
| PyTorch fp32 CUDA | ~2.5–3.3 GB | ~25–60 ms (I) | works [V] | fallback only |
| ORT GPU (community wheel) | ~1.0–1.3 GB (I) | ~15–40 ms (I) | community wheel only [V] | later optimization, not foundation |
| ORT CPU | ~1.0 GB (I) | ~100–300 ms (I) | installs fine | only if GPU contended |
| ExecuTorch | ~1 GB CPU (I) | ~100–300 ms (I) | no sm_121 GPU backend | reject |
| llama.cpp/GGUF | — | — | ModernBERT+decision head inexpressible | reject |
| vLLM pooling | +1 engine proc (~1.5 GB+) (I) | ~10–30 ms (I) | works but heavyweight | reject for this job |

All memory figures beyond raw weight bytes are **[I] estimates** to be confirmed with a 30-minute on-box measurement (torch.cuda.max_memory_allocated + RSS before/after warmup).

---

## 3. Serving pattern: in-process vs sidecar vs shared engine

**What production actually does in 2026** [V, high]:
1. **OpenAI**: separate hosted moderation classifier, called out-of-band per request (§1.6) — the sidecar/gate pattern at platform scale.
2. **Claude Code Auto Mode**: two-stage classifier gating every tool call, plus deterministic hook verdicts (exit-code semantics) — classifier consulted *by the agent framework*, not embedded in it (§1.6).
3. **NeMo Guardrails / Guardrails AI / Llama Guard integrations**: guard models are run as separate server endpoints (or at least separate model processes) and invoked as validators from the orchestration layer (§1.5).
4. **Hermes-adjacent community**: pi-jev (measured tool-call gate + jev_ask in the Pi agent), jev-router (routing gate in Claude Code) — decision model consulted in the pre-tool-call path (§1.3).
5. **Jev itself**: pure HTTP service with a typed contract (§1.1) — the market's flagship is *definitionally* a sidecar.

**Pattern analysis for our stack** [I, high]:
- **In-process library import** (classifier in the agent's own Python): lowest latency (no HTTP hop, ~1–5 ms saved) but (a) drags a CUDA context into the agent process (~300–500 MB + init cost), (b) crash-couples the gate with the agent, (c) forces the agent host to have cu130 torch — breaks portability, (d) fights Hermes's settled integration shape (hooks + auxiliary endpoints are HTTP-shaped).
- **Shared engine process** (one process hosting both the EXL3 lane and the classifier): tightest memory packing on paper (one CUDA context), but couples a 60 GB production LLM lane to a experimental model's lifecycle — any classifier OOM/reload stalls the main lane. Rejected.
- **Sidecar HTTP service** (small FastAPI/uvicorn on localhost, Jev-shaped `/decide` + typed probabilities): matches every production precedent above, matches Hermes's plugin pre_tool_call hook and `auxiliary.<task>.base_url` slots, isolates CUDA context (~0.5 GB) inside the sidecar, lets the same endpoint fall back to the hosted TypeSafe API (identical contract) when the box is down. Cost: localhost HTTP hop ≈ 1–5 ms + serialization — noise against a 38 ms model pass. **Winner.**

Memory ledger on GB10 with the 60 GB lane resident: 121 GB total ⇒ ~55–60 GB headroom observed under working load (23 GiB free *now* includes other services; the lane is the dominant consumer) [V, high on the observation; I on the exact headroom attribution]. A 2–2.5 GB sidecar is ~4% of headroom.

---

## 4. Calibration best practice

1. **Train/serve against proper scoring rules.** RLCD's core move: reward = log score + spherical score (+ ranked probability score for ordinal Score questions); the optimum of a strictly proper scoring rule is the true distribution, so a model optimized to convergence *must* report calibrated probabilities. Laya ships fitted per-cardinality temperatures ([1.637, 1.251, 1.983]) as post-hoc polish, vendor ECE 0.009 on routing. https://huggingface.co/convaiinnovations/laya [V, high that this is the published method; medium that ECE 0.009 holds outside vendor evals]
2. **Post-hoc temperature scaling on a held-out calibration split** (Guo et al., 2017 — single-temperature Platt-style rescale; standard, one parameter, no accuracy loss). Apply *per question-cardinality* as Laya does (a 3-way and an 8-way softmax need different temperatures). https://torch-uncertainty.github.io/auto_tutorials/Post_Hoc_Methods/tutorial_temperature.html [V, high]
3. **Don't assume calibration fixes conformal thresholds.** Whether temperature scaling helps or hurts conformal coverage is empirical, not settled: https://openreview.net/pdf?id=6DDaTwTvdE [V, high that this is an open question in the literature]
4. **Set gate thresholds with distribution-free guarantees, not vibes.** Split conformal prediction / **conformal risk control** (Angelopoulos, Bates, Fisch, Lei, Schuster, arXiv:2208.02814): hold out a calibration set from production-like traffic, pick the block-quantile so the expected monotone loss (e.g., false-block rate, or missed-gate rate) is bounded at level α. Recompute on drift; monitor ECE + per-threshold precision/recall continuously. https://arxiv.org/pdf/2208.02814 , https://github.com/aangelopoulos/conformal-risk [V, high]
5. **Exchangeability caveat**: conformal guarantees die under distribution shift; for an agent gate whose input distribution is whatever the user typed today, budget for periodic recalibration and treat the guarantee as soft. https://proceedings.mlr.press/v202/lu23i/lu23i.pdf [V, high]
6. **Schema-bounded ≠ correct.** Jev/Laya cannot emit an out-of-schema answer but can be confidently wrong in-schema; the probability signal is the product, so measure ECE/Brier on *your* traffic before trusting thresholds. (Arize's critique, §1.1.) [V, high]

---

## 5. Concrete recommendation for our stack (421M model on GB10 + 60 GB EXL3 lane)

**Model**: `convaiinnovations/laya` (ModernBERT-large 421M) as the resident gate — the only open-weight model with the exact Jev-shaped contract (state + Choice/Score/Noul → typed calibrated probabilities), RLCD proper-scoring-rule training, and single-forward-pass batching. Add `laya-multilingual` (322M mmBERT) only if multilingual traffic is real; `laya-typed-decisions` if the workload is exactly the typed-decisions workflows. Keep the hosted `typesafe/jev-1.13` (OpenRouter or api.typesafe.ai) as a drop-in remote fallback behind the same `/decide` contract. [I, high]

**Runtime**: **PyTorch fp16, CUDA 13, in a dedicated sidecar process.** Verified-feasible today (torch cu130 wheels run on this exact box; transformers 5.14.1 present; ModernBERT supported since transformers 4.48). Do not use llama.cpp/GGUF (head inexpressible), ExecuTorch (no sm_121 GPU backend), vLLM (second engine, wrong trade), or ONNX GPU (no official aarch64 wheel; revisit community ORT wheel only if memory pressure ever becomes real). Quantize to fp16 immediately; optionally int8 dynamic on Linear layers later if memory ever matters. [I, high; fp16 speedup magnitude medium]

**Serving shape**: localhost HTTP sidecar (FastAPI/uvicorn, port e.g. 8710) exposing (a) the existing Jev-compatible `/decide` (state + typed questions → typed answers + probabilities) and (b) an OpenAI-compatible classification route for Hermes `auxiliary.<task>.base_url` slots. Wire into Hermes via the plugin **pre_tool_call policy hook** (approve/block/modify from `p(true)` vs conformal threshold) and middleware. One process, one CUDA context, auto-restart watchdog, health endpoint. [I, high — pattern matches all §3 precedent]

**Memory budget**: fp16 weights 0.84 GB + CUDA context ~0.4 GB + activations/allocator ~0.3–0.8 GB ⇒ **budget ~2.0–2.5 GB resident** (measure and record actuals). This is ~4–5% of the headroom left by the 60 GB lane. The "3.3 GB PyTorch resident sidecar" figure is almost certainly fp32 + allocator cache — right shape, shrinkable number. [I, medium-high]

**Latency budget** (per tool call, in-lane): tokenize+forward p50 ~20–50 ms, localhost HTTP +3–6 ms, batch N questions in the same pass (10 questions ≈ 156 ms vendor-measured). Target **p95 ≤ 120 ms gate decision** excluding agent overhead; alarms at p95 > 200 ms. Fall back to hosted Jev (70–500 ms) only for the rare multilingual/overflow path. [I, medium; anchored to vendor [V] numbers]

**Verdict on the 3.3 GB PyTorch resident sidecar**: the **sidecar shape is right** (matches OpenAI/Claude Code/NeMo/pi-jev precedent and Hermes's hooks); the 3.3 GB figure is the fp32 default, not the floor — **fp16 + allocator tuning should land ~2.0–2.5 GB**, and ONNX/in-process alternatives are not better on this platform (wheel hostility / CUDA-context coupling). Ship PyTorch-fp16 sidecar now; keep an ONNX-GPU path as a documented contingency.

**Verification plan** (before trusting thresholds): run Laya fp16 on the Spark, record RSS + `torch.cuda.max_memory_allocated` and p50/p95 over 1k real prompts; compute ECE/Brier on our tool-call traffic; fit thresholds via conformal risk control on a 500+ example holdout; re-fit monthly. [I, high that this is the right procedure]

## Confidence summary
- High, verified: Jev contract/pricing/API-only; Laya architecture/temperatures/latency claims as published; Spark platform facts; torch cu130 wheels; ORT wheel gap; llama.cpp/GGUF ModernBERT gap; vLLM classify API; production serving precedents; conformal/temperature-scaling canon.
- Medium, inferred: exact on-box latency (20–50 ms) and memory floor (2.0–2.5 GB); transfer of ORT-vs-PyTorch speed deltas to GB10; vendor accuracy numbers (83.8%, ECE 0.009) outside their evals.
- Low: quality/contents of gated replicas (openJev-verdict-2.0); LLM Guard's maintenance status (single source); semantic-router characterization depth.