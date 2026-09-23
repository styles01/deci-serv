<div align="center">

# DeciServ

**A PyTorch server for System-1 decision models — local serving.**

Small, fast, non-autoregressive classifier models that answer typed questions with **calibrated probabilities in a single forward pass** — loaded resident on your GPU and served over plain localhost HTTP. Decision gates for agentic LLM stacks that cannot hallucinate, because they never generate text.

[![PyTorch](https://img.shields.io/badge/runtime-PyTorch%20CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![NVIDIA DGX Spark](https://img.shields.io/badge/built%20on-DGX%20Spark%20GB10-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
[![Hugging Face](https://img.shields.io/badge/models-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![HTTP](https://img.shields.io/badge/API-stdlib%20HTTP%20%2Fdecide-00599C?style=for-the-badge&logo=http&logoColor=white)](#the-contract)
[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-support-yellow?style=for-the-badge&logo=buy-me-a-coffee&logoColor=white)](https://buymeacoffee.com/aitamedia)
[![Follow on X](https://img.shields.io/badge/Follow%20%40jaita%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/jaita)

[Start Here](#start-here) · [The Contract](#the-contract) · [Providers](#providers) · [Why](#why-system-1) · [Measured](#measured-numbers) · [Contributing](#contributing)

</div>

> [!IMPORTANT]
> DeciServ serves **decision models, not chat models**. A System-1 model takes a *state* and returns *typed calibrated probabilities* in one forward pass (~10–40 ms). It never generates prose, never needs a KV cache, and runs happily **alongside** your main LLM lane. It is the "System 1" to your agent LLM's "System 2."

---

## Start here

```bash
# 1. get a decision model (Laya ships in-box as the reference provider)
git clone https://github.com/styles01/deci-serv && cd deci-serv
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install transformers>=4.48
huggingface-cli download convaiinnovations/laya-typed-decisions --local-dir ~/models/hf/convaiinnovations/laya

# 2. serve it (fp16 by default — half the resident memory of fp32)
python -m deciserv.server --provider laya \
    --checkpoint ~/models/hf/convaiinnovations/laya/typed-decisions \
    --port 8710 --precision fp16

# 3. ask it a typed question about a state
curl -s localhost:8710/decide -H 'Content-Type: application/json' -d '{
  "state": "User asks the agent to rm -rf the database folder during a benchmark run.",
  "questions": {
    "proceed": {"type": "choice",
                "instructions": "Should this destructive command proceed without asking the human?",
                "criteria": {"proceed": "run it", "escalate": "ask first", "refuse": "refuse"}}
  }}'
```

That's the whole product: a **resident decision gate** your agent framework consults before tool calls, with an honest probability attached to every answer.

---

## The contract

Jev-shaped — the same request shape the [TypeSafe Jev](https://openrouter.ai/typesafe/jev-1.13) API standardized, served locally from weights you own:

| Route | Method | Shape |
|---|---|---|
| `/decide` | POST | `{"state": str, "questions": {name: {"type": "choice"\|"score", "instructions": str, "criteria": {...}}}}` → `{"answers": {name: {"type", "choice"/"score", "probabilities", "confidence"}}}` |
| `/health` | GET | model, precision, device, load time |
| `/metrics` | GET | calls, p_avg_ms, p_max_ms, errors, gpu_peak_gib |

Design rules, kept deliberately boring:

- **stdlib HTTP only** (`ThreadingHTTPServer`) — no web framework in the serving path
- **fp16/bf16 on CUDA by default** — the same model in fp32 costs ~40% more resident memory for zero accuracy gain
- **`torch.inference_mode()` on every pass** — no autograd tax, no leak
- **one process, one CUDA context** — safe to run next to an inference lane (LLM, image gen) that owns the rest of the machine
- **providers are the only model-coupled code** — the HTTP layer is model-agnostic

## Providers

| Provider | Model | Params | Context | Best at | Status |
|---|---|---|---|---|---|
| **`laya`** | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) (3 ckpts) | 322–421M | 512–1024 | English/multilingual tool-gating, routing, guardrails, email triage | ✅ in-box, measured |
| *yours?* | any single-pass classifier | — | — | — | PRs welcome — implement `Provider` (4 methods) |

New provider checklist: subclass `Provider`, implement `load()` + `decide()`, add to `REGISTRY`, add a test with a frozen example transcript, and document the checkpoint + precision + measured latency/memory in this table.

## Why System-1

An agent LLM is slow, expensive, and hallucination-prone precisely when it is asked to make a *judgment* it should have made in zero tokens. The 2026 pattern (TypeSafe Jev as hosted API; Laya as open weights; production precedent in OpenAI moderation, Claude Code's Auto Mode classifier, NeMo/Guardrails validators) is to keep a **small calibrated classifier resident** and consult it from the agent framework:

| Decision | Who answers it | Latency |
|---|---|---|
| Should this tool call proceed, escalate, or be refused? | System-1 gate → your approval flow | ~13–40 ms |
| Which model/route should handle this query? | System-1 router | ~13–40 ms |
| Is this tool argument a prompt-injection attempt? | System-1 guard | ~13–40 ms |

The big model answers questions; the gate decides *whether and how* they get asked. Two orders of magnitude cheaper than asking the LLM to judge itself.

**Wiring it into an agent stack** — consult `/decide` from your `pre_tool_call` hook (Hermes), `PreToolUse` hook (Claude Code-style shells), or a validator process (NeMo Guardrails / Guardrails AI). Enforce thresholds fitted on **your** traffic (see calibration notes in the field guide below); a hosted Jev endpoint can sit behind the same contract as a fallback.

## Measured numbers

First numbers from the reference deployment — DGX Spark (GB10, 121 GB unified), `laya-typed-decisions`, fp16, n=42 calls, lane idle:

| Metric | fp32 era | **fp16 (DeciServ)** |
|---|---|---|
| Resident (RSS) | 3.3 GB | **2.3 GB** |
| GPU allocation | ~3 GB | **2.0 GB** (peak alloc 1.67 GiB) |
| Load time | — | **20.3 s** |
| Gate decision p50 | — | **13.2 ms** |
| Gate decision p95 | — | **14.6 ms** |
| Coexistence | fp32 + ComfyUI venv entanglement | **clean lane coexistence (zero lane impact)** |

> These are *first measurements on one box at one checkpoint*, lane-idle. Numbers under LLM-lane decode contention are being measured now (the memory bus is the shared resource — expect degradation, we publish what we find). Reproduce with `--precision bf16` and compare ECE on your own traffic before trusting thresholds.

## Repository map

```text
deci-serv/
├── deciserv/           The server + provider registry
│   ├── server.py       stdlib HTTP, /decide /health /metrics
│   └── providers/      Model backends (base contract + laya)
├── tests/              Frozen-transcript tests (batched == sequential, contract shapes)
├── examples/           Client snippets: Hermes pre_tool_call hook, plain requests
└── docs/               Field guide: decision models, runtimes, calibration
```

## Contributing

Contributions are welcome when they make a provider more accurate, a contract clearer, or a deployment cheaper.

1. Open an issue with the model checkpoint, precision, device, and primary source links.
2. New providers need a paired test with a frozen example transcript and the checkpoint license checked.
3. Latency/memory claims need the measurement command in the PR, not a screenshot.
4. Keep the serving path dependency-free: no web frameworks, no telemetry.

### Support the project

If this repository saved you a few gigs of unified memory, you can support the work at [Buy Me a Coffee](https://buymeacoffee.com/aitamedia), or follow [@jaita on X](https://x.com/jaita) for new providers and measurement notes.

---

## Credits

DeciServ's reference provider is [Laya](https://huggingface.co/convaiinnovations/laya) (RLCD-trained, calibrated against strictly proper scoring rules). The typed-question contract follows the Jev request shape popularized by TypeSafe. Calibration guidance builds on Guo et al. (2017) temperature scaling and conformal risk control.

Thanks to the ModernBERT, mmBERT, PyTorch, and DGX Spark communities.
