# Battery SLO columns — spec for eval/harness.py output

**Status: SPEC ONLY (harness edit is a later step, after the S0 baseline capture).**
Target file: `eval/harness.py` (functions `run_online`, `build_report`,
`print_report`, CLI in `main`). Nothing here changes the decision gates —
measurement only. The existing `latency_by_path` block stays; these columns
add to it, they do not replace it.

## Why

The battery currently reports correctness gates plus per-path p50/p95/max
(`latency_by_path`, harness.py:793-801). SLO work needs the same run to state
what the gate *costs* and *throughputs at* its latency tail: p99 and
throughput-under-tail are the columns a serving decision (cascade on/off,
cache on/off, batch vs per-question) is judged on. All numbers come from data
the harness already collects — `client_latency_ms` per case and the run's
wall clock — so the change is measurement bookkeeping, not new machinery.

## Column definitions

| column | definition | source |
|---|---|---|
| `latency_ms_p50` | nearest-rank p50 of `client_latency_ms` over non-error cases (existing `pctl`, harness.py:572) | per-case client timing |
| `latency_ms_p99` | nearest-rank p99, same population | per-case client timing |
| `qps` | successful `/decide` responses ÷ wall-clock seconds of the run's request phase | `len(results)` and the wall span around the `ThreadPoolExecutor` block in `run_online` (harness.py:616-619) |
| `decisions_per_sec_at_p99` | max sustainable rate if every decision had to stay under the p99: `1000 / latency_ms_p99` | derived, client-side |
| `slo_ok` | `latency_ms_p99 <= SLO_P99_MS` and `qps >= SLO_QPS` (SLO env-tunable, see below) | the two above |

Notes on semantics:

- Use `client_latency_ms` (end-to-end from the harness), not the server's
  `latency_ms` field — the SLO is what a caller experiences. The server-side
  figure stays in `latency_by_path` for provider comparisons.
- Error cases (`answers is None`) are excluded from latency percentiles but
  included in the denominators of nothing (they are a separate gate,
  "no case errors"); a run with errors prints `slo_ok: false` with the
  reason `errors>0` regardless of the numbers.
- `decisions_per_sec_at_p99` is per-worker (sequential client math), not the
  aggregate `qps`; the two together answer "how fast is one decision" vs
  "how much does the gate throughput". Keep both columns adjacent in output.

## Config

Two env knobs read once in `main`, echoed into the report under
`thresholds.slo` so a recording carries its own SLO envelope:

```
DECISERV_SLO_P99_MS   default 50.0   (ms; the S0 baseline sets the real bar)
DECISERV_SLO_QPS      default 40.0   (aggregate requests/sec across workers)
```

CLI mirrors: `--slo-p99-ms`, `--slo-qps`. Both optional; when unset the env
defaults apply. No SLO enforcement changes `passed` — `slo_ok` is an
INFORMATIONAL column this revision (correctness gates remain the pass/fail
authority), so adding columns cannot flip a green battery red. A later step
may promote `slo_ok` into the gate list; that promotion must be its own
reviewed change.

## Report schema additions (SCHEMA_REPORT v1 — additive only)

`build_report` gains one dict; no existing key moves or changes type:

```json
"slo": {
  "p99_ms_target": 50.0,
  "qps_target": 40.0,
  "latency_ms_p50": 13.1,
  "latency_ms_p99": 41.7,
  "qps": 57.3,
  "decisions_per_sec_at_p99": 23.98,
  "n_latency_samples": 134,
  "n_errors_excluded": 0,
  "slo_ok": true
}
```

`run_online` must return wall-clock timing so `build_report` can compute
`qps`: extend its return tuple (currently `questions, recording_cases,
scored, h` at harness.py:638) to also carry
`{"t0": float, "t1": float, "n_ok": int}` — computed immediately around the
`ThreadPoolExecutor` map (harness.py:616-619), before any post-processing.
`run_offline` returns `slo: None` (no timing to measure); `build_report`
omits `latency_ms_p50`…`slo_ok` when `slo is None` and the report says
`"slo": null` so offline re-scores stay schema-clean.

`print_report` gains one line after the latency-by-path block
(harness.py:862-864):

```
  slo: p50 13.1ms  p99 41.7ms  qps 57.3  decisions/s @ p99 24.0  [ok / MISS]
```

## Acceptance

- Online report carries all ten `slo` keys; offline report carries `"slo": null`.
- `--json` output includes `slo` (machine consumers get it without parsing stdout).
- Re-running the same battery twice against an idle server produces p99s
  within ±20% (else the column is too noisy to gate on — investigate before
  promoting `slo_ok`).
- Percentile math reuses `pctl` (harness.py:572-578); no new percentile helper.