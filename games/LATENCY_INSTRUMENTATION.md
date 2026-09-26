# LATENCY_INSTRUMENTATION — observation→action (OTA) latency for the six-game decider

*Status: instrumentation implemented and lab-verified; production A/B NOT yet run (needs a showcase restart by the operator). Design date 2026-09-26.*

**Suggestion credit (uncredited on purpose, per request): a Twitter contact suggested tracking observation-to-action p95 including queue time, alongside inference latency, comparing idle vs actively-decoding co-residency. This doc is that idea, made concrete for our stack.**

Showcase UI/games: 0xBakeer's arbiter showcase (MIT) — github.com/0xBakeer/arbiter. Vendor files carry our instrumentation only where noted below; his credit stays in every page and in NOTICE.md.

---

## 1. Pipeline anatomy — where observation and action actually happen

The live path is **browser-centric**. The six game loops do NOT run in a Python game
server; they run as JS in each showcase page (six tabs, or six iframes in the multiview
grid). The Python servers are adapters:

```
[page JS loop]  --HTTPS POST /v1/systemone-->  [:8010/:8011 serve_showcase.py adapter]
                                                --POST /decide-->  [Spark :8712 decider-2b gate]
[page JS loop]  <--answers envelope JSON-----  adapter            <--answers, latency_ms--
```

Per game tick (from `_lib/harness.mjs` and the per-page code, identical in all six
pages — verified byte-identical hook anchors):

1. `boot()` (page load): 2.5 s `/readyz` probe → live mode (or recorded fallback).
2. `requestNext(state)` — **the observation is captured here**: state is final, the
   model is asked for the NEXT move while the current one animates (sub-tick pipelining).
3. `askLive(state)` → `client.ask(...)` (`_lib/client.mjs`): one POST to the adapter.
4. Adapter `proxy_decide()`: rewrites to `{state, questions}` → gate `/decide`; gate
   `latency_ms` = its own inference time (p50 ~15.5 ms on-box; ~25–50 ms from Mac LAN).
5. Response → `decide(state, answers)` → `shield()` → stored as `ep.result`.
6. **Action applied at the next tick boundary** in `frame()` → `apply(res, now)` →
   `logic.step(s, action)` — the board actually moves here.

### Where queueing exists (all found, all instrumented)

| Queue | Where | Mechanism |
|---|---|---|
| Post-response wait | `frame()`: `t>=1` but `ep.result` null → **hold at end of glide** until it lands, apply immediately | if RTT > tick, decisions apply late — the visible "arrive in time" failure |
| Tick pacing | `apply()` immediately calls `requestNext()`; next decision's RTT overlaps the tick | pre-queue ≈ 0 by design (this is why obs→action ≈ rt + small ε) |
| Multiview compositing | :8030 iframes; CSS-scaled cells; **cross-origin (cannot read scores/timing from frames)** | the only :8030-side signal is iframe content-load latency → we probe cell load |
| Gate-side | :8712 ThreadingHTTPServer — no request queue; GPU does serialize concurrent calls | visible as inflated `latency_ms` under load, i.e. inside inference time |
| Game loop Python | none (there is no Python game loop — the games are JS) | — |

Tick periods (`logic.mjs meta.tick_ms`): paddle 60, hopper 100, snake 120, crossing 150,
mines 250, dungeon 500 ms. **Paddle is the stress case: 60 ms tick vs ~50–200 ms decision
RTT from the Mac — it always holds the glide; that is expected and is exactly what the
in-time/late split quantifies.**

## 2. Instrumentation design

Per decision, four timestamps (browser epoch, `performance.timeOrigin + now()` so
beacons are wall-clock-comparable):

| ts | Meaning | Hook site |
|---|---|---|
| `obs_ts` | observation captured (state final, decision requested) | page `requestNext()` → `window.__otaNext()` |
| `req_ts` | request sent | `client.ask()` (same as obs today; split kept for future pre-send work) |
| `resp_ts` | response received | `client.ask()` after `res.text()` |
| `action_applied_ts` | action applied in the loop | page `apply()` → `window.__otaApply(now, ep.tickStart)` |

Derived: `gate_inference_ms` (gate's own `latency_ms`), `rt_ms` (req→resp, browser),
`obs_to_action_ms` (**the OTA metric**), `postqueue_ms` (resp→apply), `wait_ms`
(apply − tickStart; >0 ⇒ applied after the tick boundary = late), `in_time` (computed
in analysis: obs_to_action ≤ tick period with margin — see §4), plus a cached
`co_resident_load_hint` (EXL3 vLLM `num_requests_running/waiting` + gate `pool_used_gb`,
polled ≤ every 2 s, ~8 ms cost via literal IP — **never `.local` names: ~3 s mDNS
penalty measured**).

### Event schema (JSONL, `games/ota_latency/<day>/ota.jsonl`)

One line per event; analyzer joins `req`+`apply` per `(game, page, seq)`:

```json
{"ts": ..., "type": "req",    "game": "snake", "page": "x7k2", "seq": 41, "tick_ms": 120,
 "obs_ts": 1790443000000, "req_ts": 1790443000001, "resp_ts": 1790443000047,
 "gate_inference_ms": 43.7, "rt_ms": 46.2, "co_resident_load_hint": {"exl3": "up",
 "exl3_running": 1.0, "exl3_waiting": 0.0, "gate_pool_used_gb": 128.6}}
{"ts": ..., "type": "apply",  "game": "snake", "page": "x7k2", "seq": 41,
 "action_applied_ts": 1790443000119, "obs_to_action_ms": 119.0, "postqueue_ms": 72.0,
 "wait_ms": 5.0, "co_resident_load_hint": {...}}
{"ts": ..., "type": "adapter", "game": "unknown", "adapter_req_ts": ..., "adapter_ms": 48.9,
 "gate_inference_ms": 45.2, "status": 200, "n_questions": 3, "n_options": 0, ...}
{"ts": ..., "type": "composite", "game": "paddle", "page": "multiview", "cell_ms": 12.3,
 "cell_ok": true, ...}
{"ts": ..., "type": "session", "pid": 8010}
```

Adapter events carry `n_questions`/`n_options` (and `_ota` context when present) but
normally `game: "unknown"` — the browser events are the source of truth for per-game
OTA; adapter rows prove the gate-side view for the same wall-clock span (and cover
requests from un-instrumented clients).

### Hook points (all additive; the serving behavior is untouched)

| # | File | Change |
|---|---|---|
| 1 | `games/serve_showcase.py` | import + `ota_install()` in `main()`; adapter timing in `proxy_decide()`; 4 tiny routes (`/__ota__/latency_client.mjs`, `/__ota__/beacon` GET+POST, `/__ota__/health.json`) |
| 2 | `vendor/arbiter/showcase/_lib/client.mjs` | 12-line block after `JSON.parse`: emits the `req` beacon, sets `resp_epoch` |
| 3 | `vendor/arbiter/showcase/<game>/index.html` ×6 | `<script src="/__ota__/latency_client.mjs?game=X">` + `__otaNext()` in `requestNext()` + `__otaApply(now, ep.tickStart)` after `logic.step()` |
| 4 | `games/multiview/index.html` | ~25-line probe: per-cell `index.html` fetch latency once per second → `composite` events (the only compositing signal a cross-origin grid can get) |
| 5 | `games/multiview/serve_multiview.py` | **unchanged** (static file server; the probe lives in the page) |

The injected page tag carries `?game=<name>` so every beacon is self-labeled.

**Files created**: `games/latency_instrumentation.py` (all measurement logic + the
served `latency_client.mjs` source), `games/ota_analyze.py` (p50/p95/p99 report),
`games/ota_loadgen.py` (EXL3 decode load, HTTP-only), `games/ota_lab.py` (offline
end-to-end lab), `games/ota_multiview_patch.py` (idempotent page patcher),
`games/multiview/snapshot/*` (**pristine originals for diff/rollback**; note
`multiview/snapshot/` shadows `multiview/snapshots/` for gitignore purposes).
Env knobs: `DECISERV_OTA=1|0`, `DECISERV_OTA_LOG`, `DECISERV_OTA_DIR`,
`DECISERV_OTA_SLO_MS`, `DECISERV_OTA_CO_RES`, `DECISERV_OTA_GATE_READYZ`.

### Rollback (no deletes — restore from snapshot or git)

```bash
cd /Users/clawdio/deci-serv
cp games/multiview/snapshot/client.mjs.orig vendor/arbiter/showcase/_lib/client.mjs
for g in snake paddle hopper crossing mines dungeon; do
  cp games/multiview/snapshot/$g/index.html.orig vendor/arbiter/showcase/$g/index.html
done
cp games/multiview/snapshot/index.html.orig games/multiview/index.html
git checkout games/serve_showcase.py   # tracked file → git restore
```

## 3. Lab validation (already run — this is what "verified" means here)

`python3 games/ota_lab.py` — fake gate (45 ms artificial latency) + the **real**
instrumented `serve_showcase.py` Handler + headless Chromium playing the instrumented
snake page; ~35 s; exit 0. Result (67 joined decisions):

```
snake: gate p50 128.9 / p95 193.1 ms | rt p50 135.0 / p95 196.9 | ota p50 139.0 / p95 208.0
       postqueue p95 94.0 | wait p95 222.0
```

Checks it proves: wire injection reaches the page; `obs_ts`→`action_applied_ts` chain
closes; join rate ≈ calls; analyzer math (percentiles) sane; first-decision boot
artifact visible and excluded. It also caught two real bugs pre-deploy (see §7).

## 4. A/B experiment protocol — idle vs EXL3-decode

Question: *do fast decisions still arrive in time when both models share the GPU?*
Metric of "in time": per-game obs→action p95 (and the in-time share) under a tick
period with margin.

### Definitions

- **Per-game tick period X**: paddle 60 ms, hopper 100, snake 120, crossing 150,
  mines 250, dungeon 500 ms (from `meta.tick_ms`).
- **Margin m = 0.5**: "in time" for game g = `obs_to_action_ms ≤ X_g × 0.5`
  (a decision that lands within half a tick still leaves render+step headroom;
  the page holds the glide when it doesn't).
- Report both the raw p50/p95/p99 table and the in-time/late share per game.

### Phase A — baseline (decider alone, EXL3 idle)

1. Confirm idle co-residency: `curl -s http://192.168.2.185:8000/metrics | grep num_requests_running` → 0–1 background, and `curl -s http://192.168.2.185:8712/readyz` → note `pool_used_gb`.
2. Ensure the six showcase pages are open and playing LIVE (six tabs at
   `http://127.0.0.1:8010/showcase/<game>/`, or one multiview tab at :8030 — multiview
   also yields `composite` events; do NOT pause; 1× speed).
3. Note the active JSONL: `curl -s http://127.0.0.1:8010/__ota__/health.json`.
4. Run **240 s** (≈2000 decisions/game at snake rates; enough for stable p99).
5. Stop, save, and summarize:
   ```bash
   cp games/ota_latency/$(date +%Y%m%d)/ota.jsonl /tmp/ota_A_idle.jsonl
   python3 games/ota_analyze.py /tmp/ota_A_idle.jsonl
   ```

### Phase B — co-resident decode load (A/B difference)

The load generator streams `chat.completions` requests at the EXL3 vLLM server —
**HTTP traffic only**; it never restarts, kills, or reconfigures anything (and never
touches `~/venvs/*`; that path is off-limits):

```bash
python3 games/ota_loadgen.py --selftest          # 5 s sanity: tokens flow, errs 0
python3 games/ota_loadgen.py --concurrency 3 --seconds 180 \
    > /tmp/ota_loadgen_B.log 2>&1 &
```
`--concurrency 3` sustains 3 back-to-back 64-token streams ≈ continuous decode
batch occupancy (vLLM continuous batching absorbs them); the on-box
`num_requests_running ≥ 3` gauge in each `co_resident_load_hint` is the proof the
condition held during every recorded decision. Keep the six game tabs at 1× speed,
don't touch them during the window. Run **240 s**, then:

```bash
curl -s http://192.168.2.185:8000/metrics | grep -E "num_requests_(running|waiting)\{"
cp games/ota_latency/$(date +%Y%m%d)/ota.jsonl /tmp/ota_B_decode.jsonl
python3 games/ota_analyze.py /tmp/ota_A_idle.jsonl /tmp/ota_B_decode.jsonl
```

(Same-day JSONLs concatenate; analyze both files explicitly as above. For a clean
separate B file, run Phase B with `DECISERV_OTA_LOG=/tmp/...` in a restarted
showcase — or just concatenate; the analyzer handles it.)

### Sampling and environment notes

- 240 s per phase, same tabs, same speed, same machine load (don't move windows,
  don't toggle 4× speed — speed multiplies the tick rate and changes the queue regime).
- Record `num_requests_running/waiting` at phase start/end (also embedded per-decision
  in `co_resident_load_hint`).
- LAN is a known confound: both phases run from the same Mac over the same LAN path;
  on-box p50 (~15.5 ms) vs LAN (~25–50 ms) difference rides in both phases equally.
- The gate payload contract must keep `instructions` on every question (gate 500s
  otherwise) — instrumentation only counts them, never rewrites.

### Report (what to compute from the analyzer output)

For each phase × game: `gate_inference_ms` p50/p95/p99, `rt_ms` p50/p95/p99,
`obs_to_action_ms` p50/p95/p99, `postqueue_ms` p95, `wait_ms` p95, in-time share
(share of decisions with `obs_to_action_ms ≤ X_g × 0.5`), and the
`exl3_running` distribution from hints.

### Decision criterion

> **"Fast decisions still arrive in time"** ⇔ in Phase B (decode load), for every
> game, obs→action **p95 ≤ X_g × 0.5** AND in-time share ≥ 99%.
> Today's expectation (paddle, X=60, threshold 30 ms from Mac LAN) will fail on
> LAN physics alone — read paddle on-box (:8011, if the tab points there) or read
> paddle's **relative** degradation A→B (the shared-bus signal) rather than its
> absolute in-time share; the absolute criterion is meaningful for snake (120 ms)
> and slower games.
>
> If Phase B degrades obs→action p95 by >25% relative to Phase A on any game
> (or `num_requests_waiting` appears in hints), the memory bus is the shared
> bottleneck — mitigations to consider afterwards (none in scope here): move the
> gate on-box, cap EXL3 concurrency, or pin the gate to a memory-bandwidth-sparse
> schedule.
> If Phase B ≈ Phase A within noise (<10% p95 shift), the gate is effectively
> free-riding on idle memory cycles and co-residency is safe for gameplay.

## 5. Experiment commands (copy-paste, operator-run)

```bash
# 0) (operator, once) restart the two showcase servers to pick up the hooks —
#    this file's author did NOT restart anything. Example for :8010:
#    kill <8010 pid>; nohup python3 games/serve_showcase.py --port 8010 \
#        --gate http://192.168.2.185:8712 >> games/ota_showcase_8010.log 2>&1 &
#    (same for :8011 with its own gate; multiview :8030 needs no restart — its
#    page is re-served fresh; hard-reload the :8030 tab once.)
# 1) sanity:
curl -s http://127.0.0.1:8010/__ota__/health.json
# 2) open six live tabs (or multiview), 1×, model policy, shield on
# 3) Phase A (240 s), then:
cp games/ota_latency/$(date +%Y%m%d)/ota.jsonl /tmp/ota_A_idle.jsonl
# 4) Phase B load:
python3 games/ota_loadgen.py --selftest
python3 games/ota_loadgen.py --concurrency 3 --seconds 180 > /tmp/ota_loadgen_B.log 2>&1 &
#    ...start the 240 s Phase A-equivalent window while it runs, then:
cp games/ota_latency/$(date +%Y%m%d)/ota.jsonl /tmp/ota_B_decode.jsonl
python3 games/ota_analyze.py /tmp/ota_A_idle.jsonl /tmp/ota_B_decode.jsonl
```

## 6. What the parent still has to do (not done here, by design)

- Restart :8010/:8011 to load the instrumented code (one operator action; multiview
  :8030 page reload happens in the browser, no server restart needed).
- Run the two 240 s phases and file the numbers against §4's criterion.

## 7. Issues found & fixed during build (worth remembering)

- The `.local` mDNS tax: `http://larryspark.local:8000` from this Mac costs **~3.0 s
  per fresh connection** (link-local IPv6 fallback) vs **~8 ms** via `192.168.2.185`.
  This silently stalled every instrumented POST when the hint probe ran in-path.
  Rule: probe URLs use literal IPs. (Also why the lab first "failed".)
- Beacon endpoint must exist in **both** GET and POST handlers (multiview beacons are
  POST; some paths hit GET) — first cut only wired GET.
- First decision per page has a ~2.5–3 s obs→action because `boot()` probes `/readyz`
  with a 2.5 s timeout before the first request — exclude it from steady-state stats
  (analyzer notes; not an SLO violation).
- The multiview page is cross-origin to the frames: its probe can only measure cell
  load latency (fetch of the iframe document), not in-frame tick timing. That's the
  compositing-layer signal, and it's enough: a stalled 8010 shows up immediately as
  cell_ms spikes.
- vLLM's `/metrics` gives `num_requests_running/waiting` — the cheap, robust
  co-residency flag (no need to inspect Hermes internals).