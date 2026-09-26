#!/usr/bin/env python3
"""ota_loadgen.py — EXL3 decode-load generator for the OTA A/B experiment.

Sends streaming chat.completions requests to the EXL3 daily driver
(default http://192.168.2.185:8000, vLLM serving Qwen3.8-Flash-Next) to keep
the decode engine busy while the decider gate serves the six games. HTTP
traffic only: it never restarts, kills, or reconfigures anything, and it never
touches ~/venvs/* — the server process is untouched. NOTE: use the literal IP;
*.local mDNS names cost ~3 s per fresh connection from this Mac.

Why streaming: a stream is pure decode for its whole lifetime, which is exactly
the co-residency condition we want to hold constant while measuring the gate.

Usage (see games/LATENCY_INSTRUMENTATION.md §A/B protocol):
    python3 games/ota_loadgen.py --concurrency 3 --seconds 120 --rps-per-worker 0.5
    python3 games/ota_loadgen.py --selftest          # 5 s at concurrency 1, then exit

Prints a one-line heartbeat every 5 s: streams done, total decode tokens,
tokens/s, mean time-to-first-token. Ctrl-C stops it cleanly.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "http://192.168.2.185:8000"
DEFAULT_MODEL = "qwen3.8-flash-next"

STOP = threading.Event()
STATS = {"streams": 0, "tokens": 0, "ttft": [], "errs": 0, "lock": threading.Lock()}


def one_stream(base: str, model: str, out_tokens: int, rps_gap: float, worker: int) -> None:
    """One streaming request: N decode steps of pure memory-bus load."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content":
                      "Write a steady plain-text description of a six-cell game grid, "
                      "one sentence per cell, keep going until told to stop."}],
        "max_tokens": out_tokens,
        "stream": True,
        "temperature": 0.7,
    }).encode()
    t_open = time.time()
    try:
        req = urllib.request.Request(
            base.rstrip("/") + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            first = None
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)
                    tok = delta["choices"][0]["delta"].get("content")
                except Exception:  # noqa: BLE001 — keepalive/comment frames
                    continue
                if tok:
                    if first is None:
                        first = time.time() - t_open
                    with STATS["lock"]:
                        STATS["tokens"] += 1
            if first is not None:
                with STATS["lock"]:
                    STATS["ttft"].append(first)
        with STATS["lock"]:
            STATS["streams"] += 1
    except Exception:  # noqa: BLE001 — count and keep going
        with STATS["lock"]:
            STATS["errs"] += 1
    if rps_gap > 0:
        STOP.wait(rps_gap)


def worker_loop(base: str, model: str, out_tokens: int, rps_gap: float, worker: int) -> None:
    while not STOP.is_set():
        one_stream(base, model, out_tokens, rps_gap, worker)


def heartbeat(every: float = 5.0) -> None:
    t0 = time.time()
    last_tokens = 0
    while not STOP.is_set():
        STOP.wait(every)
        with STATS["lock"]:
            tok, streams, errs = STATS["tokens"], STATS["streams"], STATS["errs"]
            ttft = list(STATS["ttft"])
        rate = (tok - last_tokens) / max(every, 1e-6)
        last_tokens = tok
        mt = sum(ttft) / len(ttft) * 1000 if ttft else 0.0
        print(f"[{time.time() - t0:6.1f}s] streams={streams} decode_tokens={tok} "
              f"({rate:.0f} tok/s) mean_ttft={mt:.0f}ms errs={errs}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="EXL3 decode-load generator (HTTP only).")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=3,
                    help="number of sustained streams (vLLM continuous batching absorbs these)")
    ap.add_argument("--max-tokens", type=int, default=64,
                    help="decode steps per stream (short = churn through batches)")
    ap.add_argument("--rps-per-worker", type=float, default=0.0,
                    help="pause between streams per worker, s (0 = back-to-back)")
    ap.add_argument("--seconds", type=int, default=0,
                    help="stop after N s (0 = run until Ctrl-C)")
    ap.add_argument("--selftest", action="store_true",
                    help="5 s at concurrency 1, then exit 0/1 by whether tokens flowed")
    args = ap.parse_args(argv)

    conc = 1 if args.selftest else args.concurrency
    seconds = 5 if args.selftest else args.seconds
    print(f"ota_loadgen -> {args.base} model={args.model} concurrency={conc} "
          f"max_tokens={args.max_tokens} seconds={seconds or 'until-Ctrl-C'}", flush=True)

    hb = threading.Thread(target=heartbeat, args=(5.0,), daemon=True)
    hb.start()
    threads = [threading.Thread(target=worker_loop,
                                args=(args.base, args.model, args.max_tokens,
                                      args.rps_per_worker, w), daemon=True)
               for w in range(conc)]
    for t in threads:
        t.start()
    try:
        time.sleep(seconds if seconds > 0 else 10 ** 9)
    except KeyboardInterrupt:
        pass
    finally:
        STOP.set()
        for t in threads:
            t.join(timeout=15)
    with STATS["lock"]:
        ok = STATS["tokens"] > 0 and STATS["errs"] == 0
    if args.selftest:
        print("selftest:", "OK (tokens flowed, no errors)" if ok else "FAILED", flush=True)
    return 0 if ok or not args.selftest else 1


if __name__ == "__main__":
    sys.exit(main())