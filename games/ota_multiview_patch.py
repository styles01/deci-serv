#!/usr/bin/env python3
"""games/ota_multiview_patch.py — add the OTA compositing-lag probe to the multiview page.

Adds to games/multiview/index.html a ~15-line probe that once per second fetches
each game's index.html from :8010 (same request the iframe loads actually make)
and logs per-cell latency as a JSONL line. This is the compositing-layer view:
when cell load latency rises, the iframe re-render pipeline is starving, which
is exactly where multiview 'compositing' queue effects would appear.

Run:    python3 games/ota_multiview_patch.py
Undo:   the original page is snapshotted at games/multiview/snapshot/index.html.orig
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "multiview" / "index.html"
SNAP_DIR = HERE / "multiview" / "snapshot"

MARK = "/* OTA compositing-lag probe (additive; see games/LATENCY_INSTRUMENTATION.md) */"

PROBE = """
/* OTA compositing-lag probe (additive; see games/LATENCY_INSTRUMENTATION.md) */
var otaLag = {};
GAMES.forEach(function (g) { otaLag[g] = null; });
function otaProbeCell(g) {
  var t0 = performance.now();
  fetch(BASE + g + '/index.html', { cache: 'no-store' }).then(function (r) {
    return r.text().then(function (t) { return { ok: r.ok, n: t.length }; });
  }).then(function (o) {
    var ms = performance.now() - t0;
    otaLag[g] = Math.round(ms);
    if (window.__otaBeacon) window.__otaBeacon({ type: 'composite', ts: Date.now(),
      game: g, page: 'multiview', cell_ms: ms, cell_bytes: o.n, cell_ok: !!o.ok });
  }).catch(function () {
    otaLag[g] = null;
    if (window.__otaBeacon) window.__otaBeacon({ type: 'composite', ts: Date.now(),
      game: g, page: 'multiview', cell_ms: null, cell_ok: false });
  });
}
setInterval(function () { GAMES.forEach(otaProbeCell); }, 1000);
"""

TAIL = "})();"


def main() -> int:
    src = PAGE.read_text()
    if MARK in src:
        print("already patched; nothing to do")
        return 0
    snap = SNAP_DIR / "index.html.orig"
    if not snap.exists():
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text(src)
        print(f"snapshot: {snap}")
    idx = src.rindex(TAIL)
    out = src[:idx] + PROBE + src[idx:]
    PAGE.write_text(out)
    print(f"patched {PAGE} (+{len(PROBE)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())