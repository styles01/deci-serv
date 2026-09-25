# Grid Video Pipeline — DeciServ plays Laya Arcadia (six games, tweet format)

The point: record OUR OWN replays with OUR OWN gate every time the model
improves, and render the same 2×3 1920×1080 grid video as 0xBakeer's tweet.
Re-run the whole pipeline per checkpoint — it is the model-improvement ledger.

## Pipeline v2 — HIS renderer (the right way)

0xbakeer's six-up video is his REAL showcase pages tiled, not a custom grid.
`games/serve_showcase.py` = thin adapter: serves `vendor/arbiter/showcase/`
static files + `GET /readyz` + `POST /v1/systemone` → forwards to DeciServ
`/decide` on Spark, wraps the reply in his envelope. His pages then run LIVE
on our model — his canvas boards, decision halos, panel.js telemetry bars,
HUDs, all his theme.css.

```bash
# 1. serve his showcase over our gate
python3 games/serve_showcase.py --port 8010 --gate http://192.168.2.185:8710

# 2a. watch in a browser (live, any of the six):
#     http://127.0.0.1:8010/showcase/snake/  … /hopper/ /crossing/ /paddle/ /mines/ /dungeon/

# 2b. composite video (six headless tabs → 2x3 grid → mp4)
PLAYWRIGHT_BROWSERS_PATH=/Users/clawdio/.pw-browsers \
python3 games/shoot_sixup.py --out /tmp/sixup.mp4 --seconds 16 --fps 8
```

Mac env: ~/.npm and ~/Library/Caches/ms-playwright are symlinks to an
unwritable external SSD → `npm_config_cache=/Users/clawdio/.npm-cache-local`
and `PLAYWRIGHT_BROWSERS_PATH=/Users/clawdio/.pw-browsers` (browser v1208
matches the installed python playwright 1.58 — install via
`python3 -m playwright install chromium`, NOT npx which pulls 1.63).

## Pipeline v1 — our own ASCII renderer (recorded-board truth, offline-capable)

```bash
# 1. record (Mac → Spark gate :8710; ~50s for 6 games × 1 episode)
python3 games/record_grid.py --server http://192.168.2.185:8710 \
    --out games/replays/v5 --tag v5-epoch1 --seed 13 \
    --meta-note "v5, 1 epoch, corpus v5"

# 2. frames + png + gif
python3 games/grid_replay.py --replays games/replays/v5 \
    --png /tmp/grid.png --gif /tmp/grid.gif --fps 8 \
    --title "DeciServ v5 plays Laya Arcadia — six games, one forward pass per move"

# 3. mp4 (ffmpeg at 10 fps; frames dir trick for exact control)
rm -rf /tmp/frames && mkdir -p /tmp/frames && python3 - <<'EOF'
import sys; sys.path.insert(0, "games")
from grid_replay import load_replays, replay_any, _draw_frame, GRID_W, GRID_H
from PIL import Image
raw = load_replays("games/replays/v5")
replays = {g: replay_any(g, d) for g, d in raw.items()}
n = min(max(len(r) for r in replays.values()), 300)
for ti in range(n):
    img = Image.new("RGB", (GRID_W, GRID_H), (15, 16, 20))
    _draw_frame(img, replays, ti, "DeciServ v5 plays Laya Arcadia — six games, one forward pass per move")
    img.save(f"/tmp/frames/f{ti:04d}.png")
print("frames:", n)
EOF
ffmpeg -y -framerate 10 -i /tmp/frames/f%04d.png -c:v libx264 -pix_fmt yuv420p \
    -crf 20 -movflags +faststart /tmp/deciserv_v5_grid.mp4
```

## Components

- `games/record_grid.py` — plays all 6 games against POST /decide {state,
  questions} → probabilities; logs arbiter-schema ticks (action/model_action/
  intervened/reason/events/summary/**board**) + manifest.json. The per-tick
  board snapshot is the truth the renderer draws — no re-simulation drift.
- `games/arcadia_extra.py` — paddle + dungeon ports (arbiter MIT): paddle =
  shortest state ("A paddle must be moved under a falling ball."), criteria
  carry everything (measured finding: longer states made his model worse);
  dungeon = prose rooms, same generator tables, danger score question.
- `games/grid_replay.py` — 1920×1080 renderer. `replay_any()` uses recorded
  boards when present (our files), falls back to re-sim for arbiter's
  summary-only files. Chunky pixel-tile painter (run-length rects), Menlo
  font, --title flag. GIF output also works; MP4 needs the ffmpeg step.
- Replays land in `games/replays/<tag>/` with manifest.json (scores, shield
  interventions, gate server, note).

## Facts that matter

- v4-base (laya-gate-v4-merged) game scores are near zero — expected: it is a
  safety gate, not a game agent. The grid exists to watch v5+ improve.
- Gate auto-appends its policy questions (destroys_data etc.) to every call —
  harmless, they ride along in one forward pass.
- Score questions work: dungeon `danger` returns {score, legend, probabilities}.
- Hopper physics quirk: with no usable answer it defaults to "stay", platforms
  sink 2/landing and vanish by t≈2 — looks like an empty wall. Not a renderer
  bug; the gate run fixes it by actually jumping.
- arbiter's replay files (vendor/arbiter/showcase/replay/) only carry action+
  summary per tick, no coordinates — re-sim fallback approximates his boards;
  OUR files carry boards, so ours are exact.
- 60s SSH default timeout kills interactive python on Spark; batch or background.
- Telegram delivery: write MEDIA:/tmp/deciserv_v5_grid.mp4 (25s at 10fps ≈ 250KB).