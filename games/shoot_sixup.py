#!/usr/bin/env python3
"""shoot_sixup.py — tile the six arbiter showcase pages into one 1920x1080 frame.

This is how his video is actually built: the repo's real game pages
(showcase/<game>/index.html — canvas boards, decision halos, panel.js
telemetry sidebars, theme.css) laid out 2x3 with a header bar, numbered
chips and caption chips composited on top. Playwright renders each page
LIVE against our DeciServ gate (via serve_showcase.py), screenshots each
tile, and PIL composes the grid with the same chrome as his frame.

    python3 games/shoot_sixup.py --out /tmp/sixup.png
    python3 games/shoot_sixup.py --out /tmp/sixup.mp4 --seconds 20 --fps 12
"""
from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "games"))

PORT = 8010
BASE = f"http://127.0.0.1:{PORT}"
GAMES = ["snake", "paddle", "hopper", "crossing", "mines", "dungeon"]  # his grid order
CAPTIONS = {
    "snake": None,
    "paddle": "The tick is as fast as the gate answers.",
    "hopper": None,
    "crossing": "A move and a hit-risk score, in the same call.",
    "mines": "That is 87 answers per second.",
    "dungeon": None,
}

# his theme (theme.css / board.js)
C = {
    "bg": (5, 8, 16), "chip": (26, 58, 130), "chip_tx": (219, 231, 255),
    "txt": (233, 237, 247), "dim": (125, 140, 170), "orange": (255, 140, 51),
    "cobalt": (85, 131, 255),
}


def _font(sz, italic=False):
    from PIL import ImageFont
    if italic:
        for p in ("/System/Library/Fonts/Supplemental/Arial Italic.ttf",):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    for p in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.dfont"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def compose(tiles: dict[str, "Image.Image"], title2: str | None, credit: str,
            wordmark: str) -> "Image.Image":
    from PIL import Image, ImageDraw
    W, H = 1920, 1080
    HEADER = 54
    PAD = 12
    cols, rows = 3, 2
    tw = (W - (cols + 1) * PAD) // cols
    th = (H - HEADER - (rows + 1) * PAD) // rows
    img = Image.new("RGB", (W, H), C["bg"])
    d = ImageDraw.Draw(img)
    # header bar, his style — OUR wordmark, HIS credit for the borrowed UI
    d.rectangle([24, 20, 34, 30], fill=C["orange"])
    d.text((44, 14), wordmark, fill=C["txt"], font=_font(21))
    if title2:
        w = d.textlength(title2, font=_font(19))
        d.text(((W - w) // 2, 17), title2, fill=C["dim"], font=_font(19))
    w = d.textlength(credit, font=_font(15))
    d.text((W - w - 24, 20), credit, fill=C["cobalt"], font=_font(15))
    for idx, g in enumerate(GAMES):
        col, row = idx % cols, idx // cols
        x0 = PAD + col * (tw + PAD)
        y0 = HEADER + PAD + row * (th + PAD)
        d.rounded_rectangle([x0 - 1, y0 - 1, x0 + tw + 1, y0 + th + 1],
                            radius=6, fill=(14, 21, 40), outline=(30, 40, 66))
        tile = tiles.get(g)
        if tile is not None:
            img.paste(tile.resize((tw, th), Image.LANCZOS), (x0, y0))
        # numbered chip
        label = f"{idx + 1:02d} {g}"
        cw = d.textlength(label, font=_font(13)) + 14
        d.rounded_rectangle([x0 + 2, y0 - 11, x0 + 2 + cw, y0 + 11], radius=5, fill=C["chip"])
        d.text((x0 + 9, y0 - 8), label, fill=C["chip_tx"], font=_font(13))
        cap = CAPTIONS.get(g)
        if cap:
            cwid = d.textlength(cap, font=_font(15, italic=True))
            d.text((x0 + (tw - cwid) // 2, y0 + th + 3), cap,
                   fill=(150, 162, 190), font=_font(15, italic=True))
    return img


async def shoot(seconds: float, fps: float, out_frames: Path, title2: str,
                credit: str, wordmark: str, width=1280, height=800):
    """One browser, six tabs on the real showcase pages, frame-stepped together."""
    from playwright.async_api import async_playwright

    out_frames.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        pages = {}
        for g in GAMES:
            pg = await browser.new_page(viewport={"width": width, "height": height},
                                        device_scale_factor=1)
            await pg.goto(f"{BASE}/showcase/{g}/", wait_until="domcontentloaded")
            pages[g] = pg
        await asyncio.sleep(2)  # let /readyz probes flip pages to live mode
        n_frames = int(seconds * fps)
        tick = 1.0 / fps
        for fi in range(n_frames):
            tiles = {}
            shots = await asyncio.gather(*[
                pages[g].screenshot() for g in GAMES], return_exceptions=True)
            from PIL import Image
            import io
            for g, sh in zip(GAMES, shots):
                if isinstance(sh, Exception):
                    tiles[g] = None
                else:
                    tiles[g] = Image.open(io.BytesIO(sh)).convert("RGB")
            frame = compose(tiles, title2, credit, wordmark)
            frame.save(out_frames / f"f{fi:04d}.png")
            await asyncio.sleep(max(0.0, tick - 0.0))
        await browser.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/sixup.mp4")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--keep-frames", default="/tmp/sixup_frames")
    ap.add_argument("--title2", default="Six games at once, a 421M decision model picks every move")
    ap.add_argument("--credit", default="showcase: github.com/0xBakeer/arbiter (MIT)")
    ap.add_argument("--wordmark", default="deciserv plays")
    ap.add_argument("--png-only", action="store_true")
    args = ap.parse_args(argv)

    frames = Path(args.keep_frames)
    asyncio.run(shoot(args.seconds, args.fps, frames, args.title2,
                      args.credit, args.wordmark))
    from PIL import Image
    first = sorted(frames.glob("f*.png"))
    if not first:
        print("no frames captured", file=sys.stderr)
        return 1
    png_out = str(Path(args.out).with_suffix(".png"))
    Image.open(first[-1]).save(png_out)
    print("png:", png_out)
    if not args.png_only:
        out = args.out
        cmd = ["ffmpeg", "-y", "-framerate", str(args.fps),
               "-i", str(frames / "f%04d.png"), "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", out]
        import subprocess
        subprocess.run(cmd, check=True, capture_output=True)
        print("mp4:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())