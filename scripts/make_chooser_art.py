"""Render the root chooser's images into viewer/public/chooser/.

The chooser is a static page that fetches no model, so its previews ship with
the bundle. All of them are drawn from real release artifacts:

  hero.webp      the GPT-2 Small SAE map (the Atlas and Learn dataset)
  atlas.webp     the same map, card size
  learn.webp     a close-up of unit 0 ("numbers and quantities"), the unit the
                 lesson asks visitors to find, ringed
  research.svg   the GPT-2 position power spectrum (out/gpt2/interp/fourier.json),
                 the Research first task

    .venv/bin/python scripts/make_chooser_art.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import ImageDraw

from make_thumbs import render

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
DST = ROOT / "viewer" / "public" / "chooser"
SAE = OUT / "gpt2-small__sae__blocks.8.hook_resid_pre" / "nebulai.json"
FOURIER = OUT / "gpt2" / "interp" / "fourier.json"


def spectrum_svg(doc: dict, w: int = 480, h: int = 300) -> str:
    f = np.array(doc["freqs"], dtype=float)
    p = np.array(doc["power_mean"], dtype=float)
    keep = (f > 0) & (p > 0)
    f, p = f[keep], p[keep]
    x = np.log10(f)
    y = np.log10(p)
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    pad = 22
    pts = [(pad + xi * (w - 2 * pad), h - pad - yi * (h - 2 * pad)) for xi, yi in zip(x, y)]
    line = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
    area = f"{pad},{h - pad} {line} {w - pad},{h - pad}"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
        '<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#f2b84b" stop-opacity=".55"/>'
        '<stop offset="1" stop-color="#f2b84b" stop-opacity="0"/></linearGradient>'
        '<radialGradient id="bg" cx=".3" cy=".2" r="1"><stop offset="0" stop-color="#1a2233"/>'
        '<stop offset="1" stop-color="#0e0f12"/></radialGradient></defs>'
        f'<rect width="{w}" height="{h}" fill="url(#bg)"/>'
        + "".join(
            f'<line x1="{pad}" x2="{w - pad}" y1="{pad + i * (h - 2 * pad) / 4:.1f}" '
            f'y2="{pad + i * (h - 2 * pad) / 4:.1f}" stroke="#ffffff" stroke-opacity=".06"/>'
            for i in range(5)
        )
        + f'<polygon points="{area}" fill="url(#g)"/>'
        f'<polyline points="{line}" fill="none" stroke="#f2b84b" stroke-width="2" stroke-linejoin="round"/>'
        "</svg>"
    )


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    doc = json.loads(SAE.read_text())

    hero = render(doc, 1600, 900, 60, dot=1.6)
    hero.save(DST / "hero.webp", "WEBP", quality=78, method=6)
    atlas = render(doc, 640, 400, 20, dot=1.2)
    atlas.save(DST / "atlas.webp", "WEBP", quality=82, method=6)

    pts = doc["points"]
    xy = np.array([q["xy"] for q in pts], dtype=float)
    ext = (np.percentile(xy, 92, axis=0) - np.percentile(xy, 8, axis=0)).max()
    c = np.array(pts[0]["xy"], dtype=float)
    half = np.array([ext * 0.24, ext * 0.15])
    learn = render(doc, 640, 400, 0, window=(c - half, c + half), dot=1.4)
    d = ImageDraw.Draw(learn)
    cx, cy = 320, 200
    for r, a in ((26, 255), (40, 90)):
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(77, 141, 255, a), width=3)
    # the unit itself is unclustered (grey on the map); draw it lit, as the
    # lesson's selection shows it
    d.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=(236, 242, 255))
    learn.save(DST / "learn.webp", "WEBP", quality=82, method=6)

    (DST / "research.svg").write_text(spectrum_svg(json.loads(FOURIER.read_text())))
    for f in sorted(DST.iterdir()):
        print(f"{f.name}: {math.ceil(f.stat().st_size / 1024)} kB")


if __name__ == "__main__":
    main()
