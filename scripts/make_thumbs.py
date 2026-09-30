"""Render a small preview image of every published map: out/<id>/thumb.webp.

The choosers (Research, Comparisons, the NebulAI root) show these beside each
export, so a visitor sees the actual shape of a map before asking for its
multi-megabyte file. Each thumbnail is drawn from the map's own 2-D layout and
cluster ids: nothing is invented, and a map without a layout gets no image.

    .venv/bin/python scripts/make_thumbs.py [--out out] [--force]

Additive: an existing thumbnail newer than its map is left alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 480, 300  # shown at 240x150 CSS px, so 2x for retina
PAD = 18
BG = np.array([0x0E, 0x0F, 0x12], dtype=np.float32) / 255
PALETTE = np.array(
    [
        [0x4D, 0x8D, 0xFF], [0x8A, 0x6C, 0xFF], [0xE0, 0x56, 0x9B], [0xF2, 0xB8, 0x4B],
        [0x3F, 0xC1, 0xB0], [0xFF, 0x7A, 0x59], [0x6F, 0xC3, 0xFF], [0xA6, 0xD9, 0x6A],
    ],
    dtype=np.float32,
) / 255
NOISE = np.array([0.42, 0.44, 0.52], dtype=np.float32)
GOLDEN = 0.61803398875


def render(
    doc: dict,
    W: int = W,
    H: int = H,
    PAD: int = PAD,
    window: tuple[np.ndarray, np.ndarray] | None = None,
    dot: float = 1.0,
) -> Image.Image | None:
    """Draw the map's 2-D layout. `window` (lo, hi) frames a region of layout
    space instead of the map's body (the chooser's close-up of one unit);
    `dot` scales the dot radius for larger canvases."""
    pts = [p for p in doc.get("points", []) if isinstance(p.get("xy"), list) and len(p["xy"]) == 2]
    if len(pts) < 3:
        return None
    xy = np.array([p["xy"] for p in pts], dtype=np.float64)
    cid = np.array([p.get("cluster_id", -1) if isinstance(p.get("cluster_id"), int) else -1 for p in pts])
    # frame the body of the map: a few far islands would otherwise shrink it to a dot
    # (a small probe map shows every point)
    q = 0 if len(xy) < 300 else 8
    if window is not None:
        lo, hi = window
    else:
        lo, hi = np.percentile(xy, q, axis=0), np.percentile(xy, 100 - q, axis=0)
        pad = (hi - lo) * (0.08 if q == 0 else 0.22)
        lo, hi = lo - pad, hi + pad
    span = np.maximum(hi - lo, 1e-9)
    # keep the map's aspect ratio inside the frame
    scale = min((W - 2 * PAD) / span[0], (H - 2 * PAD) / span[1])
    off = (np.array([W, H]) - span * scale) / 2
    px = (xy - lo) * scale + off
    px[:, 1] = H - px[:, 1]
    inside = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    px, cid = px[inside], cid[inside]

    col = np.where(
        (cid >= 0)[:, None],
        PALETTE[((np.maximum(cid, 0) * GOLDEN) % 1 * len(PALETTE)).astype(int) % len(PALETTE)],
        NOISE,
    )
    n = len(px)
    w = np.where(cid >= 0, 1.0, 0.35)
    # sparse maps get bigger dots, dense ones single pixels
    r = max(1, round((4 if n < 300 else 2 if n < 6000 else 1) * dot))

    acc = np.zeros((H, W, 3), dtype=np.float32)
    ix, iy = px[:, 0].astype(int), px[:, 1].astype(int)
    for dy in range(-r + 1, r):
        for dx in range(-r + 1, r):
            if dx * dx + dy * dy >= r * r:
                continue
            jx, jy = np.clip(ix + dx, 0, W - 1), np.clip(iy + dy, 0, H - 1)
            for c in range(3):
                np.add.at(acc[:, :, c], (jy, jx), col[:, c] * w)
    # soft glow: a small separable blur added back under the sharp points
    k = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    k /= k.sum()
    blur = acc.copy()
    for _ in range(2):
        for axis in (0, 1):
            blur = sum(np.roll(blur, s - 2, axis=axis) * k[s] for s in range(5))
    light = acc + blur * 1.5
    # auto-exposure: the brightest few percent of lit pixels reach near-white,
    # so a 35-point probe and a 50,000-point vocabulary read equally clearly
    lum = light.sum(axis=2)
    lit = lum[lum > 1e-6]
    ref = np.percentile(lit, 97) if lit.size else 1.0
    img = BG + (1 - BG) * (1 - np.exp(-light / max(ref, 1e-6) * 1.8))
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    root = Path(a.out)
    index = json.loads((root / "index.json").read_text())
    for d in index["datasets"]:
        src = root / d["path"]
        dst = src.parent / "thumb.webp"
        if not src.exists():
            print(f"skip {d['id']}: no map")
            continue
        if dst.exists() and not a.force and dst.stat().st_mtime >= src.stat().st_mtime:
            print(f"keep {d['id']}")
            continue
        img = render(json.loads(src.read_text()))
        if img is None:
            print(f"skip {d['id']}: no 2-D layout")
            continue
        img.save(dst, "WEBP", quality=82, method=6)
        print(f"wrote {dst} ({dst.stat().st_size // 1024} kB)")


if __name__ == "__main__":
    main()
