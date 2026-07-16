/** Shared chart theme for the Internals interp drivers (and the 2D chrome
 *  charts). Before this module every driver hand-rolled its own grid colors,
 *  axis lines, point markers, and hover-highlight logic — dozens of duplicated
 *  `[166, 173, 200, 70]` literals that drifted from the design tokens. This is
 *  the single place the chart aesthetic lives:
 *
 *   · palette      — the neon 5-stop data ramp + status colors, as [r,g,b]
 *                    tuples for deck.gl, sampled straight from tokens.ts so the
 *                    GPU side can never drift from tokens.css (guarded by
 *                    tokens-sync.test.ts) and this module can't drift from it.
 *   · grid / axis  — subtle, low-opacity DASHED strokes (req 5): a hairline
 *                    structure that never competes with the data geometry.
 *   · markers      — sharp LED/pixel indicators (req 4): diamonds/squares for
 *                    SolidPolygonLayer, plus an "active" crosshair that snaps to
 *                    the hovered plot point.
 *   · focus/dim    — hover highlighting (req 3): the focused series stays full
 *                    strength while the rest dim, computed from one alpha rule.
 *   · motion       — frame-rate-independent interpolation (req 6) so state
 *                    changes ease instead of snap-cutting.
 *
 *  Everything here is a pure function over numbers/tuples — deck.gl-shaped but
 *  deck-free — so it unit-tests without a GPU (see chart-theme.test.ts). The
 *  drivers own their data and layers; this owns how that data should LOOK. */

import { RAMP } from "../../styles/tokens";

export type RGB = [number, number, number];
export type RGBA = [number, number, number, number];
export type Vec2 = [number, number];

/** A deck.gl LineLayer datum. Grid/axis/crosshair builders emit arrays of these
 *  so a driver can drop them straight into `data` with the identity accessors
 *  `getSourcePosition: (s) => s.source` / `getTargetPosition: (s) => s.target`. */
export interface Seg {
  source: Vec2;
  target: Vec2;
}

// ── palette ────────────────────────────────────────────────────────────────
// tokens.css is the source of truth; tokens.ts mirrors it for the GPU; this
// mirrors tokens.ts for deck.gl. One chain, one direction, no free literals.

/** "#rrggbb" → [r, g, b] in 0–255 (deck.gl's color space). */
export function hexToRgb(hex: string): RGB {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

/** The neon connection ramp as 0–255 tuples (amber → orange → pink → magenta
 *  → violet). DATA ONLY — chrome uses ACCENT. Mirrors tokens.RAMP exactly. */
export const RAMP_RGB: readonly RGB[] = RAMP.map(hexToRgb);

/** Sample the ramp at t ∈ [0,1] (linear RGB lerp between stops) → [r,g,b]. */
export function rampRgb(t: number): RGB {
  const x = Math.min(Math.max(t, 0), 1) * (RAMP_RGB.length - 1);
  const i = Math.min(Math.floor(x), RAMP_RGB.length - 2);
  const f = x - i;
  const a = RAMP_RGB[i]!;
  const b = RAMP_RGB[i + 1]!;
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

/** Chrome accent (#4d8dff) — structure/crosshair/focus rings, never data. */
export const ACCENT: RGB = [77, 141, 255];
/** Warm scene-linked highlight (#f5c33b == --data-hot == ramp-0). */
export const HOT: RGB = [245, 195, 59];
/** Danger red (#ff5c7a) — the sharp LED point indicator (req 4). */
export const MARKER_HOT: RGB = [255, 92, 122];
export const SUCCESS: RGB = [62, 207, 142]; // #3ecf8e
export const WARN: RGB = [245, 177, 61]; // #f5b13d

export const TEXT_RGB: RGB = [244, 245, 247];
export const TEXT_DIM_RGB: RGB = [160, 163, 172];
export const TEXT_FAINT_RGB: RGB = [104, 108, 118];

// ── grid & axis (req 5: minimalist, low-opacity, dashed) ─────────────────────
// deck.gl alpha is 0–255. These read low on purpose: structure is a whisper.

/** Subtle dashed gridline (rgba(244,245,247,0.08) ≈ tokens --hairline). */
export const GRID_RGBA: RGBA = [244, 245, 247, 20];
/** Slightly stronger hairline (≈ --hairline-strong) for the baseline/zero axis. */
export const AXIS_RGBA: RGBA = [244, 245, 247, 41];
/** Default dash geometry for grid strokes, in plot pixels. */
export const GRID_DASH = 2;
export const GRID_GAP = 6;

// ── alpha & hover focus/dim (req 3) ──────────────────────────────────────────

/** Attach an alpha (0–1) to an [r,g,b], producing a deck.gl [r,g,b,a] (0–255). */
export function withAlpha(rgb: RGB, a: number): RGBA {
  return [rgb[0], rgb[1], rgb[2], Math.round(Math.min(Math.max(a, 0), 1) * 255)];
}

/** How far un-focused series recede when one series is focused. Low enough to
 *  clearly defer, high enough to keep context legible against deep black. */
export const DIM_ALPHA = 0.22;

/** The alpha multiplier for a series given the current hover focus. No focus →
 *  everything full strength; a focus → the focused series stays at 1 and the
 *  rest fall to DIM_ALPHA. Multiply this into a series' base alpha. */
export function seriesAlpha(hasFocus: boolean, isFocused: boolean): number {
  if (!hasFocus) return 1;
  return isFocused ? 1 : DIM_ALPHA;
}

// ── dashed strokes ───────────────────────────────────────────────────────────

/** Break the segment a→b into dash pieces for a LineLayer. `dash`/`gap` are in
 *  the same units as the endpoints (plot pixels). A degenerate (zero-length)
 *  segment yields nothing rather than a NaN-direction dash. */
export function dashedSegment(a: Vec2, b: Vec2, dash = GRID_DASH, gap = GRID_GAP): Seg[] {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy);
  if (len < 1e-6) return [];
  const ux = dx / len;
  const uy = dy / len;
  const step = Math.max(0.5, dash + gap);
  const out: Seg[] = [];
  for (let d = 0; d < len - 1e-6; d += step) {
    const e = Math.min(d + dash, len);
    out.push({ source: [a[0] + ux * d, a[1] + uy * d], target: [a[0] + ux * e, a[1] + uy * e] });
  }
  return out;
}

export interface GridSpec {
  /** Plot bounds in pixels. */
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  /** Pixel x-positions to draw vertical lines at (spanning y0→y1). */
  xs?: number[];
  /** Pixel y-positions to draw horizontal lines at (spanning x0→x1). */
  ys?: number[];
  dash?: number;
  gap?: number;
}

/** Build a full dashed grid (vertical lines at `xs`, horizontal at `ys`) as one
 *  flat Seg[] ready for a single LineLayer. Pass only `ys` for a horizontal-only
 *  grid (the common minimalist case — verticals often add noise). */
export function gridLines(spec: GridSpec): Seg[] {
  const { x0, y0, x1, y1, xs = [], ys = [], dash = GRID_DASH, gap = GRID_GAP } = spec;
  const out: Seg[] = [];
  for (const y of ys) out.push(...dashedSegment([x0, y], [x1, y], dash, gap));
  for (const x of xs) out.push(...dashedSegment([x, y0], [x, y1], dash, gap));
  return out;
}

// ── custom LED / pixel point markers (req 4) ─────────────────────────────────

export type MarkerShape = "diamond" | "square";

/** A sharp marker polygon (for SolidPolygonLayer) centered at (cx, cy) with
 *  "radius" r. Diamonds/squares read as crisp LED pixels — unlike a soft
 *  ScatterplotLayer circle they keep a hard silhouette at any zoom. */
export function markerPoly(cx: number, cy: number, r: number, shape: MarkerShape = "diamond"): Vec2[] {
  if (shape === "square") {
    return [
      [cx - r, cy - r],
      [cx + r, cy - r],
      [cx + r, cy + r],
      [cx - r, cy + r],
    ];
  }
  // diamond (45°-rotated square) — the default LED pixel look
  return [
    [cx, cy - r],
    [cx + r, cy],
    [cx, cy + r],
    [cx - r, cy],
  ];
}

/** The OUTLINE of a marker as a closed path (for a PathLayer/LineLayer): a
 *  hollow "reticle" ring that locks around the bright LED core — the target-lock
 *  look where a crisp diamond outline frames the glowing point under the cursor.
 *  Same geometry as `markerPoly`, closed back to the first vertex so the stroke
 *  completes the loop. Draw it a hair larger than the core (e.g. r * 1.7). */
export function markerRing(cx: number, cy: number, r: number, shape: MarkerShape = "diamond"): Vec2[] {
  const p = markerPoly(cx, cy, r, shape);
  return [...p, p[0]!];
}

/** Crosshair guides for the active point: a thin dashed vertical + horizontal
 *  line through (cx, cy), clipped to the plot bounds. Drop into a LineLayer
 *  colored with ACCENT for the "cursor locked on this datum" cue. */
export function crosshair(cx: number, cy: number, bounds: { x0: number; y0: number; x1: number; y1: number }, dash = 3, gap = 4): Seg[] {
  return [
    ...dashedSegment([cx, bounds.y0], [cx, bounds.y1], dash, gap),
    ...dashedSegment([bounds.x0, cy], [bounds.x1, cy], dash, gap),
  ];
}

// ── motion (req 6: physics-based, frame-rate-independent interpolation) ───────

/** Cubic ease-out — the standard chrome curve (mirrors tokens --ease-out). */
export function easeOutCubic(t: number): number {
  const x = 1 - Math.min(Math.max(t, 0), 1);
  return 1 - x * x * x;
}

/** Frame-rate-independent exponential smoothing: move `current` toward `target`
 *  by a fraction set by `lambda` (larger = snappier) over `dt` seconds. Unlike a
 *  fixed `lerp(current, target, 0.1)` this is stable across 30/60/120fps because
 *  the decay is exp(−λ·dt), not a per-frame constant. Use in a driver's frame()
 *  to animate injected/filtered data instead of snapping. */
export function damp(current: number, target: number, lambda: number, dt: number): number {
  return target + (current - target) * Math.exp(-lambda * Math.max(0, dt));
}

/** Vector form of `damp` for animating a [r,g,b] or [x,y] toward a target. */
export function dampVec<T extends number[]>(current: T, target: T, lambda: number, dt: number): T {
  const k = Math.exp(-lambda * Math.max(0, dt));
  return current.map((c, i) => target[i]! + (c - target[i]!) * k) as T;
}
