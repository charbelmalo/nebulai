/** envelope.ts — Attractors P3: the fan, as geometry.
 *
 *  The ensemble panel draws the backend's `fan` (one scalar metric per step)
 *  as a 2-D band. This module draws the same idea over the field's own three
 *  axes: at each step of a run, where did the runs of the ensemble actually
 *  sit, and how far apart were they.
 *
 *  Four rules, and each one is a refusal:
 *
 *  1. **Per-step boxes, never a tube.** A smooth surface lofted between steps
 *     interpolates a quantile surface that was never measured — the reader sees
 *     a continuous body and believes the spread between step 4 and step 5 was
 *     observed. It was not: two independent quantiles were, one at each step.
 *     So the envelope is a stack of axis-aligned wireframe boxes, one per step,
 *     with visible gaps between them.
 *
 *  2. **Only the runs that REACHED a step contribute to it** — the same rule as
 *     `fan_over()` in `src/nebulai/seer/ensemble.py`. A run that ended at step 6
 *     is not a run that sat still at step 7, so it is absent from step 7 rather
 *     than extended. `n` falls along the fan and is carried on every step so
 *     whatever draws can say so.
 *
 *  3. **No box under two runs.** A quantile of one value is that value, so a
 *     one-run step would draw a zero-width box — a picture of perfect agreement
 *     produced by having nothing to disagree with. Those steps keep their point
 *     on the median line and get no band at all.
 *
 *  4. **The quantiles are the backend's.** `ENVELOPE_LO_Q` / `ENVELOPE_HI_Q`
 *     and the linear-interpolated `quantile()` below reproduce
 *     `np.quantile(..., method="linear")` exactly, which is what the backend
 *     calls. The panel and the field must not disagree about what p10 means.
 *
 *  Deliberately pure: no THREE, no driver state. The driver feeds it live
 *  positions (so the persona cross-fade is tracked for free) and gets vertex
 *  arrays back.
 */

/** Envelope quantiles — verbatim from `seer/ensemble.py` (`ENVELOPE_LO_Q`,
 *  `ENVELOPE_HI_Q`), whose `fan_envelope` field names them `"p10_p90"`. */
export const ENVELOPE_LO_Q = 0.1;
export const ENVELOPE_HI_Q = 0.9;

/** Fewer runs than this and there is no fan at all — mirrors the backend's
 *  `MIN_RUNS_FOR_FAN`. Two runs have a range, not an envelope. */
export const MIN_RUNS_FOR_FAN = 3;

/** Fewer runs than this AT A STEP and that step gets no box (rule 3). */
export const MIN_RUNS_PER_STEP_FOR_BOX = 2;

export type Vec3 = [number, number, number];

/** One run's path through the field, in step order. Index 0 is the run's first
 *  step; a run is as long as it got. */
export type RunPath = Vec3[];

export interface EnvelopeStep {
  /** 0-based position in the run, NOT a turn id and NOT a timestamp. */
  step: number;
  /** how many runs reached this step */
  n: number;
  median: Vec3;
  /** per-axis p10 / p90. Equal to `median` on every axis when `banded` is
   *  false, in which case nothing should draw them. */
  lo: Vec3;
  hi: Vec3;
  /** whether this step has enough runs to show a spread (rule 3) */
  banded: boolean;
}

/** `np.quantile(values, q)` with the default linear interpolation, which is
 *  what `seer/ensemble.py#quantile` calls. NaN for an empty list, exactly as
 *  the backend returns `float("nan")` — a missing quantile is never 0.
 *
 *  The input is copied before sorting; callers reuse their arrays. */
export function quantile(values: readonly number[], q: number): number {
  if (values.length === 0) return Number.NaN;
  const a = [...values].sort((x, y) => x - y);
  if (a.length === 1) return a[0]!;
  const pos = (a.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return a[lo]!;
  return a[lo]! + (a[hi]! - a[lo]!) * (pos - lo);
}

/** Per-step median and p10/p90 box over an ensemble's paths.
 *
 *  The result is as long as the LONGEST run; every step reports the `n` that
 *  reached it (rule 2). Runs shorter than a step simply are not in it. */
export function envelopeSteps(runs: readonly RunPath[]): EnvelopeStep[] {
  const longest = runs.reduce((m, r) => Math.max(m, r.length), 0);
  const out: EnvelopeStep[] = [];
  for (let step = 0; step < longest; step++) {
    const xs: number[] = [];
    const ys: number[] = [];
    const zs: number[] = [];
    for (const r of runs) {
      const p = r[step];
      if (!p) continue; // this run did not reach this step
      xs.push(p[0]);
      ys.push(p[1]);
      zs.push(p[2]);
    }
    if (xs.length === 0) continue;
    const banded = xs.length >= MIN_RUNS_PER_STEP_FOR_BOX;
    const median: Vec3 = [quantile(xs, 0.5), quantile(ys, 0.5), quantile(zs, 0.5)];
    out.push({
      step,
      n: xs.length,
      median,
      lo: banded
        ? [quantile(xs, ENVELOPE_LO_Q), quantile(ys, ENVELOPE_LO_Q), quantile(zs, ENVELOPE_LO_Q)]
        : [...median],
      hi: banded
        ? [quantile(xs, ENVELOPE_HI_Q), quantile(ys, ENVELOPE_HI_Q), quantile(zs, ENVELOPE_HI_Q)]
        : [...median],
      banded,
    });
  }
  return out;
}

/** The 12 edges of one step's axis-aligned box, as 24 vertices (line-segment
 *  pairs). A wireframe, not a solid: a solid box at every step would occlude
 *  the motes the envelope is meant to describe. */
export function boxEdges(s: EnvelopeStep, into: number[]): void {
  const [x0, y0, z0] = s.lo;
  const [x1, y1, z1] = s.hi;
  const push = (ax: number, ay: number, az: number, bx: number, by: number, bz: number) => {
    into.push(ax, ay, az, bx, by, bz);
  };
  // four edges along X
  push(x0, y0, z0, x1, y0, z0);
  push(x0, y1, z0, x1, y1, z0);
  push(x0, y0, z1, x1, y0, z1);
  push(x0, y1, z1, x1, y1, z1);
  // four along Y
  push(x0, y0, z0, x0, y1, z0);
  push(x1, y0, z0, x1, y1, z0);
  push(x0, y0, z1, x0, y1, z1);
  push(x1, y0, z1, x1, y1, z1);
  // four along Z
  push(x0, y0, z0, x0, y0, z1);
  push(x1, y0, z0, x1, y0, z1);
  push(x0, y1, z0, x0, y1, z1);
  push(x1, y1, z0, x1, y1, z1);
}

/** Every banded step's wireframe, flattened. Steps below the two-run floor
 *  contribute NOTHING here — not a degenerate box, not a point. */
export function envelopeVertices(steps: readonly EnvelopeStep[]): Float32Array {
  const pts: number[] = [];
  for (const s of steps) if (s.banded) boxEdges(s, pts);
  return Float32Array.from(pts);
}

/** The median path as line-segment pairs (step i → step i+1).
 *
 *  Drawn thin on purpose, and drawn through EVERY step including the thin ones:
 *  the median of one run is that run, which is a true statement about where the
 *  survivor went. The missing box beside it is what says how alone it was. */
export function medianVertices(steps: readonly EnvelopeStep[]): Float32Array {
  const pts: number[] = [];
  for (let i = 1; i < steps.length; i++) {
    const a = steps[i - 1]!.median;
    const b = steps[i]!.median;
    pts.push(a[0], a[1], a[2], b[0], b[1], b[2]);
  }
  return Float32Array.from(pts);
}

/** The first step the envelope could not band, or null if every step held at
 *  least two runs. This is attrition made sayable: "the band stops at step 31
 *  because only one run got that far." */
export function firstThinStep(steps: readonly EnvelopeStep[]): number | null {
  for (const s of steps) if (!s.banded) return s.step;
  return null;
}
