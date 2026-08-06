/** scales.ts — the session field's axis scaling.
 *
 *  A session's quantities span four or five orders of magnitude: turns 20
 *  seconds apart and turns four hours apart; cache-writes of 300 tokens and of
 *  180,000. A linear axis gives almost all of its length to the extremes and
 *  crushes the body of the data into a wall; a log axis can't show zero, which
 *  is a real and common value here (a turn that read nothing new).
 *
 *  So each axis is an **asinh** scale: linear near zero, logarithmic once the
 *  values get large, defined at exactly zero, and smooth everywhere in between.
 *  `k` sets where the bend happens — k→0 is a plain linear axis, larger k
 *  compresses the tail harder.
 *
 *  The honesty rule is the same one the old time axis held to: the mapping is
 *  monotone and has an EXACT inverse, so every tick is labelled by inverting
 *  the scale and reads a true value. Only the spacing between readings is
 *  non-uniform. Nothing is resampled, binned or smoothed, and the tooltip
 *  always shows the raw number.
 */

export interface AxisScale {
  /** value → [0,1] along the axis */
  toUnit(v: number): number;
  /** [0,1] → value; exact inverse of toUnit for every v in [0, max] */
  toValue(u: number): number;
  /** `count` evenly-spaced positions with their true values */
  ticks(count: number): { u: number; value: number }[];
  max: number;
  k: number;
  /** false when the scale degenerated to linear (k = 0, or a flat axis) */
  curved: boolean;
}

function clamp01(u: number): number {
  return u < 0 ? 0 : u > 1 ? 1 : u;
}

/** Build an asinh scale over `[0, max]`. `k` ≤ 0 (or a degenerate max) yields a
 *  plain linear scale, which reports `curved: false` so callers never claim a
 *  compression that didn't happen. */
export function asinhScale(max: number, k: number): AxisScale {
  const linear = max > 0 && k <= 0;
  const dead = !(max > 0);
  const denom = dead || linear ? 0 : Math.asinh(k * max);

  if (dead || linear || !(denom > 0)) {
    const s = dead ? 0 : 1 / max;
    return {
      toUnit: (v) => clamp01(v * s),
      toValue: (u) => (dead ? 0 : clamp01(u) * max),
      ticks: (count) => evenTicks(count, (u) => (dead ? 0 : clamp01(u) * max)),
      max,
      k: 0,
      curved: false,
    };
  }

  const toUnit = (v: number) => clamp01(Math.asinh(k * v) / denom);
  const toValue = (u: number) => Math.sinh(clamp01(u) * denom) / k;
  return { toUnit, toValue, ticks: (count) => evenTicks(count, toValue), max, k, curved: true };
}

function evenTicks(count: number, toValue: (u: number) => number) {
  const n = Math.max(2, Math.floor(count));
  const out: { u: number; value: number }[] = [];
  for (let i = 0; i < n; i++) {
    const u = i / (n - 1);
    out.push({ u, value: toValue(u) });
  }
  return out;
}

/** A sensible starting `k` for a set of values: bend the axis where the bulk of
 *  the data actually sits, so the median lands at `target` along the axis
 *  rather than pinned against zero by a handful of outliers. Returns 0 (linear)
 *  when the spread is mild enough that curving it would be theatre.
 *
 *  `force` overrides that mild-spread guard — the user explicitly asked to ease
 *  this axis, so we curve it even on a gentle distribution. The result is still
 *  monotone and exactly invertible; only the spacing tightens. A genuinely flat
 *  axis (no positive spread at all) still returns 0, because there is nothing to
 *  bend and asinhScale would report a curve that isn't visible.
 *
 *  `target` defaults to the middle-ish 0.45 a 3-D axis wants: the cloud should
 *  fill the cube, so the body of the data belongs in the middle. A **bar strip**
 *  wants the opposite — the body low, so that the few tall bars are the ones
 *  that read — and passes a smaller target. Measured over the 17 real sessions
 *  here with ≥100 steps, 0.45 put 80% of the pace strip's cells inside an
 *  11-point band (a flat wall at half height) and used 7.4 of 10 height deciles;
 *  0.2 uses 8.9 and leaves the tail visible headroom. Only the spacing changes —
 *  the mapping stays monotone and exactly invertible at any target. */
export function suggestK(values: number[], max: number, force = false, target = 0.45): number {
  if (!(max > 0) || values.length === 0) return 0;
  const positive = values.filter((v) => v > 0).sort((a, b) => a - b);
  if (positive.length === 0) return 0;
  const median = positive[positive.length >> 1] ?? 0;
  if (!(median > 0)) return 0;
  // a mild spread (top value within ~8× the median) reads fine linearly
  if (!force && max / median < 8) return 0;
  // nothing to bend if the whole axis is one value
  if (!(max > median)) return 0;
  // asinh only ever RAISES a small value's position relative to linear, so a
  // median already at or above the target has no root to find — the bisection
  // would walk to its lower bracket and return a k indistinguishable from
  // linear. Say so directly. (Unreachable at the default target, where the
  // mild-spread guard above already covers med/max ≥ 0.125.)
  if (median / max >= target) return 0;
  // Place the median at ~`target` of the axis: solve asinh(k·med)/asinh(k·max)=target.
  // The ratio rises monotonically from med/max (as k→0, the linear limit) to 1
  // (as k→∞, where both terms go logarithmic), so a geometric bisection over a
  // deliberately wide bracket always straddles the root. The bracket is scaled
  // by 1/max because k only ever appears multiplied by a value.
  let lo = 1e-9 / max;
  let hi = 1e15 / max;
  for (let i = 0; i < 200; i++) {
    const mid = Math.sqrt(lo * hi);
    const frac = Math.asinh(mid * median) / Math.asinh(mid * max);
    if (frac > target) hi = mid;
    else lo = mid;
  }
  return Math.sqrt(lo * hi);
}
