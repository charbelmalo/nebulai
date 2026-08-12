/** Attention rollout (Abnar & Zuidema, 2020) — the math, with no renderer
 *  attached.
 *
 *  It lives on its own because it is the view's actual claim. #23 draws these
 *  numbers; if the recursion is wrong, the chart is a confident picture of
 *  nothing, and a GPU-bound driver cannot be checked by a test. Every property
 *  the paper guarantees — row-stochastic, causal, monotone in depth — is a
 *  property of THIS file and is asserted against it. */

import { decadeOn, logSpan } from "@psychix/viz/logscale";

/** One causal (source, destination) pair. */
export interface RollCell {
  /** destination (row) — the token that receives information */
  i: number;
  /** source (col) — the token information flows from */
  j: number;
}

/** The lower triangle, row-major. Rollout is strictly causal, so these are
 *  exactly the pairs that can carry a non-zero weight; the rest of the T×T
 *  lattice is excluded, not measured as zero. */
export function causalCells(T: number): RollCell[] {
  const out: RollCell[] = [];
  for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) out.push({ i, j });
  return out;
}

/** R_0 … R_{L-1}, each a T×T matrix, cumulative through that layer.
 *
 *    A_l = mean_h attn[l][h]                  head-averaged, still row-stochastic
 *    Ã_l = row_normalize(½·A_l + ½·I)         half the weight to the residual stream
 *    R_d = Ã_d · Ã_{d-1} · … · Ã_0            cumulative through layer d
 *
 *  float64 throughout: the product runs over every layer, and in float32 the
 *  row sums visibly drift off 1 by the top of a 12-layer stack — which would
 *  quietly break the one invariant that makes a row readable as a
 *  distribution. */
export function computeRollouts(
  attn: number[][][][],
  T: number,
  H: number,
  nLayer: number,
): Float64Array[][] {
  const out: Float64Array[][] = [];
  let R = Array.from({ length: T }, (_, i) => {
    const r = new Float64Array(T);
    r[i] = 1;
    return r;
  });
  for (let l = 0; l < nLayer; l++) {
    const Atil = Array.from({ length: T }, () => new Float64Array(T));
    for (let i = 0; i < T; i++) {
      let s = 0;
      for (let j = 0; j < T; j++) {
        let a = 0;
        for (let h = 0; h < H; h++) a += attn[l]?.[h]?.[i]?.[j] ?? 0;
        a /= H;
        const v = 0.5 * a + (i === j ? 0.5 : 0);
        Atil[i]![j] = v;
        s += v;
      }
      if (s > 0) for (let j = 0; j < T; j++) Atil[i]![j]! /= s;
    }
    const next = Array.from({ length: T }, () => new Float64Array(T));
    for (let i = 0; i < T; i++) {
      for (let k = 0; k < T; k++) {
        const a = Atil[i]![k]!;
        if (a === 0) continue;
        const Rk = R[k]!;
        const ni = next[i]!;
        for (let j = 0; j < T; j++) ni[j]! += a * Rk[j]!;
      }
    }
    R = next;
    out.push(R.map((row) => Float64Array.from(row)));
  }
  return out;
}

/** Map a rollout weight onto 0..1 by log₁₀, with everything at or below
 *  `floor` pinned to 0.
 *
 *  This is the ONE normalization #23 has. Both colour and column height read
 *  it, so the two channels cannot tell different stories about one weight —
 *  and because it is logarithmic, the height axis is a log axis and has to be
 *  ticked in decades. Linear height would leave every weight except the
 *  attention sink flat on the floor, which is precisely the structure the view
 *  exists to show. */
export function logNorm(v: number, floor: number): number {
  return logSpan(v, Math.log10(floor), 0);
}

/** Where a decade sits on the 0..1 axis, so a tick and the column it measures
 *  are derived from the same expression rather than two that happen to agree. */
export function decadeAt(exponent: number, floor: number): number {
  return decadeOn(exponent, Math.log10(floor), 0);
}
