/** The attention-rollout math behind #23.
 *
 *  These assert the three properties the paper guarantees — row-stochastic,
 *  strictly causal, and identity at zero layers — plus the one encoding
 *  decision the chart makes on top of them: a single log₁₀ normalization that
 *  BOTH colour and column height read, so the two channels cannot tell
 *  different stories about one weight.
 *
 *  The driver itself is GPU-bound and cannot be exercised here. That is exactly
 *  why the math lives in its own module: it is the view's actual claim, and a
 *  claim nothing can check is a claim nothing is holding up. */

import { describe, expect, it } from "vitest";
import { causalCells, computeRollouts, decadeAt, logNorm } from "../../src/scene/interp/rollout";

/** A causal, row-stochastic attention tensor: attn[layer][head][dst][src]. */
function causalAttn(nLayer: number, H: number, T: number, seed = 1): number[][][][] {
  let s = seed;
  const rnd = () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
  return Array.from({ length: nLayer }, () =>
    Array.from({ length: H }, () =>
      Array.from({ length: T }, (_, i) => {
        const row = new Array<number>(T).fill(0);
        let sum = 0;
        for (let j = 0; j <= i; j++) {
          row[j] = rnd() + 0.05;
          sum += row[j]!;
        }
        for (let j = 0; j <= i; j++) row[j]! /= sum;
        return row;
      }),
    ),
  );
}

const rowSum = (r: Float64Array) => r.reduce((a, b) => a + b, 0);

describe("computeRollouts", () => {
  it("returns one matrix per layer", () => {
    const R = computeRollouts(causalAttn(5, 3, 6), 6, 3, 5);
    expect(R).toHaveLength(5);
    expect(R[0]).toHaveLength(6);
    expect(R[0]![0]).toHaveLength(6);
  });

  it("keeps every row a distribution, at every depth", () => {
    // Each row sums to 1 — that is what makes "how much of token i comes from
    // token j" a share rather than an unscaled score. The product runs over all
    // 12 layers, so drift here would be silent and cumulative.
    const T = 9;
    const R = computeRollouts(causalAttn(12, 4, T), T, 4, 12);
    for (const Rd of R) for (const row of Rd) expect(rowSum(row)).toBeCloseTo(1, 12);
  });

  it("stays strictly causal — no weight above the diagonal", () => {
    const T = 8;
    const R = computeRollouts(causalAttn(6, 2, T), T, 2, 6);
    for (const Rd of R) {
      for (let i = 0; i < T; i++) {
        for (let j = i + 1; j < T; j++) expect(Rd[i]![j]).toBe(0);
      }
    }
  });

  it("R_0 is Ã_0 itself — the first layer, not the identity before it", () => {
    // One layer, one head: Ã_0 = row_normalize(½·A + ½·I), and R_0 = Ã_0 · I.
    // Getting this off by one would shift the whole cascade a layer.
    const A = [[[[1, 0], [0.25, 0.75]]]];
    const R = computeRollouts(A, 2, 1, 1);
    expect(R[0]![0]![0]).toBeCloseTo(1, 12);
    // row 1: ½·[0.25, 0.75] + ½·[0,1] = [0.125, 0.875], already sums to 1
    expect(R[0]![1]![0]).toBeCloseTo(0.125, 12);
    expect(R[0]![1]![1]).toBeCloseTo(0.875, 12);
  });

  it("pins R[0][0] to 1 at every depth — a constant, not a measurement", () => {
    // The first token has nothing else to attend to, so this cell is 1.0000 for
    // every prompt and every model. The stats strip excludes the diagonal for
    // exactly this reason; the test records why that exclusion is not fussiness.
    const T = 7;
    const R = computeRollouts(causalAttn(9, 3, T), T, 3, 9);
    for (const Rd of R) expect(Rd[0]![0]).toBeCloseTo(1, 12);
  });

  it("moves mass off the diagonal as depth accumulates", () => {
    // The waterfall. Self-contribution can only fall as more layers mix, so a
    // rollout whose diagonal held steady would mean the recursion is not
    // actually compounding.
    const T = 10;
    const R = computeRollouts(causalAttn(12, 4, T), T, 4, 12);
    const diag = (d: number) => {
      let s = 0;
      for (let i = 0; i < T; i++) s += R[d]![i]![i]!;
      return s / T;
    };
    expect(diag(11)).toBeLessThan(diag(0));
    for (let d = 1; d < 12; d++) expect(diag(d)).toBeLessThanOrEqual(diag(d - 1) + 1e-12);
  });

  it("treats a missing head as zero rather than throwing", () => {
    // Bundles are fetched JSON. A short tensor should degrade to a readable
    // chart, not a stack trace that blanks the page.
    const A = causalAttn(2, 2, 4);
    A[1]![1] = undefined as unknown as number[][];
    const R = computeRollouts(A, 4, 2, 2);
    for (const row of R[1]!) expect(rowSum(row)).toBeCloseTo(1, 12);
  });
});

describe("causalCells", () => {
  it("enumerates exactly the lower triangle", () => {
    const T = 6;
    const cells = causalCells(T);
    expect(cells).toHaveLength((T * (T + 1)) / 2);
    for (const c of cells) expect(c.j).toBeLessThanOrEqual(c.i);
  });

  it("is empty for an empty sequence", () => {
    expect(causalCells(0)).toEqual([]);
  });
});

describe("logNorm", () => {
  const FLOOR = 1e-4;

  it("puts each decade at an even step, so the axis can be ticked in decades", () => {
    expect(logNorm(1e-4, FLOOR)).toBe(0);
    expect(logNorm(1e-3, FLOOR)).toBeCloseTo(0.25, 12);
    expect(logNorm(1e-2, FLOOR)).toBeCloseTo(0.5, 12);
    expect(logNorm(1e-1, FLOOR)).toBeCloseTo(0.75, 12);
    expect(logNorm(1, FLOOR)).toBeCloseTo(1, 12);
  });

  it("agrees with the tick placement it is drawn against", () => {
    // The columns and the axis labels have to come from one expression. If
    // these ever disagree, every bar is silently mislabelled by the gap.
    for (let e = -4; e <= 0; e++) {
      expect(logNorm(10 ** e, FLOOR)).toBeCloseTo(decadeAt(e, FLOOR), 12);
    }
  });

  it("pins everything at or below the floor to zero", () => {
    expect(logNorm(1e-5, FLOOR)).toBe(0);
    expect(logNorm(0, FLOOR)).toBe(0);
    expect(logNorm(-1, FLOOR)).toBe(0);
  });

  it("clamps above 1 rather than letting a column leave the cage", () => {
    expect(logNorm(2, FLOOR)).toBe(1);
    expect(logNorm(1e6, FLOOR)).toBe(1);
  });

  it("reads NaN as zero, never as tall", () => {
    // A NaN weight through a naive `v <= floor` test would fall through to the
    // log branch and produce a NaN height — which, fed to an instance matrix,
    // silently kills picking for the whole mesh.
    expect(logNorm(NaN, FLOOR)).toBe(0);
  });

  it("is monotone, so a taller column is always a larger weight", () => {
    let prev = -1;
    for (const v of [1e-4, 3e-4, 1e-3, 5e-3, 1e-2, 0.1, 0.4, 1]) {
      const t = logNorm(v, FLOOR);
      expect(t).toBeGreaterThanOrEqual(prev);
      prev = t;
    }
  });
});
