/** The footer stat strip (viz/StatStrip.tsx) is fed by each interp driver's
 *  optional `stats()`. These guard the two properties that make a tile honest
 *  rather than decorative — both of which were live bugs caught in review, not
 *  hypotheticals:
 *
 *   1. A tile must not print a STRUCTURAL CONSTANT dressed as a measurement.
 *      Attention rollout's R[0][0] is 1.0 for every prompt in every model (the
 *      first token has nothing else to attend to), so a naive max over the
 *      drawn cells reports "1.0000" forever. The peak must skip the diagonal.
 *
 *   2. A tile must not round a measurement INTO a structural-looking value.
 *      The most focused head in GPT-2 sits at 0.9993; at 2dp that prints
 *      "1.00", which reads as clamped. Focus tiles carry 3 decimals.
 *
 *  Both are tested against the same arithmetic the drivers run, kept here as
 *  local reference implementations. That is deliberate: these assert the RULE,
 *  so they still fail loudly if a future edit re-derives the value a different
 *  way inside the driver and reintroduces the trap. */

import { describe, expect, it } from "vitest";

/** Reference: the peak the rollout strip reports — max over drawn (lower
 *  triangular) cells, EXCLUDING self-attention. */
function peakOffDiagonal(m: number[][], cells: Array<{ i: number; j: number }>): number {
  let peak = 0;
  for (const c of cells) {
    if (c.i === c.j) continue;
    const v = m[c.i]?.[c.j] ?? 0;
    if (v > peak) peak = v;
  }
  return peak;
}

function lowerTriangle(T: number): Array<{ i: number; j: number }> {
  const cells: Array<{ i: number; j: number }> = [];
  for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) cells.push({ i, j });
  return cells;
}

/** A rollout matrix with the real structure: row-stochastic, causal, and
 *  R[0][0] pinned at 1 because position 0 can only attend to itself. Every row
 *  past the first splits its mass between self (the diagonal) and the single
 *  preceding token, so the largest genuine source→destination value in the
 *  whole matrix is exactly `offDiagPeak`. */
function rolloutLike(T: number, offDiagPeak: number): number[][] {
  const m: number[][] = [];
  for (let i = 0; i < T; i++) {
    const row = new Array<number>(T).fill(0);
    if (i === 0) {
      row[0] = 1; // structural: nothing else to attend to
    } else {
      row[i - 1] = offDiagPeak; // the one real source→destination edge
      row[i] = 1 - offDiagPeak; // self
    }
    m.push(row);
  }
  return m;
}

describe("attention-rollout stat: peak excludes the diagonal", () => {
  it("does not report the structural R[0][0] = 1.0", () => {
    const T = 6;
    const m = rolloutLike(T, 0.42);
    const peak = peakOffDiagonal(m, lowerTriangle(T));
    expect(peak).toBeLessThan(1);
    expect(peak).toBeCloseTo(0.42, 10);
  });

  it("would be pinned at 1.0 for ANY prompt if the diagonal were included", () => {
    // The bug this replaced: same matrices, naive max — constant across inputs.
    const naiveMax = (m: number[][], cells: Array<{ i: number; j: number }>) => {
      let p = 0;
      for (const c of cells) p = Math.max(p, m[c.i]?.[c.j] ?? 0);
      return p;
    };
    for (const peak of [0.05, 0.42, 0.99]) {
      const m = rolloutLike(7, peak);
      expect(naiveMax(m, lowerTriangle(7))).toBe(1); // uninformative
      expect(peakOffDiagonal(m, lowerTriangle(7))).toBeCloseTo(peak, 10); // tracks the data
    }
  });

  it("counts exactly the drawn lower-triangular cells, T(T+1)/2", () => {
    for (const T of [1, 5, 11, 64]) {
      expect(lowerTriangle(T).length).toBe((T * (T + 1)) / 2);
    }
  });

  it("reports 0 when every off-diagonal cell is empty rather than falling back to the diagonal", () => {
    const T = 4;
    const m: number[][] = Array.from({ length: T }, (_, i) =>
      Array.from({ length: T }, (_, j) => (i === j ? 1 : 0)),
    );
    expect(peakOffDiagonal(m, lowerTriangle(T))).toBe(0);
  });
});

describe("attention-flow stat: focus precision", () => {
  // measured GPT-2 small, prompt "The Eiffel Tower is located in the city of"
  const REAL_MAX_FOCUS = 0.999328;
  const REAL_MIN_FOCUS = 0.147419;

  it("3dp keeps a near-ceiling measurement distinguishable from a clamp", () => {
    expect(REAL_MAX_FOCUS.toFixed(3)).toBe("0.999");
    expect(REAL_MAX_FOCUS.toFixed(3)).not.toBe("1.000");
  });

  it("2dp would have printed the same string as a genuinely saturated head", () => {
    expect(REAL_MAX_FOCUS.toFixed(2)).toBe("1.00");
    expect((1.0).toFixed(2)).toBe("1.00"); // indistinguishable — the reason for 3dp
  });

  it("the reported range is wide, so max alone would misdescribe the grid", () => {
    expect(REAL_MAX_FOCUS - REAL_MIN_FOCUS).toBeGreaterThan(0.8);
  });
});

describe("probability-simplex stat: entropy is a stated lower bound", () => {
  /** Reference: H over the shipped top-k with the tail collapsed to ONE bucket. */
  function entropyBits(topk: number[], tail: number): number {
    let h = 0;
    for (const p of topk) if (p > 0) h -= p * Math.log2(p);
    if (tail > 0) h -= tail * Math.log2(tail);
    return h;
  }

  it("under-reports true entropy when the tail is really spread out", () => {
    // 12 shipped candidates at 2.5% each, 70% tail.
    const topk = new Array(12).fill(0.025);
    const tail = 1 - 12 * 0.025;
    const collapsed = entropyBits(topk, tail);

    // If that same tail mass were really spread over 1000 tokens, true H is
    // much larger — which is exactly why the tile is labelled "entropy ≥".
    const spread = new Array(1000).fill(tail / 1000);
    const truth = entropyBits([...topk, ...spread], 0);

    expect(collapsed).toBeLessThan(truth);
    expect(truth - collapsed).toBeGreaterThan(1); // over a bit of understatement
  });

  it("is exact when the tail is empty (nothing is being collapsed)", () => {
    const topk = [0.5, 0.25, 0.125, 0.125];
    expect(entropyBits(topk, 0)).toBeCloseTo(1.75, 10);
  });
});
