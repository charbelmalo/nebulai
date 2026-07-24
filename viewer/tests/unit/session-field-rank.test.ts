/** Brightness is the field's loudest claim, so these pin what it is allowed to
 *  mean: monotone in output tokens (brighter is never fewer), identical for
 *  identical turns, and using the full range whatever the distribution — the
 *  three properties that let the legend say "ranked" and be telling the truth. */

import { describe, expect, it } from "vitest";
import { outputRank } from "../../src/scene/sessions/SessionFieldDriver";

const nodes = (vals: number[]) => vals.map((v) => ({ turn: { outputTokens: v } }));

describe("outputRank", () => {
  it("is monotone — brighter always means more output", () => {
    const vals = [900, 12, 4000, 300, 61, 61, 0, 128_000];
    const r = outputRank(nodes(vals));
    const byValue = vals.map((v, i) => ({ v, u: r[i]! })).sort((a, b) => a.v - b.v);
    for (let i = 1; i < byValue.length; i++) {
      expect(byValue[i]!.u).toBeGreaterThanOrEqual(byValue[i - 1]!.u);
    }
  });

  it("gives equal turns equal brightness (midrank over ties)", () => {
    const r = outputRank(nodes([50, 50, 50, 900]));
    expect(r[0]).toBe(r[1]);
    expect(r[1]).toBe(r[2]);
    expect(r[3]).toBeGreaterThan(r[0]!);
  });

  it("uses the whole range whatever the distribution", () => {
    // the two shapes a value/max ratio handles badly: a heavy tail (one huge
    // outlier drags everything else to ~0) and a near-uniform band (nothing
    // stands out at all). Rank must reach both ends on both.
    for (const vals of [
      [...Array.from({ length: 99 }, (_, i) => 400 + i), 900_000],
      Array.from({ length: 100 }, (_, i) => 1000 + i),
    ]) {
      const r = outputRank(nodes(vals));
      expect(Math.min(...r)).toBe(0);
      expect(Math.max(...r)).toBe(1);
    }
  });

  it("a run of identical values collapses to one brightness, not a spread", () => {
    // 99 turns of 400 tokens and one of 900k: the 99 are indistinguishable, so
    // they must all sit at the SAME rank rather than being fanned out to
    // manufacture a gradient the data doesn't have
    const r = outputRank(nodes([...Array.from({ length: 99 }, () => 400), 900_000]));
    const tied = new Set([...r].slice(0, 99));
    expect(tied.size).toBe(1);
    expect(r[99]).toBe(1);
  });

  it("puts the median mid-range so the gamma can do its job", () => {
    const r = [...outputRank(nodes(Array.from({ length: 501 }, (_, i) => i)))].sort(
      (a, b) => a - b,
    );
    expect(r[250]).toBeCloseTo(0.5, 6);
  });

  it("survives degenerate input", () => {
    expect([...outputRank(nodes([]))]).toEqual([]);
    expect([...outputRank(nodes([7]))]).toEqual([1]);
    // every turn identical: one flat field, no false ordering invented
    const flat = outputRank(nodes([5, 5, 5, 5]));
    expect([...flat]).toEqual([0.5, 0.5, 0.5, 0.5]);
  });
});
