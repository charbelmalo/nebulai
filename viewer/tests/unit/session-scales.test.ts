/** The asinh axes make the same claim the old time axis did — "the spacing
 *  changes, the readings don't" — so these pin it: monotone, exactly
 *  invertible, honest about degenerating to linear, and defined at zero (which
 *  a log axis would not be, and which is a real value here). */

import { describe, expect, it } from "vitest";
import { asinhScale, suggestK } from "../../src/scene/sessions/scales";

describe("asinhScale", () => {
  const max = 216_000;
  const s = asinhScale(max, 0.006);

  it("pins the ends and stays monotone", () => {
    expect(s.toUnit(0)).toBe(0);
    expect(s.toUnit(max)).toBeCloseTo(1, 12);
    let prev = -1;
    for (let v = 0; v <= max; v += max / 500) {
      const u = s.toUnit(v);
      expect(u).toBeGreaterThan(prev);
      prev = u;
    }
  });

  it("round-trips exactly — a tick label never lies", () => {
    for (const v of [0, 1, 37, 950, 12_345, 99_999, max]) {
      expect(s.toValue(s.toUnit(v))).toBeCloseTo(v, 6);
    }
  });

  it("is defined at zero, unlike a log axis", () => {
    expect(Number.isFinite(s.toUnit(0))).toBe(true);
    expect(s.toValue(0)).toBe(0);
  });

  it("actually gives the small values room", () => {
    // 10% of the range would sit at 0.1 on a linear axis; asinh lifts it well
    // clear of the floor, which is the entire point
    expect(s.toUnit(max * 0.1)).toBeGreaterThan(0.6);
    expect(asinhScale(max, 0).toUnit(max * 0.1)).toBeCloseTo(0.1, 6);
  });

  it("reports linear honestly instead of faking a curve", () => {
    const lin = asinhScale(max, 0);
    expect(lin.curved).toBe(false);
    expect(lin.toUnit(max / 2)).toBeCloseTo(0.5, 12);
    expect(s.curved).toBe(true);
  });

  it("degrades safely on a flat or empty axis", () => {
    for (const m of [0, -5, Number.NaN]) {
      const d = asinhScale(m, 0.01);
      expect(d.curved).toBe(false);
      expect(Number.isFinite(d.toUnit(10))).toBe(true);
      expect(Number.isFinite(d.toValue(0.5))).toBe(true);
    }
  });

  it("clamps out-of-range input rather than extrapolating", () => {
    expect(s.toUnit(-100)).toBe(0);
    expect(s.toUnit(max * 10)).toBe(1);
    expect(s.toValue(-1)).toBe(0);
    expect(s.toValue(4)).toBeCloseTo(max, 6);
  });

  it("labels ticks by inverting itself", () => {
    const t = s.ticks(5);
    expect(t).toHaveLength(5);
    expect(t[0]!.value).toBeCloseTo(0, 9);
    expect(t[4]!.value).toBeCloseTo(max, 6);
    for (const { u, value } of t) expect(s.toUnit(value)).toBeCloseTo(u, 9);
  });
});

describe("suggestK", () => {
  it("stays linear when the spread is mild", () => {
    const vals = Array.from({ length: 50 }, (_, i) => 100 + i * 4);
    expect(suggestK(vals, 296)).toBe(0);
  });

  it("curves a heavy-tailed axis so the median lands mid-axis", () => {
    // 200 small values and a few huge ones — the shape that made the old plot
    // a wall against the left face
    const vals = [...Array.from({ length: 200 }, () => 400), 90_000, 180_000];
    const k = suggestK(vals, 180_000);
    expect(k).toBeGreaterThan(0);
    expect(asinhScale(180_000, k).toUnit(400)).toBeCloseTo(0.45, 2);
  });

  it("degrades safely on empty or all-zero input", () => {
    expect(suggestK([], 100)).toBe(0);
    expect(suggestK([0, 0, 0], 0)).toBe(0);
  });

  it("puts the median wherever the caller asks", () => {
    // A 3-D axis wants the body mid-cube; a bar strip wants it low so the tall
    // bars read. Same data, same invertible mapping, different bend.
    const vals = [...Array.from({ length: 200 }, () => 400), 90_000, 180_000];
    for (const target of [0.45, 0.3, 0.2, 0.1]) {
      const k = suggestK(vals, 180_000, true, target);
      expect(asinhScale(180_000, k).toUnit(400)).toBeCloseTo(target, 2);
    }
  });

  it("stays exactly invertible at every target", () => {
    // The honesty rule the whole module rests on: moving the bend changes the
    // SPACING and nothing else, so a tick still reads its true value.
    const vals = [...Array.from({ length: 50 }, (_, i) => 1 + i), 9_000, 98_500];
    for (const target of [0.45, 0.2, 0.08]) {
      const s = asinhScale(98_500, suggestK(vals, 98_500, true, target));
      for (const v of vals) expect(s.toValue(s.toUnit(v))).toBeCloseTo(v, 6);
    }
  });

  it("returns linear rather than bending the wrong way", () => {
    // asinh only ever raises a small value's position. A median already above
    // the requested target has no solution, and the honest answer is "linear",
    // not a k that silently misses.
    const vals = Array.from({ length: 20 }, (_, i) => 90_000 + i);
    expect(suggestK(vals, 100_000, true, 0.2)).toBe(0);
  });

  it("leaves the default callers exactly where they were", () => {
    // The target parameter is additive: every existing call site passes three
    // arguments or fewer and must land on the same k it always did.
    const vals = [...Array.from({ length: 200 }, () => 400), 90_000, 180_000];
    expect(suggestK(vals, 180_000)).toBe(suggestK(vals, 180_000, false, 0.45));
    expect(suggestK(vals, 180_000, true)).toBe(suggestK(vals, 180_000, true, 0.45));
  });
});
