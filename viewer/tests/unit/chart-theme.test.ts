/** chart-theme.ts is the single source of the interp chart aesthetic; these
 *  guard its pure geometry/color/motion helpers so a driver migration can trust
 *  them without a GPU in the loop. */

import { describe, expect, it } from "vitest";
import {
  crosshair,
  damp,
  dampVec,
  dashedSegment,
  easeOutCubic,
  gridLines,
  hexToRgb,
  markerPoly,
  markerRing,
  RAMP_RGB,
  rampRgb,
  seriesAlpha,
  withAlpha,
} from "../../src/scene/interp/chart-theme";
import { RAMP } from "../../src/styles/tokens";

describe("palette", () => {
  it("hexToRgb parses to 0–255 tuples", () => {
    expect(hexToRgb("#000000")).toEqual([0, 0, 0]);
    expect(hexToRgb("#ffffff")).toEqual([255, 255, 255]);
    expect(hexToRgb("#4d8dff")).toEqual([77, 141, 255]);
  });

  it("RAMP_RGB mirrors tokens.RAMP exactly (no free literals)", () => {
    expect(RAMP_RGB).toHaveLength(RAMP.length);
    RAMP.forEach((hex, i) => expect(RAMP_RGB[i]).toEqual(hexToRgb(hex)));
  });

  it("rampRgb hits the exact stops at the ends and stays in gamut", () => {
    expect(rampRgb(0)).toEqual(RAMP_RGB[0]);
    expect(rampRgb(1)).toEqual(RAMP_RGB[RAMP_RGB.length - 1]);
    for (let t = 0; t <= 1.0001; t += 0.1) {
      for (const c of rampRgb(Math.min(t, 1))) {
        expect(c).toBeGreaterThanOrEqual(0);
        expect(c).toBeLessThanOrEqual(255);
      }
    }
  });

  it("withAlpha clamps and scales to 0–255", () => {
    expect(withAlpha([10, 20, 30], 1)).toEqual([10, 20, 30, 255]);
    expect(withAlpha([10, 20, 30], 0)).toEqual([10, 20, 30, 0]);
    expect(withAlpha([10, 20, 30], 2)).toEqual([10, 20, 30, 255]); // over-clamp
    expect(withAlpha([10, 20, 30], -1)).toEqual([10, 20, 30, 0]); // under-clamp
  });
});

describe("hover focus/dim", () => {
  it("no focus → everything full strength", () => {
    expect(seriesAlpha(false, false)).toBe(1);
    expect(seriesAlpha(false, true)).toBe(1);
  });
  it("with focus → focused stays, rest recede", () => {
    expect(seriesAlpha(true, true)).toBe(1);
    expect(seriesAlpha(true, false)).toBeLessThan(1);
    expect(seriesAlpha(true, false)).toBeGreaterThan(0);
  });
});

describe("dashed strokes", () => {
  it("splits a horizontal line into evenly-stepped dashes", () => {
    const segs = dashedSegment([0, 0], [20, 0], 2, 6); // step 8 over len 20 → 3 dashes
    expect(segs).toHaveLength(3);
    // first dash starts at the origin and runs `dash` long
    expect(segs[0]!.source).toEqual([0, 0]);
    expect(segs[0]!.target[0]).toBeCloseTo(2, 6);
    // dashes never overshoot the endpoint
    for (const s of segs) expect(s.target[0]).toBeLessThanOrEqual(20 + 1e-6);
  });

  it("clamps a trailing partial dash to the endpoint", () => {
    const segs = dashedSegment([0, 0], [9, 0], 2, 6); // starts at 0 and 8
    expect(segs).toHaveLength(2);
    expect(segs[1]!.source[0]).toBeCloseTo(8, 6);
    expect(segs[1]!.target[0]).toBeCloseTo(9, 6); // clamped, not 10
  });

  it("returns nothing for a degenerate (zero-length) segment", () => {
    expect(dashedSegment([5, 5], [5, 5])).toEqual([]);
  });

  it("gridLines builds horizontal-only dashed rows spanning the width", () => {
    const segs = gridLines({ x0: 0, y0: 0, x1: 100, y1: 50, ys: [10, 20, 30] });
    expect(segs.length).toBeGreaterThan(0);
    // every emitted segment is horizontal (constant y) at one of the tick rows
    for (const s of segs) {
      expect(s.source[1]).toBe(s.target[1]);
      expect([10, 20, 30]).toContain(s.source[1]);
    }
  });
});

describe("LED markers", () => {
  it("diamond has 4 vertices centered on the point", () => {
    const p = markerPoly(10, 10, 3, "diamond");
    expect(p).toHaveLength(4);
    expect(p).toContainEqual([10, 7]); // top
    expect(p).toContainEqual([13, 10]); // right
    expect(p).toContainEqual([10, 13]); // bottom
    expect(p).toContainEqual([7, 10]); // left
  });

  it("square has 4 corners at ±r", () => {
    const p = markerPoly(0, 0, 2, "square");
    expect(p).toHaveLength(4);
    expect(p).toContainEqual([-2, -2]);
    expect(p).toContainEqual([2, 2]);
  });

  it("markerRing is a closed outline (first vertex repeated at the end)", () => {
    const ring = markerRing(10, 10, 3, "diamond");
    expect(ring).toHaveLength(5); // 4 diamond vertices + closing point
    expect(ring[0]).toEqual(ring[ring.length - 1]);
    expect(ring[0]).toEqual([10, 7]); // starts at the top vertex
  });

  it("crosshair emits both a vertical and a horizontal run of dashes", () => {
    const segs = crosshair(50, 25, { x0: 0, y0: 0, x1: 100, y1: 50 });
    const vertical = segs.filter((s) => s.source[0] === 50 && s.target[0] === 50);
    const horizontal = segs.filter((s) => s.source[1] === 25 && s.target[1] === 25);
    expect(vertical.length).toBeGreaterThan(0);
    expect(horizontal.length).toBeGreaterThan(0);
  });
});

describe("motion", () => {
  it("easeOutCubic is pinned at the ends and clamps out-of-range input", () => {
    expect(easeOutCubic(0)).toBeCloseTo(0, 6);
    expect(easeOutCubic(1)).toBeCloseTo(1, 6);
    expect(easeOutCubic(-1)).toBe(0);
    expect(easeOutCubic(2)).toBe(1);
    expect(easeOutCubic(0.5)).toBeGreaterThan(0.5); // ease-OUT front-loads
  });

  it("damp leaves the value untouched at dt=0 and converges as dt grows", () => {
    expect(damp(0, 10, 5, 0)).toBeCloseTo(0, 6);
    expect(damp(0, 10, 5, 1000)).toBeCloseTo(10, 6); // effectively arrived
    const step = damp(0, 10, 5, 0.1);
    expect(step).toBeGreaterThan(0);
    expect(step).toBeLessThan(10);
  });

  it("dampVec moves each component toward its target", () => {
    const next = dampVec([0, 0, 0], [255, 0, 128], 5, 1000);
    expect(next[0]).toBeCloseTo(255, 4);
    expect(next[1]).toBeCloseTo(0, 4);
    expect(next[2]).toBeCloseTo(128, 4);
  });
});
