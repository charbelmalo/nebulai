/** The log₁₀ axis shared by the extruded charts.
 *
 *  Two charts now draw columns whose height is a log position and label that
 *  height with decade ticks. The ticks and the columns have to come from ONE
 *  expression — two that merely agree today would drift apart silently, and
 *  every bar would then be mislabelled by the gap with nothing to catch it.
 *  These tests are what holds that. */

import { describe, expect, it } from "vitest";
import { decadeOn, logSpan } from "@psychix/viz/logscale";
import { decadeAt, logNorm } from "../../src/scene/interp/rollout";

describe("logSpan", () => {
  it("puts each decade at an even step", () => {
    // ‖x‖₂ 1 … 10000 — #8's axis on a GPT-2 trace.
    expect(logSpan(1, 0, 4)).toBe(0);
    expect(logSpan(10, 0, 4)).toBeCloseTo(0.25, 12);
    expect(logSpan(100, 0, 4)).toBeCloseTo(0.5, 12);
    expect(logSpan(1000, 0, 4)).toBeCloseTo(0.75, 12);
    expect(logSpan(10000, 0, 4)).toBe(1);
  });

  it("agrees with the tick placement it is drawn against", () => {
    for (let e = 0; e <= 4; e++) expect(logSpan(10 ** e, 0, 4)).toBeCloseTo(decadeOn(e, 0, 4), 12);
    for (let e = -4; e <= 0; e++) expect(logSpan(10 ** e, -4, 0)).toBeCloseTo(decadeOn(e, -4, 0), 12);
  });

  it("clamps at both ends rather than letting a column leave the cage", () => {
    expect(logSpan(0.01, 0, 4)).toBe(0);
    expect(logSpan(1e9, 0, 4)).toBe(1);
  });

  it("reads zero, negatives and NaN as the floor, never as tall", () => {
    // A NaN height fed to an instance matrix is singular under inversion, and
    // InstancedMesh.raycast inverts it — one NaN kills picking for the whole
    // mesh. `!(v > 0)` is the guard that catches it; `v <= 0` would not.
    expect(logSpan(0, 0, 4)).toBe(0);
    expect(logSpan(-5, 0, 4)).toBe(0);
    expect(logSpan(NaN, 0, 4)).toBe(0);
  });

  it("is monotone, so a taller column is always a larger value", () => {
    let prev = -1;
    for (const v of [1, 3, 10, 47, 100, 812, 1000, 9999, 10000]) {
      const t = logSpan(v, 0, 4);
      expect(t).toBeGreaterThanOrEqual(prev);
      prev = t;
    }
  });

  it("degrades to the floor on a collapsed or inverted span", () => {
    // A one-stage trace, or a bundle whose norms are all equal, would otherwise
    // divide by zero and hand NaN straight to the instance matrix.
    expect(logSpan(100, 2, 2)).toBe(0);
    expect(logSpan(100, 4, 0)).toBe(0);
    expect(decadeOn(3, 2, 2)).toBe(0);
  });

  it("lets a tick outside the span say so", () => {
    // Not clamped, unlike logSpan: a caller ticking past the cage needs to see
    // that it landed past the cage, not to have it stack up on the edge.
    expect(decadeOn(5, 0, 4)).toBeCloseTo(1.25, 12);
    expect(decadeOn(-1, 0, 4)).toBeCloseTo(-0.25, 12);
  });
});

describe("rollout's logNorm delegates to it", () => {
  it("is exactly the [floor, 1] case", () => {
    // #23 was written first and its tests lock its behaviour; this asserts the
    // shared implementation did not change it while acquiring a second caller.
    for (const v of [1e-6, 1e-4, 5e-4, 1e-3, 0.017, 0.5, 1, 2, NaN]) {
      expect(logNorm(v, 1e-4)).toBe(logSpan(v, -4, 0));
    }
    for (let e = -4; e <= 0; e++) expect(decadeAt(e, 1e-4)).toBeCloseTo(decadeOn(e, -4, 0), 12);
  });
});
