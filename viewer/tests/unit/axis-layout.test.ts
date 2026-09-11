/** The axis layout, and the one arithmetic the CPU and the GPU must share.
 *
 *  `axisLayout()` builds the packed `iAxis` buffer — four floats per point,
 *  (realX, realY, nullX, nullY) — and `blendedPosition()` is the CPU mirror of
 *  the vertex node's `mix(mix(pos2, pos3, uMorph), axis, uAxis)`. Hover
 *  readouts, tooltips and lasso picking all go through the mirror, so if the
 *  two ever disagree the app points at one token and names another.
 *
 *  The rules under test are the two the module was written to guarantee:
 *  the real cloud and the ghost share ONE ruler, and a point nobody measured
 *  does not move.
 */

import { describe, expect, it } from "vitest";
import {
  axisLayout,
  blendedPosition,
  mapBounds,
  sharedRange,
} from "../../src/scene/axisLayout";

const BOUNDS = { cx: 0, cy: 0, halfW: 10, halfH: 10 };

function grid(n: number): Float32Array {
  const a = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    a[i * 2] = i;
    a[i * 2 + 1] = -i;
  }
  return a;
}

describe("mapBounds", () => {
  it("returns the centre and half-extents of the map's own footprint", () => {
    const b = mapBounds(Float32Array.from([-4, -2, 6, 8]));
    expect(b.cx).toBe(1);
    expect(b.cy).toBe(3);
    expect(b.halfW).toBe(5);
    expect(b.halfH).toBe(5);
  });

  it("never returns a zero half-extent — a one-point map still has a frame", () => {
    const b = mapBounds(Float32Array.from([3, 3]));
    expect(b.halfW).toBeGreaterThan(0);
    expect(b.halfH).toBeGreaterThan(0);
  });

  it("falls back to a unit frame when nothing is finite", () => {
    const b = mapBounds(Float32Array.from([NaN, NaN]));
    expect(b).toEqual({ cx: 0, cy: 0, halfW: 1, halfH: 1 });
  });

  it("ignores NaN coordinates rather than poisoning the extent", () => {
    const b = mapBounds(Float32Array.from([0, 0, NaN, NaN, 4, 4]));
    expect(b.cx).toBe(2);
    expect(b.halfW).toBe(2);
  });
});

describe("sharedRange", () => {
  it("spans every column handed in, over measured values only", () => {
    const r = sharedRange(Float32Array.from([1, 2, NaN]), Float32Array.from([-3, 0]))!;
    expect(r).toEqual([-3, 2]);
  });

  it("returns null when nothing at all was measured", () => {
    expect(sharedRange(Float32Array.from([NaN, NaN]), null)).toBeNull();
    expect(sharedRange()).toBeNull();
  });

  it("widens a degenerate range instead of returning a zero span", () => {
    const r = sharedRange(Float32Array.from([5, 5, 5]))!;
    expect(r[1]).toBeGreaterThan(r[0]);
  });
});

describe("axisLayout", () => {
  it("packs four floats per point", () => {
    const n = 5;
    const out = axisLayout(
      grid(n),
      Float32Array.from([0, 1, 2, 3, 4]),
      Float32Array.from([0, 0, 0, 0, 0]),
      Float32Array.from([4, 3, 2, 1, 0]),
      Float32Array.from([1, 1, 1, 1, 1]),
      BOUNDS,
    );
    expect(out.positions).toHaveLength(n * 4);
    expect(out.nMissing).toBe(0);
  });

  it("puts the real cloud and the ghost on ONE ruler", () => {
    // the real column spans [0, 10]; the null spans [4, 6]. On a shared ruler
    // the null must occupy the middle fifth of the frame. Normalised per-cloud
    // it would fill the whole frame — which is precisely the picture that makes
    // every random direction look as structured as the real one.
    const n = 3;
    const out = axisLayout(
      grid(n),
      Float32Array.from([0, 5, 10]),
      Float32Array.from([0, 0, 0]),
      Float32Array.from([4, 5, 6]),
      Float32Array.from([0, 0, 0]),
      BOUNDS,
    );
    expect(out.parRange).toEqual([0, 10]);
    const realX = [0, 1, 2].map((i) => out.positions[i * 4]!);
    const nullX = [0, 1, 2].map((i) => out.positions[i * 4 + 2]!);
    expect(realX[0]).toBeCloseTo(-10, 5);
    expect(realX[2]).toBeCloseTo(10, 5);
    const nullSpan = Math.max(...nullX) - Math.min(...nullX);
    const realSpan = Math.max(...realX) - Math.min(...realX);
    expect(nullSpan / realSpan).toBeCloseTo(0.2, 5);
  });

  it("leaves an unmeasured point exactly where the map put it, and counts it", () => {
    const pos2 = Float32Array.from([7, -7, 0, 0]);
    const out = axisLayout(
      pos2,
      Float32Array.from([NaN, 1]),
      Float32Array.from([NaN, 0]),
      Float32Array.from([0, 1]),
      Float32Array.from([0, 0]),
      { cx: 0, cy: 0, halfW: 10, halfH: 10 },
    );
    expect(out.nMissing).toBe(1);
    expect(out.positions[0]).toBe(7);
    expect(out.positions[1]).toBe(-7);
  });

  it("never slides an unmeasured point to the middle of the axis", () => {
    // 0 is a position on the axis and a confident one. A point with no
    // measurement must not land there just because 0 is the arithmetic default.
    const pos2 = Float32Array.from([3, 4]);
    const out = axisLayout(
      pos2,
      Float32Array.from([NaN]),
      Float32Array.from([NaN]),
      Float32Array.from([NaN]),
      Float32Array.from([NaN]),
      BOUNDS,
    );
    expect(out.positions[0]).toBe(3);
    expect(out.positions[1]).toBe(4);
    expect(out.positions[2]).toBe(3);
    expect(out.positions[3]).toBe(4);
    expect(out.nMissing).toBe(1);
  });

  it("counts a point measured on the real lane but not the null one as missing", () => {
    const out = axisLayout(
      Float32Array.from([1, 1]),
      Float32Array.from([0.5]),
      Float32Array.from([0.5]),
      Float32Array.from([NaN]),
      Float32Array.from([NaN]),
      BOUNDS,
    );
    expect(out.nMissing).toBe(1);
  });
});

describe("blendedPosition — the CPU mirror of the shader", () => {
  const pos2 = Float32Array.from([0, 0, 2, 2]);
  const pos3 = Float32Array.from([10, 10, 10, 20, 20, 20]);
  const axis = Float32Array.from([100, 100, -100, -100, 200, 200, -200, -200]);

  it("is the map position at morph 0 and axis 0", () => {
    expect(blendedPosition(1, pos2, pos3, 0, axis, 0)).toEqual([2, 2, 0]);
  });

  it("is the flythrough position at morph 1 and axis 0", () => {
    expect(blendedPosition(0, pos2, pos3, 1, axis, 0)).toEqual([10, 10, 10]);
  });

  it("is the axis position at axis 1, whatever the morph was", () => {
    expect(blendedPosition(0, pos2, pos3, 0, axis, 1)).toEqual([100, 100, 0]);
    expect(blendedPosition(0, pos2, pos3, 1, axis, 1)).toEqual([100, 100, 0]);
  });

  it("reads the null lane from the same packed vec4", () => {
    expect(blendedPosition(0, pos2, pos3, 0, axis, 1, "null")).toEqual([-100, -100, 0]);
  });

  it("flattens z as it travels onto the axis", () => {
    const [, , z] = blendedPosition(0, pos2, pos3, 1, axis, 0.5);
    expect(z).toBeCloseTo(5, 6);
  });

  it("blends the two ends linearly, exactly as mix() does", () => {
    const [x] = blendedPosition(0, pos2, pos3, 0, axis, 0.25);
    expect(x).toBeCloseTo(0 * 0.75 + 100 * 0.25, 5);
  });

  it("clamps out-of-range blends rather than extrapolating off the map", () => {
    expect(blendedPosition(0, pos2, pos3, -1, axis, -1)).toEqual([0, 0, 0]);
    expect(blendedPosition(0, pos2, pos3, 2, axis, 2)).toEqual([100, 100, 0]);
  });

  it("stays on the map when there is no axis buffer at all", () => {
    expect(blendedPosition(1, pos2, pos3, 0, null, 1)).toEqual([2, 2, 0]);
  });

  it("stays on the map when the axis buffer is short for this point", () => {
    const short = Float32Array.from([1, 1, 1, 1]);
    expect(blendedPosition(1, pos2, pos3, 0, short, 1)).toEqual([2, 2, 0]);
  });
});
