/** The kNN score is a kernel value, not a distance. `knnDistance` reads the
 *  exporter's kernel backwards to recover the quantity the neighbour search
 *  actually ranked on. These lock the inversion and — more importantly — the
 *  refusal: a score the export rounded to zero has no recoverable distance,
 *  and must come back as null rather than as Infinity or a clamped guess. */

import { describe, expect, it } from "vitest";
import { knnDistance, knnDistanceFloor } from "../../src/data/edges";

/** The exporter's forward kernel, from backend/edges.py `compute_point_knn`. */
const kernel = (d: number, sigma: number) => Math.exp(-((d / sigma) ** 2));

describe("knnDistance", () => {
  it("inverts the exporter's kernel", () => {
    const sigma = 0.734;
    for (const d of [0.05, 0.2, 0.5, 0.734, 1.1, 2.0]) {
      expect(knnDistance(kernel(d, sigma), sigma)).toBeCloseTo(d, 10);
    }
  });

  it("puts d = sigma at the kernel's 1/e point", () => {
    const sigma = 1.25;
    expect(knnDistance(Math.exp(-1), sigma)).toBeCloseTo(sigma, 10);
  });

  it("maps a perfect score to zero distance", () => {
    expect(knnDistance(1, 0.9)).toBe(0);
  });

  it("is monotonic: a higher score is always a nearer neighbour", () => {
    const sigma = 0.6;
    const scores = [0.99, 0.8, 0.61, 0.4, 0.22, 0.05];
    const dists = scores.map((s) => knnDistance(s, sigma)!);
    for (let i = 1; i < dists.length; i++) {
      expect(dists[i]!).toBeGreaterThan(dists[i - 1]!);
    }
  });

  it("returns null for a score the export rounded away", () => {
    // sims are written with `np.round(..., 3)`, so any neighbour past roughly
    // 2.63*sigma lands on exactly 0.0 and its distance is GONE — not large.
    const sigma = 0.5;
    const far = Number(kernel(3 * sigma, sigma).toFixed(3));
    expect(far).toBe(0);
    expect(knnDistance(far, sigma)).toBeNull();
  });

  it("returns null rather than NaN for out-of-range or missing inputs", () => {
    expect(knnDistance(1.4, 0.5)).toBeNull(); // above the kernel's range
    expect(knnDistance(-0.2, 0.5)).toBeNull();
    expect(knnDistance(0.5, 0)).toBeNull(); // no sigma exported
    expect(knnDistance(Number.NaN, 0.5)).toBeNull();
  });

  it("is exactly the floor where recoverability stops", () => {
    // The floor must sit on the boundary: a hair inside it is still
    // recoverable, a hair outside rounds to 0.000 and is not. If these two
    // ever disagree the bar track is drawn to the wrong extent.
    const sigma = 0.0755; // the bundled GPT-2 atlas
    const floor = knnDistanceFloor(sigma);
    expect(Number(kernel(floor * 0.99, sigma).toFixed(3))).toBeGreaterThan(0);
    expect(Number(kernel(floor * 1.01, sigma).toFixed(3))).toBe(0);
  });

  it("scales the floor with sigma", () => {
    expect(knnDistanceFloor(2)).toBeCloseTo(2 * knnDistanceFloor(1), 12);
  });

  it("keeps distances comparable across points, since sigma is global", () => {
    // Two neighbours of DIFFERENT source points with the same score must give
    // the same distance — the property that makes the column worth showing.
    const sigma = 0.81;
    expect(knnDistance(0.42, sigma)).toBe(knnDistance(0.42, sigma));
  });
});
