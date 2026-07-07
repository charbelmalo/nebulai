import { describe, expect, it } from "vitest";
import { computeHulls, convexHull } from "../../src/data/hulls";

describe("convexHull", () => {
  it("finds the square around interior points", () => {
    const hull = convexHull([
      [0, 0],
      [2, 0],
      [2, 2],
      [0, 2],
      [1, 1], // interior — must not appear
    ]);
    expect(hull).toHaveLength(4);
    expect(hull).not.toContainEqual([1, 1]);
  });

  it("passes tiny inputs through", () => {
    expect(convexHull([[1, 2]])).toEqual([[1, 2]]);
    expect(
      convexHull([
        [0, 0],
        [1, 1],
      ]),
    ).toHaveLength(2);
  });
});

describe("computeHulls", () => {
  const pos2 = new Float32Array([
    0, 0, 2, 0, 2, 2, 0, 2, 1, 1, // cluster 0: square + center
    9, 9, // cluster 1: single point
    -5, -5, // noise
  ]);
  const cids = new Int32Array([0, 0, 0, 0, 0, 1, -1]);

  it("is deterministic and skips noise", () => {
    const a = computeHulls(pos2, cids);
    const b = computeHulls(pos2, cids);
    expect(a.map((h) => h.clusterId)).toEqual([0, 1]);
    expect(a).toEqual(b);
  });

  it("anchors labels at the member centroid", () => {
    const [h0] = computeHulls(pos2, cids);
    expect(h0!.anchor).toEqual([1, 1]);
    expect(h0!.size).toBe(5);
    expect(h0!.ring).toHaveLength(8); // 4 hull vertices
  });
});
