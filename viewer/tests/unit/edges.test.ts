import { describe, expect, it } from "vitest";
import type { EdgeColumns } from "../../src/data/columns";
import { clusterDegrees, clusterNeighbors, formatCount, knnNeighbors } from "../../src/data/edges";

function makeEdges(overrides: Partial<EdgeColumns> = {}): EdgeColumns {
  return {
    space: "umap10",
    metric: "gaussian_euclidean",
    sigma: 1,
    // edges: 0–1 (0.9), 0–2 (0.4), 1–3 (0.7)
    clusterEdges: Float32Array.from([0, 1, 0.9, 0, 2, 0.4, 1, 3, 0.7]),
    knn: {
      k: 3,
      sigma: 1,
      // point 0 → [1, 2, self], point 1 → [0, padding -1, 3]
      ids: Int32Array.from([1, 2, 0, 0, -1, 3]),
      sims: Float32Array.from([0.5, 0.8, 1.0, 0.6, 0, 0.3]),
    },
    ...overrides,
  };
}

describe("clusterNeighbors", () => {
  it("finds edges in both directions, strongest first", () => {
    const nbs = clusterNeighbors(makeEdges(), 0);
    expect(nbs).toEqual([
      { other: 1, weight: expect.closeTo(0.9, 5) },
      { other: 2, weight: expect.closeTo(0.4, 5) },
    ]);
    // cluster 1 sees 0 (as b) and 3 (as a)
    expect(clusterNeighbors(makeEdges(), 1).map((n) => n.other)).toEqual([0, 3]);
  });

  it("returns empty for an unconnected cluster", () => {
    expect(clusterNeighbors(makeEdges(), 99)).toEqual([]);
  });
});

describe("knnNeighbors", () => {
  it("reads the flat row, drops self and padding, sorts by sim", () => {
    const nbs = knnNeighbors(makeEdges(), 0);
    expect(nbs).toEqual([
      { id: 2, sim: expect.closeTo(0.8, 5) },
      { id: 1, sim: expect.closeTo(0.5, 5) },
    ]);
    expect(knnNeighbors(makeEdges(), 1)).toEqual([
      { id: 0, sim: expect.closeTo(0.6, 5) },
      { id: 3, sim: expect.closeTo(0.3, 5) },
    ]);
  });

  it("returns empty when knn was not exported", () => {
    expect(knnNeighbors(makeEdges({ knn: null }), 0)).toEqual([]);
  });
});

describe("clusterDegrees", () => {
  it("sums weights per cluster over both endpoints", () => {
    const deg = clusterDegrees(makeEdges());
    expect(deg.get(0)).toBeCloseTo(1.3, 5);
    expect(deg.get(1)).toBeCloseTo(1.6, 5);
    expect(deg.get(2)).toBeCloseTo(0.4, 5);
    expect(deg.get(3)).toBeCloseTo(0.7, 5);
  });
});

describe("formatCount", () => {
  it("matches the video badge format", () => {
    expect(formatCount(950)).toBe("950");
    expect(formatCount(4900)).toBe("4.9K");
    expect(formatCount(25000)).toBe("25K");
    expect(formatCount(1000)).toBe("1K");
    expect(formatCount(9950)).toBe("10K"); // 9.95 rounds past one decimal
    expect(formatCount(1_400_000)).toBe("1.4M");
  });
});
