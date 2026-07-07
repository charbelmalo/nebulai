import { describe, expect, it } from "vitest";
import { columnarize, transferables } from "../../src/data/columns";
import { schemaVersion } from "../../src/data/schema";
import { v1Doc, v2Doc } from "./fixtures";

describe("columnarize", () => {
  it("round-trips point data into typed arrays", () => {
    const c = columnarize(v1Doc());
    expect(c.count).toBe(4);
    expect(c.pos2).toHaveLength(8);
    expect(c.pos3).toHaveLength(12);
    expect([c.pos2[2], c.pos2[3]]).toEqual([1, 0]);
    expect([c.pos3[3], c.pos3[4], c.pos3[5]]).toEqual([1, 0, 0.5]);
    expect(Array.from(c.clusterId)).toEqual([0, 0, 0, -1]);
    expect(c.labels).toEqual(["tok0", "tok1", "tok2", "tok3"]);
  });

  it("quantizes confidence to 0–255", () => {
    const c = columnarize(v1Doc());
    expect(Array.from(c.confidence)).toEqual([255, 128, 64, 0]);
  });

  it("treats a doc without schema_version as v1 with no edges", () => {
    const c = columnarize(v1Doc());
    expect(c.schema).toBe(1);
    expect(c.edges).toBeNull();
    expect(schemaVersion(v1Doc())).toBe(1);
  });

  it("columnarizes v2 edges flat", () => {
    const c = columnarize(v2Doc());
    expect(c.schema).toBe(2);
    expect(c.edges).not.toBeNull();
    expect(Array.from(c.edges!.clusterEdges)).toEqual([0, 1, Math.fround(0.42)]);
    const knn = c.edges!.knn!;
    expect(knn.k).toBe(2);
    expect(knn.ids).toHaveLength(c.count * knn.k);
    expect(knn.sims).toHaveLength(c.count * knn.k);
    // row i lives at [i*k, (i+1)*k)
    expect(Array.from(knn.ids.subarray(4 * knn.k, 5 * knn.k))).toEqual([5, 3]);
  });

  it("keeps a v2 doc exported with --edges none usable (beams off)", () => {
    const doc = v2Doc();
    delete doc.edges;
    const c = columnarize(doc);
    expect(c.schema).toBe(2);
    expect(c.edges).toBeNull();
  });

  it("lists every typed-array buffer as transferable", () => {
    const c = columnarize(v2Doc());
    const bufs = transferables(c);
    expect(bufs).toContain(c.pos2.buffer);
    expect(bufs).toContain(c.pos3.buffer);
    expect(bufs).toContain(c.clusterId.buffer);
    expect(bufs).toContain(c.confidence.buffer);
    expect(bufs).toContain(c.edges!.clusterEdges.buffer);
    expect(bufs).toContain(c.edges!.knn!.ids.buffer);
    expect(bufs).toContain(c.edges!.knn!.sims.buffer);
    expect(new Set(bufs).size).toBe(bufs.length);
  });
});
