import { describe, expect, it } from "vitest";
import { searchLabels } from "../../src/data/search";

const labels = [" ship", "Shipping", "SHIPYARD", "boat", "café", "ĠCAFÉ", "harbor"];
const clusterId = Int32Array.from([3, 3, 7, 7, -1, 5, 7]);

describe("searchLabels", () => {
  it("matches case-insensitively in both directions", () => {
    const r = searchLabels(labels, clusterId, "SHIP");
    expect(r).not.toBeNull();
    expect(Array.from(r!.matchIds)).toEqual([0, 1, 2]);
    expect(r!.total).toBe(3);

    const lower = searchLabels(labels, clusterId, "ship");
    expect(Array.from(lower!.matchIds)).toEqual([0, 1, 2]);
  });

  it("returns null on empty or whitespace-only queries (no query ≠ no matches)", () => {
    expect(searchLabels(labels, clusterId, "")).toBeNull();
    expect(searchLabels(labels, clusterId, "   ")).toBeNull();
  });

  it("distinguishes zero matches from no query", () => {
    const r = searchLabels(labels, clusterId, "zeppelin");
    expect(r).not.toBeNull();
    expect(r!.total).toBe(0);
    expect(r!.matchIds.length).toBe(0);
    expect(r!.byCluster.size).toBe(0);
  });

  it("groups matches by cluster id, noise under -1", () => {
    const r = searchLabels(labels, clusterId, "a")!;
    expect(r.byCluster.get(7)).toEqual([2, 3, 6]);
    expect(r.byCluster.get(-1)).toEqual([4]);
    expect(r.byCluster.get(5)).toEqual([5]);
    expect(r.total).toBe(5);
  });

  it("handles unicode (accented + BPE-marker labels)", () => {
    const r = searchLabels(labels, clusterId, "café")!;
    expect(Array.from(r.matchIds)).toEqual([4, 5]);
    const upper = searchLabels(labels, clusterId, "CAFÉ")!;
    expect(Array.from(upper.matchIds)).toEqual([4, 5]);
  });

  it("trims the query before matching", () => {
    const r = searchLabels(labels, clusterId, "  boat  ")!;
    expect(Array.from(r.matchIds)).toEqual([3]);
  });
});
