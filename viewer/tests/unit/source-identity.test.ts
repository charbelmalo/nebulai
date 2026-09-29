import { describe, expect, it } from "vitest";
import { columnarize, findSourceRow, sourceUnit, transferables } from "../../src/data/columns";
import { v2Doc } from "./fixtures";

describe("source identity columns", () => {
  it("keeps full-precision source evidence beside the render buffers", () => {
    const doc = v2Doc();
    doc.points[1]!.xy = [0.1234567890123, -2.000000001];
    doc.points[1]!.confidence = 0.3333333333333;
    const c = columnarize(doc);
    const u = sourceUnit(c, 1)!;
    expect(u.xy).toEqual([0.1234567890123, -2.000000001]);
    expect(u.membership).toBe(0.3333333333333);
    expect(c.pos2[2]).not.toBe(0.1234567890123); // the Float32 render copy is lossy
  });

  it("resolves non-contiguous ids", () => {
    const doc = v2Doc();
    doc.points.forEach((p, i) => (p.id = i * 7 + 3));
    const c = columnarize(doc);
    expect(c.source.identity.ok).toBe(true);
    expect(findSourceRow(c, 17, "token_embedding", 102)).toEqual({ row: 2 });
  });

  it("flags duplicate ids and duplicate units", () => {
    const doc = v2Doc();
    doc.points[3]!.id = 0;
    const c = columnarize(doc);
    expect(c.source.identity.ok).toBe(false);
    if (!c.source.identity.ok) expect(c.source.identity.reason).toMatch(/duplicate point id 0/);

    const d2 = v2Doc();
    d2.points[3]!.unit_ref = { kind: "token_embedding", index: 100 };
    const c2 = columnarize(d2);
    expect(c2.source.identity.ok).toBe(false);
  });

  it("flags rows with no identity", () => {
    const doc = v2Doc();
    delete (doc.points[2] as { unit_ref?: unknown }).unit_ref;
    const c = columnarize(doc);
    expect(c.source.identity.ok).toBe(false);
    expect(sourceUnit(c, 2)!.unitKind).toBeNull();
  });

  it("answers missing for an unknown kind or tuple", () => {
    const c = columnarize(v2Doc());
    expect(findSourceRow(c, 0, "sae_decoder(x)", 100)).toEqual({ error: "missing" });
    expect(findSourceRow(c, 0, "token_embedding", 101)).toEqual({ error: "missing" });
  });

  it("transfers the source buffers too", () => {
    const c = columnarize(v2Doc());
    const bufs = transferables(c);
    for (const b of [c.source.pointId, c.source.xy, c.source.xyz, c.source.membership]) expect(bufs).toContain(b.buffer);
  });
});
