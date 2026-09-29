import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { columnarize } from "../../src/data/columns";
import {
  buildFinding,
  exportEligibility,
  FINDING_MAX_BYTES,
  findingPin,
  parseFindingText,
  parsePin,
  pinParams,
  resolveUnit,
  validateFinding,
  verifyFinding,
  type Finding,
} from "../../src/data/finding";
import type { Dataset } from "../../src/data/loader";
import type { NebulaiDoc } from "../../src/data/schema";
import { v2Doc } from "./fixtures";

const SHA = "d".repeat(64);

function dataset(doc: NebulaiDoc, sha: string | null = SHA): Dataset {
  return {
    columns: columnarize(doc),
    hulls: [],
    parseMs: 0,
    url: "http://localhost/out/toy/nebulai.json",
    path: "toy/nebulai.json",
    bytes: 1,
    sha256: sha,
    ...(sha ? {} : { hashError: "SubtleCrypto unavailable" }),
    pinned: !!sha,
  };
}

function built(doc = v2Doc(), row = 1): Finding {
  const r = buildFinding(dataset(doc), "toy", row, { dims: 2, note: "look here" });
  if (!r.ok) throw new Error(r.reason);
  return r.finding;
}

const clone = <T>(v: T): T => JSON.parse(JSON.stringify(v));

describe("buildFinding → validate → verify", () => {
  it("round-trips through JSON and verifies against the same bytes", () => {
    const f = built();
    const text = JSON.stringify(f);
    const p = parseFindingText(text, text.length);
    expect(p.ok).toBe(true);
    if (!p.ok) return;
    expect(verifyFinding(p.finding, dataset(v2Doc()), "toy")).toEqual({ ok: true, row: 1 });
  });

  it("an edited note is still the same finding", () => {
    const f = clone(built());
    f.note = "a completely different note";
    expect(verifyFinding(f, dataset(v2Doc()), "toy").ok).toBe(true);
  });

  it("finds the unit by identity, not row order", () => {
    const f = built(v2Doc(), 4);
    const doc = v2Doc();
    doc.points.reverse();
    const r = verifyFinding(f, dataset(doc), "toy");
    expect(r).toEqual({ ok: true, row: 1 });
  });
});

describe("G05 negatives: every edited measurement is a conflict, never repaired", () => {
  const cases: [string, (f: Finding) => void, string][] = [
    ["label", (f) => (f.evidence.label = "something nicer"), "label"],
    ["layer", (f) => (f.evidence.layer = 3), "layer"],
    ["cluster", (f) => (f.evidence.cluster_id = 1), "cluster"],
    ["membership", (f) => (f.evidence.cluster_membership = 0.99), "cluster membership"],
    ["xy", (f) => (f.evidence.xy = [f.evidence.xy[0] + 1e-9, f.evidence.xy[1]]), "2-D position"],
    ["xyz", (f) => (f.evidence.xyz = [0, 0, 0.001]), "3-D position"],
    ["representation", (f) => (f.representation = "sae_decoder"), "representation"],
    ["model", (f) => (f.model.id = "gpt2"), "model"],
    ["revision", (f) => (f.model.revision = "abc123"), "model revision"],
    ["schema", (f) => (f.artifact.source_schema_version = 1), "source schema version"],
    ["metadata", (f) => (f.source_metadata = { ...f.source_metadata, namer: "hand" }), "source metadata"],
  ];
  for (const [name, edit, field] of cases) {
    it(name, () => {
      const f = clone(built());
      edit(f);
      const r = verifyFinding(f, dataset(v2Doc()), "toy");
      expect(r.ok).toBe(false);
      if (r.ok || r.code !== "conflict") throw new Error(`expected conflict, got ${JSON.stringify(r)}`);
      expect(r.conflicts).toContain(field);
    });
  }

  it("another digest is the wrong artifact, not a conflict", () => {
    const f = clone(built());
    f.artifact.sha256 = "e".repeat(64);
    expect(verifyFinding(f, dataset(v2Doc()), "toy")).toMatchObject({ ok: false, code: "wrong-artifact" });
  });

  it("another dataset id is the wrong artifact", () => {
    expect(verifyFinding(built(), dataset(v2Doc()), "other")).toMatchObject({ ok: false, code: "wrong-artifact" });
  });

  it("a unit not in the artifact is missing", () => {
    const f = clone(built());
    f.unit.index = 99999;
    expect(verifyFinding(f, dataset(v2Doc()), "toy")).toMatchObject({ ok: false, code: "missing" });
  });

  it("a duplicated identity is ambiguous and blocks export", () => {
    const doc = v2Doc();
    doc.points[2]!.id = 1;
    doc.points[2]!.unit_ref = { ...doc.points[1]!.unit_ref! };
    const ds = dataset(doc);
    expect(exportEligibility(ds).ok).toBe(false);
    expect(
      resolveUnit(ds, "toy", { datasetId: "toy", sha256: SHA, pointId: 1, unitKind: "token_embedding", unitIndex: 101 }),
    ).toMatchObject({ ok: false, code: "ambiguous" });
  });
});

describe("structure and input limits", () => {
  it("refuses oversize input before parsing it", () => {
    expect(parseFindingText("{", FINDING_MAX_BYTES + 1)).toMatchObject({ ok: false, code: "too-large" });
  });
  it("reports malformed JSON", () => {
    expect(parseFindingText("{nope", 5)).toMatchObject({ ok: false, code: "malformed" });
  });
  it("names an unsupported version and a foreign kind distinctly", () => {
    const f = clone(built()) as unknown as Record<string, unknown>;
    expect(validateFinding({ ...f, schema_version: 2 })).toMatchObject({ code: "unsupported-version" });
    expect(validateFinding({ ...f, kind: "something.else" })).toMatchObject({ code: "wrong-kind" });
  });
  it("rejects extra top-level fields, non-finite numbers and long notes", () => {
    const f = clone(built()) as unknown as Record<string, unknown>;
    expect(validateFinding({ ...f, extra: 1 })).toMatchObject({ ok: false, code: "invalid" });
    const inf = parseFindingText(JSON.stringify(f).replace(/"cluster_membership":[^,}]+/, '"cluster_membership":1e999'), 100);
    expect(inf.ok).toBe(false);
    expect(validateFinding({ ...f, note: "x".repeat(4001) })).toMatchObject({ ok: false });
    expect(validateFinding({ ...f, note: "😀".repeat(4000) }).ok).toBe(true);
  });
  it("refuses a record whose limitations were emptied", () => {
    const f = clone(built()) as unknown as Record<string, unknown>;
    expect(validateFinding({ ...f, limitations: [] }).ok).toBe(false);
  });
});

describe("export eligibility", () => {
  it("disables exact export when the bytes could not be hashed", () => {
    const e = exportEligibility(dataset(v2Doc(), null));
    expect(e.ok).toBe(false);
    if (!e.ok) expect(e.reason).toMatch(/could not be hashed/);
  });
  it("refuses a note over the limit", () => {
    const r = buildFinding(dataset(v2Doc()), "toy", 0, { dims: 2, note: "x".repeat(4001) });
    expect(r.ok).toBe(false);
  });
});

describe("pinned links", () => {
  it("round-trip identity only — no note, no evidence", () => {
    const f = built();
    const q = pinParams(findingPin(f));
    expect([...q.keys()]).toEqual(["experience", "page", "model", "view", "dims", "artifact", "point", "unit_kind", "unit_index"]);
    expect(q.toString()).not.toContain("look");
    const p = parsePin(new URLSearchParams(q.toString()));
    expect(p).toMatchObject({ ok: true, pin: findingPin(f) });
  });
  it("an ordinary link is not a pin", () => {
    expect(parsePin(new URLSearchParams("page=map&model=gpt2"))).toBeNull();
  });
  it("a partial or malformed pin is an explicit error, never an unpinned open", () => {
    const full = pinParams(findingPin(built()));
    for (const k of ["artifact", "point", "unit_kind", "unit_index", "model"]) {
      const q = new URLSearchParams(full);
      q.delete(k);
      expect(parsePin(q)).toMatchObject({ ok: false });
    }
    for (const [k, v] of [["artifact", "abc"], ["point", "-1"], ["point", "1.5"], ["unit_index", "01"], ["point", "9007199254740993"]]) {
      const q = new URLSearchParams(full);
      q.set(k!, v!);
      expect(parsePin(q)).toMatchObject({ ok: false });
    }
  });
});

// Golden: the plan's reference record, rebuilt from the real starter bytes.
const ROOT = join(__dirname, "..", "..", "..");
const STARTER = join(ROOT, "out", "gpt2-small__sae__blocks.8.hook_resid_pre", "nebulai.json");
const EXAMPLE = join(ROOT, "contracts", "finding.example.json");
describe.skipIf(!existsSync(STARTER) || !existsSync(EXAMPLE))("reference finding", () => {
  it("is reproduced exactly from the starter artifact", () => {
    const bytes = readFileSync(STARTER);
    const sha = createHash("sha256").update(bytes).digest("hex");
    const doc = JSON.parse(bytes.toString("utf8")) as NebulaiDoc;
    const ds: Dataset = { ...dataset(doc, sha), path: "gpt2-small__sae__blocks.8.hook_resid_pre/nebulai.json" };
    const row = doc.points.findIndex((p) => p.id === 0);
    const r = buildFinding(ds, "gpt2-small__sae__blocks.8.hook_resid_pre", row, { dims: 2 });
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    const example = JSON.parse(readFileSync(EXAMPLE, "utf8"));
    expect(JSON.parse(JSON.stringify(r.finding))).toEqual(example);
    expect(verifyFinding(example, ds, "gpt2-small__sae__blocks.8.hook_resid_pre")).toEqual({ ok: true, row });
  });
});
