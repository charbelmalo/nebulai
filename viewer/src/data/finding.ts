/** finding.ts — v1 atlas-unit findings: one unit's source evidence, pinned to
 *  the exact artifact bytes it came from (contracts/finding.schema.json).
 *
 *  Three jobs, kept strictly apart:
 *
 *  1. **Structure** — `parseFindingText` / `validateFinding` enforce the size
 *     cap, JSON, the supported version and the schema, with finite and safe
 *     numbers. Nothing here fetches or touches app state.
 *  2. **Resolution** — the caller resolves (dataset_id, sha256) through the
 *     TRUSTED manifest only (`findArtifact` in experience.ts). The record's
 *     own `source_path` is provenance text and is never fetched.
 *  3. **Verification** — `verifyFinding` compares a record with a Dataset
 *     whose bytes were hashed on load: exact tuple lookup, then evidence,
 *     representation, model and raw metadata. An edited note is fine; an
 *     edited measurement is a conflicting record, reported, never repaired.
 *
 *  Pinned links (`findingParams` / `parsePin`) carry only the identity tuple.
 *  They contain no note and no evidence, so a link is verified by artifact and
 *  identity alone; an imported JSON file additionally has its evidence checked.
 *
 *  Pure module — unit-tested in node. */

import { findSourceRow, sourceUnit } from "./columns";
import type { Dataset } from "./loader";
import {
  classifyRepresentation,
  isRepresentation,
  parseCuration,
  REPRESENTATION_COPY,
  type Representation,
} from "./representation";
import { SHA256_RE } from "./experience";

export const FINDING_KIND = "nebulai.atlas-unit-finding";
export const FINDING_MAX_BYTES = 256 * 1024;
export const FINDING_NOTE_MAX = 4000;

export interface Finding {
  schema_version: 1;
  kind: typeof FINDING_KIND;
  artifact: {
    dataset_id: string;
    sha256: string;
    source_path: string;
    source_schema_version: 1 | 2;
  };
  model: { id: string | null; revision: string | null };
  unit: { point_id: number; kind: string; index: number };
  representation: Representation;
  evidence: {
    label: string;
    layer: number | null;
    cluster_id: number;
    cluster_membership: number;
    xy: [number, number];
    xyz: [number, number, number];
  };
  source_metadata: Record<string, unknown>;
  reopen: { experience: "atlas"; page: "map"; view: "atlas"; dimensions: 2 | 3 };
  note: string;
  limitations: string[];
}

export type FindingErrorCode =
  | "too-large"
  | "malformed"
  | "unsupported-version"
  | "wrong-kind"
  | "invalid";

export type FindingParse =
  | { ok: true; finding: Finding }
  | { ok: false; code: FindingErrorCode; message: string };

/* ── structure ─────────────────────────────────────────────────────────── */

const isObj = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);
const isSafeNonNeg = (v: unknown): v is number =>
  typeof v === "number" && Number.isSafeInteger(v) && v >= 0;
const isFiniteNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const codePoints = (s: string) => [...s].length;

function onlyKeys(o: Record<string, unknown>, keys: readonly string[], at: string): string | null {
  for (const k of keys) if (!(k in o)) return `${at} is missing "${k}"`;
  for (const k of Object.keys(o)) if (!keys.includes(k)) return `${at} has unexpected field "${k}"`;
  return null;
}

/** Every number anywhere in the value is finite (JSON.parse turns 1e999 into
 *  Infinity, which a schema check on "number" alone would accept). */
function allFinite(v: unknown, depth = 0): boolean {
  if (depth > 64) return false;
  if (typeof v === "number") return Number.isFinite(v);
  if (Array.isArray(v)) return v.every((x) => allFinite(x, depth + 1));
  if (isObj(v)) return Object.values(v).every((x) => allFinite(x, depth + 1));
  return true;
}

const TOP = [
  "schema_version",
  "kind",
  "artifact",
  "model",
  "unit",
  "representation",
  "evidence",
  "source_metadata",
  "reopen",
  "note",
  "limitations",
] as const;

export function validateFinding(raw: unknown): FindingParse {
  const bad = (message: string): FindingParse => ({ ok: false, code: "invalid", message });
  if (!isObj(raw)) return bad("a finding must be a JSON object");
  if (raw.kind !== FINDING_KIND) {
    return {
      ok: false,
      code: "wrong-kind",
      message: `not a NebulAI atlas-unit finding (kind is ${JSON.stringify(raw.kind ?? null)})`,
    };
  }
  if (raw.schema_version !== 1) {
    return {
      ok: false,
      code: "unsupported-version",
      message: `finding version ${JSON.stringify(raw.schema_version ?? null)} is not supported; this viewer reads version 1. Your file was not changed.`,
    };
  }
  const k = onlyKeys(raw, TOP, "finding");
  if (k) return bad(k);
  if (!allFinite(raw)) return bad("finding contains a non-finite number");

  const a = raw.artifact;
  if (!isObj(a)) return bad("artifact must be an object");
  const ak = onlyKeys(a, ["dataset_id", "sha256", "source_path", "source_schema_version"], "artifact");
  if (ak) return bad(ak);
  if (typeof a.dataset_id !== "string" || !a.dataset_id) return bad("artifact.dataset_id is required");
  if (typeof a.sha256 !== "string" || !SHA256_RE.test(a.sha256))
    return bad("artifact.sha256 must be a lowercase 64-character SHA-256");
  if (typeof a.source_path !== "string" || !a.source_path) return bad("artifact.source_path is required");
  if (a.source_schema_version !== 1 && a.source_schema_version !== 2)
    return bad("artifact.source_schema_version must be 1 or 2");

  const m = raw.model;
  if (!isObj(m)) return bad("model must be an object");
  const mk = onlyKeys(m, ["id", "revision"], "model");
  if (mk) return bad(mk);
  if (!(m.id === null || typeof m.id === "string")) return bad("model.id must be a string or null");
  if (!(m.revision === null || typeof m.revision === "string"))
    return bad("model.revision must be a string or null");

  const u = raw.unit;
  if (!isObj(u)) return bad("unit must be an object");
  const uk = onlyKeys(u, ["point_id", "kind", "index"], "unit");
  if (uk) return bad(uk);
  if (!isSafeNonNeg(u.point_id)) return bad("unit.point_id must be a non-negative safe integer");
  if (typeof u.kind !== "string" || !u.kind) return bad("unit.kind is required");
  if (!isSafeNonNeg(u.index)) return bad("unit.index must be a non-negative safe integer");

  if (!isRepresentation(raw.representation)) return bad("representation is not a known class");

  const e = raw.evidence;
  if (!isObj(e)) return bad("evidence must be an object");
  const ek = onlyKeys(
    e,
    ["label", "layer", "cluster_id", "cluster_membership", "xy", "xyz"],
    "evidence",
  );
  if (ek) return bad(ek);
  if (typeof e.label !== "string") return bad("evidence.label must be a string");
  if (!(e.layer === null || isSafeNonNeg(e.layer))) return bad("evidence.layer must be a non-negative integer or null");
  if (!(typeof e.cluster_id === "number" && Number.isSafeInteger(e.cluster_id) && e.cluster_id >= -1))
    return bad("evidence.cluster_id must be an integer ≥ -1");
  if (!(isFiniteNum(e.cluster_membership) && e.cluster_membership >= 0 && e.cluster_membership <= 1))
    return bad("evidence.cluster_membership must be between 0 and 1");
  if (!(Array.isArray(e.xy) && e.xy.length === 2 && e.xy.every(isFiniteNum)))
    return bad("evidence.xy must be two numbers");
  if (!(Array.isArray(e.xyz) && e.xyz.length === 3 && e.xyz.every(isFiniteNum)))
    return bad("evidence.xyz must be three numbers");

  if (!isObj(raw.source_metadata)) return bad("source_metadata must be an object");

  const r = raw.reopen;
  if (!isObj(r)) return bad("reopen must be an object");
  const rk = onlyKeys(r, ["experience", "page", "view", "dimensions"], "reopen");
  if (rk) return bad(rk);
  if (r.experience !== "atlas" || r.page !== "map" || r.view !== "atlas")
    return bad("reopen must name the plain atlas (experience atlas, page map, view atlas)");
  if (r.dimensions !== 2 && r.dimensions !== 3) return bad("reopen.dimensions must be 2 or 3");

  if (typeof raw.note !== "string") return bad("note must be a string");
  if (codePoints(raw.note) > FINDING_NOTE_MAX) return bad(`note is longer than ${FINDING_NOTE_MAX} characters`);
  if (
    !Array.isArray(raw.limitations) ||
    raw.limitations.length === 0 ||
    !raw.limitations.every((l) => typeof l === "string" && l.length > 0)
  )
    return bad("limitations must be a non-empty list of text");

  return { ok: true, finding: raw as unknown as Finding };
}

/** Size cap → JSON → structure. `byteLength` is the file's size on disk
 *  (File.size), checked before anything is parsed. */
export function parseFindingText(text: string, byteLength: number): FindingParse {
  if (byteLength > FINDING_MAX_BYTES) {
    return {
      ok: false,
      code: "too-large",
      message: `file is ${byteLength.toLocaleString("en-US")} bytes; a finding is at most 256 KiB`,
    };
  }
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    return {
      ok: false,
      code: "malformed",
      message: `not valid JSON: ${e instanceof Error ? e.message : String(e)}`,
    };
  }
  return validateFinding(raw);
}

/* ── export ────────────────────────────────────────────────────────────── */

const CURATION_NOUN: Record<Representation, string> = {
  token_embedding: "tokens",
  token_unembedding: "tokens",
  sae_decoder: "SAE directions",
  mlp_neuron: "neurons",
  external_text_embedding: "texts",
  unknown: "units",
};

/** The interpretation limits written into every record, derived from the
 *  source metadata (never hand-maintained per dataset). */
export function findingLimitations(
  meta: Record<string, unknown>,
  rep: Representation,
  revision: string | null,
): string[] {
  const out = [REPRESENTATION_COPY[rep].limit];
  const cur = parseCuration(meta.curation);
  if (cur && cur.kept < cur.total) {
    out.push(
      `First ${cur.kept} of ${cur.total} ${CURATION_NOUN[rep]}; this is a partial subset, not a representative sample.`,
    );
  }
  out.push("Cluster membership is not confidence in the truth of a label.");
  out.push(
    revision
      ? "Artifact replay is pinned; full pipeline reproduction is not recorded."
      : "Artifact replay is pinned; the model revision and full pipeline reproduction are not recorded.",
  );
  return out;
}

export function modelRevision(meta: Record<string, unknown>): string | null {
  for (const k of ["model_revision", "revision"]) {
    const v = meta[k];
    if (typeof v === "string" && v) return v;
  }
  return null;
}

export type ExportCheck = { ok: true } | { ok: false; reason: string };

/** Can this loaded dataset produce a verified v1 record at all? */
export function exportEligibility(ds: Dataset | null): ExportCheck {
  if (!ds) return { ok: false, reason: "No map is loaded." };
  if (!ds.sha256)
    return { ok: false, reason: `Exact export is unavailable: the artifact could not be hashed (${ds.hashError ?? "unknown reason"}).` };
  if (!ds.columns.source.identity.ok)
    return { ok: false, reason: `Exact export is unavailable: this map has no unique unit identity (${ds.columns.source.identity.reason}).` };
  return { ok: true };
}

export function buildFinding(
  ds: Dataset,
  datasetId: string,
  row: number,
  opts: { dims: 2 | 3; note?: string; sourcePath?: string },
): { ok: true; finding: Finding } | { ok: false; reason: string } {
  const elig = exportEligibility(ds);
  if (!elig.ok) return elig;
  const u = sourceUnit(ds.columns, row);
  if (!u || u.pointId === null || u.unitKind === null || u.unitIndex === null)
    return { ok: false, reason: "This point has no recorded source identity." };
  if (!(u.membership >= 0 && u.membership <= 1))
    return { ok: false, reason: "This point's cluster membership is outside 0–1." };
  const note = opts.note ?? "";
  if (codePoints(note) > FINDING_NOTE_MAX)
    return { ok: false, reason: `Notes are limited to ${FINDING_NOTE_MAX} characters.` };
  const meta = ds.columns.meta as Record<string, unknown>;
  const rep = classifyRepresentation(meta);
  const revision = modelRevision(meta);
  const finding: Finding = {
    schema_version: 1,
    kind: FINDING_KIND,
    artifact: {
      dataset_id: datasetId,
      sha256: ds.sha256!,
      source_path: opts.sourcePath ?? ds.path,
      source_schema_version: ds.columns.schema,
    },
    model: { id: typeof meta.model === "string" ? meta.model : null, revision },
    unit: { point_id: u.pointId, kind: u.unitKind, index: u.unitIndex },
    representation: rep,
    evidence: {
      label: u.label,
      layer: u.layer,
      cluster_id: u.clusterId,
      cluster_membership: u.membership,
      xy: u.xy,
      xyz: u.xyz,
    },
    source_metadata: meta,
    reopen: { experience: "atlas", page: "map", view: "atlas", dimensions: opts.dims },
    note,
    limitations: findingLimitations(meta, rep, revision),
  };
  return { ok: true, finding };
}

export function findingFileName(f: Finding): string {
  const safe = f.artifact.dataset_id.replace(/[^A-Za-z0-9._-]+/g, "_");
  return `nebulai-finding-${safe}-point-${f.unit.point_id}.json`;
}

/* ── verification ──────────────────────────────────────────────────────── */

/** Deterministic JSON for deep comparison (key order independent). */
export function stableStringify(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(stableStringify).join(",")}]`;
  if (isObj(v)) {
    return `{${Object.keys(v)
      .sort()
      .map((k) => `${JSON.stringify(k)}:${stableStringify(v[k])}`)
      .join(",")}}`;
  }
  return JSON.stringify(v) ?? "null";
}

export type VerifyResult =
  | { ok: true; row: number }
  | { ok: false; code: "wrong-artifact" | "missing" | "ambiguous"; message: string }
  | { ok: false; code: "conflict"; message: string; conflicts: string[] };

/** Identity only — what a pinned link can check (it carries no evidence). */
export function resolveUnit(
  ds: Dataset,
  datasetId: string,
  pin: { datasetId: string; sha256: string; pointId: number; unitKind: string; unitIndex: number },
): VerifyResult {
  if (pin.datasetId !== datasetId || ds.sha256 !== pin.sha256) {
    return {
      ok: false,
      code: "wrong-artifact",
      message: `Expected ${pin.datasetId}@${pin.sha256.slice(0, 12)}, loaded ${datasetId}@${(ds.sha256 ?? "unhashed").slice(0, 12)}.`,
    };
  }
  const hit = findSourceRow(ds.columns, pin.pointId, pin.unitKind, pin.unitIndex);
  if ("error" in hit) {
    return hit.error === "missing"
      ? { ok: false, code: "missing", message: `No unit ${pin.unitKind} #${pin.unitIndex} with point id ${pin.pointId} in this artifact.` }
      : { ok: false, code: "ambiguous", message: `More than one row matches point id ${pin.pointId}; the artifact has no unique identity for it.` };
  }
  return { ok: true, row: hit.row };
}

/** Full check of an imported record against verified bytes. */
export function verifyFinding(f: Finding, ds: Dataset, datasetId: string): VerifyResult {
  const r = resolveUnit(ds, datasetId, {
    datasetId: f.artifact.dataset_id,
    sha256: f.artifact.sha256,
    pointId: f.unit.point_id,
    unitKind: f.unit.kind,
    unitIndex: f.unit.index,
  });
  if (!r.ok) return r;
  const u = sourceUnit(ds.columns, r.row)!;
  const meta = ds.columns.meta as Record<string, unknown>;
  const conflicts: string[] = [];
  const e = f.evidence;
  if (e.label !== u.label) conflicts.push("label");
  if (e.layer !== u.layer) conflicts.push("layer");
  if (e.cluster_id !== u.clusterId) conflicts.push("cluster");
  if (e.cluster_membership !== u.membership) conflicts.push("cluster membership");
  if (e.xy[0] !== u.xy[0] || e.xy[1] !== u.xy[1]) conflicts.push("2-D position");
  if (e.xyz[0] !== u.xyz[0] || e.xyz[1] !== u.xyz[1] || e.xyz[2] !== u.xyz[2])
    conflicts.push("3-D position");
  if (f.representation !== classifyRepresentation(meta)) conflicts.push("representation");
  const modelId = typeof meta.model === "string" ? meta.model : null;
  if (f.model.id !== modelId) conflicts.push("model");
  if (f.model.revision !== modelRevision(meta)) conflicts.push("model revision");
  if (f.artifact.source_schema_version !== ds.columns.schema) conflicts.push("source schema version");
  if (stableStringify(f.source_metadata) !== stableStringify(meta)) conflicts.push("source metadata");
  if (conflicts.length) {
    return {
      ok: false,
      code: "conflict",
      conflicts,
      message: `This record disagrees with the verified artifact on: ${conflicts.join(", ")}. It was not applied.`,
    };
  }
  return r;
}

/* ── pinned links ──────────────────────────────────────────────────────── */

export interface UnitPin {
  datasetId: string;
  sha256: string;
  pointId: number;
  unitKind: string;
  unitIndex: number;
  dims: 2 | 3;
}

export const PIN_KEYS = ["artifact", "point", "unit_kind", "unit_index"] as const;

/** The minimal hash for a pinned unit: identity only, no note, no query. */
export function pinParams(p: UnitPin): URLSearchParams {
  const q = new URLSearchParams();
  q.set("experience", "atlas");
  q.set("page", "map");
  q.set("model", p.datasetId);
  q.set("view", "atlas");
  q.set("dims", String(p.dims));
  q.set("artifact", p.sha256);
  q.set("point", String(p.pointId));
  q.set("unit_kind", p.unitKind);
  q.set("unit_index", String(p.unitIndex));
  return q;
}

export function findingPin(f: Finding): UnitPin {
  return {
    datasetId: f.artifact.dataset_id,
    sha256: f.artifact.sha256,
    pointId: f.unit.point_id,
    unitKind: f.unit.kind,
    unitIndex: f.unit.index,
    dims: f.reopen.dimensions,
  };
}

export type PinParse =
  | null
  | { ok: true; pin: UnitPin }
  | { ok: false; message: string };

const intParam = (s: string | null): number | null =>
  s !== null && /^(0|[1-9]\d*)$/.test(s) && Number.isSafeInteger(Number(s)) ? Number(s) : null;

/** null = no pin keys at all (an ordinary link). A partial or malformed pin is
 *  an explicit error — never stripped into an unpinned open. */
export function parsePin(p: URLSearchParams): PinParse {
  const present = PIN_KEYS.filter((k) => p.has(k));
  if (present.length === 0) return null;
  const missing = PIN_KEYS.filter((k) => !p.has(k));
  if (!p.get("model")) missing.push("model" as never);
  if (missing.length) {
    return { ok: false, message: `This unit link is incomplete (missing ${missing.join(", ")}), so it cannot be verified.` };
  }
  const sha = p.get("artifact")!;
  if (!SHA256_RE.test(sha)) return { ok: false, message: "The link's artifact digest is not a SHA-256." };
  const pointId = intParam(p.get("point"));
  const unitIndex = intParam(p.get("unit_index"));
  if (pointId === null) return { ok: false, message: "The link's point id is not a non-negative safe integer." };
  if (unitIndex === null) return { ok: false, message: "The link's unit index is not a non-negative safe integer." };
  const unitKind = p.get("unit_kind")!;
  if (!unitKind) return { ok: false, message: "The link's unit kind is empty." };
  return {
    ok: true,
    pin: {
      datasetId: p.get("model")!,
      sha256: sha,
      pointId,
      unitKind,
      unitIndex,
      dims: p.get("dims") === "3" ? 3 : 2,
    },
  };
}
