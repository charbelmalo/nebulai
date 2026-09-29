/** experience.ts — the trusted release manifest, `out/experience.json`.
 *
 *  `out/index.json` answers "which maps exist". This manifest answers the
 *  questions the three experiences need and the index cannot:
 *
 *  · which exact bytes are the curated starter (`default_atlas`, pinned by
 *    SHA-256 — never "whatever sorts first in the index");
 *  · where each published map lives at an IMMUTABLE content-addressed path
 *    (`artifacts/<sha256>/nebulai.json`), so a saved finding can reopen the
 *    exact artifact it was made from after the current map changes;
 *  · which optional sidecars a dataset actually ships (`sidecars`), so the
 *    viewer stops probing for files that were never published.
 *
 *  It is the ONLY thing that may turn a finding's digest into a fetch: an
 *  imported record's own `source_path` is descriptive text, never a URL.
 *
 *  Validation follows contracts/README.md: supported schema version, unique
 *  (dataset_id, sha256) entries, valid digests and byte counts, normalized
 *  relative paths with no traversal, and defaults that resolve to listed
 *  entries. An invalid manifest is reported and ignored as a whole — the
 *  viewer then behaves exactly as it did before the manifest existed, with the
 *  starter unverified, rather than trusting half of it.
 *
 *  Pure except for `loadManifest`, which takes its base explicitly. The
 *  release packager (viewer/scripts/package-experience.ts) imports the same
 *  validator, so a manifest that ships has passed the check the viewer runs. */

import { isRepresentation, type Representation } from "./representation";

export const EXPERIENCES = ["learn", "atlas", "research"] as const;
export type Experience = (typeof EXPERIENCES)[number];

export function isExperience(v: unknown): v is Experience {
  return typeof v === "string" && (EXPERIENCES as readonly string[]).includes(v);
}

export interface ManifestArtifact {
  dataset_id: string;
  sha256: string;
  /** immutable: `artifacts/<sha256>/nebulai.json`, relative to DATA_BASE */
  path: string;
  /** the dataset's mutable path in index.json at publish time */
  legacy_path: string;
  /** directory holding this dataset's sidecars (channels, interp, validation) */
  sidecar_base: string;
  bytes: number;
  label: string;
  representation: Representation;
  model_revision: string | null;
  has_interp: boolean;
  /** true for the digest the dataset currently serves at `legacy_path`.
   *  Required when a dataset lists more than one digest; optional otherwise. */
  current?: boolean;
  /** sidecar files that exist under `sidecar_base`, relative to it
   *  (e.g. "validation.json", "interp/index.json"). Absent = unknown, and the
   *  viewer probes as before; present = authoritative. */
  sidecars?: string[];
}

export interface ExperienceManifest {
  schema_version: 1;
  entry_experiences: Experience[];
  default_atlas: { dataset_id: string; sha256: string; view: "atlas"; dimensions: 2 | 3 };
  learn_intro: { lesson_id: string; dataset_id: string; sha256: string };
  research_intro: {
    dataset_id: string;
    feature: string;
    bundle_path: string;
    sha256: string;
    bytes: number;
  };
  artifacts: ManifestArtifact[];
}

export type ManifestResult =
  | { ok: true; manifest: ExperienceManifest }
  | { ok: false; errors: string[] };

export const SHA256_RE = /^[a-f0-9]{64}$/;

/** The curated Atlas starter, by id. Used ONLY when no valid manifest names
 *  one: the viewer then opens this map marked "unverified default" (its bytes
 *  are not pinned), and when the index does not list it either, it shows the
 *  chooser. There is deliberately no "first dataset in the index" fallback —
 *  index order is alphabetical, not editorial. */
export const STARTER_DATASET_ID = "gpt2-small__sae__blocks.8.hook_resid_pre";

/** A normalized relative path under DATA_BASE: no scheme, no absolute or
 *  protocol-relative start, no backslashes, no `.`/`..`/empty segments, no
 *  query/fragment/percent-escapes that could smuggle one of those past this
 *  check. Deliberately stricter than URL resolution needs to be. */
export function isSafeRelativePath(p: unknown): p is string {
  if (typeof p !== "string" || p.length === 0 || p.length > 512) return false;
  if (/[\\?#%:\s]/.test(p) || /[\u0000-\u001f\u007f]/.test(p)) return false;
  if (p.startsWith("/")) return false;
  return p.split("/").every((seg) => seg !== "" && seg !== "." && seg !== "..");
}

const isObj = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);
const isStr = (v: unknown): v is string => typeof v === "string" && v.length > 0;
const isBytes = (v: unknown): v is number =>
  typeof v === "number" && Number.isSafeInteger(v) && v > 0;

export function validateManifest(raw: unknown): ManifestResult {
  const errors: string[] = [];
  const err = (m: string) => errors.push(m);
  if (!isObj(raw)) return { ok: false, errors: ["manifest is not an object"] };
  if (raw.schema_version !== 1) {
    return { ok: false, errors: [`unsupported manifest schema_version ${String(raw.schema_version)}`] };
  }

  const entries = raw.entry_experiences;
  if (
    !Array.isArray(entries) ||
    entries.length === 0 ||
    !entries.every(isExperience) ||
    new Set(entries).size !== entries.length
  ) {
    err("entry_experiences must be a non-empty list of unique learn|atlas|research");
  }

  const artifacts: ManifestArtifact[] = [];
  if (!Array.isArray(raw.artifacts) || raw.artifacts.length === 0) {
    err("artifacts must be a non-empty list");
  } else {
    raw.artifacts.forEach((a, i) => {
      const at = `artifacts[${i}]`;
      if (!isObj(a)) return err(`${at} is not an object`);
      const before = errors.length;
      if (!isStr(a.dataset_id) || !isSafeRelativePath(a.dataset_id) || a.dataset_id.includes("/"))
        err(`${at}.dataset_id is not a plain dataset id`);
      if (typeof a.sha256 !== "string" || !SHA256_RE.test(a.sha256))
        err(`${at}.sha256 is not a lowercase SHA-256`);
      else if (a.path !== `artifacts/${a.sha256}/nebulai.json`)
        err(`${at}.path must be artifacts/<sha256>/nebulai.json`);
      if (!isSafeRelativePath(a.legacy_path)) err(`${at}.legacy_path is not a safe relative path`);
      if (!isSafeRelativePath(a.sidecar_base)) err(`${at}.sidecar_base is not a safe relative path`);
      if (!isBytes(a.bytes)) err(`${at}.bytes must be a positive integer`);
      if (!isStr(a.label)) err(`${at}.label is required`);
      if (!isRepresentation(a.representation)) err(`${at}.representation is not a known class`);
      if (!(a.model_revision === null || isStr(a.model_revision)))
        err(`${at}.model_revision must be a string or null`);
      if (typeof a.has_interp !== "boolean") err(`${at}.has_interp must be boolean`);
      if (a.current !== undefined && typeof a.current !== "boolean")
        err(`${at}.current must be boolean when present`);
      if (a.sidecars !== undefined) {
        if (
          !Array.isArray(a.sidecars) ||
          !a.sidecars.every(isSafeRelativePath) ||
          new Set(a.sidecars).size !== a.sidecars.length
        )
          err(`${at}.sidecars must be unique safe relative paths`);
      }
      if (errors.length === before) artifacts.push(a as unknown as ManifestArtifact);
    });
  }

  // uniqueness + one current digest per dataset
  const seen = new Set<string>();
  const byDataset = new Map<string, ManifestArtifact[]>();
  for (const a of artifacts) {
    const key = `${a.dataset_id}\u0000${a.sha256}`;
    if (seen.has(key)) err(`duplicate artifact ${a.dataset_id}@${a.sha256.slice(0, 12)}`);
    seen.add(key);
    const list = byDataset.get(a.dataset_id) ?? [];
    list.push(a);
    byDataset.set(a.dataset_id, list);
  }
  for (const [id, list] of byDataset) {
    const cur = list.filter((a) => a.current === true).length;
    if (cur > 1) err(`dataset ${id} marks ${cur} artifacts current`);
    if (list.length > 1 && cur === 0) err(`dataset ${id} lists ${list.length} digests but none is current`);
  }

  const resolves = (ds: unknown, sha: unknown) =>
    artifacts.some((a) => a.dataset_id === ds && a.sha256 === sha);

  const da = raw.default_atlas;
  if (!isObj(da)) err("default_atlas is required");
  else {
    if (!resolves(da.dataset_id, da.sha256)) err("default_atlas does not resolve to a listed artifact");
    if (da.view !== "atlas") err("default_atlas.view must be atlas");
    if (da.dimensions !== 2 && da.dimensions !== 3) err("default_atlas.dimensions must be 2 or 3");
  }
  const li = raw.learn_intro;
  if (!isObj(li)) err("learn_intro is required");
  else {
    if (!isStr(li.lesson_id) || !/^[a-z0-9-]+$/.test(li.lesson_id)) err("learn_intro.lesson_id is invalid");
    if (!resolves(li.dataset_id, li.sha256)) err("learn_intro does not resolve to a listed artifact");
  }
  const ri = raw.research_intro;
  if (!isObj(ri)) err("research_intro is required");
  else {
    if (!isStr(ri.dataset_id)) err("research_intro.dataset_id is required");
    if (!isStr(ri.feature)) err("research_intro.feature is required");
    if (!isSafeRelativePath(ri.bundle_path)) err("research_intro.bundle_path is not a safe relative path");
    if (typeof ri.sha256 !== "string" || !SHA256_RE.test(ri.sha256)) err("research_intro.sha256 is invalid");
    if (!isBytes(ri.bytes)) err("research_intro.bytes must be a positive integer");
  }

  return errors.length ? { ok: false, errors } : { ok: true, manifest: raw as unknown as ExperienceManifest };
}

/* ── lookups ────────────────────────────────────────────────────────────── */

/** The digest a dataset currently serves: the entry flagged `current`, or the
 *  only entry when there is just one. */
export function currentArtifact(m: ExperienceManifest | null, datasetId: string | null): ManifestArtifact | null {
  if (!m || !datasetId) return null;
  const list = m.artifacts.filter((a) => a.dataset_id === datasetId);
  return list.find((a) => a.current === true) ?? (list.length === 1 ? list[0]! : null);
}

/** Exact (dataset, digest) lookup — the only resolution a finding may use. */
export function findArtifact(
  m: ExperienceManifest | null,
  datasetId: string,
  sha256: string,
): ManifestArtifact | null {
  return m?.artifacts.find((a) => a.dataset_id === datasetId && a.sha256 === sha256) ?? null;
}

/* ── the active manifest (module registry) ─────────────────────────────── */

export type ManifestStatus =
  | { state: "unloaded" }
  | { state: "absent" }
  | { state: "invalid"; errors: string[] }
  | { state: "ok"; manifest: ExperienceManifest };

let status: ManifestStatus = { state: "unloaded" };

export function setManifestStatus(s: ManifestStatus): void {
  status = s;
}
export function manifestStatus(): ManifestStatus {
  return status;
}
export function activeManifest(): ExperienceManifest | null {
  return status.state === "ok" ? status.manifest : null;
}

/** True only when the manifest AUTHORITATIVELY says this sidecar is not
 *  published for this dataset. Unknown (no manifest, no entry, no `sidecars`
 *  list) is false, so callers keep probing exactly as before. */
export function sidecarKnownAbsent(datasetId: string, rel: string): boolean {
  const a = currentArtifact(activeManifest(), datasetId);
  if (!a || !a.sidecars) return false;
  return !a.sidecars.includes(rel);
}

/** Fetch and validate `experience.json`. Absent (404, or a dev-server SPA
 *  fallback serving HTML) is a normal state, not an error. */
export async function loadManifest(base: string, noCache = false): Promise<ManifestStatus> {
  let res: Response;
  try {
    res = await fetch(`${base}/experience.json`, noCache ? { cache: "no-store" } : undefined);
  } catch {
    return { state: "absent" };
  }
  const type = res.headers.get("content-type") ?? "";
  if (!res.ok || !type.includes("json")) return { state: "absent" };
  let raw: unknown;
  try {
    raw = await res.json();
  } catch (e) {
    return { state: "invalid", errors: [`experience.json is not JSON: ${e instanceof Error ? e.message : e}`] };
  }
  const r = validateManifest(raw);
  return r.ok ? { state: "ok", manifest: r.manifest } : { state: "invalid", errors: r.errors };
}
