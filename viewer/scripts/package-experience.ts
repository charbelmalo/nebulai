/** package-experience.ts — publish out/experience.json and the immutable,
 *  content-addressed copies of every map it lists.
 *
 *      npm run package:experience                    # ../out, writes
 *      npm run package:experience -- --check         # validate only, no writes
 *      npm run package:experience -- --out /path/out # another out root
 *
 *  For each dataset in out/index.json it hashes `<path>` (SHA-256 over the raw
 *  bytes the viewer will download), and makes
 *  `artifacts/<sha256>/nebulai.json` a HARD LINK to it — the mini has no disk
 *  for a second copy of every map. That link stays immutable only because
 *  every writer of nebulai.json replaces the file by rename
 *  (src/nebulai/backend/atomic.py); an in-place rewrite would change the
 *  artifact's bytes behind its name. So an existing artifact is always
 *  re-hashed here, and one whose bytes no longer match its directory name
 *  fails the run loudly.
 *
 *  ADDITIVE ONLY. Nothing is ever deleted: a digest from an earlier manifest
 *  stays listed (current: false) as long as its artifact is still on disk, so
 *  a finding saved against it keeps reopening the exact bytes. Retention is a
 *  separate, deliberate step (docs: DEPLOY.md, "Artifact retention").
 *
 *  The manifest is checked with the SAME validator the viewer runs
 *  (src/data/experience.ts) before it is written; an invalid one is never
 *  published. */

import { createHash } from "node:crypto";
import { existsSync, linkSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { registerHooks } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// The viewer's modules import each other without extensions (Vite resolves
// them). Teach Node's loader the same one rule so this script can import the
// validator itself rather than a copy of it.
registerHooks({
  resolve(specifier, context, next) {
    try {
      return next(specifier, context);
    } catch (e) {
      if (/^\.\.?\//.test(specifier) && !/\.[cm]?[jt]s$/.test(specifier)) {
        return next(`${specifier}.ts`, context);
      }
      throw e;
    }
  },
});

const here = dirname(fileURLToPath(import.meta.url));
const { validateManifest, STARTER_DATASET_ID } = await import("../src/data/experience.ts");
const { classifyRepresentation, datasetLabel } = await import("../src/data/representation.ts");
type ExperienceManifest = import("../src/data/experience.ts").ExperienceManifest;
type ManifestArtifact = import("../src/data/experience.ts").ManifestArtifact;

/* ── editorial choices (the plan's defaults) ─────────────────────────────── */
const LEARN_LESSON = "what-is-a-point";
const RESEARCH = { dataset_id: "gpt2", feature: "fourier-atlas", bundle_path: "gpt2/interp/fourier.json" };
/** sidecars the viewer knows how to read, relative to a dataset directory */
const SIDECARS = ["validation.json", "channels.json", "directions.json", "interp/index.json"];

/* ── args ─────────────────────────────────────────────────────────────────── */
const argv = process.argv.slice(2);
const check = argv.includes("--check");
const outIdx = argv.indexOf("--out");
const outRoot = resolve(outIdx >= 0 && argv[outIdx + 1] ? argv[outIdx + 1]! : join(here, "..", "..", "out"));

const sha256 = (buf: Buffer) => createHash("sha256").update(buf).digest("hex");
const fail = (msg: string): never => {
  console.error(`package-experience: ${msg}`);
  process.exit(1);
};

interface IndexEntry {
  id: string;
  path: string;
  has_interp?: boolean;
}

const indexPath = join(outRoot, "index.json");
if (!existsSync(indexPath)) fail(`no ${indexPath}`);
const index = JSON.parse(readFileSync(indexPath, "utf8")) as { datasets: IndexEntry[] };

const manifestPath = join(outRoot, "experience.json");
let previous: ExperienceManifest | null = null;
if (existsSync(manifestPath)) {
  const r = validateManifest(JSON.parse(readFileSync(manifestPath, "utf8")));
  if (!r.ok) fail(`existing experience.json is invalid — fix or move it first:\n  ${r.errors.join("\n  ")}`);
  else previous = r.manifest;
}

const artifacts: ManifestArtifact[] = [];
let linked = 0;

for (const entry of index.datasets) {
  const src = join(outRoot, entry.path);
  if (!existsSync(src)) fail(`${entry.id}: ${entry.path} is listed in index.json but missing`);
  const bytes = readFileSync(src);
  const digest = sha256(bytes);
  const doc = JSON.parse(bytes.toString("utf8")) as { meta: Record<string, unknown> };
  const meta = doc.meta ?? {};

  const rel = `artifacts/${digest}/nebulai.json`;
  const dst = join(outRoot, rel);
  if (existsSync(dst)) {
    const onDisk = sha256(readFileSync(dst));
    if (onDisk !== digest) fail(`${rel} no longer hashes to its name (got ${onDisk}) — something rewrote it in place`);
  } else if (!check) {
    mkdirSync(dirname(dst), { recursive: true });
    linkSync(src, dst);
    linked++;
  }

  const sidecarBase = dirname(entry.path);
  const sidecars = SIDECARS.filter((s) => existsSync(join(outRoot, sidecarBase, s)));
  const revision = [meta.model_revision, meta.revision].find((v) => typeof v === "string" && v) as string | undefined;

  artifacts.push({
    dataset_id: entry.id,
    sha256: digest,
    path: rel,
    legacy_path: entry.path,
    sidecar_base: sidecarBase,
    bytes: bytes.byteLength,
    label: datasetLabel(meta, entry.id),
    representation: classifyRepresentation(meta as never),
    model_revision: revision ?? null,
    has_interp: entry.has_interp ?? sidecars.includes("interp/index.json"),
    current: true,
    sidecars,
  });
}

// keep earlier digests whose bytes are still on disk (additive, never pruned here)
for (const old of previous?.artifacts ?? []) {
  if (artifacts.some((a) => a.dataset_id === old.dataset_id && a.sha256 === old.sha256)) continue;
  if (!index.datasets.some((d) => d.id === old.dataset_id)) continue; // dataset withdrawn from the index
  const p = join(outRoot, old.path);
  if (!existsSync(p)) continue;
  if (sha256(readFileSync(p)) !== old.sha256) fail(`${old.path} no longer hashes to its name`);
  artifacts.push({ ...old, current: false });
}

const byId = (id: string) => artifacts.find((a) => a.dataset_id === id && a.current);
const starter = byId(STARTER_DATASET_ID) ?? fail(`starter ${STARTER_DATASET_ID} is not in index.json`);
const researchFile = join(outRoot, RESEARCH.bundle_path);
if (!existsSync(researchFile)) fail(`research intro bundle ${RESEARCH.bundle_path} is missing`);
const researchBytes = readFileSync(researchFile);

const manifest: ExperienceManifest = {
  schema_version: 1,
  entry_experiences: ["learn", "atlas", "research"],
  default_atlas: { dataset_id: starter.dataset_id, sha256: starter.sha256, view: "atlas", dimensions: 2 },
  learn_intro: { lesson_id: LEARN_LESSON, dataset_id: starter.dataset_id, sha256: starter.sha256 },
  research_intro: {
    ...RESEARCH,
    sha256: sha256(researchBytes),
    bytes: researchBytes.byteLength,
  },
  artifacts: artifacts.sort((a, b) =>
    a.dataset_id === b.dataset_id ? Number(b.current) - Number(a.current) : a.dataset_id < b.dataset_id ? -1 : 1,
  ),
};

const v = validateManifest(manifest);
if (!v.ok) fail(`generated manifest failed validation:\n  ${v.errors.join("\n  ")}`);

const text = `${JSON.stringify(manifest, null, 2)}\n`;
const unchanged = existsSync(manifestPath) && readFileSync(manifestPath, "utf8") === text;
if (!check && !unchanged) {
  const tmp = `${manifestPath}.tmp`;
  writeFileSync(tmp, text);
  renameSync(tmp, manifestPath);
}

const total = artifacts.filter((a) => a.current).reduce((n, a) => n + a.bytes, 0);
console.log(
  `${check ? "checked" : "packaged"} ${artifacts.length} artifacts ` +
    `(${artifacts.filter((a) => !a.current).length} retained), ${linked} new links, ` +
    `${(total / 1e6).toFixed(1)} MB current; experience.json ${check ? "valid" : unchanged ? "unchanged" : "written"}`,
);
console.log(`  default_atlas  ${starter.dataset_id} @ ${starter.sha256.slice(0, 12)}`);
console.log(`  research_intro ${RESEARCH.bundle_path} @ ${manifest.research_intro.sha256.slice(0, 12)}`);
