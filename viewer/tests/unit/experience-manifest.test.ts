import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  currentArtifact,
  findArtifact,
  isSafeRelativePath,
  setManifestStatus,
  sidecarKnownAbsent,
  validateManifest,
  type ExperienceManifest,
} from "../../src/data/experience";

const A = "a".repeat(64);
const B = "b".repeat(64);
const R = "c".repeat(64);

function good(): ExperienceManifest {
  return {
    schema_version: 1,
    entry_experiences: ["learn", "atlas", "research"],
    default_atlas: { dataset_id: "starter", sha256: A, view: "atlas", dimensions: 2 },
    learn_intro: { lesson_id: "what-is-a-point", dataset_id: "starter", sha256: A },
    research_intro: { dataset_id: "gpt2", feature: "fourier-atlas", bundle_path: "gpt2/interp/fourier.json", sha256: R, bytes: 10 },
    artifacts: [
      {
        dataset_id: "starter",
        sha256: A,
        path: `artifacts/${A}/nebulai.json`,
        legacy_path: "starter/nebulai.json",
        sidecar_base: "starter",
        bytes: 100,
        label: "Starter",
        representation: "sae_decoder",
        model_revision: null,
        has_interp: false,
        sidecars: ["validation.json"],
      },
    ],
  };
}

const errorsOf = (m: unknown) => {
  const r = validateManifest(m);
  return r.ok ? [] : r.errors;
};

describe("validateManifest", () => {
  it("accepts the canonical shape", () => {
    expect(validateManifest(good()).ok).toBe(true);
  });

  it("rejects an unsupported version outright", () => {
    expect(errorsOf({ ...good(), schema_version: 2 })[0]).toMatch(/schema_version/);
  });

  it("rejects traversal, absolute and scheme paths", () => {
    for (const bad of ["../x.json", "/abs/x.json", "https://evil/x.json", "a/./b", "a//b", "a\\b", "a%2e%2e/b", "x?y"]) {
      const m = good();
      m.artifacts[0]!.legacy_path = bad;
      expect(errorsOf(m).join()).toMatch(/legacy_path/);
    }
  });

  it("requires the immutable path to be exactly artifacts/<sha>/nebulai.json", () => {
    const m = good();
    m.artifacts[0]!.path = `artifacts/${B}/nebulai.json`;
    expect(errorsOf(m).join()).toMatch(/path must be/);
  });

  it("rejects bad digests and byte counts", () => {
    const m = good();
    m.artifacts[0]!.sha256 = A.toUpperCase();
    expect(errorsOf(m).join()).toMatch(/sha256/);
    const n = good();
    n.artifacts[0]!.bytes = 0;
    expect(errorsOf(n).join()).toMatch(/bytes/);
  });

  it("rejects duplicate (dataset, digest) entries", () => {
    const m = good();
    m.artifacts.push({ ...m.artifacts[0]! });
    expect(errorsOf(m).join()).toMatch(/duplicate/);
  });

  it("needs exactly one current digest when a dataset lists several", () => {
    const m = good();
    m.artifacts.push({ ...m.artifacts[0]!, sha256: B, path: `artifacts/${B}/nebulai.json` });
    expect(errorsOf(m).join()).toMatch(/none is current/);
    m.artifacts[0]!.current = true;
    m.artifacts[1]!.current = true;
    expect(errorsOf(m).join()).toMatch(/marks 2 artifacts current/);
    m.artifacts[1]!.current = false;
    expect(validateManifest(m).ok).toBe(true);
    expect(currentArtifact(m, "starter")!.sha256).toBe(A);
    expect(findArtifact(m, "starter", B)!.sha256).toBe(B);
  });

  it("rejects defaults that do not resolve", () => {
    const m = good();
    m.default_atlas.sha256 = B;
    expect(errorsOf(m).join()).toMatch(/default_atlas does not resolve/);
    const n = good();
    n.learn_intro.dataset_id = "other";
    expect(errorsOf(n).join()).toMatch(/learn_intro does not resolve/);
  });

  it("rejects an unknown representation and non-boolean flags", () => {
    const m = good() as unknown as { artifacts: Record<string, unknown>[] };
    m.artifacts[0]!.representation = "vibes";
    m.artifacts[0]!.has_interp = "no";
    const e = errorsOf(m).join();
    expect(e).toMatch(/representation/);
    expect(e).toMatch(/has_interp/);
  });
});

describe("isSafeRelativePath", () => {
  it("allows plain nested paths only", () => {
    expect(isSafeRelativePath("gpt2/interp/index.json")).toBe(true);
    expect(isSafeRelativePath("")).toBe(false);
    expect(isSafeRelativePath("gpt2/../x")).toBe(false);
  });
});

describe("sidecarKnownAbsent", () => {
  afterEach(() => setManifestStatus({ state: "unloaded" }));

  it("is false (keep probing) without a manifest", () => {
    setManifestStatus({ state: "absent" });
    expect(sidecarKnownAbsent("starter", "channels.json")).toBe(false);
  });

  it("is authoritative only when the entry lists its sidecars", () => {
    const m = good();
    setManifestStatus({ state: "ok", manifest: m });
    expect(sidecarKnownAbsent("starter", "channels.json")).toBe(true);
    expect(sidecarKnownAbsent("starter", "validation.json")).toBe(false);
    expect(sidecarKnownAbsent("unlisted", "channels.json")).toBe(false);
    delete m.artifacts[0]!.sidecars;
    expect(sidecarKnownAbsent("starter", "channels.json")).toBe(false);
  });
});

// The packaged manifest in the working out/ (gitignored, so optional here).
const OUT = join(__dirname, "..", "..", "..", "out", "experience.json");
describe.skipIf(!existsSync(OUT))("packaged out/experience.json", () => {
  it("passes the viewer's own validator and pins the plan's starter", () => {
    const r = validateManifest(JSON.parse(readFileSync(OUT, "utf8")));
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.manifest.default_atlas).toEqual({
      dataset_id: "gpt2-small__sae__blocks.8.hook_resid_pre",
      sha256: "d36402d8503e82df71a9c64e48a4f9561f35ad2ac666c43eee127ff71eac10f7",
      view: "atlas",
      dimensions: 2,
    });
    expect(r.manifest.research_intro.sha256).toBe("572645fc13b08a8afcaab1d4155f0f2010c4a7c84757e3eac52bb2b67f785093");
  });
});
