import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { INTERP_FEATURES } from "../../src/scene/interp/registry";
import { REQUIREMENTS, availability, availableCount } from "../../src/scene/interp/requirements";
import type { InterpIndex } from "../../src/data/interp";

const SRC = join(__dirname, "../../src");
const read = (rel: string) => readFileSync(join(SRC, rel), "utf8");

/** loader name → the file it reads, parsed from data/interp.ts itself */
function loaderFiles(): Map<string, string> {
  const src = read("data/interp.ts");
  const out = new Map<string, string>();
  const re = /export const (load\w+) = (?:(?!\nexport )[\s\S])*?\$\{interpBase\([^)]*\)\}\/([\w]+)\.json`/g;
  for (const m of src.matchAll(re)) out.set(m[1]!, `${m[2]}.json`);
  return out;
}

/** feature id → driver class, parsed from the registry's create() lines */
function driverOf(): Map<string, string> {
  const src = read("scene/interp/registry.ts");
  const out = new Map<string, string>();
  for (const block of src.split(/\n  \{\n/).slice(1)) {
    const id = /id: "([\w-]+)"/.exec(block)?.[1];
    const cls = /create: \(\) => new (\w+)\(/.exec(block)?.[1];
    if (id && cls) out.set(id, cls);
  }
  return out;
}

const idx = (bundles: string[]): InterpIndex => ({
  meta: { model: "m", created: "" },
  bundles,
  traces: [],
});

describe("Internals analysis requirements", () => {
  it("covers exactly the registered analyses", () => {
    expect(Object.keys(REQUIREMENTS).sort()).toEqual(INTERP_FEATURES.map((f) => f.id).sort());
  });

  it("matches the loaders each driver actually calls", () => {
    const files = loaderFiles();
    const drivers = driverOf();
    expect(drivers.size).toBe(INTERP_FEATURES.length);
    for (const f of INTERP_FEATURES) {
      const cls = drivers.get(f.id)!;
      const calls = new Set(read(`scene/interp/${cls}.ts`).match(/\bload[A-Z]\w*(?=\()/g) ?? []);
      const req = REQUIREMENTS[f.id]!;
      const bundleFiles = [...calls].filter((c) => files.has(c)).map((c) => files.get(c)!);
      if (req.kind === "bundles") {
        expect(new Set(req.files), `${f.id} (${cls})`).toEqual(new Set(bundleFiles));
      } else if (req.kind === "trace") {
        expect(calls.has("loadTrace"), f.id).toBe(true);
        expect(bundleFiles, f.id).toEqual([]);
      } else if (req.kind === "sweep") {
        expect(calls.has("loadSteerBundle"), f.id).toBe(true);
      } else {
        expect(calls.size, `${f.id} is live-only`).toBe(0);
      }
    }
  });

  it("states availability from the export's bundle list", () => {
    expect(availability("fourier-atlas", idx(["fourier.json"]))).toEqual({
      state: "available",
      files: ["fourier.json"],
    });
    expect(availability("cofire-venn", idx(["cofire.json"])).state).toBe("missing");
    expect(availability("cofire-venn", idx(["sae.json", "cofire.json"])).state).toBe("available");
    expect(availability("attention-flow", idx(["trace_a.json"]))).toEqual({
      state: "available",
      files: ["trace_a.json"],
    });
    expect(availability("steer-rail", idx(["fourier.json"])).state).toBe("missing");
    expect(availability("steer-rail", idx(["intervene_x.json"])).state).toBe("available");
    expect(availability("live-nebula", idx([])).state).toBe("live");
    expect(availability("fourier-atlas", undefined).state).toBe("unknown");
    expect(availability("fourier-atlas", null).state).toBe("missing");
    expect(availability("no-such-view", idx(["fourier.json"])).state).toBe("unknown");
  });

  it("counts only analyses the export can show", () => {
    const ids = INTERP_FEATURES.map((f) => f.id);
    expect(availableCount(ids, idx([]))).toBe(0);
    expect(availableCount(ids, idx(["fourier.json", "weights.json", "trace_a.json"]))).toBe(2 + 5);
  });
});
