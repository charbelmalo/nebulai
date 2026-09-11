/** The channel sidecar, browser side.
 *
 *  Three kinds of assertion live here and they are deliberately not separated
 *  into three files, because they are three views of ONE rule:
 *
 *  1. **The closed space set is the same set on both sides.** `spaces.py` and
 *     `data/channels.ts` each carry the eight families. A tag added to one and
 *     not the other is a tag one half silently drops, so the first block reads
 *     the Python enum off disk and pins the TypeScript mirror against it.
 *  2. **Absence is never zero.** A null in the JSON becomes NaN in the column,
 *     `channelValue` returns null for it, `lowestBy` refuses to sort it to the
 *     front, and the GPU's ramp coordinate has no position for it.
 *  3. **A channel that does not line up with the map is dropped whole.** An
 *     index-shifted column mislabels every point past the gap, which is worse
 *     than having no column at all.
 */

import { readFileSync, existsSync } from "node:fs";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  SPACE_FAMILIES,
  __resetChannels,
  channelFor,
  channelValue,
  channelsFor,
  channelsLoaded,
  compatible,
  ensureChannels,
  isKnownSpace,
  lowestBy,
  parseChannel,
  parseChannelSet,
  refusalReason,
  type Channel,
} from "../../src/data/channels";
import { CHANNEL_MISSING, channelRampT } from "../../src/scene/layers/PointsLayer";

const REPO = new URL("../../../", import.meta.url);
const SPACES_PY = new URL("src/nebulai/spaces.py", REPO);
const GPT2_CHANNELS = new URL("out/gpt2/channels.json", REPO);

afterEach(() => {
  __resetChannels();
  vi.unstubAllGlobals();
});

/* ── 1. the closed set, pinned across the language boundary ───────────────── */

describe("space families ↔ spaces.py", () => {
  it("carries exactly the families the Python enum declares", () => {
    const py = readFileSync(SPACES_PY, "utf8");
    const block = py.slice(py.indexOf("class SpaceFamily"));
    const end = block.indexOf("\n_BARE");
    const values = [...block.slice(0, end).matchAll(/^\s{4}[A-Z_]+ = "([^"]+)"$/gm)].map(
      (m) => m[1]!,
    );
    expect(values.length).toBeGreaterThan(0);
    expect(values).toEqual([...SPACE_FAMILIES]);
  });

  it("accepts every parameterised form and rejects near-misses", () => {
    for (const bare of ["W_E.raw", "W_E.centered", "W_U.raw"]) {
      expect(isKnownSpace(bare)).toBe(true);
    }
    expect(isKnownSpace("resid.L11")).toBe(true);
    expect(isKnownSpace("resid.L-1")).toBe(true);
    expect(isKnownSpace("mlp_out.L0")).toBe(true);
    expect(isKnownSpace("sae.L8.gpt2-small-res-jb")).toBe(true);
    expect(isKnownSpace("text-embed.all-MiniLM-L6-v2")).toBe(true);
    expect(isKnownSpace("persona-pca.assistant-axis")).toBe(true);

    // a bare layer number with no family, a family with no layer, a family
    // that does not exist, and an SAE with no dictionary
    expect(isKnownSpace("resid.14")).toBe(false);
    expect(isKnownSpace("resid")).toBe(false);
    expect(isKnownSpace("attn_out.L3")).toBe(false);
    expect(isKnownSpace("sae.L8")).toBe(false);
    expect(isKnownSpace("")).toBe(false);
  });
});

describe("compatible() — D2's hard refusal", () => {
  it("lets a space be compared only with itself", () => {
    expect(compatible("resid.L11", "resid.L11")).toBe(true);
    expect(compatible("resid.L11", "resid.L12")).toBe(false);
  });

  it("keeps raw and centred W_E apart — the translation IS the experiment", () => {
    expect(compatible("W_E.raw", "W_E.centered")).toBe(false);
    expect(refusalReason("W_E.raw", "W_E.centered")).toContain("different bases");
  });

  it("makes an unknown space incomparable with itself", () => {
    expect(compatible("attn_out.L3", "attn_out.L3")).toBe(false);
    expect(refusalReason("attn_out.L3", "resid.L3")).toContain("unknown space");
  });

  it("returns null when there is nothing to refuse", () => {
    expect(refusalReason("sae.L8.res-jb", "sae.L8.res-jb")).toBeNull();
  });
});

/* ── 2. parsing, and what absence looks like afterwards ───────────────────── */

function raw(values: (number | null)[], over: Record<string, unknown> = {}) {
  return {
    id: "we_norm",
    label: "row norm",
    space: "W_E.raw",
    method: "l2",
    formula: "sqrt(sum(x**2))",
    fidelity: "deterministic",
    units: "l2",
    stats: { min: 1, max: 3, mean: 2, n_missing: 0 },
    values,
    ...over,
  };
}

describe("parseChannel", () => {
  it("columnarises a well-formed channel", () => {
    const ch = parseChannel(raw([1, 2, 3]), 3)!;
    expect(ch).not.toBeNull();
    expect(ch.id).toBe("we_norm");
    expect(ch.space).toBe("W_E.raw");
    expect(ch.fidelity).toBe("deterministic");
    expect([...ch.values]).toEqual([1, 2, 3]);
    expect(ch.stats.n_missing).toBe(0);
  });

  it("turns null into NaN, never into 0, and counts it", () => {
    const ch = parseChannel(raw([1, null, 3]), 3)!;
    expect(Number.isNaN(ch.values[1]!)).toBe(true);
    expect(ch.values[1]).not.toBe(0);
    expect(ch.stats.n_missing).toBe(1);
  });

  it("treats anything non-numeric as not measured", () => {
    const ch = parseChannel(raw([1, "x" as never, null]), 3)!;
    expect(ch.stats.n_missing).toBe(2);
    expect(Number.isFinite(ch.values[0]!)).toBe(true);
  });

  it("trusts the array it read over the file's own n_missing", () => {
    const ch = parseChannel(raw([1, null, 3], { stats: { n_missing: 99 } }), 3)!;
    expect(ch.stats.n_missing).toBe(1);
  });

  it("refuses a length that disagrees with the map", () => {
    expect(parseChannel(raw([1, 2]), 3)).toBeNull();
    expect(parseChannel(raw([1, 2, 3, 4]), 3)).toBeNull();
  });

  it("refuses a space outside the closed set", () => {
    expect(parseChannel(raw([1, 2, 3], { space: "attn_out.L3" }), 3)).toBeNull();
    expect(parseChannel(raw([1, 2, 3], { space: 7 }), 3)).toBeNull();
  });

  it("refuses a channel with no usable id", () => {
    expect(parseChannel(raw([1, 2, 3], { id: "" }), 3)).toBeNull();
    expect(parseChannel(raw([1, 2, 3], { id: 12 }), 3)).toBeNull();
  });

  it("keeps a direction reference when one is present (phase 1)", () => {
    const ch = parseChannel(raw([1, 2, 3], { direction: "refusal-dom" }), 3)!;
    expect(ch.direction).toBe("refusal-dom");
    expect(parseChannel(raw([1, 2, 3]), 3)!.direction).toBeUndefined();
  });
});

describe("parseChannelSet", () => {
  const doc = (over: Record<string, unknown> = {}) => ({
    meta: { model: "gpt2", revision: "abc123", n_points: 3 },
    channels: [raw([1, 2, 3]), raw([3, null, 1], { id: "we_centroid_dist" })],
    ...over,
  });

  it("indexes channels by id and carries the pinned model id", () => {
    const set = parseChannelSet(doc(), "gpt2")!;
    expect(set.model).toBe("gpt2");
    expect(set.revision).toBe("abc123");
    expect(set.nPoints).toBe(3);
    expect([...set.byId.keys()]).toEqual(["we_norm", "we_centroid_dist"]);
  });

  it("drops the individual channels it cannot read, keeping the rest", () => {
    const set = parseChannelSet(
      doc({ channels: [raw([1, 2, 3]), raw([1, 2], { id: "short" })] }),
      "gpt2",
    )!;
    expect([...set.byId.keys()]).toEqual(["we_norm"]);
  });

  it("returns null for a document with nothing renderable in it", () => {
    expect(parseChannelSet(null, "gpt2")).toBeNull();
    expect(parseChannelSet({}, "gpt2")).toBeNull();
    expect(parseChannelSet(doc({ meta: { n_points: 0 } }), "gpt2")).toBeNull();
    expect(parseChannelSet(doc({ channels: [] }), "gpt2")).toBeNull();
  });
});

describe("channelValue / lowestBy", () => {
  const ch: Channel = parseChannel(raw([3, null, 1, 2]), 4)!;

  it("returns null rather than NaN for a point with no value", () => {
    expect(channelValue(ch, 0)).toBe(3);
    expect(channelValue(ch, 1)).toBeNull();
    expect(channelValue(ch, 99)).toBeNull();
    expect(channelValue(null, 0)).toBeNull();
  });

  it("excludes missing points from the ranking instead of sorting them first", () => {
    expect(lowestBy(ch, 4)).toEqual([
      { index: 2, value: 1 },
      { index: 3, value: 2 },
      { index: 0, value: 3 },
    ]);
    expect(lowestBy(ch, 2).map((r) => r.index)).toEqual([2, 3]);
    expect(lowestBy(ch, 0)).toEqual([]);
  });
});

/* ── 3. the GPU's ramp coordinate, mirrored on the CPU ────────────────────── */

describe("channelRampT", () => {
  it("maps the measured range onto 0–1 and clamps outside it", () => {
    expect(channelRampT(1, 1, 3)).toBe(0);
    expect(channelRampT(2, 1, 3)).toBe(0.5);
    expect(channelRampT(3, 1, 3)).toBe(1);
    expect(channelRampT(9, 1, 3)).toBe(1);
    expect(channelRampT(-9, 1, 3)).toBe(0);
  });

  it("gives a point with no value no position on the scale at all", () => {
    expect(channelRampT(CHANNEL_MISSING, 1, 3)).toBeNull();
    expect(channelRampT(Number.NaN, 1, 3)).toBeNull();
  });

  it("uses a sentinel no real measurement can reach", () => {
    // the shader tests `value <= CHANNEL_MISSING / 2`; a real value would have
    // to be about -1.7e38 to be mistaken for absence
    expect(CHANNEL_MISSING).toBeLessThan(-1e38);
  });

  it("does not explode on a degenerate range", () => {
    expect(channelRampT(2, 2, 2)).toBe(0);
  });
});

/* ── 4. fetching: every failure is "this map has no channels" ─────────────── */

function stubFetch(impl: (url: string) => Promise<unknown> | unknown) {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => impl(String(url))));
}

async function settle() {
  for (let i = 0; i < 8; i++) await Promise.resolve();
}

describe("ensureChannels", () => {
  const good = {
    meta: { model: "gpt2", revision: "abc", n_points: 3 },
    channels: [raw([1, 2, 3])],
  };

  it("loads once and records that it looked", async () => {
    stubFetch(() => ({ ok: true, json: async () => good }));
    expect(channelsLoaded("gpt2")).toBe(false);
    ensureChannels("gpt2", 3, "/data");
    await settle();
    expect(channelsLoaded("gpt2")).toBe(true);
    expect(channelsFor("gpt2")!.channels).toHaveLength(1);
    expect(channelFor("gpt2", "we_norm")!.space).toBe("W_E.raw");

    ensureChannels("gpt2", 3, "/data");
    expect((globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(1);
  });

  it("records absence when there is no sidecar", async () => {
    stubFetch(() => ({ ok: false, status: 404, json: async () => ({}) }));
    ensureChannels("gpt2", 3, "/data");
    await settle();
    expect(channelsLoaded("gpt2")).toBe(true);
    expect(channelsFor("gpt2")).toBeNull();
  });

  it("never throws on a network failure", async () => {
    stubFetch(() => {
      throw new Error("offline");
    });
    ensureChannels("gpt2", 3, "/data");
    await settle();
    expect(channelsFor("gpt2")).toBeNull();
  });

  it("drops a sidecar aligned to a different build of the map", async () => {
    stubFetch(() => ({ ok: true, json: async () => good }));
    ensureChannels("gpt2", 4, "/data");
    await settle();
    expect(channelsFor("gpt2")).toBeNull();
  });

  it("distinguishes not-asked-yet from asked-and-absent", async () => {
    expect(channelsLoaded("gpt2")).toBe(false);
    expect(channelsFor("gpt2")).toBeNull();
    stubFetch(() => ({ ok: false, status: 404, json: async () => ({}) }));
    ensureChannels("gpt2", 3, "/data");
    await settle();
    expect(channelsLoaded("gpt2")).toBe(true);
    expect(channelsFor("gpt2")).toBeNull();
  });
});

/* ── 5. the real artifact, when it is on this machine ─────────────────────── */

/** Skipped rather than failed when `out/` is not linked: this asserts a
 *  property of a 1 MB build product that a fresh clone does not have. The same
 *  numbers are asserted against the Python writer in `tests/test_channels.py`,
 *  which does have the artifact when the build has been run. */
const hasArtifact = existsSync(GPT2_CHANNELS);

describe.skipIf(!hasArtifact)("out/gpt2/channels.json", () => {
  const doc = hasArtifact ? JSON.parse(readFileSync(GPT2_CHANNELS, "utf8")) : null;

  it("parses into the two phase-0 channels over 49,857 points", () => {
    const set = parseChannelSet(doc, "gpt2")!;
    expect(set.nPoints).toBe(49857);
    expect(set.model).toBe("gpt2");
    expect(set.revision).toBe("607a30d783dfa663caf39e06633721c8d4cfcd7e");
    expect([...set.byId.keys()].sort()).toEqual(["we_centroid_dist", "we_norm"]);
    for (const ch of set.channels) {
      expect(ch.space).toBe("W_E.raw");
      expect(ch.fidelity).toBe("deterministic");
      expect(ch.stats.n_missing).toBe(0);
    }
  });

  it("finds the knot by centroid distance and NOT by norm", () => {
    const set = parseChannelSet(doc, "gpt2")!;
    const norm = set.byId.get("we_norm")!;
    const dist = set.byId.get("we_centroid_dist")!;

    // the shortest rows are ordinary function words — length alone is a
    // frequency story, and this is the assertion that says so
    const shortest = lowestBy(norm, 10).map((r) => r.value);
    expect(shortest[0]).toBeCloseTo(2.454, 2);
    expect(shortest[9]).toBeLessThan(2.54);

    // the gap: the single nearest row sits a full 1.0 below the 1st percentile
    const near = lowestBy(dist, 15);
    expect(near[0]!.value).toBeCloseTo(1.534, 2);
    expect(near[10]!.value).toBeLessThan(1.62);
    expect(near[14]!.value).toBeLessThan(2.0);

    // …and the eleven under 1.62 sit in a band a thirtieth of a unit wide
    expect(near[10]!.value - near[0]!.value).toBeLessThan(0.05);

    // the two quantities are NOT each other rescaled
    const byNorm = new Set(lowestBy(norm, 15).map((r) => r.index));
    const byDist = new Set(lowestBy(dist, 15).map((r) => r.index));
    const shared = [...byDist].filter((i) => byNorm.has(i));
    expect(shared).toHaveLength(0);
  });
});
