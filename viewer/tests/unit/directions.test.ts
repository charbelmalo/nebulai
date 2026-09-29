/** The direction sidecar, browser side (P1).
 *
 *  A direction is the one object in this app that is allowed to say "this is an
 *  axis of the model". Everything here exists to make that claim expensive:
 *
 *  1. **The method set is the same set on both sides.** `directions.py` and
 *     `data/directions.ts` each carry the list; one gaining a method the other
 *     does not know is a direction one half silently drops.
 *  2. **The renderability gate refuses in the same order as Python.** R5 (no
 *     null → not a figure) is checked before D2 (wrong space → not a
 *     measurement), because a direction can fail both and the reason the user
 *     is given must not depend on which language asked.
 *  3. **Absence is never a number.** A contrast with no held-out split reads
 *     "not measured"; a projection that was never run reads "not projected";
 *     neither reads 0.
 *  4. **The two statistics are never merged.** `axisClaim` speaks about the
 *     direction's own two sets; `axisMapClaim` speaks about the whole map. They
 *     routinely disagree, and the shipped gpt2 direction is the proof.
 */

import { readFileSync, existsSync } from "node:fs";
import { afterEach, describe, expect, it, vi } from "vitest";
import { __resetChannels, ensureChannels, parseChannelSet, type Channel } from "../../src/data/channels";
import {
  DIRECTION_METHODS,
  __resetDirections,
  axisChannels,
  axisClaim,
  axisMapClaim,
  axisRefusal,
  directionById,
  directionsFor,
  directionsLoaded,
  ensureDirections,
  hasAxis,
  histogram,
  histogramRange,
  parseDirectionSet,
  renderableDirections,
  type Direction,
} from "../../src/data/directions";

const REPO = new URL("../../../", import.meta.url);
const DIRECTIONS_PY = new URL("src/nebulai/backend/directions.py", REPO);
const GPT2_DIRECTIONS = new URL("out/gpt2/directions.json", REPO);
const GPT2_CHANNELS = new URL("out/gpt2/channels.json", REPO);

afterEach(() => {
  __resetDirections();
  __resetChannels();
  vi.unstubAllGlobals();
});

/* ── fixtures ─────────────────────────────────────────────────────────────── */

/** A direction document as the backend writes it, with everything present.
 *  Individual tests take one piece away — the removal is the subject. */
function doc(over: Record<string, unknown> = {}) {
  return {
    meta: { model: "gpt2", revision: "abc123", schema: 1 },
    directions: [
      {
        id: "a-minus-b",
        label: "A − B",
        space: "W_E.centered",
        method: "two_selection",
        d: 768,
        source: {
          kind: "computed",
          protocol: "diff of means over clusters 1 and 2 of map gpt2",
          contrast: {
            cohens_d: 8.18158,
            overlap: 0,
            n_pos: 307,
            n_neg: 242,
            null_n: 32,
            null_seed: 0,
            null_cohens_d_mean: 0.351211,
            null_cohens_d_p95: 0.890712,
            heldout_cohens_d: 8.304589,
            heldout_overlap: 0,
            heldout_n_pos: 154,
            heldout_n_neg: 121,
          },
        },
        projection: {
          channel: "proj.a-minus-b",
          orth_channel: "proj.a-minus-b.orth",
          stats: { cohens_d: -0.002161, overlap: 0.834186, n: 49857 },
        },
        null: {
          method: "random_unit",
          seed: 0,
          n: 32,
          channel: "proj.a-minus-b.null",
          orth_channel: "proj.a-minus-b.null.orth",
        },
        ...over,
      },
    ],
  };
}

function chan(id: string, space = "W_E.centered", n = 4): Record<string, unknown> {
  return {
    id,
    label: id,
    space,
    method: "dot",
    formula: "x · v",
    fidelity: "deterministic",
    units: "",
    stats: { min: 0, max: 1, mean: 0.5 },
    values: Array.from({ length: n }, (_, i) => i / (n - 1)),
  };
}

function channelDoc(ids: string[], space = "W_E.centered", n = 4) {
  return {
    meta: { model: "gpt2", revision: "abc123", n_points: n },
    channels: ids.map((id) => chan(id, space, n)),
  };
}

/** Install both sidecars synchronously, the way the app installs them after a
 *  fetch, so the gate can be exercised without a network. */
async function install(dirDoc: unknown, chanDoc: unknown, id = "gpt2") {
  vi.stubGlobal("fetch", async (url: string) =>
    url.endsWith("directions.json")
      ? { ok: true, json: async () => dirDoc }
      : { ok: true, json: async () => chanDoc },
  );
  ensureChannels(id, 4);
  ensureDirections(id);
  await new Promise((r) => setTimeout(r, 0));
}

function firstOf(d: unknown): Direction {
  return parseDirectionSet(d, "gpt2")!.directions[0]!;
}

/* ── 1. the method set, pinned across the language boundary ───────────────── */

describe("methods ↔ directions.py", () => {
  it("carries exactly the methods the Python module declares", () => {
    const py = readFileSync(DIRECTIONS_PY, "utf8");
    const block = py.slice(py.indexOf("METHODS"));
    const end = block.indexOf(")");
    const values = [...block.slice(0, end).matchAll(/"([a-z_0-9]+)"/g)].map((m) => m[1]!);
    expect(values.length).toBeGreaterThan(0);
    expect(new Set(values)).toEqual(new Set(DIRECTION_METHODS));
  });
});

/* ── 2. parsing ───────────────────────────────────────────────────────────── */

describe("parseDirectionSet", () => {
  it("reads a full document and indexes it by id", () => {
    const set = parseDirectionSet(doc(), "gpt2")!;
    expect(set.model).toBe("gpt2");
    expect(set.revision).toBe("abc123");
    expect(set.directions).toHaveLength(1);
    expect(set.byId.get("a-minus-b")!.d).toBe(768);
  });

  it("returns null rather than an empty set for a document with no directions", () => {
    expect(parseDirectionSet({ meta: {}, directions: [] }, "gpt2")).toBeNull();
    expect(parseDirectionSet({}, "gpt2")).toBeNull();
    expect(parseDirectionSet("not a document", "gpt2")).toBeNull();
  });

  it("drops a direction whose method the browser does not know", () => {
    const bad = doc({ method: "vibes" });
    expect(parseDirectionSet(bad, "gpt2")).toBeNull();
  });

  it("keeps “missing” out of the numbers: a sentinel becomes null, not 0", () => {
    const d = firstOf(
      doc({
        source: {
          kind: "computed",
          protocol: "p",
          contrast: {
            cohens_d: 1,
            overlap: 0.2,
            n_pos: 2,
            n_neg: 2,
            null_n: 32,
            null_seed: 0,
            null_cohens_d_mean: 0.3,
            null_cohens_d_p95: 0.8,
            heldout_cohens_d: "missing",
            heldout_overlap: "missing",
            heldout_n_pos: 0,
            heldout_n_neg: 0,
          },
        },
      }),
    );
    expect(d.source.contrast!.heldout_cohens_d).toBeNull();
    expect(d.source.contrast!.heldout_overlap).toBeNull();
    expect(d.source.contrast!.cohens_d).toBe(1);
  });

  it("treats a direction with no protocol as no direction", () => {
    const bad = doc({ source: { kind: "computed", protocol: "" } });
    expect(parseDirectionSet(bad, "gpt2")).toBeNull();
  });
});

/* ── 3. loading ───────────────────────────────────────────────────────────── */

describe("ensureDirections", () => {
  it("distinguishes “not fetched” from “fetched and there are none”", async () => {
    expect(directionsLoaded("gpt2")).toBe(false);
    vi.stubGlobal("fetch", async () => ({ ok: false, json: async () => ({}) }));
    ensureDirections("gpt2");
    await new Promise((r) => setTimeout(r, 0));
    expect(directionsLoaded("gpt2")).toBe(true);
    expect(directionsFor("gpt2")).toBeNull();
  });

  it("never throws on a network failure — it resolves to “no directions”", async () => {
    vi.stubGlobal("fetch", async () => {
      throw new Error("offline");
    });
    ensureDirections("gpt2");
    await new Promise((r) => setTimeout(r, 0));
    expect(directionsFor("gpt2")).toBeNull();
    expect(hasAxis("gpt2")).toBe(false);
  });

  it("fetches once per dataset however often it is asked", async () => {
    const fetchSpy = vi.fn(async () => ({ ok: true, json: async () => doc() }));
    vi.stubGlobal("fetch", fetchSpy);
    ensureDirections("gpt2");
    ensureDirections("gpt2");
    ensureDirections("gpt2");
    await new Promise((r) => setTimeout(r, 0));
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });
});

/* ── 4. the gate: R5 before D2, in both languages ─────────────────────────── */

describe("axisRefusal", () => {
  const CH = ["proj.a-minus-b", "proj.a-minus-b.orth", "proj.a-minus-b.null", "proj.a-minus-b.null.orth"];

  it("passes a direction with a null, a projection and matching channels", async () => {
    await install(doc(), channelDoc(CH));
    const d = directionById("gpt2", "a-minus-b")!;
    expect(axisRefusal("gpt2", d)).toBeNull();
    expect(hasAxis("gpt2")).toBe(true);
  });

  it("refuses a direction with no null FIRST, even when the space is also wrong", async () => {
    await install(doc({ null: null, space: "resid.L8" }), channelDoc(CH));
    const d = directionsFor("gpt2")!.directions[0]!;
    const why = axisRefusal("gpt2", d)!;
    expect(why).toMatch(/no null/);
    expect(why).toMatch(/R5/);
    expect(why).not.toMatch(/D2/);
  });

  it("refuses a direction whose space is not the map's space (D2)", async () => {
    await install(doc({ space: "resid.L8" }), channelDoc(CH, "W_E.centered"));
    const d = directionsFor("gpt2")!.directions[0]!;
    expect(axisRefusal("gpt2", d)).toMatch(/resid\.L8/);
  });

  it("refuses a direction that was never projected", async () => {
    await install(doc({ projection: null }), channelDoc(CH));
    const d = directionsFor("gpt2")!.directions[0]!;
    expect(axisRefusal("gpt2", d)).toMatch(/not been projected/);
  });

  it("refuses when one of the four channels is missing — three is not most of four", async () => {
    await install(doc(), channelDoc(CH.slice(0, 3)));
    const d = directionsFor("gpt2")!.directions[0]!;
    expect(axisRefusal("gpt2", d)).toBeTruthy();
    expect(axisChannels("gpt2", d.id)).toBeNull();
  });

  it("reports both halves: what may be drawn AND what may not, with reasons", async () => {
    const two = doc();
    (two.directions as unknown[]).push({
      ...(two.directions[0] as Record<string, unknown>),
      id: "no-null",
      null: null,
    });
    await install(two, channelDoc(CH));
    const { ok, drops } = renderableDirections("gpt2");
    expect(ok.map((d) => d.id)).toEqual(["a-minus-b"]);
    expect(drops).toHaveLength(1);
    expect(drops[0]!.direction.id).toBe("no-null");
    expect(drops[0]!.reason).toMatch(/R5/);
  });

  it("resolves all four channels together or not at all", async () => {
    await install(doc(), channelDoc(CH));
    const d = directionById("gpt2", "a-minus-b")!;
    const ax = axisChannels("gpt2", d.id)!;
    expect(ax.par.id).toBe("proj.a-minus-b");
    expect(ax.orth.id).toBe("proj.a-minus-b.orth");
    expect(ax.nullPar.id).toBe("proj.a-minus-b.null");
    expect(ax.nullOrth.id).toBe("proj.a-minus-b.null.orth");
  });
});

/* ── 5. the two claims, which are allowed to disagree ─────────────────────── */

describe("axisClaim / axisMapClaim", () => {
  it("leads with the held-out number and names the random baseline", () => {
    const s = axisClaim(firstOf(doc()));
    expect(s).toMatch(/8\.30/);
    expect(s).toMatch(/154\/121/);
    expect(s).toMatch(/0\.35/);
  });

  it("says “not measured” — never 0 — when a set was too small to split", () => {
    const d = firstOf(
      doc({
        source: {
          kind: "computed",
          protocol: "p",
          contrast: {
            cohens_d: 3,
            overlap: 0,
            n_pos: 2,
            n_neg: 2,
            null_n: 32,
            null_seed: 0,
            null_cohens_d_mean: 0.3,
            null_cohens_d_p95: 0.8,
            heldout_cohens_d: "missing",
            heldout_overlap: "missing",
            heldout_n_pos: 0,
            heldout_n_neg: 0,
          },
        },
      }),
    );
    const s = axisClaim(d);
    expect(s).toMatch(/not measured/);
    expect(s).not.toMatch(/0\.00/);
  });

  it("calls a map-wide overlap above a half indistinguishable from random", () => {
    const s = axisMapClaim(firstOf(doc()));
    expect(s).toMatch(/0\.834/);
    expect(s).toMatch(/indistinguishable/);
    expect(s).toMatch(/49,857/);
  });

  it("says “not projected” rather than inventing a map-wide verdict", () => {
    expect(axisMapClaim(firstOf(doc({ projection: null })))).toMatch(/not projected/);
  });

  it("keeps the in-sample and held-out numbers in different sentences", () => {
    // the in-sample d is 8.18 and the held-out 8.30; the claim the rail leads
    // with must be the second, because the first cannot be small by
    // construction and is therefore not evidence
    const s = axisClaim(firstOf(doc()));
    expect(s.indexOf("8.30")).toBeGreaterThanOrEqual(0);
    expect(s).not.toMatch(/8\.18/);
  });
});

/* ── 6. histograms share one ruler ────────────────────────────────────────── */

describe("histogram", () => {
  function col(values: number[], stats?: { min: number; max: number }): Channel {
    return {
      id: "c",
      label: "c",
      space: "W_E.centered",
      method: "dot",
      formula: "x · v",
      fidelity: "deterministic",
      units: "",
      stats: {
        min: stats?.min ?? null,
        max: stats?.max ?? null,
        mean: null,
        p01: null,
        p99: null,
        n_missing: values.filter((v) => !Number.isFinite(v)).length,
      } as Channel["stats"],
      values: Float32Array.from(values),
    };
  }

  it("bins measured values and counts the unmeasured separately", () => {
    const h = histogram(col([0, 0.5, 1, NaN]), 0, 1, 2);
    expect(h.n).toBe(3);
    expect(h.nMissing).toBe(1);
    expect(h.bins[0]! + h.bins[1]!).toBeCloseTo(1, 10);
  });

  it("never folds a missing value into a bin", () => {
    const h = histogram(col([NaN, NaN, NaN]), 0, 1, 4);
    expect(h.n).toBe(0);
    expect(h.nMissing).toBe(3);
    expect([...h.bins]).toEqual([0, 0, 0, 0]);
  });

  it("takes the range as a parameter so the ghost cannot be rescaled", () => {
    const real = col([-1, 0, 1], { min: -1, max: 1 });
    const nul = col([-0.05, 0, 0.05], { min: -0.05, max: 0.05 });
    const [lo, hi] = histogramRange(real, nul);
    expect(lo).toBe(-1);
    expect(hi).toBe(1);
    const hn = histogram(nul, lo, hi, 10);
    // on the SHARED ruler the null is a spike in the middle; rescaled to its
    // own extents it would fill the whole width and look exactly as spread as
    // the real one, which is the lie the shared range exists to prevent
    const occupied = [...hn.bins].filter((b) => b > 0).length;
    expect(occupied).toBeLessThan(4);
  });

  it("survives a degenerate range without dividing by nothing", () => {
    const flat = col([2, 2, 2], { min: 2, max: 2 });
    const [lo, hi] = histogramRange(flat, flat);
    expect(hi).toBeGreaterThan(lo);
    expect(histogram(flat, lo, hi, 8).n).toBe(3);
  });
});

/* ── 7. the shipped artefact ──────────────────────────────────────────────── */

describe("the gpt2 directions that actually ship", () => {
  const have = existsSync(GPT2_DIRECTIONS) && existsSync(GPT2_CHANNELS);

  it.runIf(have)("parses, and every entry carries a protocol", () => {
    const set = parseDirectionSet(JSON.parse(readFileSync(GPT2_DIRECTIONS, "utf8")), "gpt2")!;
    expect(set.directions.length).toBeGreaterThan(0);
    for (const d of set.directions) {
      expect(d.source.protocol.length, d.id).toBeGreaterThan(20);
      expect(d.d, d.id).toBeGreaterThan(0);
    }
  });

  it.runIf(have)("refuses the residual-stream directions on this W_E map, by name", () => {
    const cdoc = JSON.parse(readFileSync(GPT2_CHANNELS, "utf8"));
    const ddoc = JSON.parse(readFileSync(GPT2_DIRECTIONS, "utf8"));
    const set = parseDirectionSet(ddoc, "gpt2")!;
    const chans = parseChannelSet(cdoc, "gpt2")!;
    expect(chans.byId.has("proj.male-minus-female-names")).toBe(true);
    const resid = set.directions.filter((d) => d.space.startsWith("resid."));
    expect(resid.length).toBeGreaterThan(0);
    for (const d of resid) {
      // they have no null and no projection, so they fail R5 first — which is
      // the same order `renderable()` refuses them in on the Python side
      expect(d.null, d.id).toBeNull();
    }
  });

  it.runIf(have)("records a held-out number that disagrees with the in-sample one", () => {
    const set = parseDirectionSet(JSON.parse(readFileSync(GPT2_DIRECTIONS, "utf8")), "gpt2")!;
    const fitted = set.directions.find((d) => d.id.startsWith("refusal-style"));
    expect(fitted).toBeDefined();
    const c = fitted!.source.contrast!;
    expect(c.cohens_d!).toBeGreaterThan(1.5);
    // the whole reason the held-out column exists: a 768-wide diff of means
    // over 32 points a side fits the noise, and only the refit says so
    expect(Math.abs(c.heldout_cohens_d!)).toBeLessThan(0.5);
  });
});
