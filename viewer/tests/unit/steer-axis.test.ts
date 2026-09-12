/** The phase-1 axis link out of the steer rail, and its TWO separate refusals.
 *
 *  `SteerRail`'s docstring used to end with a paragraph saying the panel could
 *  not slide a point along an axis because no bundle carried a projection of the
 *  intervened stream onto a direction. `intervene.py` now emits one for the two
 *  direction verbs, so the paragraph became a feature — and the feature's whole
 *  difficulty is that it can fail in two unrelated places:
 *
 *  1. **the intervention may have no axis at all.** `clamp` pins an SAE feature,
 *     `cap` clips a box; neither names a direction. The producer refuses in its
 *     own words and those words are what the rail shows.
 *  2. **the MAP may not be able to lay out along a direction the intervention
 *     measured perfectly.** This is D2, and it is not hypothetical: the only
 *     direction-backed sweep this repo can produce on gpt2 uses
 *     `refusal-style-v1-L8` (`resid.L8`), while the gpt2 map's points are
 *     `W_E.centered` token embeddings. The numbers are real; the map link is
 *     not available; both have to be said at once.
 *
 *  A single "axis unavailable" would collapse those two into one and teach the
 *  reader nothing about which wall was hit. So the gate returns the producer's
 *  sentence for the first and `data/directions.ts`'s sentence for the second,
 *  and these tests pin that they never swap places.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { __resetChannels, ensureChannels } from "../../src/data/channels";
import { __resetDirections, ensureDirections } from "../../src/data/directions";
import { loadSteerBundle, __resetInterpCache } from "../../src/data/interp";
import { steerAxis, runAxis } from "../../src/scene/interp/steer";
import type {
  InterveneBundle,
  InterveneRow,
  InterveneRowAxis,
  InterveneRun,
} from "../../src/data/interp";

afterEach(() => {
  __resetDirections();
  __resetChannels();
  __resetInterpCache();
  vi.unstubAllGlobals();
});

/* ── fixtures ─────────────────────────────────────────────────────────────── */

function side(tokens: string[]) {
  return {
    top: [] as [string, number][],
    text: tokens.join(""),
    tokens,
    ids: tokens.map((_, i) => i),
    logprobs: tokens.map(() => -1),
    mean_logprob: -1,
    decoding: "greedy",
  };
}

function run(axis: InterveneRun["axis"]): InterveneRun {
  return {
    prompt: "p",
    intervention: {
      verb: "add",
      layer: 8,
      alpha: 1,
      is_identity: false,
      protocol: "add 1·d at layer 8",
      direction_id: "refusal-style-v1-L8",
      space: "resid.L8",
    },
    identical: false,
    kl_bits: 1,
    resid_norm_baseline: 500,
    resid_norm_intervened: 505,
    axis,
    baseline: side([" a", " b"]),
    intervened: side([" a", " c"]),
  };
}

function rowAxis(over: Partial<InterveneRowAxis> = {}): InterveneRowAxis {
  return {
    direction_id: "refusal-style-v1-L8",
    space: "resid.L8",
    layer: 8,
    n_prompts: 2,
    proj_mean_baseline: 1.5,
    proj_mean_intervened: 2.5,
    proj_mean_delta: 1.0,
    orth_mean_delta: 0.0,
    ...over,
  };
}

function row(alpha: number, axis: InterveneRowAxis | null, refused?: string): InterveneRow {
  const r: InterveneRow = {
    alpha,
    is_identity: alpha === 0,
    protocol: `add ${alpha}·d at layer 8`,
    kl_bits_mean: alpha,
    kl_bits_max: alpha,
    identical_to_baseline: alpha === 0,
    // the run-level block is the full per-prompt measurement; the row's is its
    // mean over prompts, and they are different shapes on purpose
    runs: [
      run(
        axis
          ? {
              direction_id: axis.direction_id,
              space: axis.space,
              layer: axis.layer,
              resid_index: axis.layer + 1,
              observed_where: `hook layer ${axis.layer}`,
              token: "last",
              proj_baseline: axis.proj_mean_baseline,
              proj_intervened: axis.proj_mean_intervened,
              proj_delta: axis.proj_mean_delta,
              orth_norm_baseline: 10,
              orth_norm_intervened: 10 + axis.orth_mean_delta,
              orth_norm_delta: axis.orth_mean_delta,
              unit: "residual-stream units",
            }
          : { refused: refused ?? "no axis" },
      ),
    ],
  };
  if (axis) r.axis = axis;
  if (refused) r.axis_refused = refused;
  return r;
}

function bundle(rows: InterveneRow[], verb = "add"): InterveneBundle {
  return {
    kind: "intervention_sweep",
    model: "gpt2",
    verb,
    n_layer: 12,
    d_model: 768,
    alphas: rows.map((r) => r.alpha),
    prompts: ["prompt 0"],
    max_tokens: 8,
    rows,
    claim: "Under this protocol …",
    notes: { decoding: "greedy", control: "α = 0 installs no hook", d6: "no weights written" },
    meta: { generated: "", revision: "", digest: "" },
  };
}

/** Both sidecars, installed the way the app installs them after a fetch. */
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

function chan(id: string, space: string) {
  return {
    id,
    label: id,
    space,
    method: "dot",
    formula: "x · v",
    fidelity: "deterministic",
    units: "",
    stats: { min: 0, max: 1, mean: 0.5 },
    values: [0, 0.25, 0.5, 1],
  };
}

/** A direction document whose single entry is fully renderable, in `space`. */
function dirDoc(space: string, id = "refusal-style-v1-L8") {
  return {
    meta: { model: "gpt2", revision: "abc123", schema: 1 },
    directions: [
      {
        id,
        label: "refusal style",
        space,
        method: "diff_of_means",
        d: 768,
        source: { kind: "computed", protocol: "diff of means over two prompt sets" },
        projection: {
          channel: `proj.${id}`,
          orth_channel: `proj.${id}.orth`,
          stats: { cohens_d: 0.1, overlap: 0.9, n: 4 },
        },
        null: {
          method: "random_unit",
          seed: 0,
          n: 32,
          channel: `proj.${id}.null`,
          orth_channel: `proj.${id}.null.orth`,
        },
      },
    ],
  };
}

function chanDoc(space: string, id = "refusal-style-v1-L8") {
  return {
    meta: { model: "gpt2", revision: "abc123", n_points: 4 },
    channels: [
      chan(`proj.${id}`, space),
      chan(`proj.${id}.orth`, space),
      chan(`proj.${id}.null`, space),
      chan(`proj.${id}.null.orth`, space),
    ],
  };
}

/* ── gate one: did the intervention measure an axis at all ─────────────────── */

describe("the producer's half of the gate", () => {
  it("refuses a bundle with no axis block and says it was produced before one existed", () => {
    const b = bundle([row(0, null, undefined), row(1, null, undefined)]);
    // no axis, no axis_refused — an older bundle
    for (const r of b.rows) delete r.runs[0]!.axis;
    const ax = steerAxis(b, "gpt2", { row: 1, col: 0 });
    expect(ax.ready).toBe(false);
    if (!ax.ready) expect(ax.reason).toMatch(/carries no axis measurement/);
  });

  it("passes the producer's OWN refusal through, verbatim, for a clamp sweep", async () => {
    const why =
      "the clamp verb carries no direction, so there is no axis to project onto " +
      "— a projection here would be onto some other direction that happened to be in scope";
    const b = bundle([row(0, null, why), row(1, null, why)], "clamp");
    const ax = steerAxis(b, "gpt2", { row: 1, col: 0 });
    expect(ax.ready).toBe(false);
    if (!ax.ready) expect(ax.reason).toBe(why);
  });

  it("finds a run-level refusal even when the row level has none", () => {
    const b = bundle([row(1, null, undefined)]);
    b.rows[0]!.runs[0]!.axis = { refused: "the cap verb carries no direction" };
    const ax = steerAxis(b, "gpt2", { row: 0, col: 0 });
    expect(ax.ready).toBe(false);
    if (!ax.ready) expect(ax.reason).toMatch(/cap verb/);
  });

  it("refuses when no bundle is loaded at all", () => {
    const ax = steerAxis(null, "gpt2");
    expect(ax.ready).toBe(false);
  });
});

/* ── the measurement itself ────────────────────────────────────────────────── */

describe("the measured travel", () => {
  it("carries one stop per alpha, in the bundle's order, with the identity marked", async () => {
    await install(dirDoc("resid.L8"), chanDoc("resid.L8"));
    const b = bundle([
      row(0, rowAxis({ proj_mean_intervened: 1.5, proj_mean_delta: 0 })),
      row(1, rowAxis()),
      row(2, rowAxis({ proj_mean_intervened: 3.5, proj_mean_delta: 2 })),
    ]);
    const ax = steerAxis(b, "gpt2", { row: 2, col: 0 });
    expect(ax.ready).toBe(true);
    if (!ax.ready) return;
    expect(ax.travel.map((t) => t.alpha)).toEqual([0, 1, 2]);
    expect(ax.travel.map((t) => t.projDelta)).toEqual([0, 1, 2]);
    expect(ax.travel[0]!.isIdentity).toBe(true);
    expect(ax.travel[2]!.isIdentity).toBe(false);
    expect(ax.directionId).toBe("refusal-style-v1-L8");
    expect(ax.space).toBe("resid.L8");
    // the HOOK layer, not the artifact's
    expect(ax.layer).toBe(8);
  });

  it("points `here` at the selected row's own stop", async () => {
    await install(dirDoc("resid.L8"), chanDoc("resid.L8"));
    const b = bundle([row(0, rowAxis({ proj_mean_delta: 0 })), row(2, rowAxis({ proj_mean_delta: 2 }))]);
    const first = steerAxis(b, "gpt2", { row: 0, col: 0 });
    const last = steerAxis(b, "gpt2", { row: 1, col: 0 });
    expect(first.ready && first.here?.projDelta).toBe(0);
    expect(last.ready && last.here?.projDelta).toBe(2);
  });

  it("has no `here` when nothing is selected", async () => {
    await install(dirDoc("resid.L8"), chanDoc("resid.L8"));
    const ax = steerAxis(bundle([row(1, rowAxis())]), "gpt2");
    expect(ax.ready && ax.here).toBe(null);
  });

  it("keeps the perpendicular movement beside the along-axis one", async () => {
    await install(dirDoc("resid.L8"), chanDoc("resid.L8"));
    // a point that moved 1 along the axis while its perpendicular part moved 40
    // did not travel along the axis in any useful sense; the number has to be
    // available for the rail to say so
    const b = bundle([row(1, rowAxis({ orth_mean_delta: 40 }))]);
    const ax = steerAxis(b, "gpt2", { row: 0, col: 0 });
    expect(ax.ready && ax.here?.orthDelta).toBe(40);
  });
});

/* ── gate two: can THIS map lay out along that direction ───────────────────── */

describe("the map's half of the gate (D2)", () => {
  it("offers the link when the direction is renderable on this map", async () => {
    await install(dirDoc("resid.L8"), chanDoc("resid.L8"));
    const ax = steerAxis(bundle([row(1, rowAxis())]), "gpt2", { row: 0, col: 0 });
    expect(ax.ready).toBe(true);
    if (!ax.ready) return;
    expect(ax.mapLink).toEqual({ directionId: "refusal-style-v1-L8" });
    expect(ax.mapRefusal).toBe(null);
  });

  it("shows the numbers and REFUSES the link when the map is in another space", async () => {
    // the real gpt2 case: the direction is resid.L8, the map's channels are
    // W_E.centered. D2 refuses, and the refusal is about the spaces.
    await install(dirDoc("resid.L8"), chanDoc("W_E.centered"));
    const ax = steerAxis(bundle([row(1, rowAxis())]), "gpt2", { row: 0, col: 0 });
    expect(ax.ready).toBe(true);
    if (!ax.ready) return;
    expect(ax.here?.projDelta).toBe(1); // the measurement survives
    expect(ax.mapLink).toBe(null); // the link does not
    expect(ax.mapRefusal).toMatch(/space mismatch/);
  });

  it("refuses the link when the direction has no null, in R5's words", async () => {
    const doc = dirDoc("resid.L8");
    delete (doc.directions[0] as Record<string, unknown>).null;
    await install(doc, chanDoc("resid.L8"));
    const ax = steerAxis(bundle([row(1, rowAxis())]), "gpt2", { row: 0, col: 0 });
    expect(ax.ready && ax.mapRefusal).toMatch(/has no null/);
  });

  it("refuses the link when this map's directions.json never heard of the id", async () => {
    await install(dirDoc("resid.L8", "some-other-direction"), chanDoc("resid.L8", "some-other-direction"));
    const ax = steerAxis(bundle([row(1, rowAxis())]), "gpt2", { row: 0, col: 0 });
    expect(ax.ready).toBe(true);
    if (!ax.ready) return;
    expect(ax.mapLink).toBe(null);
    expect(ax.mapRefusal).toMatch(/has no .refusal-style-v1-L8./);
  });

  it("never returns a link and a refusal together", async () => {
    for (const space of ["resid.L8", "W_E.centered"]) {
      __resetDirections();
      __resetChannels();
      await install(dirDoc("resid.L8"), chanDoc(space));
      const ax = steerAxis(bundle([row(1, rowAxis())]), "gpt2", { row: 0, col: 0 });
      if (!ax.ready) continue;
      expect((ax.mapLink === null) !== (ax.mapRefusal === null)).toBe(true);
    }
  });
});

/* ── the run-level accessor ────────────────────────────────────────────────── */

describe("runAxis", () => {
  it("returns the block when it is a measurement", () => {
    const r = run({
      direction_id: "d",
      space: "resid.L8",
      layer: 8,
      resid_index: 9,
      observed_where: "the last token's residual stream at hook layer 8",
      token: "last",
      proj_baseline: 1,
      proj_intervened: 2,
      proj_delta: 1,
      orth_norm_baseline: 10,
      orth_norm_intervened: 10,
      orth_norm_delta: 0,
      unit: "residual-stream units",
    });
    expect(runAxis(r)?.proj_delta).toBe(1);
  });

  it("returns null for a refusal rather than a zero-filled block", () => {
    expect(runAxis(run({ refused: "the cap verb carries no direction" }))).toBe(null);
  });

  it("returns null for an older run with no axis at all", () => {
    const r = run(undefined);
    delete r.axis;
    expect(runAxis(r)).toBe(null);
  });
});


/* ── which sweep the view opens ────────────────────────────────────────────
 *
 * The driver used to fetch one hard-coded name, so a direction-backed sweep
 * sitting beside it was never seen. The choice is now made on the one property
 * that matters to the rail — whether the sweep measured a position on a
 * direction — and these pin that a manifest entry with no file on disk is
 * skipped rather than fatal.
 */

describe("picking the sweep to open", () => {
  /** Serve an index plus a map of bundle name → body (or 404 when absent). */
  function serve(bundles: string[], bodies: Record<string, unknown>) {
    const seen: string[] = [];
    vi.stubGlobal("fetch", async (url: string) => {
      seen.push(url);
      if (url.endsWith("/index.json")) {
        return {
          ok: true,
          json: async () => ({
            meta: { model: "gpt2", created: "" },
            bundles,
            traces: [],
          }),
        };
      }
      for (const [name, body] of Object.entries(bodies)) {
        if (url.endsWith(`/${name}`)) return { ok: true, json: async () => body };
      }
      return { ok: false, status: 404, json: async () => ({}) };
    });
    return seen;
  }

  const withAxis = () => bundle([row(1, rowAxis())]);
  const withoutAxis = () =>
    bundle([row(1, null, "the clamp verb carries no direction")], "clamp");

  it("prefers the sweep that measured an axis over one that did not", async () => {
    serve(["weights.json", "intervene_golden_gate.json", "intervene_refusal_add.json"], {
      "intervene_golden_gate.json": withoutAxis(),
      "intervene_refusal_add.json": withAxis(),
    });
    const b = await loadSteerBundle("gpt2", "/data");
    expect(b.rows[0]!.axis?.direction_id).toBe("refusal-style-v1-L8");
  });

  it("falls back to the first sweep that loads when none measured an axis", async () => {
    serve(["intervene_golden_gate.json"], { "intervene_golden_gate.json": withoutAxis() });
    const b = await loadSteerBundle("gpt2", "/data");
    expect(b.rows[0]!.axis_refused).toMatch(/no direction/);
  });

  it("skips a manifest entry with no file behind it", async () => {
    serve(["intervene_gone.json", "intervene_refusal_add.json"], {
      "intervene_refusal_add.json": withAxis(),
    });
    const b = await loadSteerBundle("gpt2", "/data");
    expect(b.rows[0]!.axis?.direction_id).toBe("refusal-style-v1-L8");
  });

  it("falls back to the historical name when there is no manifest", async () => {
    vi.stubGlobal("fetch", async (url: string) =>
      url.endsWith("/index.json")
        ? { ok: false, status: 404, json: async () => ({}) }
        : { ok: true, json: async () => withAxis() },
    );
    const b = await loadSteerBundle("gpt2", "/data");
    expect(b.rows[0]!.axis?.direction_id).toBe("refusal-style-v1-L8");
  });

  it("throws rather than returning an empty sweep when nothing loads", async () => {
    serve(["intervene_gone.json"], {});
    await expect(loadSteerBundle("gpt2", "/data")).rejects.toThrow(/no intervention sweep/);
  });
});

