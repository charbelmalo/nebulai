/** The Behavior page's data layer, and the three claims its encoding makes.
 *
 *  These are not smoke tests. Each section pins a decision that, if reversed,
 *  would make the page state something the study does not support:
 *
 *  1. `missing` is not `0`. A cue whose effect could not be computed must come
 *     out of every accessor as "not measured" — never as a zero bar, a
 *     zero-radius dot or a false `q > threshold`.
 *  2. Effect is encoded by AREA. Doubling Δ̂ must NOT double the radius, or a
 *     reader comparing two marks over-reads the larger one by its square.
 *  3. Significance is three-valued. `null` means "could not tell", and
 *     collapsing that into `false` is the single most common way a screen of
 *     this kind lies.
 *
 *  The last section reads BehaviorPage.tsx as text. That is deliberate: the
 *  claim contract (docs/BEHAVIORAL-DIVERGENCE-PLAN.md §1) is about the words on
 *  the screen, and nothing else in the build would catch a well-typed component
 *  that called one model the winner.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeEach, describe, expect, it } from "vitest";
import { appStore } from "../../src/app/store";
import {
  cueMarkRadius,
  cueSignificant,
  fmtMetric,
  maxMeasuredEffect,
  notRunArms,
  projectIntoLandscape,
  searchCues,
  type BehaviorCue,
  type BehaviorData,
  type BehaviorLandscape,
} from "../../src/data/behavior";

function cue(over: Partial<BehaviorCue> = {}): BehaviorCue {
  return {
    cue: "kin",
    stratum: "kinship",
    pack: "matched",
    partition: "R",
    status: "measured",
    reasons: [],
    delta_hat: 0.04,
    mmd2_between: 0.09,
    within: {},
    p_value: 0.002,
    q_value: 0.01,
    ci: [0.02, 0.06],
    manski: [0.01, 0.08],
    jsd: 0.3,
    jsd_n: 40,
    rbo: 0.42,
    location: 0.02,
    dispersion: {},
    dominant: "location",
    capability_reference: 0.12,
    arms: {},
    ...over,
  };
}

// ── missing is not zero ────────────────────────────────────────────────────

describe("a metric that was not measured never becomes a number", () => {
  it("fmtMetric says the words, it does not print 0.000", () => {
    expect(fmtMetric(null)).toBe("not measured");
    expect(fmtMetric(0)).toBe("0.000");
  });

  it("a cue with no effect gets no mark radius at all, not a radius of zero", () => {
    // A zero-radius dot reads as "measured, and the effect was nothing". The
    // truth is "we have no measurement", which needs its own mark.
    expect(cueMarkRadius(cue({ status: "missing", delta_hat: null }), 0.2)).toBeNull();
    expect(cueMarkRadius(cue({ status: "gated", delta_hat: 0.05 }), 0.2)).toBeNull();
  });

  it("a gated cue is excluded from the effect scale", () => {
    // Otherwise one gated cue carrying a large raw number would shrink every
    // honestly-measured mark on the plot.
    const cues = [
      cue({ cue: "a", delta_hat: 0.05 }),
      cue({ cue: "b", status: "gated", delta_hat: 5.0 }),
      cue({ cue: "c", status: "missing", delta_hat: null }),
    ];
    expect(maxMeasuredEffect(cues)).toBeCloseTo(0.05, 10);
  });

  it("an empty study has a max effect of 0 and does not divide by it", () => {
    expect(maxMeasuredEffect([])).toBe(0);
    // minPx, not NaN and not Infinity
    expect(cueMarkRadius(cue(), 0, 2.5, 16)).toBe(2.5);
  });
});

// ── area, not radius ───────────────────────────────────────────────────────

describe("effect is encoded by area", () => {
  it("the drawn area, not the radius, is proportional to the effect", () => {
    const max = 1.0;
    const half = cueMarkRadius(cue({ delta_hat: 0.5 }), max, 0, 10)!;
    const full = cueMarkRadius(cue({ delta_hat: 1.0 }), max, 0, 10)!;
    // radius ratio is sqrt(2), so the AREA ratio is 2 — which is the ratio of
    // the two effects. Radius encoding would make it 4.
    expect(full / half).toBeCloseTo(Math.SQRT2, 6);
    expect((full * full) / (half * half)).toBeCloseTo(2, 6);
  });

  it("the sign of the effect does not change the mark size", () => {
    const a = cueMarkRadius(cue({ delta_hat: 0.3 }), 1)!;
    const b = cueMarkRadius(cue({ delta_hat: -0.3 }), 1)!;
    expect(a).toBeCloseTo(b, 12);
  });

  it("an effect larger than the scale is clamped rather than drawn off-plot", () => {
    expect(cueMarkRadius(cue({ delta_hat: 9 }), 1, 2, 16)).toBe(16);
  });

  it("the smallest measured effect still gets a visible mark", () => {
    expect(cueMarkRadius(cue({ delta_hat: 1e-9 }), 1, 2.5, 16)).toBeGreaterThanOrEqual(2.5);
  });
});

// ── three-valued significance ──────────────────────────────────────────────

describe("significance is three-valued, and indeterminate is not a soft no", () => {
  it("a measured cue at or below the threshold passes", () => {
    expect(cueSignificant(cue({ q_value: 0.05 }), 0.05)).toBe(true);
    expect(cueSignificant(cue({ q_value: 0.051 }), 0.05)).toBe(false);
  });

  it("a cue with no q is indeterminate, not insignificant", () => {
    expect(cueSignificant(cue({ q_value: null }), 0.05)).toBeNull();
  });

  it("a gated cue is indeterminate even when it carries a q", () => {
    // The gate fired for a reason the q does not know about — a compliance
    // parity failure, say. Reporting its q as a verdict would launder that.
    expect(cueSignificant(cue({ status: "gated", q_value: 0.001 }), 0.05)).toBeNull();
  });
});

// ── search: "daddy" in at most two actions ─────────────────────────────────

describe("finding a specific cue", () => {
  const cues = [
    cue({ cue: "daddy", stratum: "kinship", pack: "matched-daddy" }),
    cue({ cue: "mother", stratum: "kinship" }),
    cue({ cue: "river", stratum: "nature", arms: {} }),
  ];

  beforeEach(() => appStore.getState().setBehaviorCue(""));

  it('"daddy" is reachable in two actions: type, then open', () => {
    // Action 1: type into the search box.
    appStore.getState().setBehaviorQuery("daddy");
    const hits = searchCues(cues, appStore.getState().behavior.query);
    expect(hits.map((c) => c.cue)).toEqual(["daddy"]);
    // Action 2: click the single result.
    appStore.getState().setBehaviorCue(hits[0]!.cue);
    expect(appStore.getState().behavior.cue).toBe("daddy");
  });

  it("search is case-insensitive and matches the stratum and the pack too", () => {
    expect(searchCues(cues, "DADDY").map((c) => c.cue)).toEqual(["daddy"]);
    expect(searchCues(cues, "kinship").map((c) => c.cue)).toEqual(["daddy", "mother"]);
    expect(searchCues(cues, "matched-daddy").map((c) => c.cue)).toEqual(["daddy"]);
  });

  it("search also matches what a model actually said", () => {
    // A reader often remembers the associate, not the cue that produced it.
    const withArm = cue({
      cue: "harbour",
      arms: {
        a: {
          model_key: "a",
          n_attempted: 10,
          n_valid: 10,
          parse_rate: 1,
          distinct_types: 3,
          entropy: 1,
          type_token_ratio: 0.3,
          reliability: null,
          top_associates: ["quay", "daddy"],
          detectors: { cue_echo: 0, exemplar_echo: 0, within_trial_duplicate: 0, prompt_copy: 0 },
          oov_fragment_ratio: null,
        },
      },
    });
    expect(searchCues([withArm], "daddy").map((c) => c.cue)).toEqual(["harbour"]);
  });

  it("an empty query returns everything rather than nothing", () => {
    expect(searchCues(cues, "   ")).toHaveLength(3);
  });

  it("opening a cue keeps the query, so the result list survives the click", () => {
    appStore.getState().setBehaviorQuery("daddy");
    appStore.getState().setBehaviorCue("daddy");
    expect(appStore.getState().behavior.query).toBe("daddy");
  });
});

// ── the fixed projection ───────────────────────────────────────────────────

describe("the landscape transform is persisted so published cues never move", () => {
  const l: BehaviorLandscape = {
    method: "pca",
    dims: 2,
    coords: [],
    pca_mean: [1, 2, 3],
    // axis-major: axis 0 is [1,0,0], axis 1 is [0,1,0]
    pca_axes: [1, 0, 0, 0, 1, 0],
    pca_axes_shape: [2, 3],
    explained_variance_ratio: [0.4, 0.2],
    projection: {
      quantity: 0.6,
      quantity_label: "these 2 axes carry 60% of the variance in the cue-word embeddings",
      total_variance: 1,
      encoder: "test",
      encoder_revision: "",
      warning: "positions come from the cue words",
    },
  };

  it("projects a new point into the EXISTING space rather than refitting one", () => {
    expect(projectIntoLandscape(l, [1, 2, 3])).toEqual([0, 0]);
    expect(projectIntoLandscape(l, [4, 6, 3])).toEqual([3, 4]);
  });

  it("refuses a vector of the wrong width instead of silently truncating", () => {
    // Silently taking the first d components would place the cue somewhere
    // plausible-looking and wrong, which is worse than an error.
    expect(() => projectIntoLandscape(l, [1, 2])).toThrow(/2 dims/);
  });
});

// ── partial studies ────────────────────────────────────────────────────────

describe("an arm that never ran is surfaced, not hidden", () => {
  const data = {
    runs: [
      {
        run_id: "r1",
        started: "",
        finished: null,
        n_trials: 10,
        n_completed: 5,
        cost_usd: null,
        halted: null,
        not_run: { grok: "no XAI_API_KEY; 1800 trials at ~$12.3456 — rerun with --approve" },
      },
      { run_id: "r2", started: "", finished: null, n_trials: 0, n_completed: 0, cost_usd: null, halted: null, not_run: {} },
    ],
  } as unknown as BehaviorData;

  it("carries the WHOLE refusal text, not its first clause", () => {
    const arms = notRunArms(data);
    expect(arms).toHaveLength(1);
    expect(arms[0]!.reason).toContain("XAI_API_KEY");
    expect(arms[0]!.reason).toContain("--approve");
  });

  it("a null cost is not a cost of zero", () => {
    // "price unknown" and "$0.0000" are different facts and the runs table
    // must not conflate them.
    expect(data.runs[0]!.cost_usd).toBeNull();
    expect(fmtMetric(data.runs[0]!.cost_usd)).toBe("not measured");
  });
});

// ── the claim contract, as written on the page ─────────────────────────────

describe("the page's own copy obeys the claim contract", () => {
  const root = join(import.meta.dirname, "..", "..", "src");
  const page = readFileSync(join(root, "chrome", "BehaviorPage.tsx"), "utf8");
  const dataMod = readFileSync(join(root, "data", "behavior.ts"), "utf8");

  /** Comments are where both files EXPLAIN the rules, quoting the very patterns
   *  the rules forbid. Strip them, or the documentation trips its own test. */
  const code = (src: string) =>
    src.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");

  it("ranks no model anywhere", () => {
    for (const banned of ["leaderboard", "outperform", "the winner", "beats "]) {
      expect(page.toLowerCase()).not.toContain(banned);
    }
  });

  it("never implies access to a closed model's internals", () => {
    const low = page.toLowerCase();
    for (const banned of ["grok's internals", "inside grok", "grok's weights"]) {
      expect(low).not.toContain(banned);
    }
    // The page must say the opposite, out loud.
    expect(low).toContain("internals");
    expect(low).toContain("no model");
  });

  it("never coalesces a missing metric to zero", () => {
    // `?? 0` on a `number | null` is the exact edit that would turn every
    // unmeasured cue into a confident zero.
    expect(code(page)).not.toMatch(/\?\?\s*0\b/);
    expect(code(dataMod)).not.toMatch(/\?\?\s*0\b/);
  });

  it("says position is lexical, not behavioural", () => {
    expect(page).toContain("Position is lexical, not behavioural");
  });

  it("sources the landscape caption from the fit rather than asserting it", () => {
    expect(page).toContain("quantity_label");
  });
});
