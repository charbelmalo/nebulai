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
  coverageNote,
  cueSignificant,
  fmtMetric,
  isExampleOnly,
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

  /*  A real artifact in this repo has this shape: `capability-control-...`
   *  compared 2 cues across both arms, which is fewer than the 2 axes a
   *  landscape needs, so the exporter emitted `status: "missing"` with no
   *  coordinates at all. */
  const missing: BehaviorLandscape = {
    method: "pca",
    dims: 2,
    status: "missing",
    reason: "2 comparable cue(s): a 2-axis projection needs more than 2.",
    coords: [],
    pca_mean: [],
    pca_axes: [],
    pca_axes_shape: [0, 0],
    explained_variance_ratio: [],
    projection: {
      quantity: null,
      quantity_label: "no projection: too few comparable cues to fit one",
      total_variance: null,
      encoder: "test",
      encoder_revision: "",
      warning: "positions come from the cue words",
    },
  };

  it("refuses to project into a landscape that was never fitted", () => {
    // [0, 0] would be a coordinate the cue never earned, at the origin of a
    // plot that does not exist.
    expect(() => projectIntoLandscape(missing, [1, 2, 3])).toThrow(/no fitted landscape/);
  });

  it("carries the refusal's reason into the error", () => {
    expect(() => projectIntoLandscape(missing, [])).toThrow(/2 comparable cue/);
  });

  it("keeps the unmeasured projection quantity null rather than zero", () => {
    expect(missing.projection.quantity).toBeNull();
    expect(missing.projection.total_variance).toBeNull();
  });
});

// ── the two different empty plots ──────────────────────────────────────────

describe("a study with no landscape is not the same as a filter with no hits", () => {
  /*  Comments in the page EXPLAIN this distinction and quote both branches, so
   *  they are stripped before the source is searched. */
  const page = readFileSync(
    join(import.meta.dirname, "..", "..", "src", "chrome", "BehaviorPage.tsx"),
    "utf8",
  )
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/^\s*\/\/.*$/gm, "");

  it("branches on the exporter's status before it looks at the points", () => {
    // `coords: []` alone is ambiguous: it is both "nothing was fitted" and
    // "your filter matched nothing". The page has to read the status.
    const onStatus = page.indexOf('l.status === "missing"');
    const onPoints = page.indexOf("pts.length === 0");
    expect(onStatus).toBeGreaterThan(-1);
    expect(onPoints).toBeGreaterThan(-1);
    expect(onStatus).toBeLessThan(onPoints);
  });

  it("tells the reader no landscape was fitted, not that their filter is wrong", () => {
    expect(page).toContain("No landscape was fitted for this study");
  });

  it("shows the exporter's own reason rather than inventing one", () => {
    expect(page).toContain("l.reason");
  });

  it("points the reader at the views that do have every collected cue", () => {
    expect(page).toMatch(/ranked and table views/);
  });

  it("keeps the filter-specific message for the filter case", () => {
    expect(page).toContain("No cue in the current filter has a landscape position");
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

describe("an artifact that is not evidence says so", () => {
  const mk = (over: Record<string, unknown>) => over as unknown as BehaviorData;

  it("does not label a strict-source study", () => {
    expect(
      isExampleOnly(
        mk({
          diagnostics: { strict_source: true },
          published: { study_id: "s", source: "", at: "", published_as: "study" },
        }),
      ),
    ).toBe(false);
  });

  it("labels anything published as an example", () => {
    expect(
      isExampleOnly(
        mk({
          diagnostics: { strict_source: true },
          published: { study_id: "s", source: "", at: "", published_as: "example" },
        }),
      ),
    ).toBe(true);
  });

  it("labels a non-strict source even with no publish stamp", () => {
    // The older-artifact case. Absent metadata is not assumed innocent: a
    // `fake` arm or the hash encoder is enough on its own.
    expect(isExampleOnly(mk({ diagnostics: { strict_source: false } }))).toBe(true);
  });

  it("says nothing when there is no artifact at all", () => {
    // "not loaded yet" must not flash an accusation about a study nobody has
    // seen; the loading state owns that moment.
    expect(isExampleOnly(null)).toBe(false);
  });

  it("the page renders the banner, and its words are unambiguous", () => {
    const root = join(import.meta.dirname, "..", "..", "src");
    const src = readFileSync(join(root, "chrome", "BehaviorPage.tsx"), "utf8");
    expect(src).toContain("isExampleOnly(data)");
    expect(src).toContain("This is an example, not evidence.");
    // and the CSS actually distinguishes it from the ordinary warning banner
    const css = readFileSync(
      join(root, "styles", "nebulai.behavior.css"),
      "utf8",
    );
    expect(css).toContain(".behavior-banner.is-example");
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

// ── coverage: the denominator a cue list cannot carry ──────────────────────

describe("a study that did not cover its cue list says so", () => {
  const mk = (over: Record<string, unknown>) => over as unknown as BehaviorData;

  it("says nothing when the study covered everything", () => {
    expect(
      coverageNote(
        mk({
          cues: [cue()],
          coverage: { cues_planned: 1, cues_analyzed: 1, complete: true, reason: "" },
        }),
      ),
    ).toBeNull();
  });

  it("states both numbers, because the cue list only carries one of them", () => {
    // 40 cues on screen is indistinguishable from a 40-cue study unless the
    // denominator is printed next to it.
    const note = coverageNote(
      mk({
        cues: [cue()],
        coverage: {
          cues_planned: 100,
          cues_analyzed: 40,
          complete: false,
          reason: "run with --cue-limit 40",
          cue_limit: 40,
        },
      }),
    );
    expect(note).toContain("40 of 100");
    expect(note).toContain("--cue-limit 40");
  });

  it("an artifact with no coverage block is not assumed complete", () => {
    // The field was added after the first studies were exported. Treating its
    // absence as "complete" would silently relabel exactly the artifacts whose
    // coverage nobody can reconstruct.
    const note = coverageNote(mk({ cues: [cue(), cue({ cue: "b" })] }));
    expect(note).not.toBeNull();
    expect(note).toContain("does not record");
    expect(note).toContain("2 cues");
  });

  it("no study at all produces no banner", () => {
    expect(coverageNote(null)).toBeNull();
  });

  it("the banner does not let partial coverage discredit the cues that ran", () => {
    // A partial study's per-cue numbers ARE the complete study's numbers for
    // those cues — cue-level truncation keeps every repeat and every block. A
    // banner that implied otherwise would throw away real measurements.
    const src = readFileSync(
      join(process.cwd(), "src/chrome/BehaviorPage.tsx"),
      "utf8",
    );
    expect(src).toContain("not weakened by the ones that did not");
    expect(src).toContain("spans only the cues listed here");
  });
});
