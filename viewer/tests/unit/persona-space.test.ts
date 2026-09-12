/** The persona space's control, and the one rule it buys.
 *
 *  §3.3 of the Attractors plan: a space whose PC1 does not clear its
 *  permutation null renders with `verdict` visible and cannot be used as the
 *  default trajectory coordinate system. That sentence is the whole mechanism
 *  that keeps risk A — a "persona axis" that is really a prompt-length axis —
 *  survivable instead of silent, so it is tested here as a property of the
 *  parser and of `isSelectableAsDefault`, not as a thing a caller remembers.
 *
 *  The fixture is the real `space.json` this repo built for
 *  SmolLM2-135M-Instruct at layer 19, trimmed to the fields the viewer reads,
 *  including the cross-check that was measured first and FAILED and was kept.
 */

import { describe, expect, it } from "vitest";
import {
  archetypeScatter,
  isSelectableAsDefault,
  parseSpace,
  verdictLabel,
  verdictNote,
  type PersonaVerdict,
} from "../../src/data/persona";

/** Trimmed from out/persona/smollm2-135m-instruct@12fd25f77366.v1.L19. */
const REAL = {
  meta: {
    space_id: "smollm2-135m-instruct@12fd25f77366.v1.L19",
    model: "HuggingFaceTB/SmolLM2-135M-Instruct",
    revision: "12fd25f77366fa6b3b4b768ec3050bf629380bac",
    layer: 19,
    pooling: "last_token_mean_over_prompt_set",
    prompt_set: { id: "personas.v1", sha256: "35cb9d1dc380e0", n: 296 },
    created: "2026-09-11T20:45:19Z",
  },
  basis: { explained_variance_ratio: [0.204613, 0.161216, 0.118952] },
  archetypes: [
    { name: "therapist", scores: [-6.35956, -11.11169, 7.29846] },
    { name: "drill sergeant", scores: [8.1, 2.2, -1.0] },
  ],
  control: {
    method: "label_permutation_within_probe",
    n: 500,
    seed: 0,
    pc1_evr: 0.204613,
    pc1_evr_null_mean: 0.172918,
    pc1_evr_null_p95: 0.191639,
    verdict: "above_null",
    p_value: 0.001996,
    cross_check: {
      method: "label_permutation_unstratified",
      n: 500,
      pc1_evr_null_p95: 0.393993,
      verdict: "below_null",
      note: "Not the verdict. Ignoring the probe strata lets a pseudo-archetype carry an unbalanced probe mix.",
    },
  },
};

function withVerdict(v: string | undefined): ReturnType<typeof parseSpace> {
  const doc = JSON.parse(JSON.stringify(REAL));
  if (v === undefined) delete doc.control;
  else doc.control.verdict = v;
  return parseSpace(doc);
}

describe("parsing a space", () => {
  it("keeps the pinned revision and the frozen prompt set's sha", () => {
    const s = parseSpace(REAL);
    expect(s.revision).toBe("12fd25f77366fa6b3b4b768ec3050bf629380bac");
    expect(s.promptSet.id).toBe("personas.v1");
    expect(s.promptSet.sha256).toBeTruthy();
    expect(s.layer).toBe(19);
  });

  it("refuses a document with no space id rather than naming it itself", () => {
    expect(() => parseSpace({ basis: {} })).toThrow(/space_id/);
  });

  it("keeps the control that was measured first and failed", () => {
    const s = parseSpace(REAL);
    expect(s.control.crossCheck?.verdict).toBe("below_null");
    expect(s.control.crossCheck?.note).toBeTruthy();
    // and does not let it become the verdict
    expect(s.control.verdict).toBe("above_null");
  });
});

describe("the default-coordinate-system rule (§3.3)", () => {
  it("selects a space whose PC1 clears its null", () => {
    expect(isSelectableAsDefault(withVerdict("above_null"))).toBe(true);
  });

  it("refuses every other verdict, including a missing control", () => {
    for (const v of ["at_null", "below_null", "nonsense", undefined]) {
      expect(isSelectableAsDefault(withVerdict(v)), String(v)).toBe(false);
    }
  });

  it("calls an unmeasured control `unknown`, never a pass", () => {
    const s = withVerdict(undefined);
    expect(s.control.verdict).toBe("unknown");
    expect(verdictNote(s)).toMatch(/no permutation control/);
    // an unmeasured control and a failed one are different facts
    expect(verdictNote(s)).not.toBe(verdictNote(withVerdict("at_null")));
  });

  it("puts both numbers in the note, because the null ships with the figure", () => {
    const note = verdictNote(withVerdict("above_null"));
    expect(note).toContain("20.5%"); // pc1 evr
    expect(note).toContain("19.2%"); // the null's p95
  });

  it("says, for below_null, that the shuffled labels did better", () => {
    expect(verdictNote(withVerdict("below_null"))).toMatch(/shuffled labels do better/);
  });

  it("labels every verdict", () => {
    const all: PersonaVerdict[] = ["above_null", "at_null", "below_null", "unknown"];
    for (const v of all) expect(verdictLabel(v), v).toBeTruthy();
  });
});

describe("the archetype scatter", () => {
  it("preserves the aspect ratio instead of stretching to the box", () => {
    const pts = archetypeScatter(parseSpace(REAL));
    expect(pts).toHaveLength(2);
    const span = Math.max(...pts.flatMap((p) => [Math.abs(p.x), Math.abs(p.y)]));
    expect(span).toBeCloseTo(1, 6); // the largest extent touches the edge…
    // …and the other axis is NOT also 1: therapist's PC2 is the big one here,
    // so its PC1 has to stay proportionally smaller
    const therapist = pts.find((p) => p.name === "therapist")!;
    expect(Math.abs(therapist.x)).toBeLessThan(Math.abs(therapist.y));
  });

  it("drops an archetype with too few components rather than padding it", () => {
    const doc = JSON.parse(JSON.stringify(REAL));
    doc.archetypes.push({ name: "truncated", scores: [1] });
    const pts = archetypeScatter(parseSpace(doc));
    expect(pts.map((p) => p.name)).not.toContain("truncated");
  });
});

/** The same recipe at a second model size, trimmed from
 *  out/persona/smollm2-360m-instruct@a10cc1512eab.v1.L20/space.json.
 *
 *  It is here because one model is not a result. SmolLM2-360M-Instruct is 2.7x
 *  the parameters and a wider residual stream (d = 960 against 576), its default
 *  layer lands at 20 rather than 19, and the thing worth recording is that the
 *  effect is in the same place and SMALLER: PC1 explains 16.6% of the archetype
 *  means' variance against 20.5% at 135M, and its null p95 is correspondingly
 *  lower (0.1537 against 0.1916), so the margin over the null shrinks from
 *  0.013 to 0.012 and the p-value goes from 0.002 to 0.008.
 *
 *  Both clear their null. Neither clears it by much, and writing the second set
 *  of numbers into the repository is the only thing that stops "the persona
 *  space clears its control" from quietly becoming a property of the recipe
 *  rather than of one measurement on one checkpoint.
 */
const REAL_360M = {
  meta: {
    space_id: "smollm2-360m-instruct@a10cc1512eab.v1.L20",
    model: "HuggingFaceTB/SmolLM2-360M-Instruct",
    revision: "a10cc1512eabd3dde888204e902eca88bddb4951",
    layer: 20,
    pooling: "last_token_mean_over_prompt_set",
    prompt_set: { id: "personas.v1", sha256: "35cb9d1dc380e0", n: 296 },
    created: "2026-09-12T01:13:00Z",
  },
  basis: { explained_variance_ratio: [0.166188, 0.117218, 0.102657] },
  archetypes: [{ name: "therapist", scores: [-14.84096, 5.97723, -1.41753] }],
  control: {
    method: "label_permutation_within_probe",
    n: 500,
    seed: 0,
    pc1_evr: 0.166188,
    pc1_evr_null_mean: 0.137741,
    pc1_evr_null_p95: 0.153749,
    verdict: "above_null",
    p_value: 0.007984,
    cross_check: {
      method: "label_permutation_unstratified",
      n: 500,
      pc1_evr_null_p95: 0.347591,
      verdict: "below_null",
      note: "Not the verdict. Ignoring the probe strata lets a pseudo-archetype carry an unbalanced probe mix.",
    },
  },
};

describe("the same control at a second model size", () => {
  const s = parseSpace(REAL_360M);

  it("is a DIFFERENT space, by id, and says which model it belongs to", () => {
    // the space id carries model, prompt-set version and layer, so a 360M space
    // can never be mistaken for the 135M one or silently used in its place
    expect(s.spaceId).toBe("smollm2-360m-instruct@a10cc1512eab.v1.L20");
    expect(s.spaceId).not.toBe(parseSpace(REAL).spaceId);
    expect(s.model).toBe("HuggingFaceTB/SmolLM2-360M-Instruct");
    expect(s.revision).toBe("a10cc1512eabd3dde888204e902eca88bddb4951");
    expect(s.layer).toBe(20);
  });

  it("shares the frozen prompt set with the 135M space", () => {
    // the same 296 archetypes at the same sha: the two spaces differ in the
    // model, not in what was asked of it, which is what makes them comparable
    expect(s.promptSet.id).toBe(parseSpace(REAL).promptSet.id);
    expect(s.promptSet.sha256).toBe(parseSpace(REAL).promptSet.sha256);
    expect(s.promptSet.n).toBe(296);
  });

  it("clears its own null, by a margin as thin as the 135M one", () => {
    expect(s.control.verdict).toBe("above_null");
    expect(isSelectableAsDefault(s)).toBe(true);
    expect(s.control.pc1Evr!).toBeGreaterThan(s.control.pc1EvrNullP95!);
    // thin: about a point and a quarter of explained variance, at both sizes
    const margin = s.control.pc1Evr! - s.control.pc1EvrNullP95!;
    const margin135 = parseSpace(REAL).control.pc1Evr! - parseSpace(REAL).control.pc1EvrNullP95!;
    expect(margin).toBeLessThan(0.02);
    expect(margin135).toBeLessThan(0.02);
  });

  it("has a LOWER pc1_evr than the smaller model, and that is not a failure", () => {
    // a bigger model spreads the archetype means over more directions; the
    // fraction on PC1 falling is not the effect weakening, and the only thing
    // that decides whether the space is usable is its own null
    expect(s.control.pc1Evr!).toBeLessThan(parseSpace(REAL).control.pc1Evr!);
    expect(s.control.pc1EvrNullP95!).toBeLessThan(parseSpace(REAL).control.pc1EvrNullP95!);
    expect(isSelectableAsDefault(s)).toBe(true);
  });

  it("keeps the unstratified cross-check that failed here too", () => {
    expect(s.control.crossCheck?.verdict).toBe("below_null");
    expect(s.control.crossCheck?.note).toBeTruthy();
    expect(s.control.verdict).toBe("above_null");
  });
});
