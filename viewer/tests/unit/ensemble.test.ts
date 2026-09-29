/** The fan: the point-estimate rule, the survivorship rule, and what Δ̂ is
 *  allowed to say.
 *
 *  The fixture is a REAL ensemble document, not a hand-written one. It is the
 *  100 ctfish runs of claude-3-5-sonnet imported into a Seer store and grouped
 *  by Palisade's own prompt variant — 50 `baseline` against 50 `spooky` —
 *  rebuilt through `build_ensemble` exactly as the server rebuilds it. The
 *  condition is an experimenter-assigned prompt variant rather than an outcome
 *  label, which is what makes it a contrast at all: grouping runs by how they
 *  turned out and then measuring that they differ behaviourally would be
 *  circular.
 *
 *  Three properties of that document make it worth testing against instead of
 *  a fixture designed to pass:
 *
 *  1. `verified_after_last_edit` has n = 18 inside a 100-run ensemble, because
 *     the 82 runs that never edited cannot answer the question and are excluded
 *     rather than counted as failures. So a document that clears the point-
 *     estimate threshold overall still carries a rate that does not — exactly
 *     the case a per-ensemble check would get wrong.
 *  2. The fan's per-step `n` falls from 100 to 2. The band at the right-hand
 *     edge is two runs wide and looks like agreement.
 *  3. Nothing in it was verified: `verified` is 0/100, and 18 runs edited.
 */

import { describe, expect, it } from "vitest";
import REAL from "../fixtures/ensemble-ctfish-claude35-baseline-vs-spooky.json";
import {
  disjointSteps,
  envelopeWidths,
  intervalDisagreement,
  missingLines,
  parseEnsemble,
  pointEstimateAllowed,
  rateInterval,
  reliabilityNote,
  shortfall,
  survivorship,
  underpoweredNote,
  wilson,
} from "../../src/data/ensemble";

const E = parseEnsemble(REAL);
const rate = (key: string) => E.rates.find((r) => r.key === key)!;
const clone = () => JSON.parse(JSON.stringify(REAL));

describe("the real document", () => {
  it("parses the ctfish prompt-variant contrast as two conditions of fifty", () => {
    expect(E.ensembleId).toBe("ctfish-claude35-baseline-vs-spooky");
    expect(E.nRuns).toBe(100);
    expect(E.nRunsRequested).toBe(100);
    expect(shortfall(E)).toBe(0);
    expect(E.conditions.map((c) => [c.condition, c.nRuns])).toEqual([
      ["baseline", 50],
      ["spooky", 50],
    ]);
    expect(E.fidelity).toBe("deterministic");
  });

  it("names the envelope it drew rather than calling it a range", () => {
    expect(E.fanEnvelope).toBe("p10_p90");
    expect(E.fanMetric).toBe("cumulative_events");
    expect(E.fanFidelity).toBe("deterministic");
  });

  it("records that the seed was written down and never handed to an agent", () => {
    expect(E.seedBase).toBe(1234);
    expect(E.seedApplied).toBe(false);
  });

  it("reproduces every shipped interval from the counts", () => {
    expect(intervalDisagreement(E)).toEqual([]);
  });

  it("reports a doctored interval instead of preferring the prettier one", () => {
    const doc = clone();
    doc.rates.edited.ci95 = [0.4, 0.9];
    const lines = intervalDisagreement(parseEnsemble(doc));
    expect(lines.some((l) => l.startsWith("edited:"))).toBe(true);
  });

  it("reports a rate that disagrees with its own counts", () => {
    const doc = clone();
    doc.rates.edited.p = 0.5; // the counts still say 18/100
    const lines = intervalDisagreement(parseEnsemble(doc));
    expect(lines.some((l) => l.includes("its own counts say"))).toBe(true);
  });
});

describe("rule 1 — below the threshold there is no point estimate", () => {
  it("reads the threshold out of the document, never from a copy here", () => {
    expect(E.pointEstimateMinRuns).toBe(20);
    const doc = clone();
    doc.point_estimate_min_runs = 50;
    // the 100-run rates are still allowed; nothing in this module said "20"
    expect(parseEnsemble(doc).pointEstimateMinRuns).toBe(50);
  });

  it("refuses every point estimate when the document states no threshold", () => {
    // the strictest available reading, not a helpfully invented 20
    const doc = clone();
    delete doc.point_estimate_min_runs;
    const e = parseEnsemble(doc);
    expect(e.pointEstimateMinRuns).toBe(Number.POSITIVE_INFINITY);
    for (const r of e.rates) expect(pointEstimateAllowed(e, r), r.key).toBe(false);
  });

  it("allows a point for the 100-run rates", () => {
    for (const key of ["completed", "failed", "verified", "edited"]) {
      expect(pointEstimateAllowed(E, rate(key)), key).toBe(true);
    }
  });

  it("refuses one for the n = 18 rate inside the same 100-run ensemble", () => {
    const r = rate("verified_after_last_edit");
    expect(r.n).toBe(18);
    expect(r.nUndecidable).toBe(82);
    expect(pointEstimateAllowed(E, r)).toBe(false);
    const note = underpoweredNote(E, r);
    expect(note).toContain("0/18");
    expect(note).toContain("20");
  });

  it("refuses one for a rate that does not exist at all", () => {
    const r = rate("files_changed");
    expect(r.fidelity).toBe("missing");
    expect(r.p).toBeNull();
    expect(pointEstimateAllowed(E, r)).toBe(false);
    expect(rateInterval(r)).toBeNull();
  });
});

describe("missing is never zero", () => {
  it("keeps the unobserved file count out of the numbers entirely", () => {
    // 18 runs edited; the corpus rewrites the board with a shell command and
    // never names a file, so "how many files" was not measured
    const r = rate("files_changed");
    expect(r.n).toBe(0);
    expect(r.p).toBeNull();
    expect(r.missing).toMatch(/never observed and is not zero/);
    expect(missingLines(E).some((l) => l.startsWith("files_changed:"))).toBe(true);
  });

  it("keeps `edited` as a real measurement beside it", () => {
    const r = rate("edited");
    expect([r.k, r.n]).toEqual([18, 100]);
    expect(r.note).toMatch(/not a count of files touched/);
  });

  it("treats an unrecognised fidelity as missing, never as a pass", () => {
    const doc = clone();
    doc.rates.edited.fidelity = "vibes";
    expect(parseEnsemble(doc).rates.find((r: any) => r.key === "edited")!.fidelity).toBe(
      "missing",
    );
  });

  it("refuses a document with no ensemble_id rather than naming it itself", () => {
    expect(() => parseEnsemble({ rates: {} })).toThrow(/ensemble_id/);
  });
});

describe("rule 2 — the band narrows because runs left it", () => {
  it("finds the attrition in the real fan", () => {
    const s = survivorship(E);
    expect(s.nAtStart).toBe(100);
    expect(s.nAtEnd).toBe(2);
    expect(s.narrowsByAttrition).toBe(true);
    expect(s.firstDropStep).not.toBeNull();
    // and the drop is not at the very end — most of the fan is already thinning
    expect(s.firstDropStep!).toBeLessThan(E.fan.length - 1);
  });

  it("says nothing about attrition when every run went the distance", () => {
    const doc = clone();
    doc.fan = [
      { step: 0, median: 1, lo: 1, hi: 1, n: 8 },
      { step: 1, median: 2, lo: 1, hi: 3, n: 8 },
    ];
    const s = survivorship(parseEnsemble(doc));
    expect(s.narrowsByAttrition).toBe(false);
    expect(s.firstDropStep).toBeNull();
  });

  it("handles a document whose fan is missing entirely", () => {
    const doc = clone();
    doc.fan = [];
    doc.fan_fidelity = "missing";
    doc.missing = { fan: "an envelope needs at least 3 runs; this ensemble has 2" };
    const e = parseEnsemble(doc);
    expect(survivorship(e)).toMatchObject({ nAtStart: 0, narrowsByAttrition: false });
    expect(envelopeWidths(e)).toEqual([]);
    expect(missingLines(e)[0]).toMatch(/^fan: an envelope needs/);
  });

  it("sorts the fan by step, so a shuffled document still draws left to right", () => {
    const doc = clone();
    doc.fan = [...doc.fan].reverse();
    const steps = parseEnsemble(doc).fan.map((f) => f.step);
    expect(steps).toEqual([...steps].sort((a, b) => a - b));
  });

  it("measures the envelope width the reader is meant to read", () => {
    const w = envelopeWidths(E);
    expect(w).toHaveLength(E.fan.length);
    // the first steps are identical in every run: the band has no width there
    expect(w[0]!.width).toBe(0);
    expect(w.some((x) => x.width > 0)).toBe(true);
  });
});

describe("two fans are only different where their bands are disjoint", () => {
  it("finds no separating step between a document and itself", () => {
    expect(disjointSteps(E, E)).toEqual([]);
  });

  it("finds the steps where the bands genuinely do not overlap", () => {
    const doc = clone();
    doc.fan = doc.fan.map((f: any) => ({ ...f, lo: f.lo + 1000, hi: f.hi + 1000 }));
    expect(disjointSteps(E, parseEnsemble(doc))).toHaveLength(E.fan.length);
  });

  it("ignores steps one fan never reached", () => {
    const doc = clone();
    doc.fan = doc.fan.slice(0, 3).map((f: any) => ({ ...f, lo: f.lo + 1000, hi: f.hi + 1000 }));
    expect(disjointSteps(E, parseEnsemble(doc))).toEqual([0, 1, 2]);
  });
});

describe("rule 4 — Δ̂ with its terms, sign intact", () => {
  it("carries the real contrast and the features it was measured on", () => {
    const r = E.reliability;
    expect(r.conditionA).toBe("baseline");
    expect(r.conditionB).toBe("spooky");
    expect(r.nA).toBe(50);
    expect(r.nB).toBe(50);
    expect(r.deltaHat).not.toBeNull();
    expect(r.featuresUsed).toContain("n_turns");
    expect(r.featuresUsed).toContain("action_edit");
  });

  it("names every feature it dropped, instead of zero-filling it", () => {
    expect(Object.keys(E.reliability.featuresDropped)).toContain("n_files_changed");
    for (const why of Object.values(E.reliability.featuresDropped)) {
      expect(why.length).toBeGreaterThan(0);
    }
  });

  it("says the separation exceeds the within-condition spread, and scopes it", () => {
    const note = reliabilityNote(E.reliability);
    expect(note).toContain("Δ̂");
    expect(note).toMatch(/says nothing about either condition on its own/);
    expect(note).toContain("between =");
    expect(note).toContain("within =");
  });

  it("reports a negative Δ̂ as an answer, not as a failure", () => {
    const note = reliabilityNote({
      ...E.reliability,
      deltaHat: -0.08,
      between: 0.02,
      withinA: 0.09,
      withinB: 0.11,
    });
    expect(note).toMatch(/NEGATIVE/);
    expect(note).toMatch(/not clipped to zero/);
    expect(note).toMatch(/real answer/);
  });

  it("refuses to call two conditions distinguished when between ≤ within", () => {
    const note = reliabilityNote({
      ...E.reliability,
      deltaHat: 0.0,
      between: 0.05,
      withinA: 0.05,
      withinB: 0.05,
    });
    expect(note).toMatch(/have not been told apart/);
  });

  it("passes the backend's own reason through when Δ̂ was not computed", () => {
    const doc = clone();
    doc.reliability = {
      statistic: "delta_hat",
      delta_hat: null,
      fidelity: "missing",
      missing: "a split half needs at least 4 runs per condition; got 3 and 3",
    };
    const e = parseEnsemble(doc);
    expect(reliabilityNote(e.reliability)).toContain("4 runs per condition");
  });

  it("does not invent a reason when the document gives none", () => {
    const doc = clone();
    doc.reliability = { statistic: "delta_hat", delta_hat: null, fidelity: "missing" };
    const note = reliabilityNote(parseEnsemble(doc).reliability);
    expect(note).toMatch(/does not say why/);
    expect(note).toMatch(/undistinguished/);
  });
});

describe("the arithmetic is the repo's one Wilson", () => {
  it("agrees with data/absorbing's implementation", () => {
    const r = rate("edited");
    const mine = wilson(r.k, r.n)!;
    expect(mine[0]).toBeCloseTo(r.ci95![0] as number, 12);
    expect(mine[1]).toBeCloseTo(r.ci95![1] as number, 12);
  });

  it("falls back to a computed interval when the document shipped none", () => {
    const doc = clone();
    doc.rates.edited.ci95 = null;
    const r = parseEnsemble(doc).rates.find((x) => x.key === "edited")!;
    const iv = rateInterval(r)!;
    expect(iv[0]).toBeGreaterThan(0);
    expect(iv[1]).toBeLessThan(1);
  });
});
