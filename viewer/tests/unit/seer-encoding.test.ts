/** The shared encoding vocabulary — the properties that keep five views from
 *  disagreeing with each other.
 *
 *  Three things are pinned here:
 *
 *  * every member of every contract enum the render layer touches has an
 *    encoding, so a new `Action` or `Effect` added in Python cannot reach the
 *    screen as an unstyled blank. This is the same drift guard
 *    `seer-contract-sync.test.ts` runs against the Python source, one layer up:
 *    that one proves we *know* the member, this one proves we can *draw* it;
 *  * the two absent fidelities are drawn as holes and in their own ink, never
 *    as a short bar in the colour of the work they failed to measure;
 *  * `rankNormalise` spends its output range on the ordering and never returns
 *    0, because 0 means "absent" everywhere else in this subsystem.
 */

import { describe, expect, it } from "vitest";
import {
  ACTIONS,
  EFFECTS,
  FIDELITIES,
  SESSION_STATES,
  type Fidelity,
} from "../../src/seer/contract";
import {
  ABSENT_INK,
  ACTION_COLOR,
  EFFECT_CAP,
  FIDELITY_TEXTURE,
  NEUTRAL_INK,
  RANK_FLOOR,
  STATE_COLOR,
  isProvisional,
  markInk,
  rankNormalise,
} from "../../src/seer/encoding";

// ── coverage ─────────────────────────────────────────────────────────────────

describe("every contract member can be drawn", () => {
  it("gives every action a hue", () => {
    for (const a of ACTIONS) {
      expect(ACTION_COLOR, a).toHaveProperty(a);
      expect(ACTION_COLOR[a], a).toMatch(/^#[0-9a-f]{6}$/i);
    }
  });

  it("gives every session state a hue", () => {
    for (const s of SESSION_STATES) {
      expect(STATE_COLOR, s).toHaveProperty(s);
      expect(STATE_COLOR[s], s).toMatch(/^#[0-9a-f]{6}$/i);
    }
  });

  it("gives every effect an end cap", () => {
    for (const e of EFFECTS) expect(EFFECT_CAP, e).toHaveProperty(e);
  });

  it("gives every fidelity a texture", () => {
    for (const f of FIDELITIES) expect(FIDELITY_TEXTURE, f).toHaveProperty(f);
  });

  it("keeps the action hues distinct, so a lane legend is readable", () => {
    const hues = ACTIONS.map((a) => ACTION_COLOR[a].toLowerCase());
    expect(new Set(hues).size).toBe(ACTIONS.length);
  });
});

// ── the absence rules ────────────────────────────────────────────────────────

describe("absent values are drawn as holes", () => {
  it("gives the two absent fidelities unfilled textures", () => {
    expect(FIDELITY_TEXTURE.missing).toBe("outline");
    expect(FIDELITY_TEXTURE.dropped_by_policy).toBe("policy");
  });

  it("never draws an absent value in the colour of the work it is missing from", () => {
    // `edit` work whose duration is missing must not read as a small edit.
    expect(markInk("edit", "missing")).toBe(ABSENT_INK.missing);
    expect(markInk("edit", "dropped_by_policy")).toBe(ABSENT_INK.dropped_by_policy);
    expect(markInk("edit", "native")).toBe(ACTION_COLOR.edit);
  });

  it("distinguishes the two absent kinds from each other", () => {
    // One is a capability gap in the agent, the other a setting on our side.
    // A researcher's next move differs, so the pixels have to differ too.
    expect(ABSENT_INK.missing).not.toBe(ABSENT_INK.dropped_by_policy);
    expect(FIDELITY_TEXTURE.missing).not.toBe(FIDELITY_TEXTURE.dropped_by_policy);
  });

  it("falls back to neutral ink for an unclassified span", () => {
    expect(markInk(null, "deterministic")).toBe(NEUTRAL_INK);
    expect(markInk(undefined, "native")).toBe(NEUTRAL_INK);
  });
});

describe("isProvisional", () => {
  it("is true while a span is still open", () => {
    expect(isProvisional({ endedAt: null, fidelity: "native" })).toBe(true);
    expect(isProvisional({})).toBe(true);
  });

  it("is true for a closed span whose duration came back absent", () => {
    // Every reconciled run is like this: thread history carries no per-item
    // clock, so the span ended but nobody clocked it. Its `0` is a construction
    // artefact, not a measurement.
    expect(isProvisional({ endedAt: 12, fidelity: "missing" })).toBe(true);
    expect(isProvisional({ endedAt: 12, fidelity: "dropped_by_policy" })).toBe(true);
  });

  it("is false only for a closed span that was actually measured", () => {
    const measured: Fidelity[] = ["native", "deterministic", "estimated", "heuristic"];
    for (const f of measured) {
      expect(isProvisional({ endedAt: 12, fidelity: f }), f).toBe(false);
    }
  });
});

// ── rank normalisation ───────────────────────────────────────────────────────

describe("rankNormalise", () => {
  it("returns positions in the input's own order", () => {
    expect(rankNormalise([30, 10, 20])).toEqual([1, RANK_FLOOR, (RANK_FLOOR + 1) / 2]);
  });

  it("never returns 0, because 0 means absent elsewhere", () => {
    for (const v of rankNormalise([0, 5, 5000])) expect(v).toBeGreaterThan(0);
    expect(Math.min(...rankNormalise([1, 2, 3]))).toBe(RANK_FLOOR);
  });

  it("spends the range on the ordering, not the magnitudes", () => {
    // The point of ranking: one enormous outlier must not flatten everything
    // else into an invisible hairline that reads as "those did nothing".
    const withOutlier = rankNormalise([1, 2, 3, 1_000_000]);
    const without = rankNormalise([1, 2, 3, 4]);
    expect(withOutlier).toEqual(without);
  });

  it("draws equal values identically", () => {
    const out = rankNormalise([7, 7, 7]);
    expect(out[0]).toBe(out[1]);
    expect(out[1]).toBe(out[2]);
  });

  it("gives tied values the mean of the ranks they span", () => {
    // [1, 1, 3]: the two 1s share ranks 0 and 1, so both sit at rank 0.5 of 2.
    const out = rankNormalise([1, 1, 3]);
    expect(out[0]).toBeCloseTo(RANK_FLOOR + 0.25 * (1 - RANK_FLOOR), 10);
    expect(out[0]).toBe(out[1]);
    expect(out[2]).toBe(1);
  });

  it("handles the degenerate inputs a live view will hand it", () => {
    expect(rankNormalise([])).toEqual([]);
    expect(rankNormalise([42])).toEqual([1]);
  });

  it("floors non-finite values rather than letting them order anything", () => {
    const out = rankNormalise([NaN, 1, 2]);
    expect(out[0]).toBe(RANK_FLOOR);
    expect(out[2]).toBe(1);
    expect(rankNormalise([NaN, NaN])).toEqual([RANK_FLOOR, RANK_FLOOR]);
  });
});
