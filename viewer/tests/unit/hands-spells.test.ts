/** The cast recognizer — the layer where a false positive actually costs
 *  something. Every number in `spells.ts` is a decision rather than a tuning
 *  constant, and this file is where those decisions are written down in a form
 *  that fails when someone loosens one.
 *
 *  `updateSpells` is pure over its state object: features in, events out, no
 *  clock and no camera. That is what makes a whole gesture expressible as a
 *  handful of synthetic frames.
 *
 *  The signature changed in the rebuild and the change is the subject of the
 *  first describe block below: this module is handed the **free** hand — the one
 *  not steering the camera — or `null`. It no longer sees the frame's whole hand
 *  list, which is what makes it structurally incapable of firing off the hand
 *  that is flying the map. */

import { describe, expect, it } from "vitest";
import { createSpellState, resetSpellState, updateSpells } from "../../src/hands/spells";
import type { HandFeatures, SpellEvent, SpellId } from "../../src/hands/types";

const DT = 1 / 30;

function makeHand(overrides: Partial<HandFeatures> = {}): HandFeatures {
  return {
    hand: "Right",
    gesture: "None",
    gestureScore: 0,
    palmNormal: { x: 0, y: 0, z: 1 },
    pointDirection: { x: 0, y: -1, z: 0 },
    palmCentre: { x: 0.5, y: 0.5, z: 0 },
    indexTip: { x: 0.5, y: 0.4, z: 0 },
    pinchGap: 0.6,
    pinching: false,
    span: 0.09,
    screenSpan: 0.18,
    curls: { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0 },
    openness: 1,
    palmVelocity: { x: 0, y: 0, z: 0 },
    approachRate: 0,
    pinchRate: 0,
    speed: 0,
    role: "idle",
    ...overrides,
  };
}

/** A palm held still: what charges, and the resting state of most tests. */
function stillPalm(overrides: Partial<HandFeatures> = {}): HandFeatures {
  return makeHand({ role: "charge", speed: 0, ...overrides });
}

/** Runs `frames` identical frames and returns everything that fired. */
function run(
  state: ReturnType<typeof createSpellState>,
  hand: HandFeatures | null,
  frames: number,
  dt = DT,
  startAt = 0,
): SpellEvent[] {
  const events: SpellEvent[] = [];
  let now = startAt;
  for (let i = 0; i < frames; i++) {
    now += dt;
    events.push(...updateSpells(state, hand, dt, now));
  }
  return events;
}

const ids = (events: readonly SpellEvent[]): SpellId[] => events.map((event) => event.id);

describe("the free hand only", () => {
  it("fires nothing at all when there is no free hand", () => {
    // The one-hand case, and the effects-off case, are the same call. This is
    // the whole uncoupling: the caller resolves which hand is steering and hands
    // this module whatever is left, so a single raised hand can fly the camera
    // through every threshold below without arming anything.
    const state = createSpellState();
    expect(run(state, null, 200, 0.05)).toHaveLength(0);
  });

  it("does not accumulate charge with no free hand, however long it is held", () => {
    // Charging invisibly would be worse than firing: the first cast after a
    // second hand appears would go off instantly, from motion made before there
    // was anything to make it with.
    const state = createSpellState();
    run(state, null, 100, 0.05);
    expect(state.charge).toBe(0);
    expect(state.chargingHand).toBeNull();
  });

  it("cannot fire a shockwave from a hand that is merely present", () => {
    // A fully charged, hard-thrusting palm — which would fire immediately as a
    // free hand — does nothing once the caller has claimed it for steering and
    // stopped passing it in.
    const state = createSpellState();
    updateSpells(state, stillPalm(), 1.15, 1.15);
    expect(state.charge).toBeCloseTo(1, 10);
    expect(run(state, null, 6, DT, 1.15)).toHaveLength(0);
  });
});

describe("charge", () => {
  it("fills over CHARGE_SECONDS while an open palm is held still", () => {
    const state = createSpellState();
    updateSpells(state, stillPalm(), 1.15, 1.15);
    expect(state.charge).toBeCloseTo(1, 10);
    expect(state.chargingHand).toBe("Right");
  });

  it("charges at a rate, not on arrival — half the time is half the charge", () => {
    const state = createSpellState();
    updateSpells(state, stillPalm(), 0.5, 0.5);
    expect(state.charge).toBeCloseTo(0.5 / 1.15, 10);
  });

  it("tolerates the tremor of an unsupported arm", () => {
    // A resting arm drifts around 0.05 viewport widths per second. STILL_SPEED
    // is 0.16 precisely so charging does not demand a steadiness nobody has.
    const state = createSpellState();
    updateSpells(state, stillPalm({ speed: 0.05 }), 0.5, 0.5);
    expect(state.charge).toBeGreaterThan(0);
  });

  it("bleeds while the palm moves rather than dumping the charge", () => {
    // Bleeding is what keeps a small correction from throwing away a second of
    // held stillness — zeroing here would make charging feel arbitrary.
    const state = createSpellState();
    updateSpells(state, stillPalm(), 1.15, 1.15);
    updateSpells(state, stillPalm({ speed: 0.6 }), 0.25, 1.4);
    expect(state.charge).toBeCloseTo(1 - 0.25 * 1.6, 10);
  });

  it("decays more gently when no palm is present at all", () => {
    const state = createSpellState();
    updateSpells(state, stillPalm(), 1.15, 1.15);
    updateSpells(state, null, 0.25, 1.4);
    // 0.8/s, half the bleed rate: a hand briefly out of frame is not the same
    // event as a hand that moved.
    expect(state.charge).toBeCloseTo(1 - 0.25 * 0.8, 10);
    expect(state.chargingHand).toBeNull();
  });

  it("does not charge from a hand the channel layer did not read as a palm", () => {
    // A pinching or pointing free hand is doing something else. Only the open
    // palm — `light` or its still sub-state `charge` — gathers.
    const state = createSpellState();
    updateSpells(state, makeHand({ role: "point", speed: 0 }), 1.15, 1.15);
    expect(state.charge).toBe(0);
  });
});

describe("shockwave", () => {
  it("fires on a thrust and spends the charge that powered it", () => {
    const state = createSpellState();
    updateSpells(state, stillPalm(), 0.5, 0.5);

    // A thrust is motion along z; `speed` is the xy magnitude, so a straight
    // push toward the camera is still "held still" as far as charging goes.
    const events = updateSpells(state, stillPalm({ approachRate: 0.33 }), DT, 0.5 + DT);

    expect(ids(events)).toEqual(["shockwave"]);
    // Power is mostly charge (0.55), on a floor of 0.3, plus vigour — which is
    // zero here because the push sits exactly on the threshold.
    const charge = (0.5 + DT) / 1.15;
    expect(events[0]?.power).toBeCloseTo(0.3 + 0.55 * charge, 10);
    expect(state.charge).toBe(0);
  });

  it("ignores a push that does not reach the thrust rate", () => {
    const state = createSpellState();
    updateSpells(state, stillPalm(), 1.15, 1.15);
    // 0.32 against a 0.33 gate. The gate sits in the gap between reach-and-
    // settle and a deliberate push, not at the noise floor.
    expect(run(state, stillPalm({ approachRate: 0.32 }), 4, DT, 1.15)).toHaveLength(0);
  });

  it("refuses an uncharged thrust however hard it is thrown", () => {
    const state = createSpellState();
    // One frame of charge is 0.029 — under the 0.12 minimum.
    const events = updateSpells(state, stillPalm({ approachRate: 2 }), DT, DT);
    expect(events).toHaveLength(0);
  });

  it("cannot double-fire from one physical push", () => {
    const state = createSpellState();
    updateSpells(state, stillPalm(), 1.15, 1.15);
    // A push spans several frames above threshold. Without the refractory
    // period each of them would be a separate shockwave.
    const thrust = stillPalm({ approachRate: 0.9 });
    expect(ids(run(state, thrust, 6, DT, 1.15))).toEqual(["shockwave"]);
  });
});

describe("snap", () => {
  it("fires when the pinch gap opens fast enough", () => {
    const state = createSpellState();
    const events = updateSpells(state, makeHand({ pinchRate: 3 }), DT, DT);
    expect(ids(events)).toEqual(["snap"]);
    const vigour = (3 - 2.6) / (2.6 * 1.5);
    expect(events[0]?.power).toBeCloseTo(0.45 + 0.55 * vigour, 10);
  });

  it("ignores a relaxed opening", () => {
    const state = createSpellState();
    // A hand simply unclenching runs well under 1; 2.5 is already a brisk
    // release and still under the gate.
    expect(run(state, makeHand({ pinchRate: 2.5 }), 5)).toHaveLength(0);
  });

  it("holds its refractory period and then re-arms", () => {
    const state = createSpellState();
    const snapping = makeHand({ pinchRate: 3 });

    expect(ids(run(state, snapping, 1))).toEqual(["snap"]);
    // 0.3 s of continuous over-threshold frames, inside the 0.35 s period.
    expect(run(state, snapping, 9)).toHaveLength(0);
    // Past it, a genuine second snap is allowed through.
    expect(ids(run(state, snapping, 3))).toEqual(["snap"]);
  });
});

describe("one gesture, one cast", () => {
  it("a thrust that also opens the pinch fires the shockwave alone", () => {
    // Releasing a push naturally opens the fingers, so a single motion can
    // satisfy both gates. The thrust returns early for that reason; without it
    // one shove would announce two casts and paint two effects.
    const state = createSpellState();
    updateSpells(state, stillPalm(), 1.15, 1.15);
    const events = updateSpells(
      state,
      stillPalm({ approachRate: 0.9, pinchRate: 4 }),
      DT,
      1.15 + DT,
    );
    expect(ids(events)).toEqual(["shockwave"]);
  });
});

describe("reset", () => {
  it("clears charge and cooldowns", () => {
    const state = createSpellState();
    updateSpells(state, stillPalm(), 1.15, 1.15);
    resetSpellState(state);
    expect(state.charge).toBe(0);
    expect(state.chargingHand).toBeNull();
    expect(state.cooldown.shockwave).toBe(0);
    expect(state.cooldown.snap).toBe(0);
  });
});
