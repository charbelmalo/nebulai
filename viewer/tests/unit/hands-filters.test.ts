/** The hand rig's signal conditioning. Vitest runs in plain Node here — no
 *  DOM, no camera — which is exactly why `filters.ts` is plain state-plus-step
 *  functions: every claim its comments make is checkable on a synthetic sample
 *  sequence. */

import { describe, expect, it } from "vitest";
import {
  createOneEuro,
  createOneEuroVec3,
  createSchmitt,
  createSpring,
  oneEuroLead,
  oneEuroStep,
  oneEuroVec3Step,
  schmittStep,
  springAtRest,
  springStep,
  type Vec3,
} from "../../src/hands/filters";

const DT = 1 / 30;

/** Deterministic ±amplitude jitter — a fixed pattern, not a random one, so a
 *  failure is reproducible. */
function jitter(i: number, amplitude: number): number {
  return Math.sin(i * 2.399963) * amplitude;
}

describe("one euro filter", () => {
  it("passes the first sample through untouched", () => {
    const state = createOneEuro();
    expect(oneEuroStep(state, 5, DT)).toBe(5);
    expect(state.velocity).toBe(0);
  });

  it("holds still and reports zero velocity on a constant signal", () => {
    const state = createOneEuro();
    for (let i = 0; i < 40; i++) oneEuroStep(state, 5, DT);
    expect(state.value).toBeCloseTo(5, 10);
    expect(state.velocity).toBeCloseTo(0, 10);
  });

  it("converges toward a step without overshooting it", () => {
    const state = createOneEuro();
    oneEuroStep(state, 0, DT);
    let previous = 0;
    for (let i = 0; i < 60; i++) {
      const value = oneEuroStep(state, 1, DT);
      expect(value).toBeGreaterThanOrEqual(previous);
      expect(value).toBeLessThanOrEqual(1);
      previous = value;
    }
    expect(previous).toBeGreaterThan(0.9);
  });

  it("attenuates jitter on a still hand", () => {
    const state = createOneEuro();
    let worst = 0;
    for (let i = 0; i < 90; i++) {
      const value = oneEuroStep(state, 0.5 + jitter(i, 0.01), DT);
      if (i > 20) worst = Math.max(worst, Math.abs(value - 0.5));
    }
    // The input swings ±0.01; a quiet cutoff has to cut that by most of itself.
    expect(worst).toBeLessThan(0.004);
  });

  it("gets out of the way when the hand moves — the whole point of 1€", () => {
    // Same ramp through two filters differing only in beta. Beta is the term
    // that opens the cutoff with speed, so the high-beta filter must sit closer
    // to the truth. A fixed low-pass would show no difference at all.
    const lazy = createOneEuro({ minCutoff: 1, beta: 0 });
    const eager = createOneEuro({ minCutoff: 1, beta: 4 });
    let lazyValue = 0;
    let eagerValue = 0;
    for (let i = 0; i < 30; i++) {
      const sample = i * 0.05;
      lazyValue = oneEuroStep(lazy, sample, DT);
      eagerValue = oneEuroStep(eager, sample, DT);
    }
    const truth = 29 * 0.05;
    expect(truth - eagerValue).toBeLessThan(truth - lazyValue);
  });

  it("leads by exactly velocity × lead time", () => {
    const state = createOneEuro();
    for (let i = 0; i < 20; i++) oneEuroStep(state, i * 0.05, DT);
    expect(oneEuroLead(state, 0.045)).toBeCloseTo(state.value + state.velocity * 0.045, 12);
  });

  it("survives a multi-second delta rather than exploding through it", () => {
    // A backgrounded tab hands the rig a delta of seconds. The step clamp is
    // what stops that from being integrated in one go.
    const state = createOneEuro();
    oneEuroStep(state, 0, DT);
    const value = oneEuroStep(state, 1, 12);
    expect(Number.isFinite(value)).toBe(true);
    expect(value).toBeGreaterThan(0);
    expect(value).toBeLessThanOrEqual(1);
  });

  it("ignores a non-finite sample instead of poisoning its state", () => {
    const state = createOneEuro();
    oneEuroStep(state, 4, DT);
    expect(oneEuroStep(state, Number.NaN, DT)).toBe(4);
    expect(state.value).toBe(4);
  });
});

describe("one euro vec3", () => {
  it("shares one cutoff across the axes so a diagonal keeps its direction", () => {
    // The failure this pins: filtering each axis on its own speed smooths x and
    // y by different amounts, so a symmetric diagonal sweep comes out bent.
    const state = createOneEuroVec3();
    const out: Vec3 = { x: 0, y: 0, z: 0 };
    for (let i = 0; i < 40; i++) {
      const t = i * 0.02;
      oneEuroVec3Step(state, { x: t, y: t, z: 0 }, DT, out);
      expect(out.x).toBeCloseTo(out.y, 12);
    }
  });
});

describe("spring", () => {
  it("arrives at its target", () => {
    const spring = createSpring(0);
    for (let i = 0; i < 300; i++) springStep(spring, 1, DT, { frequency: 2.4, damping: 0.72 });
    expect(spring.value).toBeCloseTo(1, 3);
    expect(springAtRest(spring, 1, 1e-3)).toBe(true);
  });

  it("overshoots when underdamped and does not when critically damped", () => {
    const loose = createSpring(0);
    const tight = createSpring(0);
    let loosePeak = 0;
    let tightPeak = 0;
    for (let i = 0; i < 200; i++) {
      loosePeak = Math.max(loosePeak, springStep(loose, 1, DT, { frequency: 2, damping: 0.4 }));
      tightPeak = Math.max(tightPeak, springStep(tight, 1, DT, { frequency: 2, damping: 1 }));
    }
    // The small overshoot is what reads as weight; it is a feature, not slop.
    expect(loosePeak).toBeGreaterThan(1.05);
    expect(tightPeak).toBeLessThanOrEqual(1.0001);
  });

  it("clamps a pathological timestep instead of throwing itself across the range", () => {
    const spring = createSpring(0);
    springStep(spring, 1, 30, { frequency: 3, damping: 0.7 });
    expect(Number.isFinite(spring.value)).toBe(true);
    expect(Math.abs(spring.value)).toBeLessThan(2);
  });

  it("keeps its value when handed a non-finite target", () => {
    const spring = createSpring(0.5);
    expect(springStep(spring, Number.NaN, DT, { frequency: 2, damping: 0.7 })).toBe(0.5);
  });
});

describe("schmitt trigger", () => {
  const ENGAGE = 0.34;
  const RELEASE = 0.5;
  const DWELL = 0.05;

  it("does not engage on a value that only passes through the threshold", () => {
    const state = createSchmitt();
    // One frame under the engage threshold, at 30 Hz, is 0.033 s — short of the
    // 0.05 s dwell, so a hand on its way somewhere else does not register.
    expect(schmittStep(state, 0.2, DT, ENGAGE, RELEASE, DWELL)).toBe(false);
    expect(schmittStep(state, 0.8, DT, ENGAGE, RELEASE, DWELL)).toBe(false);
  });

  it("engages once the dwell is satisfied", () => {
    const state = createSchmitt();
    expect(schmittStep(state, 0.2, DT, ENGAGE, RELEASE, DWELL)).toBe(false);
    expect(schmittStep(state, 0.2, DT, ENGAGE, RELEASE, DWELL)).toBe(true);
  });

  it("holds through the hysteresis band", () => {
    const state = createSchmitt();
    schmittStep(state, 0.2, DT, ENGAGE, RELEASE, DWELL);
    schmittStep(state, 0.2, DT, ENGAGE, RELEASE, DWELL);
    // 0.42 would never have engaged it, and does not release it either.
    expect(schmittStep(state, 0.42, DT, ENGAGE, RELEASE, DWELL)).toBe(true);
  });

  it("releases immediately, without waiting out the dwell", () => {
    const state = createSchmitt();
    schmittStep(state, 0.2, DT, ENGAGE, RELEASE, DWELL);
    schmittStep(state, 0.2, DT, ENGAGE, RELEASE, DWELL);
    // Holding a grab the operator has already let go of is worse than dropping
    // one a frame early, so release is deliberately not dwell-gated.
    expect(schmittStep(state, 0.9, DT, ENGAGE, RELEASE, DWELL)).toBe(false);
  });
});
