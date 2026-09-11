/** The one navigation law, and the hand assignment that keeps it uncoupled.
 *
 *  This file replaces `hands-nav-modes.test.ts`, which pinned three selectable
 *  control laws. The modes are gone; what stayed is the half of that file worth
 *  keeping — the *directions* and the *kind* of control, both of which are
 *  invisible to every other kind of test. A sign error produces a camera that
 *  moves smoothly and confidently the wrong way, which no snapshot, typecheck or
 *  smoke test can distinguish from correct. So each direction is asserted
 *  against the convention it has to satisfy, with the convention written down
 *  next to it rather than left in a third file.
 *
 *  What is new is the first block. The rig's central defect was that navigating
 *  and casting were read off the same physical motion, and the fix is that the
 *  free hand and the steering hand are disjoint by construction. That is a claim
 *  about this module's structure, so it is pinned here rather than left to the
 *  spell recognizer, which can no longer even see the difference.
 *
 *  `updateHandChannelTargets` is pure over its state object, so a whole gesture
 *  is a handful of synthetic frames — no clock, no camera, no DOM. */

import { describe, expect, it } from "vitest";
import {
  createHandChannels,
  handChannelsSettled,
  resetHandChannels,
  setHandEffectsEnabled,
  stepHandChannels,
  updateHandChannelTargets,
  type HandChannelState,
} from "../../src/hands/channels";
import type { HandFeatures, HandReading } from "../../src/hands/types";

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

/** An open palm at a given point in frame. `speed: 0` keeps it steering. */
function palmAt(x: number, y: number, screenSpan = 0.18): HandFeatures {
  return makeHand({ palmCentre: { x, y, z: 0 }, screenSpan });
}

function pointingAt(x: number, y: number): HandFeatures {
  return makeHand({
    gesture: "Pointing_Up",
    indexTip: { x, y, z: 0 },
    curls: { thumb: 0, index: 0.1, middle: 0.8, ring: 0.8, pinky: 0.8 },
  });
}

function reading(hands: HandFeatures[], timestamp = 0): HandReading {
  return { timestamp, hands, dominant: hands[0] ?? null, secondary: hands[1] ?? null };
}

function state(effects = false): HandChannelState {
  return createHandChannels(effects);
}

/** Runs `frames` steps with the same hands, as a held pose does. */
function hold(s: HandChannelState, hands: HandFeatures[], frames: number, from = 0): void {
  for (let frame = 0; frame < frames; frame += 1) {
    updateHandChannelTargets(s, reading(hands, (from + frame) * DT * 1000), DT);
  }
}

// ---------------------------------------------------------------------------

describe("steering and casting cannot be the same hand", () => {
  it("cannot cast at all from a single raised hand", () => {
    // The common case, and the whole point. This palm is held still (charging
    // posture), thrusting hard, and snapping its fingers — every gate the cast
    // recognizer has — while being the only hand in frame. It flies the camera
    // and fires nothing, because there is no hand left over to fire with.
    const s = state(true);
    const busy = makeHand({
      palmCentre: { x: 0.9, y: 0.5, z: 0 },
      approachRate: 0.9,
      pinchRate: 4,
    });

    let cast = 0;
    for (let frame = 0; frame < 60; frame += 1) {
      cast += updateHandChannelTargets(s, reading([busy], frame * DT * 1000), DT).spells.length;
    }

    expect(cast).toBe(0);
    expect(s.targets.panX).not.toBe(0); // it was steering the entire time
  });

  it("lets the second hand cast while the first steers", () => {
    const s = state(true);
    const steering = palmAt(0.9, 0.5);
    const free = makeHand({ hand: "Left", pinchRate: 4 });

    const events = updateHandChannelTargets(s, reading([steering, free]), DT);

    expect(events.spells.map((event) => event.id)).toEqual(["snap"]);
    // And the cast came off the free hand, not the one flying the map.
    expect(events.spells[0]?.hand).toBe("Left");
    expect(s.output.steerHand).toBe("Right");
  });

  it("casts nothing with effects switched off, however many hands are up", () => {
    const s = state(false);
    const events = updateHandChannelTargets(
      s,
      reading([palmAt(0.9, 0.5), makeHand({ hand: "Left", pinchRate: 4 })]),
      DT,
    );
    expect(events.spells).toHaveLength(0);
  });

  it("gathers no charge while effects are off", () => {
    // Charging invisibly would make the first cast after the toggle fire
    // instantly, off motion made before the operator opted in.
    const s = state(false);
    hold(s, [palmAt(0.5, 0.5), makeHand({ hand: "Left" })], 60);
    expect(s.spells.charge).toBe(0);

    setHandEffectsEnabled(s, true);
    expect(s.spells.charge).toBe(0);
  });

  it("drops a charge in progress when effects are switched off", () => {
    const s = state(true);
    hold(s, [palmAt(0.5, 0.5), makeHand({ hand: "Left" })], 40);
    expect(s.spells.charge).toBeGreaterThan(0);

    setHandEffectsEnabled(s, false);
    expect(s.spells.charge).toBe(0);
    expect(s.output.charge).toBe(0);
  });
});

describe("the open palm is a velocity, not a position", () => {
  it("keeps travelling while the palm is held off centre", () => {
    const s = state();
    const hand = palmAt(0.85, 0.5);

    hold(s, [hand], 10);
    const afterTen = s.targets.panX;
    hold(s, [hand], 10);
    const afterTwenty = s.targets.panX;

    // The defining property. Under position control a motionless hand produces a
    // motionless camera no matter how long it is held; under rate control the
    // second ten frames travel as far as the first, which is what removes the
    // clutching — and is why the pinch clutch could be deleted rather than
    // replaced.
    expect(Math.abs(afterTen)).toBeGreaterThan(0);
    expect(Math.abs(afterTwenty)).toBeCloseTo(Math.abs(afterTen) * 2, 5);
  });

  it("does nothing at all inside the dead zone", () => {
    const s = state();
    // Just inside 0.1 viewport widths of centre — where a hand rests when its
    // owner is thinking rather than steering.
    hold(s, [palmAt(0.56, 0.53)], 30);

    expect(s.targets.panX).toBe(0);
    expect(s.targets.panY).toBe(0);
  });

  it("leaves neutral smoothly rather than jumping to a tenth of full speed", () => {
    // The classic dead-zone bug: masking the threshold instead of subtracting it
    // makes the map lurch the instant the control engages.
    const s = state();
    hold(s, [palmAt(0.5 + 0.105, 0.5)], 1);

    const justOutside = Math.abs(s.targets.panX);
    const full = state();
    hold(full, [palmAt(0.5 + 0.4, 0.5)], 1);

    expect(justOutside).toBeGreaterThan(0);
    expect(justOutside).toBeLessThan(Math.abs(full.targets.panX) * 0.05);
  });

  it("flies the camera toward the hand, so the map slides the other way", () => {
    // `Camera2D.panPixels` does `cx -= dx`, so a *positive* pan channel walks the
    // camera left and the content appears to move right. A joystick pushed right
    // means "go right", so the channel must go negative. Getting this backwards
    // is invisible except by flying the thing.
    const s = state();
    hold(s, [palmAt(0.9, 0.5)], 5);
    expect(s.targets.panX).toBeLessThan(0);

    const left = state();
    hold(left, [palmAt(0.1, 0.5)], 5);
    expect(left.targets.panX).toBeGreaterThan(0);
  });

  it("uses the same sign on both axes", () => {
    // The channel's +y is already screen-down, matching the palm centre's, so a
    // second flip on the vertical axis would inexplicably invert only that one.
    const s = state();
    hold(s, [palmAt(0.9, 0.9)], 5);

    expect(Math.sign(s.targets.panX)).toBe(Math.sign(s.targets.panY));
  });

  it("zooms in as the hand approaches and out as it withdraws", () => {
    // `zoomAt` multiplies world-units-per-pixel by its factor and the rig
    // exponentiates this channel, so zooming *in* needs the channel to go down.
    // `screenSpan` rises as the hand nears the camera.
    const near = state();
    hold(near, [palmAt(0.5, 0.5, 0.18)], 1); // captures the reference span
    hold(near, [palmAt(0.5, 0.5, 0.34)], 5);
    expect(near.targets.zoom).toBeLessThan(0);

    const far = state();
    hold(far, [palmAt(0.5, 0.5, 0.18)], 1);
    hold(far, [palmAt(0.5, 0.5, 0.09)], 5);
    expect(far.targets.zoom).toBeGreaterThan(0);
  });

  it("anchors zoom to where the hand was raised, so raising it changes nothing", () => {
    const s = state();
    hold(s, [palmAt(0.5, 0.5, 0.31)], 20);

    // An arm raised close to the camera is not a zoom command; only movement
    // relative to that starting distance is.
    expect(s.targets.zoom).toBe(0);
  });

  it("spends the palm on flying rather than on point size", () => {
    // Apparent size is the zoom stick, so it cannot also drive the gain. One
    // motion, two channels, is the coupling this rebuild exists to remove — the
    // gain now moves for exactly one reason, a snap flash.
    const s = state();
    hold(s, [palmAt(0.5, 0.5, 0.18)], 1);
    hold(s, [palmAt(0.5, 0.5, 0.34)], 5);

    expect(s.targets.pointGain).toBe(1);
  });

  it("stops when the hand drops but keeps the distance already flown", () => {
    const s = state();
    hold(s, [palmAt(0.9, 0.5)], 10);
    const flown = s.targets.panX;

    hold(s, [], 10);

    // Travel is where the operator *is*, not an offset to be relaxed away — the
    // rig never knew where the camera started, so it has nowhere to return it to.
    expect(s.targets.panX).toBe(flown);
  });

  it("re-anchors zoom when the other hand takes over steering", () => {
    const s = state();
    hold(s, [palmAt(0.5, 0.5, 0.18)], 5);
    hold(s, [makeHand({ hand: "Left", palmCentre: { x: 0.5, y: 0.5, z: 0 }, screenSpan: 0.3 })], 5);

    // The new steering hand's distance is its own neutral. Inheriting the
    // previous hand's reference span would deflect the zoom stick fully the
    // instant a differently-positioned arm took over.
    expect(s.targets.zoom).toBe(0);
    expect(s.output.steerHand).toBe("Left");
  });
});

describe("the fist freezes everything", () => {
  const fist = () =>
    makeHand({ hand: "Left", gesture: "Closed_Fist", openness: 0.1, palmCentre: { x: 0.2, y: 0.5, z: 0 } });

  it("stops the stick even with the other hand fully deflected", () => {
    const s = state();
    hold(s, [palmAt(0.9, 0.5)], 10);
    const flown = s.targets.panX;

    hold(s, [palmAt(0.9, 0.5), fist()], 30);

    expect(s.targets.panX).toBe(flown);
    expect(s.output.frozen).toBe(true);
  });

  it("stops casts too, which the previous design could not", () => {
    // Freezing used to run *after* cast recognition so that Slam — thrown with a
    // closed fist — stayed reachable, which meant the one gesture whose entire
    // promise is "stop" could not protect the operator from the one gesture that
    // threw their view away. Slam is gone and the ordering is honest.
    const s = state(true);
    const events = updateHandChannelTargets(
      s,
      reading([fist(), makeHand({ hand: "Right", pinchRate: 4 })]),
      DT,
    );
    expect(events.spells).toHaveLength(0);
  });
});

describe("a pinch is recognised and inert", () => {
  it("does not fly the camera", () => {
    // A pinch leaves the other three fingers extended, so it clears the
    // open-palm openness threshold. Without the `grab` role it would fall
    // through and be read as a joystick — every pinch would fly the map.
    const s = state();
    hold(s, [makeHand({ pinching: true, palmCentre: { x: 0.9, y: 0.5, z: 0 } })], 20);

    expect(s.targets.panX).toBe(0);
    expect(s.targets.zoom).toBe(0);
    expect(s.output.roles.Right).toBe("grab");
    expect(s.output.steerHand).toBeNull();
  });
});

describe("the pointing finger", () => {
  it("emits a waypoint for a finger held still on one spot", () => {
    const s = state();
    const pointing = pointingAt(0.4, 0.3);

    // 0.6 s of stillness at 30 Hz, plus a frame to open the capture.
    let waypoint = null;
    for (let frame = 0; frame < 25; frame += 1) {
      const events = updateHandChannelTargets(s, reading([pointing], frame * DT * 1000), DT);
      if (events.waypoint) waypoint = events.waypoint;
    }

    expect(waypoint).not.toBeNull();
    expect(waypoint?.x).toBeCloseTo(0.4, 5);
    expect(waypoint?.y).toBeCloseTo(0.3, 5);
  });

  it("fires a dwell from a perfectly steady finger", () => {
    // The regression this guards: reading the dwell through `strokeExpired`
    // never fires here, because that terminator needs `points.length > 1` and a
    // motionless fingertip never records a second point. The best case a dwell
    // has was the one case that could not work.
    const s = state();
    const pointing = pointingAt(0.4, 0.3);
    let fired = false;
    for (let frame = 0; frame < 25; frame += 1) {
      if (updateHandChannelTargets(s, reading([pointing], frame * DT * 1000), DT).waypoint) {
        fired = true;
      }
    }
    expect(fired).toBe(true);
    // Exactly one point was ever captured, which is what makes the point.
    expect(s.stroke.points.length).toBeLessThanOrEqual(1);
  });

  it("reads a moving finger as a lasso rather than a dwell", () => {
    // The two readings have to be disjoint: a drawn loop resolves as a lasso at
    // 0.4 s of stillness, and only a finger that never became drawable survives
    // to 0.6 s as a waypoint.
    const s = state();
    let sawWaypoint = false;
    let sawLasso = false;

    for (let frame = 0; frame < 40; frame += 1) {
      const angle = (frame / 20) * Math.PI * 2;
      const events = updateHandChannelTargets(
        s,
        reading(
          [pointingAt(0.5 + 0.15 * Math.cos(angle), 0.5 + 0.15 * Math.sin(angle))],
          frame * DT * 1000,
        ),
        DT,
      );
      if (events.waypoint) sawWaypoint = true;
      if (events.lasso) sawLasso = true;
    }

    // A stroke only resolves once it goes still, so the pause is part of the
    // gesture rather than test scaffolding. Held just long enough to terminate
    // the loop: the fresh capture that opens on the next frame needs its own
    // full dwell, so no waypoint can follow inside this window.
    const resting = pointingAt(0.65, 0.5);
    for (let frame = 40; frame < 60; frame += 1) {
      const events = updateHandChannelTargets(s, reading([resting], frame * DT * 1000), DT);
      if (events.waypoint) sawWaypoint = true;
      if (events.lasso) sawLasso = true;
    }

    expect(sawLasso).toBe(true);
    expect(sawWaypoint).toBe(false);
  });

  it("does not steer while pointing", () => {
    const s = state();
    hold(s, [pointingAt(0.9, 0.9)], 20);
    expect(s.targets.panX).toBe(0);
    expect(s.targets.panY).toBe(0);
  });
});

describe("settling", () => {
  it("refuses to settle while a cast is still playing", () => {
    // `handChannelsSettled` is what stops the tracker's animation frame. A wave
    // whose springs have arrived is still a wave crossing the screen, and
    // stopping the loop under it freezes the most visible thing on the page.
    const s = state(true);
    hold(s, [palmAt(0.5, 0.5), makeHand({ hand: "Left" })], 40);
    updateHandChannelTargets(
      s,
      reading([palmAt(0.5, 0.5), makeHand({ hand: "Left", approachRate: 0.9 })], 1400),
      DT,
    );

    expect(s.pulse).not.toBeNull();
    stepHandChannels(s, DT, 1);
    expect(handChannelsSettled(s)).toBe(false);
  });

  it("settles once everything in flight has finished", () => {
    const s = state();
    hold(s, [palmAt(0.9, 0.5)], 10);
    hold(s, [], 1);
    // Long enough for the pan spring to arrive on the travel it was left at.
    for (let frame = 0; frame < 600; frame += 1) stepHandChannels(s, DT, frame + 1);

    expect(handChannelsSettled(s)).toBe(true);
  });

  it("a reset leaves nothing that would keep the loop alive", () => {
    // Charge and a half-drawn stroke are both advanced only by the per-reading
    // update, which a rig whose camera was just released never calls again.
    // Leaving either behind is not a stale value, it is a 60 Hz loop that runs
    // until the tab closes.
    const s = state(true);
    hold(s, [palmAt(0.5, 0.5), makeHand({ hand: "Left" })], 40);
    hold(s, [pointingAt(0.4, 0.3)], 3);
    expect(s.stroke.active).toBe(true);

    resetHandChannels(s);
    stepHandChannels(s, DT, 99);

    expect(s.spells.charge).toBe(0);
    expect(s.stroke.active).toBe(false);
    expect(handChannelsSettled(s)).toBe(true);
  });

  it("a reset does not throw away the distance already flown", () => {
    const s = state();
    hold(s, [palmAt(0.9, 0.5)], 10);
    for (let frame = 0; frame < 400; frame += 1) stepHandChannels(s, DT, frame + 1);
    const flown = s.springs.panX.value;

    resetHandChannels(s);

    // Rebased, not zeroed. These channels measure travel, so springing them back
    // to zero would hurl the map an arbitrary distance the rig never chose.
    expect(s.targets.panX).toBe(flown);
  });
});
