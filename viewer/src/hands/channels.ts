import {
  clamp,
  createSpring,
  springStep,
  type SpringConfig,
  type SpringState,
} from "./filters";
import { createSpellState, resetSpellState, updateSpells, type SpellState } from "./spells";
import {
  appendStroke,
  beginStroke,
  createStrokeCapture,
  endStroke,
  strokeExpired,
  strokeIsDrawable,
  strokeToLasso,
  type Lasso,
  type StrokeCapture,
  type StrokePoint,
} from "./strokes";
import type { HandFeatures, Handedness, HandReading, HandRole, SpellEvent } from "./types";

/**
 * The mapping layer: hand features in, atlas channel values out.
 *
 * Everything above it deals in hands, everything below it deals in a camera and
 * a set of uniforms, and keeping the translation in one place is what makes it
 * possible to change what a gesture *does* without touching either tracking or
 * rendering.
 *
 * ## One control law
 *
 * The open palm is a joystick. Displacement from the viewport centre is a pan
 * **velocity**, so the map keeps travelling while the hand is held off centre
 * and no amount of distance needs re-gripping; apparent hand size is a zoom
 * velocity on the same principle. That is the whole of continuous navigation.
 *
 * There used to be three selectable laws and a position-control pinch clutch
 * alongside them. Both are gone. The clutch was justified as the complement rate
 * control lacks — precision at the last few pixels — but `RATE_CURVE` already
 * buys that: squaring the stick means the first half of its travel is a fine
 * approach speed, so the clutch was a second answer to a question already
 * answered, and it cost a whole hand shape and a collision with the snap cast.
 * The mode picker was worse: three laws is what you ship when you have not
 * decided, and it taxes someone who only wants to look at their data.
 *
 * ## Orbit is not here at all
 *
 * Not omitted for noise — omitted because a semantic map is read like a map, not
 * inspected like an object. See the note on `HAND_LEGEND` in `types.ts`.
 *
 * ## Steering and effects can never be the same hand
 *
 * `updateSpells` is handed the hand that is *not* steering, or nothing. This is
 * the structural fix for the rig's worst defect: previously an open palm both
 * flew the camera and charged a shockwave, and a forward push both zoomed in and
 * released it, so navigating fired effects. Separating them by hand rather than
 * by threshold is what makes that impossible rather than merely unlikely — and
 * with one hand raised, the common case, no cast can fire at all.
 *
 * ## Everything else that was already true
 *
 * **Every channel is reached through a spring, never assigned.** A hand is a
 * force on the camera rather than a handle bolted to it.
 *
 * **Every channel is an offset, not an absolute.** The rig emits deltas the
 * driver composes on top of whatever the user did with the mouse, so rest is the
 * identity and releasing every hand provably returns the rig's contribution to
 * nothing rather than to some remembered pose.
 *
 * Nothing here imports three.js, deck.gl or the store. That is not incidental:
 * the unit tests run in plain Node with no DOM and no GPU.
 */

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

/**
 * Half-width of the neutral zone, in viewport widths from the centre.
 *
 * Generous — a tenth of the frame — because the cost of the two errors is wildly
 * asymmetric. Too small and the map creeps whenever the operator holds a hand up
 * to think, which reads as a broken tracker rather than as a control they are
 * touching. Too large and they have to reach slightly further to start moving,
 * which they discover in one second and never think about again.
 *
 * Note that holding a steering hand still is now *only* this. It no longer also
 * charges a cast, so "hold still and read" is a first-class posture rather than
 * a way to arm something.
 */
const RATE_DEAD_ZONE = 0.1;

/**
 * Displacement at which the stick is fully deflected, in viewport widths.
 *
 * Short of the frame edge on purpose: a hand tracked near the border of the
 * camera's field of view is exactly where MediaPipe's landmark confidence falls
 * off, so a control that only reached full speed out there would reach it least
 * reliably. Full speed lands at roughly two-thirds of the reachable box.
 */
const RATE_FULL_SCALE = 0.28;

/** Viewport widths per second of pan at full deflection. */
const RATE_PAN_SPEED = 0.85;

/**
 * Exponent shaping deflection into speed.
 *
 * Squared rather than linear so the first half of the stick's travel buys fine
 * approach speed and the last quarter buys transit speed. A linear stick has to
 * choose one or the other, and whichever it chooses is wrong for the other task.
 *
 * This now carries the fine-positioning job on its own, since the pinch clutch
 * that used to share it is gone. Flatten this curve and the first thing to break
 * is not transit — it is the operator's ability to stop somewhere precise.
 */
const RATE_CURVE = 2;

/**
 * Log-units of apparent-size change before the zoom stick leaves neutral, and
 * the change at which it is fully deflected.
 *
 * In log space because zoom is: a hand that halves its apparent distance should
 * read the same as one that doubles it, and only a log measure makes those
 * symmetric around the reference the hand was raised at.
 */
const RATE_ZOOM_DEAD = 0.1;
const RATE_ZOOM_FULL = 0.45;

/** Log-zoom units per second at full deflection — a factor of ~2.2 per second. */
const RATE_ZOOM_SPEED = 0.8;

/**
 * Bounds on the two travelling channels, in viewport widths and log-zoom units.
 *
 * These are float hygiene, not authority. Under rate control there is no
 * meaningful ceiling on how far an operator may legitimately travel, so the
 * numbers are set far past any real session rather than at a distance the map
 * might actually need.
 *
 * The zoom bound is the tighter of the two relative to its channel, and
 * deliberately: `Camera2D.clampWpp` stops the camera at its own scale limits, so
 * a zoom channel free to run far past them would wind up — the operator reverses
 * the gesture and nothing happens for several seconds while the integrator
 * unwinds slack the camera never used.
 */
const RATE_PAN_LIMIT = 3000;
const RATE_ZOOM_LIMIT = 12;

/**
 * Stillness that fires a waypoint flight, in seconds.
 *
 * Longer than `STROKE_SETTLE_SECONDS`, which is what makes the two readings of a
 * pointing finger disjoint rather than racing: a finger that has moved far
 * enough to be drawable resolves as a lasso at 0.4 s, and only a finger that
 * both stayed put and never became drawable survives to 0.6 s as a waypoint.
 */
const WAYPOINT_DWELL_SECONDS = 0.6;

/** Point-size multiplier limits, applied to the snap flash. */
const GAIN_MIN = 0.6;
const GAIN_MAX = 2;

/** Openness above which a hand counts as an open palm. */
const OPEN_PALM_THRESHOLD = 0.62;
/** Openness below which a hand counts as a closed fist holding the state. */
const CLOSED_FIST_THRESHOLD = 0.26;

/** Palm speed below which an open hand is reported as gathering charge. */
const CHARGE_ROLE_SPEED = 0.16;

/** Stillness that ends an air-drawn stroke, in seconds. */
const STROKE_SETTLE_SECONDS = 0.4;

/**
 * How fast a shockwave front crosses the viewport, in viewport widths per second.
 *
 * A wave is integrated on its own clock rather than through a spring: a spring
 * describes something being pulled toward a rest value, and a travelling front
 * has no rest value — it has a celerity.
 */
const PULSE_SPEED = 1.4;

/** Seconds for a shockwave to fade from full amplitude to nothing. */
const PULSE_LIFETIME = 1.05;

/**
 * Peak radial displacement a shockwave applies to a point, in viewport widths.
 *
 * The wave is a Gabor packet — a Gaussian envelope times one sine cycle — so it
 * carries exactly one compression front and one rarefaction. Five percent of a
 * viewport width is far more than any acoustic surface wave would produce and
 * that exaggeration is a deliberate legibility choice: the point of the gesture
 * is to show that the cloud is a real geometry rather than a picture of one. The
 * mechanism is honest — the displacement is applied to the same position
 * expression the renderer *and the id-buffer picker* read — only the amplitude
 * is theatrical.
 */
const PULSE_MAX_DISPLACEMENT = 0.05;

/** Seconds for a snap flash to decay. Short: it is a pop, not a glow. */
const FLASH_LIFETIME = 0.28;

/** Extra point gain at the peak of a snap flash. */
const FLASH_GAIN = 0.85;

const SPRING_GAIN: SpringConfig = { frequency: 2.6, damping: 0.85 };
const SPRING_ZOOM: SpringConfig = { frequency: 2, damping: 0.75 };
// The pan spring is the loosest of the set on purpose. It carries the heaviest
// visible motion, so its slight overshoot and unhurried settle are what give a
// released stick the sense of having thrown something with mass.
const SPRING_PAN: SpringConfig = { frequency: 1.5, damping: 0.62 };

// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------

export interface HandChannelOutput {
  /**
   * Whether the consumer should apply these channels at all.
   *
   * False before the rig has ever been used and again once everything has
   * settled back onto neutral, so a session that never enables hand tracking
   * renders through exactly the code path it did before the rig existed — not an
   * equivalent one.
   */
  engaged: boolean;
  /**
   * Travelled pan offset in viewport widths, +x right and +y down.
   *
   * Reported as a cumulative offset rather than a per-frame delta so that the
   * spring has something to settle *toward*; the driver differentiates it and
   * feeds the difference to `Camera2D.panPixels`, which keeps every clamp and
   * every world-space conversion in the camera that already owns them.
   */
  panX: number;
  panY: number;
  /** Travelled log-zoom offset; the driver differentiates and exponentiates it. */
  zoom: number;
  /** Multiplier on point size, carrying any live snap flash. Neutral is 1. */
  pointGain: number;
  /** True while a closed fist is holding every channel. */
  frozen: boolean;
  handsVisible: number;
  roles: Record<Handedness, HandRole>;
  /** The hand currently flying the camera, for the operator readout. */
  steerHand: Handedness | null;
  /**
   * The live shockwave: origin in mirrored viewport coordinates, front radius in
   * viewport widths, and peak displacement. Amplitude is zero when no wave is
   * running, which is what lets the driver skip the uniform write entirely.
   */
  pulse: { x: number; y: number; radius: number; amplitude: number };
  /**
   * The stroke being drawn right now, in mirrored viewport coordinates, or null.
   *
   * Live rather than only-on-completion because a lasso the operator cannot see
   * while drawing is a lasso they cannot aim.
   */
  stroke: readonly StrokePoint[] | null;
  /** Accumulated free-hand charge, 0..1, for the overlay ring. */
  charge: number;
  /** Which hand is gathering charge, or null. */
  chargingHand: Handedness | null;
}

interface SteerAnchor {
  hand: Handedness;
  referenceSpan: number;
}

export interface HandChannelState {
  /**
   * Whether the free hand may cast at all.
   *
   * Held here rather than checked by the caller because it has to gate the
   * *recognition*, not just the dispatch: charge accumulating invisibly while
   * effects are off would mean the first cast after enabling them fires
   * instantly, from a gesture made before the operator opted in.
   */
  effectsEnabled: boolean;
  output: HandChannelOutput;
  targets: {
    panX: number;
    panY: number;
    zoom: number;
    pointGain: number;
  };
  springs: {
    panX: SpringState;
    panY: SpringState;
    zoom: SpringState;
    pointGain: SpringState;
  };
  steer: SteerAnchor | null;
  stroke: StrokeCapture;
  spells: SpellState;
  /**
   * The travelling wave, integrated on its own clock.
   *
   * `age` rather than a radius is the stored quantity because the front's
   * position and its decay are both functions of elapsed time, and keeping one
   * of them as state while deriving the other invites the two to disagree after
   * a dropped frame.
   */
  pulse: { x: number; y: number; age: number; power: number } | null;
  /** Remaining life of a snap flash, and the power it was cast at. */
  flash: { age: number; power: number } | null;
  /** Clock value of the last integration, so stepping is idempotent per frame. */
  lastStepAt: number;
  /** True while at least one hand is in frame. */
  tracking: boolean;
}

export function createHandChannels(effectsEnabled = false): HandChannelState {
  return {
    effectsEnabled,
    output: {
      engaged: false,
      panX: 0,
      panY: 0,
      zoom: 0,
      pointGain: 1,
      frozen: false,
      handsVisible: 0,
      roles: { Left: "idle", Right: "idle" },
      steerHand: null,
      pulse: { x: 0.5, y: 0.5, radius: 0, amplitude: 0 },
      stroke: null,
      charge: 0,
      chargingHand: null,
    },
    targets: { panX: 0, panY: 0, zoom: 0, pointGain: 1 },
    springs: {
      panX: createSpring(0),
      panY: createSpring(0),
      zoom: createSpring(0),
      pointGain: createSpring(1),
    },
    steer: null,
    stroke: createStrokeCapture(),
    spells: createSpellState(),
    pulse: null,
    flash: null,
    lastStepAt: -1,
    tracking: false,
  };
}

/**
 * Turns the effects vocabulary on or off mid-session.
 *
 * Turning it off drops any charge and any wave in flight rather than letting
 * them finish: the operator has just said they do not want this, and an effect
 * that plays out after being switched off reads as the toggle not working.
 */
export function setHandEffectsEnabled(state: HandChannelState, enabled: boolean): void {
  if (state.effectsEnabled === enabled) return;
  state.effectsEnabled = enabled;
  if (!enabled) {
    resetSpellState(state.spells);
    state.pulse = null;
    state.flash = null;
    state.output.charge = 0;
    state.output.chargingHand = null;
    state.output.pulse.amplitude = 0;
    state.targets.pointGain = 1;
  }
}

/**
 * Returns every channel to neutral and drops every cast in flight.
 *
 * **Pan and zoom are rebased rather than zeroed, and that is the whole
 * difference between a reset and a catastrophe.** Under rate control these
 * numbers are *how far the operator has flown*, and they bear no relationship to
 * anywhere they would want to be sent back to — the rig never knew where the
 * camera started. Springing back across them would hurl the map an arbitrary
 * distance. Setting each target to the value its spring already holds leaves the
 * camera exactly where it is with nothing in flight, which is what "rest" means
 * for a channel that measures travel.
 */
export function resetHandChannels(state: HandChannelState): void {
  state.targets.panX = state.springs.panX.value;
  state.targets.panY = state.springs.panY.value;
  state.targets.zoom = state.springs.zoom.value;
  state.targets.pointGain = 1;
  state.steer = null;
  state.pulse = null;
  state.flash = null;
  resetSpellState(state.spells);
  // A half-drawn stroke and a partial charge are both unsettled work, and
  // `handChannelsSettled` correctly refuses to stop the animation frame while
  // either exists. Neither decays on its own: both are advanced only by the
  // per-reading update, which a rig whose camera has just been released never
  // calls again. Leaving them here is not a stale value — it is a 60 Hz loop
  // that runs until the tab closes.
  if (state.stroke.active) endStroke(state.stroke);
  state.output.stroke = null;
  state.output.charge = 0;
  state.output.chargingHand = null;
  state.output.pulse.amplitude = 0;
}

function assignRole(hand: HandFeatures): HandRole {
  // Recognised and inert. Without this branch a pinch — which leaves the other
  // three fingers extended and so clears the openness threshold below — would be
  // read as an open palm and fly the camera.
  if (hand.pinching) return "grab";
  if (hand.gesture === "Closed_Fist" && hand.openness < CLOSED_FIST_THRESHOLD) return "hold";
  if (hand.gesture === "Pointing_Up") return "point";
  // The canned classifier only reports Pointing_Up for a finger raised roughly
  // vertically. Drawing a lasso sends the finger everywhere, so the geometric
  // descriptor has to carry the pose once the gesture head lets go of it.
  if (hand.curls.index < 0.3 && hand.curls.middle > 0.55 && hand.curls.ring > 0.55) {
    return "point";
  }
  if (hand.openness > OPEN_PALM_THRESHOLD) {
    return hand.speed <= CHARGE_ROLE_SPEED ? "charge" : "light";
  }
  return "idle";
}

/**
 * Turns a raw deflection into a 0..1 stick drive.
 *
 * The dead zone is subtracted rather than masked, so the drive leaves neutral at
 * zero and climbs from there. Masking instead — reporting the raw magnitude once
 * it clears the threshold — makes the control jump to a tenth of full speed the
 * instant it engages, which is felt as the map lurching and is the single most
 * common way a dead zone is implemented wrongly.
 */
function stickDrive(magnitude: number, dead: number, full: number, curve: number): number {
  if (magnitude <= dead) return 0;
  const span = Math.max(full - dead, 1e-6);
  return clamp((magnitude - dead) / span, 0, 1) ** curve;
}

export interface HandChannelEvents {
  /** A completed air-drawn lasso, in mirrored viewport coordinates. */
  lasso: Lasso | null;
  /**
   * A dwelt-on point in mirrored viewport coordinates, or null.
   *
   * Reported as a bare point rather than as a resolved selection because this
   * module has no idea what is drawn at that point — resolving it needs the id
   * buffer, which lives behind the driver.
   */
  waypoint: StrokePoint | null;
  spells: readonly SpellEvent[];
}

const NO_EVENTS: HandChannelEvents = { lasso: null, waypoint: null, spells: [] };

/**
 * Recomputes channel targets from a hand reading. Called at camera cadence.
 *
 * Only targets move here. Integration happens in `stepHandChannels` at render
 * cadence, because a spring integrated at the camera's 25-30 Hz — and at that
 * cadence's jitter — reintroduces as much visible stepping as the filters just
 * removed.
 */
export function updateHandChannelTargets(
  state: HandChannelState,
  reading: HandReading,
  dt: number,
): HandChannelEvents {
  const hands = reading.hands;
  state.tracking = hands.length > 0;
  state.output.handsVisible = hands.length;
  state.output.roles = { Left: "idle", Right: "idle" };

  for (const hand of hands) {
    hand.role = assignRole(hand);
    state.output.roles[hand.hand] = hand.role;
  }

  // The freeze gate comes first, and now genuinely means everything. It used to
  // sit *after* spell recognition so that Slam — cast with a closed fist — stayed
  // reachable, which meant the one gesture whose entire promise is "stop" could
  // not protect the operator from the one gesture that threw their view away.
  // With Slam gone the ordering is free to be honest.
  const holding = hands.some((hand) => hand.role === "hold");
  state.output.frozen = holding;

  if (holding) {
    state.output.charge = state.spells.charge;
    state.output.chargingHand = state.spells.chargingHand;
    return NO_EVENTS;
  }

  // --- Hand assignment: steering first, effects strictly second --------------
  const steerHand = hands.find((hand) => hand.role === "light" || hand.role === "charge") ?? null;
  // Any hand that is not the one flying the camera. Identity comparison, not a
  // role test, because the free hand casts with several different shapes (open
  // to charge, pinched to snap) and the only property that matters is that it is
  // a different hand.
  const freeHand = state.effectsEnabled
    ? (hands.find((hand) => hand !== steerHand) ?? null)
    : null;

  const spells = updateSpells(state.spells, freeHand, dt, reading.timestamp);
  for (const spell of spells) applySpell(state, spell);

  state.output.charge = state.spells.charge;
  state.output.chargingHand = state.spells.chargingHand;

  // --- The open palm flies the camera ---------------------------------------
  if (steerHand) {
    if (!state.steer || state.steer.hand !== steerHand.hand) {
      // Anchor apparent size to however far away the hand was when it took over,
      // so the zoom stick starts at neutral rather than wherever the operator's
      // arm happened to be.
      state.steer = { hand: steerHand.hand, referenceSpan: Math.max(steerHand.screenSpan, 1e-3) };
    }
    state.output.steerHand = steerHand.hand;
    steerByRate(state, steerHand, dt);
  } else {
    state.steer = null;
    state.output.steerHand = null;
    // Dropping the hand stops the stick, and that is all it does — the travel
    // already made is where the operator *is*, not an offset to be relaxed away.
  }

  // --- The pointing finger: a drawn region, or a dwelt-on destination --------
  const drawn = updateStroke(state, hands.find((hand) => hand.role === "point") ?? null, dt);
  state.output.stroke = state.stroke.active ? state.stroke.points : null;
  return { lasso: drawn.lasso, waypoint: drawn.waypoint, spells };
}

/**
 * Flies the camera from an open palm held as a joystick.
 *
 * Two sign conventions meet here and both are easy to get backwards, so they are
 * stated rather than left to be re-derived from three files.
 *
 * **Pan.** `Camera2D.panPixels` does `cx -= dx`, so a *positive* pan channel
 * walks the camera left and the content therefore appears to move right. A stick
 * is the "go there" metaphor — pushing it right means go right, with the content
 * sliding left underneath — so this path subtracts.
 *
 * **Zoom.** `zoomAt` multiplies world-units-per-pixel by its factor, so a factor
 * under 1 zooms *in*, and the rig exponentiates this channel — meaning zooming in
 * requires the channel to go *down*. Bringing the hand closer grows `screenSpan`,
 * so approaching subtracts.
 */
function steerByRate(state: HandChannelState, hand: HandFeatures, dt: number): void {
  const steer = state.steer;
  if (!steer) return;

  // --- Pan: displacement from the frame centre is a velocity ----------------
  const offsetX = hand.palmCentre.x - 0.5;
  const offsetY = hand.palmCentre.y - 0.5;
  const deflection = Math.hypot(offsetX, offsetY);
  const drive = stickDrive(deflection, RATE_DEAD_ZONE, RATE_FULL_SCALE, RATE_CURVE);

  if (drive > 0 && deflection > 1e-6) {
    const speed = RATE_PAN_SPEED * drive * dt;
    // Both axes subtract: the channel's +y is already screen-down, matching the
    // palm centre's, so no second flip belongs here. Getting this wrong makes
    // the vertical axis invert while the horizontal one stays correct, which
    // reads as a tracking bug rather than as a sign error.
    state.targets.panX = clamp(
      state.targets.panX - (offsetX / deflection) * speed,
      -RATE_PAN_LIMIT,
      RATE_PAN_LIMIT,
    );
    state.targets.panY = clamp(
      state.targets.panY - (offsetY / deflection) * speed,
      -RATE_PAN_LIMIT,
      RATE_PAN_LIMIT,
    );
  }

  // --- Zoom: apparent size relative to where the hand was raised ------------
  const spanRatio = Math.log(Math.max(hand.screenSpan, 1e-4) / steer.referenceSpan);
  const zoomDrive = stickDrive(Math.abs(spanRatio), RATE_ZOOM_DEAD, RATE_ZOOM_FULL, RATE_CURVE);
  if (zoomDrive > 0) {
    state.targets.zoom = clamp(
      state.targets.zoom - Math.sign(spanRatio) * RATE_ZOOM_SPEED * zoomDrive * dt,
      -RATE_ZOOM_LIMIT,
      RATE_ZOOM_LIMIT,
    );
  }
}

/**
 * Applies the immediate effect of a cast.
 *
 * Both remaining casts are purely visual and live entirely inside this layer,
 * which is now a rule rather than a coincidence: a cast that needed the store or
 * the cluster list would be a navigation action wearing a gesture, and those are
 * exactly what the rebuild removed.
 */
function applySpell(state: HandChannelState, spell: SpellEvent): void {
  if (spell.id === "shockwave") {
    state.pulse = { x: spell.origin.x, y: spell.origin.y, age: 0, power: spell.power };
    return;
  }
  state.flash = { age: 0, power: spell.power };
}

interface StrokeResult {
  lasso: Lasso | null;
  waypoint: StrokePoint | null;
}

const NO_STROKE: StrokeResult = { lasso: null, waypoint: null };

/**
 * Advances the air-drawn stroke and watches it for a dwell.
 *
 * The dwell needs no capture machinery of its own, which is the reason it is
 * here rather than in a parallel state machine: this capture already tracks how
 * long the fingertip has been effectively stationary, already terminates on
 * stillness, and already refuses to become a lasso when the path is too short to
 * enclose anything. A finger held on a cluster is precisely a stroke that ran
 * out of stillness without ever becoming drawable, so reading a waypoint out of
 * it costs one branch and cannot double-fire against the lasso.
 */
function updateStroke(
  state: HandChannelState,
  pointHand: HandFeatures | null,
  dt: number,
): StrokeResult {
  const capture = state.stroke;

  if (!pointHand) {
    if (!capture.active) return NO_STROKE;
    // A finger that left the frame did not dwell on anything — it was withdrawn.
    return resolveStroke(state, false);
  }

  const tip = { x: pointHand.indexTip.x, y: pointHand.indexTip.y };

  if (!capture.active) {
    beginStroke(capture, tip);
    return NO_STROKE;
  }

  appendStroke(capture, tip, dt);

  // Stillness is read directly rather than through `strokeExpired`, because that
  // terminator requires `points.length > 1` and a finger held genuinely still
  // never produces a second point — `appendStroke` counts sub-tremor motion as
  // stillness instead of recording it. A perfectly steady point, which is the
  // *best* case a dwell has, was therefore the one case that could never fire.
  // Only a stroke that ended because it went still is a dwell; one that hit the
  // four-second ceiling with the finger still moving is a stroke that ran long,
  // and flying to wherever it happened to stop is a destination nobody chose.
  const dwelt = capture.still >= WAYPOINT_DWELL_SECONDS;

  if (dwelt || strokeExpired(capture, STROKE_SETTLE_SECONDS)) {
    return resolveStroke(state, dwelt);
  }

  return NO_STROKE;
}

function resolveStroke(state: HandChannelState, dwelt: boolean): StrokeResult {
  const drawable = strokeIsDrawable(state.stroke);
  const points = endStroke(state.stroke);
  state.output.stroke = null;
  if (drawable) return { lasso: strokeToLasso(points), waypoint: null };
  if (!dwelt) return NO_STROKE;
  const tip = points[points.length - 1];
  return { lasso: null, waypoint: tip ? { x: tip.x, y: tip.y } : null };
}

/** Distance from neutral at which a channel counts as settled. */
const REST_EPSILON = 1e-3;

/**
 * Integrates every spring toward its target and republishes the output.
 *
 * `frameTime` makes this idempotent within one render frame: the driver, the HUD
 * and the store bridge all read the same frame and any of them may be first, so
 * integrating on each call would advance the springs two or three times per
 * frame and change how the rig feels depending on which surfaces are mounted.
 */
export function stepHandChannels(
  state: HandChannelState,
  delta: number,
  frameTime: number,
): HandChannelOutput {
  if (frameTime === state.lastStepAt) return state.output;
  state.lastStepAt = frameTime;

  const { springs, targets, output } = state;

  output.panX = springStep(springs.panX, targets.panX, delta, SPRING_PAN);
  output.panY = springStep(springs.panY, targets.panY, delta, SPRING_PAN);
  output.zoom = springStep(springs.zoom, targets.zoom, delta, SPRING_ZOOM);
  output.pointGain = springStep(springs.pointGain, targets.pointGain, delta, SPRING_GAIN);

  // --- Snap flash: a short spike on top of the settled gain ------------------
  if (state.flash) {
    state.flash.age += delta;
    const life = 1 - state.flash.age / FLASH_LIFETIME;
    if (life <= 0) {
      state.flash = null;
    } else {
      // Squared rather than linear: a pop that decays linearly reads as a slow
      // dimmer, not as a flash.
      output.pointGain = clamp(
        output.pointGain + FLASH_GAIN * state.flash.power * life * life,
        GAIN_MIN,
        GAIN_MAX,
      );
    }
  }

  // --- Shockwave: a travelling front across the cloud ------------------------
  if (state.pulse) {
    state.pulse.age += delta;
    const life = 1 - state.pulse.age / PULSE_LIFETIME;
    if (life <= 0) {
      state.pulse = null;
      output.pulse.amplitude = 0;
    } else {
      output.pulse.x = state.pulse.x;
      output.pulse.y = state.pulse.y;
      output.pulse.radius = state.pulse.age * PULSE_SPEED;
      output.pulse.amplitude = PULSE_MAX_DISPLACEMENT * state.pulse.power * life * life;
    }
  } else {
    output.pulse.amplitude = 0;
  }

  output.engaged = state.tracking || !isAtRest(state);
  return output;
}

/**
 * Whether every spring has arrived at its target and stopped moving.
 *
 * This answers a different question from `engaged`, and conflating the two is a
 * live battery bug rather than a style point. `engaged` asks whether the channel
 * *values* differ from neutral — travelled pan keeps that true indefinitely,
 * which is correct, because the consumer must go on applying it. `settled` asks
 * whether the integrator still has work to do. Driving the animation frame from
 * `engaged` would leave a 60 Hz loop running forever after the first flight.
 */
export function handChannelsSettled(state: HandChannelState): boolean {
  const { springs, targets } = state;
  const settled = (spring: SpringState, target: number): boolean =>
    Math.abs(spring.value - target) <= REST_EPSILON &&
    Math.abs(spring.velocity) <= REST_EPSILON * 8;

  // A cast in flight is unsettled work regardless of where the springs are.
  // Without these the rAF stops the moment the springs arrive and a shockwave
  // freezes partway across the cloud — the springs are at rest, so every
  // existing test for stillness is satisfied while the most visible thing on
  // screen is stuck.
  if (state.pulse || state.flash) return false;
  // Charge is drawn as a growing ring, so it has to keep frames coming while it
  // fills or drains.
  if (state.spells.charge > REST_EPSILON) return false;
  // A stroke in progress is drawn point by point.
  if (state.stroke.active) return false;

  return (
    settled(springs.panX, targets.panX) &&
    settled(springs.panY, targets.panY) &&
    settled(springs.zoom, targets.zoom) &&
    settled(springs.pointGain, targets.pointGain)
  );
}

/**
 * Whether the rig is contributing nothing the consumer needs to apply.
 *
 * Pan and zoom are deliberately *not* tested against zero. They measure travel,
 * and an operator who has flown somewhere is meant to stay there — so the honest
 * question for those two is whether they are still moving, which is what
 * `handChannelsSettled` answers and what the tracker's rAF loop uses. What this
 * asks is narrower: is there any live decoration left to draw?
 */
function isAtRest(state: HandChannelState): boolean {
  if (state.pulse || state.flash) return false;
  return Math.abs(state.output.pointGain - 1) <= REST_EPSILON;
}
