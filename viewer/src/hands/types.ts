import type { Vec3 } from "./filters";

export type { Vec3 };

/** Which of the operator's hands this is, after mirror correction. */
export type Handedness = "Left" | "Right";

/**
 * The classes MediaPipe's canned gesture head emits. `None` is the null class
 * it returns when nothing scores above threshold, not an error.
 */
export type CannedGesture =
  | "None"
  | "Closed_Fist"
  | "Open_Palm"
  | "Pointing_Up"
  | "Thumb_Down"
  | "Thumb_Up"
  | "Victory"
  | "ILoveYou";

/**
 * What a hand is currently doing to the atlas. Exactly one role is active per
 * hand per frame; the arbitration order lives in `channels.ts`.
 *
 * `charge` is a sub-state of `light` rather than a peer, and which of the two a
 * hand gets depends on whether it is the steering hand: a steering palm held
 * still is simply a stick at rest, while a *free* palm held still is gathering a
 * cast. Splitting them into unrelated roles would make the overlay claim a
 * steering hand had stopped steering at the exact moment it was holding most
 * deliberately still.
 *
 * `grab` no longer drives anything. It is retained precisely so that a pinching
 * hand is *recognised and inert* rather than falling through to the openness
 * test and being read as an open palm — a pinch with the other three fingers
 * extended clears the open-palm threshold, so deleting this role would make
 * every pinch fly the camera. An explicitly inert role is the cheapest way to
 * say "seen, and deliberately ignored".
 */
export type HandRole = "idle" | "grab" | "light" | "charge" | "point" | "hold";

/**
 * A discrete gesture event — something that *fires* rather than something that
 * is held.
 *
 * The distinction matters more than it looks. Continuous channels are steered
 * and can be corrected mid-motion, so they can afford heavy smoothing and no
 * confirmation. A spell is committed the instant it is recognised and cannot be
 * taken back, which imposes two requirements the continuous path does not have:
 * it must be hard to fire by accident (hence refractory periods and velocity
 * thresholds well above hand tremor), and it must announce itself loudly enough
 * that the operator knows it fired without looking at a readout.
 *
 * **Only purely visual casts remain, and only two of them.** The set was seven,
 * and five of those were not effects at all — they were navigation and settings
 * changes wearing a gesture. `slam` reset the camera, `nextCluster` and
 * `previousCluster` flew it somewhere else, and `prism`/`aurora` toggled two
 * settings that already have sliders. Every one of them could destroy what the
 * operator was reading, off a probabilistic classifier, while their other hand
 * was steering. A cast may now change how the cloud *looks* and nothing else; if
 * a gesture would move the camera or write the store, it does not belong here.
 */
export type SpellId = "shockwave" | "snap";

export interface SpellEvent {
  id: SpellId;
  /**
   * How hard it was cast, 0..1. Carried separately from the id because the
   * same spell fired gently and fired at full charge should not look the same
   * — a discrete trigger with a continuous magnitude is what keeps an impulse
   * gesture expressive rather than binary.
   */
  power: number;
  /** Where on the viewport it originated, in mirrored viewport coordinates. */
  origin: { x: number; y: number };
  /** Which hand cast it, for the overlay. */
  hand: Handedness;
  /** `performance.now()` at the moment of the cast. */
  at: number;
}

/**
 * Human-facing name and hint for each spell, shared by the HUD and the legend.
 *
 * The hints name what the gesture does *to the atlas*, not what the rig calls
 * it internally: an operator reading the legend is deciding whether to raise
 * their hand, and "toggle the cluster hulls" is actionable where "toggle Prism"
 * is a word they would have to learn first.
 */
export const SPELL_LABEL: Record<SpellId, string> = {
  shockwave: "Shockwave",
  snap: "Snap",
};

/**
 * The whole gesture vocabulary: three hand shapes, five outcomes.
 *
 * It was fifteen outcomes off nine shapes, arbitrated by a probabilistic
 * classifier, and that is why the rig navigated badly. The failure was never a
 * threshold that wanted retuning — it was that the same physical motions were
 * read by two subsystems with nothing refereeing between them:
 *
 *  - Bringing the hand toward the camera was *both* the zoom input and the
 *    shockwave thrust, so zooming in briskly detonated the cloud.
 *  - Holding an open palm still — the posture of someone reading — was the
 *    charge input, so thinking armed the shockwave.
 *  - Opening the hand to end a pinch-drag was, by rate, a snap.
 *  - The closed fist froze the view *and*, one downward flick away, reset it;
 *    lowering a tired arm is a downward flick.
 *
 * The rebuild removes the contention rather than tuning around it. A pose now
 * means exactly one thing, and the two remaining visual casts are moved onto the
 * hand that is *not* steering (see `handEffects`), so the navigation vocabulary
 * and the effects vocabulary can never be read off the same motion.
 *
 * **Orbit is gone from the hand path entirely, and not because it was noisy.** A
 * semantic map is read the way a map is read — pan, zoom, go to — not inspected
 * the way an object is inspected. A UMAP projection has no back side worth
 * walking around: the third axis is not a dimension carrying meaning, it is
 * whatever the optimiser had left over. Orbiting it costs the operator their
 * frame of reference (there is no horizon, no up, no landmark to recover on),
 * tilts every billboarded label out of legibility, and returns nothing for it.
 * It stays available on the trackpad, where it is precise and cannot fire by
 * accident, and that is the right home for a rarely-wanted, easily-ruinous verb.
 */
export const HAND_LEGEND: readonly { pose: string; effect: string }[] = [
  { pose: "Open palm", effect: "Push it off centre to fly — further is faster" },
  { pose: "Open palm, closer/further", effect: "Zooms in and out while held" },
  { pose: "Index finger, held still", effect: "Flies to whatever you are pointing at" },
  { pose: "Index finger, moving", effect: "Draw a loop to select a region" },
  { pose: "Closed fist", effect: "Freezes everything so you can look" },
];

/**
 * The extra vocabulary the effects toggle adds, on the free hand only.
 *
 * Listed separately rather than appended to `HAND_LEGEND` because the HUD must
 * only show it while the toggle is on. A legend naming a gesture the current
 * configuration does not implement is worse than no legend at all: the operator
 * tries it, nothing happens, and they conclude the tracking is broken.
 */
export const HAND_EFFECTS_LEGEND: readonly { pose: string; effect: string }[] = [
  { pose: "Free hand, open and still", effect: "Charges — the ring fills as it gathers" },
  { pose: "Free hand, pushed forward", effect: "Shockwave: a strain front crosses the cloud" },
  { pose: "Free hand, pinch and release", effect: "Snap: flashes the whole cloud bright" },
];

/** The five fingers, thumb first, in MediaPipe's landmark order. */
export const FINGERS = ["thumb", "index", "middle", "ring", "pinky"] as const;
export type Finger = (typeof FINGERS)[number];

/**
 * The bone topology of MediaPipe's 21-landmark hand, as index pairs.
 *
 * Wrist is 0; each finger then runs four landmarks tip-ward, thumb first. The
 * last three pairs are the palm itself — knuckle to knuckle across the hand,
 * closing back to the wrist — without which a drawn skeleton is five
 * disconnected sticks radiating from a point.
 */
export const HAND_CONNECTIONS: readonly (readonly [number, number])[] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [9, 10],
  [10, 11],
  [11, 12],
  [13, 14],
  [14, 15],
  [15, 16],
  [17, 18],
  [18, 19],
  [19, 20],
  [5, 9],
  [9, 13],
  [13, 17],
  [0, 17],
] as const;

/**
 * Everything the channel layer needs from one hand in one frame, already
 * filtered, already in three.js space, already mirror-corrected.
 *
 * Nothing downstream of this interface touches a raw landmark array. That is
 * deliberate: the conversion between MediaPipe's image-space axes (+y down,
 * +z away from camera) and three.js's (+y up, +z toward the viewer), and the
 * horizontal mirror that makes the preview match the operator's mental model,
 * are each easy to apply twice or not at all. They happen once, in
 * `features.ts`, and never again.
 */
export interface HandFeatures {
  hand: Handedness;
  /** Highest-scoring canned gesture for this hand this frame. */
  gesture: CannedGesture;
  gestureScore: number;
  /** Unit vector out of the palm, three.js space. */
  palmNormal: Vec3;
  /** Unit vector along the index finger, three.js space. */
  pointDirection: Vec3;
  /** Palm centre in mirrored viewport coordinates, both axes 0..1. */
  palmCentre: Vec3;
  /** Index fingertip in mirrored viewport coordinates, both axes 0..1. */
  indexTip: Vec3;
  /**
   * Thumb-to-index distance divided by the hand's own span, so a small hand
   * far away and a large hand close by pinch at the same number.
   */
  pinchGap: number;
  /** True once the pinch has passed hysteresis and dwell. */
  pinching: boolean;
  /** Wrist-to-middle-knuckle distance in metres, from the metric landmarks. */
  span: number;
  /**
   * Apparent size in the frame, used as a proximity proxy. Rises as the hand
   * approaches the camera. Roughly 0.08 at arm's length, 0.30 up close.
   */
  screenSpan: number;
  /** Per-finger curl, 0 straight through 1 fully folded. */
  curls: Record<Finger, number>;
  /** How open the hand is overall, 0 fist through 1 flat palm. */
  openness: number;
  /**
   * Palm velocity in mirrored viewport widths per second.
   *
   * Taken from the 1€ filter's own derivative rather than differenced here. The
   * filter already maintains a smoothed derivative to drive its adaptive cutoff
   * and to predict the transport lead, and a second finite difference computed
   * off the *filtered* position would be both noisier and later than the one it
   * already has.
   */
  palmVelocity: Vec3;
  /**
   * Rate of change of apparent size, per second. Positive as the hand
   * approaches the camera, so a deliberate push at the screen is a large
   * positive value — this is what separates a thrust from a slow reach.
   */
  approachRate: number;
  /** Rate of change of the pinch gap, per second. Positive as it opens. */
  pinchRate: number;
  /** Overall palm speed in the viewport plane, a convenience for stillness tests. */
  speed: number;
  role: HandRole;
}

export interface HandReading {
  /** Monotonic timestamp in milliseconds, from `performance.now()`. */
  timestamp: number;
  hands: HandFeatures[];
  /** The hand the rig is treating as dominant, or null when none is tracked. */
  dominant: HandFeatures | null;
  /** The other tracked hand, when two are present. */
  secondary: HandFeatures | null;
}

export type HandTrackingPhase = "idle" | "requesting" | "loading" | "tracking" | "stopped" | "error";

export interface HandTrackingDiagnostics {
  /** Frames per second delivered by the camera and consumed by inference. */
  fps: number;
  /** Wall-clock milliseconds spent inside `recognizeForVideo`. */
  inferenceMs: number;
  /** Whether MediaPipe accepted the GPU delegate or fell back to CPU. */
  delegate: "GPU" | "CPU" | "unknown";
  handsVisible: number;
}

/**
 * Whether this browser can run the rig at all.
 *
 * Lives in this leaf module rather than only on the rig so the Settings page can
 * disable its toggle without importing the tracker — and with it the camera
 * lifecycle and the four asset URLs — into Seer's bundle, which has no atlas to
 * steer. It is a capability probe, not a permission check: a `true` here says
 * the APIs exist, never that the user has granted anything.
 */
export function handControlAvailable(): boolean {
  return handControlUnavailableReason() === null;
}

/**
 * Why the rig cannot run here, or `null` if it can.
 *
 * Worth separating from the boolean because the failure a user actually hits is
 * usually not "this browser has no camera". `navigator.mediaDevices` is not
 * exposed *at all* on an insecure origin, so opening this viewer over plain
 * `http://` at a LAN address — which is how it gets opened from another machine
 * — makes the probe report exactly what a camera-less browser reports. A greyed
 * out toggle reading "no camera" is a dead end when the fix is one URL away.
 */
export function handControlUnavailableReason(): string | null {
  if (typeof navigator === "undefined" || typeof WebAssembly === "undefined") {
    return "this browser cannot run the tracker";
  }
  if (typeof window !== "undefined" && !window.isSecureContext) {
    return "needs https or localhost — no camera is exposed to an insecure origin";
  }
  if (typeof navigator.mediaDevices?.getUserMedia !== "function") {
    return "this browser exposes no camera to the page";
  }
  return null;
}

export const EMPTY_HAND_DIAGNOSTICS: HandTrackingDiagnostics = {
  fps: 0,
  inferenceMs: 0,
  delegate: "unknown",
  handsVisible: 0,
};
