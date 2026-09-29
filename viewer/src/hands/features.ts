import {
  clamp,
  createOneEuro,
  createOneEuroVec3,
  createSchmitt,
  oneEuroStep,
  oneEuroVec3Lead,
  oneEuroVec3Step,
  schmittStep,
  type OneEuroState,
  type OneEuroVec3State,
  type SchmittState,
  type Vec3,
} from "./filters";
import { FINGERS, type CannedGesture, type Finger, type HandFeatures, type Handedness } from "./types";

/**
 * Landmark extraction: raw MediaPipe output in, conditioned control features
 * out. Everything about coordinate spaces is settled here and nowhere else.
 *
 * Three conversions stack up and each one is easy to apply twice or forget:
 *
 *  1. MediaPipe's axes are image axes — +x right, +y *down*, +z *away* from the
 *     camera. three.js is +y up, +z toward the viewer. So (x, -y, -z).
 *  2. The preview is mirrored, because an operator watching an unmirrored feed
 *     of their own hand cannot aim. Mirroring is a reflection in x.
 *  3. Handedness is reported by MediaPipe on the assumption that the input was
 *     already flipped horizontally (its documented selfie convention). We feed
 *     the raw camera frame, which is not flipped, so the label is inverted.
 *
 * The one thing worth knowing about (2): a normal is not a direction you can
 * reflect by reflecting its inputs. Reflecting the two palm edge vectors and
 * crossing them yields the *negative* of the correct mirrored normal, because a
 * reflection reverses handedness and a cross product is handed. So the normal
 * is built in true space and reflected once, at the end.
 */

// MediaPipe hand landmark indices.
const WRIST = 0;
const THUMB_TIP = 4;
const INDEX_MCP = 5;
const INDEX_TIP = 8;
const MIDDLE_MCP = 9;
const PINKY_MCP = 17;

const FINGER_JOINTS: Record<Finger, readonly [number, number, number, number]> = {
  thumb: [1, 2, 3, 4],
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  pinky: [17, 18, 19, 20],
};

/**
 * Whether the preview — and therefore the control space — is mirrored.
 *
 * Verify both this and `SWAP_HANDEDNESS` the same way, in one pass: hold up
 * your right hand, palm to the camera, and check that the readout says "Right"
 * and that moving your hand right moves the highlight right. If handedness is
 * inverted the palm normal points into the screen instead of out of it, and the
 * camera swings the wrong way — a failure that looks like a sign error in the
 * projection but is not.
 */
const MIRRORED_PREVIEW = true;
const SWAP_HANDEDNESS = true;

/**
 * Lead time applied to the palm and fingertip streams, in seconds.
 *
 * This buys back most of the capture-to-uniform transport delay. It is kept
 * well under the ~90 ms worst case it is compensating: past roughly 60 ms the
 * prediction error on a direction change costs more than the latency it saves.
 */
const POSITION_LEAD_SECONDS = 0.045;
const DIRECTION_LEAD_SECONDS = 0.035;

/**
 * Pinch thresholds, expressed as a fraction of the hand's own span so they hold
 * across operators and distances. Engage is tighter than release; the dwell
 * stops a hand that merely passes through a pinch from registering one.
 */
const PINCH_ENGAGE = 0.34;
const PINCH_RELEASE = 0.5;
const PINCH_DWELL_SECONDS = 0.05;

/** A hand absent this long is treated as new and its filters are reset. */
export const HAND_REACQUIRE_SECONDS = 0.35;

export interface HandFilterBank {
  palmNormal: OneEuroVec3State;
  pointDirection: OneEuroVec3State;
  palmCentre: OneEuroVec3State;
  indexTip: OneEuroVec3State;
  pinchGap: OneEuroState;
  screenSpan: OneEuroState;
  pinch: SchmittState;
  /** Seconds since this bank last received a sample. */
  absentFor: number;
  initialised: boolean;
}

export function createHandFilterBank(): HandFilterBank {
  return {
    // Directions are noisier than positions and matter more, so they get a
    // lower floor cutoff and a larger speed term: very quiet when held, fully
    // out of the way when swept.
    palmNormal: createOneEuroVec3({ minCutoff: 0.9, beta: 0.06 }),
    pointDirection: createOneEuroVec3({ minCutoff: 1.1, beta: 0.07 }),
    palmCentre: createOneEuroVec3({ minCutoff: 1.3, beta: 0.05 }),
    indexTip: createOneEuroVec3({ minCutoff: 1.7, beta: 0.09 }),
    // The pinch gap drives a threshold, so it is filtered hard. Latency on an
    // engagement is far less objectionable than a false one.
    pinchGap: createOneEuro({ minCutoff: 1.6, beta: 0.02 }),
    screenSpan: createOneEuro({ minCutoff: 0.8, beta: 0.02 }),
    pinch: createSchmitt(),
    absentFor: Number.POSITIVE_INFINITY,
    initialised: false,
  };
}

export function resetHandFilterBank(bank: HandFilterBank): void {
  const fresh = createHandFilterBank();
  bank.palmNormal = fresh.palmNormal;
  bank.pointDirection = fresh.pointDirection;
  bank.palmCentre = fresh.palmCentre;
  bank.indexTip = fresh.indexTip;
  bank.pinchGap = fresh.pinchGap;
  bank.screenSpan = fresh.screenSpan;
  bank.pinch = fresh.pinch;
  bank.absentFor = Number.POSITIVE_INFINITY;
  bank.initialised = false;
}

export interface RawPoint {
  x: number;
  y: number;
  z: number;
}

const ORIGIN: RawPoint = { x: 0, y: 0, z: 0 };

/**
 * Indexed read with a defined miss.
 *
 * `noUncheckedIndexedAccess` is on in this project, and it is on for a reason:
 * a landmark array truncated by a partial inference is a real thing MediaPipe
 * can hand us. Every read here is bounds-checked by the length guard at the top
 * of `extractHandFeatures`, so the fallback is unreachable in practice — but
 * spelling it as the origin rather than asserting non-null keeps a future
 * caller that skips the guard producing a degenerate hand instead of a crash
 * inside the camera loop.
 */
function at(points: readonly RawPoint[], index: number): RawPoint {
  return points[index] ?? ORIGIN;
}

function distance(a: RawPoint, b: RawPoint): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function normalise(out: Vec3): Vec3 {
  const length = Math.hypot(out.x, out.y, out.z);
  if (length < 1e-6) {
    out.x = 0;
    out.y = 0;
    out.z = 1;
    return out;
  }
  out.x /= length;
  out.y /= length;
  out.z /= length;
  return out;
}

/**
 * Converts a MediaPipe world-space vector into three.js space and applies the
 * preview mirror. Positions and plain directions both go through here; the palm
 * normal does not, because a normal reflects differently (see the file header).
 */
function toSceneSpace(x: number, y: number, z: number, out: Vec3): Vec3 {
  out.x = MIRRORED_PREVIEW ? -x : x;
  out.y = -y;
  out.z = -z;
  return out;
}

/**
 * Finger curl from the angle between the proximal and distal segments.
 *
 * A straight finger has its two segments parallel, so their dot product is 1; a
 * fully folded finger doubles back past perpendicular. Using an angle rather
 * than a tip-to-palm distance keeps the measure independent of hand size and of
 * how far away the operator is standing.
 */
function fingerCurl(world: readonly RawPoint[], finger: Finger): number {
  const [mcp, pip, dip, tip] = FINGER_JOINTS[finger];
  const mcpPoint = at(world, mcp);
  const pipPoint = at(world, pip);
  const dipPoint = at(world, dip);
  const tipPoint = at(world, tip);

  const proximal = {
    x: pipPoint.x - mcpPoint.x,
    y: pipPoint.y - mcpPoint.y,
    z: pipPoint.z - mcpPoint.z,
  };
  const distal = {
    x: tipPoint.x - dipPoint.x,
    y: tipPoint.y - dipPoint.y,
    z: tipPoint.z - dipPoint.z,
  };
  const proximalLength = Math.hypot(proximal.x, proximal.y, proximal.z);
  const distalLength = Math.hypot(distal.x, distal.y, distal.z);
  if (proximalLength < 1e-6 || distalLength < 1e-6) return 0;

  const alignment =
    (proximal.x * distal.x + proximal.y * distal.y + proximal.z * distal.z) /
    (proximalLength * distalLength);

  return clamp((1 - alignment) / 1.6, 0, 1);
}

export function correctHandedness(rawLabel: string): Handedness {
  const raw: Handedness = rawLabel === "Left" ? "Left" : "Right";
  if (!SWAP_HANDEDNESS) return raw;
  return raw === "Left" ? "Right" : "Left";
}

export function asCannedGesture(name: string | undefined): CannedGesture {
  switch (name) {
    case "Closed_Fist":
    case "Open_Palm":
    case "Pointing_Up":
    case "Thumb_Down":
    case "Thumb_Up":
    case "Victory":
    case "ILoveYou":
      return name;
    default:
      return "None";
  }
}

export interface ExtractOptions {
  landmarks: readonly RawPoint[];
  world: readonly RawPoint[];
  hand: Handedness;
  gesture: CannedGesture;
  gestureScore: number;
  bank: HandFilterBank;
  dt: number;
}

/**
 * Builds the conditioned feature set for one hand.
 *
 * Scratch vectors are module-scoped and reused: this runs per hand per camera
 * frame for the life of the session, and the allocation would otherwise be a
 * steady drip of garbage straight into the render loop's frame budget.
 */
const scratchNormal: Vec3 = { x: 0, y: 0, z: 0 };
const scratchPoint: Vec3 = { x: 0, y: 0, z: 0 };
const scratchCentre: Vec3 = { x: 0, y: 0, z: 0 };
const scratchTip: Vec3 = { x: 0, y: 0, z: 0 };

export function extractHandFeatures(options: ExtractOptions): HandFeatures | null {
  const { landmarks, world, hand, gesture, gestureScore, bank, dt } = options;
  if (landmarks.length < 21 || world.length < 21) return null;

  if (bank.absentFor > HAND_REACQUIRE_SECONDS && bank.initialised) {
    // The hand has been gone long enough that continuing its filters would
    // sweep the camera across the atlas as it re-enters. Start clean.
    resetHandFilterBank(bank);
  }
  bank.absentFor = 0;
  bank.initialised = true;

  const wrist = at(world, WRIST);
  const indexMcp = at(world, INDEX_MCP);
  const pinkyMcp = at(world, PINKY_MCP);
  const worldIndexTip = at(world, INDEX_TIP);

  // Palm normal, built in true (unmirrored) space so the cross product stays
  // consistent, then reflected once.
  const edgeA = {
    x: indexMcp.x - wrist.x,
    y: indexMcp.y - wrist.y,
    z: indexMcp.z - wrist.z,
  };
  const edgeB = {
    x: pinkyMcp.x - wrist.x,
    y: pinkyMcp.y - wrist.y,
    z: pinkyMcp.z - wrist.z,
  };

  // For a right hand facing the camera this cross product points away from the
  // camera, so it is negated to get the direction the palm actually faces. A
  // left hand is the mirror of that case and takes the opposite sign.
  const handSign = hand === "Right" ? -1 : 1;
  scratchNormal.x = handSign * (edgeA.y * edgeB.z - edgeA.z * edgeB.y);
  scratchNormal.y = handSign * (edgeA.z * edgeB.x - edgeA.x * edgeB.z);
  scratchNormal.z = handSign * (edgeA.x * edgeB.y - edgeA.y * edgeB.x);
  normalise(scratchNormal);

  // MediaPipe -> three.js, then the mirror, applied to the normal itself.
  const sceneNormal: Vec3 = {
    x: MIRRORED_PREVIEW ? -scratchNormal.x : scratchNormal.x,
    y: -scratchNormal.y,
    z: -scratchNormal.z,
  };

  toSceneSpace(
    worldIndexTip.x - indexMcp.x,
    worldIndexTip.y - indexMcp.y,
    worldIndexTip.z - indexMcp.z,
    scratchPoint,
  );
  normalise(scratchPoint);

  const span = distance(wrist, at(world, MIDDLE_MCP));
  const pinchGapRaw =
    span > 1e-5 ? distance(at(world, THUMB_TIP), worldIndexTip) / span : 1;

  const screenWrist = at(landmarks, WRIST);
  const screenMiddle = at(landmarks, MIDDLE_MCP);
  const screenIndexTip = at(landmarks, INDEX_TIP);
  const screenSpanRaw = Math.hypot(screenWrist.x - screenMiddle.x, screenWrist.y - screenMiddle.y);

  // Viewport coordinates: mirrored in x to match the preview, y left as-is with
  // the origin at the top so overlay drawing needs no second conversion.
  scratchCentre.x = MIRRORED_PREVIEW ? 1 - screenMiddle.x : screenMiddle.x;
  scratchCentre.y = screenMiddle.y;
  scratchCentre.z = screenMiddle.z;
  scratchTip.x = MIRRORED_PREVIEW ? 1 - screenIndexTip.x : screenIndexTip.x;
  scratchTip.y = screenIndexTip.y;
  scratchTip.z = screenIndexTip.z;

  const filteredNormal: Vec3 = { x: 0, y: 0, z: 0 };
  oneEuroVec3Step(bank.palmNormal, sceneNormal, dt, filteredNormal);
  oneEuroVec3Lead(bank.palmNormal, DIRECTION_LEAD_SECONDS, filteredNormal);
  normalise(filteredNormal);

  const filteredPoint: Vec3 = { x: 0, y: 0, z: 0 };
  oneEuroVec3Step(bank.pointDirection, scratchPoint, dt, filteredPoint);
  oneEuroVec3Lead(bank.pointDirection, DIRECTION_LEAD_SECONDS, filteredPoint);
  normalise(filteredPoint);

  const filteredCentre: Vec3 = { x: 0, y: 0, z: 0 };
  oneEuroVec3Step(bank.palmCentre, scratchCentre, dt, filteredCentre);
  oneEuroVec3Lead(bank.palmCentre, POSITION_LEAD_SECONDS, filteredCentre);

  const filteredTip: Vec3 = { x: 0, y: 0, z: 0 };
  oneEuroVec3Step(bank.indexTip, scratchTip, dt, filteredTip);
  oneEuroVec3Lead(bank.indexTip, POSITION_LEAD_SECONDS, filteredTip);

  const pinchGap = oneEuroStep(bank.pinchGap, pinchGapRaw, dt);
  const screenSpan = oneEuroStep(bank.screenSpan, screenSpanRaw, dt);
  const pinching = schmittStep(
    bank.pinch,
    pinchGap,
    dt,
    PINCH_ENGAGE,
    PINCH_RELEASE,
    PINCH_DWELL_SECONDS,
  );

  const curls = {} as Record<Finger, number>;
  for (const finger of FINGERS) {
    curls[finger] = fingerCurl(world, finger);
  }
  const openness = clamp(1 - (curls.index + curls.middle + curls.ring + curls.pinky) / 4, 0, 1);

  // Kinematics, read straight off the filters' own derivative estimates. See
  // the note on `palmVelocity` in types.ts for why these are not differenced
  // again here.
  const palmVelocity: Vec3 = {
    x: bank.palmCentre.x.velocity,
    y: bank.palmCentre.y.velocity,
    z: bank.palmCentre.z.velocity,
  };
  const speed = Math.hypot(palmVelocity.x, palmVelocity.y);

  return {
    hand,
    gesture,
    gestureScore,
    palmNormal: filteredNormal,
    pointDirection: filteredPoint,
    palmCentre: filteredCentre,
    indexTip: filteredTip,
    pinchGap,
    pinching,
    span,
    screenSpan,
    curls,
    openness,
    palmVelocity,
    approachRate: bank.screenSpan.velocity,
    pinchRate: bank.pinchGap.velocity,
    speed,
    role: "idle",
  };
}

/** Ages a bank that received no sample this frame. */
export function ageHandFilterBank(bank: HandFilterBank, dt: number): void {
  bank.absentFor += Math.max(dt, 0);
}
