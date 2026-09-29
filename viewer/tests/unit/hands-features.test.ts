/** Landmark extraction. Three coordinate conversions stack up in `features.ts`
 *  — image→scene axes, the preview mirror, and MediaPipe's selfie handedness
 *  convention — and each is easy to apply twice or not at all. The symptom of
 *  getting one wrong is a camera that swings the wrong way, which reads as a
 *  projection bug and is not one. These are the tests that say which. */

import { describe, expect, it } from "vitest";
import {
  ageHandFilterBank,
  asCannedGesture,
  correctHandedness,
  createHandFilterBank,
  extractHandFeatures,
  type HandFilterBank,
  type RawPoint,
} from "../../src/hands/features";
import type { CannedGesture, HandFeatures, Handedness } from "../../src/hands/types";

const DT = 1 / 30;

/** Where the thumb sits, which is the only thing the pinch gap measures.
 *  `near` is chosen to land inside the trigger's hysteresis band: above the
 *  0.34 engage threshold, below the 0.5 release one. */
type ThumbPose = "open" | "near" | "pinch";

interface HandOptions {
  /** 0 = every finger straight, 1 = every finger folded back on itself. */
  fold?: number;
  thumb?: ThumbPose;
  /** Overall hand size — a stand-in for standing further from the camera. */
  scale?: number;
  /** Where the hand sits in the (unmirrored) camera frame. */
  centre?: { x: number; y: number };
  /** Negates world x — a mirror image of the same pose. */
  mirror?: boolean;
}

const THUMB_CMC: Record<ThumbPose, { x: number; y: number }> = {
  open: { x: -0.09, y: 0.03 },
  near: { x: -0.045, y: 0.035 },
  pinch: { x: -0.045, y: 0.02 },
};

/**
 * A synthetic 21-landmark hand: palm in the z = 0 plane, fingers pointing up
 * (MediaPipe's world y is *down*, so "up" is negative), thumb off to the index
 * side. Deliberately built from the MediaPipe layout rather than captured from a
 * recording, so every number in an assertion is one this file chose.
 */
function makeHand(options: HandOptions = {}): { landmarks: RawPoint[]; world: RawPoint[] } {
  const fold = options.fold ?? 0;
  const centre = options.centre ?? { x: 0.5, y: 0.5 };
  const sign = options.mirror ? -1 : 1;
  const scale = options.scale ?? 1;
  const thumb = THUMB_CMC[options.thumb ?? "open"];

  const world: RawPoint[] = Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: 0 }));
  const put = (index: number, x: number, y: number): void => {
    world[index] = { x: sign * x * scale, y: y * scale, z: 0 };
  };

  put(0, 0, 0.09); // wrist
  // Metacarpal heads across the palm. Index sits at −x and pinky at +x: with
  // the palm toward the camera that is what a *right* hand looks like in the
  // raw (unmirrored) frame.
  const roots: readonly { at: number; x: number; y: number }[] = [
    { at: 1, x: thumb.x, y: thumb.y },
    { at: 5, x: -0.045, y: 0 },
    { at: 9, x: -0.015, y: 0 },
    { at: 13, x: 0.015, y: 0 },
    { at: 17, x: 0.045, y: 0 },
  ];
  for (const root of roots) {
    put(root.at, root.x, root.y);
    put(root.at + 1, root.x, root.y - 0.03); // pip
    put(root.at + 2, root.x, root.y - 0.05); // dip
    // The distal segment either continues the proximal one (straight) or
    // reverses it (folded); `fingerCurl` reads the angle between the two.
    put(root.at + 3, root.x, root.y - 0.05 + (fold > 0.5 ? 0.02 : -0.02));
  }

  // Image landmarks are the same pose under a fixed affine, so a claim about a
  // screen coordinate is traceable back to a world one.
  const landmarks = world.map((point) => ({
    x: centre.x + point.x * 2,
    y: centre.y + point.y * 2,
    z: 0,
  }));
  return { landmarks, world };
}

function extract(
  hand: Handedness,
  options: HandOptions = {},
  bank: HandFilterBank = createHandFilterBank(),
  gesture: CannedGesture = "None",
): HandFeatures | null {
  const { landmarks, world } = makeHand(options);
  return extractHandFeatures({
    landmarks,
    world,
    hand,
    gesture,
    gestureScore: gesture === "None" ? 0 : 0.9,
    bank,
    dt: DT,
  });
}

describe("handedness and gesture labels", () => {
  it("inverts MediaPipe's selfie-convention handedness", () => {
    // We feed the raw camera frame, which is not flipped; MediaPipe labels as
    // though it were. Getting this wrong flips the palm normal into the screen.
    expect(correctHandedness("Left")).toBe("Right");
    expect(correctHandedness("Right")).toBe("Left");
  });

  it("collapses an unrecognised gesture name to None", () => {
    expect(asCannedGesture("Victory")).toBe("Victory");
    expect(asCannedGesture("ILoveYou")).toBe("ILoveYou");
    expect(asCannedGesture("Thumb_Sideways")).toBe("None");
    expect(asCannedGesture(undefined)).toBe("None");
  });
});

describe("feature extraction", () => {
  it("refuses a truncated landmark array rather than reading past its end", () => {
    const { landmarks, world } = makeHand();
    const features = extractHandFeatures({
      landmarks: landmarks.slice(0, 12),
      world,
      hand: "Right",
      gesture: "None",
      gestureScore: 0,
      bank: createHandFilterBank(),
      dt: DT,
    });
    expect(features).toBeNull();
  });

  it("mirrors the viewport position exactly once, and only in x", () => {
    const features = extract("Right", { centre: { x: 0.3, y: 0.42 } });
    expect(features).not.toBeNull();
    // Middle MCP sits at world x = −0.015, so its raw image x is 0.27.
    expect(features?.palmCentre.x).toBeCloseTo(1 - 0.27, 10);
    // y keeps its top-left origin so the overlay needs no second conversion.
    expect(features?.palmCentre.y).toBeCloseTo(0.42, 10);
  });

  it("points the palm normal at the operator for a flat hand", () => {
    const features = extract("Right");
    expect(features?.palmNormal.z).toBeCloseTo(1, 6);
    expect(features?.palmNormal.x).toBeCloseTo(0, 6);
    expect(features?.palmNormal.y).toBeCloseTo(0, 6);
  });

  it("builds the normal in true space — a mirrored hand faces the same way", () => {
    // This is the file header's one subtle claim, stated as a test. A right hand
    // and its mirror image (which MediaPipe would call a left hand) are both
    // palm-to-camera, so in the mirrored preview both normals must point at the
    // operator. An implementation that reflects the two palm edge vectors and
    // *then* crosses them gets −z here, because a reflection reverses handedness
    // and the cross product is handed.
    const right = extract("Right");
    const left = extract("Left", { mirror: true });
    expect(left?.palmNormal.z).toBeCloseTo(right?.palmNormal.z ?? 0, 6);
    expect(left?.palmNormal.z).toBeCloseTo(1, 6);
  });

  it("reads curl from segment angle, so openness spans the full range", () => {
    const open = extract("Right", { fold: 0 });
    expect(open?.curls.index).toBeCloseTo(0, 10);
    expect(open?.openness).toBeCloseTo(1, 10);

    const fist = extract("Right", { fold: 1 });
    // Doubling back past perpendicular saturates the measure at 1.
    expect(fist?.curls.index).toBeCloseTo(1, 10);
    expect(fist?.openness).toBeCloseTo(0, 10);
  });

  it("scales the pinch gap by the hand's own span, not by pixels", () => {
    // Same pose at half the size: the ratio has to be identical, or thresholds
    // would depend on how far away the operator is standing.
    const near = extract("Right", { thumb: "pinch" });
    const far = extract("Right", { thumb: "pinch", scale: 0.5 });
    expect(near?.pinchGap).toBeCloseTo(far?.pinchGap ?? 0, 10);
    expect(near?.pinchGap).toBeLessThan(0.34);
    // The spans themselves are not equal — only the ratio is.
    expect(far?.span).toBeCloseTo((near?.span ?? 0) / 2, 10);
  });

  it("requires the pinch to be held before it engages", () => {
    const bank = createHandFilterBank();
    // The gap is well inside the 0.34 engage threshold — but one frame at 30 Hz
    // is 0.033 s against a 0.05 s dwell, so a hand passing through does not
    // register a pinch.
    expect(extract("Right", { thumb: "pinch" }, bank)?.pinching).toBe(false);
    expect(extract("Right", { thumb: "pinch" }, bank)?.pinching).toBe(true);
  });

  it("does not read an open hand as a pinch", () => {
    const bank = createHandFilterBank();
    for (let i = 0; i < 6; i++) {
      expect(extract("Right", {}, bank)?.pinching).toBe(false);
    }
  });

  it("keeps a pinch through the hysteresis band", () => {
    const bank = createHandFilterBank();
    extract("Right", { thumb: "pinch" }, bank);
    extract("Right", { thumb: "pinch" }, bank);

    let held: HandFeatures | null = null;
    for (let i = 0; i < 10; i++) held = extract("Right", { thumb: "near" }, bank);

    // Premise of the test: the gap has drifted past the value that would have
    // engaged it. It stays held anyway, because release is a separate, looser
    // threshold — otherwise a held grab would chatter on hand tremor alone.
    expect(held?.pinchGap ?? 0).toBeGreaterThan(0.34);
    expect(held?.pinchGap ?? 1).toBeLessThan(0.5);
    expect(held?.pinching).toBe(true);
  });
});

describe("filter bank lifetime", () => {
  it("resets a hand that left the frame instead of sweeping it back in", () => {
    const bank = createHandFilterBank();
    extract("Right", { centre: { x: 0.2, y: 0.5 } }, bank);
    extract("Right", { centre: { x: 0.2, y: 0.5 } }, bank);

    // Gone for longer than HAND_REACQUIRE_SECONDS (0.35).
    ageHandFilterBank(bank, 0.5);

    const back = extract("Right", { centre: { x: 0.8, y: 0.5 } }, bank);
    // A fresh filter takes its first sample verbatim and has no velocity, so
    // the lead term contributes nothing: this is the raw mirrored coordinate.
    expect(back?.palmCentre.x).toBeCloseTo(1 - 0.77, 10);
  });

  it("does not reset a hand that was only briefly occluded", () => {
    const bank = createHandFilterBank();
    extract("Right", { centre: { x: 0.2, y: 0.5 } }, bank);
    extract("Right", { centre: { x: 0.2, y: 0.5 } }, bank);

    ageHandFilterBank(bank, 0.1);

    const back = extract("Right", { centre: { x: 0.8, y: 0.5 } }, bank);
    // Still the same filter, so the jump is smoothed rather than taken whole.
    expect(back?.palmCentre.x).not.toBeCloseTo(1 - 0.77, 3);
  });
});
