import { describe, expect, it } from "vitest";

import {
  GUIDE_RESEARCH,
  guideResearchFor,
} from "../../src/chrome/guideResearch";

describe("Internals guide research", () => {
  /** Every live view, including the 26th.
   *
   *  `InterpFeature.id` is typed `GuideResearchId`, so a view registered with
   *  no evidence behind it is a COMPILE error and this number cannot silently
   *  fall behind the registry. What the literal still catches is the other
   *  direction: evidence left here for a view that no longer ships, and the
   *  moment someone adds a key without noticing they have added a view. It was
   *  25 through 2026-07-10 and became 26 when the intervention rail landed
   *  (ATTRACTORS-PLAN phase 4). Bump it deliberately, with the view. */
  const LIVE_FEATURES = 26;

  it("provides at least three distinct, secure references for every live feature", () => {
    const featureIds = Object.keys(GUIDE_RESEARCH) as Array<keyof typeof GUIDE_RESEARCH>;
    expect(featureIds).toHaveLength(LIVE_FEATURES);

    for (const featureId of featureIds) {
      const references = guideResearchFor(featureId);

      // three is the floor, not the quota: a view that changes the model's
      // forward pass carries a fourth (the intervention rail cites steering,
      // ablation, SAE clamping and patching practice, and dropping one of them
      // to satisfy an equality would be losing evidence to satisfy a test).
      expect(references.length, featureId).toBeGreaterThanOrEqual(3);
      expect(new Set(references.map((reference) => reference.url)).size, featureId).toBe(
        references.length,
      );
      for (const reference of references) {
        expect(reference.title, featureId).not.toBe("");
        expect(reference.citation, featureId).not.toBe("");
        expect(reference.url, featureId).toMatch(/^https:\/\//);
      }
    }
  });
});
