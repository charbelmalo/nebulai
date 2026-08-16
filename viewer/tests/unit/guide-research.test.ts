import { describe, expect, it } from "vitest";

import {
  GUIDE_RESEARCH,
  guideResearchFor,
} from "../../src/chrome/guideResearch";

describe("Internals guide research", () => {
  it("provides three distinct, secure references for all 25 live features", () => {
    const featureIds = Object.keys(GUIDE_RESEARCH) as Array<keyof typeof GUIDE_RESEARCH>;
    expect(featureIds).toHaveLength(25);

    for (const featureId of featureIds) {
      const references = guideResearchFor(featureId);

      expect(references, featureId).toHaveLength(3);
      expect(new Set(references.map((reference) => reference.url)).size, featureId).toBe(3);
      for (const reference of references) {
        expect(reference.title, featureId).not.toBe("");
        expect(reference.citation, featureId).not.toBe("");
        expect(reference.url, featureId).toMatch(/^https:\/\//);
      }
    }
  });
});
