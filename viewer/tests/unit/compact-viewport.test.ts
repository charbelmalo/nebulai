/** `isCompactViewport` is the pure seed logic behind `$compactViewport`. It
 *  is split out of chrome/state.ts specifically so this can run in plain
 *  Node — vitest has no jsdom here, so `matchMedia` doesn't exist, and
 *  `null` (the "no matchMedia" case) must read as "not compact" rather than
 *  throw or default to compact. */

import { describe, expect, it } from "vitest";
import { isCompactViewport } from "../../src/chrome/state";

describe("isCompactViewport", () => {
  it("is true when the media query matches", () => {
    expect(isCompactViewport({ matches: true })).toBe(true);
  });

  it("is false when the media query does not match", () => {
    expect(isCompactViewport({ matches: false })).toBe(false);
  });

  it("is false when there is no matchMedia at all (plain Node)", () => {
    expect(isCompactViewport(null)).toBe(false);
  });
});
