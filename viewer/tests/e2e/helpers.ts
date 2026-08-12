/** Genuinely shared e2e plumbing — the pieces every spec in both
 *  tests/e2e/nebulai/ and tests/e2e/seer/ depend on. Anything that only one
 *  instrument's specs can use (e.g. Nebulai's `bootApp`, which navigates `/`
 *  and waits on `window.__store.getState().dataset !== null` — a condition
 *  Seer's document never satisfies) lives under that instrument's own
 *  directory instead. See tests/e2e/nebulai/helpers.ts.
 */
import type { TestInfo } from "@playwright/test";

export type Rung = "webgl" | "webgpu";

export function rungOf(testInfo: TestInfo): Rung {
  return testInfo.project.name === "webgpu" ? "webgpu" : "webgl";
}
