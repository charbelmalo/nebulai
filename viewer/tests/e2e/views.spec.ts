/** Per-view smoke + screenshot goldens. `?frozen=1` pins the time uniform so
 *  every golden is a deterministic frame; goldens live per project under
 *  tests/e2e/goldens/{webgl,webgpu}/. */
import { expect, test } from "@playwright/test";
import { bootApp, rungOf, waitForSettle, type View } from "./helpers";

const VIEWS: View[] = ["atlas", "chord", "hierarchy"];

for (const view of VIEWS) {
  test(`${view}: boots clean, meta line honest, matches golden`, async ({ page }, testInfo) => {
    const rung = rungOf(testInfo);
    const { errors, tier } = await bootApp(page, rung, { view });
    test.skip(rung === "webgpu" && tier !== "webgpu", `probe fell back to ${tier}`);

    // boot fly-in + crossfade must fully settle; t is frozen so after that
    // the frame is deterministic
    await waitForSettle(page);

    // the honesty line: provenance visible in every mode
    const meta = (await page.locator(".boot-status").textContent()) ?? "";
    expect(meta).toContain("pts");
    expect(meta).toContain("% noise");
    expect(meta).toContain("namer:");
    expect(meta).toContain("edges:");

    expect(errors).toEqual([]);
    await expect(page).toHaveScreenshot(`${view}.png`);
  });
}

test("compare: boots clean with the label-space caveat (webgpu only)", async ({ page }, testInfo) => {
  const rung = rungOf(testInfo);
  test.skip(rung !== "webgpu", "CompareDriver is WebGPU-only by design");
  const { errors, tier } = await bootApp(page, rung, { view: "compare" });
  test.skip(tier !== "webgpu", `probe fell back to ${tier}`);

  await waitForSettle(page);
  const meta = (await page.locator(".boot-status").textContent()) ?? "";
  expect(meta).toContain("compare:");
  expect(meta).toContain("label space, not model geometry");
  expect(errors).toEqual([]);
  await expect(page).toHaveScreenshot("compare.png");
});

test("v1-style hierarchy gating: radio disabled only without edges", async ({ page }, testInfo) => {
  test.skip(rungOf(testInfo) === "webgpu", "chrome is identical on both rungs");
  await bootApp(page, "webgl");
  // all live exports are v2 — the hierarchy radio must be enabled
  const radio = page.locator('.legend input[type="radio"][value="hierarchy"]');
  await expect(radio).toBeEnabled();
});
