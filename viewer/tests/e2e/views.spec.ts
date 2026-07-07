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

test("atlas: confidence floor culls low-confidence points (gate direction locked)", async ({
  page,
}, testInfo) => {
  // the gate transpiles identically on both rungs; measure once on the
  // deterministic webgl rung where luminance is stable
  test.skip(rungOf(testInfo) === "webgpu", "gate is rung-independent; measured on webgl");
  await bootApp(page, "webgl", { view: "atlas" });
  await waitForSettle(page);

  // mean canvas luminance stands in for "how many points are lit" — every
  // point contributes additive brightness, so raising the floor (fewer points)
  // can only lower it. drawImage the live GL canvas into a 2D one to read px.
  const meanLuma = () =>
    page.evaluate(async () => {
      const cv = document.querySelector("canvas") as HTMLCanvasElement;
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      const off = document.createElement("canvas");
      off.width = cv.width;
      off.height = cv.height;
      const ctx = off.getContext("2d")!;
      ctx.drawImage(cv, 0, 0);
      const { data } = ctx.getImageData(0, 0, off.width, off.height);
      let sum = 0;
      for (let i = 0; i < data.length; i += 4) {
        sum += data[i]! * 0.299 + data[i + 1]! * 0.587 + data[i + 2]! * 0.114;
      }
      return sum / (data.length / 4);
    });

  const setFloor = async (f: number) => {
    await page.evaluate((v) => window.__store.getState().setSetting("confidenceFloor", v), f);
    await page.waitForTimeout(400); // continuous render loop; let the uniform land
  };

  await setFloor(0); // floor 0 = every point visible
  const open = await meanLuma();
  await setFloor(1); // floor 1 = only conf≥1 clustered points (noise dust exempt)
  const culled = await meanLuma();

  // the honesty gate: floor 0 shows all points, so the open atlas MUST be
  // brighter than the culled one. Locks direction — the inverted
  // `uConfFloor.step(iConf)` (floor 0 hid everything) would flip this.
  expect(open).toBeGreaterThan(culled * 1.05);
});

test("v1-style hierarchy gating: radio disabled only without edges", async ({ page }, testInfo) => {
  test.skip(rungOf(testInfo) === "webgpu", "chrome is identical on both rungs");
  await bootApp(page, "webgl");
  // all live exports are v2 — the hierarchy radio must be enabled
  const radio = page.locator('.legend input[type="radio"][value="hierarchy"]');
  await expect(radio).toBeEnabled();
});
