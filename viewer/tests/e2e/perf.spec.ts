/** Perf budgets from the plan, asserted via the window.__perf probe on the
 *  deterministic webgl rung: parse <800ms, boot <3s, steady-state p95
 *  ≤16.7ms, cached dataset switch <400ms. */
import { expect, test } from "@playwright/test";
import { bootApp, rungOf } from "./helpers";

test("boot, parse and steady-state frame budgets", async ({ page }, testInfo) => {
  test.skip(rungOf(testInfo) === "webgpu", "budgets asserted on the deterministic rung");
  const { errors } = await bootApp(page, "webgl", { frozen: false });

  const perf = await page.evaluate(() => window.__perf);
  expect(perf.parseMs, "worker parse+columnarize").toBeLessThan(800);
  expect(perf.bootMs, "boot to first frame").toBeLessThan(3000);

  // let the loop fill its 120-frame p95 window at steady state
  await page.waitForTimeout(2500);
  const p95 = await page.evaluate(() => window.__perf.p95FrameMs);
  expect(p95, "steady-state p95 frame time").toBeLessThanOrEqual(16.7);
  expect(errors).toEqual([]);
});

test("cached dataset switch lands under 400ms", async ({ page }, testInfo) => {
  test.skip(rungOf(testInfo) === "webgpu", "budgets asserted on the deterministic rung");
  await bootApp(page, "webgl");
  const first = await page.evaluate(() => window.__store.getState().datasetId);

  // populate the column cache with a second dataset (network-bound, uncapped)
  await page.locator("#sel-dataset").selectOption("distilgpt2");
  await page.waitForFunction(
    () =>
      window.__store.getState().datasetId === "distilgpt2" &&
      !window.__store.getState().loading.active,
    undefined,
    { timeout: 30_000 },
  );

  // switching back must hit the cache — this is the budgeted path
  const t0 = Date.now();
  await page.locator("#sel-dataset").selectOption(first!);
  await page.waitForFunction(
    (id) => window.__store.getState().datasetId === id && !window.__store.getState().loading.active,
    first,
    { timeout: 5_000 },
  );
  expect(Date.now() - t0, "cached switch").toBeLessThan(400);
});
