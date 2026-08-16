/** Perf budgets from the plan, asserted via the window.__perf probe on the
 *  deterministic webgl rung: parse <800ms, boot <3s, steady-state main-thread
 *  frame work p95 ≤16.7ms, cached dataset switch <400ms. */
import { expect, test } from "@playwright/test";
import { rungOf } from "../helpers";
import { bootApp } from "./helpers";

test("boot, parse and steady-state frame budgets", async ({ page }, testInfo) => {
  test.skip(rungOf(testInfo) === "webgpu", "budgets asserted on the deterministic rung");
  const { errors } = await bootApp(page, "webgl", { frozen: false });

  const perf = await page.evaluate(() => window.__perf);
  expect(perf.parseMs, "worker parse+columnarize").toBeLessThan(800);
  expect(perf.bootMs, "boot to first frame").toBeLessThan(3000);

  // Let the loop fill its 120-frame p95 window at steady state. Measure the
  // work NebulAI performs inside each callback, not the interval at which the
  // browser schedules rAF: headless Chrome commonly caps that scheduler at
  // 20–30 Hz even when application work is ~1ms, which is not an app
  // performance regression.
  await page.waitForTimeout(2500);
  const p95 = await page.evaluate(() => window.__perf.p95FrameWorkMs);
  expect(p95, "steady-state p95 main-thread frame work").toBeLessThanOrEqual(16.7);
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
  await page.evaluate(() => {
    window.__perf.datasetSwitchMs = undefined;
  });
  await page.locator("#sel-dataset").selectOption(first!);
  await page.waitForFunction(
    (id) => window.__store.getState().datasetId === id && !window.__store.getState().loading.active,
    first,
    { timeout: 5_000 },
  );
  const switchMs = await page.evaluate(() => window.__perf.datasetSwitchMs);
  expect(switchMs, "cached switch").toBeLessThan(400);
});
