/** Perf budgets from the plan, asserted via the window.__perf probe: parse
 *  <800ms, boot <3s, steady-state main-thread frame work p95 ≤16.7ms, cached
 *  dataset switch <400ms.
 *
 *  The frame-work budget runs on BOTH rungs. It used to skip webgpu entirely
 *  ("budgets asserted on the deterministic rung"), which meant no WebGPU perf
 *  regression was detectable by CI — and that is precisely how the rung came to
 *  carry a full-viewport pick pass and a full-res bloom chain unnoticed while
 *  being the tier we advertise as the fast one. What made webgpu "non
 *  deterministic" was GPU wall-clock, but p95FrameWorkMs measures main-thread
 *  work inside the frame callback, which is comparable across rungs. The
 *  wall-clock-sensitive budgets (parse, boot) stay on webgl. */
import { expect, test } from "@playwright/test";
import { rungOf } from "../helpers";
import { bootApp } from "./helpers";

test("boot and parse budgets", async ({ page }, testInfo) => {
  test.skip(rungOf(testInfo) === "webgpu", "wall-clock budgets on the deterministic rung");
  const { errors } = await bootApp(page, "webgl", { frozen: false });

  const perf = await page.evaluate(() => window.__perf);
  expect(perf.parseMs, "worker parse+columnarize").toBeLessThan(800);
  expect(perf.bootMs, "boot to first frame").toBeLessThan(3000);
  expect(errors).toEqual([]);
});

test("steady-state frame budget holds on this rung", async ({ page }, testInfo) => {
  const rung = rungOf(testInfo);
  const { errors, tier } = await bootApp(page, rung, { frozen: false });
  test.skip(rung === "webgpu" && tier !== "webgpu", `probe fell back to ${tier}`);

  // Let the loop fill its 120-frame p95 window at steady state. Measure the
  // work NebulAI performs inside each callback, not the interval at which the
  // browser schedules rAF: headless Chrome commonly caps that scheduler at
  // 20–30 Hz even when application work is ~1ms, which is not an app
  // performance regression.
  await page.waitForTimeout(2500);
  const p95 = await page.evaluate(() => window.__perf.p95FrameWorkMs);
  expect(p95, `steady-state p95 main-thread frame work (${rung})`).toBeLessThanOrEqual(16.7);
  expect(errors).toEqual([]);
});

/** The 3-D hover path is the one that regressed silently: it renders an id pass
 *  per pick, so a viewport-sized target quietly doubled the cloud's draw cost at
 *  30Hz. Pin the target to a single pixel so that can't come back. */
test("the id-pick target stays 1×1, not viewport-sized", async ({ page }, testInfo) => {
  const rung = rungOf(testInfo);
  const { tier } = await bootApp(page, rung, { frozen: false });
  test.skip(rung === "webgpu" && tier !== "webgpu", `probe fell back to ${tier}`);

  // drive the store rather than the select — the id picker is built by the
  // 2-D→3-D transition, not by the control that triggers it
  await page.evaluate(() => window.__store.getState().setDims(3));
  await page.waitForFunction(
    () => (window.__driver as unknown as { idPicker: unknown | null })?.idPicker != null,
    undefined,
    { timeout: 10_000 },
  );

  const rt = await page.evaluate(() => {
    const { idPicker } = window.__driver as unknown as {
      idPicker: { rt: { width: number; height: number } };
    };
    return [idPicker.rt.width, idPicker.rt.height];
  });
  expect(rt, "id-pick render target").toEqual([1, 1]);
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
