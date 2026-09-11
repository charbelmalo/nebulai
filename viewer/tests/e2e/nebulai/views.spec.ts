/** Per-view smoke + screenshot goldens. `?frozen=1` pins the time uniform so
 *  every golden is a deterministic frame; goldens live per project under
 *  tests/e2e/goldens/{webgl,webgpu}/. */
import { expect, test } from "@playwright/test";
import { rungOf } from "../helpers";
import { bootApp, waitForSettle, type View } from "./helpers";

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

test("hand control is reachable from the sidebar, and only where it steers", async ({
  page,
}, testInfo) => {
  test.skip(rungOf(testInfo) === "webgpu", "chrome is identical on both rungs");
  // Granted up front: the panel mounts the instant the setting flips and asks
  // for the camera immediately. Headless has no device, so the rig lands in its
  // error phase — which is the honest outcome here. What is under test is the
  // control's reachability, not the tracker.
  await page.context().grantPermissions(["camera"]);
  await bootApp(page, "webgl");

  // The bug this pins: the rig shipped with its only switch at the bottom of
  // the Settings overlay's General tab, so the feature was invisible from the
  // view it drives and nobody could turn it on.
  const sidebar = page.locator(".sidebar");
  const toggle = sidebar.getByRole("switch", { name: "Hand control" });
  await expect(toggle).toBeVisible();
  await expect(toggle).toBeEnabled(); // localhost is a secure context
  await expect(page.locator(".hand-rig")).toHaveCount(0);

  await toggle.click();
  await expect(page.locator(".hand-rig")).toBeVisible();
  expect(await page.evaluate(() => window.__store.getState().settings.handTracking)).toBe(true);

  // The effects toggle appears only once the rig is on, and starts OFF. There
  // used to be a three-way steering picker here; the modes are gone, and what
  // replaced them is the one thing the operator genuinely has to choose —
  // whether their free hand may throw effects at the map they are reading.
  const effects = sidebar.getByRole("switch", { name: "Hand effects" });
  await expect(effects).toBeVisible();
  expect(await page.evaluate(() => window.__store.getState().settings.handEffects)).toBe(false);

  // The legend must describe what is actually in force. With effects off, the
  // three effect gestures do nothing at all, and an operator following
  // instructions that do nothing concludes the camera is broken rather than
  // that they are reading the wrong page.
  await page.locator(".hand-rig-legend-toggle").click();
  const legend = page.locator(".hand-rig-legend");
  await expect(legend).toContainText("Open palm");
  await expect(legend).toContainText("further is faster");
  await expect(legend).toContainText("Flies to whatever you are pointing at");
  await expect(legend).toContainText("Freezes everything");
  await expect(legend).not.toContainText("Shockwave");

  await effects.click();
  expect(await page.evaluate(() => window.__store.getState().settings.handEffects)).toBe(true);
  await expect(legend).toContainText("Shockwave");
  // Named as the free hand's, everywhere it is described: the whole correction
  // is that a cast rides the hand that is not steering.
  await expect(legend).toContainText("Free hand");

  await effects.click();
  await expect(legend).not.toContainText("Shockwave");

  // Turning the rig on must not silently change the view. An earlier revision
  // flipped the atlas into 3-D on the operator's behalf, because one of the
  // three steering laws could not work in 2-D — a mode that needed the map
  // rebuilt around it was a mode that did not belong on a hand.
  expect(await page.evaluate(() => window.__store.getState().dims)).toBe(2);

  // The panel's own Turn off must be a real second exit, not decoration.
  await page.locator(".hand-rig-stop").click();
  await expect(page.locator(".hand-rig")).toHaveCount(0);
  expect(await page.evaluate(() => window.__store.getState().settings.handTracking)).toBe(false);

  // Only the atlas has a driver to steer, so the row must not advertise itself
  // in a view where turning it on would do nothing.
  await page.evaluate(() => window.__store.getState().setViewMode("chord"));
  await expect(sidebar.getByRole("switch", { name: "Hand control" })).toHaveCount(0);
});

test("channel lens: the permalink lights it, and the filter narrows the map", async ({
  page,
}, testInfo) => {
  // the lens is a colour/opacity change in the same shader on both rungs;
  // measured once on the deterministic webgl rung, like the confidence floor
  test.skip(rungOf(testInfo) === "webgpu", "lens is rung-independent; measured on webgl");

  // the exit criterion, addressed the way a reader would reach it: a URL
  const { errors } = await bootApp(page, "webgl", {
    view: "atlas",
    hash: "page=map&model=gpt2&channel=we_centroid_dist",
  });
  const hasChannels = await page.evaluate(async () => {
    const res = await fetch("out/gpt2/channels.json");
    return res.ok;
  });
  test.skip(!hasChannels, "this deploy ships no out/gpt2/channels.json");

  await page.waitForFunction(
    () => window.__store.getState().channel.id === "we_centroid_dist",
    undefined,
    { timeout: 10_000 },
  );
  await waitForSettle(page);

  // the chip in the search panel reflects the lens the URL lit — the two
  // routes into the same state must not disagree
  const chip = page.locator(".lens-chip.is-on");
  await expect(chip).toHaveText("near the centroid");

  // the readout prints RAW values, so the extreme one is the number the
  // episode quotes rather than a normalised restatement
  const first = page.locator(".lens-row").first().locator(".lens-value");
  await expect(first).toHaveText("1.534");

  // mean luminance stands in for "how many points are lit" (same argument as
  // the confidence-floor test): filtering to the knot can only dim the map
  const meanLuma = () =>
    page.evaluate(async () => {
      const cv = document.querySelector("canvas") as HTMLCanvasElement;
      // the GL canvas has no preserved drawing buffer: it must be sampled
      // inside a frame, which is why this waits two rAFs before drawImage
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

  const wide = await meanLuma();
  await page.locator(".lens-narrow").click();
  await page.waitForFunction(() => window.__store.getState().channel.window !== null, undefined, {
    timeout: 10_000,
  });
  await waitForSettle(page);
  const narrow = await meanLuma();
  expect(narrow).toBeLessThan(wide);

  // and the window travels in the channel's raw units, not as an index pair
  const hash = await page.evaluate(() => location.hash);
  expect(hash).toContain("channel=we_centroid_dist");
  expect(hash).toMatch(/crange=1\.5\d+%2C1\.\d+/);

  expect(errors).toEqual([]);
});

test("channel lens: an unknown channel in the URL is dropped, not half-applied", async ({
  page,
}, testInfo) => {
  test.skip(rungOf(testInfo) === "webgpu", "chrome is identical on both rungs");
  // an unknown channel id in the URL is dropped, not half-applied: the driver
  // is the one place that knows which channels exist, and it puts the lens
  // away rather than painting the map by a column nobody measured
  await bootApp(page, "webgl", {
    view: "atlas",
    hash: "page=map&model=gpt2&channel=not_a_channel",
  });
  await waitForSettle(page);
  await expect(page.locator(".lens-chip.is-on")).toHaveCount(0);
  expect(await page.evaluate(() => window.__store.getState().channel.id)).toBeNull();
});

test("v1-style hierarchy gating: radio disabled only without edges", async ({ page }, testInfo) => {
  test.skip(rungOf(testInfo) === "webgpu", "chrome is identical on both rungs");
  await bootApp(page, "webgl");
  // all live exports are v2 — the hierarchy radio must be enabled
  const radio = page.locator('.legend input[type="radio"][value="hierarchy"]');
  await expect(radio).toBeEnabled();
});
