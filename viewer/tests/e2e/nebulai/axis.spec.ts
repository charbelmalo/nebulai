/** The direction axis (P1), end to end, on a real GPU.
 *
 *  `axis-layout.test.ts` already pins the CPU mirror of the blend against the
 *  documented GPU mix at t = 0, 0.5 and 1. What no unit test can reach is the
 *  thing this spec is for: that the three-way blend in `PointsLayer` — map
 *  layout, 3-D layout, axis layout — actually lands on the GPU, that the null
 *  cloud is drawn beside the real one (R5), and that arriving by permalink
 *  produces the same frame as driving the rail by hand.
 *
 *  The golden is WebGPU-only by the plan's own instruction. It is also the
 *  rung where a silent failure is possible: exceeding WebGPU's eight-buffer
 *  vertex limit is rejected WITHOUT an error, and the axis attribute was the
 *  seventh and eighth lanes. A golden is the only thing that notices a cloud
 *  that quietly stopped moving.
 */
import { expect, test } from "@playwright/test";
import { rungOf } from "../helpers";
import { bootApp, waitForSettle } from "./helpers";

/** The one direction on the gpt2 map that clears both gates in
 *  `axisRefusal()`: it has a null block (R5) and its four projection channels
 *  carry its own space tag (D2). */
const DIRECTION = "male-minus-female-names";

test("axis: a direction lays the map out, with its null cloud under it", async ({
  page,
}, testInfo) => {
  const rung = rungOf(testInfo);
  test.skip(rung !== "webgpu", "the axis golden is the webgpu rung's (see header)");

  const { errors, tier } = await bootApp(page, rung, {
    hash: `model=gpt2&axis=${DIRECTION}&axist=1`,
  });
  test.skip(tier !== "webgpu", `probe fell back to ${tier}`);

  // the direction sidecar is fetched after the map; the permalink's axis is
  // applied when it lands, so wait on the STATE rather than on a duration
  await page.waitForFunction(
    (id) => {
      const s = window.__store.getState() as unknown as {
        datasetId: string | null;
        axis: { directionId: string | null; t: number; showNull: boolean };
      };
      return s.datasetId === "gpt2" && s.axis.directionId === id && s.axis.t > 0.99;
    },
    DIRECTION,
    { timeout: 30_000 },
  );
  await waitForSettle(page);

  // R5: the ghost ships with the figure. A golden of an axis with its null
  // switched off would be the picture this plan exists to refuse.
  const showNull = await page.evaluate(
    () => (window.__store.getState() as unknown as { axis: { showNull: boolean } }).axis.showNull,
  );
  expect(showNull).toBe(true);

  expect(errors).toEqual([]);
  await expect(page).toHaveScreenshot("axis.png");
});
