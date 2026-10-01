/** Atlas 3-D orbit keeps the cloud in view — the regression behind "the whole
 *  point cloud vanishes mid-orbit while the labels keep tracking".
 *
 *  Three defects compounded on the starter map, whose pos3 frame sits at
 *  z ≈ 4–10: the 3-D fit framed only the xy bounds (856 of 4,096 points on
 *  screen at rest), the view-center orbit pivot sat ~570 px above the view
 *  center (so every orbit swung the cloud around an off-screen axis), and the
 *  camera's depth window hung off the z = 0 ground point, which the pivot's
 *  compensating pan slides tens of map-widths along the view ray near the
 *  horizon — every point then fell behind the near plane while the HTML label
 *  pills, which have no depth clip, stayed put. The math is renderer-agnostic,
 *  so one rung is enough. */
import { expect, test, type Page } from "@playwright/test";
import { rungOf } from "../helpers";
import { bootApp, waitForSettle } from "./helpers";

const STARTER = "gpt2-small__sae__blocks.8.hook_resid_pre";

test.beforeEach(({}, testInfo) => {
  test.skip(rungOf(testInfo) === "webgpu", "camera math is identical on both rungs");
});

interface Probe {
  n: number;
  behindNear: number;
  pastFar: number;
  onScreen: number;
  pivot: [number, number] | null;
  el: number;
}

/** Every point through the live camera matrices, with the same morph mix
 *  PointsLayer's positionNode applies. */
function probe(page: Page): Promise<Probe> {
  return page.evaluate(() => {
    type V = { x: number; y: number; z: number; project(c: unknown): V };
    const d = window.__driver as unknown as {
      dataset: { columns: { pos2: Float32Array; pos3: Float32Array; count: number } };
      morph: number;
      camera: {
        near: number;
        far: number;
        position: { constructor: new (x: number, y: number, z: number) => V };
        matrixWorldInverse: { elements: number[] };
        projectionMatrix: { elements: number[] };
        updateMatrixWorld(): void;
      };
      cam: { viewportW: number; viewportH: number };
      orbitPivot: unknown;
      pivotWorld(): [number, number, number];
      orbitAngles(): [number, number];
    };
    const { pos2: p, pos3: q, count: n } = d.dataset.columns;
    const m = d.morph;
    d.camera.updateMatrixWorld();
    const e = d.camera.matrixWorldInverse.elements;
    const pm = d.camera.projectionMatrix.elements;
    let behindNear = 0, pastFar = 0, onScreen = 0;
    for (let i = 0; i < n; i++) {
      const x = p[i * 2]! + (q[i * 3]! - p[i * 2]!) * m;
      const y = p[i * 2 + 1]! + (q[i * 3 + 1]! - p[i * 2 + 1]!) * m;
      const z = q[i * 3 + 2]! * m;
      const depth = -(e[2]! * x + e[6]! * y + e[10]! * z + e[14]!);
      if (depth < d.camera.near) behindNear++;
      else if (depth > d.camera.far) pastFar++;
      const nx = pm[0]! * (e[0]! * x + e[4]! * y + e[8]! * z + e[12]!) + pm[12]!;
      const ny = pm[5]! * (e[1]! * x + e[5]! * y + e[9]! * z + e[13]!) + pm[13]!;
      if (Math.abs(nx) <= 1 && Math.abs(ny) <= 1) onScreen++;
    }
    let pivot: [number, number] | null = null;
    if (d.orbitPivot) {
      const [wx, wy, wz] = d.pivotWorld();
      const v = new d.camera.position.constructor(wx, wy, wz).project(d.camera);
      pivot = [((v.x + 1) / 2) * d.cam.viewportW, ((1 - v.y) / 2) * d.cam.viewportH];
    }
    return { n, behindNear, pastFar, onScreen, pivot, el: d.orbitAngles()[1] };
  });
}

test("3-D orbit to the horizon and around keeps every point drawn and on screen", async ({ page }, testInfo) => {
  const { errors } = await bootApp(page, rungOf(testInfo), { hash: `model=${STARTER}`, frozen: false });
  await page.evaluate(() => window.__store.getState().setDims(3));
  await waitForSettle(page);

  // the toggle lands inside the first 400 ms of page life — the camera must
  // still take the 3-D re-frame (an idle wheel-orbit clock of 0 read as "an
  // orbit gesture just now" and skipped it, leaving the flat fit behind)
  const fit = await page.evaluate(() => {
    const d = window.__driver as unknown as {
      cam: { wpp: number; fitFor(...a: unknown[]): [number, number, number] };
      frameBox3(): [number, number, number, number];
    };
    const b = d.frameBox3();
    return { wpp: d.cam.wpp, want: d.cam.fitFor(b[0], b[1], b[2], b[3], 72, { l: 0, r: 0, t: 0, b: 0 })[2] };
  });
  expect(fit.wpp / fit.want).toBeCloseTo(1, 3);

  // the fit frames the tilted cloud, not its xy bounds
  const rest = await probe(page);
  expect(rest.behindNear + rest.pastFar).toBe(0);
  expect(rest.onScreen).toBe(rest.n);

  // a real right-button orbit on bare canvas — pills off so none swallows it
  await page.evaluate(() => window.__store.getState().setToggle("labels", false));
  const vp = page.viewportSize()!;
  const cx = vp.width / 2, cy = vp.height / 2;
  await page.mouse.move(cx, cy);
  await page.mouse.down({ button: "right" });
  for (let i = 1; i <= 25; i++) await page.mouse.move(cx, cy + i * 12); // toward the horizon
  const tilted = await probe(page);
  expect(tilted.el).toBeGreaterThan(1.4);
  for (let i = 1; i <= 40; i++) {
    await page.mouse.move(cx - i * 15, cy + 300); // and a full turn around
    if (i % 10 === 0) {
      const s = await probe(page);
      expect(s.behindNear, `behind the near plane at step ${i}`).toBe(0);
      expect(s.pastFar).toBe(0);
      expect(s.onScreen, `on screen at step ${i}`).toBe(s.n);
      // the view-center pivot holds the screen center it was grabbed at
      expect(s.pivot).not.toBeNull();
      expect(Math.abs(s.pivot![0] - cx)).toBeLessThan(2);
      expect(Math.abs(s.pivot![1] - cy)).toBeLessThan(2);
    }
  }
  await page.mouse.up({ button: "right" });
  expect(errors).toEqual([]);
});
