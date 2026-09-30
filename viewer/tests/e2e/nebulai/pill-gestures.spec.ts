/** Cluster label pills are HTML buttons over the canvas. A drag, right-drag
 *  orbit or wheel that starts on one must still drive the map — before this
 *  they swallowed the press and the map froze under the cursor. A press that
 *  never moves is still the pill's click, and keyboard activation still
 *  works. Pure input routing, so one rung is enough. */
import { expect, test, type Page } from "@playwright/test";
import { rungOf } from "../helpers";
import { bootApp, waitForSettle } from "./helpers";

const STARTER = "gpt2-small__sae__blocks.8.hook_resid_pre";

test.beforeEach(({}, testInfo) => {
  test.skip(rungOf(testInfo) === "webgpu", "input routing is identical on both rungs");
});

interface Cam {
  cx: number;
  cy: number;
  wpp: number;
  az: number;
  el: number;
}

function cam(page: Page): Promise<Cam> {
  return page.evaluate(() => {
    const d = window.__driver as unknown as {
      cam: { cx: number; cy: number; wpp: number };
      orbitAngles(): [number, number];
    };
    const [az, el] = d.orbitAngles();
    return { cx: d.cam.cx, cy: d.cam.cy, wpp: d.cam.wpp, az, el };
  });
}

/** Center of the biggest visible pill that sits well inside the viewport. */
async function pillCenter(page: Page): Promise<{ x: number; y: number; text: string }> {
  type Hit = { x: number; y: number; text: string; a: number } | null;
  // pills fade out across the 2-D ↔ 3-D morph, so wait for them to come back
  let hit: Hit = null;
  await expect
    .poll(async () => (hit = await findPill(page)) !== null, {
      message: "a visible pill near the viewport center",
      timeout: 10_000,
    })
    .toBe(true);
  return hit!;
}

function findPill(page: Page) {
  return page.evaluate(() => {
    const vw = innerWidth, vh = innerHeight;
    let best: { x: number; y: number; text: string; a: number } | null = null;
    for (const el of document.querySelectorAll<HTMLElement>(".cluster-pill")) {
      if (getComputedStyle(el).visibility !== "visible") continue;
      const r = el.getBoundingClientRect();
      const x = r.left + r.width / 2, y = r.top + r.height / 2;
      if (x < vw * 0.3 || x > vw * 0.7 || y < vh * 0.3 || y > vh * 0.7) continue;
      // the pill must be what the pointer actually lands on
      if (document.elementFromPoint(x, y) !== el) continue;
      const a = r.width * r.height;
      if (!best || a > best.a) best = { x, y, text: el.textContent ?? "", a };
    }
    return best;
  });
}

test.describe("label pills pass gestures through to the map", () => {
  test("a left-drag that starts on a pill pans the map", async ({ page }, testInfo) => {
    const { errors } = await bootApp(page, rungOf(testInfo), { hash: `model=${STARTER}`, frozen: false });
    await waitForSettle(page);
    const p = await pillCenter(page);
    const before = await cam(page);
    await page.mouse.move(p.x, p.y);
    await page.mouse.down();
    for (let i = 1; i <= 12; i++) await page.mouse.move(p.x + i * 15, p.y + i * 8);
    await page.mouse.up();
    const after = await cam(page);
    // dragged right/down → the view center moves left/up in world space
    const moved = Math.hypot(after.cx - before.cx, after.cy - before.cy) / before.wpp;
    expect(moved, "pan distance in px").toBeGreaterThan(100);
    // a drag is not a click — no selection from the pill it started on
    expect(await page.evaluate(() => window.__store.getState().selection)).toBeNull();
    expect(errors).toEqual([]);
  });

  test("a right-drag orbit that starts on a pill orbits about that cluster", async ({ page }, testInfo) => {
    const { errors } = await bootApp(page, rungOf(testInfo), { hash: `model=${STARTER}`, frozen: false });
    await page.evaluate(() => window.__store.getState().setDims(3));
    await waitForSettle(page);
    const p = await pillCenter(page);
    const before = await cam(page);
    await page.mouse.move(p.x, p.y);
    await page.mouse.down({ button: "right" });
    for (let i = 1; i <= 15; i++) await page.mouse.move(p.x - i * 12, p.y + i * 4);
    const pivot = await page.evaluate(() => (window.__driver as unknown as { orbitPivot: unknown }).orbitPivot);
    await page.mouse.up({ button: "right" });
    const after = await cam(page);
    expect(Math.abs(after.az - before.az) + Math.abs(after.el - before.el)).toBeGreaterThan(0.2);
    expect(pivot, "an orbit pivot was grabbed").not.toBeNull();
    expect(errors).toEqual([]);
  });

  test("the wheel over a pill zooms the map", async ({ page }, testInfo) => {
    const { errors } = await bootApp(page, rungOf(testInfo), { hash: `model=${STARTER}`, frozen: false });
    await waitForSettle(page);
    const p = await pillCenter(page);
    const before = await cam(page);
    await page.mouse.move(p.x, p.y);
    for (let i = 0; i < 5; i++) await page.mouse.wheel(0, -120);
    await expect.poll(async () => (await cam(page)).wpp).toBeLessThan(before.wpp * 0.8);
    expect(errors).toEqual([]);
  });

  test("a press that never moves still selects the pill's cluster", async ({ page }, testInfo) => {
    const { errors } = await bootApp(page, rungOf(testInfo), { hash: `model=${STARTER}`, frozen: false });
    await waitForSettle(page);
    const p = await pillCenter(page);
    await page.mouse.click(p.x, p.y);
    const sel = await page.evaluate(() => window.__store.getState().selection);
    expect(sel?.kind).toBe("cluster");
    expect(errors).toEqual([]);
  });

  test("keyboard Enter on a focused pill selects its cluster once", async ({ page }, testInfo) => {
    const { errors } = await bootApp(page, rungOf(testInfo), { hash: `model=${STARTER}`, frozen: false });
    await waitForSettle(page);
    const p = await pillCenter(page);
    await page.evaluate(({ x, y }) => (document.elementFromPoint(x, y) as HTMLElement).focus(), p);
    let sets = 0;
    await page.exposeFunction("__countSel", () => sets++);
    await page.evaluate(() => {
      const s = window.__store;
      s.subscribe((a, b) => {
        if (a.selection !== b.selection) (window as unknown as { __countSel(): void }).__countSel();
      });
    });
    await page.keyboard.press("Enter");
    await expect.poll(() => page.evaluate(() => window.__store.getState().selection?.kind)).toBe("cluster");
    expect(sets).toBe(1);
    expect(errors).toEqual([]);
  });
});
