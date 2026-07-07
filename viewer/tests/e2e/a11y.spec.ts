/** Accessibility gates: axe scan on the full chrome, keyboard reachability,
 *  Escape-deselect from anywhere, and reduced-motion crossfade shortening.
 *  Chrome is identical on both rungs, so everything runs webgl-only. */
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { bootApp, rungOf } from "./helpers";

test.beforeEach(({}, testInfo) => {
  test.skip(rungOf(testInfo) === "webgpu", "chrome is identical on both rungs");
});

test("axe: no serious or critical violations", async ({ page }) => {
  await bootApp(page, "webgl");
  const results = await new AxeBuilder({ page }).analyze();
  const severe = results.violations
    .filter((v) => v.impact === "serious" || v.impact === "critical")
    .map((v) => `${v.id} (${v.impact}): ${v.help} — ${v.nodes.length} node(s)`);
  expect(severe).toEqual([]);
});

test("keyboard reaches the sidebar selects and legend radios", async ({ page }) => {
  await bootApp(page, "webgl");
  const seq: string[] = [];
  for (let i = 0; i < 40; i++) {
    await page.keyboard.press("Tab");
    seq.push(
      await page.evaluate(() => {
        const el = document.activeElement;
        if (!el || el === document.body) return "body";
        const id = el.id ? `#${el.id}` : "";
        const type = el instanceof HTMLInputElement ? `[${el.type}]` : "";
        return `${el.tagName.toLowerCase()}${type}${id}`;
      }),
    );
  }
  expect(seq).toContain("select#sel-dataset"); // sidebar
  expect(seq.some((s) => s.startsWith("input[radio]"))).toBe(true); // legend
});

test("Escape clears the selection from anywhere", async ({ page }) => {
  await bootApp(page, "webgl");
  await page.evaluate(() => window.__store.getState().setSelection({ kind: "cluster", id: 0 }));
  expect(await page.evaluate(() => window.__store.getState().selection)).not.toBeNull();
  await page.keyboard.press("Escape");
  expect(await page.evaluate(() => window.__store.getState().selection)).toBeNull();
});

test("prefers-reduced-motion shortens crossfades to 150ms", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await bootApp(page, "webgl");
  const transition = await page.evaluate(
    () => document.getElementById("scene-canvas")!.style.transition,
  );
  expect(transition).toContain("150ms");
});
