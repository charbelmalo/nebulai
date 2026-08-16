/** The shell must outlive the atlas. Nebulai's Internals and Guide pages read
 *  no dataset, and a fresh checkout has nothing baked in `out/` yet, so a
 *  missing dataset index has to degrade into a status line — never into a
 *  blank page. This spec fails the boot gate back into existence the moment
 *  someone re-adds it.
 *
 *  It cannot use `bootApp` from ./helpers: that waits on
 *  `window.__store.getState().dataset !== null`, which is exactly the condition
 *  this scenario never satisfies. Everything else follows the same shape — page
 *  errors collected from the first request, all state waits through
 *  `window.__store`, never through timing guesses. Deliberately no screenshot
 *  golden: there is no blessed image of the empty state, and minting one here
 *  would bake in whatever it happens to look like today.
 *
 *  WHAT CHANGED WHEN SEER GOT ITS OWN ENTRY: this file used to prove the point
 *  by clicking the "Seer" nav pill and asserting `store.page === "seer"`. Seer
 *  is not a page of this document any more — it is a different document
 *  (seer.html), and `setPage("seer")` on Nebulai is now a deliberate no-op. So
 *  the second test became the one that matters at this seam: the cross-
 *  instrument link is present and points at the other entry. Seer's own
 *  atlas-free boot is proved where it actually lives, in seer-entry.spec.ts. */
import { expect, test, type Page } from "@playwright/test";
import { rungOf } from "../helpers";

test.beforeEach(({}, testInfo) => {
  test.skip(rungOf(testInfo) === "webgpu", "chrome is identical on both rungs");
});

const NO_DATASETS = "no datasets in out/index.json";

/** Navigate with the dataset index 404'd out from under the app, and wait until
 *  boot has run to completion. DATA_BASE is derived from BASE_URL
 *  (src/data/base.ts), so in dev the request is
 *  `http://localhost:5173/out/index.json` — this glob matches it on any base,
 *  and the assertions below prove the interception actually landed rather than
 *  trusting the pattern. */
async function bootWithoutAtlas(page: Page): Promise<string[]> {
  const errors: string[] = [];
  page.on("pageerror", (err) => errors.push(String(err)));

  await page.route("**/out/index.json", (r) => r.fulfill({ status: 404, body: "" }));
  await page.goto("/?gpu=webgl&frozen=1");

  // `__perf.bootMs` is set only on the atlas path, so the completion signal
  // here is the status pill reporting the absence — that line is the last thing
  // bootAtlas does before returning to boot()
  await expect(page.locator(".boot-status")).toContainText(NO_DATASETS, { timeout: 45_000 });
  return errors;
}

test("no dataset index: the chrome mounts anyway and reports the absence", async ({ page }) => {
  const errors = await bootWithoutAtlas(page);

  // the route really intercepted — asserted through the app's own conclusion
  expect(await page.evaluate(() => window.__store.getState().datasets.length)).toBe(0);
  expect(await page.evaluate(() => window.__store.getState().dataset)).toBeNull();

  // …and the shell is up regardless: this is the whole point of the two-phase boot
  await expect(page.locator(".chrome-root")).toHaveCount(1);
  await expect(page.locator("nav.topnav")).toBeVisible();

  // this document is Nebulai, and its nav is Nebulai's three pages only
  expect(await page.locator(".topnav-pill").allInnerTexts()).toEqual([
    "Semantic map",
    "Internals",
    "Guide",
  ]);
  expect(await page.evaluate(() => window.__store.getState().app)).toBe("nebulai");

  // the page is degraded, not dead — neither failure handler fired
  const statusText = (await page.locator(".boot-status").textContent()) ?? "";
  expect(statusText).not.toContain("boot failed:");
  expect(statusText).not.toContain("atlas failed:");
  expect(errors).toEqual([]);
});

test("no dataset index: Seer is still reachable, as the other instrument", async ({ page }) => {
  await bootWithoutAtlas(page);

  // Seer is one subordinate link, not a fourth pill. Both halves are the
  // assertion: it must NOT be a pill (that is the segmentation), and it must
  // be present and correct (that is discoverability before the psychiX hub).
  await expect(page.locator(".topnav-pill", { hasText: "Seer" })).toHaveCount(0);

  const cross = page.locator(".topnav-cross");
  await expect(cross).toBeVisible();
  await expect(cross).toHaveAttribute("href", "./seer.html");
  await expect(cross).toContainText("Seer");
  // resolved against this document, it lands on the sibling entry rather than
  // anywhere else on the origin — a relative href is only as good as its base
  expect(await cross.evaluate((a) => (a as HTMLAnchorElement).href)).toBe(
    new URL("/seer.html", page.url()).href,
  );

  // and Nebulai cannot be talked into rendering Seer's pages from the inside
  await page.evaluate(() => window.__store.getState().setPage("seer"));
  expect(await page.evaluate(() => window.__store.getState().page)).toBe("map");
  await expect(page.locator(".seer-page")).toHaveCount(0);

  const statusText = (await page.locator(".boot-status").textContent()) ?? "";
  expect(statusText).not.toContain("boot failed:");
});

test("mobile Guide clears the two-row chrome and documents every live view", async ({
  page,
}) => {
  await page.setViewportSize({ width: 375, height: 812 });
  const errors = await bootWithoutAtlas(page);

  await page.locator(".topnav-pill", { hasText: "Guide" }).click();
  await expect(page.locator(".guide-page")).toBeVisible();
  await expect(page.locator(".guide-count")).toContainText("25 of 25");
  await expect(page.locator(".guide-card")).toHaveCount(25);
  await expect(page.locator(".guide-card-research li")).toHaveCount(75);

  const links = page.locator(".guide-card-research a");
  await expect(links).toHaveCount(75);
  for (const link of await links.all()) {
    await expect(link).toHaveAttribute("href", /^https:\/\//);
    await expect(link).toHaveAttribute("target", "_blank");
    await expect(link).toHaveAttribute("rel", "noreferrer");
  }

  const clearance = await page.evaluate(() => {
    const nav = document.querySelector(".topnav")!.getBoundingClientRect();
    const guide = document.querySelector(".guide-page")!.getBoundingClientRect();
    return { navBottom: nav.bottom, pageTop: guide.top };
  });
  expect(clearance.pageTop).toBeGreaterThanOrEqual(clearance.navBottom);

  await page.locator(".guide-card-open").first().click();
  expect(await page.evaluate(() => window.__store.getState().page)).toBe("interp");
  await expect(page.locator(".interp-page")).toBeVisible();

  const interpClearance = await page.evaluate(() => {
    const nav = document.querySelector(".topnav")!.getBoundingClientRect();
    const interp = document.querySelector(".interp-page")!.getBoundingClientRect();
    return { navBottom: nav.bottom, pageTop: interp.top };
  });
  expect(interpClearance.pageTop).toBeGreaterThanOrEqual(interpClearance.navBottom);
  expect(errors).toEqual([]);
});
