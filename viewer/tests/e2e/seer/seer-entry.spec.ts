/** Seer's own entry: /seer.html booting under its actual production
 *  condition, which is a deploy with NO baked artifacts at all.
 *
 *  boot-without-atlas.spec.ts proves the weaker, older property — Nebulai's
 *  shell survives a 404 on the dataset index. This proves the stronger one the
 *  second entry exists for: Seer never asks. Nothing is intercepted here and
 *  no route is faked; the test records every request the document makes and
 *  asserts that none of them went at `out/`. A shared import that quietly
 *  dragged the loader (or, through it, the AtlasDriver) back into Seer's graph
 *  would show up as a real request in that log — which is the only check that
 *  keeps working after a refactor renames everything.
 *
 *  It cannot use Nebulai's `bootApp` (tests/e2e/nebulai/helpers.ts) for the
 *  same reason boot-without-atlas.spec.ts cannot: that helper waits on
 *  `window.__store.getState().dataset !== null`, a condition this document
 *  will never satisfy because it has no code that could set it — which is
 *  also why bootApp lives under nebulai/ rather than in the shared
 *  tests/e2e/helpers.ts. The local helper below follows the pattern that
 *  file established — errors collected from before the first request,
 *  completion signalled by state the app itself writes, never by a sleep.
 *
 *  Console errors are deliberately NOT collected: the Live page tries to reach
 *  a collector on :8125 that no CI machine is running, and a refused
 *  connection is the correct behaviour, reported in the page's own link
 *  status. Uncaught page errors are a different matter and are still fatal. */
import { expect, test, type Page } from "@playwright/test";
import { rungOf } from "../helpers";

test.beforeEach(({}, testInfo) => {
  test.skip(rungOf(testInfo) === "webgpu", "chrome is identical on both rungs");
});

interface SeerBoot {
  /** uncaught exceptions since navigation */
  errors: string[];
  /** every URL the document requested, in order */
  requests: string[];
}

async function bootSeer(page: Page): Promise<SeerBoot> {
  const errors: string[] = [];
  const requests: string[] = [];
  page.on("pageerror", (err) => errors.push(String(err)));
  page.on("request", (r) => requests.push(r.url()));

  await page.goto("/seer.html?gpu=webgl&frozen=1");

  // the completion signal is the store's own identity write, which
  // `bootShell` performs before it mounts anything
  await page.waitForFunction(() => window.__store?.getState().app === "seer", undefined, {
    timeout: 45_000,
  });
  await expect(page.locator(".chrome-root")).toHaveCount(1);
  return { errors, requests };
}

test("seer.html boots the shared shell with no atlas artifacts", async ({ page }) => {
  const { errors, requests } = await bootSeer(page);

  // ── the actual claim: this document never reaches for the atlas ──────────
  // `out/` is where every baked artifact lives (src/data/base.ts derives
  // DATA_BASE from BASE_URL and appends it), so one substring covers the
  // index, the column blobs and compare.json alike. Non-http schemes (blob:
  // worker URLs, data: icons) are dropped rather than parsed as paths.
  const atlasRequests = requests.filter(
    (u) => /^https?:/.test(u) && new URL(u).pathname.includes("/out/"),
  );
  expect(atlasRequests).toEqual([]);

  // …and not because the atlas code is present but idle: none of it is loaded
  expect(await page.evaluate(() => window.__driver)).toBeUndefined();
  expect(await page.evaluate(() => window.__compareDriver)).toBeUndefined();
  expect(await page.evaluate(() => window.__store.getState().datasets)).toEqual([]);
  expect(await page.evaluate(() => window.__store.getState().dataset)).toBeNull();
  // index.html's layered driver stage does not exist in this document at all
  expect(await page.locator("#stage").count()).toBe(0);

  // ── it is Seer, and it says so ───────────────────────────────────────────
  expect(await page.title()).toBe("Seer — agent runs");
  await expect(page.locator(".topbar-word")).not.toContainText("nebul");
  expect(await page.evaluate(() => window.__store.getState().page)).toBe("seer");
  await expect(page.locator(".seer-page")).toBeVisible();

  // the status pill reports the rung and where the collector would be — it
  // must not invent dataset provenance it does not have
  const statusText = (await page.locator(".boot-status").textContent()) ?? "";
  expect(statusText).toContain("gpu: webgl");
  expect(statusText).toContain("collector:");
  expect(statusText).not.toContain("boot failed:");

  expect(errors).toEqual([]);
});

test("seer.html navigates its own three pages and no others", async ({ page }) => {
  await bootSeer(page);

  expect(await page.locator(".topnav-pill").allInnerTexts()).toEqual([
    "Live",
    "Transcripts",
    "Topics",
  ]);

  // the renames are the user-visible half of the segmentation: the labels
  // changed, the page ids (wire format: permalinks, store, goldens) did not
  await page.locator(".topnav-pill", { hasText: "Transcripts" }).click();
  expect(await page.evaluate(() => window.__store.getState().page)).toBe("sessions");
  await expect(page.locator(".sessions-page")).toBeVisible();

  await page.locator(".topnav-pill", { hasText: "Topics" }).click();
  expect(await page.evaluate(() => window.__store.getState().page)).toBe("snapshot");
  await expect(page.locator(".snapshot-page")).toBeVisible();

  await page.locator(".topnav-pill", { hasText: "Live" }).click();
  expect(await page.evaluate(() => window.__store.getState().page)).toBe("seer");

  // Nebulai's pages are not merely unlinked here — they are unreachable
  await page.evaluate(() => window.__store.getState().setPage("map"));
  expect(await page.evaluate(() => window.__store.getState().page)).toBe("seer");
  await page.evaluate(() => window.__store.getState().setPage("interp"));
  expect(await page.evaluate(() => window.__store.getState().page)).toBe("seer");
});

test("seer.html links back to Nebulai as the other instrument", async ({ page }) => {
  await bootSeer(page);

  const cross = page.locator(".topnav-cross");
  await expect(cross).toBeVisible();
  await expect(cross).toHaveAttribute("href", "./index.html");
  expect(await cross.evaluate((a) => (a as HTMLAnchorElement).href)).toBe(
    new URL("/index.html", page.url()).href,
  );

  // it really is a way out of this document, not a nav pill in disguise
  await cross.click();
  await page.waitForFunction(() => window.__store?.getState().app === "nebulai", undefined, {
    timeout: 45_000,
  });
  expect(await page.evaluate(() => window.__store.getState().page)).toBe("map");
});

test("an Internals permalink opened on seer.html is refused, not half-applied", async ({
  page,
}) => {
  // Nebulai's own hash, pasted into Seer's document. Its page and its
  // Internals-only keys all name things this bundle does not contain, so the
  // honest outcome is that none of them are applied — and that Seer still
  // boots onto its own first page rather than a blank one.
  const errors: string[] = [];
  page.on("pageerror", (err) => errors.push(String(err)));
  await page.goto("/seer.html?gpu=webgl&frozen=1#page=interp&feature=fourier-atlas&model=gpt2");
  await page.waitForFunction(() => window.__store?.getState().app === "seer", undefined, {
    timeout: 45_000,
  });

  expect(await page.evaluate(() => window.__store.getState().page)).toBe("seer");
  await expect(page.locator(".interp-page")).toHaveCount(0);
  await expect(page.locator(".seer-page")).toBeVisible();
  expect(errors).toEqual([]);
});
