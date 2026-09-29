/** NebulAI's three experience entries (PRODUCT-EXPERIENCES.md): the root
 *  chooser, the nested Learn/Atlas/Research documents, legacy-link
 *  forwarding, context correction with its notice, and the Other experiences
 *  / Help menus by keyboard. Chrome only, so one rung is enough. */
import { expect, test, type Page } from "@playwright/test";
import { rungOf } from "../helpers";

test.beforeEach(({}, testInfo) => {
  test.skip(rungOf(testInfo) === "webgpu", "chrome is identical on both rungs");
});

/** Collect page errors and every data request from the first byte. */
function watch(page: Page): { errors: string[]; dataRequests: string[] } {
  const errors: string[] = [];
  const dataRequests: string[] = [];
  page.on("pageerror", (err) => errors.push(String(err)));
  page.on("request", (req) => {
    if (new URL(req.url()).pathname.startsWith("/out/")) dataRequests.push(req.url());
  });
  return { errors, dataRequests };
}

async function booted(page: Page): Promise<void> {
  await page.waitForFunction(() => !!window.__store?.getState().experience, undefined, {
    timeout: 45_000,
  });
}

test("root shows the chooser and fetches no model data", async ({ page }) => {
  const { errors, dataRequests } = watch(page);
  await page.goto("/");
  await expect(page.locator(".chooser-card-link")).toHaveText([
    "Learn how maps work",
    "Explore a model",
    "Inspect research evidence",
  ]);
  for (const [name, path] of [
    ["Learn how maps work", "/learn/"],
    ["Explore a model", "/atlas/"],
    ["Inspect research evidence", "/research/"],
  ]) {
    const href = await page.getByRole("link", { name }).evaluate((a) => (a as HTMLAnchorElement).href);
    expect(new URL(href).pathname).toBe(path);
  }
  await expect(page.locator('[data-tool="seer"]')).toBeVisible();
  await page.waitForLoadState("networkidle");
  expect(dataRequests).toEqual([]);
  expect(errors).toEqual([]);
});

test("root forwards legacy links to the experience that hosts them", async ({ page }) => {
  await page.goto("/#page=interp&feature=fourier-atlas&model=gpt2");
  await page.waitForURL(/\/research\/#/);
  // the hash survives the hop (the app then mirrors its own state into it)
  expect(new URL(page.url()).hash).toContain("page=interp");
  await booted(page);
  expect(await page.evaluate(() => window.__store.getState().experience)).toBe("research");

  await page.goto("/?view=chord#model=EleutherAI__pythia-70m");
  await page.waitForURL(/\/research\/\?view=chord#/);

  await page.goto("/#model=EleutherAI__pythia-70m");
  await page.waitForURL(/\/atlas\/#model=/);

  await page.goto("/#page=guide");
  await page.waitForURL(/\/learn\/#page=guide/);
});

test("each nested entry names its experience and shows its own navigation", async ({ page }) => {
  const { errors } = watch(page);
  const cases: [string, string, string[], string][] = [
    ["/learn/", "Learn", ["Lessons"], "guide"],
    ["/atlas/", "Atlas", ["Explore"], "map"],
    ["/research/", "Research", ["Internals", "Comparisons", "Behavior study", "Methods"], "interp"],
  ];
  for (const [path, label, pills, pageId] of cases) {
    await page.goto(`${path}?gpu=webgl&frozen=1`);
    await booted(page);
    await expect(page.locator(".topbar-exp")).toHaveText(label);
    expect(await page.locator(".xnav > .topnav-pill").allInnerTexts()).toEqual(pills);
    expect(await page.evaluate(() => window.__store.getState().page)).toBe(pageId);
    await expect(page).toHaveTitle(new RegExp(`Nebul\\.AI ${label}`));
    // the wordmark goes home to the chooser
    const home = await page.locator(".topbar-home").evaluate((a) => (a as HTMLAnchorElement).href);
    expect(new URL(home).pathname).toBe("/");
  }
  expect(errors).toEqual([]);
});

test("Research never opens the starter map implicitly", async ({ page }) => {
  const { dataRequests } = watch(page);
  await page.goto("/research/?gpu=webgl&frozen=1#page=map");
  await booted(page);
  await expect(page.locator(".map-chooser")).toBeVisible();
  await page.waitForTimeout(500);
  expect(dataRequests.filter((u) => u.endsWith("/nebulai.json"))).toEqual([]);
  expect(await page.evaluate(() => window.__store.getState().datasetId)).toBeNull();
});

test("a link that its entry cannot host is corrected in place, with a notice", async ({ page }) => {
  await page.goto("/atlas/?gpu=webgl&frozen=1#page=interp&feature=fourier-atlas&model=gpt2");
  await booted(page);
  expect(new URL(page.url()).pathname).toBe("/research/");
  expect(await page.evaluate(() => window.__store.getState().experience)).toBe("research");
  const notice = page.locator(".xnav-notice");
  await expect(notice).toHaveText(/Opened in Research for this analysis\./);
  await expect(notice).toHaveAttribute("role", "status");
  await notice.getByRole("button", { name: "Dismiss notice" }).click();
  await expect(notice).toHaveCount(0);

  // an explicit experience that can host the content is kept, silently
  await page.goto("/atlas/?gpu=webgl&frozen=1#experience=research&page=map&view=atlas");
  await booted(page);
  expect(new URL(page.url()).pathname).toBe("/research/");
  await expect(page.locator(".xnav-notice")).toHaveCount(0);
});

test("Other experiences works from the keyboard and announces its state", async ({ page }) => {
  await page.goto("/learn/?gpu=webgl&frozen=1");
  await booted(page);
  const trigger = page.getByRole("button", { name: "Other experiences" });
  await trigger.focus();
  await page.keyboard.press("Enter");
  await expect(trigger).toHaveAttribute("aria-expanded", "true");
  const panel = page.locator(`#${await trigger.getAttribute("aria-controls")}`);
  await expect(panel).toBeVisible();
  await expect(panel.locator("a").first()).toContainText("NebulAI Atlas");
  await expect(panel.locator("a").nth(1)).toContainText("NebulAI Research");

  await page.keyboard.press("ArrowDown");
  await expect(panel.locator("a").first()).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(trigger).toHaveAttribute("aria-expanded", "false");
  await expect(trigger).toBeFocused();
  await expect(panel).toBeHidden();

  // a click outside closes it too
  await trigger.click();
  await expect(panel).toBeVisible();
  await page.mouse.click(5, 400);
  await expect(panel).toBeHidden();
});

test("Help in Atlas opens Learn with a way back", async ({ page }) => {
  await page.goto("/atlas/?gpu=webgl&frozen=1#model=EleutherAI__pythia-70m");
  await booted(page);
  await page.getByRole("button", { name: "Help" }).click();
  const toLearn = page.getByRole("link", { name: /Open the lessons in Learn/ });
  expect(new URL(await toLearn.evaluate((a) => (a as HTMLAnchorElement).href)).hash).toBe("#return=atlas");
  await toLearn.click();
  await page.waitForURL(/\/learn\//);
  await booted(page);
  const back = page.getByRole("link", { name: /Back to Atlas/ });
  await expect(back).toBeVisible();
  // the remembered view, model and all — not a bare entry
  expect(await back.evaluate((a) => (a as HTMLAnchorElement).href)).toMatch(/\/atlas\/.*model=EleutherAI__pythia-70m/);
});

test("Atlas hands advanced map views to Research explicitly", async ({ page }) => {
  await page.goto("/atlas/?gpu=webgl&frozen=1#model=EleutherAI__pythia-70m");
  await page.waitForFunction(() => window.__store.getState().dataset !== null, undefined, { timeout: 45_000 });
  const link = page.locator(".sidebar-handoff-link");
  await expect(link).toBeVisible();
  const href = new URL(await link.evaluate((a) => (a as HTMLAnchorElement).href));
  expect(href.pathname).toBe("/research/");
  expect(href.hash).toBe("#page=map&model=EleutherAI__pythia-70m");
  // there is no view-type select in Atlas to reach chord/hierarchy/compare
  await expect(page.locator(".sidebar select", { has: page.locator('option[value="chord"]') })).toHaveCount(0);
});

test("phone width: no horizontal scroll on any entry", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  for (const path of ["/", "/learn/", "/atlas/", "/research/"]) {
    await page.goto(`${path}?gpu=webgl&frozen=1`);
    if (path !== "/") await booted(page);
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow, path).toBeLessThanOrEqual(0);
  }
});
