/** NebulAI Learn's introductory lesson, "What is a point?" (N04L): five
 *  steps on the published starter map, completed only by what the reader
 *  does — the unit they choose and the answers they give. Covers keyboard
 *  completion, wrong choices not counting, reload and deep-link restore, the
 *  no-GPU path, a missing release manifest (recovery, never completion),
 *  390px, and existing episodes still restoring. Chrome only. */
import { expect, test, type Page } from "@playwright/test";
import { rungOf } from "../helpers";

test.beforeEach(({}, testInfo) => {
  test.skip(rungOf(testInfo) === "webgpu", "chrome is identical on both rungs");
});

const LESSON = "#lesson=what-is-a-point&lesson_step=";

function watch(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (err) => errors.push(String(err)));
  return errors;
}

async function openLesson(page: Page, step: number, gpu = "webgl"): Promise<void> {
  await page.goto(`/learn/?gpu=${gpu}&frozen=1${LESSON}${step}`);
  await expect(page.locator("#lesson-step-title")).toBeVisible({ timeout: 45_000 });
}

const title = (page: Page) => page.locator("#lesson-step-title");
const next = (page: Page) => page.getByRole("button", { name: "Next step" });
const stepOf = (page: Page) => page.evaluate(() => window.__store.getState().tour?.step ?? null);

async function chooseNumbersUnit(page: Page): Promise<void> {
  await page.locator("#rt-search").fill("number");
  const first = page.locator(".lesson-panel .rt-inspect").first();
  await expect(first).toHaveText(/number/i);
  await first.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator(".lesson-verdict.is-right")).toContainText("Found.");
}

async function answer(page: Page, text: RegExp | string): Promise<void> {
  const radio = page.getByRole("radio", { name: text });
  await radio.focus();
  await page.keyboard.press("Space");
  await expect(radio).toBeChecked();
}

test("the lesson completes by keyboard, and only on the reader's own choices", async ({ page }) => {
  const errors = watch(page);
  await openLesson(page, 0);
  await expect(title(page)).toContainText("A map of learned directions");
  await expect(title(page)).toBeFocused();
  // step 1 names the exact file it is about
  await expect(page.locator(".lesson-panel .aw-evidence")).toBeVisible();
  await next(page).focus();
  await page.keyboard.press("Enter");

  // step 2: nothing chosen → Next is disabled with its reason
  await expect(title(page)).toContainText("Find a direction about numbers");
  await expect(title(page)).toBeFocused();
  await expect(next(page)).toBeDisabled();
  await expect(page.locator("#lesson-next-reason")).toContainText("about numbers");
  // a unit whose label is not about numbers does not count
  await page.locator("#rt-search").fill("piers");
  await page.locator(".lesson-panel .rt-inspect").first().click();
  await expect(page.locator(".lesson-verdict.is-wrong")).toContainText("Not this one.");
  await expect(next(page)).toBeDisabled();
  await chooseNumbersUnit(page);
  // the same inspector as Atlas, without its close control or save actions
  await expect(page.locator(".inspector")).toBeVisible();
  await expect(page.locator(".inspector .inspector-close")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Save finding…" })).toHaveCount(0);
  await next(page).focus();
  await page.keyboard.press("Enter");

  // step 3: a wrong answer explains itself and does not complete the step
  await expect(title(page)).toContainText("What identifies this unit?");
  await answer(page, "Its label");
  await expect(page.locator(".lesson-verdict.is-wrong")).toContainText("Not quite.");
  await expect(next(page)).toBeDisabled();
  await answer(page, /Its index in this SAE/);
  await expect(page.locator(".lesson-verdict.is-right")).toContainText("sha256 digest");
  await next(page).click();

  // step 4: the meaning check
  await expect(title(page)).toContainText("What does its place on the map mean?");
  await answer(page, /strongly active on text I entered/);
  await expect(next(page)).toBeDisabled();
  await answer(page, /decoder vector points/);
  await next(page).click();

  // step 5: explicit finish, then the save actions and Continue in Atlas
  await expect(title(page)).toContainText("Keep it, or keep exploring");
  await expect(page.getByRole("button", { name: "Save finding…" })).toBeVisible();
  const atlas = page.getByRole("link", { name: "Continue in Atlas" });
  const href = await atlas.getAttribute("href");
  expect(href).toContain("/atlas/#experience=atlas&page=map");
  expect(href).toContain("artifact=");
  expect(href).not.toContain("lesson");
  await page.getByRole("button", { name: "Finish lesson" }).click();
  await expect(page.locator(".lesson-complete")).toContainText("Lesson complete.");
  await expect(page.locator(".lesson-complete")).toBeFocused();
  expect(await page.evaluate(() => localStorage.getItem("nebulai.lessons.completed"))).toContain(
    "what-is-a-point",
  );
  // answers stay in the tab: the address carries the step, nothing else
  const hash = await page.evaluate(() => location.hash);
  expect(hash).toContain("lesson=what-is-a-point&lesson_step=4");
  expect(hash).not.toMatch(/answer|index|geometry/);

  // back to the catalog: the card says Completed and holds focus
  await page.getByRole("button", { name: "Back to lessons" }).click();
  await expect(page.locator(".lesson-card-kicker")).toContainText("Completed");
  await expect(page.getByRole("button", { name: "Take the lesson again" })).toBeFocused();
  expect(errors).toEqual([]);
});

test("a reload restores the step and the reader's answers; a deep link does not invent them", async ({
  page,
}) => {
  await openLesson(page, 1);
  await chooseNumbersUnit(page);
  await next(page).click();
  await answer(page, /Its index in this SAE/);
  await page.reload();
  await expect(title(page)).toContainText("What identifies this unit?", { timeout: 45_000 });
  await expect(page.getByRole("radio", { name: /Its index in this SAE/ })).toBeChecked();
  await expect(next(page)).toBeEnabled();
  await expect(page.locator(".inspector-title")).toHaveText("numbers and quantities");

  // a fresh tab opened at step 4 has no chosen unit: it says so, and cannot finish
  const fresh = await page.context().newPage();
  await openLesson(fresh, 4);
  await expect(fresh.locator(".lesson-need")).toContainText("none is chosen yet");
  await expect(fresh.getByRole("button", { name: "Finish lesson" })).toHaveCount(0);
  await fresh.getByRole("button", { name: "Go to step 2" }).click();
  await expect(fresh.locator("#lesson-step-title")).toContainText("Find a direction about numbers");
  await fresh.close();
});

test("without a GPU the lesson is complete from the list alone", async ({ page }) => {
  const errors = watch(page);
  await openLesson(page, 0, "static");
  expect(await page.evaluate(() => window.__store.getState().renderer)).toBe("unavailable");
  await expect(page.locator(".lesson-panel .aw-note")).toContainText("Every step works from the list");
  await next(page).click();
  await chooseNumbersUnit(page);
  await next(page).click();
  await answer(page, /Its index in this SAE/);
  await next(page).click();
  await answer(page, /decoder vector points/);
  await next(page).click();
  await page.getByRole("button", { name: "Finish lesson" }).click();
  await expect(page.locator(".lesson-complete")).toContainText("Lesson complete.");
  expect(errors).toEqual([]);
});

test("a missing release manifest blocks the lesson with recovery, and never completes it", async ({ page }) => {
  await page.route("**/experience.json", (r) => r.fulfill({ status: 404, body: "" }));
  await page.goto(`/learn/?gpu=webgl&frozen=1${LESSON}2`);
  const alert = page.getByRole("alert");
  await expect(alert).toContainText("This lesson's map is not available", { timeout: 45_000 });
  await expect(alert).toContainText("No other map was substituted");
  await expect(page.getByRole("heading", { name: "This lesson's map is not available" })).toBeFocused();
  await expect(page.getByRole("button", { name: "Try again" })).toBeVisible();
  // nothing was opened under the lesson, and no step was marked done
  expect(await page.evaluate(() => window.__store.getState().datasetId)).toBeNull();
  await expect(page.locator(".lesson-complete")).toHaveCount(0);
  expect(await stepOf(page)).toBe(2);
  expect(await page.evaluate(() => window.__store.getState().tour?.blocked)).toBeTruthy();

  // the catalog says the same thing before anyone starts
  await page.getByRole("button", { name: "Back to lessons" }).click();
  await expect(page.getByRole("button", { name: "Start the lesson" })).toBeDisabled();
  await expect(page.locator(".lesson-card .episode-gate")).toContainText("Not available here");
});

test("the catalog starts the lesson, and Exit returns to it", async ({ page }) => {
  await page.goto("/learn/?gpu=webgl&frozen=1");
  const start = page.getByRole("button", { name: /Start the lesson|Take the lesson again/ });
  await expect(start).toBeEnabled({ timeout: 45_000 });
  await start.click();
  await expect(title(page)).toContainText("A map of learned directions", { timeout: 45_000 });
  expect(await page.evaluate(() => location.hash)).toContain("lesson=what-is-a-point&lesson_step=0");
  await page.getByRole("button", { name: "Exit lesson" }).click();
  await expect(page.locator(".lesson-card")).toBeVisible();
  expect(await page.evaluate(() => window.__store.getState().page)).toBe("guide");
});

test("at 390px the lesson and the map share one screen behind a switch", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openLesson(page, 1);
  await chooseNumbersUnit(page);
  // the unit is inline, and the step controls stay in reach
  await expect(page.locator(".lesson-inline-unit .inspector")).toBeVisible();
  await expect(next(page)).toBeInViewport();
  await page.getByRole("button", { name: "Map", exact: true }).click();
  await expect(page.locator(".lesson-panel")).toBeHidden();
  await page.getByRole("button", { name: "Lesson", exact: true }).click();
  await expect(page.locator(".lesson-panel")).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});

test("existing guided episodes still restore their model and step in Learn", async ({ page }) => {
  await page.goto("/learn/?gpu=webgl&frozen=1#episode=induction&step=1");
  await page.waitForFunction(() => window.__store?.getState().tour?.id === "induction", undefined, {
    timeout: 45_000,
  });
  expect(await stepOf(page)).toBe(1);
  expect(await page.evaluate(() => window.__store.getState().experience)).toBe("learn");
  await expect(page.locator(".interp-tourbar")).toBeVisible();
});

test("Play on a catalog episode opens its stage, not the catalog", async ({ page }) => {
  await page.goto("/learn/?gpu=webgl&frozen=1");
  const card = page.locator(".episode-card", { hasText: "The Induction Circuit" });
  await expect(card.getByRole("button", { name: /Play this episode/ })).toBeEnabled({ timeout: 45_000 });
  await card.getByRole("button", { name: /Play this episode/ }).click();
  await expect(page.locator(".interp-tourbar")).toBeVisible();
  expect(await page.evaluate(() => window.__store.getState().page)).toBe("interp");
});
