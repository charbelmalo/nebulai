/** NebulAI Research's entry to Internals (N04R): an explicit model-export
 *  chooser that loads no model map, then the first task — Position Patterns
 *  by Frequency — on the exact bundle the release manifest pins. Covers
 *  keyboard completion, the verbatim export, a missing or altered bundle
 *  (recovery, nothing substituted), the no-GPU path, deep-link restore,
 *  honest per-export availability and the legacy Atlas redirect. Chrome only. */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { expect, test, type Page } from "@playwright/test";
import { rungOf } from "../helpers";

test.beforeEach(({}, testInfo) => {
  test.skip(rungOf(testInfo) === "webgpu", "chrome is identical on both rungs");
});

const TASK = "#page=interp&model=gpt2&feature=fourier-atlas";
const BUNDLE = "**/out/gpt2/interp/fourier.json";

/** every model-map request the page makes (`<id>/nebulai.json`) */
function watch(page: Page): { errors: string[]; maps: string[] } {
  const errors: string[] = [];
  const maps: string[] = [];
  page.on("pageerror", (err) => errors.push(String(err)));
  page.on("request", (r) => {
    if (/\/nebulai(\.[0-9a-f]+)?\.json(\?|$)/.test(r.url())) maps.push(r.url());
  });
  return { errors, maps };
}

async function pinned(page: Page): Promise<{ sha256: string; bytes: number }> {
  return page.evaluate(async () => {
    const m = await (await fetch("/out/experience.json")).json();
    return m.research_intro;
  });
}

const task = (page: Page) => page.locator("aside.research-task");

async function pick(page: Page, name: RegExp): Promise<void> {
  const radio = page.getByRole("radio", { name });
  await radio.focus();
  await page.keyboard.press("Space");
  await expect(radio).toBeChecked();
}

test("Research opens on an export chooser and loads no model map", async ({ page }) => {
  const { errors, maps } = watch(page);
  await page.goto("/research/?gpu=webgl&frozen=1");
  await expect(page.getByRole("heading", { level: 1, name: "Choose a model export" })).toBeVisible({
    timeout: 45_000,
  });
  const gpt2 = page.getByRole("button", { name: /^GPT-2\s*Recommended/ });
  await expect(gpt2).toContainText(/24 of 26 analyses available/);
  await expect(page.getByRole("button", { name: /^GPT-2 Medium\b/ })).toContainText(/18 of 26/);
  await expect(page.locator(".research-chooser-more summary")).toContainText(/map but no internals export/);
  // nothing has been chosen, so the address names no model
  expect(new URL(page.url()).hash).not.toContain("model=");
  await page.waitForTimeout(500);
  expect(maps).toEqual([]);
  expect(errors).toEqual([]);
});

test("the first task completes by keyboard and exports the exact pinned bytes", async ({ page }) => {
  const { errors, maps } = watch(page);
  await page.goto("/research/?gpu=webgl&frozen=1");
  const gpt2 = page.getByRole("button", { name: /^GPT-2\s*Recommended/ });
  await expect(gpt2).toBeVisible({ timeout: 45_000 });
  await gpt2.focus();
  await page.keyboard.press("Enter");

  await expect(task(page)).toContainText("Verified");
  await expect(page.locator(".interp-rail-count")).toHaveText("24 of 26 available");
  await expect(page.locator(".interp-canvas-host")).toBeVisible();
  const hash = new URL(page.url()).hash;
  expect(hash).toContain("model=gpt2");
  expect(hash).toContain("feature=fourier-atlas");

  // 1. a wrong answer is explained and does not count
  await pick(page, /attends to earlier positions/);
  await expect(task(page).locator(".research-feedback.is-wrong")).toContainText("Nothing here was measured on a prompt");
  await pick(page, /stored position embeddings repeat/);
  // 2. the strongest pattern: 2 cycles per window, period 512
  await pick(page, /Record frequency 2$/);
  await expect(task(page)).toContainText("frequency 2 repeats every 512 positions");
  // 3. the caveat
  await pick(page, /uses a given frequency/);
  await expect(page.locator(".research-complete")).toBeEmpty();
  // 4. the download is the published file, byte for byte
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    task(page).getByRole("button", { name: /Download gpt2-fourier\.json/ }).press("Enter"),
  ]);
  expect(download.suggestedFilename()).toBe("gpt2-fourier.json");
  const bytes = readFileSync((await download.path())!);
  const pin = await pinned(page);
  expect(bytes.length).toBe(pin.bytes);
  expect(createHash("sha256").update(bytes).digest("hex")).toBe(pin.sha256);

  await expect(page.locator(".research-complete")).toContainText("Task complete");
  expect(maps).toEqual([]);
  expect(errors).toEqual([]);
});

test("a deep link restores the task and its progress without a model map", async ({ page }) => {
  const { maps } = watch(page);
  await page.goto(`/research/?gpu=webgl&frozen=1${TASK}`);
  await expect(task(page)).toContainText("Verified", { timeout: 45_000 });
  await pick(page, /stored position embeddings repeat/);
  await page.reload();
  await expect(task(page)).toContainText("Verified", { timeout: 45_000 });
  await expect(page.getByRole("radio", { name: /stored position embeddings repeat/ })).toBeChecked();
  expect(await page.evaluate(() => window.__store.getState().datasetId)).toBeNull();
  expect(maps).toEqual([]);

  // All exports returns to the chooser and drops the model from the address
  await page.getByRole("button", { name: "All exports" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Choose a model export" })).toBeVisible();
  await expect.poll(() => new URL(page.url()).hash).not.toContain("model=");
});

test("a missing bundle shows recovery and substitutes nothing", async ({ page }) => {
  const { errors } = watch(page);
  await page.route(BUNDLE, (r) => r.fulfill({ status: 404, body: "not found" }));
  await page.goto(`/research/?gpu=webgl&frozen=1${TASK}`);
  await expect(task(page).getByRole("heading", { level: 2 })).toHaveText("The analysis file is not on this server", {
    timeout: 45_000,
  });
  await expect(task(page).getByRole("alert")).toContainText("the server answered 404");
  await expect(page.locator(".interp-status.is-error")).toContainText("Chart withheld");
  expect(await page.evaluate(() => "__interpDriver" in window)).toBe(false);

  // the file comes back: Try again verifies it and the chart opens
  await page.unroute(BUNDLE);
  await task(page).getByRole("button", { name: "Try again" }).click();
  await expect(task(page)).toContainText("Verified");
  await expect(page.locator(".interp-status.is-error")).toHaveCount(0);
  await expect.poll(() => page.evaluate(() => "__interpDriver" in window)).toBe(true);
  expect(errors).toEqual([]);
});

test("an altered bundle is refused, even though it parses", async ({ page }) => {
  await page.route(BUNDLE, async (r) => {
    const res = await r.fetch();
    const json = await res.json();
    json.power_mean[2] = 1;
    await r.fulfill({ response: res, body: JSON.stringify(json) });
  });
  await page.goto(`/research/?gpu=webgl&frozen=1${TASK}`);
  await expect(task(page).getByRole("heading", { level: 2 })).toHaveText("The file here is not the released one", {
    timeout: 45_000,
  });
  await task(page).getByRole("button", { name: "Choose another model" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Choose a model export" })).toBeVisible();
});

test("without a GPU the task is complete through the table", async ({ page }) => {
  const { errors } = watch(page);
  await page.goto(`/research/?gpu=static${TASK}`);
  await expect(task(page)).toContainText("Verified", { timeout: 45_000 });
  await expect(page.locator(".interp-static")).toContainText("needs WebGL or WebGPU");
  await expect(page.locator(".interp-static")).toContainText("discrete Fourier transform");
  await expect(page.locator(".interp-static a[download]")).toHaveText("fourier.json");
  await pick(page, /stored position embeddings repeat/);
  await pick(page, /Record frequency 1$/);
  await pick(page, /uses a given frequency/);
  await Promise.all([
    page.waitForEvent("download"),
    task(page).getByRole("button", { name: /Download gpt2-fourier\.json/ }).click(),
  ]);
  await expect(page.locator(".research-complete")).toContainText("Task complete");
  expect(errors).toEqual([]);
});

test("each export states which analyses it has", async ({ page }) => {
  await page.goto("/research/?gpu=webgl&frozen=1");
  await page.getByRole("button", { name: /^GPT-2 Medium\b/ }).click({ timeout: 45_000 });
  await expect(page.locator(".interp-rail-count")).toHaveText("18 of 26 available");
  // the SAE views and the grokking toy are gpt2-only
  const sae = page.locator(".interp-feature.is-unavailable");
  await expect(sae.first()).toBeVisible();
  expect(await sae.count()).toBeGreaterThanOrEqual(6);
  await expect(page.locator(".interp-model-note")).toContainText("GPT-2 small");
  // tours run in Learn, so Research links there
  await page.getByRole("combobox", { name: "Model" }).selectOption("gpt2");
  const tour = page.locator("a.interp-tour-btn").first();
  await expect(tour).toHaveAttribute("href", /\/learn\/#episode=[\w-]+&step=0$/);
});

test("the legacy Atlas analysis link opens the Research task", async ({ page }) => {
  const { maps } = watch(page);
  await page.goto(`/atlas/?gpu=webgl&frozen=1${TASK}`);
  await page.waitForURL(/\/research\//);
  await expect(task(page)).toContainText("Verified", { timeout: 45_000 });
  expect(maps).toEqual([]);
});

test("the task stacks under the chart at 390px", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`/research/?gpu=webgl&frozen=1${TASK}`);
  await expect(task(page)).toContainText("Verified", { timeout: 45_000 });
  const chart = await page.locator(".interp-chart").boundingBox();
  const panel = await task(page).boundingBox();
  expect(panel!.y).toBeGreaterThanOrEqual(chart!.y + chart!.height - 1);
  expect(panel!.x + panel!.width).toBeLessThanOrEqual(391);
});

test("an analysis opened from Methods survives the export choice", async ({ page }) => {
  const { maps } = watch(page);
  await page.goto("/research/?gpu=webgl&frozen=1#page=guide");
  const card = page.locator(".guide-card", { hasText: "Map of SAE Feature Directions" }).first();
  await card.locator(".guide-card-open").click({ timeout: 45_000 });
  const chooser = page.locator(".research-chooser");
  await expect(chooser.locator(".research-chooser-request")).toContainText("You asked for");
  // the SAE views are exported for gpt2 only, and the chooser says so per export
  await expect(page.getByRole("button", { name: /^GPT-2\s*Recommended/ })).toContainText(/Includes /);
  await expect(page.getByRole("button", { name: /^GPT-2 Medium\b/ })).toContainText(/Does not include /);
  const requested = await page.evaluate(() => window.__store.getState().interp.featureId);
  await page.getByRole("button", { name: /^GPT-2\s*Recommended/ }).click();
  await expect(page.locator(".interp-page")).toBeVisible();
  expect(await page.evaluate(() => window.__store.getState().interp.featureId)).toBe(requested);
  await expect(page.locator("aside.research-task")).toHaveCount(0);
  expect(maps).toEqual([]);
});

test("a link naming an analysis but no export asks for the export", async ({ page }) => {
  await page.goto("/research/?gpu=webgl&frozen=1#page=interp&feature=logit-lens-tunnel");
  await expect(page.locator(".research-chooser-request")).toContainText("You asked for", {
    timeout: 45_000,
  });
  await page.locator(".research-chooser-request button").click();
  await expect(page.locator(".research-chooser-request")).toHaveCount(0);
  await page.getByRole("button", { name: /^GPT-2\s*Recommended/ }).click();
  await expect(page.locator("aside.research-task")).toContainText("Verified");
});
