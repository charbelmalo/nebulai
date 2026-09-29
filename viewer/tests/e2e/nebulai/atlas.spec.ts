/** NebulAI Atlas's first task (PRODUCT-EXPERIENCES.md): search → inspect →
 *  save → reopen an exact unit, by map, by list, by keyboard and without a
 *  GPU — and every G05 negative case ending in a readable refusal that
 *  opens nothing. Chrome and data only, so one rung is enough. */
import { readFile, writeFile } from "node:fs/promises";
import { expect, test, type Page } from "@playwright/test";
import { rungOf } from "../helpers";

test.beforeEach(({}, testInfo) => {
  test.skip(rungOf(testInfo) === "webgpu", "chrome is identical on both rungs");
});

const STARTER = "gpt2-small__sae__blocks.8.hook_resid_pre";
const STARTER_SHA = "d36402d8503e82df71a9c64e48a4f9561f35ad2ac666c43eee127ff71eac10f7";
const UNIT_KIND = "sae_decoder(gpt2-small-res-jb, blocks.8.hook_resid_pre)";

function watch(page: Page): { errors: string[]; mapFetches: string[] } {
  const errors: string[] = [];
  const mapFetches: string[] = [];
  page.on("pageerror", (err) => errors.push(String(err)));
  page.on("request", (req) => {
    if (req.url().endsWith("/nebulai.json")) mapFetches.push(req.url());
  });
  return { errors, mapFetches };
}

async function booted(page: Page): Promise<void> {
  await page.waitForFunction(() => !!window.__store?.getState().experience, undefined, { timeout: 45_000 });
}

async function mapLoaded(page: Page): Promise<void> {
  await page.waitForFunction(() => window.__store.getState().dataset !== null, undefined, { timeout: 45_000 });
}

const pinState = (page: Page) => page.evaluate(() => window.__store.getState().pin);
const pinCode = (page: Page) =>
  page.evaluate(() => {
    const pin = window.__store.getState().pin;
    return pin.status === "error" ? pin.code : null;
  });

function pinHash(p: Record<string, string>): string {
  const q = new URLSearchParams({
    experience: "atlas",
    page: "map",
    model: STARTER,
    view: "atlas",
    dims: "2",
    artifact: STARTER_SHA,
    point: "0",
    unit_kind: UNIT_KIND,
    unit_index: "0",
    ...p,
  });
  for (const [k, v] of Object.entries(p)) if (v === "") q.delete(k);
  return q.toString();
}

test("the starter opens with its evidence and point 0 first in the list", async ({ page }) => {
  const { errors } = watch(page);
  await page.goto("/atlas/?gpu=webgl&frozen=1");
  await mapLoaded(page);
  const ws = page.getByRole("complementary", { name: "Atlas workspace" });
  await expect(ws.getByRole("heading", { level: 2 })).toHaveText("GPT-2 Small · SAE decoder directions · Layer 8");
  await expect(ws).toContainText("4,096 of 24,576 SAE decoder directions");
  await expect(ws).toContainText("Labels: Neuronpedia");
  await expect(ws).toContainText("do not establish activation on an input");
  await expect(page.locator(".rt-summary")).toHaveText("Showing 1–50 of 4,096, in artifact order.");
  await expect(page.locator(".rt-inspect").first()).toHaveAccessibleName("Inspect numbers and quantities");
  await expect(page.locator(".rt-table tbody tr").first().locator("td").first()).toHaveText("0");

  await ws.getByText("About this map").click();
  await expect(ws.locator(".aw-dl")).toContainText("49.46% unclustered");
  await expect(ws.locator(".aw-dl")).toContainText("(noise_fraction 0.4946)");
  await expect(ws.locator(".aw-dl")).toContainText(`sha256 ${STARTER_SHA.slice(0, 12)}`);
  // the pager reaches the last page
  await expect(page.locator(".rt-page")).toHaveText("Page 1 of 82");
  expect(errors).toEqual([]);
});

test("search → inspect by keyboard → Escape returns to the row", async ({ page }) => {
  await page.goto("/atlas/?gpu=webgl&frozen=1");
  await mapLoaded(page);
  const search = page.getByLabel("Search labels");
  await search.fill("numbers and quantities");
  await expect(page.locator(".rt-summary")).toContainText("of 4,096 labels contain “numbers and quantities”");
  const row = page.getByRole("button", { name: "Inspect numbers and quantities" }).first();
  await row.focus();
  await page.keyboard.press("Enter");

  const insp = page.locator(".inspector");
  await expect(insp.getByRole("heading", { level: 2 })).toHaveText("numbers and quantities");
  await expect(insp.getByRole("heading", { level: 2 })).toBeFocused();
  await expect(insp).toContainText(UNIT_KIND);
  await expect(insp).toContainText("index 0 · point ID 0");
  await expect(insp).toContainText("Unclustered");
  await expect(insp).toContainText("projection coordinates, not an activation");
  await expect(insp).toContainText("Cluster membership is not confidence in the truth of a label.");
  expect(await page.evaluate(() => window.__store.getState().selection)).toEqual({ kind: "point", id: 0 });
  // the legend steps aside while a unit is open
  await expect(page.locator(".legend")).toHaveCount(0);
  // the selected row is marked for assistive tech
  await expect(page.locator('.rt-table tr[aria-current="true"]')).toHaveCount(1);

  await page.keyboard.press("Escape");
  await expect(insp).toHaveCount(0);
  await expect(row).toBeFocused();
  expect(await page.evaluate(() => window.__store.getState().selection)).toBeNull();

  // a no-match search says what search does, and can be cleared
  await search.fill("zzzqqq-no-such-label");
  await expect(page.locator(".rt-summary")).toHaveText("No label contains “zzzqqq-no-such-label”.");
  await page.getByRole("button", { name: "Clear search" }).click();
  await expect(search).toHaveValue("");
});

test("save a finding, reopen it from the file, and from its link", async ({ page, context }, testInfo) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto("/atlas/?gpu=webgl&frozen=1");
  await mapLoaded(page);
  await page.getByRole("button", { name: "Inspect words related to piers" }).click();
  await page.getByRole("button", { name: "Save finding…" }).click();
  await page.getByLabel("Note (optional)").fill("pier sense, not peer");
  const [dl] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: "Download finding" }).click(),
  ]);
  expect(dl.suggestedFilename()).toBe(`nebulai-finding-${STARTER}-point-1.json`);
  const file = testInfo.outputPath("finding.json");
  await dl.saveAs(file);
  const finding = JSON.parse(await readFile(file, "utf8"));
  expect(finding.kind).toBe("nebulai.atlas-unit-finding");
  expect(finding.artifact.sha256).toBe(STARTER_SHA);
  expect(finding.unit).toEqual({ point_id: 1, kind: UNIT_KIND, index: 1 });
  expect(finding.note).toBe("pier sense, not peer");

  await page.getByRole("button", { name: "Copy unit link" }).click();
  await expect(page.locator(".inspector-copy-state")).toContainText("Link copied");
  const link = await page.evaluate(() => navigator.clipboard.readText());
  expect(new URL(link).pathname).toBe("/atlas/");
  expect(link).not.toContain("pier");

  // a second visit: import the file
  await page.goto("/atlas/?gpu=webgl&frozen=1");
  await mapLoaded(page);
  await page.locator('[data-testid="finding-file"]').setInputFiles(file);
  await expect(page.locator(".inspector-badge")).toContainText("Verified against artifact sha256 d36402d8503e");
  await expect(page.locator(".inspector-badge")).toContainText("from a saved finding");
  await expect(page.locator(".inspector-imported")).toContainText("pier sense, not peer");
  expect(await page.evaluate(() => window.__store.getState().selection)).toEqual({ kind: "point", id: 1 });
  // the address bar now carries the pin (identity only, never the note)
  await expect.poll(() => new URL(page.url()).hash).toContain("point=1");
  expect(page.url()).not.toContain("pier");

  // the copied link, in a fresh page
  const fresh = await context.newPage();
  await fresh.goto(link.replace("/atlas/", "/atlas/?gpu=webgl&frozen=1"));
  await fresh.waitForFunction(() => window.__store?.getState().pin?.status === "ok", undefined, { timeout: 45_000 });
  await expect(fresh.locator(".inspector-badge")).toContainText("from a unit link");
  await expect(fresh.locator(".inspector h2")).toHaveText("words related to piers");
});

test("G05: malformed, wrong-kind and conflicting files are refused readably", async ({ page }, testInfo) => {
  await page.goto("/atlas/?gpu=webgl&frozen=1");
  await mapLoaded(page);
  const input = page.locator('[data-testid="finding-file"]');
  const notice = page.locator(".pin-notice");

  const bad = testInfo.outputPath("bad.json");
  await writeFile(bad, "{ not json");
  await input.setInputFiles(bad);
  await expect(notice).toHaveAttribute("role", "alert");
  await expect(notice).toContainText("This file can't be opened");
  expect(await pinCode(page)).toBe("malformed");
  await notice.getByRole("button", { name: "Choose a map" }).click();
  await expect(notice).toHaveCount(0);

  await writeFile(bad, JSON.stringify({ schema_version: 1, kind: "something.else" }));
  await input.setInputFiles(bad);
  await expect(notice).toBeVisible();
  expect(await pinCode(page)).toBe("wrong-kind");
  await notice.getByRole("button", { name: "Choose a map" }).click();

  // a genuine record whose label was edited no longer matches the map
  await page.getByRole("button", { name: "Inspect numbers and quantities" }).click();
  const [dl] = await Promise.all([
    page.waitForEvent("download"),
    (async () => {
      await page.getByRole("button", { name: "Save finding…" }).click();
      await page.getByRole("button", { name: "Download finding" }).click();
    })(),
  ]);
  const good = testInfo.outputPath("good.json");
  await dl.saveAs(good);
  const f = JSON.parse(await readFile(good, "utf8"));
  f.evidence.label = "something the map never said";
  await writeFile(bad, JSON.stringify(f));
  await page.keyboard.press("Escape");
  await input.setInputFiles(bad);
  await expect(notice).toContainText("Record conflicts with the map");
  await expect(notice).toContainText("differ: label");
  expect(await page.evaluate(() => window.__store.getState().selection)).toBeNull();
});

test("G05: incomplete, wrong-kind and wrong-artifact links open nothing", async ({ page }) => {
  const { mapFetches } = watch(page);
  // incomplete: the pin keys are partial
  await page.goto(`/atlas/?gpu=webgl&frozen=1#${pinHash({ point: "" })}`);
  await booted(page);
  const notice = page.locator(".pin-notice");
  await expect(notice).toContainText("Unit link can't be opened");
  await expect(notice).toContainText("missing point");
  await page.waitForTimeout(300);
  expect(await page.evaluate(() => window.__store.getState().datasetId)).toBeNull();
  // the refused link stays in the address bar, so it can be reported
  expect(new URL(page.url()).hash).toContain("unit_index=0");

  // a digest the manifest does not publish: refused without fetching
  const before = mapFetches.length;
  // a hash-only goto is a same-document navigation; boot reads the hash once
  await page.goto("about:blank");
  await page.goto(`/atlas/?gpu=webgl&frozen=1#${pinHash({ artifact: "0".repeat(64) })}`);
  await booted(page);
  await expect(notice).toContainText("Exact artifact unavailable");
  await expect(notice).toContainText("not substituted");
  await expect(notice).toContainText(`sha256 ${STARTER_SHA.slice(0, 12)}`);
  await page.waitForTimeout(300);
  expect(mapFetches.length).toBe(before);
  expect(await page.evaluate(() => window.__store.getState().datasetId)).toBeNull();
  // recovery is explicit
  await notice.getByRole("button", { name: "Open the latest map" }).click();
  await mapLoaded(page);
  expect(await page.evaluate(() => window.__store.getState().datasetId)).toBe(STARTER);

  // right artifact, a unit kind the map does not have
  // a hash-only goto is a same-document navigation; boot reads the hash once
  await page.goto("about:blank");
  await page.goto(`/atlas/?gpu=webgl&frozen=1#${pinHash({ unit_kind: "mlp_neuron(gpt2, h.8)" })}`);
  await booted(page);
  await expect(notice).toContainText("Unit not found");
  expect(await page.evaluate(() => window.__store.getState().selection)).toBeNull();
});

test("phone width: search, inspect and return without leaving the page", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/atlas/?gpu=webgl&frozen=1");
  await mapLoaded(page);
  const sw = page.getByRole("group", { name: "Show" });
  await expect(sw.getByRole("button", { name: "Map" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".atlas-workspace")).toBeHidden();
  await sw.getByRole("button", { name: "Results" }).click();
  await page.getByLabel("Search labels").fill("piers");
  await page.getByRole("button", { name: "Inspect words related to piers" }).click();
  const insp = page.locator(".inspector");
  await expect(insp).toBeVisible();
  const box = await insp.boundingBox();
  expect(box!.width).toBeGreaterThan(340);
  await insp.getByRole("button", { name: /Back to results/ }).click();
  await expect(page.locator(".atlas-workspace")).toBeVisible();
  // back keeps the selection; only the panel changed
  expect(await page.evaluate(() => window.__store.getState().selection)).toEqual({ kind: "point", id: 1 });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});

test("no GPU: the list is the interface and reaches the same evidence", async ({ page }) => {
  const { errors } = watch(page);
  await page.goto("/atlas/?gpu=static");
  await mapLoaded(page);
  await expect(page.locator(".aw-note")).toContainText("needs WebGL or WebGPU");
  await expect(page.getByRole("button", { name: "3D" })).toBeDisabled();
  await page.getByRole("button", { name: "Inspect numbers and quantities" }).click();
  await expect(page.locator(".inspector")).toContainText("index 0 · point ID 0");
  await expect(page.getByRole("button", { name: "Save finding…" })).toBeEnabled();

  await page.setViewportSize({ width: 390, height: 844 });
  await page.keyboard.press("Escape");
  const sw = page.getByRole("group", { name: "Show" });
  await expect(sw.getByRole("button", { name: "Map" })).toBeDisabled();
  await expect(sw.getByRole("button", { name: "Results" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".atlas-workspace")).toBeVisible();
  expect(errors).toEqual([]);
});
