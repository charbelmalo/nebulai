/** smoke-release.mjs — check a NebulAI release where it is actually served.
 *
 *    node scripts/smoke-release.mjs https://research.elysiumsystems.net
 *    node scripts/smoke-release.mjs http://127.0.0.1:8766 --all-artifacts
 *
 *  The argument is the ORIGIN; the app is expected at /psychiX/nebulai-maps/
 *  and Seer at /psychiX/seer/, exactly as in production, on a static host with
 *  no SPA rewrite (a missing entry must 404, not fall back to index.html).
 *
 *  HTTP: every entry document, the release manifest, the research bundle and
 *  the Atlas starter artifact (every artifact with --all-artifacts — ~150 MB,
 *  so only on a fast link) are fetched and hashed against the manifest.
 *  Browser (headless Chromium, WebGL): the root chooser loads no map; Research
 *  verifies its pinned bundle; Learn lists its lesson; Atlas opens its starter;
 *  a pinned unit link reopens the exact unit; the legacy Internals link lands
 *  in Research; Seer boots its own shell. Same-origin 4xx/5xx responses and
 *  page errors fail the run.
 *
 *  Exit 0 = every check passed. Each check prints one line. */

import { createHash } from "node:crypto";
import { chromium } from "@playwright/test";

const origin = (process.argv[2] ?? "").replace(/\/$/, "");
if (!/^https?:\/\//.test(origin)) {
  console.error("usage: node scripts/smoke-release.mjs <origin> [--all-artifacts]");
  process.exit(2);
}
const ALL = process.argv.includes("--all-artifacts");
const APP = `${origin}/psychiX/nebulai-maps/`;
const SEER = `${origin}/psychiX/seer/`;
const OUT = `${APP}out/`;

let failed = 0;
const ok = (msg) => console.log(`ok    ${msg}`);
const bad = (msg) => {
  failed++;
  console.log(`FAIL  ${msg}`);
};
const sha = (buf) => createHash("sha256").update(buf).digest("hex");
const bust = (u) => `${u}${u.includes("?") ? "&" : "?"}smoke=${Date.now()}`;

async function get(url) {
  const r = await fetch(bust(url), { redirect: "follow", cache: "no-store" });
  return { r, buf: Buffer.from(await r.arrayBuffer()) };
}

/* ── HTTP ───────────────────────────────────────────────────────────────── */

for (const [name, url] of [
  ["root", APP],
  ["learn", `${APP}learn/`],
  ["atlas", `${APP}atlas/`],
  ["research", `${APP}research/`],
  ["seer", SEER],
]) {
  const { r, buf } = await get(url);
  const html = buf.toString("utf8");
  const type = r.headers.get("content-type") ?? "";
  // every entry must reference assets at the app's ROOT base, never a nested one
  const nested = /(src|href)="\/psychiX\/nebulai-maps\/(learn|atlas|research)\/assets\//.test(html);
  if (r.status === 200 && type.includes("text/html") && /<script[^>]+type="module"/.test(html) && !nested)
    ok(`${name} entry 200 text/html`);
  else bad(`${name} entry: ${r.status} ${type}${nested ? " (nested asset path)" : ""}`);
}
{
  const { r } = await get(`${APP}no-such-entry/`);
  if (r.status === 404) ok("unknown entry 404s (no SPA fallback)");
  else bad(`unknown entry answered ${r.status}`);
}

const { r: mr, buf: mbuf } = await get(`${OUT}experience.json`);
let manifest = null;
try {
  manifest = JSON.parse(mbuf.toString("utf8"));
  if (mr.status === 200 && manifest.schema_version === 1) ok(`manifest 200, ${manifest.artifacts.length} artifacts`);
  else bad(`manifest ${mr.status} schema ${manifest.schema_version}`);
} catch (e) {
  bad(`manifest ${mr.status}: ${e}`);
}

if (manifest) {
  const ri = manifest.research_intro;
  const { r, buf } = await get(`${OUT}${ri.bundle_path}`);
  if (r.status === 200 && buf.length === ri.bytes && sha(buf) === ri.sha256)
    ok(`research bundle ${ri.bundle_path} matches sha256 ${ri.sha256.slice(0, 12)}`);
  else bad(`research bundle ${r.status} ${buf.length} B sha ${sha(buf).slice(0, 12)}`);

  const starter = manifest.default_atlas.sha256;
  const pick = ALL ? manifest.artifacts : manifest.artifacts.filter((a) => a.sha256 === starter);
  for (const a of pick) {
    const { r, buf } = await get(`${OUT}${a.path}`);
    const h = sha(buf);
    if (r.status === 200 && h === a.sha256 && buf.length === a.bytes) ok(`artifact ${a.dataset_id} sha256 ok`);
    else bad(`artifact ${a.dataset_id}: ${r.status} ${buf.length} B sha ${h.slice(0, 12)}`);
    // the legacy path must still serve the same bytes for a CURRENT artifact
    if (a.current !== false && a.legacy_path) {
      const { buf: lb } = await get(`${OUT}${a.legacy_path}`);
      if (sha(lb) === a.sha256) ok(`legacy ${a.legacy_path} = artifact`);
      else bad(`legacy ${a.legacy_path} differs from its current artifact`);
    }
  }
  if (!ALL) console.log(`      (hashed the starter only; --all-artifacts hashes all ${manifest.artifacts.length})`);
}

/* ── browser ────────────────────────────────────────────────────────────── */

const browser = await chromium.launch({ args: ["--use-angle=metal", "--ignore-gpu-blocklist"] });

async function visit(name, url, check, { width = 1280, height = 800 } = {}) {
  const ctx = await browser.newContext({ viewport: { width, height } });
  const page = await ctx.newPage();
  const errors = [];
  const maps = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("response", (res) => {
    const u = res.url();
    if (u.startsWith(origin) && res.status() >= 400) errors.push(`${res.status()} ${u.slice(origin.length)}`);
  });
  page.on("request", (req) => {
    if (/\/nebulai\.json(\?|$)/.test(req.url())) maps.push(req.url());
  });
  try {
    await page.goto(url, { waitUntil: "domcontentloaded" });
    const note = await check(page, maps);
    if (errors.length) bad(`${name}: ${errors.slice(0, 4).join("; ")}`);
    else ok(`${name}${note ? ` — ${note}` : ""}`);
  } catch (e) {
    bad(`${name}: ${String(e.message ?? e).split("\n")[0]}${errors.length ? ` [${errors.slice(0, 3).join("; ")}]` : ""}`);
  } finally {
    await ctx.close();
  }
}

const T = 90_000;

await visit("root chooser, no map fetched", `${APP}?gpu=webgl`, async (page, maps) => {
  await page.locator(".chooser-card-link").first().waitFor({ timeout: T });
  const n = await page.locator(".chooser-card-link").count();
  await page.waitForTimeout(1500);
  if (n !== 3) throw new Error(`${n} cards`);
  if (maps.length) throw new Error(`fetched ${maps.length} map(s)`);
  return "3 cards";
});

await visit("research first task verified", `${APP}research/?gpu=webgl`, async (page, maps) => {
  const countEl = page.locator(".research-export.is-recommended .research-export-count");
  await countEl.filter({ hasText: /of \d+ analyses available/ }).waitFor({ timeout: T });
  const count = (await countEl.textContent()) ?? "";
  await page.locator(".research-export.is-recommended").click();
  await page.locator("aside.research-task").getByText("Verified").first().waitFor({ timeout: T });
  if (maps.length) throw new Error(`fetched ${maps.length} map(s)`);
  return count.trim();
});

await visit("legacy Internals link lands in Research", `${APP}atlas/?gpu=webgl#page=interp&feature=fourier-atlas&model=gpt2`, async (page) => {
  await page.waitForURL(/\/research\//, { timeout: T });
  await page.locator("aside.research-task").getByText("Verified").first().waitFor({ timeout: T });
});

await visit("learn lists its lesson", `${APP}learn/?gpu=webgl`, async (page) => {
  await page.locator(".lesson-card").first().waitFor({ timeout: T });
});

await visit("atlas opens its starter", `${APP}atlas/?gpu=webgl`, async (page) => {
  await page.locator(".rt-table tbody tr").first().waitFor({ timeout: T });
  return ((await page.locator(".rt-summary").textContent()) ?? "").trim();
});

if (manifest) {
  const a = manifest.default_atlas;
  const q = new URLSearchParams({
    experience: "atlas",
    page: "map",
    model: a.dataset_id,
    view: "atlas",
    dims: String(a.dimensions),
    artifact: a.sha256,
    point: "0",
    unit_kind: "sae_decoder(gpt2-small-res-jb, blocks.8.hook_resid_pre)",
    unit_index: "0",
  });
  await visit("pinned unit link reopens the exact unit", `${APP}atlas/?gpu=webgl#${q}`, async (page) => {
    const badge = page.locator(".inspector-badge");
    await badge.waitFor({ timeout: T });
    await page.waitForFunction(
      () => /Verified against artifact/.test(document.querySelector(".inspector-badge")?.textContent ?? ""),
      undefined,
      { timeout: T },
    );
    return ((await page.locator(".inspector h2").textContent()) ?? "").trim();
  });
}

await visit("research task at 390px", `${APP}research/?gpu=static#page=interp&model=gpt2&feature=fourier-atlas`, async (page) => {
  await page.locator("aside.research-task").getByText("Verified").first().waitFor({ timeout: T });
  const scroll = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  if (scroll > 0) throw new Error(`horizontal scroll ${scroll}px`);
}, { width: 390, height: 844 });

await visit("seer boots its own shell", `${SEER}?gpu=webgl`, async (page) => {
  await page.waitForFunction(() => window.__store?.getState().app === "seer", undefined, { timeout: T });
});

await browser.close();
console.log(failed ? `\n${failed} check(s) FAILED` : "\nall checks passed");
process.exit(failed ? 1 : 0);
