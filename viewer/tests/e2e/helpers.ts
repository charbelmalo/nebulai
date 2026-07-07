/** Shared e2e plumbing: boot the app on a given gpu rung, collect console
 *  errors from the very first request, and wait for the deep-linked view.
 *  All state waits go through `window.__store` / `window.__perf` (exposed by
 *  main.ts for exactly this purpose) — never through timing guesses. */
import type { Page, TestInfo } from "@playwright/test";

export type Rung = "webgl" | "webgpu";
export type View = "atlas" | "chord" | "hierarchy" | "compare";

export function rungOf(testInfo: TestInfo): Rung {
  return testInfo.project.name === "webgpu" ? "webgpu" : "webgl";
}

export interface BootResult {
  /** console errors + uncaught page errors since navigation */
  errors: string[];
  /** the tier the capability probe actually landed on */
  tier: string;
}

export async function bootApp(
  page: Page,
  rung: Rung,
  opts: { view?: View; frozen?: boolean } = {},
): Promise<BootResult> {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") {
      const loc = msg.location();
      errors.push(loc.url ? `${msg.text()} [${loc.url}]` : msg.text());
    }
  });
  page.on("pageerror", (err) => errors.push(String(err)));
  page.on("response", (res) => {
    if (res.status() >= 400) errors.push(`${res.status()} ${res.url()}`);
  });

  const params = new URLSearchParams({ gpu: rung });
  if (opts.frozen !== false) params.set("frozen", "1"); // pin t for goldens
  if (opts.view && opts.view !== "atlas") params.set("view", opts.view);
  await page.goto(`/?${params.toString()}`);

  await page.waitForFunction(
    () => window.__perf.bootMs !== undefined && window.__store.getState().dataset !== null,
    undefined,
    { timeout: 45_000 },
  );
  const tier = await page.evaluate(
    () => window.__store.getState().capabilities?.tier ?? "unknown",
  );

  // the compare deep link only fires on the webgpu tier — don't wait for a
  // view switch that by design never happens on a fallback rung
  const waitView = opts.view && opts.view !== "atlas" && !(opts.view === "compare" && tier !== "webgpu");
  if (waitView) {
    await page.waitForFunction((v) => window.__store.getState().viewMode === v, opts.view, {
      timeout: 20_000,
    });
  }
  return { errors, tier };
}

/** Wait until the atlas camera/morph stop moving (the boot fly-in, view
 *  crossfades), then a beat more for label-overlay CSS transitions — the
 *  goldens need a fully settled frame, not a fixed sleep. */
export async function waitForSettle(page: Page): Promise<void> {
  await page.waitForFunction(
    () => {
      const d = window.__driver as unknown as {
        cam: { cx: number; cy: number; wpp: number };
        morph: number;
      };
      if (!d) return true;
      const key = JSON.stringify([d.cam.cx, d.cam.cy, d.cam.wpp, d.morph]);
      const w = window as unknown as { __settleKey?: string; __settleAt?: number };
      if (w.__settleKey !== key) {
        w.__settleKey = key;
        w.__settleAt = performance.now();
        return false;
      }
      return performance.now() - (w.__settleAt ?? 0) > 400;
    },
    undefined,
    { timeout: 20_000, polling: 100 },
  );
  await page.waitForTimeout(400);
}
