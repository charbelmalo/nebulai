/** boot-shell.ts — the half of boot that both instruments run, extracted from
 *  main.ts's `boot()` when Seer got its own Vite entry.
 *
 *  Step 3 split main.ts into `boot()` (probe the GPU tier, mount the chrome,
 *  read the permalink, apply it) and `bootAtlas()` (everything that needs
 *  baked artifacts). This module is the first half made reusable, and NOTHING
 *  more: it does not know what a dataset is, it never touches `out/`, and it
 *  imports no driver. Seer calls it and stops; Nebulai calls it, runs
 *  `bootAtlas` in between, and finishes.
 *
 *  It is deliberately two functions rather than one with a callback, because
 *  the permalink has to be applied AFTER the app-specific middle runs —
 *  `applyUrlState` routes view switches through actions that `bootAtlas`
 *  registers — but the pill, the probe and the chrome have to come BEFORE it,
 *  so a failed or absent atlas still leaves a page standing:
 *
 *      const shell = await bootShell(APP);
 *      …app-specific boot, allowed to fail…
 *      finishShellBoot(shell.urlState);
 *
 *  Note what is NOT here: the stylesheet. Each entry imports its own sheet
 *  (styles/nebulai.css / styles/seer.css) so a document downloads the partials
 *  for its own three pages, and so chrome.responsive.css keeps winning the
 *  cascade by being last — which only the entry can guarantee. */

import { probeCapabilities, type Capabilities } from "@psychix/viz/capabilities";
import { mountChrome } from "../chrome/mount";
import type { AppShell } from "../chrome/apps/types";
import { applyUrlState, readUrlState, startUrlSync, type UrlState } from "../chrome/urlState";
import { appStore } from "./store";

declare global {
  interface Window {
    __perf: {
      parseMs?: number;
      bootMs?: number;
      /** Browser-scheduled requestAnimationFrame interval. Diagnostic only:
       * headless browsers may cap this independently of application work. */
      p95FrameMs?: number;
      /** Main-thread work performed by NebulAI inside one frame callback. */
      p95FrameWorkMs?: number;
      /** Most recent dataset change, measured wholly inside the app. */
      datasetSwitchMs?: number;
    };
    __store: typeof appStore;
  }
}

export interface BootedShell {
  caps: Capabilities;
  /** the permalink as read at boot — hand it back to `finishShellBoot` */
  urlState: UrlState;
  /** #chrome, with the boot pill already inside it */
  chrome: HTMLElement;
  /** the boot progress bar; Nebulai drives it from the dataset fetch, Seer
   *  never touches it (it has nothing to download at boot) */
  progress: HTMLDivElement;
  /** write the status pill — the one line that is always on screen */
  say(text: string): void;
}

/** Probe, mount, read the permalink. Everything in here is unconditional and
 *  nothing in it can be starved by missing data. */
export async function bootShell(app: AppShell): Promise<BootedShell> {
  window.__perf = {};
  window.__store = appStore; // e2e tests read state through this

  document.title = app.documentTitle;
  appStore.getState().setApp(app.id); // also lands `page` on this app's first

  const chrome = document.getElementById("chrome")!;
  const progress = document.createElement("div");
  progress.className = "boot-progress";
  const status = document.createElement("div");
  status.className = "boot-status";
  // the MetaLine truncates to one line on compact viewports (chrome.base.css);
  // tapping it reveals the full provenance string instead of leaving it clipped
  status.addEventListener("click", () => status.classList.toggle("is-expanded"));
  chrome.append(progress, status);
  const say = (text: string) => {
    status.textContent = text;
  };

  const caps = await probeCapabilities();
  appStore.getState().setCapabilities(caps);

  // The chrome goes up before any data is asked for. It is safe this early:
  // mountChrome appends its own root next to the boot pill rather than
  // replacing it, and every action the chrome can fire routes through
  // app/actions.ts's optional-chained handler, which no-ops until the app
  // registers the real set (Nebulai does so in bootAtlas; Seer registers none,
  // because none of its three pages command a driver through that bridge).
  mountChrome(chrome, app);

  // permalink: `#model=` picks Nebulai's boot dataset; the rest of the hash
  // state is applied by finishShellBoot once the app shell is wired
  return { caps, urlState: readUrlState(), chrome, progress, say };
}

/** Apply the remaining hash state and start mirroring the store into it.
 *  Called on EVERY path — atlas, no-atlas and failed-atlas alike — so a
 *  permalink is honoured even when the thing it points at could not load. */
export function finishShellBoot(urlState: UrlState): void {
  applyUrlState(urlState);
  startUrlSync();
}
