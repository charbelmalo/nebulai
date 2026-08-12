/** state/shell.ts — signal mirrors for the ShellSlice (capabilities, global
 *  settings, the Settings overlay, the current page), plus the two pieces of
 *  chrome layout state that never touch the store at all: the compact-viewport
 *  flag and the one-panel-at-a-time coordination built on top of it.
 *
 *  The viewport/panel block lives with the shell mirrors rather than in its own
 *  module because `openPanel` writes all four panel signals in one pass — split
 *  them across files and the coordination rule stops being readable in one
 *  place, which is the only thing keeping it honest.
 *
 *  One stale pointer to be aware of: the $compactViewport comment below says
 *  "same rationale as $compareTour above". $compareTour moved to
 *  state/atlas.ts when this file was split out; its comment is kept verbatim
 *  rather than re-worded, so read "above" as "in state/atlas.ts". */

import { signal } from "@preact/signals";
import { appStore, type AppId, type Page, type Settings } from "../../app/store";
import type { Capabilities } from "@psychix/viz/capabilities";

const s = appStore.getState();

/** Which instrument this document is. Written once, by the entry module's
 *  `setApp` before the chrome mounts — it is mirrored rather than read
 *  straight off the store only so a component that branches on it re-renders
 *  if it ever stops being write-once. */
export const $app = signal<AppId>(s.app);

export const $capabilities = signal<Capabilities | null>(s.capabilities);
export const $settings = signal<Settings>(s.settings);
export const $settingsOpen = signal<boolean>(s.settingsOpen);
export const $page = signal<Page>(s.page);

/** Compact-viewport signal (phone widths, or a coarse pointer under 900px —
 *  tablets in portrait). Also NOT a store mirror: the viewport is a property
 *  of the window, not app data, and nothing outside the chrome layer needs
 *  it. Mirroring it through the store would mean every resize/rotation
 *  writes state that no driver or non-chrome code reads — same rationale as
 *  $compareTour above.
 *
 *  `isCompactViewport` is split out as a pure function so the default-seed
 *  logic is unit-testable without a DOM: vitest runs in plain Node, so a
 *  bare `matchMedia()` at module scope would throw on import (see
 *  src/viz/capabilities.ts for the same guard). `null` (no matchMedia
 *  available) reads as "not compact". */
const COMPACT_QUERY =
  "(max-width: 720px), ((pointer: coarse) and (max-width: 900px))";
const mql = typeof matchMedia !== "undefined" ? matchMedia(COMPACT_QUERY) : null;

export function isCompactViewport(mql: { matches: boolean } | null): boolean {
  return mql?.matches ?? false;
}

export const $compactViewport = signal<boolean>(isCompactViewport(mql));
mql?.addEventListener("change", (e) => {
  $compactViewport.value = e.matches;
});

/** One-at-a-time panel coordination on compact viewports (a product
 *  decision, not a UX nicety invented here). On desktop every panel keeps
 *  opening/closing independently, exactly like before these were promoted
 *  out of each component's local `useSignal`.
 *
 *  Seeded from `$compactViewport.peek()` — read ONCE at module init, never
 *  re-subscribed. If the seed instead tracked the live signal, a viewport
 *  change after boot (a phone rotation, a resize) would silently override
 *  whatever the user had just opened or closed by hand. Peek-once means the
 *  initial guess can only be wrong for the first render; after that, only
 *  openPanel() and each panel's own collapse button touch these signals. */
const compactAtBoot = $compactViewport.peek();
export const $sidebarOpen = signal<boolean>(!compactAtBoot);
export const $searchCollapsed = signal<boolean>(compactAtBoot);
export const $legendCollapsed = signal<boolean>(compactAtBoot);
export const $compareCollapsed = signal<boolean>(compactAtBoot);

export type PanelName = "sidebar" | "search" | "legend" | "compare";

/** Opens `which`. On a compact viewport this also collapses every other
 *  panel, so only one ever covers the (mostly canvas-sized) screen at a
 *  time; on desktop there is no coupling — each panel opens independently,
 *  same as before this file grew shared panel state. */
export function openPanel(which: PanelName): void {
  if ($compactViewport.value) {
    $sidebarOpen.value = which === "sidebar";
    $searchCollapsed.value = which !== "search";
    $legendCollapsed.value = which !== "legend";
    $compareCollapsed.value = which !== "compare";
    return;
  }
  switch (which) {
    case "sidebar":
      $sidebarOpen.value = true;
      break;
    case "search":
      $searchCollapsed.value = false;
      break;
    case "legend":
      $legendCollapsed.value = false;
      break;
    case "compare":
      $compareCollapsed.value = false;
      break;
  }
}
