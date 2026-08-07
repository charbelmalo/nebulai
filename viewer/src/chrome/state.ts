/** zustand → @preact/signals bridge. One signal per store slice the chrome
 *  reads; a single store subscription keeps them in sync. Writes go the other
 *  way through store setters / app actions — signals are read-only mirrors. */

import { signal } from "@preact/signals";
import {
  appStore,
  type Appearance,
  type CompareUI,
  type InterpSelection,
  type InterpUI,
  type MapQuery,
  type Page,
  type Probing,
  type Progress,
  type Selection,
  type SessionsState,
  type Settings,
  type SnapshotState,
  type Toggles,
  type TourRef,
  type ViewMode,
} from "../app/store";
import type { Capabilities } from "../app/capabilities";
import type { CompareTourState } from "../app/actions";
import type { CompareData } from "../data/compare";
import type { Dataset } from "../data/loader";
import type { DatasetEntry } from "../data/schema";

const s = appStore.getState();

export const $capabilities = signal<Capabilities | null>(s.capabilities);
export const $datasets = signal<DatasetEntry[]>(s.datasets);
export const $datasetId = signal<string | null>(s.datasetId);
export const $dataset = signal<Dataset | null>(s.dataset);
export const $loading = signal(s.loading);
export const $viewMode = signal<ViewMode>(s.viewMode);
export const $dims = signal<2 | 3>(s.dims);
export const $selection = signal<Selection | null>(s.selection);
export const $mapQuery = signal<MapQuery>(s.mapQuery);
export const $toggles = signal<Toggles>(s.toggles);
export const $settings = signal<Settings>(s.settings);
export const $appearance = signal<Appearance>(s.appearance);
export const $probing = signal<Probing>(s.probing);
export const $progress = signal<Progress>(s.progress);
export const $settingsOpen = signal<boolean>(s.settingsOpen);
export const $page = signal<Page>(s.page);
export const $snapshot = signal<SnapshotState>(s.snapshot);
export const $sessions = signal<SessionsState>(s.sessions);
export const $interp = signal<InterpUI>(s.interp);
export const $interpSelection = signal<InterpSelection | null>(s.interpSelection);
export const $tour = signal<TourRef | null>(s.tour);
export const $compareData = signal<CompareData | null>(s.compareData);
export const $compare = signal<CompareUI>(s.compare);

/** The one signal here that does NOT mirror a store slice: the compare layout
 *  tour lives on a GPU uniform inside the driver, and main.ts pumps it here as
 *  the driver reports. Putting it in the store would mean a write per frame and
 *  would break "chrome writes the store; the driver only reads". null = the
 *  compare driver hasn't been built yet. */
export const $compareTour = signal<CompareTourState | null>(null);

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
 *  src/app/capabilities.ts for the same guard). `null` (no matchMedia
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

appStore.subscribe((st) => {
  if (st.capabilities !== $capabilities.value) $capabilities.value = st.capabilities;
  if (st.datasets !== $datasets.value) $datasets.value = st.datasets;
  if (st.datasetId !== $datasetId.value) $datasetId.value = st.datasetId;
  if (st.dataset !== $dataset.value) $dataset.value = st.dataset;
  if (st.loading !== $loading.value) $loading.value = st.loading;
  if (st.viewMode !== $viewMode.value) $viewMode.value = st.viewMode;
  if (st.dims !== $dims.value) $dims.value = st.dims;
  if (st.selection !== $selection.value) $selection.value = st.selection;
  if (st.mapQuery !== $mapQuery.value) $mapQuery.value = st.mapQuery;
  if (st.toggles !== $toggles.value) $toggles.value = st.toggles;
  if (st.settings !== $settings.value) $settings.value = st.settings;
  if (st.appearance !== $appearance.value) $appearance.value = st.appearance;
  if (st.probing !== $probing.value) $probing.value = st.probing;
  if (st.progress !== $progress.value) $progress.value = st.progress;
  if (st.settingsOpen !== $settingsOpen.value) $settingsOpen.value = st.settingsOpen;
  if (st.page !== $page.value) $page.value = st.page;
  if (st.snapshot !== $snapshot.value) $snapshot.value = st.snapshot;
  if (st.sessions !== $sessions.value) $sessions.value = st.sessions;
  if (st.interp !== $interp.value) $interp.value = st.interp;
  if (st.interpSelection !== $interpSelection.value)
    $interpSelection.value = st.interpSelection;
  if (st.tour !== $tour.value) $tour.value = st.tour;
  if (st.compareData !== $compareData.value) $compareData.value = st.compareData;
  if (st.compare !== $compare.value) $compare.value = st.compare;
});
