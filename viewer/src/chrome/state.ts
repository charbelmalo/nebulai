/** zustand → @preact/signals bridge. One signal per store slice the chrome
 *  reads; a single store subscription keeps them in sync. Writes go the other
 *  way through store setters / app actions — signals are read-only mirrors. */

import { signal } from "@preact/signals";
import {
  appStore,
  type Appearance,
  type CompareUI,
  type InterpUI,
  type Page,
  type Probing,
  type Progress,
  type Selection,
  type SessionsState,
  type Settings,
  type SnapshotState,
  type Toggles,
  type ViewMode,
} from "../app/store";
import type { Capabilities } from "../app/capabilities";
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
export const $compareData = signal<CompareData | null>(s.compareData);
export const $compare = signal<CompareUI>(s.compare);

appStore.subscribe((st) => {
  if (st.capabilities !== $capabilities.value) $capabilities.value = st.capabilities;
  if (st.datasets !== $datasets.value) $datasets.value = st.datasets;
  if (st.datasetId !== $datasetId.value) $datasetId.value = st.datasetId;
  if (st.dataset !== $dataset.value) $dataset.value = st.dataset;
  if (st.loading !== $loading.value) $loading.value = st.loading;
  if (st.viewMode !== $viewMode.value) $viewMode.value = st.viewMode;
  if (st.dims !== $dims.value) $dims.value = st.dims;
  if (st.selection !== $selection.value) $selection.value = st.selection;
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
  if (st.compareData !== $compareData.value) $compareData.value = st.compareData;
  if (st.compare !== $compare.value) $compare.value = st.compare;
});
