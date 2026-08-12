/** state/sync.ts — the ONE store subscription that drives every signal in this
 *  directory.
 *
 *  It stays single on purpose. The obvious-looking alternative — each state
 *  module subscribing for its own slice — would register eight listeners that
 *  zustand walks on every single `set`, and it would hand the update ordering
 *  over to module-evaluation order, so a store write that touches two slices
 *  could land in the signals in a different sequence than it does today. One
 *  listener, one pass, one fixed write order: the body below is a verbatim
 *  port of the subscribe that lived at the bottom of chrome/state.ts, field
 *  order and identity checks included. Anything added to a state module must
 *  be appended to THIS function, not given a subscription of its own.
 *
 *  The `!==` guards are identity checks, not deep equality, and that is the
 *  whole contract: the store replaces the object it writes, so reference
 *  inequality means "actually changed" and skipping on equality is what keeps
 *  an unrelated `set` from waking every subscriber of every signal.
 *
 *  One seam the split introduced: the eight state modules each take their own
 *  `appStore.getState()` snapshot to seed their signals, where the single file
 *  they came from took one snapshot for all of them. That is equivalent today —
 *  module evaluation is synchronous, nothing writes the store at import time,
 *  and this subscribe registers after all eight have evaluated. It stops being
 *  equivalent the moment any state module writes the store while importing:
 *  modules evaluated earlier would keep a pre-write value forever, because the
 *  subscribe below only fires on LATER `set` calls. Keep import-time store
 *  writes out of `state/`. */

import { appStore } from "../../app/store";
import { $appearance } from "./appearance";
import {
  $compare,
  $compareData,
  $dataset,
  $datasetId,
  $datasets,
  $dims,
  $loading,
  $mapQuery,
  $selection,
  $toggles,
  $viewMode,
} from "./atlas";
import { $interp, $interpSelection, $tour } from "./interp";
import { $probing, $progress } from "./probing";
import { $seer } from "./seer";
import { $sessions } from "./sessions";
import { $app, $capabilities, $page, $settings, $settingsOpen } from "./shell";
import { $snapshot } from "./snapshot";

appStore.subscribe((st) => {
  if (st.app !== $app.value) $app.value = st.app;
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
  if (st.seer !== $seer.value) $seer.value = st.seer;
});
