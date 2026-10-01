/** state/atlas.ts — signal mirrors for the AtlasSlice: the dataset list, the
 *  loaded dataset, the active view, the pick/query/toggle surface, and the
 *  compare-view pair. `hover` and `morphT` deliberately have no signal — they
 *  change at pointer/frame rate and only the drivers read them, so mirroring
 *  them would wake Preact for nothing. */

import { signal } from "@preact/signals";
import {
  appStore,
  type AtlasPanel,
  type AxisUI,
  type ChannelUI,
  type CompareUI,
  type LoadFailure,
  type MapQuery,
  type PinState,
  type RendererState,
  type Selection,
  type Toggles,
  type ViewMode,
} from "../../app/store";
import type { CompareTourState } from "../../app/actions";
import type { CompareData } from "../../data/compare";
import type { Dataset } from "../../data/loader";
import type { DatasetEntry } from "../../data/schema";

const s = appStore.getState();

export const $datasets = signal<DatasetEntry[]>(s.datasets);
export const $datasetId = signal<string | null>(s.datasetId);
export const $dataset = signal<Dataset | null>(s.dataset);
export const $loading = signal(s.loading);
/** the dataset a request is fetching right now (null when idle) */
export const $pendingDatasetId = signal<string | null>(s.pendingDatasetId);
/** why the last dataset request did not commit (null when it did) */
export const $loadError = signal<LoadFailure | null>(s.loadError);
export const $viewMode = signal<ViewMode>(s.viewMode);
export const $dims = signal<2 | 3>(s.dims);
export const $selection = signal<Selection | null>(s.selection);
export const $mapQuery = signal<MapQuery>(s.mapQuery);
export const $toggles = signal<Toggles>(s.toggles);
/** The channel lens: which per-point scalar the map is painted by, and the
 *  filter window in that channel's raw units. */
export const $channel = signal<ChannelUI>(s.channel);
/** The direction axis: which direction the map is laid out on, how far the
 *  blend has travelled, and whether the null cloud is in the picture. */
export const $axis = signal<AxisUI>(s.axis);
export const $compareData = signal<CompareData | null>(s.compareData);
/** exact-unit pin from a link or an imported finding, and how far it got */
export const $pin = signal<PinState>(s.pin);
export const $inspectorOpen = signal<boolean>(s.inspectorOpen);
export const $atlasPanel = signal<AtlasPanel>(s.atlasPanel);
/** map renderer readiness: `unavailable` = the results list is the whole UI */
export const $renderer = signal<RendererState>(s.renderer);
export const $unverifiedDefault = signal<boolean>(s.unverifiedDefault);
export const $compare = signal<CompareUI>(s.compare);

/** The one signal here that does NOT mirror a store slice: the compare layout
 *  tour lives on a GPU uniform inside the driver, and main.ts pumps it here as
 *  the driver reports. Putting it in the store would mean a write per frame and
 *  would break "chrome writes the store; the driver only reads". null = the
 *  compare driver hasn't been built yet. */
export const $compareTour = signal<CompareTourState | null>(null);
