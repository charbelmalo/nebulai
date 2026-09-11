/** atlas.ts — Nebulai's Map page: the loaded dataset, the driver-backed view
 *  the user is looking at (atlas / chord / hierarchy / compare), and every
 *  pick, hover, query and toggle that hangs off it. This is the biggest slice
 *  because it is the whole first instrument's data surface; Seer composes none
 *  of it.
 *
 *  `setDataset` reaches into the interp slice (`interpSelection`, `tour`) —
 *  unit ids and tour copy are both per-model, so a model switch has to clear
 *  them or the UI would keep pointing at units that no longer exist. That
 *  cross-slice write is only type-legal because this StateCreator is typed
 *  against the full `AppState`. */

import type { StateCreator } from "zustand";
import type { CompareData } from "../../data/compare";
import type { Dataset } from "../../data/loader";
import type { DatasetEntry } from "../../data/schema";
import { searchLabels, type SearchResults } from "../../data/search";
import type { AppState } from "../store";

export type ViewMode = "atlas" | "chord" | "hierarchy" | "compare";

export interface Selection {
  kind: "cluster" | "point";
  id: number;
}

/** Map-page keyword search. `results` is derived from `text` against the
 *  current dataset's labels at write time (labels never leave the main
 *  thread, so the store is the one place with both). null results = no
 *  query; a zero-match query keeps a non-null results object. */
export interface MapQuery {
  text: string;
  results: SearchResults | null;
}

export interface Toggles {
  territories: boolean;
  labels: boolean;
  beams: boolean;
  halos: boolean; // pulsing hub rings (the "radial bubbles")
  noise: boolean;
  legend: boolean;
}

/** Compare-view UI state. `hiddenModels` holds source indices toggled off in
 *  the legend; `state` indexes CompareData.states (default 1 = semantic). */
export interface CompareUI {
  state: number;
  hiddenModels: number[];
  sharedOnly: boolean;
}

/** The channel lens: colour the map by a per-point scalar from `channels.json`.
 *
 *  Deliberately NOT a `ViewMode`. The map keeps its layout, its clusters and its
 *  picking; only the paint changes, and turning the lens off returns the exact
 *  frame you had (R1 — a layout state is not a new coordinate system, and this
 *  is less than a layout state). A `ViewMode` would have forced every driver to
 *  learn about channels and would have made "look at the norms" feel like
 *  leaving the map.
 *
 *  `window` is in the channel's own raw units, so it survives being read out
 *  loud ("everything below 2.0") and put in a URL. `null` means the full
 *  measured range: no point is dimmed, which is the only honest default — any
 *  preset window would be the app asserting where the interesting values are. */
export interface ChannelUI {
  /** channel id from the dataset's `channels.json`, or null = lens off */
  id: string | null;
  /** filter window [lo, hi] in raw units; null = the whole measured range */
  window: [number, number] | null;
}

export interface AtlasSlice {
  datasets: DatasetEntry[];
  datasetId: string | null;
  dataset: Dataset | null;
  compareData: CompareData | null;
  compare: CompareUI;
  loading: { active: boolean; loaded: number; total: number };
  viewMode: ViewMode;
  dims: 2 | 3;
  morphT: number; // 0 = flat map, 1 = flythrough; drivers ease toward dims
  hover: Selection | null;
  selection: Selection | null;
  mapQuery: MapQuery;
  toggles: Toggles;
  channel: ChannelUI;

  setDatasets(d: DatasetEntry[]): void;
  setDataset(id: string, d: Dataset, opts?: { keepTour?: boolean }): void;
  setCompareData(d: CompareData | null): void;
  setCompareState(i: number): void;
  toggleCompareModel(sourceIdx: number): void;
  setCompareSharedOnly(v: boolean): void;
  setLoading(active: boolean, loaded?: number, total?: number): void;
  setViewMode(m: ViewMode): void;
  setDims(d: 2 | 3): void;
  setMorphT(t: number): void;
  setHover(s: Selection | null): void;
  setSelection(s: Selection | null): void;
  setMapQuery(text: string): void;
  setToggle(key: keyof Toggles, value: boolean): void;
  setChannel(id: string | null, window?: [number, number] | null): void;
  setChannelWindow(window: [number, number] | null): void;
}

export const createAtlasSlice: StateCreator<AppState, [], [], AtlasSlice> = (set, get) => ({
  datasets: [],
  datasetId: null,
  dataset: null,
  compareData: null,
  compare: { state: 1, hiddenModels: [], sharedOnly: false },
  loading: { active: false, loaded: 0, total: 0 },
  viewMode: "atlas",
  dims: 2,
  morphT: 0,
  hover: null,
  selection: null,
  mapQuery: { text: "", results: null },
  toggles: { territories: true, labels: true, beams: true, halos: true, noise: true, legend: true },
  channel: { id: null, window: null },

  setDatasets: (datasets) => set({ datasets }),
  // unit ids are per-model, so a dataset switch clears the cross-view pick too
  setDataset: (datasetId, dataset, opts) =>
    set({
      datasetId,
      dataset,
      hover: null,
      selection: null,
      // match ids are per-dataset row indices — a stale query on a new
      // vocabulary would highlight arbitrary points
      mapQuery: { text: "", results: null },
      interpSelection: null,
      // An episode step that changes model must not clear the episode running
      // it. Every other caller still clears the tour, because a tour's copy
      // names units of the model it was written for.
      ...(opts?.keepTour ? {} : { tour: null }),
      // channel ids AND their ranges are per-model: `we_norm` on gpt2 and on
      // pythia-70m are different numbers in different units, so a window
      // carried across would filter on a scale that no longer exists
      channel: { id: null, window: null },
    }),
  setCompareData: (compareData) => set({ compareData }),
  setCompareState: (state) => set((s) => ({ compare: { ...s.compare, state } })),
  toggleCompareModel: (sourceIdx) =>
    set((s) => ({
      compare: {
        ...s.compare,
        hiddenModels: s.compare.hiddenModels.includes(sourceIdx)
          ? s.compare.hiddenModels.filter((i) => i !== sourceIdx)
          : [...s.compare.hiddenModels, sourceIdx],
      },
    })),
  setCompareSharedOnly: (sharedOnly) => set((s) => ({ compare: { ...s.compare, sharedOnly } })),
  setLoading: (active, loaded = 0, total = 0) => set({ loading: { active, loaded, total } }),
  setViewMode: (viewMode) => set({ viewMode, selection: null, hover: null }),
  // beams/flare are drawn in the 2-D map plane — a dimension switch clears
  // the selection rather than rendering edges at stale coordinates
  // selection survives the dimension flip — beams glide pos2→pos3 with the
  // points; only hover is cleared (its pick frame changes under the cursor)
  setDims: (dims) => set({ dims, hover: null }),
  setMorphT: (morphT) => set({ morphT }),
  setHover: (hover) => set({ hover }),
  setSelection: (selection) => set({ selection }),
  setMapQuery: (text) => {
    const ds = get().dataset;
    const results = ds ? searchLabels(ds.columns.labels, ds.columns.clusterId, text) : null;
    set({ mapQuery: { text, results } });
  },
  setToggle: (key, value) =>
    set((s) => ({ toggles: { ...s.toggles, [key]: value } })),
  // switching channel drops the old window: it was expressed in the previous
  // channel's units, and silently reusing the numbers would filter centroid
  // distances by a norm's bounds
  setChannel: (id, window = null) => set({ channel: { id, window: id ? window : null } }),
  setChannelWindow: (window) => set((s) => ({ channel: { ...s.channel, window } })),
});
