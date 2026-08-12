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

  setDatasets(d: DatasetEntry[]): void;
  setDataset(id: string, d: Dataset): void;
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

  setDatasets: (datasets) => set({ datasets }),
  // unit ids are per-model, so a dataset switch clears the cross-view pick too
  setDataset: (datasetId, dataset) =>
    set({
      datasetId,
      dataset,
      hover: null,
      selection: null,
      // match ids are per-dataset row indices — a stale query on a new
      // vocabulary would highlight arbitrary points
      mapQuery: { text: "", results: null },
      interpSelection: null,
      tour: null,
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
});
