/** Shared app state — the coordination point between drivers, overlays, and
 *  chrome. Toolkits never share GPU contexts; they share THIS. Camera matrices
 *  land here in M1 so SVG/HTML overlays project through the same numbers the
 *  active driver renders with. */

import { createStore } from "zustand/vanilla";
import type { Capabilities } from "./capabilities";
import type { CompareData } from "../data/compare";
import type { Dataset } from "../data/loader";
import type { DatasetEntry } from "../data/schema";

export type ViewMode = "atlas" | "chord" | "hierarchy" | "compare";

export interface Selection {
  kind: "cluster" | "point";
  id: number;
}

export interface Toggles {
  territories: boolean;
  labels: boolean;
  beams: boolean;
  noise: boolean;
  legend: boolean;
}

/** "Additional" tab knobs — render-quality settings, all live-applied. */
export interface Settings {
  pointScale: number; // × base point size, 0.5–2
  bloom: boolean; // post bloom (webgpu tier only; forced off by reduced motion)
  confidenceFloor: number; // hide points below this confidence, 0–1
}

/** Compare-view UI state. `hiddenModels` holds source indices toggled off in
 *  the legend; `state` indexes CompareData.states (default 1 = semantic). */
export interface CompareUI {
  state: number;
  hiddenModels: number[];
  sharedOnly: boolean;
}

export interface AppState {
  capabilities: Capabilities | null;
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
  toggles: Toggles;
  settings: Settings;

  setCapabilities(c: Capabilities): void;
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
  setToggle(key: keyof Toggles, value: boolean): void;
  setSetting<K extends keyof Settings>(key: K, value: Settings[K]): void;
}

export const appStore = createStore<AppState>()((set) => ({
  capabilities: null,
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
  toggles: { territories: true, labels: true, beams: true, noise: true, legend: true },
  settings: { pointScale: 1, bloom: true, confidenceFloor: 0 },

  setCapabilities: (capabilities) => set({ capabilities }),
  setDatasets: (datasets) => set({ datasets }),
  setDataset: (datasetId, dataset) => set({ datasetId, dataset, hover: null, selection: null }),
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
  setDims: (dims) => set({ dims, selection: null, hover: null }),
  setMorphT: (morphT) => set({ morphT }),
  setHover: (hover) => set({ hover }),
  setSelection: (selection) => set({ selection }),
  setToggle: (key, value) =>
    set((s) => ({ toggles: { ...s.toggles, [key]: value } })),
  setSetting: (key, value) =>
    set((s) => ({ settings: { ...s.settings, [key]: value } })),
}));
