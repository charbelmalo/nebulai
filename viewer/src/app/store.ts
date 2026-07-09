/** Shared app state — the coordination point between drivers, overlays, and
 *  chrome. Toolkits never share GPU contexts; they share THIS. Camera matrices
 *  land here in M1 so SVG/HTML overlays project through the same numbers the
 *  active driver renders with. */

import { createStore } from "zustand/vanilla";
import type { Capabilities } from "./capabilities";
import type { CompareData } from "../data/compare";
import type { Dataset } from "../data/loader";
import type { DatasetEntry } from "../data/schema";
import type { SessionAnalysis } from "../chrome/sessionlog";

export type ViewMode = "atlas" | "chord" | "hierarchy" | "compare";

/** Top-level page — nav-bar controlled. `map` is the semantic cloud (all the
 *  driver-backed views); `snapshot` is the per-topic conversation-log map;
 *  `interp` is the Internals gallery (mechanistic-interpretability drivers, each
 *  rendering one real computed quantity from an interp bundle); `guide`
 *  documents the exact math + source data behind every live feature. */
export type Page = "map" | "snapshot" | "interp" | "guide" | "sessions";

/** Internals-page UI state. `featureId` selects which InterpDriver owns the
 *  interp canvas (must match a registered feature id in scene/interp/registry).
 *  The active model is read from `datasetId` — the bundles live per-model.
 *  `traceSlug` selects which per-prompt forward trace the forward-group features
 *  render (weight-group features ignore it). "" means "use the first trace". */
export interface InterpUI {
  featureId: string;
  traceSlug: string;
}

/** One saved topic filter — a named bag of keywords the snapshot map watches
 *  for in conversation logs. Ships with a couple of defaults (design,
 *  shaders). Users add more from either the Snapshot Map page or Settings. */
export interface TopicPreset {
  id: string;
  name: string;
  keywords: string[];
}

/** A parsed conversation log — a sequence of turns with role + text. The
 *  snapshot pipeline runs entirely client-side; the raw log is kept in memory
 *  only. */
export interface ConversationTurn {
  role: string;
  text: string;
  ts?: number;
}

export interface SnapshotLog {
  id: string;
  name: string;
  turns: ConversationTurn[];
  loadedAt: number;
}

export interface SnapshotState {
  logs: SnapshotLog[];
  activeLogId: string | null;
  topics: TopicPreset[];
  activeTopicId: string;
  turnIndex: number; // 0 = first turn, logs.turns.length-1 = last
  playing: boolean;
}

/** Sessions-page state — analysed agent-mode session transcripts (rich, real
 *  quantities), which the 3-D plotter renders as trajectories. `analyses` are
 *  DERIVED summaries (never raw text); they persist to IndexedDB across app
 *  sessions via `chrome/sessionStore.ts` and rehydrate on boot. `activeIds`
 *  selects which sessions are overlaid on the plot. */
export interface SessionsState {
  analyses: SessionAnalysis[];
  activeIds: string[];
  hydrated: boolean; // true once the IndexedDB rehydrate pass has run
}

export interface Selection {
  kind: "cluster" | "point";
  id: number;
}

export interface Toggles {
  territories: boolean;
  labels: boolean;
  beams: boolean;
  halos: boolean; // pulsing hub rings (the "radial bubbles")
  noise: boolean;
  legend: boolean;
}

/** Global render-quality settings — live-applied. Per-view appearance knobs
 *  live in `appearance` below; general chrome/render knobs live here. */
export interface Settings {
  pointScale: number; // × base point size, 0.5–2
  bloom: boolean; // post bloom (webgpu tier only; forced off by reduced motion)
  confidenceFloor: number; // hide points below this confidence, 0–1
  theme: "dark" | "light" | "auto";
  labelDensity: number; // 0.2–2 — culling threshold multiplier for cluster labels
  animationSpeed: number; // 0.25–2 — global time-uniform multiplier
  reducedMotion: boolean; // manual override; caps.reducedMotion still wins on init
}

/** Per-view appearance settings. Every graph type gets its own tab in the
 *  Settings page; every knob a driver honors MUST live here — this is the
 *  single source of truth for graph looks. New driver knobs belong in the
 *  matching sub-interface, not the driver's private state. */
export interface Appearance {
  atlas: {
    hullOpacity: number; // 0–1 — territory hull fill alpha
    beamWidth: number; // 0.25–3 — connection line width scale
    haloIntensity: number; // 0–1 — pulsing hub ring strength
    background: "vignette" | "flat" | "grid"; // stage background
    orbitEnabled: boolean; // slow camera orbit in 3D flythrough
    orbitSpeed: number; // 0.1–3
  };
  chord: {
    ribbonOpacity: number; // 0–1
    curveTension: number; // 0–1 (0 = straight, 1 = maximum bezier)
    labelRotation: boolean; // rotate rim labels tangentially
    showTicks: boolean;
  };
  hierarchy: {
    linkStroke: number; // 0.5–3
    nodeSize: number; // 0.5–3
    fanAngle: number; // 60–360 — arc span in degrees
    colorBy: "cluster" | "depth" | "confidence";
  };
  compare: {
    swatchSize: number; // 4–20 px
    strokeOnHover: boolean;
    dimOthers: boolean; // when a model is highlighted, dim the rest
  };
}

/** Model probing config — live probing tests a model endpoint before it's
 *  used for cluster naming or embedding. Progress state is separate and
 *  transient (mirrors the pipeline stages the backend actually reports). */
export interface Probing {
  endpoint: string; // custom OpenAI-compatible base URL, empty = default chain
  apiKey: string; // never persisted, kept in memory only
  model: string; // e.g. "llama3.2:3b" or "gpt-4o-mini"
  liveProbe: boolean; // ping /models on config change
  probeIntervalMs: number; // 5000–60000 — recurring health check
  autoRun: boolean; // rebuild the map when config changes
  useM4Worker: boolean; // route through 192.168.0.200 (m4worker-bridge)
}

export type ProbeStage =
  | "idle"
  | "probing"
  | "loading"
  | "reducing"
  | "clustering"
  | "naming"
  | "exporting"
  | "rendering"
  | "done"
  | "error";

/** Live progress — populated by the probe/build pipeline. Progress bars in
 *  the Settings page bind directly to these fields. */
export interface Progress {
  stage: ProbeStage;
  pct: number; // 0–1
  message: string; // human-readable status line
  latencyMs: number | null; // last successful probe RTT
  history: { id: number; t: number; stage: ProbeStage; message: string }[]; // event log
  error: string | null;
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
  appearance: Appearance;
  probing: Probing;
  progress: Progress;
  settingsOpen: boolean; // Settings page overlay visibility
  page: Page;
  snapshot: SnapshotState;
  sessions: SessionsState;
  interp: InterpUI;

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
  setAppearance<G extends keyof Appearance, K extends keyof Appearance[G]>(
    graph: G,
    key: K,
    value: Appearance[G][K],
  ): void;
  setProbing<K extends keyof Probing>(key: K, value: Probing[K]): void;
  setProgress(patch: Partial<Progress>): void;
  pushProgressEvent(stage: ProbeStage, message: string): void;
  resetProgress(): void;
  setSettingsOpen(open: boolean): void;
  setPage(p: Page): void;
  setInterpFeature(id: string): void;
  setInterpTrace(slug: string): void;
  addSnapshotLog(log: SnapshotLog): void;
  removeSnapshotLog(id: string): void;
  setActiveLog(id: string | null): void;
  setActiveTopic(id: string): void;
  setTurnIndex(i: number): void;
  setPlaying(v: boolean): void;
  addTopicPreset(t: TopicPreset): void;
  updateTopicPreset(id: string, patch: Partial<TopicPreset>): void;
  removeTopicPreset(id: string): void;
  setSessionAnalyses(list: SessionAnalysis[]): void;
  addSessionAnalysis(a: SessionAnalysis): void;
  removeSessionAnalysis(id: string): void;
  toggleSessionActive(id: string): void;
  clearSessionAnalyses(): void;
  setSessionsHydrated(v: boolean): void;
}

/** Preset topic filters shipped by default. Users can add more from the
 *  Snapshot Map page or the Settings → Snapshot tab. Keep lists tight — the
 *  match is case-insensitive substring; long lists dilute the signal. */
export const DEFAULT_TOPICS: TopicPreset[] = [
  {
    id: "design",
    name: "Design keywords",
    keywords: [
      "typography",
      "spacing",
      "radius",
      "grid",
      "tokens",
      "layout",
      "hover",
      "focus",
      "empty state",
      "component",
      "hierarchy",
      "affordance",
      "accessibility",
      "contrast",
      "keyboard",
    ],
  },
  {
    id: "shaders",
    name: "Shader effects",
    keywords: [
      "bloom",
      "vignette",
      "chromatic aberration",
      "SSAO",
      "godrays",
      "fresnel",
      "raymarch",
      "SDF",
      "post-processing",
      "TSL",
      "WGSL",
      "GLSL",
      "compute",
      "uniform",
      "vertex",
      "fragment",
    ],
  },
  {
    id: "interaction",
    name: "Interaction craft",
    keywords: [
      "hit target",
      "safe triangle",
      "aria-activedescendant",
      "focus trap",
      "escape",
      "arrow keys",
      "roving tabindex",
      "scroll padding",
      "submenu",
      "combobox",
      "tooltip",
      "dropdown",
    ],
  },
];

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
  toggles: { territories: true, labels: true, beams: true, halos: true, noise: true, legend: true },
  settings: {
    pointScale: 1,
    bloom: true,
    confidenceFloor: 0,
    theme: "dark",
    labelDensity: 1,
    animationSpeed: 1,
    reducedMotion: false,
  },
  appearance: {
    atlas: {
      hullOpacity: 0.28,
      beamWidth: 1,
      haloIntensity: 0.7,
      background: "vignette",
      orbitEnabled: false,
      orbitSpeed: 1,
    },
    chord: {
      ribbonOpacity: 0.55,
      curveTension: 0.6,
      labelRotation: true,
      showTicks: true,
    },
    hierarchy: {
      linkStroke: 1,
      nodeSize: 1,
      fanAngle: 300,
      colorBy: "cluster",
    },
    compare: {
      swatchSize: 10,
      strokeOnHover: true,
      dimOthers: true,
    },
  },
  probing: {
    endpoint: "",
    apiKey: "",
    model: "",
    liveProbe: false,
    probeIntervalMs: 15000,
    autoRun: false,
    useM4Worker: false,
  },
  progress: {
    stage: "idle",
    pct: 0,
    message: "",
    latencyMs: null,
    history: [],
    error: null,
  },
  settingsOpen: false,
  page: "map",
  snapshot: {
    logs: [],
    activeLogId: null,
    topics: DEFAULT_TOPICS,
    activeTopicId: DEFAULT_TOPICS[0]?.id ?? "",
    turnIndex: 0,
    playing: false,
  },
  sessions: { analyses: [], activeIds: [], hydrated: false },
  interp: { featureId: "weight-spectrum", traceSlug: "" },

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
  setAppearance: (graph, key, value) =>
    set((s) => ({
      appearance: {
        ...s.appearance,
        [graph]: { ...s.appearance[graph], [key]: value },
      },
    })),
  setProbing: (key, value) =>
    set((s) => ({ probing: { ...s.probing, [key]: value } })),
  setProgress: (patch) => set((s) => ({ progress: { ...s.progress, ...patch } })),
  pushProgressEvent: (stage, message) =>
    set((s) => {
      const last = s.progress.history[s.progress.history.length - 1];
      const id = (last?.id ?? 0) + 1;
      return {
        progress: {
          ...s.progress,
          stage,
          message,
          history: [...s.progress.history.slice(-49), { id, t: Date.now(), stage, message }],
        },
      };
    }),
  resetProgress: () =>
    set({
      progress: { stage: "idle", pct: 0, message: "", latencyMs: null, history: [], error: null },
    }),
  setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
  setPage: (page) => set({ page }),
  setInterpFeature: (featureId) =>
    set((s) => ({ interp: { ...s.interp, featureId } })),
  setInterpTrace: (traceSlug) =>
    set((s) => ({ interp: { ...s.interp, traceSlug } })),
  addSnapshotLog: (log) =>
    set((s) => ({
      snapshot: {
        ...s.snapshot,
        logs: [...s.snapshot.logs, log],
        activeLogId: log.id,
        turnIndex: Math.max(0, log.turns.length - 1),
      },
    })),
  removeSnapshotLog: (id) =>
    set((s) => {
      const logs = s.snapshot.logs.filter((l) => l.id !== id);
      const activeLogId =
        s.snapshot.activeLogId === id ? (logs[0]?.id ?? null) : s.snapshot.activeLogId;
      return { snapshot: { ...s.snapshot, logs, activeLogId, turnIndex: 0 } };
    }),
  setActiveLog: (id) =>
    set((s) => ({ snapshot: { ...s.snapshot, activeLogId: id, turnIndex: 0 } })),
  setActiveTopic: (activeTopicId) =>
    set((s) => ({ snapshot: { ...s.snapshot, activeTopicId } })),
  setTurnIndex: (turnIndex) =>
    set((s) => ({ snapshot: { ...s.snapshot, turnIndex: Math.max(0, turnIndex) } })),
  setPlaying: (playing) => set((s) => ({ snapshot: { ...s.snapshot, playing } })),
  addTopicPreset: (t) =>
    set((s) => ({ snapshot: { ...s.snapshot, topics: [...s.snapshot.topics, t] } })),
  updateTopicPreset: (id, patch) =>
    set((s) => ({
      snapshot: {
        ...s.snapshot,
        topics: s.snapshot.topics.map((t) => (t.id === id ? { ...t, ...patch } : t)),
      },
    })),
  removeTopicPreset: (id) =>
    set((s) => {
      const topics = s.snapshot.topics.filter((t) => t.id !== id);
      const activeTopicId =
        s.snapshot.activeTopicId === id
          ? (topics[0]?.id ?? "")
          : s.snapshot.activeTopicId;
      return { snapshot: { ...s.snapshot, topics, activeTopicId } };
    }),
  // ── sessions (3-D plotter) ───────────────────────────────────────────────
  setSessionAnalyses: (list) =>
    set((s) => ({
      sessions: {
        ...s.sessions,
        analyses: list,
        // keep any still-present active ids; default to showing the newest one
        activeIds: (() => {
          const ids = new Set(list.map((a) => a.id));
          const kept = s.sessions.activeIds.filter((id) => ids.has(id));
          if (kept.length) return kept;
          const first = list[0]?.id;
          return first ? [first] : [];
        })(),
        hydrated: true,
      },
    })),
  addSessionAnalysis: (a) =>
    set((s) => {
      // de-dup by id; newest first so the list reads most-recent-on-top
      const analyses = [a, ...s.sessions.analyses.filter((x) => x.id !== a.id)];
      return {
        sessions: {
          ...s.sessions,
          analyses,
          activeIds: [a.id, ...s.sessions.activeIds.filter((id) => id !== a.id)],
        },
      };
    }),
  removeSessionAnalysis: (id) =>
    set((s) => ({
      sessions: {
        ...s.sessions,
        analyses: s.sessions.analyses.filter((a) => a.id !== id),
        activeIds: s.sessions.activeIds.filter((x) => x !== id),
      },
    })),
  toggleSessionActive: (id) =>
    set((s) => ({
      sessions: {
        ...s.sessions,
        activeIds: s.sessions.activeIds.includes(id)
          ? s.sessions.activeIds.filter((x) => x !== id)
          : [...s.sessions.activeIds, id],
      },
    })),
  clearSessionAnalyses: () =>
    set((s) => ({ sessions: { ...s.sessions, analyses: [], activeIds: [] } })),
  setSessionsHydrated: (hydrated) =>
    set((s) => ({ sessions: { ...s.sessions, hydrated } })),
}));
