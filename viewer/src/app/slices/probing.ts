/** probing.ts — the endpoints Nebulai talks to and the live pipeline progress
 *  it reports back. Config (`probing`) and progress (`progress`) sit together
 *  because they are two halves of one story: you point the app at a model
 *  server, and the stages the backend reports come straight back through the
 *  same Settings page.
 *
 *  EVERY ENDPOINT HERE IS NEBULAI'S. `liveUrl` (:8123) is the Internals live
 *  probe server and `buildUrl` (:8124) is the map build server. The one
 *  Seer-owned endpoint that used to sit alongside them — the SessionSeer
 *  capture server on :8125 — moved out to `slices/seer.ts` when Seer got its
 *  own entry point, so a Seer document composes Seer's endpoint without also
 *  inheriting a build server it has nothing to build with. `VITE_SEER_URL`
 *  still names it; only the store path changed
 *  (`probing.seerUrl` → `seer.serverUrl`). */

import type { StateCreator } from "zustand";
import type { AppState } from "../store";

/** Map-builder parameters — mirrors the build server's /build/start params
 *  (which mirror the `nebulai tokens` CLI flags). 0 means "auto"/"full" for
 *  the counts; the server omits those flags so the CLI default applies. */
export interface BuildParams {
  maxTokens: number; // 0 = full curated vocab
  nNeighbors: number; // UMAP n_neighbors
  seed: number; // UMAP seed
  minClusterSize: number; // 0 = auto (HDBSCAN default heuristic)
  minSamples: number; // 0 = auto
  clusterMethod: "leaf" | "eom";
  namer: "auto" | "ollama" | "openai" | "openrouter" | "anthropic" | "none";
  edges: "knn" | "cluster" | "none";
  force: boolean; // recompute cached UMAP reductions
  embedHost: string; // [source=api|probe] embeddings endpoint base URL
  embedModel: string; // [source=api|probe] embedding model name
  embedApi: "ollama" | "openai"; // [source=api|probe] transport
  // an OpenAI-compatible chat server: names clusters for every source, and
  // GENERATES the concepts for a probe
  llmHost: string;
  llmModel: string; // "" = first chat model the server lists
  // [source=probe] only — there is no model, so these define the whole cloud
  probeSeed: string; // the word the cloud grows from
  depth: number; // BFS expansion depth
  breadth: number; // concepts proposed per term
  sensitivity: number; // cosine floor against the seed (0..1)
  generator: "auto" | "ollama" | "openai" | "openrouter" | "anthropic";
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
  useBridgeEndpoint: boolean; // route naming/embedding through a custom bridge endpoint
  liveUrl: string; // Internals #25 live probe server (nebulai live_server)
  buildUrl: string; // map build server (nebulai build_server)
  buildModel: string; // HF model id to build (curated pick or custom)
  // W_E rows · third-party text embeddings · no model at all (LLM-grown probe)
  buildSource: "hf" | "api" | "probe";
  buildParams: BuildParams;
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

export interface ProbingSlice {
  probing: Probing;
  progress: Progress;

  setProbing<K extends keyof Probing>(key: K, value: Probing[K]): void;
  setBuildParam<K extends keyof BuildParams>(key: K, value: BuildParams[K]): void;
  setProgress(patch: Partial<Progress>): void;
  pushProgressEvent(stage: ProbeStage, message: string): void;
  resetProgress(): void;
}

export const createProbingSlice: StateCreator<AppState, [], [], ProbingSlice> = (set) => ({
  probing: {
    endpoint: "",
    apiKey: "",
    model: "",
    liveProbe: false,
    probeIntervalMs: 15000,
    autoRun: false,
    useBridgeEndpoint: false,
    // Bridge endpoints (live Internals driver, on-demand build server, probe)
    // are env-driven so one codebase serves both dev and a static deploy:
    //   • dev / local: default to loopback so a locally-run nebulai server
    //     (live_server on :8123, build_server on :8124) is picked up with no
    //     config.
    //   • static deploy: the build passes empty VITE_* vars (see
    //     docs/DEPLOY-STATIC.md) so these blank out and the features become
    //     bring-your-own-endpoint — a visitor pastes their own URL in Settings
    //     and nothing is contacted until they do.
    liveUrl: import.meta.env.VITE_LIVE_URL ?? "http://127.0.0.1:8123",
    buildUrl: import.meta.env.VITE_BUILD_URL ?? "http://127.0.0.1:8124",
    buildModel: "gpt2",
    buildSource: "hf",
    buildParams: {
      maxTokens: 0,
      nNeighbors: 30,
      seed: 42,
      minClusterSize: 0,
      minSamples: 0,
      clusterMethod: "leaf",
      namer: "auto",
      edges: "knn",
      force: false,
      embedHost: import.meta.env.VITE_EMBED_HOST ?? "http://localhost:11434",
      embedModel: "mxbai-embed-large",
      embedApi: "ollama",
      llmHost: import.meta.env.VITE_LLM_HOST ?? "http://localhost:8050",
      llmModel: "",
      probeSeed: "",
      depth: 2,
      breadth: 12,
      sensitivity: 0.35,
      generator: "auto",
    },
  },
  progress: {
    stage: "idle",
    pct: 0,
    message: "",
    latencyMs: null,
    history: [],
    error: null,
  },

  setProbing: (key, value) =>
    set((s) => ({ probing: { ...s.probing, [key]: value } })),
  setBuildParam: (key, value) =>
    set((s) => ({
      probing: {
        ...s.probing,
        buildParams: { ...s.probing.buildParams, [key]: value },
      },
    })),
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
});
