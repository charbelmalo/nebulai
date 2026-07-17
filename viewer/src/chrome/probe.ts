/** Live probing + REAL map builds. The build side talks to the local build
 *  server (`python -m nebulai.backend.build_server`, port 8124), which runs
 *  the actual `nebulai tokens` pipeline as a subprocess — progress here is
 *  the pipeline's own stage output, nothing simulated. */

import { signal } from "@preact/signals";
import { requestRefreshDatasets } from "../app/actions";
import { appStore, type ProbeStage } from "../app/store";

let probeTimer: number | null = null;

/** Fire a single /models probe against the configured endpoint. Updates
 *  `progress.latencyMs` and pushes an event on success or error. */
export async function probeEndpoint(): Promise<void> {
  const st = appStore.getState();
  const { endpoint, apiKey, useBridgeEndpoint } = st.probing;
  const base = useBridgeEndpoint
    ? "http://localhost:8100"
    : (endpoint || "").replace(/\/+$/, "");
  if (!base) {
    st.pushProgressEvent("error", "No endpoint configured");
    st.setProgress({ stage: "error", error: "endpoint empty" });
    return;
  }
  st.pushProgressEvent("probing", `GET ${base}/v1/models`);
  st.setProgress({ stage: "probing", pct: 0.1, error: null });
  const t0 = performance.now();
  try {
    const res = await fetch(`${base}/v1/models`, {
      headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : undefined,
    });
    const latencyMs = performance.now() - t0;
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    st.setProgress({ stage: "done", pct: 1, latencyMs, message: `${res.status} OK` });
    st.pushProgressEvent("done", `probe ok in ${latencyMs.toFixed(0)}ms`);
  } catch (e) {
    const latencyMs = performance.now() - t0;
    const msg = e instanceof Error ? e.message : String(e);
    st.setProgress({ stage: "error", latencyMs, error: msg });
    st.pushProgressEvent("error", `probe failed: ${msg}`);
  }
}

// ── build server client ─────────────────────────────────────────────────────

/** One curated (or previously built) model as reported by /build/models. */
export interface BuildModelInfo {
  id: string;
  label: string;
  interp: boolean;
  built: boolean;
  /** reduced.params.json contents when a UMAP cache exists — the fast
   *  re-cluster path is available when the current params match this. */
  cached_reduce: Record<string, unknown> | null;
}

export interface BuildModelsPayload {
  models: BuildModelInfo[];
  namers: string[];
  sources: { id: string; label: string }[];
}

/** null until the first successful /build/models fetch. */
export const $buildModels = signal<BuildModelsPayload | null>(null);
export const $buildHealth = signal<"unknown" | "ok" | "down">("unknown");

function buildBase(): string {
  return appStore.getState().probing.buildUrl.replace(/\/+$/, "");
}

export async function fetchBuildModels(): Promise<void> {
  try {
    const res = await fetch(`${buildBase()}/build/models`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    $buildModels.value = (await res.json()) as BuildModelsPayload;
    $buildHealth.value = "ok";
  } catch {
    $buildModels.value = null;
    $buildHealth.value = "down";
  }
}

/** Store BuildParams → /build/start params. 0-valued counts are omitted so
 *  the CLI defaults apply (full vocab / auto cluster sizing). */
function serverParams(): Record<string, unknown> {
  const { buildParams: bp, buildSource } = appStore.getState().probing;
  const p: Record<string, unknown> = {
    n_neighbors: bp.nNeighbors,
    seed: bp.seed,
    cluster_method: bp.clusterMethod,
    namer: bp.namer,
    edges: bp.edges,
  };
  if (bp.maxTokens > 0) p.max_tokens = bp.maxTokens;
  if (bp.minClusterSize > 0) p.min_cluster_size = bp.minClusterSize;
  if (bp.minSamples > 0) p.min_samples = bp.minSamples;
  if (bp.force) p.force = true;
  if (buildSource === "api") {
    p.embed_host = bp.embedHost;
    p.embed_model = bp.embedModel;
    p.embed_api = bp.embedApi;
  }
  return p;
}

/** Do the current params hit the model's UMAP cache? (= the seconds-fast
 *  re-cluster teaching loop instead of a minutes-long reduce). Mirrors the
 *  reduce_params dict in cli.py; api builds aren't reported by /build/models,
 *  so they never claim the fast path. */
export function cacheMatches(m: BuildModelInfo | undefined): boolean {
  const { buildModel, buildSource, buildParams: bp } = appStore.getState().probing;
  const c = m?.cached_reduce;
  if (!c || buildSource !== "hf") return false;
  return (
    c.model === buildModel &&
    (c.max_tokens ?? null) === (bp.maxTokens > 0 ? bp.maxTokens : null) &&
    c.center === true &&
    c.cluster_dim === 10 &&
    c.n_neighbors === bp.nNeighbors &&
    c.seed === bp.seed
  );
}

let pollTimer: number | null = null;
let lastStage: string | null = null;

function stopPolling(): void {
  if (pollTimer != null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function startPolling(): void {
  stopPolling();
  lastStage = null;
  pollTimer = window.setInterval(() => void pollBuildStatus(), 1000);
}

function fmtElapsed(s: number): string {
  const m = Math.floor(s / 60);
  return m > 0 ? `${m}m ${Math.floor(s % 60)}s` : `${Math.floor(s)}s`;
}

export async function pollBuildStatus(): Promise<void> {
  const st = appStore.getState();
  try {
    const res = await fetch(`${buildBase()}/build/status`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const s = await res.json();
    const stage = (s.stage ?? "idle") as ProbeStage;
    st.setProgress({
      stage,
      pct: s.pct ?? 0,
      message: s.running ? `${s.message || "…"} — ${fmtElapsed(s.elapsed_s ?? 0)} elapsed` : s.message || "",
      error: s.error ?? null,
    });
    if (stage !== lastStage) {
      st.pushProgressEvent(stage, s.message || stage);
      lastStage = stage;
    }
    if (!s.running) {
      stopPolling();
      void fetchBuildModels(); // built/cached flags just changed
      if (s.done && s.dataset_id) {
        st.pushProgressEvent("rendering", `hot-swapping to ${s.dataset_id}`);
        requestRefreshDatasets(s.dataset_id);
      }
    }
  } catch (e) {
    stopPolling();
    const msg = e instanceof Error ? e.message : String(e);
    st.setProgress({ stage: "error", error: `lost build server: ${msg}` });
    st.pushProgressEvent("error", `status poll failed: ${msg}`);
    $buildHealth.value = "down";
  }
}

/** POST /build/start with the configured model/source/params, then poll.
 *  `force` overrides the stored force flag (the re-cluster button passes
 *  false so a matching UMAP cache is reused). */
export async function startBuild(opts: { force?: boolean } = {}): Promise<void> {
  const st = appStore.getState();
  const { buildModel, buildSource } = st.probing;
  if (!buildModel.trim()) {
    st.setProgress({ stage: "error", error: "no model selected" });
    return;
  }
  const params = serverParams();
  if (opts.force !== undefined) {
    if (opts.force) params.force = true;
    else delete params.force;
  }
  st.resetProgress();
  st.pushProgressEvent("probing", `POST /build/start ${buildModel} (${buildSource})`);
  st.setProgress({ stage: "probing", pct: 0.02, message: `starting ${buildModel}…` });
  try {
    const res = await fetch(`${buildBase()}/build/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: buildModel, source: buildSource, params }),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.error ?? `HTTP ${res.status}`);
    startPolling();
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    st.setProgress({ stage: "error", error: msg });
    st.pushProgressEvent("error", `build start failed: ${msg}`);
  }
}

export async function cancelBuild(): Promise<void> {
  const st = appStore.getState();
  try {
    await fetch(`${buildBase()}/build/cancel`, { method: "POST" });
    st.pushProgressEvent("idle", "cancel requested");
    // the poll loop observes the server flip to idle/"cancelled"
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    st.pushProgressEvent("error", `cancel failed: ${msg}`);
  }
}

/** Optional recurring live-probe based on `probing.liveProbe`. Called from
 *  main.ts on config change; safe to call repeatedly. */
export function scheduleLiveProbe(): void {
  if (probeTimer != null) {
    clearInterval(probeTimer);
    probeTimer = null;
  }
  const p = appStore.getState().probing;
  if (!p.liveProbe) return;
  probeTimer = window.setInterval(probeEndpoint, Math.max(5000, p.probeIntervalMs));
}
