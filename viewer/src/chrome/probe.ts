/** Live probing + simulated map-build progress. The real backend runs in
 *  Python; this module drives the viewer-side state so the progress strip
 *  animates whether it's talking to a real endpoint or a mock schedule. */

import { appStore, type ProbeStage } from "../app/store";

let probeTimer: number | null = null;
let buildTimer: number | null = null;

/** Fire a single /models probe against the configured endpoint. Updates
 *  `progress.latencyMs` and pushes an event on success or error. */
export async function probeEndpoint(): Promise<void> {
  const st = appStore.getState();
  const { endpoint, apiKey, useM4Worker } = st.probing;
  const base = useM4Worker
    ? "http://192.168.0.200:8100"
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

/** Schedule of stages the map-build steps through. Timings are illustrative;
 *  a real pipeline replaces this with backend events. */
const BUILD_SCHEDULE: { stage: ProbeStage; ms: number; msg: string }[] = [
  { stage: "probing", ms: 400, msg: "handshake…" },
  { stage: "loading", ms: 800, msg: "streaming units" },
  { stage: "reducing", ms: 1600, msg: "UMAP 10-D → 3-D → 2-D" },
  { stage: "clustering", ms: 900, msg: "HDBSCAN leaf selection" },
  { stage: "naming", ms: 1500, msg: "auto-namer chain" },
  { stage: "exporting", ms: 500, msg: "writing nebulai.json" },
  { stage: "rendering", ms: 600, msg: "drawing atlas" },
  { stage: "done", ms: 0, msg: "map ready" },
];

export function startBuildProbe(): void {
  cancelBuildProbe();
  const st = appStore.getState();
  st.resetProgress();
  st.pushProgressEvent("probing", "starting build");
  let elapsed = 0;
  const total = BUILD_SCHEDULE.reduce((a, s) => a + s.ms, 0);
  let idx = 0;

  const tick = () => {
    if (idx >= BUILD_SCHEDULE.length) {
      buildTimer = null;
      return;
    }
    const step = BUILD_SCHEDULE[idx];
    if (!step) {
      buildTimer = null;
      return;
    }
    appStore.getState().setProgress({
      stage: step.stage,
      message: step.msg,
      pct: Math.min(1, elapsed / Math.max(1, total)),
    });
    appStore.getState().pushProgressEvent(step.stage, step.msg);
    elapsed += step.ms;
    idx++;
    buildTimer = window.setTimeout(tick, step.ms);
  };
  tick();
}

export function cancelBuildProbe(): void {
  if (buildTimer != null) {
    clearTimeout(buildTimer);
    buildTimer = null;
    const st = appStore.getState();
    if (st.progress.stage !== "done") {
      st.setProgress({ stage: "idle", message: "cancelled" });
      st.pushProgressEvent("idle", "cancelled");
    }
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
