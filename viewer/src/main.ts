/** Boot: probe GPU tier → discover datasets → load through the worker →
 *  hand the canvas to the AtlasDriver and run the frame loop. The status pill
 *  doubles as the MetaLine — dataset provenance stays visible in every mode. */

import "./styles/tokens.css";
import "./styles/craft-tokens.css";
import "./styles/chrome.css";

import { registerActions } from "./app/actions";
import { probeCapabilities } from "./app/capabilities";
import { appStore, type ViewMode } from "./app/store";
import { mountChrome } from "./chrome/mount";
import { loadCompare } from "./data/compare";
import { loadDataset, loadIndex } from "./data/loader";
import { AtlasDriver } from "./scene/drivers/AtlasDriver";
import { ChordDriver } from "./scene/drivers/ChordDriver";
import { CompareDriver } from "./scene/drivers/CompareDriver";
import { HierarchyDriver } from "./scene/drivers/HierarchyDriver";

declare global {
  interface Window {
    __perf: { parseMs?: number; bootMs?: number; p95FrameMs?: number };
    __store: typeof appStore;
    __driver?: AtlasDriver;
    __compareDriver?: CompareDriver;
    __chordDriver?: ChordDriver;
    __hierDriver?: HierarchyDriver;
  }
}

window.__perf = {};
window.__store = appStore; // e2e tests read state through this

const chrome = document.getElementById("chrome")!;
const progress = document.createElement("div");
progress.className = "boot-progress";
const status = document.createElement("div");
status.className = "boot-status";
chrome.append(progress, status);

function say(text: string) {
  status.textContent = text;
}

/** The honesty line: dataset provenance stays visible in every mode. */
export function metaLine(): string {
  const { datasetId, dataset, viewMode, compareData } = appStore.getState();
  if (viewMode === "compare" && compareData) {
    const m = compareData.meta;
    return [
      `compare: ${m.models.length} models`,
      `${m.n_points} cluster concepts`,
      `${compareData.stats.n_shared_concepts} shared`,
      `embed: ${m.embed_model} (label space, not model geometry)`,
    ].join(" · ");
  }
  if (!datasetId || !dataset) return "no dataset";
  const m = dataset.columns.meta;
  const e = dataset.columns.edges;
  const parts = [
    m.model ?? datasetId,
    m.unit,
    `${m.n_points.toLocaleString("en-US")} pts`,
    `${m.n_clusters} clusters`,
    `${(m.noise_fraction * 100).toFixed(1)}% noise`,
    `namer: ${m.namer}`,
    e ? `edges: ${e.metric}@${e.space}` : "edges: none (v1 export)",
  ];
  return parts.filter(Boolean).join(" · ");
}

async function boot() {
  const t0 = performance.now();
  const caps = await probeCapabilities();
  appStore.getState().setCapabilities(caps);
  say(`gpu: ${caps.tier} — loading datasets…`);

  const index = await loadIndex();
  appStore.getState().setDatasets(index.datasets);
  const first = index.datasets[0];
  if (!first) {
    say("no datasets in out/index.json — run `uv run nebulai tokens` first");
    return;
  }

  appStore.getState().setLoading(true);
  const ds = await loadDataset(first.path, (loaded, total) => {
    appStore.getState().setLoading(true, loaded, total);
    progress.style.width = `${((loaded / total) * 100).toFixed(1)}%`;
    say(`${first.id} — ${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`);
  });
  appStore.getState().setDataset(first.id, ds);
  appStore.getState().setLoading(false);

  window.__perf.parseMs = ds.parseMs;
  progress.style.width = "100%";
  progress.classList.add("is-done");

  if (caps.tier === "static") {
    say(`${metaLine()} · gpu: static (no WebGPU/WebGL — static fallback lands in M4)`);
    return;
  }

  const canvas = document.getElementById("scene-canvas") as HTMLCanvasElement;
  const driver = new AtlasDriver();
  await driver.init(canvas, caps.tier);
  window.__driver = driver; // e2e + debugging handle

  // view-manager state — declared before applySize so the resize handler can
  // see the compare driver once it exists
  const FADE_MS = caps.reducedMotion ? 150 : 300;
  let compareDriver: CompareDriver | null = null;
  let compareCanvas: HTMLCanvasElement | null = null;
  let chordDriver: ChordDriver | null = null;
  let chordCanvas: HTMLCanvasElement | null = null;
  let hierDriver: HierarchyDriver | null = null;
  let hierCanvas: HTMLCanvasElement | null = null;
  let activeMode: ViewMode = "atlas";
  let fadeUntil = 0;

  const stage = document.getElementById("stage")!;
  const applySize = () => {
    const dpr = window.devicePixelRatio || 1;
    driver.resize(stage.clientWidth, stage.clientHeight, dpr);
    compareDriver?.resize(stage.clientWidth, stage.clientHeight, dpr);
    chordDriver?.resize(stage.clientWidth, stage.clientHeight, dpr);
    hierDriver?.resize(stage.clientWidth, stage.clientHeight, dpr);
  };
  applySize();
  new ResizeObserver(applySize).observe(stage);

  driver.setDataset(ds);
  say(`${metaLine()} · gpu: ${caps.tier}`);

  // compare.json is optional (run `nebulai compare`); discovery is
  // non-blocking so the atlas never waits on it
  if (caps.tier === "webgpu") {
    loadCompare()
      .then((cd) => appStore.getState().setCompareData(cd))
      .catch(() => void 0);
  }

  // ── view manager: atlas ↔ compare ↔ chord crossfade ────────────────────
  // One driver per canvas: AtlasDriver keeps #scene-canvas; CompareDriver and
  // ChordDriver each get a lazily-created sibling canvas. Switching crossfades
  // opacity and swaps pointer-events + a mode class on the stage.
  function makeAuxCanvas(id: string): HTMLCanvasElement {
    const c = document.createElement("canvas");
    c.id = id;
    c.style.opacity = "0";
    c.style.pointerEvents = "none";
    c.style.transition = `opacity ${FADE_MS}ms ease`;
    canvas.after(c);
    return c;
  }

  async function ensureCompareDriver(): Promise<CompareDriver> {
    if (compareDriver) return compareDriver;
    const cd = appStore.getState().compareData;
    if (!cd) throw new Error("no comparison export — run `uv run nebulai compare <models…>`");
    compareCanvas = makeAuxCanvas("compare-canvas");
    const d = new CompareDriver();
    await d.init(compareCanvas);
    d.setData(cd);
    d.resize(stage.clientWidth, stage.clientHeight, window.devicePixelRatio || 1);
    compareDriver = d;
    window.__compareDriver = d;
    return d;
  }

  async function ensureChordDriver(): Promise<ChordDriver> {
    if (chordDriver) return chordDriver;
    const dsNow = appStore.getState().dataset;
    if (!dsNow) throw new Error("no dataset loaded");
    chordCanvas = makeAuxCanvas("chord-canvas");
    const d = new ChordDriver();
    await d.init(chordCanvas, caps.tier);
    d.resize(stage.clientWidth, stage.clientHeight, window.devicePixelRatio || 1);
    d.setDataset(dsNow);
    chordDriver = d;
    window.__chordDriver = d;
    return d;
  }

  async function ensureHierDriver(): Promise<HierarchyDriver> {
    if (hierDriver) return hierDriver;
    const dsNow = appStore.getState().dataset;
    if (!dsNow) throw new Error("no dataset loaded");
    hierCanvas = makeAuxCanvas("hier-canvas");
    const d = new HierarchyDriver();
    await d.init(hierCanvas, caps.tier); // lazy-imports deck.gl inside
    d.resize(stage.clientWidth, stage.clientHeight, window.devicePixelRatio || 1);
    d.setDataset(dsNow);
    hierDriver = d;
    window.__hierDriver = d;
    return d;
  }

  canvas.style.transition = `opacity ${FADE_MS}ms ease`;

  async function switchViewMode(mode: ViewMode): Promise<void> {
    if (mode === activeMode) return;
    if (mode === "compare") await ensureCompareDriver();
    if (mode === "chord") await ensureChordDriver();
    if (mode === "hierarchy") await ensureHierDriver();
    activeMode = mode;
    fadeUntil = performance.now() + FADE_MS + 120;
    appStore.getState().setViewMode(mode);
    stage.classList.toggle("mode-compare", mode === "compare");
    stage.classList.toggle("mode-chord", mode === "chord");
    stage.classList.toggle("mode-hierarchy", mode === "hierarchy");
    const show = (c: HTMLCanvasElement | null, on: boolean) => {
      if (!c) return;
      c.style.opacity = on ? "1" : "0";
      c.style.pointerEvents = on ? "auto" : "none";
    };
    canvas.style.opacity = mode === "atlas" ? "1" : "0";
    canvas.style.pointerEvents = mode === "atlas" ? "" : "none";
    show(compareCanvas, mode === "compare");
    show(chordCanvas, mode === "chord");
    show(hierCanvas, mode === "hierarchy");
    say(`${metaLine()} · gpu: ${caps.tier}`);
  }

  mountChrome(chrome);
  registerActions({
    async switchDataset(id) {
      const st = appStore.getState();
      if (st.loading.active || id === st.datasetId) return;
      const entry = st.datasets.find((d) => d.id === id);
      if (!entry) return;
      st.setLoading(true);
      try {
        const next = await loadDataset(entry.path, (loaded, total) => {
          appStore.getState().setLoading(true, loaded, total);
          say(`${id} — ${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`);
        });
        appStore.getState().setDataset(id, next);
        driver.setDataset(next);
        chordDriver?.setDataset(next);
        hierDriver?.setDataset(next);
        say(`${metaLine()} · gpu: ${caps.tier}`);
      } finally {
        appStore.getState().setLoading(false);
      }
    },
    switchViewMode,
  });

  // deep links for e2e + `nebulai compare` handoff
  const deepView = new URLSearchParams(location.search).get("view");
  if (deepView === "compare" && caps.tier === "webgpu") {
    loadCompare()
      .then((cd) => {
        if (cd) {
          appStore.getState().setCompareData(cd);
          return switchViewMode("compare");
        }
      })
      .catch(() => void 0);
  } else if (deepView === "chord" || deepView === "hierarchy") {
    switchViewMode(deepView).catch(() => void 0);
  }

  window.__perf.bootMs = performance.now() - t0;
  console.info(
    `[nebulai] boot ${window.__perf.bootMs.toFixed(0)}ms, worker parse ${ds.parseMs.toFixed(0)}ms, ` +
      `${ds.hulls.length} hulls, schema v${ds.columns.schema}`,
  );

  // ── frame loop ─────────────────────────────────────────────────────────
  // ?frozen=1 pins the time uniform for screenshot goldens; the loop still
  // runs so camera tweens and picking stay live.
  const frozen = new URLSearchParams(location.search).has("frozen");
  const frameDts: number[] = [];
  let last = performance.now();
  let frames = 0;

  const loop = (now: number) => {
    const dt = now - last;
    last = now;
    // during the crossfade all live drivers render; afterwards only the active one
    const fading = now < fadeUntil;
    const t = frozen ? 0 : now / 1000;
    if (activeMode === "atlas" || fading) driver.frame(dt, t);
    if ((activeMode === "compare" || fading) && compareDriver) compareDriver.frame(dt, t);
    if ((activeMode === "chord" || fading) && chordDriver) chordDriver.frame(dt, t);
    if ((activeMode === "hierarchy" || fading) && hierDriver) hierDriver.frame(dt, t);

    frameDts.push(dt);
    if (frameDts.length > 120) frameDts.shift();
    if (++frames % 60 === 0) {
      const sorted = [...frameDts].sort((a, b) => a - b);
      window.__perf.p95FrameMs = sorted[Math.floor(sorted.length * 0.95)];
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

boot().catch((e) => {
  console.error(e);
  say(`boot failed: ${e instanceof Error ? e.message : e}`);
});
