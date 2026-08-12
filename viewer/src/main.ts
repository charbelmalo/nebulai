/** Nebulai's entry — the instrument that maps what a model knows
 *  (Semantic map · Internals · Guide). Seer's entry is `src/seer-main.ts`;
 *  the two share `app/boot-shell.ts`, the chrome and @psychix/viz, and share
 *  no page components or drivers at all.
 *
 *  Boot happens in two phases, because the shell must not depend on the atlas.
 *
 *  `bootShell()` (shared, see app/boot-shell.ts) is unconditional — probe the
 *  GPU tier, mount the Preact chrome, read the permalink. Nothing in it can be
 *  starved by missing data. `bootAtlas()` below holds everything that needs
 *  baked artifacts: discover datasets, load one through the worker, hand the
 *  canvas to the AtlasDriver, wire the view manager and run the frame loop. It
 *  returns quietly when there is nothing to render, rather than taking the
 *  page down with it.
 *
 *  The split exists because a second instrument — Seer, the agent-run
 *  observability app — shares this shell and runs with ZERO atlas artifacts in
 *  `out/`. Under the old linear boot a missing out/index.json aborted before a
 *  single pixel of chrome appeared, which held every non-map page hostage to a
 *  dataset it never reads. It is still worth keeping now that Seer boots from
 *  its own HTML: a Nebulai checkout with nothing built yet lands on a standing
 *  page that says so. The status pill doubles as the MetaLine — dataset
 *  provenance stays visible in every mode. */

import "@psychix/viz/tokens.css";
import "@psychix/viz/craft-tokens.css";
import "./styles/nebulai.css";

import { registerActions } from "./app/actions";
import { bootShell, finishShellBoot, type BootedShell } from "./app/boot-shell";
import { appStore, type ViewMode } from "./app/store";
import { NEBULAI_APP } from "./chrome/apps/nebulai";
import { $compareTour } from "./chrome/state";
import { registerInterpUrlHooks } from "./chrome/urlState";
import { loadCompare } from "./data/compare";
import { evictDataset, loadDataset, loadIndex } from "./data/loader";
import { DATA_BASE } from "./data/base";
import { isLiveTrace } from "./data/interp";
import { findFeature } from "./scene/interp/registry";
import { AtlasDriver } from "./scene/drivers/AtlasDriver";
import { ChordDriver } from "./scene/drivers/ChordDriver";
import { CompareDriver } from "./scene/drivers/CompareDriver";
import { HierarchyDriver } from "./scene/drivers/HierarchyDriver";

declare global {
  interface Window {
    __driver?: AtlasDriver;
    __compareDriver?: CompareDriver;
    __chordDriver?: ChordDriver;
    __hierDriver?: HierarchyDriver;
    __sessionDriver?: { describe(): unknown };
  }
}

// The permalink layer is instrument-neutral by design; the two Internals-only
// hash keys are validated by things only this entry may import (the 25-driver
// registry, the live-trace prefix). Registered before bootShell reads the
// hash. See chrome/urlState.ts.
registerInterpUrlHooks({
  knownFeature: (id) => !!findFeature(id),
  shareableTrace: (slug) => !isLiveTrace(slug),
});


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
  const shell = await bootShell(NEBULAI_APP);
  shell.say(`gpu: ${shell.caps.tier} — loading datasets…`);

  // A dead atlas must not be a dead page — Internals and Guide owe it nothing,
  // and neither does a fresh checkout with an empty out/. Report the failure on
  // the status pill and carry on with the shell standing.
  try {
    await bootAtlas(shell, t0);
  } catch (e) {
    console.error(e);
    shell.say(`atlas failed: ${e instanceof Error ? e.message : e}`);
  }

  // permalink: apply the remaining hash state now that actions are registered,
  // then keep the hash mirroring the store so every view is shareable. This
  // sits here rather than inside bootAtlas so it runs exactly once on every
  // path — atlas, no-atlas and failed-atlas alike — and still lands after
  // registerActions, which bootAtlas reaches before it resolves.
  finishShellBoot(shell.urlState);
}

async function bootAtlas(shell: BootedShell, t0: number) {
  const { caps, urlState, progress, say } = shell;
  // out/index.json is genuinely optional: a Seer-only checkout has no baked
  // artifacts at all. A missing or empty index is a fact to report, not a
  // failure to throw — the shell is already up and every non-map page works
  // without a single atlas byte.
  const index = await loadIndex().catch((e) => {
    console.warn("[nebulai] no dataset index —", e instanceof Error ? e.message : e);
    return null;
  });
  const datasets = index?.datasets ?? [];
  appStore.getState().setDatasets(datasets);
  const first = datasets.find((d) => d.id === urlState.model) ?? datasets[0];
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
  // non-blocking so the atlas never waits on it. The old `tier === "webgpu"`
  // gate went away with the driver's raw-WGSL rewrite — it is TSL now, so it
  // renders on the forceWebGL rung too (minus bloom, like the atlas).
  loadCompare()
    .then((cd) => appStore.getState().setCompareData(cd))
    .catch(() => void 0);

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
    d.onTour = (s) => ($compareTour.value = s);
    d.setReducedMotion(caps.reducedMotion);
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

  /** Load a dataset entry and hand it to every live driver. `noCache` skips
   *  both the in-memory column cache and the browser HTTP cache — used after
   *  a rebuild overwrites the artifact on disk. */
  async function loadAndShow(entry: { id: string; path: string }, noCache = false) {
    const st = appStore.getState();
    st.setLoading(true);
    progress.classList.remove("is-done");
    progress.style.width = "0%";
    try {
      if (noCache) evictDataset(entry.path);
      const next = await loadDataset(
        entry.path,
        (loaded, total) => {
          appStore.getState().setLoading(true, loaded, total);
          progress.style.width = `${((loaded / total) * 100).toFixed(1)}%`;
          say(`${entry.id} — ${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`);
        },
        DATA_BASE,
        noCache,
      );
      appStore.getState().setDataset(entry.id, next);
      driver.setDataset(next);
      chordDriver?.setDataset(next);
      hierDriver?.setDataset(next);
      say(`${metaLine()} · gpu: ${caps.tier}`);
    } finally {
      appStore.getState().setLoading(false);
      progress.style.width = "100%";
      progress.classList.add("is-done");
    }
  }

  registerActions({
    async switchDataset(id) {
      const st = appStore.getState();
      if (st.loading.active || id === st.datasetId) return;
      const entry = st.datasets.find((d) => d.id === id);
      if (!entry) return;
      await loadAndShow(entry);
    },
    switchViewMode,
    async refreshDatasets(datasetId) {
      if (appStore.getState().loading.active) return;
      const index = await loadIndex(DATA_BASE, true);
      appStore.getState().setDatasets(index.datasets);
      const entry = index.datasets.find((d) => d.id === datasetId);
      if (!entry) return; // built into a different out root than the one served
      await loadAndShow(entry, true);
      appStore.getState().pushProgressEvent("done", `map ready — ${datasetId}`);
      appStore.getState().setProgress({ stage: "done", pct: 1 });
    },
    flyToCluster(id) {
      if (appStore.getState().viewMode === "atlas") driver.flyToCluster(id);
    },
    flyToPoint(id) {
      if (appStore.getState().viewMode === "atlas") driver.flyToPoint(id);
    },
    compareTour(cmd) {
      const d = compareDriver;
      if (!d) return; // the panel only renders once the driver exists
      switch (cmd.kind) {
        case "toggle":
          d.togglePlay();
          break;
        case "play":
          d.play();
          break;
        case "pause":
          d.pause();
          break;
        case "restart":
          d.restart();
          break;
        case "seek":
          d.pause();
          d.seek(cmd.u);
          break;
        case "speed":
          d.setSpeed(cmd.mult);
          break;
        case "pick":
          d.pickState(cmd.state);
          break;
      }
    },
  });

  // deep links for e2e + `nebulai compare` handoff
  const deepView = new URLSearchParams(location.search).get("view");
  if (deepView === "compare") {
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
  // bootShell itself failed (or something after the try/catch did): the pill
  // may not exist yet, so fall back to creating one rather than losing the
  // only report of why the page is blank.
  console.error(e);
  const status =
    document.querySelector<HTMLElement>(".boot-status") ??
    document.getElementById("chrome")!.appendChild(
      Object.assign(document.createElement("div"), { className: "boot-status" }),
    );
  status.textContent = `boot failed: ${e instanceof Error ? e.message : e}`;
});
