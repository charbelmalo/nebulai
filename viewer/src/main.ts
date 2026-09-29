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
 *  provenance stays visible in every mode.
 *
 *  PAGE-AWARE BOOT (N03). Every action is registered before any GPU work, and
 *  no page waits on a map it does not show:
 *
 *  · the index and the release manifest (out/experience.json) load together;
 *  · an explicit `model` in the permalink is honoured or refused with a
 *    recovery state — never silently swapped for another map;
 *  · the map page with no model opens the manifest's pinned `default_atlas`,
 *    or, with no valid manifest, the known starter id marked "unverified
 *    default"; if the index lists neither, the chooser — never datasets[0];
 *  · Internals, Behavior and Episodes load nothing; opening the map later
 *    fetches the starter then;
 *  · each dataset request carries an identity: a newer one aborts the older
 *    fetch, and a superseded result can neither commit nor report progress;
 *  · the renderer boots alongside the fetch. Until it is ready, view-mode
 *    requests wait for it; on the static tier they are refused, while every
 *    data action still works. */

import "@psychix/viz/tokens.css";
import "@psychix/viz/craft-tokens.css";
import "./styles/nebulai.css";

import { registerActions, requestEpisodeStep, type CompareTourCommand } from "./app/actions";
import "./chrome/episodes";
import { bootShell, finishShellBoot, type BootedShell } from "./app/boot-shell";
import type { Capabilities } from "@psychix/viz/capabilities";
import { appStore, type ViewMode } from "./app/store";
import { NEBULAI_APP } from "./chrome/apps/nebulai";
import { $compareTour } from "./chrome/state";
import { applyTourStep, findTour } from "./chrome/tours";
import { registerInterpUrlHooks } from "./chrome/urlState";
import { loadCompare } from "./data/compare";
import {
  evictDataset,
  loadDataset,
  loadIndex,
  LoadError,
  type Dataset,
  type LoadOptions,
} from "./data/loader";
import {
  activeManifest,
  currentArtifact,
  findArtifact,
  type ManifestArtifact,
  loadManifest,
  setManifestStatus,
  STARTER_DATASET_ID,
} from "./data/experience";
import { APP_ROOT, DATA_BASE } from "./data/base";
import {
  defaultPage,
  experienceFromPath,
  isExperience,
  resolveExperience,
  type Experience,
} from "./app/experience";
import { NEBULAI_EXPERIENCES } from "./chrome/apps/nav";
import type { UrlState } from "./chrome/urlState";
import { isLiveTrace } from "./data/interp";
import { parsePin, resolveUnit, verifyFinding, type Finding, type UnitPin } from "./data/finding";
import type { PinSource } from "./app/store";
import { handRig } from "./hands/rig";
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
  knownEpisode: (id) => !!findTour(id),
  runEpisode: (id, step) => requestEpisodeStep(id, step),
  parsePin,
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

/** The GPU half: one driver per view plus the view manager and the frame
 *  loop. Created only on a rendering tier, after the data actions exist. */
interface Gfx {
  setDataset(ds: Dataset): void;
  switchViewMode(mode: ViewMode): Promise<void>;
  flyToCluster(id: number): void;
  flyToPoint(id: number): void;
  compareTour(cmd: CompareTourCommand): void;
}

interface ShowOptions {
  noCache?: boolean;
  keepTour?: boolean;
  unverifiedDefault?: boolean;
  /** a pinned open: read exactly these bytes (a manifest entry, current or
   *  retained) instead of the dataset's current artifact */
  artifact?: ManifestArtifact;
  /** a pinned open with no manifest: the mutable path must hash to this */
  expectSha256?: string;
}

/** Which experience this document was opened as: the entry HTML declares it
 *  (`<meta name="nebulai-experience">`), and the path is the fallback. */
function entryExperience(): Experience | null {
  const meta = document
    .querySelector<HTMLMetaElement>('meta[name="nebulai-experience"]')
    ?.getAttribute("content");
  if (isExperience(meta)) return meta;
  return experienceFromPath(location.pathname, new URL(APP_ROOT).pathname);
}

/** Resolve Learn / Atlas / Research before the chrome paints (see
 *  app/experience.ts for the precedence). When the link belongs to another
 *  experience than the entry it was opened at, the address bar moves to that
 *  entry in place — same bundle, same data root — and a one-line notice says
 *  why, so the correction is never silent. */
function resolveNebulaiContext(u: UrlState): void {
  const entry = entryExperience();
  const legacy = new URLSearchParams(location.search).get("view");
  const legacyView =
    legacy === "chord" || legacy === "hierarchy" || legacy === "compare" ? legacy : undefined;
  const intent = {
    page: u.page,
    view: u.view ?? legacyView,
    episode: !!u.episode,
    model: !!u.model,
    finding: !!u.pin,
  };
  const r = resolveExperience({ entry, explicit: u.experience, intent });
  // a nested document always has an entry; this only guards a misnamed file
  const exp: Experience = r.experience ?? entry ?? "atlas";
  if (entry && exp !== entry) {
    const path = new URL(`${exp}/`, APP_ROOT).pathname;
    history.replaceState(null, "", `${path}${location.search}${location.hash}`);
  }
  document.title = NEBULAI_EXPERIENCES[exp].documentTitle;
  document.body.dataset.experience = exp;
  const st = appStore.getState();
  st.setExperience(exp, r.notice, u.returnTo ?? null);
  if (!u.page) st.setPage(defaultPage(exp, intent));
}

async function boot() {
  const t0 = performance.now();
  const shell = await bootShell(NEBULAI_APP, resolveNebulaiContext);
  shell.say(`gpu: ${shell.caps.tier} — loading datasets…`);

  // Learn hosts map and Internals pages only while a lesson runs. When the
  // lesson ends (Exit, last step, or a model switch that clears it), the
  // visitor returns to the Lessons catalog instead of an unguided stage.
  appStore.subscribe((s, prev) => {
    if (s.experience !== "learn" || !prev.tour || s.tour) return;
    if (s.page !== "guide") s.setPage("guide");
  });

  // A dead atlas must not be a dead page — Internals and Guide owe it nothing,
  // and neither does a fresh checkout with an empty out/. Report the failure on
  // the status pill and carry on with the shell standing.
  let openInitial: () => void = () => void 0;
  try {
    openInitial = await bootAtlas(shell, t0);
  } catch (e) {
    console.error(e);
    shell.say(`atlas failed: ${e instanceof Error ? e.message : e}`);
  }

  // permalink: apply the remaining hash state now that actions are registered,
  // then keep the hash mirroring the store so every view is shareable. This
  // sits here rather than inside bootAtlas so it runs exactly once on every
  // path — atlas, no-atlas and failed-atlas alike.
  finishShellBoot(shell.urlState);
  openInitial();
}

/** Registers every action, starts the renderer, and returns the function that
 *  opens the boot dataset — run after the permalink is applied, so an episode
 *  deep link can claim the map before the starter would. */
async function bootAtlas(shell: BootedShell, t0: number): Promise<() => void> {
  const { caps, urlState, progress, say } = shell;

  // out/index.json is genuinely optional (a fresh checkout has no baked
  // artifacts at all), and so is the manifest. Both are facts to report, not
  // failures to throw.
  const [index, manifest] = await Promise.all([
    loadIndex().catch((e) => {
      console.warn("[nebulai] no dataset index —", e instanceof Error ? e.message : e);
      return null;
    }),
    loadManifest(DATA_BASE),
  ]);
  setManifestStatus(manifest);
  if (manifest.state === "invalid") {
    console.warn("[nebulai] experience.json ignored:", manifest.errors.join("; "));
  }
  const datasets = index?.datasets ?? [];
  appStore.getState().setDatasets(datasets);

  /* ── renderer (nullable until ready) ─────────────────────────────────── */
  let gfx: Gfx | null = null;
  let resolveGfx!: (g: Gfx | null) => void;
  const gfxReady = new Promise<Gfx | null>((r) => (resolveGfx = r));

  /* ── dataset requests, with identity ─────────────────────────────────── */
  let seq = 0;
  let inflight: AbortController | null = null;
  let lastRequest: { id: string; opts: ShowOptions } | null = null;
  let firstCommit = true;

  const resetProgress = () => {
    progress.classList.remove("is-done");
    progress.style.width = "0%";
  };
  const doneProgress = () => {
    progress.style.width = "100%";
    progress.classList.add("is-done");
  };
  const statusLine = () =>
    `${metaLine()} · gpu: ${caps.tier}${appStore.getState().unverifiedDefault ? " · unverified default" : ""}`;

  /** Load `id` and install it everywhere. Resolves true when THIS request
   *  committed; false when it failed (recorded in `loadError`) or was
   *  superseded by a newer request. */
  async function show(id: string, opts: ShowOptions = {}): Promise<boolean> {
    const entry = appStore.getState().datasets.find((d) => d.id === id);
    if (!entry) {
      appStore.getState().failLoad({
        datasetId: id,
        kind: "unknown-model",
        message: `No published map is named “${id}”.`,
      });
      say(`no map named ${id} — pick one from the list`);
      return false;
    }
    const my = ++seq;
    inflight?.abort();
    const ctrl = new AbortController();
    inflight = ctrl;
    lastRequest = { id, opts };
    appStore.getState().beginLoad(id);
    resetProgress();

    const onProgress: LoadOptions["onProgress"] = (loaded, total) => {
      if (my !== seq) return;
      appStore.getState().setLoading(true, loaded, total);
      if (total > 0) progress.style.width = `${((loaded / total) * 100).toFixed(1)}%`;
      say(
        total > 0
          ? `${id} — ${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`
          : `${id} — ${(loaded / 1e6).toFixed(1)} MB`,
      );
    };

    try {
      // A published artifact is read from its immutable path and checked
      // against its digest. A refresh after a local rebuild reads the mutable
      // path instead — the manifest describes the release, not the rebuild.
      const art = opts.artifact ?? (opts.noCache ? null : currentArtifact(activeManifest(), id));
      let ds: Dataset;
      if (opts.noCache) evictDataset(entry.path);
      if (art) {
        try {
          ds = await loadDataset(art.path, {
            onProgress,
            expectedSha256: art.sha256,
            signal: ctrl.signal,
          });
        } catch (e) {
          // the immutable copy is missing (a partial deploy): the mutable path
          // still serves a map, just not a pinned one
          if (!(e instanceof LoadError) || e.kind !== "fetch") throw e;
          console.warn(`[nebulai] ${art.path} unavailable, reading ${entry.path}`);
          // a pinned open still insists on the pinned bytes: the mutable
          // path is acceptable only if it hashes to the same digest
          ds = await loadDataset(entry.path, {
            onProgress,
            signal: ctrl.signal,
            expectedSha256: opts.artifact ? art.sha256 : undefined,
          });
        }
      } else {
        ds = await loadDataset(entry.path, {
          onProgress,
          noCache: opts.noCache,
          signal: ctrl.signal,
          expectedSha256: opts.expectSha256,
        });
      }
      if (my !== seq) return false;
      appStore.getState().setDataset(id, ds, {
        keepTour: opts.keepTour,
        unverifiedDefault: opts.unverifiedDefault,
      });
      gfx?.setDataset(ds);
      if (firstCommit) {
        firstCommit = false;
        window.__perf.parseMs = ds.parseMs;
      }
      say(statusLine());
      return true;
    } catch (e) {
      if (my !== seq) return false;
      if (e instanceof LoadError && e.kind === "aborted") {
        appStore.getState().endLoad();
        return false;
      }
      const kind = e instanceof LoadError && e.kind !== "aborted" ? e.kind : "fetch";
      const message = e instanceof Error ? e.message : String(e);
      appStore.getState().failLoad({
        datasetId: id,
        kind,
        message,
        expected: e instanceof LoadError ? e.expected : undefined,
        actual: e instanceof LoadError ? e.actual : undefined,
      });
      say(`${id} failed to load — ${message}`);
      console.error(`[nebulai] ${id}:`, e);
      return false;
    } finally {
      if (my === seq) {
        inflight = null;
        doneProgress();
      }
    }
  }

  /** Which map "open the starter" means right now. */
  function starterChoice(): { id: string; unverifiedDefault: boolean } | null {
    const m = activeManifest();
    const listed = (id: string) => appStore.getState().datasets.some((d) => d.id === id);
    if (m && listed(m.default_atlas.dataset_id)) {
      return { id: m.default_atlas.dataset_id, unverifiedDefault: false };
    }
    if (listed(STARTER_DATASET_ID)) return { id: STARTER_DATASET_ID, unverifiedDefault: true };
    return null;
  }

  async function openStarter(): Promise<void> {
    const pick = starterChoice();
    if (!pick) {
      const noIndex = appStore.getState().datasets.length === 0;
      appStore.getState().failLoad({
        datasetId: null,
        kind: noIndex ? "index" : "no-starter",
        message: noIndex
          ? "No dataset index was found."
          : "No starter map is published here. Pick a map from the list.",
      });
      if (noIndex) say("no datasets in out/index.json — run `uv run nebulai tokens` first");
      return;
    }
    await show(pick.id, { unverifiedDefault: pick.unverifiedDefault });
  }

  /** See AppActions.openPinned. */
  async function openPinned(pin: UnitPin, source: PinSource, finding?: Finding): Promise<void> {
    const st0 = appStore.getState();
    const pending = { status: "pending" as const, pin, source, finding };
    st0.setPin(pending);
    const stillMine = () => appStore.getState().pin === pending;
    const fail = (code: string, title: string, message: string, extra: { expected?: string; actual?: string; conflicts?: string[] } = {}) => {
      if (!stillMine()) return;
      appStore.getState().setPin({ status: "error", pin, source, code, title, message, ...extra });
    };

    if (!st0.datasets.some((d) => d.id === pin.datasetId)) {
      fail("unknown-map", "Map unavailable", `No published map is named “${pin.datasetId}” here, so this unit cannot be opened.`);
      return;
    }
    const manifest = activeManifest();
    const exact = findArtifact(manifest, pin.datasetId, pin.sha256);
    const loaded = st0.datasetId === pin.datasetId ? st0.dataset : null;

    if (!(loaded && loaded.sha256 === pin.sha256)) {
      if (exact) {
        if (!(await show(pin.datasetId, { artifact: exact }))) {
          const err = appStore.getState().loadError;
          if (!err) {
            if (stillMine()) appStore.getState().setPin({ status: "none" }); // superseded
            return;
          }
          appStore.getState().clearLoadError();
          fail(
            err.kind === "digest" ? "digest" : "fetch",
            "Exact artifact unavailable",
            err.kind === "digest"
              ? "The published bytes for this unit's map no longer match its recorded digest. Nothing was opened."
              : `The map this unit was saved from could not be read (${err.message}). Nothing was opened.`,
            { expected: pin.sha256, actual: err.actual },
          );
          return;
        }
      } else {
        // not in the trusted manifest: when the digest this map serves now
        // is known, the pinned bytes are simply not published — say so
        // without fetching anything
        // A manifest that does not list this map at all says nothing about
        // its versions, so that case falls through to digest verification.
        const listed = !!manifest?.artifacts.some((a) => a.dataset_id === pin.datasetId);
        const serving = loaded?.sha256 ?? currentArtifact(manifest, pin.datasetId)?.sha256 ?? null;
        if (listed || serving) {
          fail(
            "wrong-artifact",
            "Exact artifact unavailable",
            "This unit was saved from a version of the map that is not published here. The current map was not substituted.",
            { expected: pin.sha256, actual: serving ?? undefined },
          );
          return;
        }
        // no manifest entry: the mutable path is acceptable only if its
        // bytes hash to the pinned digest
        if (!(await show(pin.datasetId, { expectSha256: pin.sha256 }))) {
          const err = appStore.getState().loadError;
          if (!err) {
            if (stillMine()) appStore.getState().setPin({ status: "none" });
            return;
          }
          appStore.getState().clearLoadError();
          fail(
            err.kind === "digest" ? "wrong-artifact" : "fetch",
            "Exact artifact unavailable",
            err.kind === "digest"
              ? "The map published here is a different version from the one this unit was saved from. Nothing was opened."
              : `The map could not be read (${err.message}). Nothing was opened.`,
            { expected: pin.sha256, actual: err.actual },
          );
          return;
        }
      }
    }
    if (!stillMine()) return;

    const st = appStore.getState();
    const ds = st.dataset!;
    const r = finding ? verifyFinding(finding, ds, pin.datasetId) : resolveUnit(ds, pin.datasetId, pin);
    if (!r.ok) {
      const title =
        r.code === "conflict" ? "Record conflicts with the map" : r.code === "missing" ? "Unit not found" : r.code === "ambiguous" ? "Unit is ambiguous" : "Exact artifact unavailable";
      fail(r.code, title, r.message, r.code === "conflict" ? { conflicts: r.conflicts } : {});
      return;
    }
    // reopen on the plain atlas, in the dimensions the record names
    if (st.viewMode !== "atlas") {
      const g = gfx ?? (await gfxReady);
      if (g) await g.switchViewMode("atlas");
      else appStore.getState().setViewMode("atlas");
    }
    if (!stillMine()) return;
    const s2 = appStore.getState();
    if (s2.mapQuery.text) s2.setMapQuery("");
    if (s2.channel.id) s2.setChannel(null);
    if (s2.axis.directionId) s2.setAxisDirection(null);
    s2.setDims(pin.dims);
    s2.setPin({ status: "ok", pin, row: r.row, source, note: finding?.note || undefined });
    s2.setSelection({ kind: "point", id: r.row });
    s2.setInspectorOpen(true);
    if (s2.page !== "map") s2.setPage("map");
    const g = gfx ?? (await gfxReady);
    g?.flyToPoint(r.row);
  }

  /* ── actions: all registered before any GPU work ─────────────────────── */
  registerActions({
    async switchDataset(id) {
      const st = appStore.getState();
      // re-picking the map on screen is a no-op, unless it cancels a
      // different pending request
      if (id === st.datasetId && st.pendingDatasetId === null) return;
      if (id === st.pendingDatasetId) return;
      const started = performance.now();
      try {
        await show(id);
      } finally {
        // Measure the application path, not Playwright's selectOption and
        // cross-process polling overhead. The e2e budget warms the cache first,
        // so this covers cache lookup, store/driver handoff, and rendered state.
        window.__perf.datasetSwitchMs = performance.now() - started;
      }
    },
    async switchViewMode(mode) {
      // queued until the renderer exists; refused where it never will
      const g = gfx ?? (await gfxReady);
      if (!g) return;
      await g.switchViewMode(mode);
    },
    async refreshDatasets(datasetId) {
      const index = await loadIndex(DATA_BASE, true);
      appStore.getState().setDatasets(index.datasets);
      if (!index.datasets.some((d) => d.id === datasetId)) return; // built into another out root
      if (await show(datasetId, { noCache: true })) {
        appStore.getState().pushProgressEvent("done", `map ready — ${datasetId}`);
        appStore.getState().setProgress({ stage: "done", pct: 1 });
      }
    },
    /** One episode step. The only place a dataset is installed WITHOUT
     *  clearing the tour — because here the tour is what asked for it. */
    async runEpisodeStep(episodeId, step) {
      const tour = findTour(episodeId);
      const spec = tour?.steps[step];
      if (!tour || !spec) return;
      const st = appStore.getState();

      if (spec.dataset && spec.dataset !== st.datasetId) {
        // A missing dataset is NOT a reason to narrate over whatever map
        // happens to be loaded: the captions quote that model's numbers. The
        // episode's manifest already refuses to offer it, and this is the
        // second line of defence for a deep link that skipped the offer.
        if (!st.datasets.some((d) => d.id === spec.dataset)) {
          console.warn(`[nebulai] episode ${episodeId} needs dataset ${spec.dataset}`);
          return;
        }
        if (!(await show(spec.dataset, { keepTour: true }))) return;
      }

      appStore.getState().setTour({ id: episodeId, step });
      if (spec.page) appStore.getState().setPage(spec.page);
      applyTourStep(tour, step);
    },
    async retryLoad() {
      const st = appStore.getState();
      const err = st.loadError;
      if (!err) return;
      if (err.kind === "index") {
        const idx = await loadIndex(DATA_BASE, true).catch(() => null);
        if (!idx) {
          say("no datasets in out/index.json — still missing");
          return;
        }
        appStore.getState().setDatasets(idx.datasets);
        if (manifestStateOk() === false) setManifestStatus(await loadManifest(DATA_BASE, true));
        await openStarter();
        return;
      }
      if (err.kind === "no-starter" || err.kind === "unknown-model") return; // needs a pick
      const again = lastRequest && lastRequest.id === err.datasetId ? lastRequest : null;
      if (again) await show(again.id, { ...again.opts, noCache: again.opts.noCache });
      else if (err.datasetId) await show(err.datasetId);
    },
    openStarter,
    openPinned,
    flyToCluster(id) {
      gfx?.flyToCluster(id);
    },
    flyToPoint(id) {
      gfx?.flyToPoint(id);
    },
    compareTour(cmd) {
      gfx?.compareTour(cmd);
    },
  });

  function manifestStateOk(): boolean {
    return activeManifest() !== null;
  }

  /* ── renderer, alongside the first fetch ─────────────────────────────── */
  if (caps.tier === "static") {
    appStore.getState().setRenderer("unavailable");
    resolveGfx(null);
  } else {
    initGfx(caps, say, statusLine)
      .then((g) => {
        gfx = g;
        const ds = appStore.getState().dataset;
        if (ds) g.setDataset(ds);
        appStore.getState().setRenderer("ready");
        resolveGfx(g);
      })
      .catch((e) => {
        console.error("[nebulai] renderer failed", e);
        appStore.getState().setRenderer("unavailable");
        say(`renderer unavailable — ${e instanceof Error ? e.message : e}`);
        resolveGfx(null);
      });
  }

  // The boot budget ends when the boot map is committed (or refused) AND the
  // renderer has settled either way.
  const markBooted = async (first: Promise<unknown>) => {
    await Promise.allSettled([first, gfxReady]);
    if (window.__perf.bootMs !== undefined) return;
    window.__perf.bootMs = performance.now() - t0;
    const ds = appStore.getState().dataset;
    console.info(
      `[nebulai] boot ${window.__perf.bootMs.toFixed(0)}ms` +
        (ds ? `, worker parse ${ds.parseMs.toFixed(0)}ms, ${ds.hulls.length} hulls, schema v${ds.columns.schema}` : ", no map"),
    );
    if (caps.tier === "static" && ds) say(`${statusLine()} (no WebGPU/WebGL — results list only)`);
  };

  /* ── what to open at boot (runs after the permalink is applied) ───────── */
  return () => {
    const st = appStore.getState();
    let first: Promise<unknown> = Promise.resolve();
    if (datasets.length === 0) {
      st.failLoad({ datasetId: null, kind: "index", message: "No dataset index was found." });
      say("no datasets in out/index.json — run `uv run nebulai tokens` first");
    } else if (st.pendingDatasetId !== null || st.datasetId !== null) {
      // the permalink (an episode step) already claimed a map
      first = waitForLoad();
    } else if (urlState.pin) {
      // a pinned unit: verify first; a broken pin loads NOTHING, not the
      // latest map under an exact-looking link
      if (urlState.pin.ok) first = openPinned(urlState.pin.pin, "link");
      else
        st.setPin({
          status: "error",
          pin: null,
          source: "link",
          code: "invalid-link",
          title: "Unit link can't be opened",
          message: urlState.pin.message,
        });
    } else if (urlState.model) {
      first = show(urlState.model);
    } else if (st.page === "map" && st.experience === "atlas") {
      // Atlas opens on its curated starter. Research's Comparisons never
      // loads it implicitly — a research view asks for its map by name.
      first = openStarter();
    } else {
      say(`gpu: ${caps.tier} · ${datasets.length} maps available`);
    }
    void markBooted(first);

    // Opening the map later, with nothing on it, fetches the starter then.
    appStore.subscribe((s, prev) => {
      if (s.page !== "map" || prev.page === "map") return;
      if (s.experience !== "atlas") return;
      if (s.datasetId || s.pendingDatasetId || s.loadError) return;
      void openStarter();
    });
  };
}

/** Resolves once no dataset request is pending. */
function waitForLoad(): Promise<void> {
  return new Promise((resolve) => {
    const check = () => appStore.getState().pendingDatasetId === null;
    if (check()) return resolve();
    const off = appStore.subscribe(() => {
      if (check()) {
        off();
        resolve();
      }
    });
  });
}

/** Build every GPU driver, the view manager and the frame loop. */
async function initGfx(
  caps: Capabilities,
  say: (t: string) => void,
  statusLine: () => string,
): Promise<Gfx> {
  const tier = caps.tier as Exclude<Capabilities["tier"], "static">;
  const canvas = document.getElementById("scene-canvas") as HTMLCanvasElement;
  const driver = new AtlasDriver();
  await driver.init(canvas, tier);
  window.__driver = driver; // e2e + debugging handle

  // Webcam hand control (src/hands). Pointing the rig at the driver costs
  // nothing on its own — no camera, no model, no frame loop — until the
  // Settings toggle turns it on, which is what `watchSettings` waits for.
  handRig.setTarget(driver);
  handRig.watchSettings();

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

  // compare.json is optional (run `nebulai compare`); discovery is
  // non-blocking so the atlas never waits on it.
  loadCompare()
    .then((cd) => appStore.getState().setCompareData(cd))
    .catch(() => void 0);

  // ── view manager: atlas ↔ compare ↔ chord crossfade ────────────────────
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
    await d.init(chordCanvas, tier);
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
    await d.init(hierCanvas, tier); // lazy-imports deck.gl inside
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
    say(statusLine());
  }

  // deep links for e2e + `nebulai compare` handoff. Chord and hierarchy need
  // a map, so they wait for the boot map rather than failing before it lands.
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
    const go = () => switchViewMode(deepView).catch(() => void 0);
    if (appStore.getState().dataset) void go();
    else {
      const off = appStore.subscribe((s) => {
        if (!s.dataset) return;
        off();
        void go();
      });
    }
  }

  // ── frame loop ─────────────────────────────────────────────────────────
  // ?frozen=1 pins the time uniform for screenshot goldens; the loop still
  // runs so camera tweens and picking stay live. Every driver's frame() is a
  // no-op until it has a dataset.
  const frozen = new URLSearchParams(location.search).has("frozen");
  const frameDts: number[] = [];
  const frameWork: number[] = [];
  let last = performance.now();
  let frames = 0;

  const loop = (now: number) => {
    const dt = now - last;
    last = now;
    const workStarted = performance.now();
    const fading = now < fadeUntil;
    const t = frozen ? 0 : now / 1000;
    if (activeMode === "atlas" || fading) driver.frame(dt, t);
    if ((activeMode === "compare" || fading) && compareDriver) compareDriver.frame(dt, t);
    if ((activeMode === "chord" || fading) && chordDriver) chordDriver.frame(dt, t);
    if ((activeMode === "hierarchy" || fading) && hierDriver) hierDriver.frame(dt, t);

    frameDts.push(dt);
    frameWork.push(performance.now() - workStarted);
    if (frameDts.length > 120) frameDts.shift();
    if (frameWork.length > 120) frameWork.shift();
    if (++frames % 60 === 0) {
      const sorted = [...frameDts].sort((a, b) => a - b);
      const sortedWork = [...frameWork].sort((a, b) => a - b);
      window.__perf.p95FrameMs = sorted[Math.floor(sorted.length * 0.95)];
      window.__perf.p95FrameWorkMs = sortedWork[Math.floor(sortedWork.length * 0.95)];
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);

  return {
    setDataset(ds) {
      driver.setDataset(ds);
      chordDriver?.setDataset(ds);
      hierDriver?.setDataset(ds);
    },
    switchViewMode,
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
  };
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
