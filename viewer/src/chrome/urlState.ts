/** Permalink layer — mirrors the shareable slice of app state into
 *  `location.hash` so any view a user lands on can be sent to someone else as
 *  a URL. Read once at boot (`readUrlState` + `applyUrlState`), then mirrored
 *  forever after (`startUrlSync`) via `history.replaceState` so back/forward
 *  history is never spammed by exploration.
 *
 *  Format: `#page=interp&model=gpt2&feature=live-nebula&trace=<slug>` /
 *  `#page=map&model=gpt2&view=chord&dims=3`. Only keys meaningful for the
 *  active page are written, so links stay short and honest. The legacy
 *  `?view=` search param (e2e + `nebulai compare` handoff) is untouched. */

import { APP_PAGES, appStore, type Page, type ViewMode } from "../app/store";
import { requestViewMode } from "../app/actions";

const VIEWS: readonly ViewMode[] = ["atlas", "chord", "hierarchy", "compare"];

/** The two Internals-only hash keys (`feature`, `trace`) can only be validated
 *  by things that live on Nebulai's side of the split: `feature` against the
 *  25-driver interp registry, `trace` against the live-trace prefix in
 *  data/interp. Importing either here would drag the whole Internals gallery
 *  — and three.js with it — into Seer's bundle, for a permalink key Seer's
 *  three pages cannot even express. So the app that owns those pages injects
 *  the validators at boot (src/main.ts) and this module stays instrument-
 *  neutral. Unregistered means "reject", which is the correct answer on Seer:
 *  an interp permalink opened there is not a link it can honour.
 *
 *  Everything else about the permalink stays shared and single. */
export interface InterpUrlHooks {
  /** is this a feature id the Internals rail actually ships? */
  knownFeature(id: string): boolean;
  /** may this trace slug go in a shareable URL? Live traces exist only in the
   *  tab that captured them, so a link to one would land on an honest error. */
  shareableTrace(slug: string): boolean;
  /** is this an episode id this bundle can actually play? Same argument as
   *  `knownFeature`: the episode registry reaches the interp registry, and
   *  Seer must not grow a three.js import for a key its pages cannot express. */
  knownEpisode?(id: string): boolean;
  /** play episode `id` at `step`. Asynchronous inside — a step may name a model
   *  that still has to be fetched — so it is injected rather than imported. */
  runEpisode?(id: string, step: number): void;
}

const NO_INTERP: InterpUrlHooks = { knownFeature: () => false, shareableTrace: () => false };
let interpHooks: InterpUrlHooks = NO_INTERP;

/** Called by the entry that owns the Internals page, before `readUrlState`. */
export function registerInterpUrlHooks(hooks: InterpUrlHooks): void {
  interpHooks = hooks;
}

export interface UrlState {
  page?: Page;
  model?: string;
  feature?: string;
  trace?: string;
  view?: ViewMode;
  dims?: 2 | 3;
  /** map-page keyword search query */
  q?: string;
  /** map-page channel lens: a channel id from the model's `channels.json` */
  channel?: string;
  /** the lens's filter window, `lo,hi` in the channel's own raw units */
  crange?: [number, number];
  /** episode being played, and how far into it */
  episode?: string;
  step?: number;
}

/** Parse the current hash. Unknown keys/values are dropped, never guessed. */
export function readUrlState(): UrlState {
  const p = new URLSearchParams(location.hash.replace(/^#/, ""));
  const out: UrlState = {};
  // only the running instrument's own three pages are addressable: a
  // `#page=seer` on Nebulai's document names a page that is not in this
  // bundle, so it is dropped here rather than half-applied downstream
  const pages = APP_PAGES[appStore.getState().app] as readonly string[];
  const page = p.get("page");
  if (page && pages.includes(page)) out.page = page as Page;
  const model = p.get("model");
  if (model) out.model = model;
  const feature = p.get("feature");
  if (feature && interpHooks.knownFeature(feature)) out.feature = feature;
  const trace = p.get("trace");
  if (trace && interpHooks.shareableTrace(trace)) out.trace = trace;
  const view = p.get("view");
  if (view && (VIEWS as readonly string[]).includes(view)) out.view = view as ViewMode;
  const dims = p.get("dims");
  if (dims === "2" || dims === "3") out.dims = Number(dims) as 2 | 3;
  const q = p.get("q");
  if (q && q.trim()) out.q = q;
  // A channel id cannot be validated here: which channels exist is a property
  // of an artifact that has not been fetched yet. So it is carried through
  // as-is and the driver drops it when the dataset has no such channel — the
  // same place that decides whether the lens can be lit at all, which keeps
  // "unknown channel" from having two different answers.
  const channel = p.get("channel");
  if (channel && channel.trim()) out.channel = channel.trim();
  const crange = p.get("crange");
  if (crange) {
    const parts = crange.split(",").map(Number);
    const lo = parts[0];
    const hi = parts[1];
    // a half-parsed window would filter on a bound the user never chose
    if (
      lo !== undefined &&
      hi !== undefined &&
      Number.isFinite(lo) &&
      Number.isFinite(hi) &&
      lo <= hi
    ) {
      out.crange = [lo, hi];
    }
  }
  const episode = p.get("episode");
  if (episode && (interpHooks.knownEpisode?.(episode) ?? false)) {
    out.episode = episode;
    const step = Number(p.get("step") ?? "0");
    out.step = Number.isInteger(step) && step >= 0 ? step : 0;
  }
  return out;
}

/** Apply a parsed permalink to the store. Call AFTER the boot dataset load and
 *  `registerActions` (view switching goes through the app-shell handler; the
 *  model itself is picked at boot from `UrlState.model`, not here). */
export function applyUrlState(u: UrlState): void {
  const st = appStore.getState();
  if (u.feature) st.setInterpFeature(u.feature);
  if (u.trace) st.setInterpTrace(u.trace);
  if (u.dims) st.setDims(u.dims);
  // after the boot dataset load, so the labels to search are resident
  if (u.q) st.setMapQuery(u.q);
  if (u.channel) st.setChannel(u.channel, u.crange ?? null);
  if (u.page) st.setPage(u.page);
  if (u.view && u.view !== "atlas") requestViewMode(u.view);
  // last, because an episode step rewrites page, model, channel and selection:
  // it must not be overwritten by the very keys it exists to supersede
  if (u.episode) interpHooks.runEpisode?.(u.episode, u.step ?? 0);
}

function buildHash(): string {
  const st = appStore.getState();
  const p = new URLSearchParams();
  p.set("page", st.page);
  if (st.datasetId) p.set("model", st.datasetId);
  if (st.page === "map") {
    if (st.viewMode !== "atlas") p.set("view", st.viewMode);
    if (st.dims === 3) p.set("dims", "3");
    if (st.mapQuery.text.trim()) p.set("q", st.mapQuery.text);
    if (st.channel.id) {
      p.set("channel", st.channel.id);
      // the window travels in RAW units, so the link states what it filtered on
      // even to someone reading the URL rather than opening it
      if (st.channel.window) p.set("crange", st.channel.window.join(","));
    }
  } else if (st.page === "interp") {
    p.set("feature", st.interp.featureId);
    // live traces exist only in this tab's memory — a permalink to one would
    // land on an honest error, so they're never written into the hash
    if (st.interp.traceSlug && interpHooks.shareableTrace(st.interp.traceSlug))
      p.set("trace", st.interp.traceSlug);
  }
  // an episode is a position in a narrative, not a property of one page: it has
  // to survive the step that carries it from Internals to the Map
  if (st.tour) {
    p.set("episode", st.tour.id);
    p.set("step", String(st.tour.step));
  }
  return `#${p.toString()}`;
}

/** Keep the hash mirroring the store. Debounced a microtask so a burst of
 *  store writes (dataset switch clears selection etc.) coalesces into one
 *  write. NOT rAF-debounced: browsers freeze rAF entirely in occluded tabs,
 *  which would leave the hash stale exactly when a user copies a link from a
 *  backgrounded window. */
export function startUrlSync(): void {
  let queued = false;
  const sync = () => {
    queued = false;
    const h = buildHash();
    if (location.hash !== h) history.replaceState(null, "", h);
  };
  appStore.subscribe(() => {
    if (queued) return;
    queued = true;
    queueMicrotask(sync);
  });
  sync();
}

/** The current view as a shareable absolute URL (hash is always in sync once
 *  `startUrlSync` has run). */
export function shareUrl(): string {
  return `${location.origin}${location.pathname}${location.search}${buildHash()}`;
}
