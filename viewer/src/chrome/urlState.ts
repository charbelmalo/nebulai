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

import { appStore, type Page, type ViewMode } from "../app/store";
import { requestViewMode } from "../app/actions";
import { isLiveTrace } from "../data/interp";
import { findFeature } from "../scene/interp/registry";

const PAGES: readonly Page[] = ["map", "snapshot", "interp", "guide", "sessions"];
const VIEWS: readonly ViewMode[] = ["atlas", "chord", "hierarchy", "compare"];

export interface UrlState {
  page?: Page;
  model?: string;
  feature?: string;
  trace?: string;
  view?: ViewMode;
  dims?: 2 | 3;
  /** map-page keyword search query */
  q?: string;
}

/** Parse the current hash. Unknown keys/values are dropped, never guessed. */
export function readUrlState(): UrlState {
  const p = new URLSearchParams(location.hash.replace(/^#/, ""));
  const out: UrlState = {};
  const page = p.get("page");
  if (page && (PAGES as readonly string[]).includes(page)) out.page = page as Page;
  const model = p.get("model");
  if (model) out.model = model;
  const feature = p.get("feature");
  if (feature && findFeature(feature)) out.feature = feature;
  const trace = p.get("trace");
  if (trace && !isLiveTrace(trace)) out.trace = trace;
  const view = p.get("view");
  if (view && (VIEWS as readonly string[]).includes(view)) out.view = view as ViewMode;
  const dims = p.get("dims");
  if (dims === "2" || dims === "3") out.dims = Number(dims) as 2 | 3;
  const q = p.get("q");
  if (q && q.trim()) out.q = q;
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
  if (u.page) st.setPage(u.page);
  if (u.view && u.view !== "atlas") requestViewMode(u.view);
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
  } else if (st.page === "interp") {
    p.set("feature", st.interp.featureId);
    // live traces exist only in this tab's memory — a permalink to one would
    // land on an honest error, so they're never written into the hash
    if (st.interp.traceSlug && !isLiveTrace(st.interp.traceSlug))
      p.set("trace", st.interp.traceSlug);
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
