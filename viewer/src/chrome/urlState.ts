/** Permalink layer — mirrors the shareable slice of app state into
 *  `location.hash` so any view a user lands on can be sent to someone else as
 *  a URL. Read once at boot (`readUrlState` + `applyUrlState`), then mirrored
 *  forever after (`startUrlSync`) via `history.replaceState` so back/forward
 *  history is never spammed by exploration.
 *
 *  Format: `#page=interp&model=gpt2&feature=live-nebula&trace=<slug>` /
 *  `#page=map&model=gpt2&view=chord&dims=3` /
 *  `#page=behavior&bview=ranked&cue=daddy&q=kin`. Only keys meaningful for the
 *  active page are written, so links stay short and honest. The legacy
 *  `?view=` search param (e2e + `nebulai compare` handoff) is untouched.
 *
 *  The Behavior keys are prefixed `b` where they would otherwise collide with
 *  a map key of the same name but different meaning (`bview` is a cue
 *  presentation, `view` is a driver). `cue` is NOT validated against a cue
 *  list: the study artifact is fetched by the page, not by this module, so a
 *  hash naming a cue absent from the published study opens the page with no
 *  cue selected rather than being silently rewritten to something else. */

import {
  APP_PAGES,
  appStore,
  type BehaviorView,
  type Page,
  type ViewMode,
} from "../app/store";
import { requestViewMode } from "../app/actions";
import { isExperience, type Experience } from "../app/experience";
import type { PinParse } from "../data/finding";

/** The pinned-unit hash keys (data/finding.ts PIN_KEYS), repeated here as
 *  plain strings so writing them needs no NebulAI-only import. */
const PIN_KEYS = ["artifact", "point", "unit_kind", "unit_index"] as const;

const VIEWS: readonly ViewMode[] = ["atlas", "chord", "hierarchy", "compare"];
const BVIEWS: readonly BehaviorView[] = ["landscape", "ranked", "table"];

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
  /** validate Learn's `lesson` / `lesson_step` keys against the lesson
   *  registry; returns the namespaced tour id and a valid step */
  parseLesson?(lesson: string | null, step: string | null): { tourId: string; step: number } | null;
  /** parse the pinned-unit keys (NebulAI only; Seer has no atlas units) */
  parsePin?(p: URLSearchParams): PinParse;
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
  /** map-page keyword search query; also the Behavior page's cue search */
  q?: string;
  /** map-page channel lens: a channel id from the model's `channels.json` */
  channel?: string;
  /** the lens's filter window, `lo,hi` in the channel's own raw units */
  crange?: [number, number];
  /** map-page direction axis: a direction id from the model's `directions.json` */
  axis?: string;
  /** how far the map has travelled onto that axis, 0–1 */
  axist?: number;
  /** `0` when the null cloud was switched OFF. Written only in that case, so
   *  the ghost is in every link that does not explicitly say otherwise. */
  axisnull?: boolean;
  /** episode being played, and how far into it */
  episode?: string;
  step?: number;
  /** Behavior: which cue is open */
  cue?: string;
  /** Behavior: landscape | ranked | table */
  bview?: BehaviorView;
  /** NebulAI experience named by the link. Kept RAW: whether it can host the
   *  rest of the link is app/experience.ts's decision, and an unknown value is
   *  simply ignored there. */
  experience?: string;
  /** contextual help's way back (`return=atlas`) */
  returnTo?: Experience;
  /** pinned unit: a verified identity tuple, or why the keys are unusable.
   *  Absent when the link names no pin at all. */
  pin?: Exclude<PinParse, null>;
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
  // Same argument as `channel` above: which directions exist is a property of
  // an artifact nobody has fetched yet, so the id travels as-is and the driver
  // drops it if this map cannot draw it. That keeps "unknown direction" with
  // exactly one answer, in the place that already owns R5 and D2.
  const axis = p.get("axis");
  if (axis && axis.trim()) {
    out.axis = axis.trim();
    const t = Number(p.get("axist") ?? "1");
    // a permalink naming an axis but no position means "all the way onto it" —
    // the picture the link was written to show
    out.axist = Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : 1;
    if (p.get("axisnull") === "0") out.axisnull = false;
  }
  const episode = p.get("episode");
  if (episode && (interpHooks.knownEpisode?.(episode) ?? false)) {
    out.episode = episode;
    const step = Number(p.get("step") ?? "0");
    out.step = Number.isInteger(step) && step >= 0 ? step : 0;
  }
  // Learn's lesson keys ride the same runner as episodes, under a namespaced
  // tour id; a recognised lesson wins over a stray episode key
  const lesson = interpHooks.parseLesson?.(p.get("lesson"), p.get("lesson_step")) ?? null;
  if (lesson) {
    out.episode = lesson.tourId;
    out.step = lesson.step;
  }
  const cue = p.get("cue");
  if (cue && cue.trim()) out.cue = cue;
  const bview = p.get("bview");
  if (bview && (BVIEWS as readonly string[]).includes(bview)) out.bview = bview as BehaviorView;
  const experience = p.get("experience");
  if (experience) out.experience = experience;
  const ret = p.get("return");
  if (isExperience(ret)) out.returnTo = ret;
  const pin = interpHooks.parsePin?.(p) ?? null;
  if (pin) out.pin = pin;
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
  // after the boot dataset load, so the labels to search are resident.
  // One hash key, two pages: `q` is the map's label search and the Behavior
  // page's cue search, and a permalink only ever names one page.
  if (u.q) {
    if (u.page === "behavior") st.setBehaviorQuery(u.q);
    else st.setMapQuery(u.q);
  }
  if (u.channel) st.setChannel(u.channel, u.crange ?? null);
  if (u.axis) {
    st.setAxisDirection(u.axis);
    st.setAxisT(u.axist ?? 1);
    if (u.axisnull === false) st.setAxisNull(false);
  }
  if (u.bview) st.setBehaviorView(u.bview);
  if (u.cue) st.setBehaviorCue(u.cue);
  if (u.page) st.setPage(u.page);
  if (u.view && u.view !== "atlas") requestViewMode(u.view);
  // last, because an episode step rewrites page, model, channel and selection:
  // it must not be overwritten by the very keys it exists to supersede
  if (u.episode) interpHooks.runEpisode?.(u.episode, u.step ?? 0);
}

/** The model the address named at boot. Until the first load starts,
 *  resolves or fails, the store has no dataset yet — and a hash rewritten in
 *  that window would drop `model=`, so a link copied (or a return address
 *  remembered) a moment after opening would lose the map it was opened on. */
let bootModel: string | null = null;
/** The pin keys exactly as the boot link carried them. A malformed pin is
 *  kept in the address while its error is on screen, so the link the visitor
 *  pasted is still the one they can copy back and inspect. */
let bootPinRaw: [string, string][] = [];

function buildHash(): string {
  const st = appStore.getState();
  if (bootModel && (st.datasetId || st.pendingDatasetId || st.loadError || st.interpModel))
    bootModel = null;
  // Internals names its own model. Research's Internals never borrows the
  // map's: with no export chosen, the address carries none (the chooser).
  const model =
    st.page === "interp" && st.interpModel
      ? st.interpModel
      : st.experience === "research" && st.page === "interp"
        ? bootModel
        : (st.pendingDatasetId ?? st.datasetId ?? bootModel);
  const p = new URLSearchParams();
  // the experience is written explicitly so a link pasted anywhere under the
  // app — including the root chooser — reopens in the same context
  if (st.experience) p.set("experience", st.experience);
  p.set("page", st.page);
  if (model) p.set("model", model);
  const pin = st.pin;
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
    if (st.axis.directionId) {
      p.set("axis", st.axis.directionId);
      // the blend position travels too: "which direction" and "how far along
      // it" are different pictures and a link has to be able to name either
      p.set("axist", st.axis.t.toFixed(2));
      // only the OFF state is written. A link that says nothing about the null
      // opens with the null on, which is the only default R5 allows.
      if (!st.axis.showNull) p.set("axisnull", "0");
    }
    // a pin is written only while it is honest: the verified unit is still
    // the selection, it is still loading, or its link failed and is on screen
    if (pin.status === "ok" || pin.status === "pending") {
      p.set("model", pin.pin.datasetId);
      p.set("artifact", pin.pin.sha256);
      p.set("point", String(pin.pin.pointId));
      p.set("unit_kind", pin.pin.unitKind);
      p.set("unit_index", String(pin.pin.unitIndex));
    } else if (pin.status === "error" && pin.source === "link") {
      if (pin.pin) {
        p.set("model", pin.pin.datasetId);
        p.set("artifact", pin.pin.sha256);
        p.set("point", String(pin.pin.pointId));
        p.set("unit_kind", pin.pin.unitKind);
        p.set("unit_index", String(pin.pin.unitIndex));
      } else {
        for (const [k, v] of bootPinRaw) p.set(k, v);
      }
    }
  } else if (st.page === "behavior") {
    if (st.behavior.view !== "landscape") p.set("bview", st.behavior.view);
    if (st.behavior.cue) p.set("cue", st.behavior.cue);
    if (st.behavior.query.trim()) p.set("q", st.behavior.query);
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
    if (st.tour.id.startsWith("lesson:")) {
      p.set("lesson", st.tour.id.slice("lesson:".length));
      p.set("lesson_step", String(st.tour.step));
    } else {
      p.set("episode", st.tour.id);
      p.set("step", String(st.tour.step));
    }
  }
  if (st.returnTo) p.set("return", st.returnTo);
  return `#${p.toString()}`;
}

/** Keep the hash mirroring the store. Debounced a microtask so a burst of
 *  store writes (dataset switch clears selection etc.) coalesces into one
 *  write. NOT rAF-debounced: browsers freeze rAF entirely in occluded tabs,
 *  which would leave the hash stale exactly when a user copies a link from a
 *  backgrounded window. */
export function startUrlSync(): void {
  bootModel = readUrlState().model ?? null;
  const raw = new URLSearchParams(location.hash.replace(/^#/, ""));
  bootPinRaw = PIN_KEYS.filter((k) => raw.has(k)).map((k) => [k, raw.get(k)!]);
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
