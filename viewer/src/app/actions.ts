/** Chrome → app command bridge. Chrome components stay dumb: they call
 *  these; main.ts (which owns the driver + loader) registers the handlers.
 *  Keeps Preact free of driver/loader imports and the dependency arrow
 *  pointing one way: chrome → store/actions ← app shell. */

import type { PinSource, ViewMode } from "./store";
import type { Finding, UnitPin } from "../data/finding";

export interface AppActions {
  switchDataset(id: string): Promise<void>;
  switchViewMode(mode: ViewMode): Promise<void>;
  /** After a build finishes: re-fetch out/index.json (cache-busted), evict the
   *  rebuilt dataset and hot-swap to it — bypasses switchDataset's same-id
   *  early-return so rebuilding the currently shown map refreshes in place. */
  refreshDatasets(datasetId: string): Promise<void>;
  /** Camera fly-tos on the atlas (search-result / legend clicks). No-ops off
   *  the atlas view — the other views have no camera to fly. */
  flyToCluster(id: number): void;
  flyToPoint(id: number): void;
  /** Compare's layout tour. The driver owns the playhead (it lives on a GPU
   *  uniform), so these are commands, not store writes — the tour reports back
   *  through `$compareTour`. No-ops off the compare view. */
  compareTour(cmd: CompareTourCommand): void;
  /** Play one step of an episode (P5).
   *
   *  A step may name a different model from the one on screen, and loading a
   *  model is asynchronous, which is exactly why this cannot live in
   *  `tours.ts`: the store write that installs a dataset must happen here,
   *  where the loader is. It installs the dataset with `{ keepTour: true }` so
   *  the episode driving the switch survives its own step — every other caller
   *  still clears the tour, because a tour's captions name units of the model
   *  they were written for.
   *
   *  Resolves once the step is fully applied, so a caller can await the step
   *  before advancing rather than racing the fetch. */
  runEpisodeStep(episodeId: string, step: number): Promise<void>;
  /** Repeat whatever failed last: the index fetch, or the dataset request
   *  recorded in `loadError`. A no-op when nothing failed. */
  retryLoad(): Promise<void>;
  /** Open the curated starter map (manifest `default_atlas`, else the
   *  unverified default). Used when the map page opens with nothing loaded. */
  openStarter(): Promise<void>;
  /** Open one exact unit: resolve (dataset, digest) through the trusted
   *  manifest, load those bytes, find the identity tuple, and select it on
   *  the plain atlas. `finding` (an imported record) additionally has its
   *  evidence checked. Every outcome lands in `pin`; a failure never changes
   *  the map on screen. */
  openPinned(pin: UnitPin, source: PinSource, finding?: Finding): Promise<void>;
}

export type CompareTourCommand =
  | { kind: "toggle" }
  | { kind: "play" }
  | { kind: "pause" }
  | { kind: "restart" }
  | { kind: "seek"; u: number }
  | { kind: "speed"; mult: number }
  /** A layout radio was picked: leave the tour and morph there.
   *
   *  This exists because the store alone cannot carry the intent. The radio
   *  shows the TOUR's stage, not the store's, so picking the layout the store
   *  already holds is a no-op change — the driver would never hear it and the
   *  tour would keep playing under a radio the user just clicked. */
  | { kind: "pick"; state: number };

/** A snapshot of the compare layout tour. It lives here rather than in the
 *  driver so the transport UI can read it without chrome importing a driver —
 *  this module is the sanctioned meeting point between the two halves.
 *
 *  NOTE what this is NOT: compare has no time axis, so unlike the sessions
 *  playhead this progress is a position in a PRESENTATION, not a clock. The
 *  readout must name layout states; showing it as seconds would invent a
 *  quantity the data doesn't have. */
export interface CompareTourState {
  playing: boolean;
  /** 0..1 through the whole tour */
  progress: number;
  /** index of the layout being shown (the one being left, mid-transition) */
  stage: number;
  /** raw state key, e.g. "semantic" */
  stageName: string;
  /** 0 while parked on `stage`, →1 as it morphs into `stage + 1` */
  blend: number;
  /** number of layout states in the tour */
  stages: number;
  /** progress (0..1) at which each layout is fully settled — the driver owns
   *  the dwell/transition weighting, so it hands the exact stop positions over
   *  rather than making the UI guess them and draw ticks that lie. */
  stops: number[];
  speed: number;
  hasData: boolean;
}

let handlers: AppActions | null = null;

export function registerActions(a: AppActions): void {
  handlers = a;
}

export function requestDataset(id: string): void {
  handlers?.switchDataset(id).catch((e) => console.error("[nebulai] dataset switch failed", e));
}

export function requestViewMode(mode: ViewMode): void {
  handlers?.switchViewMode(mode).catch((e) => console.error("[nebulai] view switch failed", e));
}

export function requestRefreshDatasets(datasetId: string): void {
  handlers
    ?.refreshDatasets(datasetId)
    .catch((e) => console.error("[nebulai] dataset refresh failed", e));
}

export function requestFlyToCluster(id: number): void {
  handlers?.flyToCluster(id);
}

export function requestFlyToPoint(id: number): void {
  handlers?.flyToPoint(id);
}

export function requestCompareTour(cmd: CompareTourCommand): void {
  handlers?.compareTour(cmd);
}

export function requestRetryLoad(): void {
  handlers?.retryLoad().catch((e) => console.error("[nebulai] retry failed", e));
}

export function requestStarter(): void {
  handlers?.openStarter().catch((e) => console.error("[nebulai] starter failed", e));
}

export function requestOpenPinned(pin: UnitPin, source: PinSource, finding?: Finding): void {
  handlers?.openPinned(pin, source, finding).catch((e) => console.error("[nebulai] pinned open failed", e));
}

/** Fire-and-forget an episode step (for click handlers). */
export function requestEpisodeStep(episodeId: string, step: number): void {
  handlers
    ?.runEpisodeStep(episodeId, step)
    .catch((e) => console.error("[nebulai] episode step failed", e));
}

/** Awaitable form, for callers that must know the step landed (URL restore,
 *  tests) rather than assuming it did. Resolves immediately when no shell has
 *  registered handlers — a chrome-only render has nothing to drive. */
export function runEpisodeStep(episodeId: string, step: number): Promise<void> {
  return handlers?.runEpisodeStep(episodeId, step) ?? Promise.resolve();
}
