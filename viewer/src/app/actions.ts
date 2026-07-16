/** Chrome → app command bridge. Chrome components stay dumb: they call
 *  these; main.ts (which owns the driver + loader) registers the handlers.
 *  Keeps Preact free of driver/loader imports and the dependency arrow
 *  pointing one way: chrome → store/actions ← app shell. */

import type { ViewMode } from "./store";

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
