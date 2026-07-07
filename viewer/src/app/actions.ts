/** Chrome → app command bridge. Chrome components stay dumb: they call
 *  these; main.ts (which owns the driver + loader) registers the handlers.
 *  Keeps Preact free of driver/loader imports and the dependency arrow
 *  pointing one way: chrome → store/actions ← app shell. */

import type { ViewMode } from "./store";

export interface AppActions {
  switchDataset(id: string): Promise<void>;
  switchViewMode(mode: ViewMode): Promise<void>;
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
