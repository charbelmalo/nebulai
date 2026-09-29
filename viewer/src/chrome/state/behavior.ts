/** state/behavior.ts — signal mirror for the BehaviorSlice: which cue is open,
 *  how the cue set is being shown, what is filtered, and the reader-facing
 *  flags. Read-only, like every signal in this directory; the Behavior page
 *  writes through the store's setters and hears the result back here. */

import { signal } from "@preact/signals";
import { appStore, type BehaviorUI } from "../../app/store";

export const $behavior = signal<BehaviorUI>(appStore.getState().behavior);
