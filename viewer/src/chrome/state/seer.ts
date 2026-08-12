/** state/seer.ts — signal mirror for the SeerSlice: the SessionSeer capture
 *  server the Settings page edits. Separate from `$probing` (Nebulai's
 *  endpoints) for the same reason the slices are — see slices/seer.ts. */

import { signal } from "@preact/signals";
import { appStore, type SeerConfig } from "../../app/store";

const s = appStore.getState();

export const $seer = signal<SeerConfig>(s.seer);
