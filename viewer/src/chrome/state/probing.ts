/** state/probing.ts — signal mirrors for the ProbingSlice: the endpoint
 *  config the Settings page edits, and the live pipeline progress the build /
 *  probe run reports back into it. */

import { signal } from "@preact/signals";
import { appStore, type Probing, type Progress } from "../../app/store";

const s = appStore.getState();

export const $probing = signal<Probing>(s.probing);
export const $progress = signal<Progress>(s.progress);
