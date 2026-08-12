/** state/appearance.ts — signal mirror for the AppearanceSlice. One signal for
 *  the whole tree: the store replaces the `appearance` object wholesale on
 *  every knob change, so a per-graph split here would buy no extra precision
 *  and would just multiply the identity checks the sync pass has to run. */

import { signal } from "@preact/signals";
import { appStore, type Appearance } from "../../app/store";

const s = appStore.getState();

export const $appearance = signal<Appearance>(s.appearance);
