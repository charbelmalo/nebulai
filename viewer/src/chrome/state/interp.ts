/** state/interp.ts — signal mirrors for the InterpSlice: which Internals
 *  feature owns the canvas, the cross-view entity pick, and the guided-tour
 *  pointer. Read-only mirrors, like every signal in this directory — chrome
 *  writes through the store's setters and hears the result back here. */

import { signal } from "@preact/signals";
import {
  appStore,
  type InterpSelection,
  type InterpUI,
  type TourRef,
} from "../../app/store";

const s = appStore.getState();

export const $interp = signal<InterpUI>(s.interp);
export const $interpSelection = signal<InterpSelection | null>(s.interpSelection);
export const $tour = signal<TourRef | null>(s.tour);
