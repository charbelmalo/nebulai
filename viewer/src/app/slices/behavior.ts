/** behavior.ts — the Behavior page: which cue is open, how the cue set is being
 *  shown, what is filtered, and the reader-facing knobs that belong in Settings.
 *
 *  This slice holds UI state only. The study artifact itself is fetched by
 *  `data/behavior.ts` and never enters the store: it is immutable, it is
 *  potentially a few MB, and putting it here would make every unrelated store
 *  write re-notify subscribers holding it.
 *
 *  One field deserves its own paragraph. `showIndeterminate` defaults to TRUE
 *  and the Settings row that hides it is worded as a declutter, not as a
 *  filter, because "indeterminate" is a real answer in this study — a cue whose
 *  q could not be computed is not a cue with no effect. A default that hid
 *  those would quietly convert "we could not tell" into "nothing there", which
 *  is the exact failure the plan's claim contract (§1) exists to prevent. */

import type { StateCreator } from "zustand";
import type { AppState } from "../store";

/** How the cue set is presented. `landscape` is the fixed PCA scatter,
 *  `ranked` the sorted list, `table` the full metric grid. All three read the
 *  same cues; none of them is a different analysis. */
export type BehaviorView = "landscape" | "ranked" | "table";

/** Which cues are drawn. `all` includes gated and missing cues as their own
 *  marks; `measured` keeps only cues with a computed effect. Gated and missing
 *  are never silently dropped — choosing to hide them is an explicit act with
 *  a visible control, because the count of what is NOT measured is part of
 *  reading the study honestly. */
export type BehaviorFilter = "all" | "measured" | "significant";

export interface BehaviorUI {
  view: BehaviorView;
  /** the open cue in the inspector; "" = none */
  cue: string;
  /** free-text search across cue, stratum, pack and top associates */
  query: string;
  filter: BehaviorFilter;
  /** ranked/table sort key */
  sort: "effect" | "q" | "alpha" | "reliability";
  /** draw cues whose significance is indeterminate (see the header note) */
  showIndeterminate: boolean;
  /** show the read-only sample responses panel for the open cue */
  showSamples: boolean;
  /** reveal text a cue pack marked sensitive; off by default (§8.6) */
  revealSensitive: boolean;
}

export interface BehaviorSlice {
  behavior: BehaviorUI;
  setBehaviorView(view: BehaviorView): void;
  setBehaviorCue(cue: string): void;
  setBehaviorQuery(query: string): void;
  setBehaviorFilter(filter: BehaviorFilter): void;
  setBehaviorSort(sort: BehaviorUI["sort"]): void;
  setBehaviorFlag(key: "showIndeterminate" | "showSamples" | "revealSensitive", on: boolean): void;
}

export const createBehaviorSlice: StateCreator<AppState, [], [], BehaviorSlice> = (set) => ({
  behavior: {
    view: "landscape",
    cue: "",
    query: "",
    filter: "all",
    sort: "effect",
    showIndeterminate: true,
    showSamples: false,
    revealSensitive: false,
  },

  setBehaviorView: (view) => set((s) => ({ behavior: { ...s.behavior, view } })),
  /*  Opening a cue does NOT clear the query: the reader got here by searching,
   *  and dropping the search on click would make "find daddy, then look at it"
   *  a round trip instead of two clicks. */
  setBehaviorCue: (cue) => set((s) => ({ behavior: { ...s.behavior, cue } })),
  setBehaviorQuery: (query) => set((s) => ({ behavior: { ...s.behavior, query } })),
  setBehaviorFilter: (filter) => set((s) => ({ behavior: { ...s.behavior, filter } })),
  setBehaviorSort: (sort) => set((s) => ({ behavior: { ...s.behavior, sort } })),
  setBehaviorFlag: (key, on) => set((s) => ({ behavior: { ...s.behavior, [key]: on } })),
});
