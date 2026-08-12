/** interp.ts — Nebulai's Internals page: which mechanistic-interpretability
 *  driver owns the canvas, which forward trace it reads, the cross-view entity
 *  pick that links 25 charts into one instrument, and where the user is in a
 *  guided tour.
 *
 *  Two boundary crossings run through here and both are deliberate. Inbound:
 *  `atlas.setDataset` clears `interpSelection` and `tour` (unit ids and tour
 *  copy are per-model), and `shell.setSetting` clears `interpSelection` when
 *  cross-linking is switched off. Outbound: `setInterpSelection` READS
 *  `settings.crossLink` before it commits. Typing every StateCreator against
 *  the full `AppState` is what keeps all three legal. */

import type { StateCreator } from "zustand";
import type { AppState } from "../store";

/** Internals-page UI state. `featureId` selects which InterpDriver owns the
 *  interp canvas (must match a registered feature id in scene/interp/registry).
 *  The active model is read from `datasetId` — the bundles live per-model.
 *  `traceSlug` selects which per-prompt forward trace the forward-group features
 *  render (weight-group features ignore it). "" means "use the first trace". */
export interface InterpUI {
  featureId: string;
  traceSlug: string;
}

/** Cross-view entity selection (Internals). Clicking a head / token position /
 *  SAE feature in one view publishes it here; every other mounted or later-
 *  mounted view that knows the entity highlights it (registry `linksTo`
 *  declares which). This is what turns 25 charts into one instrument: pick
 *  L4H11 in Head Fingerprints and Composition Web / Induction / Ablation /
 *  OV Eigen light the same head up. Selection is a claim about IDENTITY only
 *  ("this is the same unit"), never about causality. */
export type InterpSelection =
  | { kind: "head"; layer: number; head: number }
  | { kind: "token"; pos: number }
  | { kind: "saeFeature"; id: number };

/** Pointer into a guided tour (chrome/tours.ts owns the tour content — the
 *  store only tracks WHERE the user is so chrome can render the overlay). */
export interface TourRef {
  id: string;
  step: number;
}

export function sameInterpSelection(
  a: InterpSelection | null,
  b: InterpSelection | null,
): boolean {
  if (a === null || b === null) return a === b;
  if (a.kind !== b.kind) return false;
  if (a.kind === "head" && b.kind === "head")
    return a.layer === b.layer && a.head === b.head;
  if (a.kind === "token" && b.kind === "token") return a.pos === b.pos;
  if (a.kind === "saeFeature" && b.kind === "saeFeature") return a.id === b.id;
  return false;
}

export interface InterpSlice {
  interp: InterpUI;
  /** null = nothing picked. Cleared on model switch (unit ids are per-model). */
  interpSelection: InterpSelection | null;
  /** Active guided tour (Internals) — which tour and which step. null = none.
   *  Cleared on model switch: tours quote model-specific bundle numbers. */
  tour: TourRef | null;

  setInterpFeature(id: string): void;
  setInterpTrace(slug: string): void;
  setInterpSelection(sel: InterpSelection | null): void;
  setTour(tour: TourRef | null): void;
}

export const createInterpSlice: StateCreator<AppState, [], [], InterpSlice> = (set) => ({
  interp: { featureId: "weight-spectrum", traceSlug: "" },
  interpSelection: null,
  tour: null,

  setInterpFeature: (featureId) =>
    set((s) => ({ interp: { ...s.interp, featureId } })),
  setInterpTrace: (traceSlug) =>
    set((s) => ({ interp: { ...s.interp, traceSlug } })),
  setInterpSelection: (interpSelection) =>
    set((s) => (s.settings.crossLink || interpSelection === null ? { interpSelection } : s)),
  setTour: (tour) => set({ tour }),
});
