/** InterpDriver — the contract for the mechanistic-interpretability features
 *  (the "Internals" page). Same spirit as SceneDriver: one driver owns the
 *  interp canvas at a time; drivers coordinate only through the store and the
 *  shared canvas/overlay they're handed. Each driver renders exactly ONE real
 *  computed quantity from an interp bundle and exposes exact hover values.
 *
 *  Kept separate from SceneDriver because interp drivers take an overlay host
 *  for their tooltip/labels and are keyed to a model id (not a Dataset). */

import type { GpuTier } from "../../app/capabilities";
import type { InterpSelection } from "../../app/store";

export interface InterpDriver {
  /** When false, the host skips the per-frame RAF entirely — the view is static
   *  (deck redraws on demand) so there's nothing to animate and no reason to
   *  spin the main thread at 60fps. Defaults to animated when omitted. */
  readonly animated?: boolean;
  /** Init GPU/deck against the shared interp canvas. `overlay` is an absolutely
   *  positioned host for HTML tooltips/labels (HTML-first law). */
  init(canvas: HTMLCanvasElement, tier: GpuTier, overlay: HTMLElement): Promise<void>;
  /** Load + render the bundle for this model id (e.g. "gpt2"). Forward-group
   *  features also receive the selected `trace` slug (which per-prompt forward
   *  pass to render); weight-group features ignore it. May reject if the model
   *  has no bundle for this feature — the host surfaces that honestly. */
  setModel(model: string, trace?: string): Promise<void>;
  /** Cross-view entity link (optional opt-in): the host pushes the global
   *  `interpSelection` here on mount and on every change. A driver that knows
   *  the entity highlights it; publishing goes the other way — the driver
   *  calls `appStore.getState().setInterpSelection(...)` on click. Views that
   *  opt in declare it in the registry's `linksTo` so the rail can badge them. */
  setSelection?(sel: InterpSelection | null): void;
  /** dt ms, t elapsed seconds (pinned at 0 under ?frozen for goldens). */
  frame(dt: number, t: number): void;
  resize(width: number, height: number, dpr: number): void;
  dispose(): void;
}

export type InterpGroup = "weights" | "forward" | "sae" | "trained" | "live";

/** A single colored key in a feature's legend — `rgb` is a bare "r,g,b" string
 *  so it can drop straight into `rgb(...)` and MUST match the exact color the
 *  driver renders (the legend is a contract, not decoration). */
export interface LegendKey {
  label: string;
  rgb: string;
}

/** One entry in the Internals feature rail. `n` is the feature's number in the
 *  25-feature spec; `blurb` states the exact real quantity (shown as the
 *  legend subtitle and on /guide). Only IMPLEMENTED features are registered —
 *  the rail never lists a view that isn't backed by real data yet. `legend`
 *  and `note` are the encoding key shown in the on-canvas legend card. */
export interface InterpFeature {
  id: string;
  n: number;
  label: string;
  group: InterpGroup;
  blurb: string;
  /** The exact quantity/formula the view renders, in one plain-text line (kept
   *  ASCII-mathy, not LaTeX — it drops straight into /guide). REQUIRED so a live
   *  feature can never ship without stating its math. */
  math: string;
  /** Which bundle field(s) the numbers come from, and how they were computed
   *  offline (the provenance line on /guide). REQUIRED for the same reason. */
  source: string;
  legend?: LegendKey[];
  note?: string;
  /** Which corner the legend card docks to. Defaults to top-right; radial views
   *  set "bl", the two-column attention view sets "br" (a narrower card that
   *  tucks into the reserved right gutter), and views whose data hugs the right
   *  column set "tl" — always the corner that covers the least data. */
  legendCorner?: "tr" | "tl" | "bl" | "br";
  /** True when the view renders one bundled prompt at a time and should get the
   *  prompt selector, even outside the "forward" group (e.g. the SAE piano-roll
   *  renders per-prompt encoder activations but belongs to the sae group). All
   *  forward-group features are implicitly per-trace. */
  perTrace?: boolean;
  /** True for forward-group features that carry their OWN prompt set (e.g. the
   *  patching pairs) — suppresses the shared trace selector, which would
   *  otherwise imply it filters a view it doesn't affect. */
  ownPrompts?: boolean;
  /** True for full-bleed boards where the open legend card would sit on top of
   *  data in every corner — the legend then defaults to its collapsed pill
   *  (the user's toggle still overrides, sticky for the session). */
  legendCollapsed?: boolean;
  /** Entity kinds this view can follow via cross-view linking (implements
   *  `setSelection`). Only list kinds the driver REALLY highlights — the rail
   *  badges views that can follow the current pick, and a badge that does
   *  nothing would be a lie. */
  linksTo?: Array<InterpSelection["kind"]>;
  create: () => InterpDriver;
}
