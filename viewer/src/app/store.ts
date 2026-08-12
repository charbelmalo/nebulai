/** Shared app state — the coordination point between drivers, overlays, and
 *  chrome. Toolkits never share GPU contexts; they share THIS. Camera matrices
 *  land here in M1 so SVG/HTML overlays project through the same numbers the
 *  active driver renders with. */

/*  The state itself now lives in `slices/`, one module per product surface,
 *  and this file is two things: the composition that fuses them into the one
 *  `appStore` every consumer already imports, and the barrel that keeps every
 *  existing `from "../app/store"` import working untouched. Nothing was
 *  renamed and nothing moved out of reach — the decomposition is invisible
 *  from the outside on purpose.
 *
 *  Why slices at all: there are really two instruments in here. Nebulai (Map ·
 *  Internals · Guide) maps what a model knows; Seer (Live · Transcripts ·
 *  Topics) maps what an agent did. Splitting the store along that seam is what
 *  let the second app entry (`src/seer-main.ts`) happen: `shell` carries the
 *  `app` identity and `APP_PAGES`, and Seer's own endpoint config sits in its
 *  own `seer` slice rather than inside Nebulai's `probing`.
 *
 *  The two apps still share ONE store type. That is deliberate: the signal
 *  bridge, the permalink layer and the Settings overlay are single, and a
 *  slice a document never reads costs it a few default fields, not a bundle.
 *  What must NOT be shared is components and drivers — that separation is
 *  enforced at the entry/mount boundary (chrome/apps/), not here.
 *
 *  Every StateCreator in `slices/` is typed against the FULL `AppState`, not
 *  against its own slice. That is load-bearing, not stylistic: several actions
 *  legitimately write across the seam — `atlas.setDataset` clears the interp
 *  slice's `interpSelection` and `tour` because unit ids and tour copy are
 *  per-model, `shell.setSetting` clears `interpSelection` when cross-linking
 *  goes off, and `interp.setInterpSelection` reads `settings.crossLink` before
 *  it commits. Narrow the typing to a single slice and all three stop
 *  compiling.
 *
 *  The composition below has NO overlapping keys — each field and action is
 *  contributed by exactly one slice — so the spread order cannot change any
 *  default. */

import { createStore } from "zustand/vanilla";
import { createAppearanceSlice, type AppearanceSlice } from "./slices/appearance";
import { createAtlasSlice, type AtlasSlice } from "./slices/atlas";
import { createInterpSlice, type InterpSlice } from "./slices/interp";
import { createProbingSlice, type ProbingSlice } from "./slices/probing";
import { createSeerSlice, type SeerSlice } from "./slices/seer";
import { createSessionsSlice, type SessionsSlice } from "./slices/sessions";
import { createShellSlice, type ShellSlice } from "./slices/shell";
import { createSnapshotSlice, type SnapshotSlice } from "./slices/snapshot";

export interface AppState
  extends ShellSlice,
    AtlasSlice,
    InterpSlice,
    ProbingSlice,
    AppearanceSlice,
    SnapshotSlice,
    SessionsSlice,
    SeerSlice {}

export const appStore = createStore<AppState>()((...a) => ({
  ...createShellSlice(...a),
  ...createAtlasSlice(...a),
  ...createInterpSlice(...a),
  ...createProbingSlice(...a),
  ...createAppearanceSlice(...a),
  ...createSnapshotSlice(...a),
  ...createSessionsSlice(...a),
  ...createSeerSlice(...a),
}));

/*  ── barrel ────────────────────────────────────────────────────────────────
 *  Everything below is re-exported from its new home so the 40-odd consumers
 *  that import from this module keep compiling with no edit. Anything a
 *  consumer imports today MUST stay listed here; the slice modules are an
 *  implementation detail nobody outside `app/` should have to know about
 *  until the per-app stores land. */

export { sameInterpSelection } from "./slices/interp";
export { DEFAULT_TOPICS } from "./slices/snapshot";

export { APP_PAGES } from "./slices/shell";
export type { AppId, Page, Settings, ShellSlice } from "./slices/shell";
export type {
  AtlasSlice,
  CompareUI,
  MapQuery,
  Selection,
  Toggles,
  ViewMode,
} from "./slices/atlas";
export type { InterpSelection, InterpSlice, InterpUI, TourRef } from "./slices/interp";
export type { BuildParams, ProbeStage, Probing, ProbingSlice, Progress } from "./slices/probing";
export type { Appearance, AppearanceSlice } from "./slices/appearance";
export type {
  ConversationTurn,
  SnapshotLog,
  SnapshotSlice,
  SnapshotState,
  TopicPreset,
} from "./slices/snapshot";
export type { SessionsSlice, SessionsState } from "./slices/sessions";
export type { SeerConfig, SeerSlice } from "./slices/seer";
