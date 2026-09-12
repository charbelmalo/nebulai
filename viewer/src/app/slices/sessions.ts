/** sessions.ts — Seer's Live page: the analysed agent-run transcripts the 3-D
 *  plotter draws as trajectories, and which of them are currently overlaid.
 *
 *  Only DERIVED summaries live here, never raw transcript text, and they
 *  outlive the tab: `chrome/sessionStore.ts` persists them to IndexedDB and
 *  rehydrates on boot, which is what `hydrated` gates on.
 *
 *  NOT HERE, deliberately: `resetSessionsAppearance`. It reads as a sessions
 *  action but it writes `appearance.sessions`, so it belongs to the appearance
 *  slice that owns that field — see slices/appearance.ts, which also explains
 *  why `Appearance` has not been cut along the product line yet. */

import type { StateCreator } from "zustand";
import type { SessionAnalysis } from "../../chrome/sessionlog";
import type { AppState } from "../store";

/** Sessions-page state — analysed agent-mode session transcripts (rich, real
 *  quantities), which the 3-D plotter renders as trajectories. `analyses` are
 *  DERIVED summaries (never raw text); they persist to IndexedDB across app
 *  sessions via `chrome/sessionStore.ts` and rehydrate on boot. `activeIds`
 *  selects which sessions are overlaid on the plot. */
export interface SessionsState {
  analyses: SessionAnalysis[];
  activeIds: string[];
  hydrated: boolean; // true once the IndexedDB rehydrate pass has run

  // ── Attractors P2 (D4/D5): the persona coordinate system ────────────────
  /** The space every trajectory is currently placed in, or null for the
   *  field's own derived axes. A space whose permutation control did not clear
   *  its null may be LOADED and looked at but is never set here by default —
   *  `data/persona.ts#isSelectableAsDefault` is the one place that decides,
   *  and `personaVerdict` carries the answer to whatever is drawing. */
  personaSpaceId: string | null;
  /** The selected space's control verdict, so a panel can state it without
   *  refetching the space. `unknown` covers both "no space" and "a space whose
   *  control was never run" — the panel text tells those apart, the geometry
   *  treats both as not-a-default. */
  personaVerdict: "above_null" | "at_null" | "below_null" | "unknown";
  /** Placements by run id. Each is the verbatim `placement.json` document:
   *  coordinates for the turns that could be placed, and a `skipped` list with
   *  a reason per turn that could not. A turn missing from `points` is never
   *  drawn at the origin. */
  placements: Record<string, PlacementDoc>;

  // ── Attractors P3: the fan ───────────────────────────────────────────────
  /** The ensemble currently on screen, or null when a single run is. Selecting
   *  one does NOT deselect the runs: an ensemble is a way of reading the runs
   *  that are already there, and the fan is drawn over them. */
  ensembleId: string | null;
  /** The ensemble's member run ids, verbatim from its document. The field
   *  draws the fan over the ones it actually has and reports how many that
   *  was; it never infers membership from whatever runs are on screen,
   *  because membership is a claim about protocol identity that only the
   *  ensemble document can make.
   *
   *  NOT here: whether the envelope is drawn. That is a look, so it lives in
   *  `appearance.sessions.showEnvelope` with the rest of the field's knobs
   *  and has exactly one home in Settings. */
  ensembleRunIds: string[];
}

/** `placement.json` as the viewer reads it — `seer place` writes it, the seer
 *  server serves it at `/seer/run/<id>/placement`, and nothing in between
 *  reinterprets it. */
export interface PlacementDoc {
  run_id: string;
  space_id: string;
  model: string;
  revision: string;
  layer: number;
  /** R7: the viewer picks the glyph off this field and nothing else. */
  placement_source: string;
  fidelity: string;
  verdict: string;
  pc1_evr: number | null;
  pc1_evr_null_p95: number | null;
  n_points: number;
  n_skipped: number;
  n_dropped_by_policy: number;
  n_missing: number;
  points: { index: number; seq: number; coords: [number, number]; role?: string }[];
  skipped: { index: number; reason?: string; fidelity?: string }[];
}

export interface SessionsSlice {
  sessions: SessionsState;

  setSessionAnalyses(list: SessionAnalysis[]): void;
  addSessionAnalysis(a: SessionAnalysis): void;
  removeSessionAnalysis(id: string): void;
  toggleSessionActive(id: string): void;
  clearSessionAnalyses(): void;
  setSessionsHydrated(v: boolean): void;

  /** Select the persona space, with the verdict that came with it. Passing a
   *  verdict other than `above_null` is allowed on purpose — a failed space is
   *  viewable — but the caller has to have LOOKED at the verdict to call this,
   *  and whatever draws can then state it. */
  setPersonaSpace(spaceId: string | null, verdict?: SessionsState["personaVerdict"]): void;
  setPlacement(runId: string, doc: PlacementDoc | null): void;

  /** Put an ensemble on screen, or clear it with null. `runIds` are the
   *  document's own members; omitting them clears the group rather than
   *  keeping the last ensemble's. */
  setEnsemble(ensembleId: string | null, runIds?: string[]): void;
}

export const createSessionsSlice: StateCreator<AppState, [], [], SessionsSlice> = (set) => ({
  sessions: {
    analyses: [],
    activeIds: [],
    hydrated: false,
    personaSpaceId: null,
    personaVerdict: "unknown",
    placements: {},
    ensembleId: null,
    ensembleRunIds: [],
  },

  // ── sessions (3-D plotter) ───────────────────────────────────────────────
  setSessionAnalyses: (list) =>
    set((s) => ({
      sessions: {
        ...s.sessions,
        analyses: list,
        // keep any still-present active ids; default to showing the newest one
        activeIds: (() => {
          const ids = new Set(list.map((a) => a.id));
          const kept = s.sessions.activeIds.filter((id) => ids.has(id));
          if (kept.length) return kept;
          const first = list[0]?.id;
          return first ? [first] : [];
        })(),
        hydrated: true,
      },
    })),
  addSessionAnalysis: (a) =>
    set((s) => {
      // de-dup by id; newest first so the list reads most-recent-on-top
      const analyses = [a, ...s.sessions.analyses.filter((x) => x.id !== a.id)];
      return {
        sessions: {
          ...s.sessions,
          analyses,
          activeIds: [a.id, ...s.sessions.activeIds.filter((id) => id !== a.id)],
        },
      };
    }),
  removeSessionAnalysis: (id) =>
    set((s) => ({
      sessions: {
        ...s.sessions,
        analyses: s.sessions.analyses.filter((a) => a.id !== id),
        activeIds: s.sessions.activeIds.filter((x) => x !== id),
      },
    })),
  toggleSessionActive: (id) =>
    set((s) => ({
      sessions: {
        ...s.sessions,
        activeIds: s.sessions.activeIds.includes(id)
          ? s.sessions.activeIds.filter((x) => x !== id)
          : [...s.sessions.activeIds, id],
      },
    })),
  clearSessionAnalyses: () =>
    set((s) => ({ sessions: { ...s.sessions, analyses: [], activeIds: [] } })),
  setSessionsHydrated: (hydrated) =>
    set((s) => ({ sessions: { ...s.sessions, hydrated } })),

  setPersonaSpace: (personaSpaceId, personaVerdict = "unknown") =>
    set((s) => ({
      sessions: {
        ...s.sessions,
        personaSpaceId,
        personaVerdict: personaSpaceId ? personaVerdict : "unknown",
        // placements belong to a space; keeping them across a space change
        // would draw last space's coordinates under this space's name
        placements: personaSpaceId === s.sessions.personaSpaceId ? s.sessions.placements : {},
      },
    })),
  setPlacement: (runId, doc) =>
    set((s) => {
      const placements = { ...s.sessions.placements };
      if (doc) placements[runId] = doc;
      else delete placements[runId];
      return { sessions: { ...s.sessions, placements } };
    }),
  setEnsemble: (ensembleId, runIds) =>
    set((s) => ({
      sessions: { ...s.sessions, ensembleId, ensembleRunIds: ensembleId ? (runIds ?? []) : [] },
    })),
});
