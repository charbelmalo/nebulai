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
}

export interface SessionsSlice {
  sessions: SessionsState;

  setSessionAnalyses(list: SessionAnalysis[]): void;
  addSessionAnalysis(a: SessionAnalysis): void;
  removeSessionAnalysis(id: string): void;
  toggleSessionActive(id: string): void;
  clearSessionAnalyses(): void;
  setSessionsHydrated(v: boolean): void;
}

export const createSessionsSlice: StateCreator<AppState, [], [], SessionsSlice> = (set) => ({
  sessions: { analyses: [], activeIds: [], hydrated: false },

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
});
