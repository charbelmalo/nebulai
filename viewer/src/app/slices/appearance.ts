/** appearance.ts — per-view look knobs, the single source of truth for how
 *  every graph type draws itself.
 *
 *  THIS IS THE ONE REMAINING CROSS-PRODUCT SEAM. `Appearance` straddles both
 *  instruments: `atlas` / `chord` / `hierarchy` / `compare` are Nebulai's, and
 *  `sessions` is Seer's. It stays whole on purpose. `setAppearance` is generic
 *  over `<G extends keyof Appearance, K extends keyof Appearance[G]>`, so the
 *  interface IS the contract every Settings-page tab binds through — cutting it
 *  in two today would mean two setters, two tab wirings and a rewrite of
 *  SettingsPage, all to buy nothing until a second app entry actually exists.
 *  Step 6 of the segmentation plan narrows it per-app; until then, one slice.
 *
 *  `resetSessionsAppearance` lives here rather than in the sessions slice
 *  because it writes `appearance.sessions` — the slice that OWNS the field owns
 *  the writer, even when the field belongs to the other product. */

import type { StateCreator } from "zustand";
import {
  DEFAULT_SESSIONS_APPEARANCE,
  type SessionsAppearance,
} from "../../scene/sessions/appearance";
import type { AppState } from "../store";

/** Per-view appearance settings. Every graph type gets its own tab in the
 *  Settings page; every knob a driver honors MUST live here — this is the
 *  single source of truth for graph looks. New driver knobs belong in the
 *  matching sub-interface, not the driver's private state. */
export interface Appearance {
  atlas: {
    hullOpacity: number; // 0–1 — territory hull fill alpha
    beamWidth: number; // 0.25–3 — connection line width scale
    haloIntensity: number; // 0–1 — pulsing hub ring strength
    background: "vignette" | "flat" | "grid"; // stage background
    orbitEnabled: boolean; // slow camera orbit in 3D flythrough
    orbitSpeed: number; // 0.1–3
  };
  chord: {
    ribbonOpacity: number; // 0–1
    curveTension: number; // 0–1 (0 = straight, 1 = maximum bezier)
    labelRotation: boolean; // rotate rim labels tangentially
    showTicks: boolean;
  };
  hierarchy: {
    linkStroke: number; // 0.5–3
    nodeSize: number; // 0.5–3
    fanAngle: number; // 60–360 — arc span in degrees
    colorBy: "cluster" | "depth" | "confidence";
  };
  compare: {
    swatchSize: number; // 4–20 px
    strokeOnHover: boolean;
    dimOthers: boolean; // when a model is highlighted, dim the rest
  };
  /** Sessions emissive-particle field. Full knob set + default palette live in
   *  scene/sessions/appearance.ts (kept out of this file so the driver, which
   *  imports the store, and the store don't form an import cycle). */
  sessions: SessionsAppearance;
}

export interface AppearanceSlice {
  appearance: Appearance;

  setAppearance<G extends keyof Appearance, K extends keyof Appearance[G]>(
    graph: G,
    key: K,
    value: Appearance[G][K],
  ): void;
  /** Restore every Sessions appearance knob to its shipped default in one shot
   *  (the tab's Reset button — setAppearance is per-key). */
  resetSessionsAppearance(): void;
}

export const createAppearanceSlice: StateCreator<AppState, [], [], AppearanceSlice> = (set) => ({
  appearance: {
    atlas: {
      hullOpacity: 0.28,
      beamWidth: 1,
      haloIntensity: 0.7,
      background: "vignette",
      orbitEnabled: false,
      orbitSpeed: 1,
    },
    chord: {
      ribbonOpacity: 0.55,
      curveTension: 0.6,
      labelRotation: true,
      showTicks: true,
    },
    hierarchy: {
      linkStroke: 1,
      nodeSize: 1,
      fanAngle: 300,
      colorBy: "cluster",
    },
    compare: {
      swatchSize: 10,
      strokeOnHover: true,
      dimOthers: true,
    },
    sessions: { ...DEFAULT_SESSIONS_APPEARANCE },
  },

  setAppearance: (graph, key, value) =>
    set((s) => ({
      appearance: {
        ...s.appearance,
        [graph]: { ...s.appearance[graph], [key]: value },
      },
    })),
  resetSessionsAppearance: () =>
    set((s) => ({
      appearance: {
        ...s.appearance,
        sessions: {
          ...DEFAULT_SESSIONS_APPEARANCE,
          categoryColors: { ...DEFAULT_SESSIONS_APPEARANCE.categoryColors },
        },
      },
    })),
});
