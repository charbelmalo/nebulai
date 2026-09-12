/** shell.ts — the frame both instruments live inside: which instrument is
 *  running, what the device can do, the global render/quality settings, which
 *  page is showing, and whether the Settings overlay is up. Nothing here is
 *  specific to Nebulai's semantic maps or to Seer's agent runs, which is
 *  exactly why it is its own slice: both entries — `src/main.ts` and
 *  `src/seer-main.ts` — compose THIS one, and neither has to drag the other's
 *  data slices along to get a nav bar.
 *
 *  `setSetting` reaches across the slice boundary into `interpSelection` — see
 *  the comment on that branch for why. That cross-write is legal only because
 *  every StateCreator in this directory is typed against the full `AppState`
 *  rather than its own slice; keep it that way. */

import type { StateCreator } from "zustand";
import type { Capabilities } from "@psychix/viz/capabilities";
import type { AppState } from "../store";

/** Which instrument this document is. Set once at boot by the entry module
 *  (`main.ts` → "nebulai", `seer-main.ts` → "seer") and never again: it is the
 *  identity of the page, not a mode you can toggle. Crossing between the two
 *  is a navigation to the other entry's HTML, not a store write. */
export type AppId = "nebulai" | "seer";

/** Top-level page — nav-bar controlled. `map` is the semantic cloud (all the
 *  driver-backed views); `snapshot` is the per-topic conversation-log map
 *  (shown as "Topics"); `interp` is the Internals gallery (mechanistic-
 *  interpretability drivers, each rendering one real computed quantity from an
 *  interp bundle); `guide` documents the exact math + source data behind every
 *  live feature; `sessions` is the transcript plotter (shown as
 *  "Transcripts"); `seer` is SessionSeer's live view — capture and comparison
 *  of Codex / Claude / Hermes agent runs, served by `seer serve`.
 *
 *  `behavior` is the association study of docs/BEHAVIORAL-DIVERGENCE-PLAN.md.
 *  It is a page rather than a view mode because it reads a different artifact
 *  (`out/behavior/behavior.json`), makes a different kind of claim, and must
 *  never be mistaken for the weight-geometry maps: everything else in this
 *  instrument answers "what can this layer write", and this one answers "what
 *  did these deployments actually say". Keeping them on separate pages is what
 *  stops a reader carrying a causal reading from one to the other.
 *
 *  The union stays all seven on purpose: ONE shell type serves both
 *  instruments, so the chrome, the permalink layer and the signal bridge stay
 *  single. Which of the seven a given document may actually reach is
 *  `APP_PAGES`. */
export type Page =
  | "map"
  | "behavior"
  | "snapshot"
  | "interp"
  | "guide"
  | "sessions"
  | "seer";

/** The pages each instrument owns, in nav order — the authority for both
 *  "what may `setPage` accept" and "where does this app boot". Labels are NOT
 *  here: they are chrome, and live in `chrome/apps/nav.ts`, which is pinned
 *  against this table by tests/unit/app-pages.test.ts so the two cannot drift.
 *
 *  `setPage` silently drops a page the running app does not own rather than
 *  throwing: the callers are a permalink and a nav click, and neither has a
 *  sensible failure mode beyond "stay where you are". */
export const APP_PAGES: Record<AppId, readonly Page[]> = {
  nebulai: ["map", "behavior", "interp", "guide"],
  seer: ["seer", "sessions", "snapshot"],
};

/** Global render-quality settings — live-applied. Per-view appearance knobs
 *  live in `appearance` below; general chrome/render knobs live here. */
export interface Settings {
  pointScale: number; // × base point size, 0.5–2
  bloom: boolean; // post bloom (webgpu tier only; forced off by reduced motion)
  confidenceFloor: number; // hide points below this confidence, 0–1
  theme: "dark" | "light" | "auto";
  labelDensity: number; // 0.2–2 — culling threshold multiplier for cluster labels
  animationSpeed: number; // 0.25–2 — global time-uniform multiplier
  reducedMotion: boolean; // manual override; caps.reducedMotion still wins on init
  /** Internals cross-view linking: clicking a head/token/SAE feature in one
   *  view highlights it in every other view that shows the same unit. */
  crossLink: boolean;
  /** Webcam hand control (src/hands). Off by default and never auto-enabled:
   *  turning it on is what triggers the camera permission prompt and the ~19 MB
   *  of MediaPipe runtime + model, so it has to be an explicit act. */
  handTracking: boolean;
  /** Whether the free — non-steering — hand may cast the two visual effects.
   *  Off by default: the rig's job is navigation, and an effect fired by someone
   *  who was trying to pan is the failure that made the previous vocabulary
   *  unusable. Held here rather than on the rig because it is a preference that
   *  must survive the tracker being torn down and rebuilt, and because the
   *  Settings UI has to read it without importing the tracker. */
  handEffects: boolean;
}

export interface ShellSlice {
  app: AppId;
  capabilities: Capabilities | null;
  settings: Settings;
  settingsOpen: boolean; // Settings page overlay visibility
  page: Page;

  /** Declare which instrument is running and land on its first page. Called
   *  exactly once, by the entry module, before the chrome mounts. */
  setApp(app: AppId): void;
  setCapabilities(c: Capabilities): void;
  setSetting<K extends keyof Settings>(key: K, value: Settings[K]): void;
  setSettingsOpen(open: boolean): void;
  setPage(p: Page): void;
}

export const createShellSlice: StateCreator<AppState, [], [], ShellSlice> = (set) => ({
  // Nebulai is the default only because a store constructed outside an entry
  // (unit tests, the signal bridge's seed snapshot) has to be *something*; the
  // entry module overwrites it before the chrome mounts.
  app: "nebulai",
  capabilities: null,
  settings: {
    pointScale: 1,
    bloom: true,
    confidenceFloor: 0,
    theme: "dark",
    labelDensity: 1,
    animationSpeed: 1,
    reducedMotion: false,
    crossLink: true,
    handTracking: false,
    handEffects: false,
  },
  settingsOpen: false,
  page: "map",

  setApp: (app) => set({ app, page: APP_PAGES[app][0]! }),
  setCapabilities: (capabilities) => set({ capabilities }),
  setSetting: (key, value) =>
    set((s) => ({
      settings: { ...s.settings, [key]: value },
      // switching cross-view linking off also drops the live pick — a frozen
      // highlight with no way to change it would read as data, not UI state
      ...(key === "crossLink" && value === false ? { interpSelection: null } : {}),
    })),
  setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
  // a page the running instrument does not own is dropped, not thrown on —
  // see APP_PAGES. Nebulai can never land on `seer`, Seer can never land on
  // `map`, whichever way the request arrived (nav click or permalink).
  setPage: (page) =>
    set((s) => (APP_PAGES[s.app].includes(page) ? { page } : {})),
});
