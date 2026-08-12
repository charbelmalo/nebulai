/** snapshot.ts — Seer's Transcripts/Topics surface: parsed conversation logs,
 *  the saved topic filters the snapshot map watches for, and the playhead over
 *  a log's turns.
 *
 *  Everything here is client-side and in-memory — the raw log text never
 *  leaves the tab and never reaches the store beyond what a `ConversationTurn`
 *  holds. This slice is entirely Seer's; Nebulai composes none of it. */

import type { StateCreator } from "zustand";
import type { AppState } from "../store";

/** One saved topic filter — a named bag of keywords the snapshot map watches
 *  for in conversation logs. Ships with a couple of defaults (design,
 *  shaders). Users add more from either the Snapshot Map page or Settings. */
export interface TopicPreset {
  id: string;
  name: string;
  keywords: string[];
}

/** A parsed conversation log — a sequence of turns with role + text. The
 *  snapshot pipeline runs entirely client-side; the raw log is kept in memory
 *  only. */
export interface ConversationTurn {
  role: string;
  text: string;
  ts?: number;
}

export interface SnapshotLog {
  id: string;
  name: string;
  turns: ConversationTurn[];
  loadedAt: number;
}

export interface SnapshotState {
  logs: SnapshotLog[];
  activeLogId: string | null;
  topics: TopicPreset[];
  activeTopicId: string;
  turnIndex: number; // 0 = first turn, logs.turns.length-1 = last
  playing: boolean;
}

/** Preset topic filters shipped by default. Users can add more from the
 *  Snapshot Map page or the Settings → Snapshot tab. Keep lists tight — the
 *  match is case-insensitive substring; long lists dilute the signal. */
export const DEFAULT_TOPICS: TopicPreset[] = [
  {
    id: "design",
    name: "Design keywords",
    keywords: [
      "typography",
      "spacing",
      "radius",
      "grid",
      "tokens",
      "layout",
      "hover",
      "focus",
      "empty state",
      "component",
      "hierarchy",
      "affordance",
      "accessibility",
      "contrast",
      "keyboard",
    ],
  },
  {
    id: "shaders",
    name: "Shader effects",
    keywords: [
      "bloom",
      "vignette",
      "chromatic aberration",
      "SSAO",
      "godrays",
      "fresnel",
      "raymarch",
      "SDF",
      "post-processing",
      "TSL",
      "WGSL",
      "GLSL",
      "compute",
      "uniform",
      "vertex",
      "fragment",
    ],
  },
  {
    id: "interaction",
    name: "Interaction craft",
    keywords: [
      "hit target",
      "safe triangle",
      "aria-activedescendant",
      "focus trap",
      "escape",
      "arrow keys",
      "roving tabindex",
      "scroll padding",
      "submenu",
      "combobox",
      "tooltip",
      "dropdown",
    ],
  },
];

export interface SnapshotSlice {
  snapshot: SnapshotState;

  addSnapshotLog(log: SnapshotLog): void;
  removeSnapshotLog(id: string): void;
  setActiveLog(id: string | null): void;
  setActiveTopic(id: string): void;
  setTurnIndex(i: number): void;
  setPlaying(v: boolean): void;
  addTopicPreset(t: TopicPreset): void;
  updateTopicPreset(id: string, patch: Partial<TopicPreset>): void;
  removeTopicPreset(id: string): void;
}

export const createSnapshotSlice: StateCreator<AppState, [], [], SnapshotSlice> = (set) => ({
  snapshot: {
    logs: [],
    activeLogId: null,
    topics: DEFAULT_TOPICS,
    activeTopicId: DEFAULT_TOPICS[0]?.id ?? "",
    turnIndex: 0,
    playing: false,
  },

  addSnapshotLog: (log) =>
    set((s) => ({
      snapshot: {
        ...s.snapshot,
        logs: [...s.snapshot.logs, log],
        activeLogId: log.id,
        turnIndex: Math.max(0, log.turns.length - 1),
      },
    })),
  removeSnapshotLog: (id) =>
    set((s) => {
      const logs = s.snapshot.logs.filter((l) => l.id !== id);
      const activeLogId =
        s.snapshot.activeLogId === id ? (logs[0]?.id ?? null) : s.snapshot.activeLogId;
      return { snapshot: { ...s.snapshot, logs, activeLogId, turnIndex: 0 } };
    }),
  setActiveLog: (id) =>
    set((s) => ({ snapshot: { ...s.snapshot, activeLogId: id, turnIndex: 0 } })),
  setActiveTopic: (activeTopicId) =>
    set((s) => ({ snapshot: { ...s.snapshot, activeTopicId } })),
  setTurnIndex: (turnIndex) =>
    set((s) => ({ snapshot: { ...s.snapshot, turnIndex: Math.max(0, turnIndex) } })),
  setPlaying: (playing) => set((s) => ({ snapshot: { ...s.snapshot, playing } })),
  addTopicPreset: (t) =>
    set((s) => ({ snapshot: { ...s.snapshot, topics: [...s.snapshot.topics, t] } })),
  updateTopicPreset: (id, patch) =>
    set((s) => ({
      snapshot: {
        ...s.snapshot,
        topics: s.snapshot.topics.map((t) => (t.id === id ? { ...t, ...patch } : t)),
      },
    })),
  removeTopicPreset: (id) =>
    set((s) => {
      const topics = s.snapshot.topics.filter((t) => t.id !== id);
      const activeTopicId =
        s.snapshot.activeTopicId === id
          ? (topics[0]?.id ?? "")
          : s.snapshot.activeTopicId;
      return { snapshot: { ...s.snapshot, topics, activeTopicId } };
    }),
});
