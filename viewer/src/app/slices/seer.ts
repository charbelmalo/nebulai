/** seer.ts — the second instrument's own configuration.
 *
 *  Exactly one field today: the SessionSeer capture server that `seer serve`
 *  puts on :8125. It lived in `probing.seerUrl` until Seer got its own Vite
 *  entry, alongside Nebulai's live-probe (:8123) and build (:8124) servers.
 *  That was one endpoint list serving two instruments; now each owns its own,
 *  and a Seer document no longer carries the config for a map builder it
 *  cannot run.
 *
 *  The env var did NOT change: `VITE_SEER_URL` still names this endpoint with
 *  the same loopback default and the same static-deploy behaviour (an empty
 *  VITE_* blanks it out and the page becomes bring-your-own-endpoint — see
 *  the note in slices/probing.ts). Only the store path moved:
 *  `probing.seerUrl` → `seer.serverUrl`.
 *
 *  It is a one-field object rather than a bare `seerUrl: string` because this
 *  is where Seer's config accretes — a run-retention cap, a poll interval, a
 *  default agent — and a slice that has to be reshaped the first time it grows
 *  a second field was never worth the saved keystroke. */

import type { StateCreator } from "zustand";
import type { AppState } from "../store";

export interface SeerConfig {
  /** SessionSeer capture server base URL (`seer serve`). Empty = not
   *  configured; the client refuses to guess one rather than inventing a
   *  host — see `seerBase()` in src/seer/client.ts. */
  serverUrl: string;
}

export interface SeerSlice {
  seer: SeerConfig;

  setSeerConfig<K extends keyof SeerConfig>(key: K, value: SeerConfig[K]): void;
}

export const createSeerSlice: StateCreator<AppState, [], [], SeerSlice> = (set) => ({
  seer: {
    serverUrl: import.meta.env.VITE_SEER_URL ?? "http://127.0.0.1:8125",
  },

  setSeerConfig: (key, value) => set((s) => ({ seer: { ...s.seer, [key]: value } })),
});
