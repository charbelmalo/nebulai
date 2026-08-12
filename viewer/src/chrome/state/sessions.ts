/** state/sessions.ts — signal mirror for the SessionsSlice: the analysed agent
 *  runs and which of them are overlaid on the 3-D plot. Note that the Sessions
 *  LOOK is not here — those knobs live under `appearance.sessions`, so the
 *  Sessions page reads both `$sessions` and `$appearance`. */

import { signal } from "@preact/signals";
import { appStore, type SessionsState } from "../../app/store";

const s = appStore.getState();

export const $sessions = signal<SessionsState>(s.sessions);
