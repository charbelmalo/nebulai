/** state/snapshot.ts — signal mirror for the SnapshotSlice: the loaded
 *  conversation logs, the topic presets and the turn playhead, all in one
 *  object because the store replaces `snapshot` wholesale on every write. */

import { signal } from "@preact/signals";
import { appStore, type SnapshotState } from "../../app/store";

const s = appStore.getState();

export const $snapshot = signal<SnapshotState>(s.snapshot);
