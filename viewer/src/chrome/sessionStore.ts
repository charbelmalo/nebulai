/** sessionStore.ts — durable, cross-session persistence for analysed agent
 *  sessions, backed by the browser's IndexedDB.
 *
 *  PRIVACY: only the *derived* `SessionAnalysis` is persisted — token totals,
 *  the per-turn trajectory, the tool histogram, file paths, plus short
 *  (≤240-char) prompt/response previews that power the turn inspector. The
 *  full raw transcript text (complete prompts, model prose, tool output) is
 *  NEVER stored; it stays in volatile memory for the lifetime of the parse
 *  and is dropped. IndexedDB is local to this browser profile on this machine
 *  and is never transmitted.
 *
 *  This is what makes "analysis persistence from session to session" real: an
 *  analysis loaded today is still on the Sessions trajectory tomorrow, so you
 *  can compare the shape of many sessions accumulated over time. Clearing is a
 *  one-call `clearSessionAnalyses()` (surfaced in Settings → Sessions).
 */

import { buildComposition, type SessionAnalysis } from "./sessionlog";

const DB_NAME = "nebulai";
const DB_VERSION = 1;
const STORE = "sessions";

let dbPromise: Promise<IDBDatabase> | null = null;

function openDB(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("IndexedDB unavailable — session analyses cannot persist"));
      return;
    }
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const os = db.createObjectStore(STORE, { keyPath: "id" });
        os.createIndex("loadedAt", "loadedAt", { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("IndexedDB open failed"));
  });
  return dbPromise;
}

function tx<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return openDB().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const t = db.transaction(STORE, mode);
        const req = run(t.objectStore(STORE));
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error ?? new Error("IndexedDB request failed"));
      }),
  );
}

/** Persist (or overwrite) one analysed session. Resolves once committed. */
export function saveSessionAnalysis(a: SessionAnalysis): Promise<void> {
  return tx<IDBValidKey>("readwrite", (s) => s.put(a)).then(() => undefined);
}

/** Fill in fields the fold gained after a record was written.
 *
 *  Records are stored as-is, with no schema version, so a row saved by an
 *  older build is missing whatever the analysis has learned to compute since.
 *  Every gap is filled with EMPTY, never with a reconstruction: the raw
 *  transcript is deliberately not kept (see the privacy note above), so there
 *  is nothing to recompute from, and a plausible-looking guess derived from
 *  the fields that did survive would be a fabricated measurement.
 *
 *  Empty is also readable downstream — a card with no outcome bars is showing
 *  you that it has no outcome data, which is the truth. `SessionsStats` says so
 *  in words when the record is old enough to be missing them. */
export function normalizeSessionAnalysis(a: SessionAnalysis): SessionAnalysis {
  if (!Array.isArray(a.toolOutcomes)) a.toolOutcomes = [];
  if (typeof a.unattributedErrors !== "number") a.unattributedErrors = 0;
  // The context composition is the one gap that CAN be filled honestly, and so
  // it is the one exception to the empty-not-reconstructed rule above. It reads
  // nothing but per-turn token counts, and those are persisted — running the
  // same pure function over the same real numbers is not a guess. What does not
  // survive is the usage-consistency check, so `exact` goes in as null: unknown,
  // which is neither the true it might have been nor a false we cannot support.
  if (!a.context && Array.isArray(a.turns)) a.context = buildComposition(a.turns, null);
  return a;
}

/** All persisted analyses, most-recently-loaded first. Returns [] if the
 *  store is empty or IndexedDB is unavailable (degrades, never throws). */
export function loadAllSessionAnalyses(): Promise<SessionAnalysis[]> {
  return tx<SessionAnalysis[]>("readonly", (s) => s.getAll() as IDBRequest<SessionAnalysis[]>)
    .then((rows) => rows.map(normalizeSessionAnalysis).sort((a, b) => b.loadedAt - a.loadedAt))
    .catch(() => []);
}

/** Forget one session (by id). */
export function deleteSessionAnalysis(id: string): Promise<void> {
  return tx<undefined>("readwrite", (s) => s.delete(id) as IDBRequest<undefined>).then(
    () => undefined,
  );
}

/** Forget every persisted session. */
export function clearSessionAnalyses(): Promise<void> {
  return tx<undefined>("readwrite", (s) => s.clear() as IDBRequest<undefined>).then(() => undefined);
}

/** Best-effort byte estimate of what's persisted, for the Settings readout. */
export async function estimateSessionStorage(): Promise<{ count: number; bytes: number }> {
  const all = await loadAllSessionAnalyses();
  let bytes = 0;
  for (const a of all) bytes += JSON.stringify(a).length;
  return { count: all.length, bytes };
}
