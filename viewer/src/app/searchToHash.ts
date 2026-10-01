/** Query-string deep links (`?lesson=what-is-a-point&lesson_step=2`) are the
 *  form people type and paste from docs, but every view key lives in the hash
 *  (chrome/urlState.ts). Move the recognised keys from the query into the hash
 *  once, before anything reads it, so both spellings open the same view. A key
 *  already in the hash wins; unknown query keys (`frozen`, capability flags,
 *  the legacy `view`) stay where they are. Dependency-free: the chooser entry
 *  runs it too. */
export const HASH_KEYS = [
  "experience",
  "page",
  "lesson",
  "lesson_step",
  "episode",
  "step",
  "model",
  "feature",
  "q",
  "dims",
  "return",
  "artifact",
  "unit_index",
  "point",
] as const;

export function promoteSearchToHash(loc: Pick<Location, "search" | "hash" | "pathname"> = location): string | null {
  const search = new URLSearchParams(loc.search);
  const hash = new URLSearchParams(loc.hash.replace(/^#/, ""));
  let moved = false;
  for (const k of HASH_KEYS) {
    const v = search.get(k);
    if (v === null) continue;
    search.delete(k);
    if (!hash.has(k)) hash.set(k, v);
    moved = true;
  }
  if (!moved) return null;
  const s = search.toString();
  const h = hash.toString();
  const next = `${loc.pathname}${s ? `?${s}` : ""}${h ? `#${h}` : ""}`;
  if (typeof history !== "undefined" && loc === location) history.replaceState(null, "", next);
  return next;
}
