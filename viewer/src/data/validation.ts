/** The distortion signal for a loaded map, read from out/compare/metrics.json.
 *
 *  `nebulai validate` measures three things the map's own geometry cannot tell
 *  you — trustworthiness (did UMAP preserve the ORIGINAL-space neighbourhoods),
 *  seed stability (would the same map be drawn on a different day), and a null
 *  baseline (what the identical pipeline scores on column-shuffled vectors).
 *  Until now all three stopped at the CLI and the README, so a browser user got
 *  no distortion signal at all — including for the one shipped map whose null
 *  OUT-SCORES it.
 *
 *  THE VERDICT IS NOT COMPUTED HERE. `margin`, `below_null_floor` and
 *  `not_comparable` are derived once in Python (backend/metrics.py:margin_flags)
 *  and written into metrics.json by `nebulai metrics`. This module only renders
 *  what that file already decided. The rules are subtle — a null that resolved a
 *  very different cluster count is answering a different question, and a
 *  NEGATIVE margin is an inverted result rather than a weak one — and a second
 *  implementation in TypeScript would drift from the table it is supposed to
 *  agree with. Same reason SESSIONSEER-LIVE.md §4.2 refuses to let the field
 *  layer recompute the projection it draws over.
 *
 *  Absence is "not measured", never "measured as clean": a map missing from
 *  metrics.json renders NO readout rather than a passing one. Validation
 *  re-runs UMAP, so it is opt-in and most maps legitimately have no row.
 */

import { signal } from "@preact/signals";
import { DATA_BASE } from "./base";

export interface MapVerdict {
  id: string;
  /** neighbourhood preservation from the original space; 1.0 faithful, ~0.5 chance */
  trustworthiness: number | null;
  /** mean pairwise ARI across UMAP seeds; 1.0 = same map every seed */
  stability_ari: number | null;
  silhouette: number | null;
  null_silhouette: number | null;
  /** silhouette − null_silhouette. `null` when the map was never validated. */
  margin: number | null;
  /** the null clustered outside 0.5–2x the map's k, so the margin compares
   *  two different questions and is not evidence either way. */
  not_comparable: boolean;
  /** the null scored HIGHER than the real map — the separation on screen is
   *  the construction procedure, not the model. */
  below_null_floor: boolean;
}

/** null = not loaded yet; an empty Map = loaded and nothing to show. */
export const $verdicts = signal<Map<string, MapVerdict> | null>(null);

let started = false;

/** Fetch once per session, lazily. Never throws: any failure (no file, offline,
 *  malformed) resolves to an empty Map so the UI renders no readout instead of
 *  hanging or erroring. A static deploy that ships no metrics.json is a
 *  supported configuration, not a fault. */
export function ensureVerdicts(base = DATA_BASE): void {
  if (started) return;
  started = true;
  void (async () => {
    try {
      const res = await fetch(`${base}/compare/metrics.json`);
      if (!res.ok) throw new Error(String(res.status));
      const doc = (await res.json()) as { maps?: MapVerdict[] };
      const byId = new Map<string, MapVerdict>();
      for (const m of doc.maps ?? []) if (m?.id) byId.set(m.id, m);
      $verdicts.value = byId;
    } catch {
      $verdicts.value = new Map();
    }
  })();
}

/** The verdict for one dataset id, or null when it has not been validated (or
 *  metrics.json predates the verdict fields — in which case `margin` is absent
 *  and the caller must not infer one). */
export function verdictFor(id: string | null): MapVerdict | null {
  if (!id) return null;
  return $verdicts.value?.get(id) ?? null;
}
