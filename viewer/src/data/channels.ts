/** Per-point scalars for a loaded map, read from `out/<id>/channels.json`.
 *
 *  A channel is one number per point, index-aligned to `nebulai.json`'s points:
 *  the raw `W_E` row norm and distance-to-centroid that make the glitch knot
 *  visible (phase 0), and — from phase 1 — every direction projection and its
 *  null. One sidecar carries both, so a map that has one has the other's
 *  plumbing already.
 *
 *  This module follows `data/validation.ts` exactly, for the same three
 *  reasons:
 *
 *  · **Fetch once, per dataset, lazily.** Most maps have no channels and a
 *    static deploy may ship none at all.
 *  · **Never throw.** Any failure — no file, offline, malformed, wrong length —
 *    resolves to "this dataset has no channels". Absence is *not measured*,
 *    never *measured as clean*: a map with no `channels.json` shows no channel
 *    UI, rather than a channel UI reading zero everywhere.
 *  · **Refuse in the loader, not in the UI.** `compatible()` is D2's hard
 *    refusal on cross-space projection, and it lives here so no view can route
 *    around it by asking a different question.
 *
 *  `missing` is carried as `NaN` in the Float32Array (the JSON has `null`),
 *  never as 0. Consumers MUST test `Number.isFinite` before drawing — a
 *  missing value that renders at the origin is exactly the failure
 *  `seer/contract.py` exists to prevent, and `channelValue()` below returns
 *  `null` for it so the type system helps.
 */

import { sidecarKnownAbsent } from "./experience";
import { signal } from "@preact/signals";
import { DATA_BASE } from "./base";

/** How a channel's numbers came to be known — `seer/contract.py`'s vocabulary,
 *  restricted to what a per-point scalar can honestly carry. */
export type ChannelFidelity = "deterministic" | "estimated" | "heuristic" | "missing";

export interface ChannelStats {
  /** over the MEASURED entries only; null when nothing was measured */
  min: number | null;
  max: number | null;
  mean: number | null;
  /** how many points have no value. Never folded into the range. */
  n_missing: number;
}

export interface Channel {
  id: string;
  label: string;
  /** a tag from the closed set in `src/nebulai/spaces.py` (see SPACE_FAMILIES) */
  space: string;
  method: string;
  formula: string;
  fidelity: ChannelFidelity;
  units: string;
  stats: ChannelStats;
  /** n values; NaN where the quantity was not measured */
  values: Float32Array;
  /** the direction this projection came from, when it is one (phase 1) */
  direction?: string;
}

export interface ChannelSet {
  datasetId: string;
  model: string;
  revision: string;
  nPoints: number;
  channels: Channel[];
  byId: Map<string, Channel>;
}

/* ── the closed space set (mirrors src/nebulai/spaces.py) ─────────────────── */

/** The eight families. Pinned against the Python enum by
 *  `tests/unit/channels.test.ts`, so adding one on either side without the
 *  other fails a test rather than shipping a tag one half silently drops. */
export const SPACE_FAMILIES = [
  "W_E.raw",
  "W_E.centered",
  "W_U.raw",
  "resid",
  "mlp_out",
  "sae",
  "text-embed",
  "persona-pca",
] as const;

const BARE = new Set(["W_E.raw", "W_E.centered", "W_U.raw"]);
const LAYER = /^L(-?\d+)$/;
const REF = /^[A-Za-z0-9][A-Za-z0-9._/-]*$/;

/** True when `tag` is inside the closed set. Unknown tags are not spaces, and
 *  a quantity in an unknown space is comparable with nothing — including
 *  itself. */
export function isKnownSpace(tag: string): boolean {
  if (typeof tag !== "string" || !tag) return false;
  if (BARE.has(tag)) return true;
  const dot = tag.indexOf(".");
  if (dot < 0) return false;
  const head = tag.slice(0, dot);
  const rest = tag.slice(dot + 1);
  if (head === "resid" || head === "mlp_out") return LAYER.test(rest);
  if (head === "sae") {
    const d2 = rest.indexOf(".");
    if (d2 < 0) return false;
    return LAYER.test(rest.slice(0, d2)) && REF.test(rest.slice(d2 + 1));
  }
  if (head === "text-embed" || head === "persona-pca") return REF.test(rest);
  return false;
}

/** May a quantity in space `a` be plotted against one in space `b`?
 *
 *  Only when they are the same space. D2 approved a hard refusal over a
 *  warning: `resid.L13` and `resid.L14` are different bases, `W_E.raw` and
 *  `W_E.centered` differ by the translation the glitch experiment is about,
 *  and two SAEs at one layer are two dictionaries. A picture you cannot make
 *  is one you cannot mistake for a finding.
 */
export function compatible(a: string, b: string): boolean {
  if (!isKnownSpace(a) || !isKnownSpace(b)) return false;
  return a === b;
}

/** Why `a` and `b` may not be compared, or null when they may. The UI prints
 *  this rather than inventing its own wording, so the browser refuses in the
 *  same words `nebulai.spaces.refusal_reason` does. */
export function refusalReason(a: string, b: string): string | null {
  if (compatible(a, b)) return null;
  if (!isKnownSpace(a)) return `unknown space “${a}”`;
  if (!isKnownSpace(b)) return `unknown space “${b}”`;
  return `space mismatch: ${a} vs ${b} — different bases, never comparable`;
}

/* ── loading ──────────────────────────────────────────────────────────────── */

/** dataset id → its channels, or null for "fetched, and there are none".
 *  A dataset absent from the map has not been fetched yet. */
export const $channels = signal<Map<string, ChannelSet | null>>(new Map());

const started = new Set<string>();

function normalizeFidelity(v: unknown): ChannelFidelity {
  return v === "estimated" || v === "heuristic" || v === "missing" ? v : "deterministic";
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Turn one raw channel object into typed columns, or null when it is not
 *  renderable. Rejection reasons, all of which would otherwise mislead:
 *  a missing/odd id, a space outside the closed set, or a length that
 *  disagrees with the map (an index-shifted channel mislabels every point
 *  past the gap). */
export function parseChannel(raw: unknown, nPoints: number): Channel | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  const id = o.id;
  if (typeof id !== "string" || !id) return null;
  const space = o.space;
  if (typeof space !== "string" || !isKnownSpace(space)) return null;
  const vals = o.values;
  if (!Array.isArray(vals) || vals.length !== nPoints) return null;

  const values = new Float32Array(nPoints);
  let missing = 0;
  for (let i = 0; i < nPoints; i++) {
    const v = vals[i];
    // null (and anything non-finite) is NOT measured. NaN propagates that all
    // the way to the GPU, where the filter drops it; 0 would draw a point in
    // the low-norm knot that was never measured to be there.
    if (typeof v === "number" && Number.isFinite(v)) values[i] = v;
    else {
      values[i] = Number.NaN;
      missing++;
    }
  }

  const s = (o.stats ?? {}) as Record<string, unknown>;
  return {
    id,
    label: typeof o.label === "string" ? o.label : id,
    space,
    method: typeof o.method === "string" ? o.method : "",
    formula: typeof o.formula === "string" ? o.formula : "",
    fidelity: normalizeFidelity(o.fidelity),
    units: typeof o.units === "string" ? o.units : "",
    stats: {
      min: num(s.min),
      max: num(s.max),
      mean: num(s.mean),
      // trust the file's count only if it agrees with what we actually read;
      // the array in hand is the ground truth
      n_missing: missing,
    },
    values,
    ...(typeof o.direction === "string" ? { direction: o.direction } : {}),
  };
}

/** Columnarise a whole `channels.json` document. Returns null when the doc is
 *  absent, malformed, or describes a different map. Never throws. */
export function parseChannelSet(doc: unknown, datasetId: string): ChannelSet | null {
  if (!doc || typeof doc !== "object") return null;
  const d = doc as Record<string, unknown>;
  const meta = (d.meta ?? {}) as Record<string, unknown>;
  const nPoints = typeof meta.n_points === "number" ? meta.n_points : 0;
  if (!(nPoints > 0) || !Array.isArray(d.channels)) return null;

  const channels: Channel[] = [];
  for (const raw of d.channels) {
    const ch = parseChannel(raw, nPoints);
    if (ch) channels.push(ch);
  }
  if (channels.length === 0) return null;

  return {
    datasetId,
    model: typeof meta.model === "string" ? meta.model : datasetId,
    revision: typeof meta.revision === "string" ? meta.revision : "",
    nPoints,
    channels,
    byId: new Map(channels.map((c) => [c.id, c])),
  };
}

/** Fetch this dataset's channels once, lazily. Safe to call on every render.
 *
 *  `expectedPoints` is the loaded map's point count: a channel file whose meta
 *  disagrees with the map in front of the user is aligned to a DIFFERENT map,
 *  so it is dropped whole rather than rendered over the wrong points. */
export function ensureChannels(
  datasetId: string,
  expectedPoints?: number,
  base = DATA_BASE,
): void {
  if (!datasetId || started.has(datasetId)) return;
  started.add(datasetId);
  void (async () => {
    let set: ChannelSet | null = null;
    try {
      // the release manifest can say authoritatively that no channels ship
      // for this map; then there is nothing to ask for, and no 404 to log
      const absent = base === DATA_BASE && sidecarKnownAbsent(datasetId, "channels.json");
      const res = absent ? null : await fetch(`${base}/${datasetId}/channels.json`);
      if (res?.ok) {
        set = parseChannelSet(await res.json(), datasetId);
        if (set && expectedPoints !== undefined && set.nPoints !== expectedPoints) set = null;
      }
    } catch {
      set = null;
    }
    const next = new Map($channels.value);
    next.set(datasetId, set);
    $channels.value = next;
  })();
}

/** This dataset's channels, or null when it has none (or they have not
 *  arrived yet — the UI renders nothing either way, which is the same honest
 *  answer). */
export function channelsFor(datasetId: string | null): ChannelSet | null {
  if (!datasetId) return null;
  return $channels.value.get(datasetId) ?? null;
}

/** Has this dataset's sidecar been fetched yet — whatever the answer was?
 *
 *  The difference between "no channels" and "not asked yet" has to be visible
 *  to the episode gate: an episode whose data is still in flight is *pending*,
 *  and calling it unavailable would be a small lie that resolves itself a
 *  moment later. `channelsFor()` deliberately collapses both to null (the UI
 *  draws nothing either way); this is the one caller that needs them apart. */
export function channelsLoaded(datasetId: string | null): boolean {
  return !!datasetId && $channels.value.has(datasetId);
}

export function channelFor(datasetId: string | null, channelId: string | null): Channel | null {
  if (!channelId) return null;
  return channelsFor(datasetId)?.byId.get(channelId) ?? null;
}

/** One point's value, or null when it was not measured. Returning null rather
 *  than NaN is deliberate: callers must handle absence, and `null` is the
 *  shape TypeScript will make them handle. */
export function channelValue(ch: Channel | null, index: number): number | null {
  if (!ch || index < 0 || index >= ch.values.length) return null;
  const v = ch.values[index]!;
  return Number.isFinite(v) ? v : null;
}

/** The `n` points with the lowest measured values, ascending. Missing points
 *  are excluded, not sorted to the front — "not measured" is not "smallest".
 *  This is what the SearchPanel's ranked lowest-norm list reads. */
export function lowestBy(ch: Channel, n: number): { index: number; value: number }[] {
  const out: { index: number; value: number }[] = [];
  for (let i = 0; i < ch.values.length; i++) {
    const v = ch.values[i]!;
    if (Number.isFinite(v)) out.push({ index: i, value: v });
  }
  out.sort((a, b) => a.value - b.value);
  return out.slice(0, Math.max(0, n));
}

/** Reset the module for tests (and for a hard dataset-index refresh). */
export function __resetChannels(): void {
  started.clear();
  $channels.value = new Map();
}
