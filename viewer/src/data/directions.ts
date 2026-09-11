/** Directions for a loaded map, read from `out/<id>/directions.json`.
 *
 *  A direction is one unit vector in one named space, plus the provenance that
 *  makes it readable and the two statistics that make it checkable. The
 *  per-point numbers it produces do NOT live here — they are channels, in the
 *  same `channels.json` the glitch lens uses — so this module is small and its
 *  only real job is to decide which directions may be drawn.
 *
 *  That decision is the whole point, and it is made here rather than in a view
 *  so no view can route around it (the same argument as `compatible()` in
 *  `data/channels.ts`):
 *
 *  · **R5 — a direction with no null is not renderable.** Not "renderable
 *    without a ghost": not renderable. The ghost is not decoration, it is the
 *    only thing that says whether the axis is an axis.
 *  · **D2 — the projection channel must exist and must carry the direction's
 *    own space tag.** A projection of `resid.L14` points onto a `W_E.centered`
 *    direction is a number, not a measurement.
 *
 *  Like every loader in `data/`, this one never throws: a missing file, an
 *  offline fetch or a malformed document all resolve to "this dataset has no
 *  directions", which the UI renders as no axis affordance at all rather than
 *  as an axis reading zero.
 */

import { signal } from "@preact/signals";
import { DATA_BASE } from "./base";
import { type Channel, channelFor, channelsFor, channelsLoaded, isKnownSpace } from "./channels";

/** How a direction came to exist. Mirrors `METHODS` in
 *  `src/nebulai/backend/directions.py`; pinned by `tests/unit/directions.test.ts`
 *  so adding one on either side without the other fails a test. */
export const DIRECTION_METHODS = [
  "diff_of_means",
  "pca",
  "sae_decoder",
  "lora_rank1",
  "probe",
  "two_selection",
] as const;
export type DirectionMethod = (typeof DIRECTION_METHODS)[number];

/** The map-wide comparison: is this axis more than a random axis, over every
 *  point? For a contrast between two clusters the honest answer is usually
 *  no. */
export interface DirectionStats {
  cohens_d: number | null;
  overlap: number | null;
  n: number | null;
}

/** The comparison a made direction actually claims: its own two sets.
 *
 *  `cohens_d` is in sample and therefore cannot be small — a diff of means is
 *  fitted on exactly those points. `heldout_cohens_d` refits on half of each
 *  set and scores the other half, and is the only number here a reader may
 *  treat as a measurement. It is `null` — rendered as "not measured", never as
 *  0 — when a set was too small to split.
 */
export interface DirectionContrast {
  cohens_d: number | null;
  overlap: number | null;
  n_pos: number | null;
  n_neg: number | null;
  null_n: number | null;
  null_cohens_d_mean: number | null;
  null_cohens_d_p95: number | null;
  heldout_cohens_d: number | null;
  heldout_overlap: number | null;
  heldout_n_pos: number | null;
  heldout_n_neg: number | null;
}

export interface DirectionSource {
  /** "computed" here, or "imported" from a published artefact */
  kind: string;
  /** the frozen identity of the two sets / the repo and commit sha. Never
   *  empty: a direction whose protocol is unknown is not a direction. */
  protocol: string;
  contrast: DirectionContrast | null;
}

export interface DirectionNull {
  method: string;
  seed: number;
  n: number;
  channel: string;
  orth_channel: string;
}

export interface Direction {
  id: string;
  label: string;
  /** a tag from the closed set in `src/nebulai/spaces.py` */
  space: string;
  method: DirectionMethod;
  d: number;
  source: DirectionSource;
  /** the two real channels, and how the axis compares with its null map-wide */
  projection: { channel: string; orth_channel: string; stats: DirectionStats } | null;
  /** the ghost. Absent → not renderable (R5). */
  null: DirectionNull | null;
}

export interface DirectionSet {
  datasetId: string;
  model: string;
  revision: string;
  directions: Direction[];
  byId: Map<string, Direction>;
}

/** The four columns an axis needs to draw: real parallel/orthogonal and the
 *  null's. All four or nothing — a half-resolved axis is exactly the figure
 *  without its ghost that R5 forbids. */
export interface AxisChannels {
  direction: Direction;
  par: Channel;
  orth: Channel;
  nullPar: Channel;
  nullOrth: Channel;
}

/* ── parsing ──────────────────────────────────────────────────────────────── */

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function int(v: unknown, fallback: number): number {
  return typeof v === "number" && Number.isFinite(v) ? Math.trunc(v) : fallback;
}

function parseContrast(raw: unknown): DirectionContrast | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  return {
    cohens_d: num(o.cohens_d),
    overlap: num(o.overlap),
    n_pos: num(o.n_pos),
    n_neg: num(o.n_neg),
    null_n: num(o.null_n),
    null_cohens_d_mean: num(o.null_cohens_d_mean),
    null_cohens_d_p95: num(o.null_cohens_d_p95),
    // the backend writes the string "missing" when a set was too small to
    // split; it arrives here as null and must render as "not measured"
    heldout_cohens_d: num(o.heldout_cohens_d),
    heldout_overlap: num(o.heldout_overlap),
    heldout_n_pos: num(o.heldout_n_pos),
    heldout_n_neg: num(o.heldout_n_neg),
  };
}

/** One raw direction object, or null when it is not one.
 *
 *  Rejected: no id, a space outside the closed set, a method outside the
 *  closed set, a non-positive width, or an empty `source.protocol`. Every one
 *  of those would render as a labelled axis whose label is not backed by
 *  anything. */
export function parseDirection(raw: unknown): Direction | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  const id = o.id;
  if (typeof id !== "string" || !id) return null;
  const space = o.space;
  if (typeof space !== "string" || !isKnownSpace(space)) return null;
  const method = o.method;
  if (typeof method !== "string" || !(DIRECTION_METHODS as readonly string[]).includes(method)) {
    return null;
  }
  const d = num(o.d);
  if (d === null || d <= 0) return null;

  const src = (o.source ?? {}) as Record<string, unknown>;
  const protocol = typeof src.protocol === "string" ? src.protocol : "";
  if (!protocol) return null;

  const pj = (o.projection ?? null) as Record<string, unknown> | null;
  let projection: Direction["projection"] = null;
  if (pj && typeof pj.channel === "string" && pj.channel) {
    const st = (pj.stats ?? {}) as Record<string, unknown>;
    projection = {
      channel: pj.channel,
      orth_channel: typeof pj.orth_channel === "string" ? pj.orth_channel : "",
      stats: { cohens_d: num(st.cohens_d), overlap: num(st.overlap), n: num(st.n) },
    };
  }

  const nl = (o.null ?? null) as Record<string, unknown> | null;
  let nul: DirectionNull | null = null;
  if (nl && typeof nl.channel === "string" && nl.channel) {
    nul = {
      method: typeof nl.method === "string" ? nl.method : "random_unit",
      seed: int(nl.seed, 0),
      n: int(nl.n, 0),
      channel: nl.channel,
      orth_channel: typeof nl.orth_channel === "string" ? nl.orth_channel : "",
    };
  }

  return {
    id,
    label: typeof o.label === "string" && o.label ? o.label : id,
    space,
    method: method as DirectionMethod,
    d: Math.trunc(d),
    source: {
      kind: typeof src.kind === "string" ? src.kind : "computed",
      protocol,
      contrast: parseContrast(src.contrast),
    },
    projection,
    null: nul,
  };
}

/** Columnarise a whole `directions.json`. Null when absent or malformed. */
export function parseDirectionSet(doc: unknown, datasetId: string): DirectionSet | null {
  if (!doc || typeof doc !== "object") return null;
  const d = doc as Record<string, unknown>;
  if (!Array.isArray(d.directions)) return null;
  const meta = (d.meta ?? {}) as Record<string, unknown>;

  const directions: Direction[] = [];
  for (const raw of d.directions) {
    const dir = parseDirection(raw);
    if (dir) directions.push(dir);
  }
  if (directions.length === 0) return null;

  return {
    datasetId,
    model: typeof meta.model === "string" ? meta.model : datasetId,
    revision: typeof meta.revision === "string" ? meta.revision : "",
    directions,
    byId: new Map(directions.map((x) => [x.id, x])),
  };
}

/* ── loading ──────────────────────────────────────────────────────────────── */

/** dataset id → its directions, or null for "fetched, and there are none". */
export const $directions = signal<Map<string, DirectionSet | null>>(new Map());

const started = new Set<string>();

/** Fetch this dataset's directions once, lazily. Safe on every render. */
export function ensureDirections(datasetId: string, base = DATA_BASE): void {
  if (!datasetId || started.has(datasetId)) return;
  started.add(datasetId);
  void (async () => {
    let set: DirectionSet | null = null;
    try {
      const res = await fetch(`${base}/${datasetId}/directions.json`);
      if (res.ok) set = parseDirectionSet(await res.json(), datasetId);
    } catch {
      set = null;
    }
    const next = new Map($directions.value);
    next.set(datasetId, set);
    $directions.value = next;
  })();
}

export function directionsFor(datasetId: string | null): DirectionSet | null {
  if (!datasetId) return null;
  return $directions.value.get(datasetId) ?? null;
}

/** Has this dataset's sidecar been fetched yet — whatever the answer was?
 *  "no directions" and "not asked yet" are different, and the episode gate
 *  needs them apart. */
export function directionsLoaded(datasetId: string | null): boolean {
  return !!datasetId && $directions.value.has(datasetId);
}

export function directionById(datasetId: string | null, id: string | null): Direction | null {
  if (!id) return null;
  return directionsFor(datasetId)?.byId.get(id) ?? null;
}

/* ── the renderability gate ───────────────────────────────────────────────── */

/** Why this direction may not be drawn, or null when it may.
 *
 *  Mirrors `renderable()` in `src/nebulai/backend/directions.py` — same two
 *  rules, same order, and the test file pins the wording so the browser and
 *  the CLI cannot drift into refusing for different reasons. */
export function axisRefusal(datasetId: string | null, dir: Direction): string | null {
  if (!dir.null || !dir.null.channel) {
    return `“${dir.id}” has no null — a direction without its ghost is not a figure (R5)`;
  }
  if (!dir.projection || !dir.projection.channel) {
    return `“${dir.id}” has not been projected onto this map yet`;
  }
  if (!channelsLoaded(datasetId)) return "channels have not been fetched yet";

  const need: [string, string][] = [
    [dir.projection.channel, "projection"],
    [dir.projection.orth_channel, "orthogonal"],
    [dir.null.channel, "null projection"],
    [dir.null.orth_channel, "null orthogonal"],
  ];
  for (const [id, role] of need) {
    const ch = id ? channelFor(datasetId, id) : null;
    if (!ch) return `the ${role} channel “${id}” is not in this map's channels.json`;
    if (ch.space !== dir.space) {
      return `space mismatch: direction ${dir.space} vs channel ${ch.space} — different bases, never comparable`;
    }
  }
  return null;
}

/** The four columns, or null with the reason on `axisRefusal`. */
export function axisChannels(datasetId: string | null, id: string | null): AxisChannels | null {
  const dir = directionById(datasetId, id);
  if (!dir || axisRefusal(datasetId, dir) !== null) return null;
  const par = channelFor(datasetId, dir.projection!.channel);
  const orth = channelFor(datasetId, dir.projection!.orth_channel);
  const nullPar = channelFor(datasetId, dir.null!.channel);
  const nullOrth = channelFor(datasetId, dir.null!.orth_channel);
  if (!par || !orth || !nullPar || !nullOrth) return null;
  return { direction: dir, par, orth, nullPar, nullOrth };
}

/** Every direction of this dataset that may be drawn, and every one that may
 *  not with its reason. Both halves, always — a direction dropped silently is
 *  a direction the user thinks was never made. */
export function renderableDirections(datasetId: string | null): {
  ok: Direction[];
  drops: { direction: Direction; reason: string }[];
} {
  const set = directionsFor(datasetId);
  const ok: Direction[] = [];
  const drops: { direction: Direction; reason: string }[] = [];
  if (!set) return { ok, drops };
  for (const dir of set.directions) {
    const why = axisRefusal(datasetId, dir);
    if (why === null) ok.push(dir);
    else drops.push({ direction: dir, reason: why });
  }
  return { ok, drops };
}

/** Does this dataset have any direction at all whose space matches its
 *  channels? Used to decide whether the axis affordance appears. */
export function hasAxis(datasetId: string | null): boolean {
  return renderableDirections(datasetId).ok.length > 0;
}

/* ── histograms (the rail draws these; the numbers come from the sidecar) ─── */

export interface Histogram {
  /** bin edges, `bins.length + 1` of them */
  edges: Float64Array;
  /** fraction of measured values in each bin — sums to 1 (or 0 when n = 0) */
  bins: Float64Array;
  /** how many values were measured */
  n: number;
  /** how many were not. Never folded into `n`. */
  nMissing: number;
}

/** Bin a channel's measured values on an explicit range.
 *
 *  The real and null histograms MUST share a range or the ghost is drawn on a
 *  different ruler — which is why the range is a parameter rather than
 *  computed per call. `histogramRange()` below produces the shared one. */
export function histogram(ch: Channel, lo: number, hi: number, nBins = 48): Histogram {
  const edges = new Float64Array(nBins + 1);
  const span = hi - lo;
  for (let i = 0; i <= nBins; i++) edges[i] = lo + (span * i) / nBins;
  const bins = new Float64Array(nBins);
  let n = 0;
  let nMissing = 0;
  for (let i = 0; i < ch.values.length; i++) {
    const v = ch.values[i]!;
    if (!Number.isFinite(v)) {
      nMissing++;
      continue;
    }
    n++;
    if (span <= 0) continue;
    let b = Math.floor(((v - lo) / span) * nBins);
    if (b < 0) b = 0;
    if (b >= nBins) b = nBins - 1;
    bins[b]! += 1;
  }
  if (n > 0) for (let i = 0; i < nBins; i++) bins[i]! /= n;
  return { edges, bins, n, nMissing };
}

/** The range both histograms share: the union of the two channels' measured
 *  extents, from the stats the backend already computed where it can. */
export function histogramRange(a: Channel, b: Channel): [number, number] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const ch of [a, b]) {
    const s = ch.stats;
    if (s.min !== null) lo = Math.min(lo, s.min);
    if (s.max !== null) hi = Math.max(hi, s.max);
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    // stats absent: fall back to the columns themselves
    for (const ch of [a, b]) {
      for (let i = 0; i < ch.values.length; i++) {
        const v = ch.values[i]!;
        if (!Number.isFinite(v)) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
  if (hi <= lo) return [lo - 0.5, lo + 0.5];
  return [lo, hi];
}

/** The one-line claim an axis is allowed to make about itself.
 *
 *  Deliberately not a sentence about what the direction *means*: the held-out
 *  effect size and the random-direction baseline, in that order, or a plain
 *  statement that neither was measured. D3 permits exactly one causal sentence
 *  per figure and this is not it — nothing here says the model uses this axis.
 */
export function axisClaim(dir: Direction): string {
  const c = dir.source.contrast;
  if (!c) return "no contrast statistics were recorded for this direction";
  if (c.heldout_cohens_d === null) {
    return "held-out separation: not measured — a set of fewer than four members cannot be split";
  }
  const base =
    c.null_cohens_d_mean === null
      ? "no random-direction baseline was recorded"
      : `random directions on the same two sets average |d| ${c.null_cohens_d_mean.toFixed(2)}`;
  return (
    `separates its own two sets by d = ${c.heldout_cohens_d.toFixed(2)} on held-out members ` +
    `(n = ${c.heldout_n_pos ?? 0}/${c.heldout_n_neg ?? 0}); ${base}`
  );
}

/** What the map-wide comparison says, in words. Usually "this is not a special
 *  axis of the whole map", which is the finding, not a failure. */
export function axisMapClaim(dir: Direction): string {
  const s = dir.projection?.stats;
  if (!s || s.overlap === null || s.n === null) return "not projected onto this map";
  const verdict =
    s.overlap > 0.5
      ? "indistinguishable from a random direction across the map as a whole"
      : "separated from its null across the map as a whole";
  return `over all ${s.n.toLocaleString()} points: histogram overlap ${s.overlap.toFixed(3)} — ${verdict}`;
}

/** Reset for tests. */
export function __resetDirections(): void {
  started.clear();
  $directions.value = new Map();
}

/** Exposed only so tests can assert the loader and the channel gate agree
 *  without reaching into `data/channels.ts`. */
export function __channelsPresent(datasetId: string | null): boolean {
  return channelsFor(datasetId) !== null;
}
