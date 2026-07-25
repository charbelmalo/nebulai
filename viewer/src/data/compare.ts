/** Cross-model comparison data (`out/compare/compare.json`). Each point is a
 *  cluster centroid from one model, carrying its position in every layout
 *  state — the CompareDriver interpolates between states on the GPU. At ~840
 *  points the file is small (~400KB): plain fetch, no worker, and the CPU
 *  hover loop the atlas outgrew is perfectly fine here. */

import { DATA_BASE } from "./base";

export interface ComparePoint {
  source: string;
  source_idx: number;
  title: string;
  size: number;
  meta_cluster: number;
  shared: boolean;
  color: [number, number, number];
  positions: Record<string, [number, number, number]>;
}

export interface CompareData {
  meta: {
    models: string[];
    n_points: number;
    n_meta_clusters: number;
    embed_model: string;
  };
  states: string[];
  colors: Record<string, [number, number, number]>;
  stats: {
    n_shared_concepts: number;
    n_unique_per_model: Record<string, number>;
    jaccard: Record<string, number>;
  };
  points: ComparePoint[];
}

/** Per-instance attribute arrays for the compare field.
 *
 *  Packed to fit WebGPU's 8-vertex-buffer ceiling: the sprite quad already
 *  costs 2 (position + uv), so the four layout positions, the model colour and
 *  the four per-point scalars ride in FIVE buffers — each vec4 carries a state
 *  position in .xyz and one scalar in .w. That leaves a spare slot; going one
 *  attribute wider is rejected silently (see the ChordDriver note in the
 *  nebulai-viz-threejs skill).
 *
 *      a     = (native.xyz,      radius)
 *      b     = (semantic.xyz,    sourceIdx)
 *      c     = (byModel.xyz,     shared)
 *      d     = (byConcept.xyz,   sizeRank)
 *      color = model rgb
 */
export interface CompareAttributes {
  a: Float32Array;
  b: Float32Array;
  c: Float32Array;
  d: Float32Array;
  color: Float32Array;
  count: number;
  /** World radius every state is normalised to — what the camera fits to. */
  extent: number;
}

/** The common radius each layout state is rescaled to (see normaliseState). */
export const STATE_EXTENT = 10;

/** Centre a layout state on its own origin and scale it to STATE_EXTENT.
 *
 *  WHY THIS IS NOT CHEATING. The four states are four unrelated coordinate
 *  systems — a per-model UMAP, a joint embedding, and two synthetic grid
 *  arrangements. No distance in one is comparable to a distance in another, so
 *  their relative sizes are an artefact of how each was produced, not a fact
 *  about the models. A single uniform scale + translation per state preserves
 *  every within-state relationship (ratios, angles, ordering, clustering) and
 *  removes only that artefact.
 *
 *  It also fixes two real defects: the camera had to fit the union of all four,
 *  which framed the semantic layout at ~28% of the stage; and a fixed
 *  world-space sprite size meant apparent point density changed by an order of
 *  magnitude between states, so "denser" read as a property of the layout when
 *  it was a property of the export.
 *
 *  Scale keys on the 98th-percentile radius, not the max: a handful of
 *  stragglers would otherwise shrink the bulk to nothing. They stay on screen —
 *  the camera fit leaves margin — they just sit outside the nominal sphere. */
function normaliseState(buf: Float32Array, n: number): void {
  if (n === 0) return;
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      const v = buf[i * 4 + k]!;
      if (v < lo[k]!) lo[k] = v;
      if (v > hi[k]!) hi[k] = v;
    }
  }
  const cx = (lo[0]! + hi[0]!) / 2;
  const cy = (lo[1]! + hi[1]!) / 2;
  const cz = (lo[2]! + hi[2]!) / 2;
  const dists = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    dists[i] = Math.hypot(buf[i * 4]! - cx, buf[i * 4 + 1]! - cy, buf[i * 4 + 2]! - cz);
  }
  const sorted = Array.from(dists).sort((x, y) => x - y);
  const p98 = sorted[Math.min(n - 1, Math.floor(0.98 * (n - 1)))]! || sorted[n - 1]! || 1;
  const k = STATE_EXTENT / p98;
  for (let i = 0; i < n; i++) {
    buf[i * 4] = (buf[i * 4]! - cx) * k;
    buf[i * 4 + 1] = (buf[i * 4 + 1]! - cy) * k;
    buf[i * 4 + 2] = (buf[i * 4 + 2]! - cz) * k;
  }
}

/** Midrank of each point's token count, normalised to 0..1.
 *
 *  Same reasoning as the sessions field: cluster sizes are heavy-tailed, so a
 *  value/max ratio leaves almost every point in the bottom of the ramp and the
 *  field reads as flat confetti. A rank always spans the full ramp. It stays
 *  honest because it is monotone in the raw token count and ties share a
 *  midrank — the tooltip prints the real number. */
export function sizeRank(points: ComparePoint[]): Float32Array {
  const n = points.length;
  const out = new Float32Array(n);
  if (n === 0) return out;
  if (n === 1) {
    out[0] = 1;
    return out;
  }
  const order = Array.from({ length: n }, (_, i) => i).sort(
    (x, y) => points[x]!.size - points[y]!.size,
  );
  const denom = n - 1;
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && points[order[j + 1]!]!.size === points[order[i]!]!.size) j++;
    const midrank = (i + j) / 2 / denom; // ties share the middle of their run
    for (let k = i; k <= j; k++) out[order[k]!] = midrank;
    i = j + 1;
  }
  return out;
}

/** Radius is sqrt-scaled against the largest cluster, as the original viewer
 *  did — area, not radius, tracks token count. */
export function compareAttributes(data: CompareData): CompareAttributes {
  const pts = data.points;
  const states = data.states;
  const n = pts.length;
  let maxSize = 1;
  for (const p of pts) maxSize = Math.max(maxSize, p.size);
  const rank = sizeRank(pts);

  const a = new Float32Array(n * 4);
  const b = new Float32Array(n * 4);
  const c = new Float32Array(n * 4);
  const d = new Float32Array(n * 4);
  const color = new Float32Array(n * 3);
  const packs = [a, b, c, d];

  for (let i = 0; i < n; i++) {
    const p = pts[i]!;
    for (let s = 0; s < 4; s++) {
      const pos = p.positions[states[s]!] ?? [0, 0, 0];
      const buf = packs[s]!;
      buf[i * 4] = pos[0];
      buf[i * 4 + 1] = pos[1];
      buf[i * 4 + 2] = pos[2];
    }
    a[i * 4 + 3] = 0.28 + 0.95 * Math.sqrt(p.size / maxSize);
    b[i * 4 + 3] = p.source_idx;
    c[i * 4 + 3] = p.shared ? 1 : 0;
    d[i * 4 + 3] = rank[i]!;
    color[i * 3] = p.color[0];
    color[i * 3 + 1] = p.color[1];
    color[i * 3 + 2] = p.color[2];
  }
  // after packing, so the .w scalars are already in place and untouched
  for (const buf of packs) normaliseState(buf, n);
  return { a, b, c, d, color, count: n, extent: STATE_EXTENT };
}

let cached: CompareData | null | undefined;

/** null = no comparison exported yet (run `nebulai compare` first). */
export async function loadCompare(base = DATA_BASE): Promise<CompareData | null> {
  if (cached !== undefined) return cached;
  try {
    const res = await fetch(`${base}/compare/compare.json`);
    cached = res.ok ? ((await res.json()) as CompareData) : null;
  } catch {
    cached = null;
  }
  return cached;
}
