/** Beam-data lookups over the columnar edges block (schema v2). Weights are
 *  gaussian similarities computed in 10-D u_cluster space — the honesty
 *  guardrail: beams describe cluster-space proximity, never display-space. */

import type { EdgeColumns } from "./columns";

export interface ClusterNeighbor {
  other: number;
  weight: number;
}

export interface PointNeighbor {
  id: number;
  sim: number;
}

/** All cluster edges touching `cid`, strongest first. */
export function clusterNeighbors(edges: EdgeColumns, cid: number): ClusterNeighbor[] {
  const out: ClusterNeighbor[] = [];
  const e = edges.clusterEdges;
  for (let i = 0; i < e.length; i += 3) {
    const a = e[i]!;
    const b = e[i + 1]!;
    if (a === cid) out.push({ other: b, weight: e[i + 2]! });
    else if (b === cid) out.push({ other: a, weight: e[i + 2]! });
  }
  out.sort((x, y) => y.weight - x.weight);
  return out;
}

/** Point `i`'s kNN row, strongest first; empty when knn wasn't exported. */
export function knnNeighbors(edges: EdgeColumns, i: number): PointNeighbor[] {
  const knn = edges.knn;
  if (!knn) return [];
  const out: PointNeighbor[] = [];
  for (let j = i * knn.k; j < (i + 1) * knn.k; j++) {
    const id = knn.ids[j]!;
    if (id < 0 || id === i) continue; // padding / self
    out.push({ id, sim: knn.sims[j]! });
  }
  out.sort((x, y) => y.sim - x.sim);
  return out;
}

/** Recover the Euclidean distance in 10-D cluster space that produced a stored
 *  kNN score, or null when it cannot be recovered.
 *
 *  The exporter stores `sim = exp(-(d/sigma)^2)`, so this is just that kernel
 *  read backwards: `d = sigma * sqrt(-ln sim)`. Worth surfacing because the
 *  score itself is a kernel value with no unit — 0.6 means nothing on its own —
 *  while the distance is the quantity the neighbour search actually ranked on.
 *  `sigma` is one global constant for the whole export (the median neighbour
 *  distance), so distances are comparable across every point, not just within
 *  one row.
 *
 *  Returns null rather than a number when the score cannot support one:
 *  scores are rounded to 3 decimals on export, so a genuinely distant
 *  neighbour arrives as exactly 0 and its distance is not merely large, it is
 *  UNRECOVERABLE. Callers must render that as "—". Reporting Infinity, or a
 *  huge number from a clamped epsilon, would be inventing precision the export
 *  threw away. */
export function knnDistance(sim: number, sigma: number): number | null {
  if (!(sim > 0) || !(sigma > 0) || sim > 1) return null;
  // max(0, …) normalises the -0 that `-Math.log(1)` produces for an exact
  // score of 1, so a zero distance formats as "0.00" and not "-0.00".
  return sigma * Math.sqrt(Math.max(0, -Math.log(sim)));
}

/** The distance past which a neighbour's score rounds away entirely.
 *
 *  Scores are written with `np.round(sim, 3)`, so anything below 0.0005 lands
 *  on exactly 0.000 and `knnDistance` can no longer recover it. That is not a
 *  hole in the data, though — a score of 0 still tells us the neighbour is
 *  FARTHER than this, which is worth saying instead of showing a blank. */
export function knnDistanceFloor(sigma: number): number {
  return sigma * Math.sqrt(-Math.log(0.0005));
}

/** Hub score per cluster = sum of edge weights touching it (used to pick
 *  which clusters get pulsing halos). Returns clusterId → degree. */
export function clusterDegrees(edges: EdgeColumns): Map<number, number> {
  const deg = new Map<number, number>();
  const e = edges.clusterEdges;
  for (let i = 0; i < e.length; i += 3) {
    const a = e[i]!;
    const b = e[i + 1]!;
    const w = e[i + 2]!;
    deg.set(a, (deg.get(a) ?? 0) + w);
    deg.set(b, (deg.get(b) ?? 0) + w);
  }
  return deg;
}

/** Video-style compact counts for beam badges: 4900 → "4.9K", 25000 → "25K". */
export function formatCount(n: number): string {
  if (n < 1000) return String(n);
  if (n < 1e6) {
    const k = n / 1000;
    return `${k < 10 ? (Math.round(k * 10) / 10).toString() : Math.round(k)}K`;
  }
  const m = n / 1e6;
  return `${m < 10 ? (Math.round(m * 10) / 10).toString() : Math.round(m)}M`;
}
