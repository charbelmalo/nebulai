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
