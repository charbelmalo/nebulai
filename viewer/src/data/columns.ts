/** Columnarize a parsed nebulai.json into typed arrays for zero-copy transfer
 *  out of the parse worker and direct GPU upload. Pure function — unit-tested
 *  in node without a DOM. */

import type { NebulaiCluster, NebulaiDoc, NebulaiMeta } from "./schema";
import { schemaVersion } from "./schema";

export interface EdgeColumns {
  space: string;
  metric: string;
  sigma: number;
  /** flat triples [a, b, weight] × nClusterEdges */
  clusterEdges: Float32Array;
  /** per-point kNN, flat n*k; null when exported with cluster-only edges */
  knn: { k: number; sigma: number; ids: Int32Array; sims: Float32Array } | null;
}

export interface Columns {
  meta: NebulaiMeta;
  schema: 1 | 2;
  count: number;
  pos2: Float32Array; // n*2
  pos3: Float32Array; // n*3
  clusterId: Int32Array; // n, -1 = noise
  confidence: Uint8Array; // n, quantized 0–255 (opacity only needs 8 bits)
  labels: string[]; // stays on the main thread, never uploaded
  clusters: NebulaiCluster[];
  edges: EdgeColumns | null;
}

export function columnarize(doc: NebulaiDoc): Columns {
  const n = doc.points.length;
  const pos2 = new Float32Array(n * 2);
  const pos3 = new Float32Array(n * 3);
  const clusterId = new Int32Array(n);
  const confidence = new Uint8Array(n);
  const labels = new Array<string>(n);

  for (let i = 0; i < n; i++) {
    const p = doc.points[i]!;
    pos2[i * 2] = p.xy[0];
    pos2[i * 2 + 1] = p.xy[1];
    pos3[i * 3] = p.xyz[0];
    pos3[i * 3 + 1] = p.xyz[1];
    pos3[i * 3 + 2] = p.xyz[2];
    clusterId[i] = p.cluster_id;
    confidence[i] = Math.round(Math.min(Math.max(p.confidence, 0), 1) * 255);
    labels[i] = p.label;
  }

  let edges: EdgeColumns | null = null;
  if (doc.edges) {
    const ce = doc.edges.cluster_edges;
    const clusterEdges = new Float32Array(ce.length * 3);
    for (let i = 0; i < ce.length; i++) {
      clusterEdges[i * 3] = ce[i]![0];
      clusterEdges[i * 3 + 1] = ce[i]![1];
      clusterEdges[i * 3 + 2] = ce[i]![2];
    }
    edges = {
      space: doc.edges.space,
      metric: doc.edges.metric,
      sigma: doc.edges.sigma,
      clusterEdges,
      knn: doc.edges.knn
        ? {
            k: doc.edges.knn.k,
            sigma: doc.edges.knn.sigma,
            ids: Int32Array.from(doc.edges.knn.ids),
            sims: Float32Array.from(doc.edges.knn.sims),
          }
        : null,
    };
  }

  return {
    meta: doc.meta,
    schema: schemaVersion(doc),
    count: n,
    pos2,
    pos3,
    clusterId,
    confidence,
    labels,
    clusters: doc.clusters,
    edges,
  };
}

/** The ArrayBuffers to hand to postMessage as transferables (zero-copy). */
export function transferables(c: Columns): ArrayBuffer[] {
  const bufs = [
    c.pos2.buffer,
    c.pos3.buffer,
    c.clusterId.buffer,
    c.confidence.buffer,
  ];
  if (c.edges) {
    bufs.push(c.edges.clusterEdges.buffer);
    if (c.edges.knn) bufs.push(c.edges.knn.ids.buffer, c.edges.knn.sims.buffer);
  }
  return bufs as ArrayBuffer[];
}
