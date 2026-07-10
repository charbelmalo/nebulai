/** Types for nebulai.json (the Phase-1 → Phase-2 contract) and out/index.json.
 *  Source of truth: docs/DETAILS.md §5 / skills/nebulai/references/backend-contract.md. */

export interface NebulaiPoint {
  id: number;
  unit_ref: { kind: string; index: number };
  label: string;
  confidence: number; // HDBSCAN membership probability → opacity
  layer: number | null;
  xy: [number, number];
  xyz: [number, number, number];
  cluster_id: number; // -1 = noise
}

export interface NebulaiCluster {
  id: number;
  title: string;
  size: number;
  centroid: [number, number, number]; // u3 (display) space
}

/** Schema v2 similarity edges — computed in 10-D u_cluster space, never display. */
export interface NebulaiEdges {
  space: string; // "umap10"
  metric: string; // "gaussian_euclidean"
  k_cluster: number;
  sigma: number;
  cluster_edges: [number, number, number][]; // [a, b, weight], a < b
  knn?: {
    k: number;
    sigma: number;
    ids: number[]; // flat n_points * k
    sims: number[]; // flat n_points * k
  };
}

export interface NebulaiMeta {
  schema_version?: number; // absent = v1
  n_points: number;
  n_clusters: number;
  noise_fraction: number;
  namer: string;
  created: string;
  model?: string;
  unit?: string;
  [key: string]: unknown; // Units.meta provenance passes through verbatim
}

export interface NebulaiDoc {
  meta: NebulaiMeta;
  points: NebulaiPoint[];
  clusters: NebulaiCluster[];
  edges?: NebulaiEdges;
}

export interface DatasetEntry {
  id: string;
  model: string;
  path: string;
  schema_version: number;
  n_points: number;
  n_clusters: number;
  noise_fraction: number;
  namer: string | null;
  has_edges: boolean;
  /** true when out/<id>/interp/index.json exists. Optional: index.json files
   *  written before this field simply omit it and the viewer falls back to a
   *  network probe. */
  has_interp?: boolean;
}

export interface DatasetIndex {
  datasets: DatasetEntry[];
}

/** v1 files predate schema_version; anything unknown is treated as v1 so the
 *  viewer degrades instead of failing. Edge presence is separate — a v2 file
 *  exported with `--edges none` has no edges block and beams stay off. */
export function schemaVersion(doc: NebulaiDoc): 1 | 2 {
  return doc.meta.schema_version === 2 ? 2 : 1;
}
