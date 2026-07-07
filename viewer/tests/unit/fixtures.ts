import type { NebulaiDoc, NebulaiPoint } from "../../src/data/schema";

function pt(
  id: number,
  cluster: number,
  conf: number,
  xy: [number, number],
  xyz: [number, number, number],
): NebulaiPoint {
  return {
    id,
    unit_ref: { kind: "token_embedding", index: id + 100 },
    label: `tok${id}`,
    confidence: conf,
    layer: null,
    xy,
    xyz,
    cluster_id: cluster,
  };
}

/** Legacy export: no schema_version, no edges. */
export function v1Doc(): NebulaiDoc {
  return {
    meta: {
      n_points: 4,
      n_clusters: 1,
      noise_fraction: 0.25,
      namer: "centroid",
      created: "2026-01-01T00:00:00+00:00",
      model: "toy",
      unit: "token_embedding",
    },
    points: [
      pt(0, 0, 1.0, [0, 0], [0, 0, 0]),
      pt(1, 0, 0.5, [1, 0], [1, 0, 0.5]),
      pt(2, 0, 0.25, [1, 1], [1, 1, 1]),
      pt(3, -1, 0.0, [5, 5], [5, 5, 5]),
    ],
    clusters: [{ id: 0, title: "a cluster", size: 3, centroid: [0.7, 0.3, 0.5] }],
  };
}

/** Schema v2 with cluster edges + point kNN (k=2, 4 points → flat length 8). */
export function v2Doc(): NebulaiDoc {
  const doc = v1Doc();
  doc.meta.schema_version = 2;
  doc.points.push(pt(4, 1, 0.9, [-1, -1], [-1, -1, -1]));
  doc.points.push(pt(5, 1, 0.8, [-1.2, -1], [-1.2, -1, -1]));
  doc.meta.n_points = 6;
  doc.meta.n_clusters = 2;
  doc.clusters.push({ id: 1, title: "another", size: 2, centroid: [-1.1, -1, -1] });
  doc.edges = {
    space: "umap10",
    metric: "gaussian_euclidean",
    k_cluster: 5,
    sigma: 2.5,
    cluster_edges: [[0, 1, 0.42]],
    knn: {
      k: 2,
      sigma: 0.4,
      ids: [1, 2, 0, 2, 0, 1, 2, 1, 5, 3, 4, 3],
      sims: [0.9, 0.6, 0.9, 0.7, 0.6, 0.7, 0.1, 0.1, 0.95, 0.05, 0.95, 0.05],
    },
  };
  return doc;
}
