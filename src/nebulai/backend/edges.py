"""Similarity edges for the Phase-2 viewer's connection beams.

Everything here is computed in the 10-D ``u_cluster`` space — the same space
HDBSCAN ran in — never in the 2-D/3-D display projections. Beam presence and
weight must reflect the geometry the clustering saw, not the layout the eye
sees; the export stamps that provenance (``space``/``metric``/``sigma``) so
the viewer can say so.

Metric note: HDBSCAN runs Euclidean on ``u_cluster``, so edges do too.
Cosine is correct on the *original* embedding rows but saturates on UMAP
output coordinates (they sit far from the origin — measured spread on gpt2
was [0.912, 1.0]). Weights are a Gaussian kernel ``exp(-(d/sigma)^2)`` with
``sigma`` = the median distance of the candidate set, so weights are
comparable within one export and the scale is recorded, not implied.
"""

import numpy as np


def _cluster_centroids(
    u_cluster: np.ndarray, cluster_ids: np.ndarray
) -> tuple[list[int], np.ndarray]:
    cids = sorted({int(c) for c in cluster_ids if c >= 0})
    centroids = np.stack(
        [u_cluster[cluster_ids == cid].mean(axis=0) for cid in cids]
    )
    return cids, centroids


def compute_cluster_edges(
    u_cluster: np.ndarray,
    cluster_ids: np.ndarray,
    k_cluster: int = 5,
) -> tuple[list[list[float]], float]:
    """Top-k nearest cluster pairs as ``[a, b, weight]`` triples, plus sigma.

    An edge survives if it is in *either* endpoint's top-k (union), deduped
    with ``a < b``. Weight = ``exp(-(d/sigma)^2)`` over centroid Euclidean
    distance, ``sigma`` = median off-diagonal centroid distance.
    """
    cids = sorted({int(c) for c in cluster_ids if c >= 0})
    if len(cids) < 2:
        return [], 0.0
    _, centroids = _cluster_centroids(u_cluster, cluster_ids)

    diff = centroids[:, None, :] - centroids[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    off_diag = dist[~np.eye(len(cids), dtype=bool)]
    sigma = float(np.median(off_diag))
    np.fill_diagonal(dist, np.inf)

    k = min(k_cluster, len(cids) - 1)
    pairs: dict[tuple[int, int], float] = {}
    for i in range(len(cids)):
        top = np.argpartition(dist[i], k - 1)[:k]
        for j in top:
            a, b = sorted((cids[i], cids[int(j)]))
            d = dist[i, int(j)]
            pairs[(a, b)] = float(np.exp(-((d / max(sigma, 1e-12)) ** 2)))
    edges = [[a, b, round(w, 3)] for (a, b), w in sorted(pairs.items())]
    return edges, sigma


def compute_point_knn(u_cluster: np.ndarray, k_point: int = 6) -> dict:
    """Exact per-point kNN (Euclidean) as flat columnar arrays.

    ``ids`` and ``sims`` are length ``n * k`` so the viewer can memcpy them
    into typed arrays; row ``i`` occupies ``[i*k, (i+1)*k)``. Self is
    excluded. ``sims`` uses the same Gaussian kernel as cluster edges, with
    its own ``sigma`` = median neighbor distance.
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(u_cluster)
    k = min(k_point, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(u_cluster)
    dist, idx = nn.kneighbors(u_cluster)
    dist, idx = dist[:, 1:], idx[:, 1:]  # column 0 is self (distance 0)
    sigma = float(np.median(dist))
    sims = np.round(np.exp(-((dist / max(sigma, 1e-12)) ** 2)), 3)
    return {
        "k": k,
        "sigma": round(sigma, 4),
        "ids": idx.astype(np.int64).ravel().tolist(),
        "sims": sims.ravel().tolist(),
    }


def compute_edges(
    u_cluster: np.ndarray,
    cluster_ids: np.ndarray,
    k_cluster: int = 5,
    k_point: int = 6,
    include_knn: bool = True,
) -> dict:
    """The ``edges`` block of nebulai.json (schema v2)."""
    cluster_edges, sigma = compute_cluster_edges(
        u_cluster, cluster_ids, k_cluster=k_cluster
    )
    edges = {
        "space": "umap10",
        "metric": "gaussian_euclidean",
        "k_cluster": k_cluster,
        "sigma": round(sigma, 4),
        "cluster_edges": cluster_edges,
    }
    if include_knn:
        edges["knn"] = compute_point_knn(u_cluster, k_point=k_point)
    return edges
