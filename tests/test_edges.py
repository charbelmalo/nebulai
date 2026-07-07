"""Edges module tests — synthetic 10-D blobs, no UMAP/HDBSCAN needed."""

import json
from pathlib import Path

import numpy as np
import pytest

from nebulai.backend.edges import (
    compute_cluster_edges,
    compute_edges,
    compute_point_knn,
)


@pytest.fixture()
def blobs() -> tuple[np.ndarray, np.ndarray]:
    """Four well-separated 10-D blobs of 25 points, plus 10 noise points."""
    rng = np.random.default_rng(42)
    centers = rng.normal(size=(4, 10)) * 10.0
    pts, cids = [], []
    for cid, c in enumerate(centers):
        pts.append(c + rng.normal(scale=0.1, size=(25, 10)))
        cids.extend([cid] * 25)
    pts.append(rng.normal(scale=10.0, size=(10, 10)))
    cids.extend([-1] * 10)
    return np.vstack(pts).astype(np.float32), np.array(cids)


def test_cluster_edges_deduped_and_bounded(blobs):
    u, cids = blobs
    edges, sigma = compute_cluster_edges(u, cids, k_cluster=2)
    assert edges, "separated blobs must still yield top-k edges"
    assert sigma > 0
    seen = set()
    for a, b, w in edges:
        assert a < b, "edges must be deduped a<b"
        assert (a, b) not in seen
        seen.add((a, b))
        assert 0.0 < w <= 1.0, "gaussian weight out of range"
        assert a >= 0 and b >= 0, "noise (-1) must never appear in edges"


def test_cluster_edges_union_of_topk(blobs):
    u, cids = blobs
    # k = n_clusters - 1 → complete graph: C(4,2) = 6 edges
    edges, _ = compute_cluster_edges(u, cids, k_cluster=3)
    assert len(edges) == 6


def test_cluster_edges_weights_have_dynamic_range(blobs):
    """Guard against metric saturation (cosine on UMAP output was ~all 1.0)."""
    u, cids = blobs
    edges, _ = compute_cluster_edges(u, cids, k_cluster=3)
    w = np.array([e[2] for e in edges])
    assert w.max() - w.min() > 0.1


def test_cluster_edges_degenerate_cases(blobs):
    u, cids = blobs
    assert compute_cluster_edges(u, np.full_like(cids, -1)) == ([], 0.0)
    assert compute_cluster_edges(u, np.zeros_like(cids)) == ([], 0.0)


def test_point_knn_shape_and_self_exclusion(blobs):
    u, cids = blobs
    n = len(u)
    knn = compute_point_knn(u, k_point=6)
    k = knn["k"]
    assert k == 6
    assert len(knn["ids"]) == n * k, "ids must be flat N*k"
    assert len(knn["sims"]) == n * k, "sims must be flat N*k"
    assert knn["sigma"] > 0
    ids = np.asarray(knn["ids"]).reshape(n, k)
    sims = np.asarray(knn["sims"]).reshape(n, k)
    for i in range(n):
        assert i not in ids[i], "kNN must exclude self"
    assert sims.min() >= 0.0 and sims.max() <= 1.0
    # tight blobs: nearest neighbors of a clustered point share its cluster
    assert (cids[ids[0]] == cids[0]).all()


def test_point_knn_k_clamped():
    u = np.random.default_rng(0).normal(size=(4, 10))
    knn = compute_point_knn(u, k_point=10)
    assert knn["k"] == 3


def test_compute_edges_block_shape(blobs):
    u, cids = blobs
    edges = compute_edges(u, cids)
    assert edges["space"] == "umap10"
    assert edges["metric"] == "gaussian_euclidean"
    assert edges["k_cluster"] == 5
    assert edges["sigma"] > 0
    assert "cluster_edges" in edges and "knn" in edges
    no_knn = compute_edges(u, cids, include_knn=False)
    assert "knn" not in no_knn
    json.dumps(edges)  # must be JSON-serializable (no numpy scalars)


def test_backfill_round_trip(blobs, tmp_path: Path):
    """Simulate `nebulai edges`: v1 doc + cached u_cluster → v2 doc."""
    from nebulai.backend.export import SCHEMA_VERSION

    u, cids = blobs
    doc = {
        "meta": {"n_points": len(u)},
        "points": [{"id": i, "cluster_id": int(c)} for i, c in enumerate(cids)],
        "clusters": [],
    }
    jp = tmp_path / "nebulai.json"
    jp.write_text(json.dumps(doc))

    loaded = json.loads(jp.read_text())
    loaded["edges"] = compute_edges(
        u, np.array([p["cluster_id"] for p in loaded["points"]])
    )
    loaded["meta"]["schema_version"] = SCHEMA_VERSION
    jp.write_text(json.dumps(loaded, ensure_ascii=False))

    back = json.loads(jp.read_text())
    assert back["meta"]["schema_version"] == 2
    assert back["edges"]["cluster_edges"]
    assert len(back["edges"]["knn"]["ids"]) == len(u) * back["edges"]["knn"]["k"]
