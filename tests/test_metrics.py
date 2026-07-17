"""Structural-metrics tests — counts recomputed from points, silhouette from
the reduction cache, and the honest label footnote. Offline: builds tiny
nebulai.json / reduced.npz fixtures in tmp."""

import json

import numpy as np

from nebulai.backend.metrics import compute_map_metrics, format_table


def _write_map(d, points, meta_extra=None, u_cluster=None):
    d.mkdir(parents=True, exist_ok=True)
    doc = {
        "meta": {"model": "m", "unit": "token_embedding", **(meta_extra or {})},
        "points": [{"cluster_id": c} for c in points],
        "clusters": [],
    }
    (d / "nebulai.json").write_text(json.dumps(doc))
    if u_cluster is not None:
        np.savez_compressed(d / "reduced.npz", u_cluster=u_cluster.astype(np.float32))


def test_counts_recomputed_from_points(tmp_path):
    d = tmp_path / "map_a"
    _write_map(d, points=[0, 0, 1, 1, 1, -1])  # 2 clusters, 1 noise of 6
    m = compute_map_metrics(d)
    assert m["n_points"] == 6
    assert m["n_clusters"] == 2
    assert m["noise_fraction"] == round(1 / 6, 4)
    assert m["median_cluster_size"] == 2  # sizes {2,3} -> median 2.5 -> int 2
    assert m["silhouette"] is None  # no reduced.npz


def test_silhouette_present_when_reduction_cached(tmp_path):
    d = tmp_path / "map_b"
    # two tight, well-separated blobs in 3-D -> high silhouette
    rng = np.random.RandomState(0)
    a = rng.randn(20, 3) * 0.01 + np.array([0, 0, 0])
    b = rng.randn(20, 3) * 0.01 + np.array([10, 10, 10])
    u = np.vstack([a, b])
    points = [0] * 20 + [1] * 20
    _write_map(d, points=points, u_cluster=u)
    m = compute_map_metrics(d)
    assert m["silhouette"] is not None
    assert m["silhouette"] > 0.9


def test_format_table_footnote_only_counts_labels_none(tmp_path):
    tok = tmp_path / "tok"
    sae = tmp_path / "sae"
    _write_map(tok, [0, 0, 1], meta_extra={"unit": "token_embedding"})
    _write_map(
        sae,
        [0, 0, 1],
        meta_extra={
            "unit": "sae_decoder(r, layers.21.mlp)",
            "labels_source": "none",
            "n_labeled": 0,
        },
    )
    table = format_table([compute_map_metrics(tok), compute_map_metrics(sae)])
    # the token map is NOT counted as unlabeled; only the --labels none SAE is
    assert "n/a for 1 of 2 maps" in table
