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


# --- the validation columns ----------------------------------------------


def _write_validation(d, *, null_sil, null_k):
    (d / "validation.json").write_text(
        json.dumps(
            {
                "trustworthiness": {"trustworthiness": 0.75},
                "stability": {"mean_ari": 0.5},
                "null_baseline": {"silhouette": null_sil, "n_clusters": null_k},
            }
        )
    )


def test_validation_columns_absent_until_a_map_is_validated(tmp_path):
    d = tmp_path / "plain"
    _write_map(d, [0, 0, 1])
    table = format_table([compute_map_metrics(d)])
    # an all-"n/a" block reads as a failed measurement, not an unrequested one
    assert "margin" not in table
    assert "null.sil" not in table


def test_margin_is_flagged_when_the_null_clustered_at_another_granularity(tmp_path):
    """The real case: gpt2-small's SAE map has 69 clusters, its null resolved
    16. Silhouette rises as a partition coarsens, so 0.5246 vs 0.5737 is not
    the real map losing to noise — it is two different questions. The number is
    still printed (hiding it would be worse), but marked as not-evidence."""
    d = tmp_path / "sae"
    rng = np.random.RandomState(0)
    u = np.vstack([rng.randn(20, 3) * 0.01, rng.randn(20, 3) * 0.01 + 10])
    _write_map(d, [0] * 20 + [1] * 20, u_cluster=u)
    def margin_cell(dataset_dir) -> str:
        # the footnote explains the `?`, so only the data row can be inspected
        table = format_table([compute_map_metrics(dataset_dir)])
        return table.splitlines()[2].split()[-1]

    _write_validation(d, null_sil=0.2, null_k=16)  # 16 vs 2 clusters -> 8x
    assert compute_map_metrics(d)["null_clusters"] == 16
    assert margin_cell(d).endswith("?")

    _write_validation(d, null_sil=0.2, null_k=3)  # 3 vs 2 -> within 0.5-2x
    assert not margin_cell(d).endswith("?")
    assert margin_cell(d).startswith("+0.7")  # ~0.9986 real - 0.2 null


def test_margin_reports_a_negative_result_rather_than_swallowing_it(tmp_path):
    d = tmp_path / "weak"
    rng = np.random.RandomState(0)
    u = np.vstack([rng.randn(20, 3) * 0.01, rng.randn(20, 3) * 0.01 + 10])
    _write_map(d, [0] * 20 + [1] * 20, u_cluster=u)
    _write_validation(d, null_sil=0.999, null_k=2)
    table = format_table([compute_map_metrics(d)])
    assert "-0." in table  # a map under its own floor says so
