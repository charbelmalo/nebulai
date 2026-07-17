"""Structural metrics for a map — the quantitative side of the comparison.

The whole point of the three front-ends is that they are three decompositions
of one model on an identical back-end, so their maps can be *measured* against
each other, not just eyeballed. This module computes those numbers with ONE
shared function, so Plan A / B / C are scored identically:

  - n_points, n_clusters, noise_fraction  — how much structure was found
  - silhouette                            — how well-separated that structure is
  - mean / median cluster size            — how the mass is distributed

Silhouette is computed in `u_cluster` (the 10-D UMAP space HDBSCAN actually
clustered), over non-noise points only, with euclidean distance (the space is a
euclidean embedding, and that is the geometry HDBSCAN saw). It is sampled for
large maps because silhouette is O(n²).

The headline the comparison is meant to show: the SAE map should separate into
more, cleaner clusters (more clusters / lower noise / higher silhouette) than
the raw-neuron map — SAE features are trained to be monosemantic, raw neurons
are polysemantic. This module makes that claim checkable rather than asserted.
"""

import json
from pathlib import Path

import numpy as np

# silhouette is O(n^2); above this many non-noise points we subsample
_SILHOUETTE_SAMPLE_CAP = 8000


def _silhouette(u_cluster: np.ndarray, labels: np.ndarray) -> float | None:
    """Mean silhouette over non-noise points, or None if it is undefined
    (fewer than 2 clusters, or too few points)."""
    mask = labels >= 0
    X = u_cluster[mask]
    y = labels[mask]
    n = len(X)
    n_clusters = len(set(y.tolist()))
    if n_clusters < 2 or n <= n_clusters:
        return None
    from sklearn.metrics import silhouette_score

    sample = _SILHOUETTE_SAMPLE_CAP if n > _SILHOUETTE_SAMPLE_CAP else None
    score = silhouette_score(
        X, y, metric="euclidean", sample_size=sample, random_state=0
    )
    return round(float(score), 4)


def compute_map_metrics(dataset_dir: Path) -> dict:
    """Structural metrics for one exported map directory.

    Reads `nebulai.json` (cluster ids/labels + stamped meta) and, when present,
    `reduced.npz` (`u_cluster`, the clustering space) for the silhouette. All
    counts are recomputed from the points so the metrics are self-consistent
    even if meta drifts. `silhouette` is None when it is undefined or when the
    reduction cache is absent."""
    dataset_dir = Path(dataset_dir)
    doc = json.loads((dataset_dir / "nebulai.json").read_text())
    meta = doc["meta"]
    cluster_ids = np.array([p["cluster_id"] for p in doc["points"]], dtype=int)

    n_points = int(len(cluster_ids))
    labelled = cluster_ids[cluster_ids >= 0]
    cluster_labels = sorted(set(labelled.tolist()))
    n_clusters = len(cluster_labels)
    n_noise = int((cluster_ids < 0).sum())
    sizes = np.array(
        [int((cluster_ids == c).sum()) for c in cluster_labels], dtype=int
    )

    silhouette = None
    npz = dataset_dir / "reduced.npz"
    if npz.exists():
        u_cluster = np.load(npz)["u_cluster"]
        if len(u_cluster) == n_points:
            silhouette = _silhouette(u_cluster, cluster_ids)

    return {
        "id": dataset_dir.name,
        "model": meta.get("model"),
        "unit": meta.get("unit"),
        "n_points": n_points,
        "n_clusters": n_clusters,
        "noise_fraction": round(n_noise / max(n_points, 1), 4),
        "silhouette": silhouette,
        "mean_cluster_size": round(float(sizes.mean()), 1) if n_clusters else None,
        "median_cluster_size": int(np.median(sizes)) if n_clusters else None,
        "n_labeled": meta.get("n_labeled"),
        "labels_source": meta.get("labels_source"),
        "hdbscan": meta.get("hdbscan"),
    }


def format_table(rows: list[dict]) -> str:
    """A fixed-width comparison table, one map per row. Label detection is not
    a column: with `--labels none` maps every unit is a placeholder, so a
    detection score would be hollow — we print the labeled-unit count instead
    and say so."""
    from .compare import _source_label

    def sil(r: dict) -> str:
        return "n/a" if r["silhouette"] is None else f"{r['silhouette']:.4f}"

    headers = ["map", "points", "clusters", "noise", "silhouette", "med.size"]
    table = []
    for r in rows:
        table.append(
            [
                _source_label(r),
                str(r["n_points"]),
                str(r["n_clusters"]),
                f"{r['noise_fraction'] * 100:.1f}%",
                sil(r),
                "–" if r["median_cluster_size"] is None else str(r["median_cluster_size"]),
            ]
        )

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table)) if table else len(headers[i])
        for i in range(len(headers))
    ]

    def fmt(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    lines = [fmt(headers), fmt(["-" * w for w in widths])]
    lines += [fmt(row) for row in table]

    # honesty footnote on labels — count only maps deliberately built with
    # `--labels none` (0 labeled units). The token map's labels are the token
    # strings themselves, so it is NOT counted here even though it carries no
    # auto-interp feature labels.
    unlabeled = [
        r
        for r in rows
        if r.get("labels_source") is not None and (r.get("n_labeled") or 0) == 0
    ]
    if unlabeled:
        lines.append("")
        lines.append(
            f"  label-detection: n/a for {len(unlabeled)} of {len(rows)} maps "
            "(--labels none → 0 labeled units); structural separation only."
        )
    return "\n".join(lines)
