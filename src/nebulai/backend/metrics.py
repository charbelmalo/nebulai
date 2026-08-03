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

    # independent validation, when `nebulai validate` has been run for this map.
    # Absent by default: these re-run UMAP, so they are opt-in rather than part
    # of every build. None here means "not measured", never "measured as zero".
    val = {}
    val_path = dataset_dir / "validation.json"
    if val_path.exists():
        v = json.loads(val_path.read_text())
        val = {
            "trustworthiness": (v.get("trustworthiness") or {}).get("trustworthiness"),
            "stability_ari": (v.get("stability") or {}).get("mean_ari"),
            "null_silhouette": (v.get("null_baseline") or {}).get("silhouette"),
            # carried so the margin can be qualified: silhouette is not
            # comparable across wildly different cluster counts, so a null that
            # partitioned at a different granularity than the real map produces
            # a margin that means nothing. See `_margin()`.
            "null_clusters": (v.get("null_baseline") or {}).get("n_clusters"),
        }

    return {
        **val,
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

    def num(v, places: int = 4) -> str:
        return "n/a" if v is None else f"{v:.{places}f}"

    def sil(r: dict) -> str:
        return num(r["silhouette"])

    def margin(r: dict) -> str:
        """silhouette − null floor, flagged when the two aren't comparable.

        Silhouette rises as a partition gets coarser, so a null that resolved a
        very different number of clusters than the real map is being scored on a
        different question. `nebulai validate` rescales min_cluster_size for
        subsampling, but HDBSCAN still picks its own cluster count — when that
        lands outside 0.5-2x the real map's, the margin is marked `?`.
        """
        s, n = r.get("silhouette"), r.get("null_silhouette")
        if s is None or n is None:
            return "n/a"
        nk, rk = r.get("null_clusters"), r.get("n_clusters")
        suspect = bool(rk and nk) and not (0.5 <= nk / rk <= 2.0)
        return f"{s - n:+.4f}" + ("?" if suspect else "")

    # the independent-validation columns only appear once at least one map has
    # actually been validated — an all-"n/a" block would read as a failed
    # measurement rather than one that was never asked for
    validated = any(
        r.get(k) is not None
        for r in rows
        for k in ("trustworthiness", "stability_ari", "null_silhouette")
    )

    headers = ["map", "points", "clusters", "noise", "silhouette", "med.size"]
    if validated:
        headers += ["trust", "seed.ARI", "null.sil", "null.k", "margin"]
    table = []
    for r in rows:
        cells = [
            _source_label(r),
            str(r["n_points"]),
            str(r["n_clusters"]),
            f"{r['noise_fraction'] * 100:.1f}%",
            sil(r),
            "–" if r["median_cluster_size"] is None else str(r["median_cluster_size"]),
        ]
        if validated:
            cells += [
                num(r.get("trustworthiness")),
                num(r.get("stability_ari")),
                num(r.get("null_silhouette")),
                "n/a" if r.get("null_clusters") is None else str(r["null_clusters"]),
                margin(r),
            ]
        table.append(cells)

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

    if validated:
        lines.append("")
        lines.append(
            "  trust    neighbourhood preservation from the ORIGINAL space "
            "(1.0 = faithful; ~0.5 = chance).\n"
            "  seed.ARI mean pairwise agreement across UMAP seeds "
            "(1.0 = same map every seed; ~0 = chance).\n"
            "  null.sil silhouette of the SAME pipeline on column-shuffled "
            "vectors — the floor 'silhouette' has to clear.\n"
            "  null.k   clusters the null resolved. Silhouette rises as a "
            "partition coarsens, so this has to\n"
            "           sit near the map's own cluster count for the margin to "
            "be a like-for-like comparison.\n"
            "  margin   silhouette − null.sil. `?` marks a null that clustered "
            "outside 0.5-2x the map's k —\n"
            "           there the two numbers answer different questions and the "
            "margin is not evidence.\n"
            "           Small samples also inflate the null (UMAP invents "
            "separable islands from noise),\n"
            "           so a margin scored on a subsample is a conservative "
            "reading, not an optimistic one."
        )
    return "\n".join(lines)
