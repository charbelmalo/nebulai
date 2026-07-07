"""Export the map as nebulai.json — the contract the Phase-2 viewer loads."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..units import Units


SCHEMA_VERSION = 2


def export_json(
    path: Path,
    units: Units,
    u2: np.ndarray,
    u3: np.ndarray,
    cluster_ids: np.ndarray,
    probs: np.ndarray,
    titles: dict[int, str],
    namer_used: str,
    u_cluster: np.ndarray | None = None,
    edges_mode: str = "knn",
) -> dict:
    points = []
    for i in range(len(units)):
        points.append(
            {
                "id": i,
                "unit_ref": {
                    "kind": units.meta.get("unit", "unit"),
                    "index": int(units.ids[i]),
                },
                "label": units.labels[i],
                "confidence": round(float(probs[i]), 3),
                "layer": units.meta.get("layer"),
                "xy": [round(float(v), 4) for v in u2[i]],
                "xyz": [round(float(v), 4) for v in u3[i]],
                "cluster_id": int(cluster_ids[i]),
            }
        )

    clusters = []
    for cid in sorted({int(c) for c in cluster_ids if c >= 0}):
        members = np.where(cluster_ids == cid)[0]
        centroid = u3[members].mean(axis=0)
        clusters.append(
            {
                "id": cid,
                "title": titles.get(cid, ""),
                "size": int(len(members)),
                "centroid": [round(float(v), 4) for v in centroid],
            }
        )

    n_noise = int((cluster_ids < 0).sum())
    doc = {
        "meta": {
            **units.meta,
            "schema_version": SCHEMA_VERSION,
            "n_points": len(units),
            "n_clusters": len(clusters),
            "noise_fraction": round(n_noise / max(len(units), 1), 4),
            "namer": namer_used,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "points": points,
        "clusters": clusters,
    }
    # edges are computed in the clustering space (u_cluster), never u2/u3 —
    # beam similarity must reflect the geometry HDBSCAN saw, not the layout
    if u_cluster is not None and edges_mode != "none":
        from .edges import compute_edges

        doc["edges"] = compute_edges(
            u_cluster, cluster_ids, include_knn=(edges_mode == "knn")
        )
    path.write_text(json.dumps(doc, ensure_ascii=False))
    return doc["meta"]
