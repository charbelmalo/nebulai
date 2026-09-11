"""Export the map as nebulai.json — the contract the Phase-2 viewer loads."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..units import Units


SCHEMA_VERSION = 2


def public_meta(meta: dict) -> dict:
    """`Units.meta` minus its side channels.

    Everything in `meta` is copied verbatim into `nebulai.json`, which is what
    makes each artifact self-describing — and which means a front-end cannot use
    `meta` to hand the CLI a 50,000-element array without that array landing in
    the export. Keys beginning with `_` are the escape hatch: they travel from a
    front-end to the CLI (the token front-end's raw-space glitch channels use
    `_channels`) and are dropped here, so `nebulai.json` keeps its shape and its
    schema version.
    """
    return {k: v for k, v in meta.items() if not k.startswith("_")}


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
            **public_meta(units.meta),
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
