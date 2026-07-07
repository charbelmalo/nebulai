"""Cross-model comparison: combine several models' clouds and categorize them.

The honest move (see docs/DETAILS.md): raw embedding geometries from different
models don't share a basis, so we DON'T concatenate them. Instead each model's
*named clusters* are embedded in a neutral third-party space (mxbai-embed-large
on the M4), then co-reduced and re-clustered. A meta-cluster that draws from
several models is a **shared concept**; one from a single model is **unique**.

The export carries, per meta-point (= one source-model cluster), its position
in several *layout states* so the WebGPU viewer can smoothly interpolate
between them:

  native   — each model's own 3D cloud, offset into its own quadrant
  semantic — the unified concept space (matching concepts converge)
  by_model — models fanned into columns (each model's footprint)
  by_concept — points collapsed onto their meta-cluster (shared knots pop)
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from .cluster import cluster_units
from .embed import embed_texts
from .reduce import reduce_vectors

# stable, high-contrast per-model colors (RGB 0..1)
_PALETTE = [
    [0.20, 0.70, 1.00],  # blue
    [1.00, 0.45, 0.30],  # orange
    [0.45, 0.90, 0.45],  # green
    [0.85, 0.45, 1.00],  # purple
    [1.00, 0.82, 0.25],  # gold
    [0.30, 0.95, 0.85],  # teal
]


def _load_model(json_path: Path) -> dict:
    d = json.loads(json_path.read_text())
    members: dict[int, list[str]] = {}
    for p in d["points"]:
        c = int(p["cluster_id"])
        if c >= 0:
            members.setdefault(c, []).append(p["label"])
    clusters = []
    for c in d["clusters"]:
        cid = int(c["id"])
        clusters.append(
            {
                "cluster_id": cid,
                "title": c["title"],
                "size": int(c["size"]),
                "centroid": np.asarray(c["centroid"], dtype=np.float32),
                "members": members.get(cid, [])[:12],
            }
        )
    return {"model": d["meta"]["model"], "clusters": clusters}


def _normalize(P: np.ndarray, scale: float = 10.0) -> np.ndarray:
    P = P - P.mean(axis=0)
    r = np.abs(P).max() + 1e-8
    return (P * (scale / r)).astype(np.float32)


def _grid_offsets(n: int, spacing: float = 26.0) -> list[np.ndarray]:
    """Corner offsets so each model's native cloud sits in its own quadrant."""
    cols = int(np.ceil(np.sqrt(n)))
    offs = []
    for i in range(n):
        gx, gy = i % cols, i // cols
        offs.append(np.array([gx * spacing, -gy * spacing, 0.0], dtype=np.float32))
    center = np.mean(offs, axis=0) if offs else np.zeros(3)
    return [o - center for o in offs]


def build_comparison(
    json_paths: list[Path],
    embed_host: str,
    embed_model: str = "mxbai-embed-large",
    seed: int = 42,
) -> dict:
    models = [_load_model(p) for p in json_paths]
    model_ids = [m["model"] for m in models]

    # one meta-point per (model, cluster)
    src, titles, sizes, texts, native = [], [], [], [], []
    for mi, m in enumerate(models):
        for c in m["clusters"]:
            src.append(mi)
            titles.append(c["title"])
            sizes.append(c["size"])
            native.append(c["centroid"])
            texts.append(f"{c['title']}. tokens: " + ", ".join(c["members"]))
    src = np.asarray(src)
    sizes = np.asarray(sizes, dtype=np.float32)

    # --- neutral semantic space + meta clustering ---
    # Meta-points are already-aggregated concepts, so we want FINE granularity:
    # min_samples=1 (not cluster.py's default 5, which would force every
    # meta-cluster to span several models and erase all "unique" concepts).
    E = embed_texts(texts, host=embed_host, model=embed_model)
    u_cluster, u3, _u2 = reduce_vectors(E, cluster_dim=10, n_neighbors=15, seed=seed)
    meta_ids, _probs = cluster_units(
        u_cluster, min_cluster_size=3, min_samples=1, method="leaf"
    )

    semantic = _normalize(u3)

    # --- native state: each model's own cloud, normalized then quadranted ---
    native = np.asarray(native, dtype=np.float32)
    offs = _grid_offsets(len(models))
    native_state = np.zeros_like(semantic)
    for mi in range(len(models)):
        idx = np.where(src == mi)[0]
        if len(idx):
            native_state[idx] = _normalize(native[idx], scale=9.0) + offs[mi]

    # --- by_model: fan models into columns, keep semantic y/z ---
    by_model = semantic.copy()
    col = (src - src.mean()) * 22.0
    by_model[:, 0] = col + semantic[:, 0] * 0.18

    # --- by_concept: collapse onto meta-cluster centroid (shared knots pop) ---
    by_concept = semantic.copy()
    for cid in set(int(x) for x in meta_ids if x >= 0):
        idx = np.where(meta_ids == cid)[0]
        c = semantic[idx].mean(axis=0)
        by_concept[idx] = c + (semantic[idx] - c) * 0.14

    # --- categorize meta-clusters: shared vs unique ---
    meta_clusters = []
    shared_pt = np.zeros(len(src), dtype=bool)
    for cid in sorted(set(int(x) for x in meta_ids if x >= 0)):
        idx = np.where(meta_ids == cid)[0]
        contributing = sorted(set(int(src[i]) for i in idx))
        is_shared = len(contributing) > 1
        shared_pt[idx] = is_shared
        rep = titles[idx[int(np.argmax(sizes[idx]))]]
        meta_clusters.append(
            {
                "id": cid,
                "title": rep,
                "models": [model_ids[k] for k in contributing],
                "n_models": len(contributing),
                "shared": is_shared,
                "size": int(len(idx)),
            }
        )

    # per-model concept sets (meta-cluster ids each model reaches)
    reach = {
        mi: set(int(meta_ids[i]) for i in np.where(src == mi)[0] if meta_ids[i] >= 0)
        for mi in range(len(models))
    }
    jaccard = {}
    for a, b in combinations(range(len(models)), 2):
        inter = len(reach[a] & reach[b])
        union = len(reach[a] | reach[b]) or 1
        jaccard[f"{model_ids[a]} vs {model_ids[b]}"] = round(inter / union, 3)

    n_shared = sum(1 for mc in meta_clusters if mc["shared"])
    unique = {
        model_ids[mi]: sum(
            1
            for mc in meta_clusters
            if not mc["shared"] and model_ids[mi] in mc["models"]
        )
        for mi in range(len(models))
    }

    points = []
    for i in range(len(src)):
        points.append(
            {
                "source": model_ids[int(src[i])],
                "source_idx": int(src[i]),
                "title": titles[i],
                "size": int(sizes[i]),
                "meta_cluster": int(meta_ids[i]),
                "shared": bool(shared_pt[i]),
                "color": _PALETTE[int(src[i]) % len(_PALETTE)],
                "positions": {
                    "native": native_state[i].round(3).tolist(),
                    "semantic": semantic[i].round(3).tolist(),
                    "by_model": by_model[i].round(3).tolist(),
                    "by_concept": by_concept[i].round(3).tolist(),
                },
            }
        )

    return {
        "meta": {
            "models": model_ids,
            "n_points": len(points),
            "n_meta_clusters": len(meta_clusters),
            "embed_model": embed_model,
        },
        "states": ["native", "semantic", "by_model", "by_concept"],
        "colors": {model_ids[i]: _PALETTE[i % len(_PALETTE)] for i in range(len(models))},
        "stats": {
            "n_shared_concepts": n_shared,
            "n_unique_per_model": unique,
            "jaccard": jaccard,
        },
        "points": points,
        "meta_clusters": meta_clusters,
    }


def export_comparison(out_path: Path, comparison: dict) -> None:
    out_path.write_text(json.dumps(comparison))
