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

# Stable, high-contrast per-model colors (RGB 0..1). Indexing is `% len`, so
# this list is a correctness constraint, not decoration: comparing more maps
# than there are colors silently paints two clouds identically in the one view
# whose entire job is telling them apart. Keep it at least as long as the
# number of maps in out/ (18 and growing — the four corpus models pushed it
# past the 16 this list held, which `test_palette_covers_every_built_map`
# caught).
_PALETTE = [
    [0.20, 0.70, 1.00],  # blue
    [1.00, 0.45, 0.30],  # orange
    [0.45, 0.90, 0.45],  # green
    [0.85, 0.45, 1.00],  # purple
    [1.00, 0.82, 0.25],  # gold
    [0.30, 0.95, 0.85],  # teal
    [1.00, 0.55, 0.75],  # pink
    [0.60, 0.80, 0.30],  # lime
    [0.55, 0.60, 1.00],  # periwinkle
    [1.00, 0.70, 0.40],  # apricot
    [0.40, 0.85, 0.65],  # jade
    [0.90, 0.40, 0.55],  # rose
    [0.70, 0.65, 0.95],  # lilac
    [0.95, 0.90, 0.55],  # sand
    [0.35, 0.75, 0.80],  # steel
    [0.80, 0.55, 0.35],  # bronze
    [0.50, 0.85, 1.00],  # ice
    [0.95, 0.60, 0.20],  # amber
    [0.65, 0.90, 0.70],  # mint
    [0.75, 0.50, 0.85],  # orchid
    # Extended 2026-09-12 when the corpus models' W_U maps and the api-embedding
    # contrast map pushed out/ past twenty. The guard in tests/test_compare.py
    # is what caught it: the modulo below would have quietly given two maps the
    # same colour, and a legend that lies is worse than a missing legend.
    [0.30, 0.60, 0.85],  # denim
    [0.90, 0.75, 0.60],  # linen
    [0.55, 0.95, 0.40],  # spring
    [1.00, 0.40, 0.40],  # coral
    [0.45, 0.55, 0.70],  # slate
    [0.85, 0.85, 0.95],  # frost
]


def _source_label(meta: dict) -> str:
    """A short, human-readable identity for one map, distinguishing front-ends
    of the SAME model (token vs SAE vs neuron) — which `meta.model` alone
    cannot, so all three collapse into one cloud if keyed on the model id.

    Derived from the geometry origin (`meta.unit`): e.g. "SmolLM2-135M · SAE
    features", "SmolLM2-135M · MLP neurons", "SmolLM2-135M · tokens"."""
    model = str(meta.get("model", "?"))
    short = model.split("/")[-1]
    unit = str(meta.get("unit", ""))
    if unit.startswith("sae_decoder"):
        kind = "SAE features"
    elif unit.startswith("mlp_neuron"):
        kind = "MLP neurons"
    elif unit.startswith("api_text_embedding"):
        kind = "API embeddings"
    elif unit.startswith("probe_concept"):
        kind = "probe concepts"
    elif unit.startswith("token_embedding") or unit == "token_embedding":
        kind = "tokens"
    else:
        kind = unit or "units"
    return f"{short} · {kind}"


def _unique_labels(labels: list[str]) -> list[str]:
    """Make identity labels unique (append ' #2', ' #3', … on collision) so two
    maps that derive the same label never merge silently."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label} #{seen[label]}")
    return out


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
    return {
        "model": d["meta"]["model"],
        "label": _source_label(d["meta"]),
        "clusters": clusters,
    }


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
    embed_api: str = "ollama",
    embed_api_key: str | None = None,
) -> dict:
    models = [_load_model(p) for p in json_paths]
    # identify each cloud by its front-end/unit label, NOT meta.model — three
    # decompositions of one model (tokens/SAE/neurons) share a model id and
    # would otherwise collapse into a single legend entry / color / jaccard key
    model_ids = _unique_labels([m["label"] for m in models])

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
    E = embed_texts(
        texts,
        host=embed_host,
        model=embed_model,
        api=embed_api,
        api_key=embed_api_key,
    )
    u_cluster, u3, _u2 = reduce_vectors(E, cluster_dim=10, n_neighbors=15, seed=seed)
    meta_ids, _probs = cluster_units(
        u_cluster, min_cluster_size=3, min_samples=1, method="leaf"
    )

    semantic = _normalize(u3)

    # --- native state: each model's own cloud, normalized then quadranted ---
    #
    # WHAT THIS DESTROYS, deliberately: `_normalize(..., scale=9.0)` is applied
    # PER MODEL, so every cloud is rescaled to the same radius regardless of how
    # spread its own UMAP was. Cross-model distance in this state is therefore
    # meaningless in both directions — a model whose concepts genuinely sit
    # further apart is squeezed to look like one whose concepts sit close, and
    # the gap between two quadrants is `_grid_offsets`, a constant.
    #
    # It is still the right default. Each model got its OWN UMAP fit, and two
    # independent non-linear fits share no axes, no orientation and no scale, so
    # their raw coordinates were never comparable to begin with — normalizing
    # does not throw away a comparison, it declines to imply one. The comparable
    # states are `semantic`/`by_concept`, which put every model through ONE fit
    # of a shared embedding space.
    #
    # Quadrant assignment and palette index both follow `models` order, i.e. the
    # order the datasets were passed on the command line. Nothing about a
    # model's position here is a property of the model.
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
            "source_models": [m["model"] for m in models],
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


# ---------------------------------------------------------------------------
# Route B — orthogonal Procrustes over shared tokens (README roadmap)
# ---------------------------------------------------------------------------
#
# Everything above is Route A: it never touches raw geometry, because two
# models' embedding spaces have no shared basis. Route B asks a narrower
# question that *can* be answered in the raw spaces:
#
#     For two models with the SAME tokenizer, is there a single rigid rotation
#     that carries one model's token cloud onto the other's?
#
# If one exists, the two geometries agree up to a change of basis and the
# difference between their maps is a difference of coordinates, not of content.
# If none exists, they genuinely arrange the vocabulary differently.
#
# Three constraints make the answer mean something, and all three are enforced
# rather than merely documented:
#
# * **Same tokenizer only.** Aligning across tokenizers would pair token id 42
#   of one vocabulary with an unrelated string in the other; the result would be
#   noise with a rotation matrix attached.
# * **Held-out evaluation.** A rotation fitted on all 49,857 tokens and scored
#   on the same 49,857 is fitting, not testing. The reported residual is
#   measured on tokens the fit never saw.
# * **A permutation null.** A residual of 0.4 means nothing without knowing what
#   a *wrong* pairing scores. The null shuffles which row of B each row of A is
#   matched to and refits, holding both clouds' internal structure fixed and
#   destroying only the correspondence — which is exactly the hypothesis.

_ROUTE_B_MIN_SHARED = 256


class RouteBError(ValueError):
    """Route B was asked for a pair it cannot honestly answer for."""


def _orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """The R minimising ‖A R − B‖_F subject to RᵀR = I.

    For equal widths this is the classical solution `R = U Vᵀ` from the SVD of
    `AᵀB`. The same formula solves the unequal-width case with R semi-orthogonal
    (`d_a × d_b`), which is what lets gpt2 (768-d) be compared with gpt2-medium
    (1024-d) without throwing away dimensions by hand. That case is a projection
    and the result says so: a projection can only lose structure, so a *low*
    residual across widths is still evidence while a high one is partly an
    artifact of the width gap.
    """
    u, _, vt = np.linalg.svd(A.T @ B, full_matrices=False)
    return u @ vt


def _residual(A: np.ndarray, B: np.ndarray, R: np.ndarray) -> float:
    """‖A R − B‖²_F / ‖B‖²_F — 0 is a perfect fit, 1 is no better than zero."""
    denom = float((B * B).sum())
    if denom <= 0:
        return float("nan")
    return float(((A @ R - B) ** 2).sum() / denom)


def _prepare(V: np.ndarray) -> np.ndarray:
    """Centre, then scale to unit Frobenius norm.

    Orthogonal Procrustes has no scale parameter, so two clouds of different
    overall magnitude would report a large residual for a reason that has
    nothing to do with their shape. Normalising both leaves the residual
    measuring what it is meant to: relative arrangement.
    """
    X = np.asarray(V, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    n = float(np.sqrt((X * X).sum()))
    return X / n if n > 0 else X


def route_b_procrustes(
    model_a: str,
    model_b: str,
    *,
    max_tokens: int | None = None,
    n_permutations: int = 200,
    holdout_fraction: float = 0.5,
    seed: int = 0,
    remote: bool | None = False,
    units_loader=None,
) -> dict:
    """Fit and test a rigid alignment between two same-family token clouds.

    Returns a JSON-serialisable report. Raises `RouteBError` when the pair
    cannot support the question — a different tokenizer, or too few shared
    tokens — rather than returning a number that would look like an answer.
    """
    if units_loader is None:
        from ..frontends.tokens import load_token_units

        units_loader = load_token_units

    ua = units_loader(model_a, center=False, max_tokens=max_tokens, remote=remote)
    ub = units_loader(model_b, center=False, max_tokens=max_tokens, remote=remote)

    # Tokenizer identity is checked on the token STRINGS, not on the vocab size:
    # two tokenizers can agree on a count and disagree on every entry.
    la, lb = list(ua.labels), list(ub.labels)
    index_b = {s: i for i, s in enumerate(lb)}
    pairs = [(i, index_b[s]) for i, s in enumerate(la) if s in index_b]
    overlap = len(pairs) / max(len(la), len(lb)) if la and lb else 0.0
    if len(pairs) < _ROUTE_B_MIN_SHARED:
        raise RouteBError(
            f"{model_a} and {model_b} share only {len(pairs)} token strings "
            f"({overlap:.1%} of the larger vocabulary). Route B answers a "
            f"question about a shared vocabulary; below {_ROUTE_B_MIN_SHARED} "
            f"shared tokens there is no such vocabulary, and a rotation fitted "
            f"on what remains would describe the overlap rather than the "
            f"models. Use the Route A comparison for cross-tokenizer pairs."
        )

    ia = np.array([p[0] for p in pairs], dtype=int)
    ib = np.array([p[1] for p in pairs], dtype=int)
    A = _prepare(np.asarray(ua.vectors)[ia])
    B = _prepare(np.asarray(ub.vectors)[ib])

    rng = np.random.default_rng(seed)
    n = len(A)
    perm = rng.permutation(n)
    n_fit = max(1, int(round(n * (1.0 - holdout_fraction))))
    fit_idx, test_idx = perm[:n_fit], perm[n_fit:]
    if len(test_idx) < _ROUTE_B_MIN_SHARED // 4:
        raise RouteBError(
            f"holdout_fraction={holdout_fraction} leaves {len(test_idx)} "
            f"evaluation tokens, too few to distinguish a real alignment from a "
            f"lucky one."
        )

    R = _orthogonal_procrustes(A[fit_idx], B[fit_idx])
    resid_fit = _residual(A[fit_idx], B[fit_idx], R)
    resid_held = _residual(A[test_idx], B[test_idx], R)
    resid_full = _residual(A, B, _orthogonal_procrustes(A, B))

    # Permutation null: the same two clouds, the wrong correspondence. Refitting
    # inside the loop is the point — the null must be "the best rotation
    # available to a wrong pairing", not "this rotation applied to a wrong
    # pairing", which would be trivial to beat.
    null: list[float] = []
    for _ in range(n_permutations):
        Rn = _orthogonal_procrustes(A[fit_idx], B[fit_idx][rng.permutation(len(fit_idx))])
        shuffled_test = B[test_idx][rng.permutation(len(test_idx))]
        null.append(_residual(A[test_idx], shuffled_test, Rn))
    null_arr = np.asarray([v for v in null if np.isfinite(v)], dtype=np.float64)
    # Lower residual = better alignment, so the tail of interest is the LEFT one.
    r = int((null_arr <= resid_held).sum())
    p_value = (r + 1) / (len(null_arr) + 1) if len(null_arr) else float("nan")

    return {
        "route": "B",
        "method": "orthogonal Procrustes over shared tokens, held-out residual",
        "model_a": model_a,
        "model_b": model_b,
        "dim_a": int(A.shape[1]),
        "dim_b": int(B.shape[1]),
        "square_rotation": bool(A.shape[1] == B.shape[1]),
        "n_shared_tokens": int(n),
        "vocab_overlap": round(float(overlap), 6),
        "n_fit": int(len(fit_idx)),
        "n_heldout": int(len(test_idx)),
        "residual_fit": round(resid_fit, 6),
        "residual_heldout": round(resid_held, 6),
        "residual_full_insample": round(resid_full, 6),
        "alignment_heldout": round(1.0 - resid_held, 6),
        "null_residual_mean": round(float(null_arr.mean()), 6) if len(null_arr) else None,
        "null_residual_min": round(float(null_arr.min()), 6) if len(null_arr) else None,
        "n_permutations_effective": int(len(null_arr)),
        "p_value": round(float(p_value), 6) if np.isfinite(p_value) else None,
        "seed": int(seed),
        "interpretation": (
            "residual_heldout is the fraction of the target cloud's variance the "
            "fitted rotation fails to explain on tokens it never saw; p_value is "
            "the (r+1)/(B+1) permutation probability of matching it under a "
            "shuffled token correspondence. A low residual with a small p means "
            "the two models arrange this shared vocabulary the same way up to a "
            "change of basis. It does NOT mean the models behave alike, and it "
            "ranks neither of them."
        ),
    }
