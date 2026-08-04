"""Independent validation for a map — metrics that are NOT the construction.

`metrics.py` scores the map with silhouette, computed in `u_cluster` — the same
10-D UMAP space HDBSCAN clustered in. That is a useful descriptive number, but
it cannot tell you whether the clusters exist in the model's original space or
whether they are an artifact of one lucky UMAP seed: it grades the projection
using the projection's own geometry.

This module adds the three checks that close that loop, in increasing order of
how much they can embarrass the map:

  1. trustworthiness  — did UMAP preserve the ORIGINAL-space neighbourhoods?
                        Reads `vectors`, so it is independent of HDBSCAN
                        entirely. Low trustworthiness means the layout invented
                        adjacency that the model does not have.
  2. seed stability   — re-run reduce+cluster under several seeds and score the
                        clusterings against each other with ARI. Answers "would
                        I have drawn the same map on a different day?"
  3. null baseline    — run the identical pipeline on column-shuffled vectors,
                        which preserve every per-dimension marginal but destroy
                        the correlations between dimensions. UMAP+HDBSCAN will
                        happily find "clusters" in that. Whatever silhouette it
                        scores is the FLOOR any real number has to clear.

Nothing here is cheap: 2 and 3 re-run UMAP, which is the expensive step (and
`seed >= 0` forces UMAP single-threaded). They are opt-in, subsample-capped,
and every returned dict records the sample size it actually used so a number
can never be read as covering more points than it saw.

TWO CAVEATS, both found while building this and both load-bearing when you read
the output:

  - The null baseline is only meaningful AT THE MAP'S REAL SCALE. On a few
    hundred points UMAP manufactures clean separable islands out of shuffled
    noise, and the null can out-score the real map (measured: silhouette 0.88
    on shuffled data vs 0.43 on three genuinely separated gaussian blobs,
    n=180). That is not a bug in the shuffle — it is the sharpest possible
    statement of why silhouette-on-a-UMAP-space is weak evidence. Do not read
    a subsampled null as a floor for a full-size map.

  - Seed stability is dominated by `cluster_selection_method`. On three clean
    blobs, `eom` recovers exactly 3 clusters with ARI 1.00 across seeds, while
    `leaf` — the project default, chosen because `eom` collapses token maps
    into one mega-cluster — splits them into 8 and drops to ARI 0.37. So a low
    stability score on a real map is a statement about the leaf/eom trade-off
    as much as about the model. Report the method alongside the number.
"""

import numpy as np

# trustworthiness is O(n^2) in both time and memory (it builds a full pairwise
# distance matrix on each side); 5000 points is ~200MB per side in float64
_TRUST_SAMPLE_CAP = 5000

# re-running UMAP per seed is the expensive part — cap harder than trustworthiness
_STABILITY_SAMPLE_CAP = 4000


def _subsample(n: int, cap: int, seed: int) -> np.ndarray | None:
    """Row indices to score on, or None to use every row."""
    if n <= cap:
        return None
    return np.sort(np.random.RandomState(seed).choice(n, size=cap, replace=False))


def scale_cluster_kwargs(cluster_kwargs: dict, n_full: int, n_sample: int) -> dict:
    """Rescale `min_cluster_size` so a subsample is clustered at the same
    GRANULARITY as the full map.

    Without this the comparison is silently apples-to-oranges. HDBSCAN's
    min_cluster_size is an absolute point count, so carrying the full map's
    value onto a subsample changes what "a cluster" means: SmolLM2's token map
    resolves to min_cluster_size=48 over 48,636 points (~1 cluster floor per
    1000 points), and applying 48 to a 4,000-point null asks for clusters 12x
    larger in relative terms — which produced 6 giant clusters and a silhouette
    of 0.43, against 51 clusters and 0.21 for GPT-2's null under a different
    stamped value. Neither number described its own map.

    Scaling keeps the ratio fixed, so the null and the stability sweep cluster
    the sample the way the pipeline would have clustered a map of that size.
    """
    if n_sample >= n_full or "min_cluster_size" not in cluster_kwargs:
        return dict(cluster_kwargs)
    out = dict(cluster_kwargs)
    out["min_cluster_size"] = max(
        2, round(cluster_kwargs["min_cluster_size"] * n_sample / n_full)
    )
    return out


def trustworthiness_score(
    vectors: np.ndarray,
    embedding: np.ndarray,
    n_neighbors: int = 15,
    sample_cap: int = _TRUST_SAMPLE_CAP,
    seed: int = 0,
) -> dict:
    """How well `embedding` preserves the k-NN structure of `vectors`.

    1.0 means every embedding neighbourhood was already a neighbourhood in the
    original space; ~0.5 is what you get from an unrelated layout. This is the
    one metric here that does not involve HDBSCAN at all, so a map can have a
    great silhouette and poor trustworthiness — that combination means the
    clusters are real *in the projection* and the projection is not faithful.

    CAVEAT, and it matters: when subsampled, neighbourhoods are computed within
    the sample. A point's true nearest neighbours may not have been sampled, so
    the score is noisier than a full-set score and is not directly comparable
    across maps with different sample sizes. `n_scored` is returned so the
    caller can say which it is.
    """
    from sklearn.manifold import trustworthiness

    n = len(vectors)
    if n != len(embedding):
        raise ValueError(f"length mismatch: {n} vectors vs {len(embedding)} embedded")
    if n <= n_neighbors + 1:
        return {"trustworthiness": None, "n_scored": n, "n_neighbors": n_neighbors}

    idx = _subsample(n, sample_cap, seed)
    X = vectors if idx is None else vectors[idx]
    Y = embedding if idx is None else embedding[idx]
    # k must stay below n/2 for trustworthiness to be defined
    k = min(n_neighbors, (len(X) - 1) // 2)
    score = trustworthiness(X, Y, n_neighbors=k)
    return {
        "trustworthiness": round(float(score), 4),
        "n_scored": int(len(X)),
        "n_neighbors": int(k),
        "subsampled": idx is not None,
    }


def seed_stability(
    vectors: np.ndarray,
    seeds: tuple[int, ...] = (42, 1, 2, 3),
    reduce_kwargs: dict | None = None,
    cluster_kwargs: dict | None = None,
    sample_cap: int = _STABILITY_SAMPLE_CAP,
    sample_seed: int = 0,
) -> dict:
    """Re-run reduce+cluster under each seed; score every PAIR of clusterings
    with the adjusted Rand index.

    ARI is 1.0 for identical partitions and ~0.0 for the agreement you would
    expect by chance, so it is already corrected for cluster count — a map that
    fragments differently every seed scores near zero even if the cluster
    *count* is stable.

    Noise is scored as its own label rather than dropped. That is deliberate:
    if reseeding moves a third of the points in and out of noise, that IS
    instability, and excluding noise would hide exactly the failure this metric
    exists to catch.

    Note this recomputes a reference clustering per seed on the SAME (possibly
    subsampled) vectors rather than comparing against the shipped labels — the
    shipped labels were fit on all points, so comparing them to a subsample's
    clustering would measure the subsampling, not the seed.
    """
    from sklearn.metrics import adjusted_rand_score

    from .cluster import cluster_units
    from .reduce import reduce_vectors

    reduce_kwargs = dict(reduce_kwargs or {})
    cluster_kwargs = dict(cluster_kwargs or {})
    reduce_kwargs.pop("seed", None)

    idx = _subsample(len(vectors), sample_cap, sample_seed)
    X = vectors if idx is None else vectors[idx]

    labelings: dict[int, np.ndarray] = {}
    per_seed = []
    for s in seeds:
        u_cluster, _u3, _u2 = reduce_vectors(X, seed=s, **reduce_kwargs)
        ids, _probs = cluster_units(u_cluster, **cluster_kwargs)
        labelings[s] = ids
        per_seed.append(
            {
                "seed": int(s),
                "n_clusters": len({int(c) for c in ids if c >= 0}),
                "noise_fraction": round(float((ids < 0).mean()), 4),
            }
        )

    pairs = []
    ordered = list(seeds)
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            a, b = ordered[i], ordered[j]
            pairs.append(
                {
                    "seeds": [int(a), int(b)],
                    "ari": round(float(adjusted_rand_score(labelings[a], labelings[b])), 4),
                }
            )

    aris = [p["ari"] for p in pairs]
    return {
        "mean_ari": round(float(np.mean(aris)), 4) if aris else None,
        "min_ari": round(float(np.min(aris)), 4) if aris else None,
        "n_scored": int(len(X)),
        "subsampled": idx is not None,
        "seeds": [int(s) for s in seeds],
        "per_seed": per_seed,
        "pairs": pairs,
    }


def null_baseline(
    vectors: np.ndarray,
    reduce_kwargs: dict | None = None,
    cluster_kwargs: dict | None = None,
    sample_cap: int = _STABILITY_SAMPLE_CAP,
    seed: int = 42,
) -> dict:
    """The same pipeline on structure-free data — the floor a real map must clear.

    Each column is shuffled independently. That preserves the exact marginal
    distribution of every dimension (so the null has the same scale, sparsity
    and anisotropy as the real embedding matrix) while destroying the
    correlations BETWEEN dimensions, which is where representational structure
    actually lives.

    UMAP+HDBSCAN still returns clusters on this. That is the point: if the real
    map's silhouette is 0.45 and the null scores 0.38, the separation you are
    looking at is mostly the procedure, not the model.
    """
    from .cluster import cluster_units
    from .metrics import _silhouette
    from .reduce import reduce_vectors

    reduce_kwargs = dict(reduce_kwargs or {})
    cluster_kwargs = dict(cluster_kwargs or {})
    reduce_kwargs.pop("seed", None)

    idx = _subsample(len(vectors), sample_cap, seed)
    X = vectors if idx is None else vectors[idx]

    rng = np.random.RandomState(seed)
    shuffled = np.empty_like(X)
    for c in range(X.shape[1]):
        shuffled[:, c] = rng.permutation(X[:, c])

    u_cluster, _u3, _u2 = reduce_vectors(shuffled, seed=seed, **reduce_kwargs)
    ids, _probs = cluster_units(u_cluster, **cluster_kwargs)
    return {
        "silhouette": _silhouette(u_cluster, ids),
        "n_clusters": len({int(c) for c in ids if c >= 0}),
        "noise_fraction": round(float((ids < 0).mean()), 4),
        "n_scored": int(len(X)),
        "subsampled": idx is not None,
    }


# --- reloading the original vectors --------------------------------------
#
# `reduced.npz` caches only the UMAP outputs (u_cluster/u3/u2), never the
# source vectors — 50k x 768 float32 is ~150MB per map, which would dwarf the
# 2.6MB reduction cache and the whole point of the static artifact. So
# trustworthiness, which needs the ORIGINAL space, has to reload the front-end.
# Every front-end stamps enough of its own arguments into meta to be replayed
# exactly, which is what makes this possible at all.


def reload_units(meta: dict):
    """Re-run the front-end that built this map, from its stamped meta.

    Returns a `Units`. Raises with a specific reason when the map cannot be
    replayed — an unreadable reason is worse than no number, because a silently
    wrong reload would validate a DIFFERENT point set than the one on screen.
    """
    unit = str(meta.get("unit", ""))
    centered = bool(meta.get("centered", False))
    kept = meta.get("kept")

    if unit == "token_embedding":
        from ..frontends.tokens import load_token_units

        # `kept` is len(ids) after curation, and max_tokens truncates after
        # curation too — so passing it back reproduces the identical set
        # whether or not the original build capped the vocab.
        return load_token_units(meta["model"], center=centered, max_tokens=kept)

    if unit.startswith("mlp_neuron"):
        from ..frontends.neurons import load_neuron_units

        return load_neuron_units(
            meta["model_repo"],
            layer=int(meta["layer"]),
            max_neurons=kept,
            center=centered,
            labels_source=str(meta.get("labels_source", "none")),
        )

    if unit.startswith("sae_decoder"):
        from ..frontends.sae import load_sae_units

        return load_sae_units(
            meta["sae_release"],
            sae_id=meta["sae_id"],
            max_features=int(kept),
            center=centered,
            labels_source=str(meta.get("labels_source", "none")),
        )

    if unit.startswith("api_text_embedding"):
        raise ValueError(
            "api_text_embedding maps cannot be revalidated offline: the vectors "
            f"came from a live embedding service ({meta.get('embed_model')} @ "
            f"{meta.get('embed_host')}) and are not reproducible from meta. "
            "Re-run the build against a reachable host to validate this map."
        )

    if unit.startswith("probe_concept"):
        # Probe is non-replayable TWICE OVER, and the second reason is the one
        # that rules out ever adding a reload path here. The embeddings came
        # from a live service (same problem as api_text_embedding), but the
        # concept set itself was *sampled from an LLM* — re-running the
        # generator at temperature returns a different set of concepts, of a
        # different size, so a "reload" would hand trustworthiness a point set
        # that is not the one on screen. The length guard in `validate_map`
        # would usually catch that, but not always: it only compares counts,
        # and two different concept sets can be the same size.
        #
        # The null baseline is separately meaningless at this scale — probe
        # maps run to tens of points, and the module docstring's first caveat
        # (UMAP manufactures separable islands out of shuffled noise below a
        # few hundred points) applies with full force.
        raise ValueError(
            "probe_concept maps cannot be revalidated: the concepts were "
            f"sampled from a generator ({meta.get('generator')}) and embedded "
            f"by a live service ({meta.get('embed_model')} @ "
            f"{meta.get('embed_host')}), so replaying meta yields a DIFFERENT "
            f"point set, not this one ({kept} concepts kept of "
            f"{meta.get('n_proposed')} proposed). Probe maps are exploratory "
            "sketches of a text-embedding space, not measured artifacts — "
            "there is no offline number to compute here."
        )

    raise ValueError(f"no reload path for unit type {unit!r}")


def validate_map(
    dataset_dir,
    trust_neighbors: int = 15,
    stability_seeds: tuple[int, ...] = (42, 1, 2, 3),
    trust_sample_cap: int = _TRUST_SAMPLE_CAP,
    stability_sample_cap: int = _STABILITY_SAMPLE_CAP,
    skip_stability: bool = False,
    skip_null: bool = False,
) -> dict:
    """All three independent checks for one exported map directory.

    Trustworthiness grades the SHIPPED reduction from `reduced.npz` — it has to
    be the layout that is actually on screen, not a re-fit. Stability and the
    null baseline necessarily re-run UMAP, so they carry their own sample caps.
    """
    import json
    from pathlib import Path

    dataset_dir = Path(dataset_dir)
    meta = json.loads((dataset_dir / "nebulai.json").read_text())["meta"]

    npz = dataset_dir / "reduced.npz"
    if not npz.exists():
        raise FileNotFoundError(
            f"{npz} is missing — trustworthiness grades the shipped reduction, "
            "so the map must be rebuilt (or its cache restored) before it can "
            "be validated."
        )
    u_cluster = np.load(npz)["u_cluster"]

    units = reload_units(meta)
    vectors = units.vectors
    if len(vectors) != len(u_cluster):
        raise ValueError(
            f"reload produced {len(vectors)} units but the shipped reduction has "
            f"{len(u_cluster)} — meta does not replay this build exactly, so any "
            "score would describe a different point set."
        )

    # replay the clustering params the map was actually built with, so the
    # stability/null numbers describe THIS map's pipeline and not the defaults.
    # Maps built before `hdbscan` was stamped resolve the same n-dependent
    # defaults the build would have used, so every map is treated alike.
    from .cluster import resolve_cluster_params

    hdb = dict(meta.get("hdbscan") or {})
    cluster_kwargs = resolve_cluster_params(
        len(u_cluster),
        hdb.get("min_cluster_size"),
        hdb.get("min_samples"),
        hdb.get("method", "leaf"),
    )
    reduce_kwargs = {"cluster_dim": int(u_cluster.shape[1])}

    # subsampled runs must cluster at the full map's granularity, not its
    # absolute min_cluster_size — see scale_cluster_kwargs
    n_full = len(vectors)
    scaled = scale_cluster_kwargs(
        cluster_kwargs, n_full, min(stability_sample_cap, n_full)
    )

    out: dict = {
        "id": dataset_dir.name,
        "unit": meta.get("unit"),
        "cluster_method": cluster_kwargs.get("method", "leaf"),
        "cluster_kwargs": cluster_kwargs,
        "cluster_kwargs_scaled": scaled,
        "trustworthiness": trustworthiness_score(
            vectors, u_cluster, n_neighbors=trust_neighbors, sample_cap=trust_sample_cap
        ),
    }
    if not skip_stability:
        out["stability"] = seed_stability(
            vectors,
            seeds=stability_seeds,
            reduce_kwargs=reduce_kwargs,
            cluster_kwargs=scaled,
            sample_cap=stability_sample_cap,
        )
    if not skip_null:
        out["null_baseline"] = null_baseline(
            vectors,
            reduce_kwargs=reduce_kwargs,
            cluster_kwargs=scaled,
            sample_cap=stability_sample_cap,
        )
    return out
