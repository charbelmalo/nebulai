import numpy as np


def resolve_cluster_params(
    n: int,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    method: str = "leaf",
) -> dict:
    """The clustering params HDBSCAN will actually see, with defaults filled in.

    Kept as one function so `cluster_units` and any caller that needs to *record*
    the params (export meta, the metrics table) resolve them identically — the
    `None -> max(15, n // 1000)` default must never drift between the run and
    what gets stamped into `nebulai.json`."""
    return {
        "method": method,
        "min_cluster_size": max(15, n // 1000)
        if min_cluster_size is None
        else int(min_cluster_size),
        "min_samples": 5 if min_samples is None else int(min_samples),
    }


def cluster_units(
    u_cluster: np.ndarray,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    method: str = "leaf",
) -> tuple[np.ndarray, np.ndarray]:
    """HDBSCAN over the clustering-space embedding.

    Returns (cluster_ids, probabilities). cluster_id -1 is noise; the
    membership probability doubles as the point-confidence signal
    (mapped to opacity in the viz).

    Default is leaf selection: on token-embedding UMAP spaces, eom
    collapses ~everything into one mega-cluster (the vocab core is one
    connected density blob), while leaf recovers the fine concept
    groups at the cost of a higher noise fraction.
    """
    from sklearn.cluster import HDBSCAN

    n = len(u_cluster)
    p = resolve_cluster_params(n, min_cluster_size, min_samples, method)
    min_cluster_size = p["min_cluster_size"]
    min_samples = p["min_samples"]

    h = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=method,
    )
    ids = h.fit_predict(u_cluster)
    probs = getattr(h, "probabilities_", np.ones(n))
    return np.asarray(ids, dtype=int), np.asarray(probs, dtype=np.float32)
