"""UMAP reductions: a mid-dimensional space for clustering, 3D for the
flythrough, and a 2D view derived from the 3D one so both views stay aligned
(independent 2D/3D fits would scatter points differently)."""

import numpy as np


def reduce_vectors(
    vectors: np.ndarray,
    cluster_dim: int = 10,
    n_neighbors: int = 30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import umap
    from sklearn.decomposition import PCA

    common: dict = {"n_neighbors": n_neighbors, "metric": "cosine"}
    if seed >= 0:
        # deterministic, but forces single-threaded layout; pass -1 for speed
        common["random_state"] = seed

    # HDBSCAN runs on this — min_dist=0 packs clusters densely, and staying
    # above 2-3 dims avoids clustering artifacts the projection invented
    u_cluster = umap.UMAP(
        n_components=cluster_dim, min_dist=0.0, **common
    ).fit_transform(vectors)

    u3 = umap.UMAP(n_components=3, min_dist=0.1, **common).fit_transform(vectors)
    u2 = PCA(n_components=2, random_state=0).fit_transform(u3)

    return (
        np.asarray(u_cluster, dtype=np.float32),
        np.asarray(u3, dtype=np.float32),
        np.asarray(u2, dtype=np.float32),
    )
