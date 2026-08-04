"""Guardrails on the smallest clouds — the probe front-end makes them routinely."""

import numpy as np
import pytest

from nebulai.backend.reduce import reduce_vectors


def _vecs(n, d=32):
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, d)).astype(np.float32)


def test_a_cloud_too_small_to_reduce_says_what_to_change():
    with pytest.raises(ValueError, match="only 8 points to reduce"):
        reduce_vectors(_vecs(8), cluster_dim=10, n_neighbors=10)


def test_the_error_names_the_probe_knobs_that_fix_it():
    with pytest.raises(ValueError, match="--breadth"):
        reduce_vectors(_vecs(5), cluster_dim=10)


def test_n_neighbors_larger_than_the_cloud_is_clamped_not_fatal():
    # 20 points, 30 neighbours asked for: clamp and render rather than fail
    c, u3, u2 = reduce_vectors(_vecs(20), cluster_dim=5, n_neighbors=30, seed=1)
    assert c.shape == (20, 5) and u3.shape == (20, 3) and u2.shape == (20, 2)
    assert np.isfinite(u3).all()
