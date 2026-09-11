"""Route B: orthogonal Procrustes over shared tokens (README roadmap item).

Route A's whole argument is that two models' embedding spaces have no shared
basis, so their raw coordinates must never be concatenated. Route B does not
contradict that — it tests it. If a single rigid rotation carries one cloud onto
the other, then for that pair the bases differ by exactly a rotation and the
difference between their maps is a difference of coordinates. If no rotation
does, the models really do arrange the vocabulary differently.

The danger with any alignment method is that it always returns *something*.
`AᵀB`'s SVD exists for any two matrices, so a residual on its own is not
evidence. Three things turn it into evidence and each gets a section here:
a held-out residual, a permutation null that refits, and a refusal for pairs
where the question is not defined.

The tests inject a synthetic `units_loader`, so they pin the maths without
downloading a checkpoint. The real gpt2-family numbers are produced by
`nebulai compare --route-b` and recorded in the README.
"""

import numpy as np
import pytest

from nebulai.backend.compare import (
    RouteBError,
    _orthogonal_procrustes,
    _prepare,
    _residual,
    route_b_procrustes,
)
from nebulai.units import Units

VOCAB = [f"tok{i}" for i in range(2000)]


def _units(vectors, labels=None):
    labels = labels or VOCAB[: len(vectors)]
    return Units(
        ids=list(range(len(vectors))),
        vectors=np.asarray(vectors, dtype=np.float32),
        labels=list(labels),
        meta={},
    )


def _random_rotation(d, rng):
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    return q * np.sign(np.diag(r))


def _loader(table):
    def load(model, *, center=False, max_tokens=None, remote=None):
        return table[model]

    return load


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------


def test_a_pure_rotation_is_recovered_exactly():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 24))
    R = _random_rotation(24, rng)
    A, B = _prepare(X), _prepare(X @ R)
    assert _residual(A, B, _orthogonal_procrustes(A, B)) == pytest.approx(0.0, abs=1e-12)


def test_the_fitted_map_is_orthogonal():
    rng = np.random.default_rng(1)
    A, B = _prepare(rng.normal(size=(300, 16))), _prepare(rng.normal(size=(300, 16)))
    R = _orthogonal_procrustes(A, B)
    assert np.allclose(R.T @ R, np.eye(16), atol=1e-10)


def test_scale_is_removed_before_fitting():
    """Otherwise a cloud that is merely ten times larger reports a huge
    residual for a reason that has nothing to do with its shape."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 12))
    A, B = _prepare(X), _prepare(10.0 * X)
    assert _residual(A, B, _orthogonal_procrustes(A, B)) == pytest.approx(0.0, abs=1e-12)


def test_a_translation_is_removed_before_fitting():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 12))
    A, B = _prepare(X), _prepare(X + 7.0)
    assert _residual(A, B, _orthogonal_procrustes(A, B)) == pytest.approx(0.0, abs=1e-12)


def test_unrelated_clouds_leave_almost_all_the_variance_unexplained():
    rng = np.random.default_rng(4)
    A = _prepare(rng.normal(size=(600, 32)))
    B = _prepare(rng.normal(size=(600, 32)))
    assert _residual(A, B, _orthogonal_procrustes(A, B)) > 0.8


# --------------------------------------------------------------------------
# end to end: a rotated copy, a noisy copy, and an unrelated cloud
# --------------------------------------------------------------------------


def _report(model_b_vectors, *, seed=0, n_perm=40, labels_b=None):
    rng = np.random.default_rng(99)
    X = rng.normal(size=(1200, 24))
    table = {"a": _units(X), "b": _units(model_b_vectors, labels_b)}
    return route_b_procrustes(
        "a", "b", n_permutations=n_perm, seed=seed, units_loader=_loader(table)
    )


def test_a_rotated_copy_aligns_on_held_out_tokens():
    rng = np.random.default_rng(99)
    X = rng.normal(size=(1200, 24))
    rep = _report(X @ _random_rotation(24, np.random.default_rng(5)))
    assert rep["residual_heldout"] < 1e-6
    assert rep["alignment_heldout"] == pytest.approx(1.0, abs=1e-6)
    assert rep["square_rotation"] is True
    assert rep["n_shared_tokens"] == 1200


def test_an_unrelated_cloud_does_not_align_and_is_not_significant():
    """The negative control. An alignment method that cannot fail here is
    measuring the SVD's willingness to return a matrix, not the models."""
    rep = _report(np.random.default_rng(6).normal(size=(1200, 24)))
    assert rep["residual_heldout"] > 0.8
    assert rep["p_value"] > 0.05


def test_a_noisy_rotated_copy_lands_between_the_two():
    rng = np.random.default_rng(99)
    X = rng.normal(size=(1200, 24))
    noisy = X @ _random_rotation(24, np.random.default_rng(7))
    noisy = noisy + 0.5 * np.random.default_rng(8).normal(size=noisy.shape)
    rep = _report(noisy)
    assert 0.05 < rep["residual_heldout"] < 0.5
    assert rep["p_value"] < 0.05


# --------------------------------------------------------------------------
# held-out evaluation and the null
# --------------------------------------------------------------------------


def test_the_fit_and_heldout_residuals_are_measured_on_disjoint_tokens():
    rep = _report(np.random.default_rng(10).normal(size=(1200, 24)))
    assert rep["n_fit"] + rep["n_heldout"] == rep["n_shared_tokens"]
    assert rep["n_fit"] > 0 and rep["n_heldout"] > 0


def test_the_in_sample_residual_is_reported_separately_and_is_never_larger():
    """Both numbers are printed so a reader can see the overfitting gap rather
    than being handed whichever one flattered the result."""
    rep = _report(np.random.default_rng(11).normal(size=(1200, 24)))
    assert rep["residual_full_insample"] <= rep["residual_heldout"] + 1e-9


def test_the_p_value_cannot_be_zero():
    rng = np.random.default_rng(99)
    X = rng.normal(size=(1200, 24))
    rep = _report(X @ _random_rotation(24, np.random.default_rng(12)), n_perm=40)
    floor = 1.0 / (rep["n_permutations_effective"] + 1)
    # The report rounds for display; the floor is what matters.
    assert rep["p_value"] == pytest.approx(floor, abs=1e-5)
    assert rep["p_value"] > 0


def test_the_null_is_refit_and_so_is_hard_to_beat_by_construction():
    """A null that applied the TRUE rotation to a shuffled pairing would be
    beaten by anything; this one gives the shuffled pairing its own best
    rotation, so a significant result means the correspondence carries the
    signal — not the rotation."""
    rep = _report(np.random.default_rng(13).normal(size=(1200, 24)))
    assert rep["null_residual_min"] is not None
    assert rep["null_residual_mean"] > 0.5


def test_the_report_is_reproducible_from_its_seed():
    rng = np.random.default_rng(99)
    X = rng.normal(size=(1200, 24))
    noisy = X @ _random_rotation(24, np.random.default_rng(14)) + 0.3 * rng.normal(
        size=(1200, 24)
    )
    assert _report(noisy, seed=5) == _report(noisy, seed=5)


def test_a_different_seed_changes_the_split_but_not_the_verdict():
    rng = np.random.default_rng(99)
    X = rng.normal(size=(1200, 24))
    rot = X @ _random_rotation(24, np.random.default_rng(15))
    a, b = _report(rot, seed=1), _report(rot, seed=2)
    assert a["residual_heldout"] < 1e-6
    assert b["residual_heldout"] < 1e-6


# --------------------------------------------------------------------------
# unequal widths
# --------------------------------------------------------------------------


def test_unequal_widths_use_a_semi_orthogonal_map_and_say_so():
    """gpt2 (768-d) vs gpt2-medium (1024-d). Refusing the pair outright would
    be over-strict; pretending it is a rotation would be a lie."""
    rng = np.random.default_rng(20)
    X = rng.normal(size=(1200, 24))
    wide = np.hstack([X @ _random_rotation(24, rng), rng.normal(size=(1200, 8))])
    table = {"narrow": _units(X), "wide": _units(wide)}
    rep = route_b_procrustes(
        "narrow", "wide", n_permutations=20, units_loader=_loader(table)
    )
    assert rep["square_rotation"] is False
    assert rep["dim_a"] == 24 and rep["dim_b"] == 32
    assert "projection" in rep["method"] or rep["square_rotation"] is False
    # The extra 8 dimensions are pure noise the 24-d source cannot explain, so
    # the residual is bounded away from zero — correctly.
    assert rep["residual_heldout"] > 0.05


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


def test_a_disjoint_vocabulary_is_refused_not_scored():
    rng = np.random.default_rng(30)
    table = {
        "a": _units(rng.normal(size=(1200, 24))),
        "b": _units(rng.normal(size=(1200, 24)), [f"other{i}" for i in range(1200)]),
    }
    with pytest.raises(RouteBError) as exc:
        route_b_procrustes("a", "b", units_loader=_loader(table))
    assert "share only 0 token strings" in str(exc.value)
    assert "Route A" in str(exc.value), "the refusal must name the alternative"


def test_a_small_overlap_is_refused_even_though_a_matrix_could_be_fitted():
    rng = np.random.default_rng(31)
    labels_b = VOCAB[:100] + [f"other{i}" for i in range(1100)]
    table = {
        "a": _units(rng.normal(size=(1200, 24))),
        "b": _units(rng.normal(size=(1200, 24)), labels_b),
    }
    with pytest.raises(RouteBError):
        route_b_procrustes("a", "b", units_loader=_loader(table))


def test_only_the_shared_tokens_are_aligned_when_the_overlap_is_partial():
    """A shared-but-reordered vocabulary must still line up by STRING. Pairing
    by row index would silently align unrelated tokens."""
    rng = np.random.default_rng(32)
    X = rng.normal(size=(1200, 24))
    R = _random_rotation(24, rng)
    order = rng.permutation(1200)
    table = {
        "a": _units(X),
        "b": _units((X @ R)[order], [VOCAB[i] for i in order]),
    }
    rep = route_b_procrustes("a", "b", n_permutations=20, units_loader=_loader(table))
    assert rep["n_shared_tokens"] == 1200
    assert rep["residual_heldout"] < 1e-6


def test_a_holdout_that_leaves_too_few_tokens_is_refused():
    rng = np.random.default_rng(33)
    table = {"a": _units(rng.normal(size=(1200, 24))), "b": _units(rng.normal(size=(1200, 24)))}
    with pytest.raises(RouteBError) as exc:
        route_b_procrustes("a", "b", holdout_fraction=0.01, units_loader=_loader(table))
    assert "too few" in str(exc.value)


# --------------------------------------------------------------------------
# the claim contract
# --------------------------------------------------------------------------


def test_the_report_carries_its_own_interpretation_and_refuses_to_rank():
    rep = _report(np.random.default_rng(40).normal(size=(1200, 24)), n_perm=20)
    text = rep["interpretation"].lower()
    assert "ranks neither" in text
    assert "behave" in text, "the caption must disclaim behaviour"
    for banned in ("better", "worse", "best", "leaderboard", "outperform"):
        assert banned not in text
