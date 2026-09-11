"""The estimator, the permutation test, and the A/A calibration (§6.4–§6.7).

Every number the Behavior study reports is produced here, so this file is where
the study's honesty is either enforced or lost. Four properties matter more than
the rest, and each has its own section below:

1. **Δ̂ is zero when nothing differs.** `MMD²(A,B)` alone is *not* zero for two
   samples from one distribution — it is a positive finite-sample quantity that
   grows as `n` shrinks. Subtracting the within-model split-half term is what
   removes that floor. A test that only checked "Δ̂ is big when the models
   differ" would pass for an estimator that reports a difference between a
   distribution and itself.
2. **The permutation p is calibrated.** Under the null it must be roughly
   uniform, so a nominal 5 % test rejects about 5 % of the time. The A/A harness
   below measures that rate rather than asserting it from theory.
3. **The p-floor is real.** With few trials per block there are only so many
   distinct within-block label assignments, and if `1/N` sits above the
   BY-adjusted threshold nothing can be confirmed *at any effect size*. That is
   a design fact the study must print before spending money, not a surprise
   found afterwards.
4. **Missingness is bracketed, not imputed.** Manski bounds that span the effect
   floor block confirmation regardless of the point estimate.

The A/A harness uses modest permutation counts so the file stays runnable in
seconds; its assertions are correspondingly loose bands, chosen to catch a
broken estimator rather than to certify a precise rejection rate.
"""

import math

import numpy as np
import pytest

from nebulai.behavior.stats import (
    benjamini_yekutieli,
    bootstrap_ci,
    compliance_parity,
    decompose,
    delta_hat,
    entropy,
    jsd,
    manski_bounds,
    median_bandwidth,
    miller_madow_entropy,
    mmd2,
    p_floor,
    permutation_p,
    rbo,
    split_half_reliability,
    type_token_ratio,
)


def _sphere(rng, n, d=16, shift=0.0, scale=1.0):
    """n unit vectors, optionally shifted then re-normalized (the study's
    vectors are always L2-normalized, so the tests must be too)."""
    v = rng.normal(size=(n, d)) * scale
    v[:, 0] += shift
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _blocks(n, k):
    return np.arange(n) % k


# --------------------------------------------------------------------------
# 1. the estimator
# --------------------------------------------------------------------------


def test_the_unbiased_mmd2_of_a_sample_against_itself_is_negative():
    """Not a bug, and worth a test so nobody "fixes" it.

    The unbiased estimator drops the diagonal from the two within-sample terms
    but not from the cross term. Fed the same array twice, the cross term
    therefore includes n self-similarities of exactly 1 that the within terms
    excluded, and the estimate lands below zero. It is the price of an estimator
    with no positive floor — which is the property Δ̂ needs, because a floor
    would make every A/A control look like a difference.
    """
    rng = np.random.default_rng(0)
    X = _sphere(rng, 40)
    assert mmd2(X, X, median_bandwidth(X)) < 0


def test_mmd2_of_two_disjoint_halves_of_one_sample_is_near_zero():
    """The honest version of "distance from itself": split, don't repeat."""
    rng = np.random.default_rng(0)
    X = _sphere(rng, 120)
    v = mmd2(X[:60], X[60:], median_bandwidth(X))
    assert abs(v) < 0.02


def test_raw_mmd2_between_two_draws_from_one_distribution_is_not_zero():
    """The reason Δ̂ subtracts a within-model term at all."""
    rng = np.random.default_rng(1)
    A, B = _sphere(rng, 30), _sphere(rng, 30)
    assert mmd2(A, B, median_bandwidth(A, B)) != pytest.approx(0.0, abs=1e-6)


def test_delta_hat_is_near_zero_for_two_samples_from_one_distribution():
    rng = np.random.default_rng(2)
    vals = []
    for _ in range(12):
        A, B = _sphere(rng, 60), _sphere(rng, 60)
        vals.append(delta_hat(A, B, median_bandwidth(A, B), draws=25, rng=rng).delta_hat)
    assert abs(float(np.mean(vals))) < 0.01, f"A/A Δ̂ centred at {np.mean(vals):.4f}"


def test_delta_hat_rises_with_a_real_location_difference():
    """Monotone in the true separation — and note that the small-shift value is
    allowed to be slightly negative. Δ̂ is a difference of two noisy estimates,
    so at an effect below the noise it scatters around zero on both sides; a
    version clamped at zero would report a one-sided quantity and every
    near-null cue would look like a small positive effect."""
    rng = np.random.default_rng(3)
    A = _sphere(rng, 60)
    bw = median_bandwidth(A)
    near = delta_hat(A, _sphere(rng, 60, shift=0.3), bw, rng=rng).delta_hat
    mid = delta_hat(A, _sphere(rng, 60, shift=1.5), bw, rng=rng).delta_hat
    far = delta_hat(A, _sphere(rng, 60, shift=4.0), bw, rng=rng).delta_hat
    assert far > mid > near
    assert abs(near) < 0.01
    assert far > 0.1


def test_delta_hat_reports_its_parts():
    rng = np.random.default_rng(4)
    A, B = _sphere(rng, 40), _sphere(rng, 40, shift=1.0)
    d = delta_hat(A, B, median_bandwidth(A, B), rng=rng)
    assert d.delta_hat == pytest.approx(d.mmd2_between - 0.5 * (d.within_a + d.within_b))
    assert (d.n_a, d.n_b) == (40, 40)
    assert d.form == "difference"


def test_the_standardized_form_divides_instead_of_subtracting():
    rng = np.random.default_rng(5)
    A, B = _sphere(rng, 40), _sphere(rng, 40, shift=1.0)
    bw = median_bandwidth(A, B)
    d = delta_hat(A, B, bw, rng=np.random.default_rng(9), form="standardized")
    assert d.form == "standardized"
    assert d.delta_hat == pytest.approx(d.mmd2_between / (0.5 * (d.within_a + d.within_b)))


def test_the_bandwidth_follows_the_median_heuristic_in_its_sigma_convention():
    """Pairwise distances 3, 4, 5 → median squared distance 16.

    The kernel is written `exp(−d²/2σ²)`, so the heuristic's σ is
    `sqrt(median(d²)/2)`, not `median(d)`. Pinning the convention matters
    because `mmd_bandwidth` is a frozen manifest field: a later reader who
    assumed the other convention would reproduce the study at a kernel width
    off by √2 and get different effect sizes from the same data.
    """
    X = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    assert median_bandwidth(X) == pytest.approx(math.sqrt(16 / 2))


# --------------------------------------------------------------------------
# 2. the permutation test, and the A/A calibration harness
# --------------------------------------------------------------------------


def test_the_p_value_can_never_be_reported_as_zero():
    """(r+1)/(B+1): a permutation test does not license p = 0."""
    rng = np.random.default_rng(6)
    A, B = _sphere(rng, 40), _sphere(rng, 40, shift=4.0)
    p, obs, n_eff = permutation_p(
        A, B, _blocks(40, 4), _blocks(40, 4), median_bandwidth(A, B),
        n_permutations=40, draws=6, seed=1,
    )
    assert p > 0
    assert p == pytest.approx(1.0 / (n_eff + 1))
    assert obs > 0


def test_a_real_difference_produces_a_small_p():
    rng = np.random.default_rng(7)
    A, B = _sphere(rng, 48), _sphere(rng, 48, shift=2.0)
    p, _, _ = permutation_p(
        A, B, _blocks(48, 4), _blocks(48, 4), median_bandwidth(A, B),
        n_permutations=60, draws=6, seed=2,
    )
    assert p < 0.05


@pytest.mark.slow
def test_the_aa_calibration_rejects_at_about_the_nominal_rate():
    """The A/A arm of §6.7.1, run for real at a small scale.

    Both "models" are the same distribution, so every rejection is a false one.
    The band is wide because 40 replicates at 60 permutations cannot pin a 5 %
    rate tightly — it is set to catch an estimator that is *systematically*
    anticonservative (a broken within-model correction shows up here at 30-60 %,
    not at 12 %).
    """
    rng = np.random.default_rng(1234)
    n_rep, alpha = 40, 0.05
    ps = []
    for r in range(n_rep):
        A, B = _sphere(rng, 40), _sphere(rng, 40)
        p, _, _ = permutation_p(
            A, B, _blocks(40, 4), _blocks(40, 4), median_bandwidth(A, B),
            n_permutations=60, draws=5, seed=r,
        )
        if not math.isnan(p):
            ps.append(p)
    rate = float(np.mean([p <= alpha for p in ps]))
    assert len(ps) == n_rep
    assert rate <= 0.20, f"A/A rejection rate {rate:.3f} at nominal {alpha}"
    assert float(np.mean(ps)) > 0.25, "p-values should not pile up near zero under the null"


def test_shuffling_labels_destroys_a_real_effect():
    """The shuffled-label control: the same data, model labels randomized."""
    rng = np.random.default_rng(8)
    A, B = _sphere(rng, 48), _sphere(rng, 48, shift=2.0)
    pooled = np.vstack([A, B])
    idx = rng.permutation(len(pooled))
    SA, SB = pooled[idx[:48]], pooled[idx[48:]]
    bw = median_bandwidth(pooled)
    real = delta_hat(A, B, bw, rng=np.random.default_rng(0)).delta_hat
    shuffled = delta_hat(SA, SB, bw, rng=np.random.default_rng(0)).delta_hat
    assert real > 0
    assert shuffled < real / 4


# --------------------------------------------------------------------------
# 3. the p-floor
# --------------------------------------------------------------------------


def test_the_p_floor_is_the_reciprocal_of_the_assignment_count():
    # 2 blocks, 3 of each model per block: C(6,3) = 20 per block, 400 total.
    assert p_floor([3, 3], [3, 3]) == pytest.approx(1.0 / 400)


def test_a_single_tiny_block_cannot_reach_any_useful_alpha():
    """One block of 2+2 gives C(4,2)=6 assignments, so p >= 1/6 = 0.167.

    No effect size can rescue this: the study must either add trials per block
    or stop claiming confirmation, and it has to know that before it spends.
    """
    floor = p_floor([2], [2])
    assert floor == pytest.approx(1 / 6)
    assert floor > 0.05


def test_the_floor_falls_as_trials_per_block_rise():
    assert p_floor([10] * 4, [10] * 4) < p_floor([4] * 4, [4] * 4) < p_floor([2] * 4, [2] * 4)


def test_a_huge_design_saturates_instead_of_overflowing():
    assert p_floor([60] * 8, [60] * 8) == pytest.approx(1e-18)


def test_the_floor_is_compared_against_the_by_threshold_not_alpha():
    """§6.7.1's actual check, spelled out: with m cues the BY penalty c(m)
    raises every adjusted q, so a floor that looks fine against 0.05 can still
    make confirmation impossible once multiplicity is paid for."""
    m, q = 300, 0.05
    cm = sum(1.0 / i for i in range(1, m + 1))
    smallest_useful = q / (m * cm)  # what the most significant cue must beat
    assert smallest_useful == pytest.approx(2.65e-5, rel=0.05)
    # 2 blocks of 3+3: 400 assignments, floor 1/400 — fine against a bare 0.05,
    # and 94x too coarse to confirm anything once 300 cues are paid for.
    coarse = p_floor([3, 3], [3, 3])
    assert coarse < q
    assert coarse > smallest_useful
    # 6 blocks of 12+12 clears it with room to spare.
    assert p_floor([12] * 6, [12] * 6) < smallest_useful


# --------------------------------------------------------------------------
# multiple testing
# --------------------------------------------------------------------------


def test_by_is_stricter_than_the_raw_p_and_never_exceeds_one():
    adj, rej = benjamini_yekutieli([0.001, 0.02, 0.3, 0.8], q=0.05)
    assert all(a <= 1.0 for a in adj)
    assert adj[0] > 0.001
    assert rej[0] is True
    assert rej[-1] is False


def test_by_adjusted_values_are_monotone_in_the_raw_values():
    raw = [0.0001, 0.003, 0.01, 0.04, 0.2, 0.9]
    adj, _ = benjamini_yekutieli(raw, q=0.05)
    assert adj == sorted(adj)


def test_by_pays_the_harmonic_penalty():
    """The c(m) factor is the whole difference from BH; losing it would make
    every reported q too small by ~2.9x at m = 300."""
    adj, _ = benjamini_yekutieli([0.001, 0.001, 0.001, 0.001], q=0.05)
    cm = sum(1.0 / i for i in range(1, 5))
    assert adj[0] == pytest.approx(min(1.0, 0.001 * 4 * cm / 4))


def test_an_empty_family_is_not_an_error():
    assert benjamini_yekutieli([], q=0.05) == ([], [])


# --------------------------------------------------------------------------
# bootstrap and decomposition
# --------------------------------------------------------------------------


def test_the_bootstrap_interval_brackets_the_point_estimate():
    """Without the duplicate-induced bias correction this fails: the replicate
    distribution sits entirely above the observed Δ̂."""
    rng = np.random.default_rng(11)
    A, B = _sphere(rng, 60), _sphere(rng, 60, shift=1.5)
    bw = median_bandwidth(A, B)
    point = delta_hat(A, B, bw, draws=5, rng=np.random.default_rng(3)).delta_hat
    lo, hi = bootstrap_ci(
        A, B, _blocks(60, 4), _blocks(60, 4), bw, n_bootstrap=120, draws=5, seed=3
    )
    assert lo < point < hi


def test_an_aa_interval_covers_zero():
    rng = np.random.default_rng(12)
    A, B = _sphere(rng, 60), _sphere(rng, 60)
    lo, hi = bootstrap_ci(
        A, B, _blocks(60, 4), _blocks(60, 4), median_bandwidth(A, B),
        n_bootstrap=120, draws=5, seed=4,
    )
    assert lo <= 0 <= hi


def _blob(rng, n, centre, tightness=0.25, d=16):
    """A concentrated cloud around a unit direction, re-normalized.

    Note that `_sphere(..., scale=s)` is NOT this: scaling Gaussian draws by a
    constant and then normalizing gives a uniform direction, so `scale` alone
    cannot make a tight cluster. Getting that wrong turns an intended
    location-only contrast into a location-AND-dispersion one.
    """
    c = np.zeros(d)
    c[: len(centre)] = centre
    c = c / np.linalg.norm(c)
    v = c + tightness * rng.normal(size=(n, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_a_shifted_pair_is_called_location_driven():
    rng = np.random.default_rng(13)
    a = _blob(rng, 80, [1.0, 0.0])
    b = _blob(rng, 80, [0.0, 1.0])
    d = decompose(a, b)
    assert d.dispersion_a == pytest.approx(d.dispersion_b, rel=0.15)
    assert d.dominant == "location"


def test_a_pair_that_differs_only_in_spread_is_called_dispersion_driven():
    """Two clouds on the same centre, one much tighter. Calling this an
    association difference would be the single most misleading thing the study
    could print."""
    rng = np.random.default_rng(14)
    tight = _blob(rng, 80, [1.0, 0.0], tightness=0.05)
    wide = _blob(rng, 80, [1.0, 0.0], tightness=0.9)
    d = decompose(tight, wide)
    assert d.dispersion_b > d.dispersion_a
    assert d.dominant == "dispersion"


# --------------------------------------------------------------------------
# distribution statistics
# --------------------------------------------------------------------------


def test_entropy_of_a_point_mass_is_zero_and_of_a_uniform_is_log_k():
    assert entropy({"a": 10}) == pytest.approx(0.0)
    assert entropy([5, 5, 5, 5]) == pytest.approx(math.log(4))


def test_miller_madow_corrects_upward_by_the_expected_amount():
    counts = np.array([5.0, 5.0, 5.0, 5.0])
    plug = entropy(list(counts.astype(int)))
    assert miller_madow_entropy(counts) == pytest.approx(plug + 3 / (2 * 20))


def test_jsd_of_identical_tables_is_zero_and_reports_both_n():
    a = {"butter": 10, "toast": 6, "flour": 4}
    val, na, nb = jsd(a, dict(a))
    assert val == pytest.approx(0.0, abs=1e-12)
    assert (na, nb) == (20, 20)


def test_jsd_of_disjoint_tables_approaches_log_two():
    val, _, _ = jsd({"a": 200, "b": 200}, {"c": 200, "d": 200}, correction="none")
    assert val == pytest.approx(math.log(2), abs=1e-6)


def test_the_bias_correction_changes_small_sample_jsd():
    """The whole point: with few trials per cue the plug-in estimator reports a
    divergence that is mostly type-count artifact."""
    a, b = {"x": 3, "y": 2}, {"x": 2, "z": 3}
    plug, _, _ = jsd(a, b, correction="none")
    corrected, _, _ = jsd(a, b, correction="miller-madow")
    assert corrected != pytest.approx(plug)


def test_an_empty_side_is_nan_not_zero():
    val, na, nb = jsd({}, {"a": 3})
    assert math.isnan(val)
    assert (na, nb) == (0, 3)


def test_rbo_is_one_for_identical_rankings_and_zero_for_disjoint_ones():
    """With the truncated sum this would be 0.271 for identical three-item
    lists — the number a reader would take as "they barely agreed"."""
    assert rbo(["a", "b", "c"], ["a", "b", "c"], 0.9) == pytest.approx(1.0, abs=1e-9)
    assert rbo(["a", "b", "c"], ["x", "y", "z"], 0.9) == pytest.approx(0.0, abs=1e-9)


def test_rbo_orders_a_partial_overlap_between_the_two_extremes():
    same_head = rbo(["a", "b", "c"], ["a", "y", "z"], 0.9)
    same_tail = rbo(["a", "b", "c"], ["x", "y", "c"], 0.9)
    assert 0.0 < same_tail < same_head < 1.0, "top-weighting must be visible"


def test_rbo_handles_lists_of_unequal_length():
    v = rbo(["a", "b"], ["a", "b", "c", "d"], 0.9)
    assert 0.0 < v <= 1.0
    assert not math.isnan(v)


def test_rbo_weights_the_top_of_the_list_more_at_smaller_p():
    """p is a manifest field precisely because it moves the answer this much."""
    a, b = ["a", "b", "c", "d"], ["a", "x", "y", "z"]
    assert rbo(a, b, 0.5) > rbo(a, b, 0.98)


def test_type_token_ratio():
    assert type_token_ratio(["a", "b", "a", "c"]) == pytest.approx(0.75)
    assert type_token_ratio([]) == 0.0


# --------------------------------------------------------------------------
# 4. compliance parity and Manski bounds
# --------------------------------------------------------------------------


def test_compliance_parity_is_a_gate_not_a_warning():
    assert compliance_parity(0.95, 0.92, 0.05) is True
    assert compliance_parity(0.95, 0.60, 0.05) is False


def test_with_nothing_missing_the_bounds_collapse_to_the_point_estimate():
    rng = np.random.default_rng(15)
    A, B = _sphere(rng, 40), _sphere(rng, 40, shift=1.0)
    mb = manski_bounds(
        A, B, median_bandwidth(A, B), missing_a=0, missing_b=0, effect_floor=0.02, seed=1
    )
    assert mb.lo == mb.hi
    assert mb.spans_floor is False


def test_heavy_missingness_widens_the_bounds():
    rng = np.random.default_rng(16)
    A, B = _sphere(rng, 40), _sphere(rng, 40, shift=1.0)
    bw = median_bandwidth(A, B)
    light = manski_bounds(A, B, bw, missing_a=2, missing_b=2, effect_floor=0.02, seed=2)
    heavy = manski_bounds(A, B, bw, missing_a=30, missing_b=30, effect_floor=0.02, seed=2)
    assert (heavy.hi - heavy.lo) > (light.hi - light.lo)


def test_bounds_that_span_the_floor_are_flagged():
    """The flag is what blocks confirmation; the point estimate is irrelevant
    once the worst case reaches the floor."""
    rng = np.random.default_rng(17)
    A, B = _sphere(rng, 40), _sphere(rng, 40, shift=1.2)
    bw = median_bandwidth(A, B)
    mb = manski_bounds(A, B, bw, missing_a=30, missing_b=2, effect_floor=0.01, seed=3)
    assert mb.lo <= 0.01 <= mb.hi
    assert mb.spans_floor is True
    # The same data with everything observed clears the same floor: the flag is
    # about the missingness, not about the effect.
    full = manski_bounds(A, B, bw, missing_a=0, missing_b=0, effect_floor=0.01, seed=3)
    assert full.lo > 0.01
    assert full.spans_floor is False


# --------------------------------------------------------------------------
# reliability, and why it is never read alone
# --------------------------------------------------------------------------


def test_split_half_reliability_is_high_for_a_concentrated_sample():
    rng = np.random.default_rng(18)
    X = _sphere(rng, 60, scale=0.15) + np.array([1.0] + [0.0] * 15)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    assert split_half_reliability(X, seed=1) > 0.95


def test_a_degenerate_constant_responder_scores_a_perfect_one():
    """Gate 2 alone would pass a model that echoes the cue every single time.
    This test exists to pin that fact in place, not to celebrate it: the
    informativeness floor of §6.5.2 is what actually catches this arm."""
    X = np.tile(_sphere(np.random.default_rng(19), 1), (40, 1))
    assert split_half_reliability(X, seed=1) == pytest.approx(1.0, abs=1e-9)


def test_too_few_trials_is_nan_rather_than_a_confident_number():
    assert math.isnan(split_half_reliability(_sphere(np.random.default_rng(20), 3)))
