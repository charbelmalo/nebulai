"""Per-cue evidence assembly and the §6.5 gates that decide what may be claimed.

BEHAVIORAL-DIVERGENCE-PLAN.md §6.5. A cue's status is a conjunction of named
checks, not a score, and the reason this file is long is the same reason
`analyze.py` is one long conditional: every way a cue can fail to be evidence
has to be distinguishable from every other way, and from success.

The four properties under test, each of which has a cheap wrong version that
would look identical on a dashboard:

  * **Assembly.** A `CueResult` carries both arms' profiles, the effect, its
    uncertainty, and the reasons — so a reader can see what produced the number.
  * **BY across cues.** 180 cues at α = 0.05 produce ~9 false positives by
    construction. The family-wise correction is applied across the whole cue
    family, and a cue that does not survive it is demoted with the reason
    recorded, not quietly dropped from the display.
  * **Compliance parity (§6.5.1).** When one arm parses far less often than the
    other, the survivors are a non-random subsample of its behaviour. The pair
    is `incomparable` — a distinct state from "no difference found".
  * **Informativeness (§6.5.2).** A model that echoes the cue every time scores
    a perfect split-half reliability with zero semantic content, so the floor
    and the echo detectors are what stop degeneracy from reading as a finding.

And the rule that cuts across all of them: a metric that could not be computed
is `None` with a reason, never `0`. A zero is a measurement.
"""

import math

import numpy as np
import pytest

from nebulai.behavior import analyze as A
from nebulai.behavior import stats as S
from nebulai.behavior.contract import CANARY_CUE, Cue, Manifest, ModelRef, TrialRecord
from nebulai.behavior.embed import HashEmbedder
from nebulai.behavior.normalize import detect, parse_associates
from nebulai.behavior.protocol import EXEMPLAR_ANSWERS, PRIMARY, default_frames

BANDWIDTH = 1.0


def _manifest(**kw) -> Manifest:
    base = dict(
        study_id="t_analyze",
        created="2026-09-11T00:00:00Z",
        models=[ModelRef("A", "fake", "fake"), ModelRef("B", "fake", "fake")],
        cues=[Cue("hot", "control_neutral")],
        frames=default_frames(),
        # Small on purpose: the permutation and bootstrap counts are what make
        # this file fast, and neither is what is under test here (they have
        # their own coverage in tests/test_behavior_stats.py).
        min_valid_trials=4,
        split_half_draws=4,
        n_permutations=40,
        n_bootstrap=20,
        reliability_floor=0.0,
        effect_floor=0.0,
        seed=7,
    )
    base.update(kw)
    m = Manifest(**base)  # type: ignore[arg-type]
    m.freeze("2026-09-11T00:00:00Z")
    return m


def _trial(cue: str, model_key: str, raw: str, repeat: int, *, blocks: int = 2) -> TrialRecord:
    """One stored trial, built exactly the way `runner._one` builds it.

    Going through the real parser and the real detectors matters: a fixture that
    hand-set `valid` and the marks would test the gates against a shape the
    runner never produces.
    """
    prompt = PRIMARY.render(cue)
    parsed = parse_associates(raw)
    marks = detect(parsed, cue=cue, prompt=prompt, exemplar_answers=EXEMPLAR_ANSWERS)
    return TrialRecord(
        study_id="t_analyze",
        arm="discovery",
        cue=cue,
        frame_id=PRIMARY.id,
        model_key=model_key,
        repeat=repeat,
        block=repeat % blocks,
        prompt=prompt,
        prompt_sha="sha256:test",
        raw_output=raw,
        associates=parsed.associates,
        valid=parsed.valid,
        invalid_reason=parsed.reason,
        usage={
            "marks": {
                "cue_echo": marks.cue_echo,
                "exemplar_echo": marks.exemplar_echo,
                "within_trial_duplicate": marks.within_trial_duplicate,
                "prompt_copy": marks.prompt_copy,
                "distinct_types": marks.distinct_types,
            }
        },
    )


REFUSAL = "I'm sorry, I can't help with that."

_POOL_A = [
    "cold, warm, fire", "warm, sun, heat", "cold, heat, sun",
    "fire, warm, cold", "heat, sun, warm", "sun, cold, fire",
]
_POOL_B = [
    "spicy, pepper, chilli", "pepper, curry, spicy", "chilli, curry, pepper",
    "spicy, chilli, curry", "curry, pepper, chilli", "pepper, spicy, curry",
]


def _rows(cue: str, model_key: str, texts: list[str], *, n: int = 12, invalid: int = 0):
    """`n` trials cycling through `texts`, with the last `invalid` unparseable."""
    out = []
    for i in range(n):
        raw = REFUSAL if i >= n - invalid else texts[i % len(texts)]
        out.append(_trial(cue, model_key, raw, i))
    return out


def _profiles(rows, m=None):
    return A.build_profiles(rows, m or _manifest(), HashEmbedder())


# --------------------------------------------------------------------------
# profile construction
# --------------------------------------------------------------------------


def test_the_refusal_fixture_really_is_unparseable():
    """The compliance tests below are meaningless if this parses.

    Pinned as its own assertion so a parser change breaks here, with a clear
    message, rather than silently turning every parity test into a no-op.
    """
    assert parse_associates(REFUSAL).valid is False


def test_profiles_are_keyed_by_cue_then_model():
    rows = _rows("hot", "A", _POOL_A) + _rows("hot", "B", _POOL_B) + _rows("salt", "A", _POOL_A)
    prof = _profiles(rows)
    assert set(prof) == {"hot", "salt"}
    assert set(prof["hot"]) == {"A", "B"}
    assert set(prof["salt"]) == {"A"}


def test_canary_trials_are_dropped_before_any_metric_sees_them():
    """§5.5.1: excluded from every semantic metric. Dropped in exactly one
    place, so no downstream metric can re-admit them by accident."""
    rows = _rows("hot", "A", _POOL_A) + [
        _trial(CANARY_CUE, "A", "glass, door, view", i) for i in range(4)
    ]
    assert CANARY_CUE not in _profiles(rows)


def test_attempted_counts_every_trial_and_valid_counts_only_the_parsed_ones():
    """`n_attempted` is the denominator of the parity gate, so an unparseable
    trial must stay in it — dropping the row would make a refusing model look
    perfectly compliant."""
    p = _profiles(_rows("hot", "A", _POOL_A, n=10, invalid=4))["hot"]["A"]
    assert p.n_attempted == 10
    assert p.n_valid == 6
    assert p.parse_rate == pytest.approx(0.6)


def test_a_profile_carries_one_vector_and_one_block_label_per_valid_trial():
    """The permutation test indexes vectors by block; a length mismatch would
    silently misalign every trial with someone else's collection time."""
    p = _profiles(_rows("hot", "A", _POOL_A, n=8))["hot"]["A"]
    assert p.vectors is not None and p.blocks is not None
    assert len(p.vectors) == len(p.blocks) == p.n_valid
    assert p.vectors.shape[1] == HashEmbedder().dim


def test_type_counts_and_entropy_come_from_the_pooled_associates():
    p = _profiles(_rows("hot", "A", _POOL_A, n=6))["hot"]["A"]
    assert set(p.counts) == {"cold", "warm", "fire", "sun", "heat"}
    assert p.distinct_types == 5
    assert p.entropy == pytest.approx(S.entropy(p.counts))
    assert p.top_ranked[:1] != [], "the ranked list feeds RBO and must not be empty"


def test_an_arm_with_no_valid_trials_has_no_vectors_and_no_reliability():
    p = _profiles(_rows("hot", "A", _POOL_A, n=6, invalid=6))["hot"]["A"]
    assert p.n_valid == 0
    assert p.vectors is None
    assert p.reliability is None, "unmeasurable reliability is None, never 0.0"
    assert p.counts == {}


def test_too_few_trials_to_halve_leaves_reliability_missing_not_nan():
    """SOURCE FIX (analyze._profile): `split_half_reliability` returns NaN below
    four trials, and the raw NaN was being stored. That silently passed gate 2
    (`nan < floor` is False) and reached the exporter, where `json.dumps` writes
    the bare token `NaN` — invalid JSON, and not the `null` §8.5 requires. It is
    now routed through `_f`, so an uncomputable reliability is `None` and the
    gate fires.
    """
    p = _profiles(_rows("hot", "A", _POOL_A, n=3))["hot"]["A"]
    assert p.n_valid == 3
    assert math.isnan(S.split_half_reliability(p.vectors, draws=4, seed=7)), (
        "the underlying statistic really is NaN here; the profile is what must not be"
    )
    assert p.reliability is None


def test_an_uncomputable_reliability_fails_gate_2_instead_of_passing_it_silently():
    """The second half of the same fix: `nan < floor` is False in Python, so an
    un-sanitized NaN sailed through the reliability gate as if it had passed."""
    r = _compare(
        a_rows=_rows("hot", "A", _POOL_A, n=12), b_rows=_rows("hot", "B", _POOL_B, n=3)
    )
    assert r.arms["B"].reliability is None
    assert any(x.startswith("gate 2") and "B" in x for x in r.reasons)


def test_the_detector_rates_are_per_trial_shares_of_every_attempt():
    rows = _rows("hot", "A", ["hot, hot, hot"], n=4) + _rows("hot", "A", _POOL_A, n=4)
    # both halves are the same model at the same cue, so they pool into one arm
    for i, r in enumerate(rows):
        r.repeat = i
    p = _profiles(rows)["hot"]["A"]
    assert p.cue_echo_rate == pytest.approx(0.5)
    assert p.duplicate_rate == pytest.approx(0.5)
    assert p.exemplar_echo_rate == 0.0


def test_the_exemplar_echo_detector_catches_a_model_copying_the_few_shot_answer():
    """§6.5.2: copying `butter, toast, flour` is maximally reliable and carries
    no association at all."""
    p = _profiles(_rows("hot", "A", [", ".join(EXEMPLAR_ANSWERS)], n=6))["hot"]["A"]
    assert p.exemplar_echo_rate == 1.0


def test_the_fragmentation_proxy_is_a_share_of_associates_and_is_labelled_a_proxy():
    plain = _profiles(_rows("hot", "A", _POOL_A, n=6))["hot"]["A"]
    odd = _profiles(_rows("hot", "A", ["c0ld, w4rm, f1re"], n=6))["hot"]["A"]
    assert plain.oov_fragment_ratio == 0.0
    assert odd.oov_fragment_ratio == 1.0


# --------------------------------------------------------------------------
# cue-level assembly
# --------------------------------------------------------------------------


def _compare(m=None, *, a_rows=None, b_rows=None, **kw) -> A.CueResult:
    m = m or _manifest()
    rows = (a_rows if a_rows is not None else _rows("hot", "A", _POOL_A)) + (
        b_rows if b_rows is not None else _rows("hot", "B", _POOL_B)
    )
    prof = _profiles(rows, m)["hot"]
    return A.compare_cue(
        "hot", "control_neutral", prof["A"], prof["B"], m,
        bandwidth=BANDWIDTH, permutations=40, **kw,
    )


def test_a_cue_result_carries_both_arms_and_every_reported_quantity():
    r = _compare()
    assert set(r.arms) == {"A", "B"}
    assert set(r.within) == {"A", "B"} and set(r.dispersion) == {"A", "B"}
    assert set(r.jsd_n) == {"A", "B"}
    for field in ("delta_hat", "mmd2_between", "p_value", "ci_lo", "ci_hi", "jsd", "rbo"):
        assert getattr(r, field) is not None, f"{field} is computable here and must be reported"
    assert 0.0 <= r.p_value <= 1.0
    assert r.ci_lo <= r.ci_hi


def test_two_arms_with_disjoint_associates_produce_a_positive_effect_and_zero_rbo():
    """The sanity floor of the whole file: if this does not separate, nothing
    below is testing what it says it is."""
    r = _compare()
    assert r.delta_hat > 0
    assert r.rbo == 0.0  # no shared terms at any rank
    assert r.jsd > 0.5  # disjoint support: the JS divergence is near its max
    assert r.jsd_n == {"A": 36, "B": 36}  # 12 trials × 3 associates each


def test_the_partition_label_travels_with_the_result():
    """§6.5.3 turns on this label: a displayed magnitude must be from the
    confirmation partition, and the artifact has to be able to say which one a
    number came from."""
    assert _compare().partition == "discovery"
    assert _compare(partition="confirmation").partition == "confirmation"


def test_the_pack_label_does_not_change_the_threshold_a_cue_is_held_to():
    """§5.3: matched controls are grouped, not privileged."""
    plain = _compare()
    packed = _compare(pack="daddy")
    assert packed.pack == "daddy" and plain.pack == ""
    assert packed.delta_hat == pytest.approx(plain.delta_hat)
    assert packed.status == plain.status


def test_the_decomposition_names_which_component_dominates():
    """§6.4: "the two models disagree" and "one model is more scattered than the
    other" are different findings, and the artifact must not merge them."""
    r = _compare()
    assert r.dominant in ("location", "dispersion")
    assert r.location is not None


def test_nothing_in_a_cue_result_names_a_winner():
    """§1.2 / the package docstring: `Δ̂` is symmetric and the artifact has no
    ranking field. A sign convention here would be a leaderboard in disguise."""
    r = _compare()
    assert not hasattr(r, "winner")
    forbidden = ("better", "worse", "winner", "beats", "outperform", "rank")
    text = " ".join(r.reasons).lower()
    assert not any(w in text for w in forbidden)


# --------------------------------------------------------------------------
# a metric that could not be computed
# --------------------------------------------------------------------------


def test_a_cue_with_one_silent_arm_gets_a_reason_and_no_numbers():
    """Not a zero effect. "Nothing came back" and "the two agreed" are opposite
    findings and a 0.0 would render as the second one."""
    r = _compare(b_rows=_rows("hot", "B", _POOL_B, n=6, invalid=6))
    assert r.delta_hat is None
    assert r.p_value is None and r.q_value is None
    assert r.ci_lo is None and r.ci_hi is None
    assert r.status == "insufficient evidence"
    assert r.reasons == ["no valid trials for at least one model"]


def test_too_few_valid_trials_records_gate_1_with_both_counts():
    m = _manifest(min_valid_trials=20)
    r = _compare(m)
    assert any(x.startswith("gate 1") for x in r.reasons)
    assert "12/12" in " ".join(r.reasons)  # the numbers, so the shortfall is legible
    assert r.status == "insufficient evidence"


def test_an_effect_below_the_frozen_floor_is_no_detected_deviation_not_a_failure():
    """§6.5 gate 5. "We looked and found nothing above the floor we declared" is
    a result; `insufficient evidence` would be a different, weaker claim."""
    m = _manifest(effect_floor=10.0)  # unreachably high, so every cue is below it
    r = _compare(m)
    assert r.status == "no detected deviation"
    assert any("below the frozen smallest effect of interest" in x for x in r.reasons)
    assert r.delta_hat is not None, "the measured effect is still reported"


def test_a_below_floor_effect_that_also_failed_another_gate_is_insufficient_evidence():
    """The distinction matters: "nothing there" requires that everything else
    was in order. With a broken gate upstream the honest answer is that we
    cannot say."""
    m = _manifest(effect_floor=10.0, min_valid_trials=99)
    assert _compare(m).status == "insufficient evidence"


def test_an_effect_inside_the_capability_reference_is_attributed_to_capability():
    """§5.7 / §4.2.1: a difference no larger than the same-family scale contrast
    is not evidence of a difference in learned association."""
    r = _compare(capability_reference=10.0)
    assert r.status == "capability-attributable"
    assert r.capability_reference == 10.0
    assert any("scale and competence" in x for x in r.reasons)


def test_the_capability_reference_is_none_when_no_such_study_exists():
    """Absent, not zero — a 0.0 reference would attribute every effect to
    capability at the first comparison."""
    assert _compare().capability_reference is None


# --------------------------------------------------------------------------
# compliance parity (§6.5.1)
# --------------------------------------------------------------------------


def test_a_large_parse_rate_gap_makes_the_pair_incomparable():
    """Not "no difference", not "weak evidence": the two samples are not
    samples of the same thing, and no n rescues that."""
    m = _manifest(compliance_parity_max=0.25)
    r = _compare(m, b_rows=_rows("hot", "B", _POOL_B, n=12, invalid=6))  # 1.00 vs 0.50
    assert r.status == "incomparable"
    assert any(x.startswith("gate 3") for x in r.reasons)
    assert "non-random subsample" in " ".join(r.reasons)


def test_an_incomparable_pair_stops_before_the_bounds_are_computed():
    """The Manski bounds describe a comparison. Publishing them for a pair that
    is not comparable would dress the refusal up as a measurement."""
    m = _manifest(compliance_parity_max=0.25)
    r = _compare(m, b_rows=_rows("hot", "B", _POOL_B, n=12, invalid=6))
    assert r.manski_lo is None and r.manski_hi is None


def test_a_gap_inside_the_declared_tolerance_is_not_gated():
    """The threshold is frozen in the manifest, so the gate is a declared
    tolerance rather than a judgement made after seeing the rates."""
    m = _manifest(compliance_parity_max=0.25)
    r = _compare(m, b_rows=_rows("hot", "B", _POOL_B, n=12, invalid=2))  # 1.00 vs 0.83
    assert r.status != "incomparable"
    assert not any(x.startswith("gate 3") for x in r.reasons)


def test_the_parity_threshold_is_the_manifests_and_nothing_elses():
    """Same data, two frozen tolerances, two verdicts — which is exactly why the
    tolerance has to be declared before collection."""
    rows_b = _rows("hot", "B", _POOL_B, n=12, invalid=6)
    strict = _compare(_manifest(compliance_parity_max=0.25), b_rows=rows_b)
    loose = _compare(_manifest(compliance_parity_max=0.75), b_rows=rows_b)
    assert strict.status == "incomparable"
    assert loose.status != "incomparable"


def test_missing_trials_widen_the_manski_bounds_around_the_point_estimate():
    """§6.5.1: the unparsed trials are unobserved by construction, so the honest
    statement about them is a bracket, not an imputation."""
    clean = _compare()
    lossy = _compare(b_rows=_rows("hot", "B", _POOL_B, n=12, invalid=2))
    assert clean.manski_lo == pytest.approx(clean.manski_hi), "nothing missing, nothing to bracket"
    assert lossy.manski_lo < lossy.manski_hi


def test_a_bound_that_spans_the_effect_floor_blocks_confirmation():
    """The point estimate clears the floor and the cue still cannot be confirmed.

    With two of twelve trials unparsed the bracket here is roughly
    [0.33, 0.58] around a Δ̂ of 0.53, so a floor of 0.45 sits inside it: the
    worst case for the unobserved trials puts the effect below the smallest
    effect declared interesting, and a confirmation would be asserting something
    the data cannot exclude.
    """
    m = _manifest(effect_floor=0.45)
    r = _compare(m, b_rows=_rows("hot", "B", _POOL_B, n=12, invalid=2))
    assert r.manski_lo < m.effect_floor < r.manski_hi
    assert r.delta_hat > m.effect_floor, "the point estimate alone would have passed"
    assert any("worst-case bound" in x for x in r.reasons)
    assert r.status != "confirmed"


# --------------------------------------------------------------------------
# informativeness floor and the echo detectors (§6.5.2)
# --------------------------------------------------------------------------


def _degenerate_rows(model_key="B", n=12):
    return _rows("hot", model_key, ["hot, hot, hot"], n=n)


def test_an_arm_that_echoes_the_cue_fails_the_informativeness_floor():
    """The whole reason gate 4 exists: this arm's split-half reliability is a
    perfect 1.0, because every trial is identical. Reliability alone would call
    it the best-behaved arm in the study."""
    m = _manifest(min_distinct_types=3, min_associate_entropy=0.8)
    r = _compare(m, b_rows=_degenerate_rows())
    prof = r.arms["B"]
    assert prof.reliability == pytest.approx(1.0, abs=1e-6)
    assert prof.distinct_types == 1
    assert prof.cue_echo_rate == 1.0
    reasons = " ".join(r.reasons)
    assert "gate 4" in reasons
    assert "distinct associate types" in reasons
    assert "entropy" in reasons
    assert "echoes the cue or the exemplar" in reasons
    assert r.status == "insufficient evidence"


def test_the_echo_gate_needs_a_majority_of_trials_not_a_single_slip():
    """A model that echoes once in twelve is not degenerate, and a gate that
    fired on one trial would disqualify almost every real arm."""
    rows = _rows("hot", "B", _POOL_B, n=11) + [_trial("hot", "B", "hot, hot, hot", 11)]
    r = _compare(b_rows=rows)
    assert r.arms["B"].cue_echo_rate == pytest.approx(1 / 12)
    assert not any("echoes the cue" in x for x in r.reasons)


def test_a_low_entropy_arm_fails_even_when_it_echoes_nothing():
    """Degeneracy does not have to look like echoing: three words repeated for
    every trial carry no more association than one."""
    m = _manifest(min_distinct_types=3, min_associate_entropy=1.5)
    r = _compare(m, b_rows=_rows("hot", "B", ["pepper, curry, spicy"], n=12))
    assert r.arms["B"].cue_echo_rate == 0.0
    assert r.arms["B"].distinct_types == 3  # passes the type floor
    assert any("entropy" in x for x in r.reasons)  # and still fails the entropy floor


def test_a_differential_fragmentation_gap_is_recorded_as_partly_representational():
    """§6.4.4: the encoder puts fragmented strings in low-density regions, so
    part of such an effect is about the representation, not the association.
    The differential is reported; the level never is."""
    m = _manifest(oov_differential_max=0.25)
    r = _compare(m, b_rows=_rows("hot", "B", ["c0ld, w4rm, f1re"], n=12))
    assert any("representational, not associative" in x for x in r.reasons)


def test_a_reliability_below_the_floor_names_the_arm_that_failed_it():
    """Two arms, one gate: a reason that did not name the arm would leave the
    reader unable to tell which half of the comparison is unusable."""
    m = _manifest(reliability_floor=0.99)
    r = _compare(m, b_rows=_rows("hot", "B", _POOL_B, n=12))
    gate2 = [x for x in r.reasons if x.startswith("gate 2")]
    assert gate2, "an unreliable arm must be called out"
    assert any(x.split()[2] in ("A", "B") for x in gate2)


def test_a_clean_comparison_reaches_suggestive_and_no_further():
    """Discovery cannot produce more than `suggestive` no matter how clean it
    is: `confirmed` requires arms R and G, which have not been collected."""
    m = _manifest(reliability_floor=-1.0, min_associate_entropy=0.0, min_distinct_types=1)
    r = _compare(m)
    assert r.reasons == []
    assert r.status == "suggestive"


# --------------------------------------------------------------------------
# BY across the cue family
# --------------------------------------------------------------------------


def _result(cue: str, p: float | None, status: str = "suggestive") -> A.CueResult:
    r = A.CueResult(cue=cue, stratum="control_neutral")
    r.p_value = p
    r.delta_hat = 0.5
    r.status = status
    return r


def test_by_q_values_are_assigned_to_every_testable_cue():
    m = _manifest()
    results = [_result(f"c{i}", p) for i, p in enumerate([0.001, 0.01, 0.2, 0.6])]
    A.finalize_family(results, m)
    assert all(r.q_value is not None for r in results)
    assert all(r.q_value >= r.p_value for r in results), "BY only ever inflates a p"
    qs = [r.q_value for r in sorted(results, key=lambda r: r.p_value)]
    assert qs == sorted(qs), "q must be monotone in p"


def test_the_by_penalty_is_the_harmonic_one_not_bh():
    """BY, not BH: per-cue tests share an embedder, a frame and collection
    blocks, so positive dependence cannot be assumed. The `c(m) = Σ 1/i` factor
    is the price of not having to argue about the dependence structure — for
    four cues, c(4) = 1 + 1/2 + 1/3 + 1/4 = 2.0833.
    """
    m = _manifest()
    results = [_result(f"c{i}", 0.01) for i in range(4)]
    A.finalize_family(results, m)
    c4 = sum(1.0 / i for i in range(1, 5))
    assert results[0].q_value == pytest.approx(0.01 * 4 * c4 / 4)
    assert results[0].q_value > 0.01 * 4 / 4, "a BH q would be smaller; this must not be BH"


def test_a_cue_that_does_not_survive_by_is_demoted_with_the_reason_recorded():
    """Not dropped from the artifact: "we tested this and it did not survive the
    family correction" is itself the finding, and the q-value is shown."""
    m = _manifest(q_threshold=0.05)
    results = [_result("weak", 0.04)] + [_result(f"c{i}", 0.5) for i in range(9)]
    A.finalize_family(results, m)
    weak = results[0]
    assert weak.q_value > 0.05
    assert weak.status == "no detected deviation"
    assert any(x.startswith("gate 6") for x in weak.reasons)
    assert f"{weak.q_value:.4f}" in " ".join(weak.reasons)


def test_a_cue_that_survives_by_keeps_its_status():
    m = _manifest(q_threshold=0.05)
    results = [_result("strong", 1e-6)] + [_result(f"c{i}", 0.5) for i in range(9)]
    A.finalize_family(results, m)
    assert results[0].q_value <= 0.05
    assert results[0].status == "suggestive"
    assert not any(x.startswith("gate 6") for x in results[0].reasons)


def test_by_never_promotes_a_cue_that_failed_an_earlier_gate():
    """Gate 6 is a filter, not a rescue: surviving the family correction cannot
    undo an incomparable parse-rate gap or a missing effect."""
    m = _manifest()
    results = [_result("x", 1e-9, status="incomparable"), _result("y", 1e-9, status="suggestive")]
    A.finalize_family(results, m)
    assert results[0].status == "incomparable"


def test_an_untestable_cue_gets_a_null_q_and_does_not_enter_the_family():
    """A cue with no p-value has nothing to correct. Giving it `q = 0` would
    make the most broken cue in the study look like its strongest result, and
    counting it in `m` would penalise every real cue for it."""
    m = _manifest()
    with_gap = [_result("a", 0.01), _result("broken", None), _result("b", 0.02)]
    without = [_result("a", 0.01), _result("b", 0.02)]
    A.finalize_family(with_gap, m)
    A.finalize_family(without, m)
    assert with_gap[1].q_value is None
    assert with_gap[0].q_value == pytest.approx(without[0].q_value)


def test_a_nan_p_value_is_treated_as_untestable_rather_than_as_zero():
    m = _manifest()
    results = [_result("nan", float("nan")), _result("ok", 0.01)]
    A.finalize_family(results, m)
    assert results[0].q_value is None


def test_an_empty_family_is_returned_unchanged_rather_than_raising():
    m = _manifest()
    assert A.finalize_family([], m) == []
    only_broken = [_result("broken", None)]
    assert A.finalize_family(only_broken, m)[0].q_value is None


# --------------------------------------------------------------------------
# joining the confirmation arms (§6.5 gates 7-8)
# --------------------------------------------------------------------------


def _triple(r_status="suggestive", g_status="suggestive"):
    d = _result("hot", 0.001)
    d.delta_hat = 0.42
    r = _result("hot", 0.002, status=r_status)
    r.delta_hat = 0.31
    g = _result("hot", 0.003, status=g_status)
    return {"hot": d}, {"hot": r}, {"hot": g}


def test_reproduced_on_both_arms_is_confirmed_and_shows_the_confirmation_number():
    """§6.5.3: the displayed magnitude comes from arm R, never from discovery —
    discovery effects are selection-inflated by construction."""
    d, r, g = _triple()
    out = A.confirm(d, r, g, _manifest())["hot"]
    assert out.status == "confirmed"
    assert out.partition == "confirmation"
    assert out.delta_hat == 0.31, "the discovery effect must not be what is shown"
    assert "shown here only as the selection reason" in out.reasons[0]
    assert "0.4200" in out.reasons[0], "discovery survives only as the reason it was tested"


def test_a_non_strict_source_can_never_reach_confirmed():
    """§5.5: a synthetic adapter, a stand-in embedder or a moving alias caps the
    achievable status, and the artifact says which cap applied."""
    d, r, g = _triple()
    out = A.confirm(d, r, g, _manifest(), strict_source=False)["hot"]
    assert out.status == "suggestive"
    assert any("capped at suggestive" in x for x in out.reasons)


def test_an_effect_that_does_not_survive_a_different_frame_is_frame_specific():
    """A finding, not a failure — and explicitly not the same thing as a false
    positive, which is why it has its own state."""
    d, r, g = _triple(g_status="no detected deviation")
    out = A.confirm(d, r, g, _manifest())["hot"]
    assert out.status == "frame-specific"
    assert any("not the same thing as a false positive" in x for x in out.reasons)


def test_a_discovery_hit_that_arm_r_did_not_reproduce_stays_suggestive():
    d, r, g = _triple(r_status="no detected deviation")
    out = A.confirm(d, r, g, _manifest())["hot"]
    assert out.status == "suggestive"
    assert any("selection noise" in x for x in out.reasons)


def test_a_cue_with_no_arm_r_reports_that_rather_than_a_verdict():
    d, _, g = _triple()
    out = A.confirm(d, {}, g, _manifest())["hot"]
    assert "arm R was not collected for this cue" in out.reasons
    assert out.status == "suggestive", "the discovery status is left as it was, not upgraded"
    assert out.partition == "discovery"


# --------------------------------------------------------------------------
# calibration controls (§6.7)
# --------------------------------------------------------------------------


def test_the_aa_control_reports_missing_rather_than_a_zero_effect_on_thin_data():
    p = _profiles(_rows("hot", "A", _POOL_A, n=4))["hot"]["A"]
    out = A.aa_control(p, _manifest(), BANDWIDTH, draws=2)
    assert out["status"] == "missing"
    assert "fewer than 8" in out["reason"]
    assert "delta_hat_mean" not in out, "a missing control reports no numbers at all"


def test_the_aa_control_splits_one_model_against_itself():
    """If this does not come out near zero, every between-model number in the
    study is suspect and no effect size rescues it."""
    p = _profiles(_rows("hot", "A", _POOL_A, n=16))["hot"]["A"]
    out = A.aa_control(p, _manifest(), BANDWIDTH, draws=4)
    assert out["status"] == "measured"
    assert out["model_key"] == "A"
    assert out["n_splits"] == 4
    assert abs(out["delta_hat_mean"]) < 0.05  # same distribution split in half
    assert 0.0 <= out["false_positive_rate_at_0.05"] <= 1.0


def test_the_positive_control_is_a_within_model_pass_rate_per_cue():
    """§6.7.2: instrument validity, not cross-model agreement. A model that
    cannot produce `hot -> cold` is not producing association data, and no
    divergence number computed from it means anything."""
    rows = _rows("hot", "A", ["cold, warm, fire"], n=4) + _rows("hot", "B", ["pepper, x, y"], n=4)
    out = A.positive_control_report(rows, threshold=0.5)
    assert out["pass_rate"] == {"A": 1.0, "B": 0.0}
    assert out["passed"] == {"A": True, "B": False}
    assert out["per_cue"]["hot"]["B"] == 0.0


def test_a_non_control_cue_contributes_nothing_to_the_control_report():
    """The control is defined over a declared cue list; scoring an arbitrary cue
    against it would be scoring against the author's first guess."""
    out = A.positive_control_report(_rows("cupboard", "A", _POOL_A, n=4))
    assert out["pass_rate"] == {}
    assert out["n_trials"] == {}


def test_a_model_with_no_control_trials_has_no_pass_rate_rather_than_zero():
    out = A.positive_control_report([])
    assert out["pass_rate"] == {}
    assert out["passed"] == {}


# --------------------------------------------------------------------------
# the NaN/None discipline, one level down
# --------------------------------------------------------------------------


def test_the_internal_float_coercion_maps_nan_to_none_and_keeps_a_real_zero():
    """`_f` is what stands between an uncomputable statistic and a reported 0.

    A real 0.0 must survive it — "measured, and it was zero" is a finding — and
    NaN must not, because it is not a number the artifact can carry.
    """
    assert A._f(float("nan")) is None
    assert A._f(None) is None
    assert A._f(0.0) == 0.0
    assert A._f(np.float64(1.5)) == 1.5


def test_no_reported_field_of_a_cue_result_is_ever_nan():
    """Every float that leaves `compare_cue` has been through `_f`, so the
    exporter never has to decide what to do with a NaN."""
    r = _compare(b_rows=_rows("hot", "B", _POOL_B, n=12, invalid=4))
    for name, value in vars(r).items():
        if isinstance(value, float):
            assert not math.isnan(value), f"{name} is NaN"
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, float):
                    assert not math.isnan(v), f"{name}[{k}] is NaN"
