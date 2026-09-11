"""`behavior.json` — the artifact the Behavior page reads, and what it may say.

BEHAVIORAL-DIVERGENCE-PLAN.md §7.1, §8.1, §8.5, §9.5. The exporter is the last
place a careful analysis can be turned into a misleading picture, so the
properties pinned here are about what the file is allowed to contain rather than
about arithmetic:

  * **`build_export` is pure.** No I/O, no clock-dependent verdicts, no mutation
    of the results it is handed — so the artifact can be diffed against another
    run and the difference is the study, not the exporter.
  * **`missing` is `null`. Everywhere.** A metric that could not be computed is
    not a zero, and it is not a `NaN` either: §8.5 renders the two differently,
    and `JSON.parse` rejects the third.
  * **The landscape ships its transform.** §5.4 grows the cue set by two orders
    of magnitude while §7.1/§10.3 promise permalinkable cue positions; only a
    persisted `(mean, axes)` pair lets a new cue be placed instead of refitting
    and moving every already-published cue.
  * **The caption is sourced from the projection**, not asserted in copy
    (§8.1) — the number and the sentence come from the same fit.
  * **`trustworthiness` is reported, including when it cannot be computed.**
  * **The claim ranks nothing.** §1.1 is the strongest sentence the package may
    produce, and the exporter is where it is written down.
"""

import copy
import json
import math

import numpy as np
import pytest

from nebulai.behavior import BEHAVIOR_SCHEMA_VERSION
from nebulai.behavior.analyze import ArmProfile, CueResult
from nebulai.behavior.contract import Cue, Manifest, ModelRef
from nebulai.behavior.embed import HashEmbedder
from nebulai.behavior.export import (
    _r,
    build_export,
    cue_landscape,
    trustworthiness,
    write_export,
)
from nebulai.behavior.protocol import default_frames

CUE_WORDS = [
    "hot", "salt", "cat", "king", "day", "black", "daddy", "father",
    "mother", "window", "bread", "paper", "button", "shelf", "towel",
]


def _manifest(**kw) -> Manifest:
    base = dict(
        study_id="t_export",
        created="2026-09-11T00:00:00Z",
        models=[
            ModelRef("A", "fake", "fake", label="synthetic A"),
            ModelRef("B", "xai", "grok-4-0709", label="Grok 4 (0709)"),
        ],
        cues=[Cue("hot", "control_neutral"), Cue("daddy", "identity", pack="daddy")],
        frames=default_frames(),
    )
    base.update(kw)
    m = Manifest(**base)  # type: ignore[arg-type]
    m.freeze("2026-09-11T00:00:00Z")
    return m


def _profile(key: str, **kw) -> ArmProfile:
    base = dict(
        n_attempted=12, n_valid=12, parse_rate=1.0, distinct_types=5,
        entropy=1.4, type_token_ratio=0.4, reliability=0.82,
        top_ranked=["cold", "warm", "fire"], cue_echo_rate=0.0,
        exemplar_echo_rate=0.0, duplicate_rate=0.0, prompt_copy_rate=0.0,
        oov_fragment_ratio=0.0,
    )
    base.update(kw)
    return ArmProfile(model_key=key, **base)  # type: ignore[arg-type]


def _full_result(cue: str = "hot") -> CueResult:
    r = CueResult(cue=cue, stratum="control_neutral", pack="", partition="confirmation")
    r.arms = {"A": _profile("A"), "B": _profile("B")}
    r.delta_hat = 0.1234567
    r.mmd2_between = 0.2
    r.within = {"A": 0.05, "B": 0.06}
    r.p_value = 0.0012345678
    r.q_value = 0.0234567
    r.ci_lo, r.ci_hi = 0.05, 0.19
    r.manski_lo, r.manski_hi = 0.04, 0.21
    r.jsd = 0.31
    r.jsd_n = {"A": 36, "B": 36}
    r.rbo = 0.42
    r.location = 0.09
    r.dispersion = {"A": 0.01, "B": 0.02}
    r.dominant = "location"
    r.status = "confirmed"
    r.reasons = ["tested because discovery ranked it"]
    return r


def _empty_result(cue: str = "broken") -> CueResult:
    """A cue where nothing could be computed — the case every `null` is for."""
    r = CueResult(cue=cue, stratum="identity")
    r.arms = {
        "A": _profile("A", n_valid=0, parse_rate=0.0, reliability=None, oov_fragment_ratio=None),
        "B": _profile("B", n_valid=0, parse_rate=0.0, reliability=None, oov_fragment_ratio=None),
    }
    r.status = "insufficient evidence"
    r.reasons = ["no valid trials for at least one model"]
    return r


def _landscape(words=None, dims: int = 2) -> dict:
    return cue_landscape(words or CUE_WORDS, HashEmbedder(), dims)


def _export(results=None, **kw) -> dict:
    base = dict(
        landscape=_landscape(),
        diagnostics={"mmd_bandwidth": 0.9, "strict_source": False},
        runs=[{"total": 24, "spent_usd": 0.0}],
    )
    base.update(kw)
    return build_export(_manifest(), results or [_full_result()], **base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# build_export is pure
# --------------------------------------------------------------------------


def test_build_export_writes_nothing_to_disk(tmp_path, monkeypatch):
    """Purity is what makes the artifact testable at all: if assembling it
    touched the filesystem, every property below would need a real study."""
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    _export()
    assert set(tmp_path.iterdir()) == before


def test_build_export_does_not_mutate_the_results_it_is_given():
    """The caller still owns its objects. A reasons list appended to in here
    would show up in a second export of the same run."""
    r = _full_result()
    snapshot = copy.deepcopy(vars(r))
    _export([r])
    assert vars(r) == snapshot


def test_build_export_does_not_mutate_the_manifest_or_the_landscape():
    m = _manifest()
    land = _landscape()
    before_hash, before_land = m.frozen_hash, copy.deepcopy(land)
    build_export(m, [_full_result()], landscape=land, diagnostics={}, runs=[])
    m.verify_integrity()
    assert m.frozen_hash == before_hash
    assert land == before_land


def test_two_exports_of_the_same_study_differ_only_in_their_timestamp():
    a, b = _export(), _export()
    assert a.pop("generated") and b.pop("generated")
    assert a == b


def test_the_artifact_declares_the_schema_version_the_viewer_fails_closed_on():
    """§9.5: the viewer refuses an unknown version rather than rendering a
    partial study, which only works if the version is in the file."""
    assert _export()["schema"] == BEHAVIOR_SCHEMA_VERSION


def test_the_cue_index_maps_each_cue_to_its_position_in_the_cue_list():
    """The landscape's coordinate rows are positional, so this index is the only
    thing tying a coordinate to a cue."""
    payload = _export([_full_result("hot"), _full_result("salt"), _empty_result("cat")])
    assert payload["cue_index"] == {"hot": 0, "salt": 1, "cat": 2}
    assert [c["cue"] for c in payload["cues"]] == ["hot", "salt", "cat"]


# --------------------------------------------------------------------------
# missing is null, everywhere
# --------------------------------------------------------------------------


def _walk(obj, path="$"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, obj


def test_an_uncomputable_cue_exports_nulls_and_never_zeros():
    """The single most consequential line in the file. A 0.0 effect with a
    p-value of 0.0 is a spectacular finding; "we could not compute this" is not,
    and the two must not render the same."""
    c = _export([_empty_result()])["cues"][0]
    for field in ("delta_hat", "mmd2_between", "p_value", "q_value", "jsd", "rbo", "location"):
        assert c[field] is None, f"{field} became {c[field]!r} instead of null"
    assert c["ci"] == [None, None]
    assert c["manski"] == [None, None]
    assert c["arms"]["A"]["reliability"] is None
    assert c["arms"]["A"]["oov_fragment_ratio"] is None


def test_no_metric_of_an_uncomputable_cue_is_a_zero_anywhere_in_the_payload():
    """A blanket sweep rather than a field list, so a field added later cannot
    quietly default to 0 without this failing."""
    payload = _export([_empty_result()])
    numeric = {
        p: v
        for p, v in _walk(payload["cues"][0])
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    # The only numbers a cue with no data may carry are its own counts: how many
    # trials were attempted (12) and how many parsed (0). Those are measured.
    allowed = {"$.arms.A.n_attempted", "$.arms.B.n_attempted",
               "$.arms.A.n_valid", "$.arms.B.n_valid",
               "$.arms.A.parse_rate", "$.arms.B.parse_rate",
               "$.arms.A.distinct_types", "$.arms.B.distinct_types",
               "$.arms.A.entropy", "$.arms.B.entropy",
               "$.arms.A.type_token_ratio", "$.arms.B.type_token_ratio",
               "$.arms.A.detectors.cue_echo", "$.arms.B.detectors.cue_echo",
               "$.arms.A.detectors.exemplar_echo", "$.arms.B.detectors.exemplar_echo",
               "$.arms.A.detectors.within_trial_duplicate",
               "$.arms.B.detectors.within_trial_duplicate",
               "$.arms.A.detectors.prompt_copy", "$.arms.B.detectors.prompt_copy"}
    assert set(numeric) <= allowed, f"unexpected numbers on an uncomputable cue: {numeric}"


def test_a_real_zero_still_exports_as_zero():
    """The other side of the rule: "measured, and it was zero" is a finding, so
    `None` must be the only thing that becomes null."""
    r = _full_result()
    r.delta_hat = 0.0
    r.rbo = 0.0
    c = _export([r])["cues"][0]
    assert c["delta_hat"] == 0.0 and c["delta_hat"] is not None
    assert c["rbo"] == 0.0


def test_an_unmeasured_manifest_field_exports_as_null_not_as_a_default():
    """§5.1's reasoning-token p95 and §5.5.1's fingerprint audit are both
    "not measured here", and the artifact has to be able to say so."""
    man = _export()["manifest"]
    assert man["reasoning_tokens_p95"] is None
    assert man["fingerprint_available"] is None
    assert man["p_floor"] is None
    assert man["statistics"]["mmd_bandwidth"] is None


def test_a_non_finite_metric_is_exported_as_null_rather_than_as_nan():
    """SOURCE FIX (export._r): NaN and inf were being `round()`-ed and passed
    through, and `json.dumps` writes them as the bare tokens `NaN`/`Infinity`.
    No conforming JSON reader accepts either, so ONE un-computable statistic
    made the whole artifact unparseable instead of marking one field as not
    measured. `_r` now maps every non-finite value to `None`.
    """
    r = _full_result()
    r.delta_hat = float("nan")
    r.rbo = float("inf")
    c = _export([r])["cues"][0]
    assert c["delta_hat"] is None
    assert c["rbo"] is None
    assert "NaN" not in json.dumps(c) and "Infinity" not in json.dumps(c)


def test_the_rounding_helper_keeps_zero_and_drops_every_non_finite_value():
    assert _r(None) is None
    assert _r(0.0) == 0.0
    assert _r(float("nan")) is None
    assert _r(float("inf")) is None and _r(float("-inf")) is None
    assert _r(0.123456789) == 0.12346  # 5 dp by default
    assert _r(0.123456789, 8) == 0.12345679


def test_the_written_file_is_parseable_json_with_no_bare_nan_token(tmp_path):
    p = write_export(tmp_path / "behavior.json", _export([_full_result(), _empty_result()]))
    text = p.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["schema"] == BEHAVIOR_SCHEMA_VERSION


def test_writing_a_payload_that_still_carries_a_nan_fails_loudly(tmp_path):
    """SOURCE FIX (export.write_export): `allow_nan=False`, as the belt to
    `_r`'s braces. A NaN that reached the payload by another route — a
    diagnostics block, a field added later — used to produce a `behavior.json`
    whose `JSON.parse` throws in the viewer with no indication of the cause.
    Failing at write time names the file and the rule instead.
    """
    payload = _export(diagnostics={"mmd_bandwidth": float("nan")})
    with pytest.raises(ValueError) as exc:
        write_export(tmp_path / "behavior.json", payload)
    assert "non-finite" in str(exc.value)
    assert "null" in str(exc.value)
    assert not (tmp_path / "behavior.json").exists(), "a bad artifact must not be left behind"


def test_write_export_creates_the_study_directory(tmp_path):
    p = write_export(tmp_path / "nested" / "out" / "behavior.json", _export())
    assert p.exists()


# --------------------------------------------------------------------------
# the landscape and its persisted transform
# --------------------------------------------------------------------------


def test_the_landscape_is_pca_and_says_so():
    """§7.1.1 measured it: UMAP at n=180 produced a silhouette of 0.88 on
    SHUFFLED data against 0.43 on the real data — it manufactures islands at
    this sample size and a reader cannot tell one from a finding."""
    land = _landscape()
    assert land["method"] == "pca"
    assert land["dims"] == 2
    assert len(land["coords"]) == len(CUE_WORDS)
    assert all(len(row) == 2 for row in land["coords"])


def test_the_landscape_ships_the_transform_not_only_the_coordinates():
    """Without `(mean, axes)` a new cue can only be placed by refitting, and a
    refit moves every cue already published under a permalink."""
    land = _landscape()
    d = HashEmbedder().dim
    assert land["pca_axes_shape"] == [2, d]  # [dims, d], axis-major
    assert len(land["pca_mean"]) == d
    assert len(land["pca_axes"]) == 2 * d


def test_the_shipped_transform_reproduces_the_shipped_coordinates():
    """The contract the viewer relies on: `(x - pca_mean) @ pca_axes` must land
    a cue exactly where this fit already put it, or "placing" a new cue would
    put it in a different space from every existing one."""
    emb = HashEmbedder()
    land = cue_landscape(CUE_WORDS, emb, 2)
    dims, d = land["pca_axes_shape"]
    axes = np.asarray(land["pca_axes"], dtype=np.float64).reshape(dims, d).T  # (d, dims)
    mean = np.asarray(land["pca_mean"], dtype=np.float64)
    placed = (np.asarray(emb.encode(CUE_WORDS), dtype=np.float64) - mean) @ axes
    np.testing.assert_allclose(placed, np.asarray(land["coords"]), atol=1e-4)


def test_placing_a_cue_that_was_not_in_the_fit_uses_the_same_axes():
    """The whole reason the transform is persisted: the original cues must not
    move when a new one arrives."""
    emb = HashEmbedder()
    land = cue_landscape(CUE_WORDS, emb, 2)
    dims, d = land["pca_axes_shape"]
    axes = np.asarray(land["pca_axes"]).reshape(dims, d).T
    mean = np.asarray(land["pca_mean"])
    new = (np.asarray(emb.encode(["thimble"]), dtype=np.float64) - mean) @ axes
    assert new.shape == (1, 2)
    assert np.isfinite(new).all()
    # and the refit alternative really would have moved things, which is the
    # failure mode being avoided
    refit = cue_landscape(CUE_WORDS + ["thimble"], emb, 2)
    assert refit["coords"][0] != land["coords"][0]


def test_the_projection_caption_is_sourced_from_the_fit_not_asserted_in_copy():
    """§8.1: the number in the sentence and the number in the field are the same
    number, so the caption cannot drift from the projection it describes."""
    land = _landscape()
    proj = land["projection"]
    assert proj["quantity"] == pytest.approx(sum(land["explained_variance_ratio"]), abs=1e-4)
    assert f"{proj['quantity']:.0%}" in proj["quantity_label"]
    assert "cue-word embeddings" in proj["quantity_label"]


def test_the_projection_records_which_encoder_produced_the_positions():
    land = _landscape()
    assert land["projection"]["encoder"] == HashEmbedder().id
    assert land["projection"]["encoder_revision"] == HashEmbedder().revision


def test_the_landscape_warns_that_position_is_about_the_cue_words_only():
    """§7.1: neither model gets its own projection, because two projections side
    by side read as comparable internal geometry, which this is not."""
    warning = _landscape()["projection"]["warning"]
    assert "not from any model's behaviour" in warning
    assert "lexical-semantic similarity" in warning


def test_the_landscape_is_deterministic_for_the_same_cues_and_encoder():
    """A permalinked position that moved between two runs of the same study
    would make every shared link wrong."""
    assert _landscape()["coords"] == _landscape()["coords"]


def test_a_three_dimensional_landscape_exports_three_axes():
    land = _landscape(dims=3)
    assert land["dims"] == 3
    assert land["pca_axes_shape"][0] == 3
    assert len(land["explained_variance_ratio"]) == 3
    assert all(len(row) == 3 for row in land["coords"])


# --------------------------------------------------------------------------
# trustworthiness
# --------------------------------------------------------------------------


def test_trustworthiness_is_nan_when_there_are_too_few_points_to_compute_it():
    """n <= k + 1 leaves no neighbourhood to compare, so there is no number.

    NaN rather than 1.0 or 0.0: the caller has to decide how to render "not
    computable", and both of those would be a verdict on the projection.
    """
    rng = np.random.default_rng(0)
    high = rng.normal(size=(11, 8))
    low = rng.normal(size=(11, 2))
    assert math.isnan(trustworthiness(high, low, k=10))
    assert math.isnan(trustworthiness(high[:5], low[:5], k=10))


def test_trustworthiness_is_computed_once_there_are_enough_points():
    rng = np.random.default_rng(0)
    high = rng.normal(size=(40, 8))
    low = high[:, :2]
    t = trustworthiness(high, low, k=10)
    assert not math.isnan(t)
    assert 0.0 <= t <= 1.0


def test_a_projection_that_preserves_neighbours_scores_above_a_scrambled_one():
    """The statistic has to be able to tell the difference, or reporting it
    beside the landscape says nothing."""
    rng = np.random.default_rng(1)
    high = rng.normal(size=(60, 6))
    faithful = high[:, :2]
    scrambled = rng.normal(size=(60, 2))
    assert trustworthiness(high, faithful, k=10) > trustworthiness(high, scrambled, k=10)


def test_a_nan_trustworthiness_would_be_refused_at_write_time(tmp_path):
    """Ties the two rules together: the diagnostic is allowed to be
    un-computable, but it may not be written as a bare `NaN`."""
    rng = np.random.default_rng(0)
    t = trustworthiness(rng.normal(size=(11, 8)), rng.normal(size=(11, 2)), k=10)
    payload = _export(diagnostics={"trustworthiness": t})
    with pytest.raises(ValueError):
        write_export(tmp_path / "behavior.json", payload)
    # the correct thing for a caller to do with it
    payload["diagnostics"]["trustworthiness"] = _r(t)
    assert write_export(tmp_path / "behavior.json", payload).exists()
    assert json.loads((tmp_path / "behavior.json").read_text())["diagnostics"][
        "trustworthiness"
    ] is None


# --------------------------------------------------------------------------
# the claim
# --------------------------------------------------------------------------


LEADERBOARD_WORDS = (
    "best", "worst", "better", "worse", "winner", "loser", "beats", "outperform",
    "superior", "inferior", "leader", "top model", "wins", "stronger", "smarter",
    "more capable", "compares favourably", "compares favorably",
)


def test_the_claim_ranks_no_model_and_carries_no_leaderboard_language():
    """§1.2. The strongest sentence this package may produce is §1.1's, and the
    claim string is where that promise is actually kept."""
    claim = _export()["claim"].lower()
    for word in LEADERBOARD_WORDS:
        assert word not in claim, f"the claim contains leaderboard language: {word!r}"


def test_the_only_mention_of_ranking_in_the_claim_is_the_denial_of_it():
    """"rank" is allowed exactly once and only in the negative.

    A blanket ban on the word would forbid the sentence that does the work, so
    the test checks the sense instead: one occurrence, inside "no model is
    ranked".
    """
    claim = _export()["claim"].lower()
    assert claim.count("rank") == 1
    assert "no model is ranked" in claim


def test_the_claim_says_what_it_covers_and_what_it_does_not():
    claim = _export()["claim"]
    assert "protocol" in claim
    assert "no model is ranked" in claim
    assert "internals" in claim, "§1.2: nothing here describes a model's internals"


def test_no_model_names_appear_in_the_claim_so_it_cannot_read_as_a_verdict():
    claim = _export()["claim"]
    for m in _manifest().models:
        assert m.pinned not in claim
        assert (m.label not in claim) if m.label else True


def test_the_artifact_has_no_winner_field_at_any_depth():
    payload = _export([_full_result(), _empty_result()])
    banned = {"winner", "loser", "rank", "ranking", "score", "leaderboard", "best_model"}
    for path, _ in _walk(payload):
        leaf = path.rsplit(".", 1)[-1].split("[")[0]
        assert leaf not in banned, f"{path} is a ranking field"


def test_the_agreement_claim_matches_the_resolution_frozen_in_the_manifest():
    """§6.5.4: two English sentence-transformers on overlapping web data are not
    two representation KINDS, so under `downgrade` the copy must say
    "not specific to one of two similar encoders" and never
    "embedder-independent"."""
    down = build_export(
        _manifest(second_embedder_resolution="downgrade"), [_full_result()],
        landscape=_landscape(), diagnostics={}, runs=[],
    )["manifest"]["embedder"]
    up = build_export(
        _manifest(second_embedder_resolution="strengthen"), [_full_result()],
        landscape=_landscape(), diagnostics={}, runs=[],
    )["manifest"]["embedder"]
    assert down["agreement_claim"] == "not specific to one of two similar encoders"
    assert "independent" not in down["agreement_claim"]
    assert up["agreement_claim"] == "agrees across two representations of different kinds"


# --------------------------------------------------------------------------
# provenance the artifact must carry to be auditable on its own
# --------------------------------------------------------------------------


def test_the_artifact_carries_both_hashes_and_the_freeze_stamp():
    """A reader with the file alone must be able to ask "which study is this?"
    and "was it collected the same way as that one?"."""
    m = _manifest()
    man = _export()["manifest"]
    assert man["hash"] == m.frozen_hash
    assert man["protocol_hash"] == m.protocol_hash()
    assert man["frozen_at"] == "2026-09-11T00:00:00Z"
    assert man["strict"] is True


def test_the_exact_pinned_model_ids_travel_into_the_artifact():
    """§5.5: never a family name, never an alias — the artifact names the exact
    deployments the claim is about."""
    pinned = [x["pinned"] for x in _export()["manifest"]["models"]]
    assert pinned == ["fake", "grok-4-0709"]


def test_the_frames_are_listed_by_id_and_role_without_their_text():
    """Roles are what a reader needs to check the partition discipline; the
    template text lives in the manifest, which is hashed."""
    frames = _export()["manifest"]["frames"]
    assert {f["role"] for f in frames} == {"primary", "heldout", "canary"}
    assert all(set(f) == {"id", "role"} for f in frames)


def test_the_frozen_statistics_block_travels_with_the_numbers_it_defines():
    stats = _export()["manifest"]["statistics"]
    for key in ("delta_hat_form", "effect_floor", "q_threshold", "multiple_testing",
                "rbo_p", "jsd_correction", "compliance_parity_max"):
        assert key in stats
    assert stats["multiple_testing"] == "BY"


def test_diagnostics_runs_and_samples_pass_through_untouched():
    diag = {"strict_source": False, "canary": {"status": "missing"}}
    runs = [{"total": 24, "by_arm": {"discovery": {"trials": 24}}}]
    samples = {"hot": [{"model_key": "A", "raw": "cold, warm, fire"}]}
    payload = _export(diagnostics=diag, runs=runs, samples=samples)
    assert payload["diagnostics"] == diag
    assert payload["runs"] == runs
    assert payload["samples"] == samples


def test_omitted_samples_export_as_an_empty_object_not_as_null():
    """Nothing was requested, so nothing is there — distinct from a sample block
    that was requested and could not be produced."""
    assert _export()["samples"] == {}


def test_every_cue_carries_the_reasons_it_is_in_the_state_it_is_in():
    """§6.5: a status is a conjunction of named checks, and collapsing it to a
    label would hide which one failed."""
    payload = _export([_full_result(), _empty_result()])
    for c in payload["cues"]:
        assert c["status"]
        assert c["reasons"], f"{c['cue']} has a status with no stated reason"
