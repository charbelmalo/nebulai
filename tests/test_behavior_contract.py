"""The frozen manifest — the object every number in the study is defined against.

BEHAVIORAL-DIVERGENCE-PLAN.md §9.1 and §9.4. A behavioral divergence study has
no internal check on itself: the same trials analyzed with `effect_floor = 0.02`
and with `effect_floor = 0.002` produce different verdicts, and neither run can
tell you which threshold was chosen before the numbers were seen. The manifest
is the only thing that can, so the properties pinned here are the ones that make
it evidence rather than a config file:

  * the freeze is a CONTENT hash, so the same study frozen twice is the same
    hash and a single edited character is a different one;
  * the hash covers held-out frame text, so a Lane B frame provably existed
    before discovery ran;
  * `protocol_hash` answers a narrower question — "were these two studies
    collected the same way?" — and therefore must move for a sampler change and
    stay still for a threshold change;
  * `reasoning_tokens_p95 = None` means MISSING. §5.1's budget exists to bound a
    quantity nobody has measured on this machine, and a zero would claim it had
    been measured and found to be nothing;
  * a pinned model id is carried verbatim, never normalized, aliased or
    substituted.
"""

import json
from dataclasses import FrozenInstanceError

import pytest

from nebulai.behavior.contract import (
    CANARY_CUE,
    MANIFEST_FORMAT_VERSION,
    Cue,
    Manifest,
    ManifestError,
    ModelRef,
    PromptFrame,
    load_manifest,
    manifest_from_dict,
    prompt_sha,
)
from nebulai.behavior.protocol import default_frames

FROZEN_AT = "2026-09-11T00:00:00Z"


def _manifest(**kw) -> Manifest:
    base = dict(
        study_id="t_contract",
        created="2026-09-11T00:00:00Z",
        models=[
            ModelRef("A", "fake", "fake", label="synthetic A"),
            ModelRef("B", "xai", "grok-4-0709", label="Grok 4 (0709)"),
        ],
        cues=[Cue("hot", "control_neutral"), Cue("daddy", "identity", pack="daddy")],
        frames=default_frames(),
    )
    base.update(kw)
    return Manifest(**base)  # type: ignore[arg-type]


def _frozen(**kw) -> Manifest:
    m = _manifest(**kw)
    m.freeze(FROZEN_AT)
    return m


# --------------------------------------------------------------------------
# freezing produces a stable hash
# --------------------------------------------------------------------------


def test_two_identical_manifests_freeze_to_the_same_hash():
    """Content-addressed, not identity-addressed.

    If the hash depended on anything incidental — object identity, dict order,
    the wall clock — two people could not check that they are looking at the
    same study, which is the only thing the hash is for.
    """
    assert _frozen().frozen_hash == _frozen().frozen_hash


def test_the_hash_survives_a_json_round_trip_through_disk(tmp_path):
    """The hash must be re-derivable from the file, not only from the object.

    `load_manifest` re-computes it on every load; if serialization lost or
    reordered a field the study would fail integrity on its own artifact.
    """
    m = _frozen()
    p = m.save(tmp_path / "manifest.json")
    again = load_manifest(p)
    assert again.frozen_hash == m.frozen_hash
    assert again.compute_hash() == m.frozen_hash


def test_the_hash_is_a_sha256_and_says_so():
    h = _frozen().frozen_hash
    assert h is not None
    algo, digest = h.split(":", 1)
    assert algo == "sha256"
    assert len(digest) == 64  # a bare hex sha256, so a reader can verify it


def test_freezing_records_when_and_is_reported_by_is_frozen():
    m = _manifest()
    assert m.is_frozen is False
    m.freeze(FROZEN_AT)
    assert m.is_frozen is True
    assert m.frozen_at == FROZEN_AT


def test_editorial_and_self_referential_fields_are_outside_the_hash():
    """`notes`, `created` and the freeze stamp itself must not move the hash.

    `frozen_hash` is excluded of necessity — it cannot hash itself — and
    `created`/`notes` are excluded so that re-typing a comment does not look
    like a protocol change. Everything else is in.
    """
    a = _frozen(notes="first pass", created="2026-01-01T00:00:00Z")
    b = _frozen(notes="rewritten after review", created="2026-05-05T00:00:00Z")
    assert a.frozen_hash == b.frozen_hash


@pytest.mark.parametrize(
    "field,value",
    [
        ("effect_floor", 0.002),
        ("q_threshold", 0.10),
        ("trials_per_cue", 80),
        ("delta_hat_form", "standardized"),
        ("compliance_parity_max", 0.5),
        ("rbo_p", 0.95),
        ("jsd_correction", "plugin"),
        ("min_distinct_types", 2),
        ("seed", 43),
        ("embedder_sha", "deadbeef"),
        ("second_embedder_resolution", "strengthen"),
    ],
)
def test_every_researcher_degree_of_freedom_moves_the_hash(field, value):
    """§9.4: a free parameter outside the hash is an undeclared degree of freedom.

    Each of these decides what a reported number MEANS, so changing one after
    the fact must be detectable. The parametrization is the point — a single
    spot check would pass against a hash over three fields.
    """
    assert _frozen().frozen_hash != _frozen(**{field: value}).frozen_hash


def test_held_out_frame_text_is_inside_the_hash():
    """§5.4: arm G's frames must provably predate discovery.

    If held-out text were outside the hash, a frame could be written after
    seeing which cues looked interesting and the study would still verify —
    which is precisely the generalization test being faked.
    """
    frames = default_frames()
    edited = [
        PromptFrame(f.id, f.role, f.template + " Please.", f.stop)
        if f.role == "heldout"
        else f
        for f in frames
    ]
    assert _frozen(frames=frames).frozen_hash != _frozen(frames=edited).frozen_hash


def test_a_cue_added_after_the_freeze_is_visible_in_the_hash():
    small = _frozen()
    big = _frozen(cues=[Cue("hot", "control_neutral"), Cue("daddy", "identity", pack="daddy"),
                        Cue("salt", "control_neutral")])
    assert small.frozen_hash != big.frozen_hash


# --------------------------------------------------------------------------
# a frozen manifest refuses mutation
# --------------------------------------------------------------------------


def test_a_post_freeze_edit_is_caught_by_verify_integrity():
    m = _frozen()
    m.effect_floor = 0.0001  # the edit that would manufacture a finding
    with pytest.raises(ManifestError) as exc:
        m.verify_integrity()
    msg = str(exc.value)
    assert "edited after it was frozen" in msg
    assert m.frozen_hash in msg and m.compute_hash() in msg  # both, so it is diffable


def test_loading_an_edited_manifest_file_fails_at_load_not_at_analysis(tmp_path):
    """The failure must arrive before the numbers do.

    A manifest checked only at report time has already let a full analysis run
    and be looked at, and a threshold seen next to a result it produced cannot
    be un-seen.
    """
    m = _frozen()
    p = m.save(tmp_path / "manifest.json")
    d = json.loads(p.read_text())
    d["effect_floor"] = 0.0001
    p.write_text(json.dumps(d))
    with pytest.raises(ManifestError):
        load_manifest(p)


def test_re_freezing_is_refused_and_the_message_says_what_to_do_instead():
    m = _frozen()
    with pytest.raises(ManifestError) as exc:
        m.freeze("2026-10-01T00:00:00Z")
    msg = str(exc.value)
    assert "already frozen" in msg
    assert "study revision" in msg  # the supported path, named in the refusal
    assert m.frozen_at == FROZEN_AT, "the refused freeze must not have half-applied"


def test_an_unfrozen_manifest_may_not_have_a_trial_collected_against_it():
    with pytest.raises(ManifestError) as exc:
        _manifest().require_frozen()
    assert "draft" in str(exc.value)
    assert "preregistered" in str(exc.value)


def test_verify_integrity_is_silent_on_a_draft():
    """A draft has no hash to violate; only a freeze creates the obligation."""
    _manifest().verify_integrity()


def test_model_cue_and_frame_rows_are_frozen_dataclasses():
    """The nested rows cannot be edited in place at all.

    `Manifest` itself must stay mutable (freeze() writes two fields), so the
    tamper-evidence there is the hash. One level down there is no such need, so
    the stronger guarantee is used instead.
    """
    m = _frozen()
    for obj, attr, value in (
        (m.models[0], "pinned", "something-else"),
        (m.cues[0], "stratum", "identity"),
        (m.frames[0], "template", "{cue}"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(obj, attr, value)


def test_a_study_with_one_arm_cannot_be_frozen():
    """Divergence is a two-place relation; there is nothing to compare here."""
    with pytest.raises(ManifestError) as exc:
        _manifest(models=[ModelRef("A", "fake", "fake")]).freeze(FROZEN_AT)
    assert "at least two model arms" in str(exc.value)


def test_a_study_with_no_cues_cannot_be_frozen():
    with pytest.raises(ManifestError) as exc:
        _manifest(cues=[]).freeze(FROZEN_AT)
    assert "nothing to measure" in str(exc.value)


# --------------------------------------------------------------------------
# protocol_hash: moves for the protocol, still for everything else
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", 0.7),
        ("top_p", 0.9),
        ("max_output_tokens", 64),
        ("trials_per_cue", 80),
    ],
)
def test_protocol_hash_moves_when_the_sampler_or_task_shape_changes(field, value):
    assert _manifest().protocol_hash() != _manifest(**{field: value}).protocol_hash()


def test_protocol_hash_moves_when_any_frame_text_changes():
    """Two studies that sent different words did not share a protocol."""
    frames = default_frames()
    edited = [
        PromptFrame(f.id, f.role, f.template.replace("Give", "Provide"), f.stop)
        if f.role == "primary"
        else f
        for f in frames
    ]
    assert _manifest(frames=frames).protocol_hash() != _manifest(frames=edited).protocol_hash()


@pytest.mark.parametrize(
    "field,value",
    [
        ("effect_floor", 0.2),
        ("q_threshold", 0.01),
        ("cues", [Cue("bread", "control_neutral")]),
        ("embedder_id", "sentence-transformers/all-mpnet-base-v2"),
        ("max_cost_usd", 99.0),
        ("n_permutations", 5000),
        ("notes", "unrelated"),
    ],
)
def test_protocol_hash_is_unmoved_by_a_field_that_is_not_the_protocol(field, value):
    """The whole point of the second hash.

    "Were these collected the same way?" and "do these mean the same thing?"
    are different questions. A protocol hash that moved with the cue set could
    not answer the first one, and §5.4 grows the cue set by two orders of
    magnitude under a fixed protocol.
    """
    assert _manifest().protocol_hash() == _manifest(**{field: value}).protocol_hash()


def test_the_two_hashes_are_not_the_same_hash():
    m = _frozen()
    assert m.protocol_hash() != m.frozen_hash


def test_the_protocol_hash_does_not_require_a_freeze():
    """It is a question about text that exists, answerable on a draft."""
    assert _manifest().protocol_hash().startswith("sha256:")


# --------------------------------------------------------------------------
# missing is not zero
# --------------------------------------------------------------------------


def test_reasoning_tokens_p95_defaults_to_none_not_zero():
    """§5.1: nobody has measured Grok's reasoning-token distribution here.

    A 0 would say it was measured and found to be nothing, which would shrink
    the very budget that exists because the quantity is unknown.
    """
    assert _manifest().reasoning_tokens_p95 is None


def test_reasoning_tokens_p95_serializes_as_json_null(tmp_path):
    raw = _frozen().save(tmp_path / "m.json").read_text(encoding="utf-8")
    assert '"reasoning_tokens_p95": null' in raw
    assert '"reasoning_tokens_p95": 0' not in raw


def test_a_none_p95_round_trips_as_none_and_is_never_coerced(tmp_path):
    p = _frozen().save(tmp_path / "m.json")
    back = load_manifest(p)
    assert back.reasoning_tokens_p95 is None
    assert back.reasoning_tokens_p95 is not False, "None must not decay to a falsy number"


def test_a_measured_p95_round_trips_as_the_number(tmp_path):
    """The other half: `None` must mean missing, so a real 0 must be storable."""
    p = _frozen(reasoning_tokens_p95=252).save(tmp_path / "m.json")
    assert load_manifest(p).reasoning_tokens_p95 == 252
    zero = _frozen(reasoning_tokens_p95=0, study_id="t_zero")
    assert zero.reasoning_tokens_p95 == 0
    assert zero.frozen_hash != _frozen(study_id="t_zero").frozen_hash, (
        "a measured zero and an unmeasured None are different studies"
    )


@pytest.mark.parametrize("field", ["fingerprint_available", "mmd_bandwidth", "p_floor"])
def test_every_not_yet_audited_field_starts_as_none_and_stays_null(field, tmp_path):
    """§5.5.1 / §6.4: each of these is "not yet determined", not a value.

    `fingerprint_available=False` is a real audited answer; `None` is "the
    audit has not run", and the artifact must be able to say which.
    """
    m = _frozen()
    assert getattr(m, field) is None
    assert f'"{field}": null' in json.dumps(m.to_dict())
    assert getattr(load_manifest(m.save(tmp_path / "m.json")), field) is None


# --------------------------------------------------------------------------
# a pinned model id is never substituted
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pinned", ["grok-4-latest", "grok-4:latest", "grok-beta", "xai/latest", "grok-3-preview"]
)
def test_a_strict_study_refuses_a_moving_alias(pinned):
    """§5.5: a subject that can change underneath the study is not reproducible.

    The refusal is at manifest construction, before a single trial, because
    afterwards there is no way to tell which weights produced which trial.
    """
    with pytest.raises(ManifestError) as exc:
        _manifest(models=[ModelRef("A", "fake", "fake"), ModelRef("B", "xai", pinned)])
    msg = str(exc.value)
    assert "moving alias" in msg
    assert "confirmed" in msg  # says what the cost is, not just that it is refused


def test_exploratory_mode_allows_an_alias_and_the_manifest_records_that_it_did():
    """Allowed, but `strict=False` is itself in the hash.

    `analyze.py` caps such a study at `suggestive`; the gate that makes that
    enforceable is that the flag cannot be flipped after collection without
    breaking the hash.
    """
    loose = _frozen(
        strict=False,
        models=[ModelRef("A", "fake", "fake"), ModelRef("B", "xai", "grok-4-latest")],
    )
    assert loose.model("B").pinned == "grok-4-latest"
    assert loose.frozen_hash != _frozen(
        strict=True, models=[ModelRef("A", "fake", "fake"), ModelRef("B", "xai", "grok-4-0709")]
    ).frozen_hash


def test_an_exact_dated_release_is_not_mistaken_for_an_alias():
    """The alias check must not have false positives: a pin containing the
    letters of an alias inside a dated id is still a pin."""
    m = _frozen(
        models=[ModelRef("A", "fake", "fake"), ModelRef("B", "xai", "grok-4-0709-preview-0915")]
    )
    assert m.model("B").pinned == "grok-4-0709-preview-0915"


def test_a_pinned_id_survives_the_round_trip_character_for_character(tmp_path):
    """No casefolding, no family shortening, no provider prefix stripping."""
    pinned = "grok-4-0709"
    p = _frozen(
        models=[ModelRef("A", "fake", "fake"), ModelRef("B", "xai", pinned, revision="abc123")]
    ).save(tmp_path / "m.json")
    back = load_manifest(p)
    assert back.model("B").pinned == pinned
    assert back.model("B").revision == "abc123"
    assert json.loads(p.read_text())["models"][1]["pinned"] == pinned


def test_changing_the_pinned_id_is_a_different_study():
    a = _frozen()
    b = _frozen(models=[ModelRef("A", "fake", "fake"), ModelRef("B", "xai", "grok-4-0915")])
    assert a.frozen_hash != b.frozen_hash


def test_an_arm_without_a_pinned_id_is_refused():
    with pytest.raises(ManifestError) as exc:
        _manifest(models=[ModelRef("A", "fake", "fake"), ModelRef("B", "xai", "")])
    assert "pinned id" in str(exc.value)


def test_duplicate_arm_keys_are_refused_so_no_arm_can_shadow_another():
    with pytest.raises(ManifestError) as exc:
        _manifest(models=[ModelRef("A", "fake", "fake"), ModelRef("A", "xai", "grok-4-0709")])
    assert "duplicate model arm keys" in str(exc.value)


def test_asking_for_an_unknown_arm_or_frame_raises_rather_than_guessing():
    m = _frozen()
    with pytest.raises(ManifestError):
        m.model("C")
    with pytest.raises(ManifestError):
        m.frame("lane_z")


# --------------------------------------------------------------------------
# the rest of the schema's load-bearing invariants
# --------------------------------------------------------------------------


def test_a_study_with_no_primary_frame_is_refused():
    """Discovery and arm R both run on the primary frame; without one there is
    no partition for them to run on."""
    with pytest.raises(ManifestError) as exc:
        _manifest(frames=[f for f in default_frames() if f.role != "primary"])
    assert "no primary prompt frame" in str(exc.value)


def test_duplicate_cue_text_is_refused():
    """Two rows for one cue would double that cue's weight in the BY family and
    give it two chances at the same threshold."""
    with pytest.raises(ManifestError) as exc:
        _manifest(cues=[Cue("hot", "control_neutral"), Cue("hot", "identity")])
    assert "duplicate cue text" in str(exc.value)


@pytest.mark.parametrize("bad", ["cohen_d", "", "Difference"])
def test_an_unknown_primary_statistic_form_is_refused_by_name(bad):
    with pytest.raises(ManifestError) as exc:
        _manifest(delta_hat_form=bad)
    assert "delta_hat_form" in str(exc.value)


@pytest.mark.parametrize("bad", ["ignore", "both", ""])
def test_an_unknown_second_embedder_resolution_is_refused(bad):
    """§6.5.4: the UI copy is generated from this field, so an unrecognized
    value would silently produce the stronger of the two claims."""
    with pytest.raises(ManifestError) as exc:
        _manifest(second_embedder_resolution=bad)
    assert "second_embedder_resolution" in str(exc.value)


def test_a_frame_without_a_cue_placeholder_fails_when_it_is_rendered():
    f = PromptFrame(id="broken", role="primary", template="no placeholder here")
    with pytest.raises(ManifestError) as exc:
        f.render("hot")
    assert "{cue}" in str(exc.value)


def test_rendering_substitutes_the_cue_verbatim_including_its_spelling():
    """§6.2: cue text is stored and sent verbatim; normalization is a derived
    view, never a replacement."""
    f = PromptFrame(id="p", role="primary", template="{cue} -> ")
    assert f.render("Daddy") == "Daddy -> "


def test_primary_and_heldout_are_partitioned_by_role():
    m = _frozen()
    assert m.primary_frame.role == "primary"
    assert [f.id for f in m.heldout_frames] == ["lane_b_listing", "lane_b_terse"]
    assert m.primary_frame not in m.heldout_frames


def test_the_canary_cue_cannot_collide_with_a_real_cue():
    """`analyze.build_profiles` drops rows by this exact string, so a cue that
    could equal it would be silently deleted from the study."""
    assert CANARY_CUE == "__canary__"
    with pytest.raises(ManifestError):
        # dunder-wrapped text is not a word anyone would preregister, but the
        # duplicate check is what actually keeps two rows from colliding
        _manifest(cues=[Cue(CANARY_CUE, "control_neutral"), Cue(CANARY_CUE, "identity")])


def test_the_serialized_form_carries_its_own_format_version(tmp_path):
    d = _frozen().to_dict()
    assert d["manifest_format_version"] == MANIFEST_FORMAT_VERSION
    # ... and the version tag is not itself part of the content hash input,
    # since `manifest_from_dict` strips it before reconstructing.
    assert manifest_from_dict(d).compute_hash() == _frozen().frozen_hash


def test_rank_weights_round_trip_as_a_tuple_not_a_list(tmp_path):
    """`trial_vector` slices them positionally; a list would still work, but the
    hash must not change just because JSON has no tuples."""
    m = _frozen()
    back = load_manifest(m.save(tmp_path / "m.json"))
    assert back.rank_weights == m.rank_weights
    assert isinstance(back.rank_weights, tuple)
    assert back.compute_hash() == m.frozen_hash


def test_prompt_sha_is_content_addressed_and_tagged():
    a, b = prompt_sha("hot -> "), prompt_sha("hot ->")
    assert a.startswith("sha256:") and a != b
    assert prompt_sha("hot -> ") == a
