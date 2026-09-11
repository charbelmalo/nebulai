"""Directions: the constructors, the projection identity, and the null.

Four properties carry most of this file:

* a direction is **unit-normalised** and a zero difference is **refused** — two
  sets with the same mean have no direction between them, and returning an
  arbitrary unit vector there would manufacture a separation out of nothing;
* `parallel² + orthogonal² = ‖p‖²` — that identity is what makes the axis layout
  a re-coordinatisation of the point rather than a new embedding;
* the null distribution is **centred and scales as 1/√d** — a projection onto a
  random unit vector has mean 0 and standard deviation ‖p‖/√d, so a real
  histogram that looks like the ghost is telling the truth about itself;
* a direction with no null block is **not renderable**, mechanically (R5), and
  a projection channel in the wrong space is dropped with a stated reason (D2).
"""

import json

import numpy as np
import pytest

from nebulai.backend.channels import Channel, write_channels
from nebulai.backend.directions import (
    DEFAULT_NULL_N,
    DIRECTIONS_FILENAME,
    METHODS,
    Direction,
    DirectionError,
    contrast,
    diff_of_means,
    drop_directions,
    from_lora_rank1,
    from_two_selections,
    null_directions,
    pca_component,
    project,
    projection_channels,
    read_directions,
    renderable,
    separation,
    write_directions,
)


def _dir(**kw):
    base = dict(
        id="d1",
        label="test direction",
        space="W_E.centered",
        method="diff_of_means",
        vector=np.array([3.0, 4.0, 0.0]),
        source={"kind": "computed", "protocol": "unit test"},
    )
    base.update(kw)
    return Direction(**base)


# ── the object itself ─────────────────────────────────────────────────────────


def test_vector_is_unit_normalised_at_construction():
    d = _dir()
    assert np.isclose(np.linalg.norm(d.vector), 1.0)
    assert np.allclose(d.vector, [0.6, 0.8, 0.0])
    assert d.d == 3


def test_zero_vector_is_refused_rather_than_nudged():
    with pytest.raises(DirectionError, match="zero norm"):
        _dir(vector=np.zeros(4))


def test_nonfinite_vector_is_refused():
    with pytest.raises(DirectionError, match="NaN or inf"):
        _dir(vector=np.array([1.0, np.nan, 0.0]))


def test_unknown_space_is_refused_by_the_closed_set():
    from nebulai.spaces import UnknownSpaceError

    with pytest.raises(UnknownSpaceError):
        _dir(space="resid.14")  # missing the L


def test_unknown_method_is_refused():
    with pytest.raises(DirectionError, match="not in"):
        _dir(method="vibes")
    assert "diff_of_means" in METHODS


def test_protocol_is_mandatory():
    with pytest.raises(DirectionError, match="protocol is mandatory"):
        _dir(source={"kind": "computed"})


def test_source_kind_is_closed():
    with pytest.raises(DirectionError, match="computed"):
        _dir(source={"kind": "downloaded", "protocol": "x"})


def test_json_round_trip_preserves_the_vector_and_the_provenance():
    d = _dir()
    d.projection = {"channel": "proj.d1", "orth_channel": "proj.d1.orth"}
    d.null = {"method": "random_unit", "seed": 0, "n": 4, "channel": "proj.d1.null"}
    back = Direction.from_json(json.loads(json.dumps(d.to_json())))
    assert back.id == d.id and back.space == d.space and back.method == d.method
    assert np.allclose(back.vector, d.vector, atol=1e-6)
    assert back.source["protocol"] == "unit test"
    assert back.renderable


def test_a_direction_without_a_null_is_not_renderable():
    d = _dir()
    d.projection = {"channel": "proj.d1"}
    assert not d.renderable


# ── constructors ──────────────────────────────────────────────────────────────


def test_diff_of_means_points_from_neg_to_pos():
    pos = np.array([[2.0, 0.0], [4.0, 0.0]])
    neg = np.array([[0.0, 0.0], [0.0, 0.0]])
    d = diff_of_means(pos, neg, "W_E.centered", id="x", label="x", protocol="p")
    assert np.allclose(d.vector, [1.0, 0.0])
    assert d.source["n_pos"] == 2 and d.source["n_neg"] == 2
    assert d.source["kind"] == "computed"


def test_diff_of_means_refuses_two_different_widths():
    with pytest.raises(DirectionError, match="not the same space"):
        diff_of_means(
            np.zeros((2, 3)) + 1, np.zeros((2, 4)), "W_E.raw", id="x", label="x", protocol="p"
        )


def test_diff_of_means_refuses_identical_means():
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(DirectionError, match="same mean"):
        diff_of_means(m, m, "W_E.raw", id="x", label="x", protocol="p")


def test_pca_component_recovers_a_planted_axis_and_reports_its_variance():
    rng = np.random.default_rng(7)
    t = rng.standard_normal(400) * 5.0
    m = np.stack([t, rng.standard_normal(400) * 0.05], axis=1)
    d = pca_component(m, 0, "W_E.centered", id="pc1", label="pc1", protocol="p")
    assert abs(abs(float(d.vector[0])) - 1.0) < 1e-2
    assert d.source["explained_variance_ratio"] > 0.99
    assert d.method == "pca"


def test_pca_component_k_out_of_range_is_refused():
    with pytest.raises(DirectionError, match="out of range"):
        pca_component(np.eye(3), 5, "W_E.raw", id="x", label="x", protocol="p")


def test_from_lora_rank1_is_an_explicit_phase_5_refusal():
    with pytest.raises(NotImplementedError, match="phase 5"):
        from_lora_rank1("/nowhere/adapter.safetensors")


def test_from_two_selections_uses_row_indices_by_default():
    v = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 3.0]])
    d = from_two_selections([0, 1], [2, 3], v, "W_E.centered")
    assert np.allclose(d.vector, np.array([2.0, -2.0]) / np.linalg.norm([2.0, -2.0]))
    assert d.method == "two_selection"
    assert d.source["n_pos"] == 2 and d.source["n_neg"] == 2


def test_from_two_selections_honours_explicit_ids():
    v = np.array([[1.0, 0.0], [0.0, 1.0]])
    d = from_two_selections(["a"], ["b"], v, "W_E.raw", ids=["a", "b"])
    assert np.allclose(d.vector, np.array([1.0, -1.0]) / np.sqrt(2))


def test_from_two_selections_refuses_overlapping_selections():
    v = np.eye(3)
    with pytest.raises(DirectionError, match="both selections"):
        from_two_selections([0, 1], [1, 2], v, "W_E.raw")


def test_from_two_selections_refuses_unknown_ids():
    v = np.eye(3)
    with pytest.raises(DirectionError, match="not rows"):
        from_two_selections([0], [99], v, "W_E.raw")


def test_from_two_selections_refuses_an_empty_side():
    v = np.eye(3)
    with pytest.raises(DirectionError, match="non-empty"):
        from_two_selections([0], [], v, "W_E.raw")


# ── projection ────────────────────────────────────────────────────────────────


def test_projection_identity_parallel_squared_plus_orthogonal_squared():
    rng = np.random.default_rng(3)
    p = rng.standard_normal((256, 16))
    d = _dir(vector=rng.standard_normal(16))
    par, orth = project(p, d)
    assert np.allclose(par**2 + orth**2, (p * p).sum(axis=1), atol=1e-8)


def test_projection_of_the_direction_itself_is_its_own_length():
    d = _dir(vector=np.array([0.0, 2.0, 0.0]))
    par, orth = project(np.array([[0.0, 7.0, 0.0]]), d)
    assert np.isclose(par[0], 7.0)
    assert np.isclose(orth[0], 0.0, atol=1e-6)


def test_projection_is_signed_and_orthogonal_is_not():
    d = _dir(vector=np.array([1.0, 0.0, 0.0]))
    par, orth = project(np.array([[-5.0, 1.0, 0.0]]), d)
    assert par[0] < 0
    assert orth[0] > 0


def test_projection_refuses_a_dimensionality_mismatch():
    d = _dir(vector=np.ones(4))
    with pytest.raises(DirectionError, match="cannot project"):
        project(np.zeros((2, 7)), d)


def test_projection_accepts_a_bare_vector_as_well_as_a_direction():
    par, _ = project(np.array([[2.0, 0.0]]), np.array([4.0, 0.0]))
    assert np.isclose(par[0], 2.0)  # the raw vector is normalised on the way in


# ── the null ──────────────────────────────────────────────────────────────────


def test_null_directions_are_unit_vectors_and_reproducible():
    a = null_directions(32, 8, seed=5)
    b = null_directions(32, 8, seed=5)
    assert a.shape == (8, 32)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, null_directions(32, 8, seed=6))


def test_null_projection_is_centred_and_its_spread_scales_as_one_over_sqrt_d():
    """‖p‖/√d, measured at two dimensions an order of magnitude apart.

    This is the number the ghost histogram's width means. If it drifted, a real
    distribution could clear a null that had quietly become narrow.
    """
    rng = np.random.default_rng(11)
    for d_dim in (64, 1024):
        p = rng.standard_normal((4000, d_dim))
        p /= np.linalg.norm(p, axis=1, keepdims=True)  # unit-length points
        nulls = null_directions(d_dim, 64, seed=0)
        par = np.einsum("ij,ij->i", p, nulls[np.arange(4000) % 64])
        assert abs(float(par.mean())) < 0.01
        assert abs(float(par.std()) - 1.0 / np.sqrt(d_dim)) < 0.15 / np.sqrt(d_dim)


def test_null_directions_refuses_degenerate_shapes():
    with pytest.raises(DirectionError):
        null_directions(0, 4)
    with pytest.raises(DirectionError):
        null_directions(4, 0)


def test_separation_is_zero_for_two_draws_of_the_same_distribution():
    rng = np.random.default_rng(2)
    a = rng.standard_normal(5000)
    b = rng.standard_normal(5000)
    s = separation(a, b)
    assert abs(s["cohens_d"]) < 0.1
    assert s["overlap"] > 0.9
    assert s["n"] == 5000


def test_separation_is_signed_and_large_when_the_real_sits_above_its_null():
    rng = np.random.default_rng(2)
    s = separation(rng.standard_normal(4000) + 4.0, rng.standard_normal(4000))
    assert s["cohens_d"] > 3.5
    assert s["overlap"] < 0.1
    s2 = separation(rng.standard_normal(4000) - 4.0, rng.standard_normal(4000))
    assert s2["cohens_d"] < -3.5


# ── the four channels ─────────────────────────────────────────────────────────


def test_projection_channels_fill_in_the_blocks_and_are_index_aligned():
    rng = np.random.default_rng(1)
    p = rng.standard_normal((200, 12))
    d = _dir(vector=rng.standard_normal(12))
    chans = projection_channels(d, p, n_null=8, seed=3)
    assert [c.id for c in chans] == [
        "proj.d1",
        "proj.d1.orth",
        "proj.d1.null",
        "proj.d1.null.orth",
    ]
    assert all(len(c) == 200 for c in chans)
    assert all(c.space == d.space for c in chans)
    assert d.projection["channel"] == "proj.d1"
    assert d.null == {
        "method": "random_unit",
        "seed": 3,
        "n": 8,
        "channel": "proj.d1.null",
        "orth_channel": "proj.d1.null.orth",
    }
    assert d.renderable


def test_projection_channels_refuse_a_width_mismatch_with_both_numbers():
    d = _dir(vector=np.ones(5))
    with pytest.raises(DirectionError, match="5-dimensional.*9-dimensional"):
        projection_channels(d, np.zeros((3, 9)))


def test_projection_channel_values_match_project():
    rng = np.random.default_rng(4)
    p = rng.standard_normal((64, 7))
    d = _dir(vector=rng.standard_normal(7))
    par, orth = project(p, d)
    chans = projection_channels(d, p, n_null=4)
    assert np.allclose(np.asarray(chans[0].values), par)
    assert np.allclose(np.asarray(chans[1].values), orth)


def test_default_null_n_is_what_the_cli_advertises():
    assert DEFAULT_NULL_N == 32


# ── the registry file ─────────────────────────────────────────────────────────


def test_write_and_read_round_trip(tmp_path):
    p = tmp_path / DIRECTIONS_FILENAME
    write_directions(p, model="gpt2", revision="abc", directions=[_dir()])
    doc = read_directions(p)
    assert doc["meta"]["model"] == "gpt2"
    assert doc["meta"]["revision"] == "abc"
    assert [d["id"] for d in doc["directions"]] == ["d1"]


def test_merge_keeps_earlier_directions(tmp_path):
    p = tmp_path / DIRECTIONS_FILENAME
    write_directions(p, model="gpt2", revision="abc", directions=[_dir(id="a")])
    write_directions(p, model="gpt2", revision="abc", directions=[_dir(id="b")])
    assert [d["id"] for d in read_directions(p)["directions"]] == ["a", "b"]


def test_merge_rewrites_an_id_rather_than_duplicating_it(tmp_path):
    p = tmp_path / DIRECTIONS_FILENAME
    write_directions(p, model="gpt2", revision="abc", directions=[_dir(id="a", label="one")])
    write_directions(p, model="gpt2", revision="abc", directions=[_dir(id="a", label="two")])
    ds = read_directions(p)["directions"]
    assert len(ds) == 1 and ds[0]["label"] == "two"


def test_merge_refuses_to_mix_two_models(tmp_path):
    p = tmp_path / DIRECTIONS_FILENAME
    write_directions(p, model="gpt2", revision="a", directions=[_dir(id="a")])
    write_directions(p, model="distilgpt2", revision="b", directions=[_dir(id="b")])
    assert [d["id"] for d in read_directions(p)["directions"]] == ["b"]


def test_duplicate_ids_in_one_write_are_refused(tmp_path):
    with pytest.raises(DirectionError, match="duplicate"):
        write_directions(
            tmp_path / DIRECTIONS_FILENAME,
            model="gpt2",
            revision="a",
            directions=[_dir(id="x"), _dir(id="x")],
        )


def test_read_of_a_missing_or_broken_file_is_absence_not_a_crash(tmp_path):
    assert read_directions(tmp_path / "nope.json") is None
    bad = tmp_path / DIRECTIONS_FILENAME
    bad.write_text("{ not json")
    assert read_directions(bad) is None


def test_drop_removes_by_id(tmp_path):
    p = tmp_path / DIRECTIONS_FILENAME
    write_directions(p, model="gpt2", revision="a", directions=[_dir(id="x"), _dir(id="y")])
    assert drop_directions(p, ["x"]) == 1
    assert [d["id"] for d in read_directions(p)["directions"]] == ["y"]


# ── renderability: R5 and D2, mechanically ────────────────────────────────────


def _built(tmp_path, n=32, space="W_E.centered"):
    rng = np.random.default_rng(0)
    p = rng.standard_normal((n, 6))
    d = _dir(vector=rng.standard_normal(6), space=space)
    chans = projection_channels(d, p, n_null=4)
    write_channels(
        tmp_path / "channels.json",
        model="gpt2",
        revision="r",
        n_points=n,
        channels=chans,
    )
    write_directions(tmp_path / DIRECTIONS_FILENAME, model="gpt2", revision="r", directions=[d])
    return json.loads((tmp_path / DIRECTIONS_FILENAME).read_text()), json.loads(
        (tmp_path / "channels.json").read_text()
    )


def test_a_fully_projected_direction_is_renderable(tmp_path):
    ddoc, cdoc = _built(tmp_path)
    ok, drops = renderable(ddoc, cdoc)
    assert [d["id"] for d in ok] == ["d1"]
    assert drops == []


def test_no_null_block_means_dropped_with_a_reason(tmp_path):
    ddoc, cdoc = _built(tmp_path)
    ddoc["directions"][0].pop("null")
    ok, drops = renderable(ddoc, cdoc)
    assert ok == []
    assert drops[0][0] == "d1" and "null" in drops[0][1]


def test_no_projection_block_means_dropped_with_the_command_that_fixes_it(tmp_path):
    ddoc, cdoc = _built(tmp_path)
    ddoc["directions"][0].pop("projection")
    ok, drops = renderable(ddoc, cdoc)
    assert ok == []
    assert "nebulai direction project" in drops[0][1]


def test_a_missing_projection_channel_means_dropped(tmp_path):
    ddoc, cdoc = _built(tmp_path)
    cdoc["channels"] = [c for c in cdoc["channels"] if c["id"] != "proj.d1"]
    ok, drops = renderable(ddoc, cdoc)
    assert ok == []
    assert "not in channels.json" in drops[0][1]


def test_a_projection_channel_in_another_space_is_dropped_by_d2(tmp_path):
    ddoc, cdoc = _built(tmp_path)
    for c in cdoc["channels"]:
        if c["id"] == "proj.d1":
            c["space"] = "W_E.raw"
    ok, drops = renderable(ddoc, cdoc)
    assert ok == []
    assert "D2" in drops[0][1]


def test_no_channels_file_at_all_drops_everything_rather_than_guessing(tmp_path):
    ddoc, _ = _built(tmp_path)
    ok, drops = renderable(ddoc, None)
    assert ok == []
    assert len(drops) == 1


def test_absent_registry_is_absence_not_an_empty_finding():
    assert renderable(None, None) == ([], [])


# ── channels.json shares one file with the glitch lens ────────────────────────


def test_projecting_a_direction_does_not_delete_the_lens_channels(tmp_path):
    """The lifetime argument, tested: `direction project` merges into the same
    sidecar the glitch lens lives in, and must not take it out."""
    lens = Channel(
        id="we_norm",
        label="row norm",
        space="W_E.raw",
        method="l2",
        formula="norm(W_E[t])",
        values=np.arange(16, dtype=np.float64),
    )
    write_channels(tmp_path / "channels.json", model="gpt2", revision="r", n_points=16, channels=[lens])
    rng = np.random.default_rng(0)
    d = _dir(vector=rng.standard_normal(6))
    chans = projection_channels(d, rng.standard_normal((16, 6)), n_null=4)
    write_channels(
        tmp_path / "channels.json", model="gpt2", revision="r", n_points=16, channels=chans
    )
    ids = [c["id"] for c in json.loads((tmp_path / "channels.json").read_text())["channels"]]
    assert "we_norm" in ids
    assert "proj.d1" in ids and "proj.d1.null" in ids


# ── the contrast a made direction actually claims ─────────────────────────────


def test_contrast_measures_the_two_sets_the_direction_was_built_from():
    """`separation` asks whether the axis spreads the whole map; `contrast`
    asks whether it separates its own two sets. For a made direction the second
    is the claim, and it ships its own null."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal((80, 24)) + np.eye(1, 24, 0)[0] * 4.0
    b = rng.standard_normal((70, 24)) - np.eye(1, 24, 0)[0] * 4.0
    d = diff_of_means(a, b, "W_E.raw", id="c", label="c", protocol="p")
    c = d.source["contrast"]
    assert c["n_pos"] == 80 and c["n_neg"] == 70
    assert c["cohens_d"] > 5.0
    assert c["overlap"] < 0.05
    # the control: random unit directions on the same two sets. They are not
    # zero — a mean gap of 8 in one of 24 dimensions leaks into any axis — but
    # they are a different order of magnitude from the real one.
    assert c["null_cohens_d_mean"] < c["cohens_d"] / 4
    assert c["null_cohens_d_p95"] < c["cohens_d"] / 2


def test_contrast_is_near_zero_when_the_two_sets_are_one_distribution():
    rng = np.random.default_rng(4)
    pool = rng.standard_normal((800, 32))
    d = diff_of_means(pool[:400], pool[400:], "W_E.raw", id="c", label="c", protocol="p")
    c = d.source["contrast"]
    # in sample a diff-of-means always separates the sets it was fitted on,
    # which is exactly why the held-out number is the one that means anything
    assert c["cohens_d"] > 0.4  # in sample it always separates
    assert abs(c["heldout_cohens_d"]) < 0.4
    assert c["heldout_overlap"] > 0.7


def test_contrast_holds_out_half_of_each_set():
    rng = np.random.default_rng(5)
    a = rng.standard_normal((41, 12)) + 3.0
    b = rng.standard_normal((23, 12)) - 3.0
    c = contrast(a, b, a.mean(axis=0) - b.mean(axis=0))
    assert c["heldout_n_pos"] == 41 - 41 // 2
    assert c["heldout_n_neg"] == 23 - 23 // 2
    assert c["in_sample"] is True


def test_a_set_too_small_to_split_reports_missing_not_a_number():
    """R2/§2.2: absence is `missing`, never an optimistic stand-in."""
    rng = np.random.default_rng(6)
    a, b = rng.standard_normal((3, 8)) + 2.0, rng.standard_normal((9, 8))
    c = contrast(a, b, a.mean(axis=0) - b.mean(axis=0))
    assert c["heldout_cohens_d"] == "missing"
    assert c["heldout_overlap"] == "missing"
    assert c["heldout_n_pos"] == 0


def test_two_selections_carries_the_same_contrast_block():
    rng = np.random.default_rng(7)
    v = rng.standard_normal((60, 16))
    v[:30] += np.eye(1, 16, 2)[0] * 5.0
    d = from_two_selections(list(range(30)), list(range(30, 60)), v, "W_E.raw")
    assert d.source["contrast"]["cohens_d"] > 2.0
    assert d.source["contrast"]["n_pos"] == 30


def test_contrast_is_reproducible_for_a_fixed_seed():
    rng = np.random.default_rng(8)
    a, b = rng.standard_normal((40, 10)) + 1.0, rng.standard_normal((40, 10))
    v = a.mean(axis=0) - b.mean(axis=0)
    assert contrast(a, b, v, seed=11) == contrast(a, b, v, seed=11)
    assert contrast(a, b, v, seed=11) != contrast(a, b, v, seed=12)


# ── frozen prompt sets ───────────────────────────────────────────────────────
#
# The prompt sets exist because every published refusal direction is the wrong
# width for every model here, so the only available refusal-*style* direction is
# one computed from data this repository can show you. The tests below are about
# that word "frozen": if the data can move without the protocol string changing,
# the protocol string is decoration.


def test_the_refusal_style_set_is_two_matched_halves():
    from nebulai.backend.prompt_sets import REFUSAL_STYLE as ps

    assert len(ps.pos) == len(ps.neg) == 32
    assert len(set(ps.pos)) == 32 and len(set(ps.neg)) == 32
    # no string appears on both sides — a shared member would pull the two means
    # toward each other and make the contrast look weaker than the data is
    assert not (set(ps.pos) & set(ps.neg))


def test_the_two_halves_open_with_the_same_verbs():
    """Surface form is matched deliberately, and the test says by how much.

    Not because matching removes the topic difference — it does not, and the
    set's own caveat says so — but because an unmatched pair would make the
    direction partly a direction about sentence shape, which is a third thing
    nobody wants in there.
    """
    from nebulai.backend.prompt_sets import REFUSAL_STYLE as ps

    first = lambda xs: [x.split()[0] for x in xs]  # noqa: E731
    assert first(ps.pos) == first(ps.neg)


def test_the_sha_changes_when_a_single_character_does():
    from dataclasses import replace

    from nebulai.backend.prompt_sets import REFUSAL_STYLE as ps

    before = ps.sha
    moved = replace(ps, pos=(ps.pos[0] + ".",) + ps.pos[1:])
    assert moved.sha != before
    # and is stable across calls: a digest that drifted would make every
    # protocol string written from it unverifiable
    assert ps.sha == before


def test_the_protocol_names_the_set_the_model_the_layer_and_the_position():
    from nebulai.backend.prompt_sets import REFUSAL_STYLE as ps

    proto = ps.protocol(model="gpt2", revision="607a30d7", layer=8, position=-1)
    for needle in ("refusal-style-v1", ps.sha, "gpt2", "607a30d7", "resid.L8", "-1"):
        assert needle in proto, needle
    # and it carries the caveat, so the sentence that distrusts the direction
    # travels inside the direction rather than beside it
    assert "base model" in proto


def test_a_direction_fitted_on_the_set_records_the_sha_in_its_source():
    from nebulai.backend.directions import diff_of_means
    from nebulai.backend.prompt_sets import REFUSAL_STYLE as ps

    rng = np.random.default_rng(3)
    a = rng.normal(size=(32, 16))
    b = rng.normal(size=(32, 16)) + 1.0
    d = diff_of_means(
        a,
        b,
        "resid.L8",
        id="x",
        label="x",
        protocol=ps.protocol(model="gpt2", revision="r", layer=8, position=-1),
        source={"prompt_set": ps.id, "prompt_set_sha": ps.sha},
    )
    assert d.source["prompt_set_sha"] == ps.sha
    assert d.space == "resid.L8"
    # and it is NOT renderable on a W_E map, which is the whole point of the
    # space tag travelling with the vector
    assert d.null is None
