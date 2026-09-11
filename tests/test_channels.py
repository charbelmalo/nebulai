"""The closed space set and the `channels.json` sidecar.

Two properties are worth more than the rest of this file put together:

* a channel whose length disagrees with the map is **refused**, not truncated —
  an index-shifted channel mislabels every point past the gap and looks fine;
* a missing value is serialised as `null`, never as `0` — the whole point of the
  glitch lens is that "near the centroid" is a claim, and an unmeasured token
  drawn at distance 0 would be the strongest such claim in the file.
"""

import json

import numpy as np
import pytest

from nebulai import spaces
from nebulai.backend.channels import (
    CHANNELS_FILENAME,
    Channel,
    ChannelError,
    channel_ids,
    channels_from_meta,
    drop_channels,
    find_channel,
    read_channels,
    validate,
    write_channels,
)
from nebulai.backend.export import public_meta


# ── spaces: the closed set ────────────────────────────────────────────────────


def test_bare_families_parse_and_round_trip():
    for tag in ("W_E.raw", "W_E.centered", "W_U.raw"):
        assert str(spaces.parse(tag)) == tag
        assert spaces.is_known(tag)


def test_layered_and_referenced_families_round_trip():
    for tag in (
        "resid.L0",
        "resid.L11",
        "mlp_out.L8",
        "sae.L8.gpt2-small-res-jb",
        "text-embed.sentence-transformers/all-MiniLM-L6-v2",
        "persona-pca.assistant-axis",
    ):
        assert str(spaces.parse(tag)) == tag
        assert spaces.is_known(tag)


@pytest.mark.parametrize(
    "tag",
    [
        "",
        "W_E",
        "W_E.raw.extra",
        "resid",
        "resid.14",  # layers are spelled L14; a bare number is not a space
        "resid.Lx",
        "mlp_out",
        "sae.L8",  # an SAE without its dictionary is not a space
        "sae.8.repo",
        "text-embed",
        "logits",
        "attn_out.L3",
    ],
)
def test_unknown_tags_are_refused_not_guessed(tag):
    assert not spaces.is_known(tag)
    with pytest.raises(spaces.UnknownSpaceError):
        spaces.parse(tag)


def test_raw_and_centered_embeddings_are_never_comparable():
    """D2. The translation between them is the glitch experiment's subject."""
    assert not spaces.compatible("W_E.raw", "W_E.centered")
    assert spaces.refusal_reason("W_E.raw", "W_E.centered")
    assert spaces.compatible("W_E.raw", "W_E.raw")
    assert spaces.refusal_reason("W_E.raw", "W_E.raw") is None


def test_different_layers_and_different_dictionaries_are_different_spaces():
    assert not spaces.compatible("resid.L11", "resid.L12")
    assert not spaces.compatible("resid.L11", "mlp_out.L11")
    assert not spaces.compatible("sae.L8.res-jb", "sae.L8.other-sae")


def test_unknown_space_is_comparable_with_nothing_including_itself():
    assert not spaces.compatible("attn.L3", "attn.L3")
    assert "unknown" in (spaces.refusal_reason("attn.L3", "W_E.raw") or "")


def test_only_text_embed_is_foreign_geometry():
    """R7: foreign data wears foreign clothes, so the viewer must be able to ask."""
    assert not spaces.parse("text-embed.all-MiniLM-L6-v2").model_internal
    for tag in ("W_E.raw", "resid.L4", "sae.L8.repo", "persona-pca.axis"):
        assert spaces.parse(tag).model_internal


def test_helpers_build_the_same_tags_the_parser_accepts():
    assert str(spaces.we_space(centered=False)) == "W_E.raw"
    assert str(spaces.we_space(centered=True)) == "W_E.centered"
    assert str(spaces.resid(7)) == "resid.L7"


# ── channels: construction ────────────────────────────────────────────────────


def _chan(**kw) -> Channel:
    base = dict(
        id="we_norm",
        label="‖W_E row‖",
        space="W_E.raw",
        method="l2_norm",
        formula="||W_E[i]||_2",
        values=[1.0, 2.0, 3.0],
    )
    base.update(kw)
    return Channel(**base)


def test_channel_refuses_a_space_outside_the_closed_set():
    with pytest.raises(spaces.UnknownSpaceError):
        _chan(space="embedding")


def test_channel_refuses_an_unknown_fidelity():
    with pytest.raises(ChannelError):
        _chan(fidelity="probably")


def test_channel_refuses_an_empty_id():
    with pytest.raises(ChannelError):
        _chan(id="")


def test_stats_ignore_missing_and_count_them():
    ch = _chan(values=[1.0, np.nan, 3.0])
    s = ch.stats()
    assert s["min"] == 1.0 and s["max"] == 3.0 and s["mean"] == 2.0
    assert s["n_missing"] == 1


def test_stats_of_an_entirely_unmeasured_channel_are_not_zero():
    s = _chan(values=[np.nan, np.nan]).stats()
    assert s["n_missing"] == 2
    for k in ("min", "max", "mean"):
        assert np.isnan(s[k]), f"{k} must stay NaN, not collapse to 0"


def test_missing_serialises_as_null_never_as_zero():
    doc = _chan(values=[1.5, np.nan, 3.0]).to_json()
    assert doc["values"] == [1.5, None, 3.0]
    assert doc["stats"]["n_missing"] == 1
    # and the whole object must survive a JSON round trip: bare NaN is not JSON
    assert json.loads(json.dumps(doc))["values"][1] is None


# ── channels: validation ──────────────────────────────────────────────────────


def test_short_channel_is_refused_rather_than_padded():
    with pytest.raises(ChannelError, match="index-shifted"):
        validate([_chan(values=[1.0, 2.0])], n_points=3)


def test_duplicate_ids_are_refused():
    with pytest.raises(ChannelError, match="duplicate"):
        validate([_chan(), _chan()], n_points=3)


# ── channels: the file ────────────────────────────────────────────────────────


def test_write_read_round_trip(tmp_path):
    p = write_channels(
        tmp_path / CHANNELS_FILENAME,
        model="gpt2",
        revision="abc123",
        n_points=3,
        channels=[_chan()],
    )
    doc = read_channels(p)
    assert doc is not None
    assert doc["meta"]["model"] == "gpt2"
    assert doc["meta"]["revision"] == "abc123"
    assert doc["meta"]["n_points"] == 3
    assert channel_ids(doc) == ["we_norm"]
    assert find_channel(doc, "we_norm")["space"] == "W_E.raw"
    assert find_channel(doc, "nope") is None


def test_absent_and_malformed_files_read_as_absent(tmp_path):
    assert read_channels(tmp_path / "nothing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert read_channels(bad) is None
    notdoc = tmp_path / "arr.json"
    notdoc.write_text("[1,2,3]")
    assert read_channels(notdoc) is None


def test_merge_keeps_other_channels_and_replaces_by_id(tmp_path):
    """`direction project` appends beside the glitch lens without a rebuild."""
    p = tmp_path / CHANNELS_FILENAME
    write_channels(p, model="gpt2", revision="r", n_points=3, channels=[_chan()])
    write_channels(
        p,
        model="gpt2",
        revision="r",
        n_points=3,
        channels=[_chan(id="axis_refusal", values=[9.0, 9.0, 9.0])],
    )
    assert sorted(channel_ids(read_channels(p))) == ["axis_refusal", "we_norm"]

    write_channels(
        p, model="gpt2", revision="r", n_points=3, channels=[_chan(values=[7.0, 7.0, 7.0])]
    )
    doc = read_channels(p)
    assert sorted(channel_ids(doc)) == ["axis_refusal", "we_norm"]
    assert find_channel(doc, "we_norm")["values"] == [7.0, 7.0, 7.0]


def test_merge_refuses_channels_aligned_to_a_different_map(tmp_path):
    p = tmp_path / CHANNELS_FILENAME
    write_channels(p, model="gpt2", revision="r", n_points=3, channels=[_chan()])
    # same file, different point count: the old channel is aligned to a map that
    # is no longer there, so it must not survive
    write_channels(
        p,
        model="gpt2",
        revision="r",
        n_points=2,
        channels=[_chan(id="other", values=[1.0, 2.0])],
    )
    assert channel_ids(read_channels(p)) == ["other"]


def test_drop_channels(tmp_path):
    p = tmp_path / CHANNELS_FILENAME
    write_channels(
        p,
        model="gpt2",
        revision="r",
        n_points=3,
        channels=[_chan(), _chan(id="we_centroid_dist")],
    )
    assert drop_channels(p, ["we_norm"]) == 1
    assert channel_ids(read_channels(p)) == ["we_centroid_dist"]
    assert drop_channels(tmp_path / "absent.json", ["x"]) == 0


# ── the front-end → CLI side channel ──────────────────────────────────────────


def test_channels_from_meta_reads_the_underscore_side_channel():
    meta = {
        "_channels": [
            {
                "id": "we_norm",
                "label": "raw ‖W_E row‖",
                "space": "W_E.raw",
                "method": "l2_norm",
                "formula": "||W_E[i]||_2",
                "units": "l2",
                "values": [1.0, 2.0],
            }
        ]
    }
    chans = channels_from_meta(meta)
    assert [c.id for c in chans] == ["we_norm"]
    assert chans[0].units == "l2"
    assert chans[0].fidelity == "deterministic"


def test_a_front_end_with_no_channels_yields_none_not_zeros():
    assert channels_from_meta({}) == []
    assert channels_from_meta({"_channels": []}) == []


def test_public_meta_strips_the_side_channel_from_the_export():
    """50,000 floats must never land in `nebulai.json`; its schema is unchanged."""
    meta = {"model": "gpt2", "_channels": [{"id": "we_norm", "values": [0.0] * 50_000}]}
    out = public_meta(meta)
    assert out == {"model": "gpt2"}
    assert "_channels" not in out


# ── the real artifacts, when they are present ─────────────────────────────────


def test_shipped_gpt2_channels_are_aligned_and_honest():
    """Skipped on a checkout without `out/` — asserted hard when it is there.

    This is the only test that looks at the real map, and it exists because the
    numbers in the SolidGoldMagikarp episode are quoted from this file.
    """
    from pathlib import Path

    root = Path("out/gpt2")
    if not (root / CHANNELS_FILENAME).exists() or not (root / "nebulai.json").exists():
        pytest.skip("out/gpt2 not present in this checkout")
    doc = read_channels(root / CHANNELS_FILENAME)
    assert doc is not None
    n = doc["meta"]["n_points"]
    mapdoc = json.loads((root / "nebulai.json").read_text())
    assert n == len(mapdoc["points"]), "channels must be aligned to the map's points"
    assert set(channel_ids(doc)) == {"we_norm", "we_centroid_dist"}
    for ch in doc["channels"]:
        assert len(ch["values"]) == n
        assert ch["space"] == "W_E.raw", "the map is centred; these numbers are not"
        assert ch["fidelity"] == "deterministic"
        assert ch["stats"]["n_missing"] == 0
    cd = find_channel(doc, "we_centroid_dist")["values"]
    labels = [p["label"] for p in mapdoc["points"]]
    nearest = sorted(range(n), key=lambda i: cd[i])[:10]
    knot = {labels[i] for i in nearest}
    # the ten nearest the RAW centroid are all in the glitch cluster; these four
    # are quoted in the episode copy
    for tok in (" externalToEVA", "quickShip", " TheNitrome", " RandomRedditor"):
        assert tok in knot, f"{tok!r} left the near-centroid knot"
