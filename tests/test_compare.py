"""Comparison identity tests — three decompositions of ONE model (tokens /
SAE / neurons) must stay distinct clouds. They share `meta.model`, so keying
on it collapses them; identity comes from the front-end/unit label instead.
Pure + offline (the helpers don't touch the network or embedder)."""

from pathlib import Path

from nebulai.backend.compare import (
    _PALETTE,
    _site_tag,
    _source_label,
    _titles_are_placeholders,
    _unique_labels,
)


SOURCE_LABEL_CASES = [
    (
        "tokens",
        {"model": "HuggingFaceTB/SmolLM2-135M", "unit": "token_embedding"},
        "SmolLM2-135M · tokens",
    ),
    (
        "sae",
        {
            "model": "HuggingFaceTB/SmolLM2-135M",
            "unit": "sae_decoder(EleutherAI/sae-SmolLM2-135M-64x, layers.21.mlp)",
        },
        "SmolLM2-135M · SAE features L21",
    ),
    (
        "neurons",
        {
            "model": "HuggingFaceTB/SmolLM2-135M",
            "unit": "mlp_neuron(HuggingFaceTB/SmolLM2-135M, model.layers.21.mlp.down_proj)",
        },
        "SmolLM2-135M · MLP neurons L21",
    ),
    (
        "api-embeddings",
        {"model": "gpt2", "unit": "api_text_embedding(mxbai-embed-large)"},
        "gpt2 · API embeddings",
    ),
]


def test_source_label_distinguishes_frontends_of_one_model():
    labels = [_source_label(meta) for _id, meta, _exp in SOURCE_LABEL_CASES]
    for (_id, _meta, expected), got in zip(SOURCE_LABEL_CASES, labels):
        assert got == expected, _id
    # the SmolLM2 trio (first three) must be three DIFFERENT identities
    trio = labels[:3]
    assert len(set(trio)) == 3


def test_unique_labels_suffixes_collisions():
    assert _unique_labels(["a", "b", "a", "a", "b"]) == [
        "a",
        "b",
        "a #2",
        "a #3",
        "b #2",
    ]


def test_unique_labels_noop_when_distinct():
    trio = [
        "SmolLM2-135M · tokens",
        "SmolLM2-135M · SAE features L21",
        "SmolLM2-135M · MLP neurons L21",
    ]
    assert _unique_labels(trio) == trio


# --- palette --------------------------------------------------------------
# Colors are assigned `_PALETTE[i % len(_PALETTE)]`, so a palette shorter than
# the roster paints two clouds identically in the ONE view whose purpose is
# telling them apart — and it does so silently. This caught exactly that when
# the comparison grew from 4 maps to 8 against a 6-color palette.


def test_palette_colors_are_distinct():
    assert len({tuple(c) for c in _PALETTE}) == len(_PALETTE)


def test_palette_covers_every_built_map():
    out = Path(__file__).resolve().parents[1] / "out"
    if not out.is_dir():  # a fresh clone has no artifacts; nothing to guard
        return
    n_maps = sum(1 for d in out.iterdir() if (d / "nebulai.json").is_file())
    assert len(_PALETTE) >= n_maps, (
        f"{n_maps} maps in out/ but only {len(_PALETTE)} colors — "
        "comparing them all would reuse a color"
    )


def test_palette_channels_are_unit_range_floats():
    for c in _PALETTE:
        assert len(c) == 3
        assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in c)


# --- the site tag ---------------------------------------------------------
# A depth series is five maps of one model with one front-end. Before the site
# tag they all derived the same label and `_unique_labels` numbered them " #2"
# … " #5" — unique, and useless: the legend could not say which depth a cloud
# came from. These tests pin that the tag carries the layer and that it does
# not leak into the labels of front-ends that have no site.

DEPTH_SERIES = [
    "mlp_neuron(mistralai/Mistral-Nemo-Instruct-2407, model.layers.4.mlp.down_proj)",
    "mlp_neuron(mistralai/Mistral-Nemo-Instruct-2407, model.layers.12.mlp.down_proj)",
    "mlp_neuron(mistralai/Mistral-Nemo-Instruct-2407, model.layers.20.mlp.down_proj)",
    "mlp_neuron(mistralai/Mistral-Nemo-Instruct-2407, model.layers.28.mlp.down_proj)",
    "mlp_neuron(mistralai/Mistral-Nemo-Instruct-2407, model.layers.36.mlp.down_proj)",
]


def test_site_tag_reads_the_layer_index():
    assert _site_tag(DEPTH_SERIES[0]) == "L4"
    assert _site_tag(DEPTH_SERIES[-1]) == "L36"
    # gpt2's hook path spells the layer differently and still resolves
    assert _site_tag("mlp_neuron(gpt2, h.8.mlp.c_proj)") == "L8"
    assert _site_tag("sae_decoder(gpt2-small-res-jb, blocks.8.hook_resid_pre)") == "L8"


def test_site_tag_is_empty_when_the_unit_names_no_site():
    assert _site_tag("token_embedding") == ""
    assert _site_tag("api_text_embedding(mxbai-embed-large)") == ""
    assert _site_tag("probe_concept(mxbai-embed-large)") == ""


def test_site_tag_falls_back_to_the_path_when_no_number_is_in_it():
    # Never invent a layer number that is not in the unit string: an unnumbered
    # site is reported verbatim rather than as L0 or as nothing.
    assert _site_tag("mlp_neuron(org/m, mlp.down_proj)") == "mlp.down_proj"


def test_a_depth_series_derives_five_distinct_labels():
    labels = [
        _source_label({"model": "mistralai/Mistral-Nemo-Instruct-2407", "unit": u})
        for u in DEPTH_SERIES
    ]
    assert len(set(labels)) == 5, labels
    # and the distinction is the layer, not a collision counter
    assert labels == _unique_labels(labels)
    assert "#2" not in " ".join(labels)
    assert labels[0].endswith("MLP neurons L4")
    assert labels[-1].endswith("MLP neurons L36")


def test_token_and_api_labels_are_unchanged_by_the_site_tag():
    assert (
        _source_label({"model": "gpt2", "unit": "token_embedding"}) == "gpt2 · tokens"
    )
    assert (
        _source_label({"model": "gpt2", "unit": "api_text_embedding(mxbai-embed-large)"})
        == "gpt2 · API embeddings"
    )


# --- placeholder titles are not concepts ----------------------------------
# `--labels none` maps are titled "unlabeled neurons (cluster 7)" by
# name.placeholder_titles, and every such map uses the SAME string shape. The
# comparison's concept space is an embedding of those titles, so before this
# guard a five-layer depth series on one model reported pairwise concept
# overlaps of 0.5-0.625 -- a measurement of the placeholder generator. These
# tests pin the detector and the shape of the refusal.

PLACEHOLDERS = [
    "unlabeled neurons (cluster 0)",
    "unlabeled neurons (cluster 1)",
    "unlabeled neurons (cluster 14)",
]
REAL_TITLES = ["punctuation and brackets", "German inflections", "digits"]


def test_placeholder_titles_detected_from_the_namer_stamp():
    # the stamp is authoritative even when a title happens to look real
    assert _titles_are_placeholders(
        {"namer": "none(all-placeholder-labels)"}, REAL_TITLES
    )


def test_placeholder_titles_detected_from_the_title_shape():
    # fallback for artifacts built before the stamp existed
    assert _titles_are_placeholders({"namer": "claude-cli"}, PLACEHOLDERS)
    assert _titles_are_placeholders({}, PLACEHOLDERS)


def test_named_map_is_not_flagged():
    assert not _titles_are_placeholders({"namer": "claude-cli"}, REAL_TITLES)
    # one placeholder among real titles is not an all-placeholder map
    assert not _titles_are_placeholders(
        {"namer": "claude-cli"}, REAL_TITLES + PLACEHOLDERS[:1]
    )


def test_a_map_with_no_clusters_is_not_flagged_as_placeholder():
    # `all([])` is True, which would flag an empty map as unnamed and hide a
    # real bug behind an honesty message.
    assert not _titles_are_placeholders({"namer": "claude-cli"}, [])


def test_unnamed_pairs_are_null_not_zero(monkeypatch, tmp_path):
    """The end-to-end shape: a named map, two placeholder-titled maps, and the
    stats they produce. Pairs touching a placeholder map must be None, the
    named-vs-named pair must be a number, and the placeholder maps' clusters
    must not be counted as shared concepts with each other."""
    import json

    import numpy as np

    from nebulai.backend import compare as C

    def write(name: str, model: str, unit: str, titles: list[str], namer: str):
        pts, cls = [], []
        for i, t in enumerate(titles):
            for j in range(3):
                pts.append({"cluster_id": i, "label": f"{name}-u{i}-{j}"})
            cls.append(
                {
                    "id": i,
                    "title": t,
                    "size": 3,
                    "centroid": [float(i), float(i + 1), 0.0],
                }
            )
        d = tmp_path / name
        d.mkdir()
        (d / "nebulai.json").write_text(
            json.dumps(
                {
                    "meta": {"model": model, "unit": unit, "namer": namer},
                    "points": pts,
                    "clusters": cls,
                }
            )
        )
        return d / "nebulai.json"

    named = write(
        "named", "gpt2", "token_embedding",
        ["punctuation", "digits", "German", "verbs", "suffixes", "whitespace"],
        "claude-cli",
    )
    n4 = write(
        "n4", "org/m", "mlp_neuron(org/m, model.layers.4.mlp.down_proj)",
        [f"unlabeled neurons (cluster {i})" for i in range(6)],
        "none(all-placeholder-labels)",
    )
    n8 = write(
        "n8", "org/m", "mlp_neuron(org/m, model.layers.8.mlp.down_proj)",
        [f"unlabeled neurons (cluster {i})" for i in range(6)],
        "none(all-placeholder-labels)",
    )
    named2 = write(
        "named2", "distilgpt2", "token_embedding",
        ["punctuation", "digits", "French", "nouns", "prefixes", "newlines"],
        "claude-cli",
    )

    # a deterministic stand-in for the embedder: identical strings embed
    # identically, which is exactly the condition that made placeholder titles
    # collide in the first place.
    def fake_embed(texts, **kwargs):
        rng = np.random.default_rng(0)
        table: dict[str, np.ndarray] = {}
        out = []
        for t in texts:
            key = t.split(".")[0]
            if key not in table:
                table[key] = rng.normal(size=8).astype(np.float32)
            out.append(table[key])
        return np.asarray(out, dtype=np.float32)

    monkeypatch.setattr(C, "embed_texts", fake_embed)
    comp = C.build_comparison(
        [named, n4, n8, named2], embed_host="", embed_model="fake"
    )

    stats = comp["stats"]
    # four maps -> six pairs; five of them touch a placeholder-titled map
    assert len(stats["jaccard"]) == 6
    unmeasured = [k for k, v in stats["jaccard"].items() if v is None]
    assert len(unmeasured) == 5, stats["jaccard"]
    # the one pair of named maps still gets a number, so the guard refuses the
    # unmeasurable pairs rather than disabling the statistic
    measured = {k: v for k, v in stats["jaccard"].items() if v is not None}
    assert len(measured) == 1, measured
    (mk, mv), = measured.items()
    assert "MLP neurons" not in mk
    assert isinstance(mv, float)
    # `missing` is None, never 0.0 -- the two are different claims
    assert all(v is None or isinstance(v, float) for v in stats["jaccard"].values())
    for k in unmeasured:
        # spelled as an absence, so a reader (and the viewer) cannot mistake it
        # for "these two share no concepts"
        assert stats["jaccard"][k] is None
    assert sorted(stats["unnamed_models"]) == sorted(
        [m for m in comp["meta"]["models"] if "MLP neurons" in m]
    )
    assert "not measured, not zero" in stats["unnamed_reason"]
    # two placeholder maps landing in one meta-cluster is not a shared concept
    for mc in comp["meta_clusters"]:
        if mc["shared"]:
            assert mc["n_models_named"] > 1, mc
