"""Comparison identity tests — three decompositions of ONE model (tokens /
SAE / neurons) must stay distinct clouds. They share `meta.model`, so keying
on it collapses them; identity comes from the front-end/unit label instead.
Pure + offline (the helpers don't touch the network or embedder)."""

from pathlib import Path

from nebulai.backend.compare import _PALETTE, _source_label, _unique_labels


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
        "SmolLM2-135M · SAE features",
    ),
    (
        "neurons",
        {
            "model": "HuggingFaceTB/SmolLM2-135M",
            "unit": "mlp_neuron(HuggingFaceTB/SmolLM2-135M, model.layers.21.mlp.down_proj)",
        },
        "SmolLM2-135M · MLP neurons",
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
        "SmolLM2-135M · SAE features",
        "SmolLM2-135M · MLP neurons",
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
