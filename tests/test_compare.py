"""Comparison identity tests — three decompositions of ONE model (tokens /
SAE / neurons) must stay distinct clouds. They share `meta.model`, so keying
on it collapses them; identity comes from the front-end/unit label instead.
Pure + offline (the helpers don't touch the network or embedder)."""

from nebulai.backend.compare import _source_label, _unique_labels


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
