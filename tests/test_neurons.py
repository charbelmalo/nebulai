"""MLP-neuron front-end pure-function tests — the honesty-contract string
builders, subset/label merging, and runtime orientation verification. No
network, no safetensors, no HF: everything here imports and runs offline."""

import argparse

import numpy as np
import pytest

from nebulai.frontends.neurons import (
    MoeSelectionRequired,
    labels_for,
    layer_down_proj_keys,
    mlp_dims,
    model_tag_for,
    neuron_dataset_id,
    neuron_tensor_path,
    neuron_unit_string,
    orient_neuron_rows,
    placeholder_titles,
    resolve_neuron_key,
    subset_indices,
    text_config,
)


# --- neuron_unit_string -----------------------------------------------------


def test_neuron_unit_string_exact():
    assert (
        neuron_unit_string("gpt2", "h.8.mlp.c_proj")
        == "mlp_neuron(gpt2, h.8.mlp.c_proj)"
    )


# --- model_tag_for ----------------------------------------------------------

MODEL_TAG_CASES = [
    # the canonical repo maps to the model tag it is true of
    ("canonical-repo-gets-tag", "openai-community/gpt2", "gpt2"),
    # a bare gpt2 already names itself
    ("bare-gpt2-names-itself", "gpt2", "gpt2"),
    # any other repo names itself — never claim a provenance it doesn't have
    ("other-repo-names-itself", "someone/custom-gpt2", "someone/custom-gpt2"),
]


@pytest.mark.parametrize(
    "_id,repo,expected", MODEL_TAG_CASES, ids=[c[0] for c in MODEL_TAG_CASES]
)
def test_model_tag_for(_id, repo, expected):
    assert model_tag_for(repo) == expected


# --- neuron_tensor_path -----------------------------------------------------

TENSOR_PATH_CASES = [
    ("layer-8", 8, "h.8.mlp.c_proj"),
    ("layer-0", 0, "h.0.mlp.c_proj"),
]


@pytest.mark.parametrize(
    "_id,layer,expected", TENSOR_PATH_CASES, ids=[c[0] for c in TENSOR_PATH_CASES]
)
def test_neuron_tensor_path(_id, layer, expected):
    assert neuron_tensor_path(layer) == expected


# --- neuron_dataset_id ------------------------------------------------------

DATASET_CASES = [
    (
        "plain",
        ("gpt2", "h.8.mlp.c_proj"),
        "gpt2__neurons__h.8.mlp.c_proj",
    ),
    (
        "slashed-model-tag",
        ("someone/custom", "h.8.mlp.c_proj"),
        "someone__custom__neurons__h.8.mlp.c_proj",
    ),
]


@pytest.mark.parametrize(
    "_id,args,expected", DATASET_CASES, ids=[c[0] for c in DATASET_CASES]
)
def test_neuron_dataset_id(_id, args, expected):
    assert neuron_dataset_id(*args) == expected


# --- subset_indices ---------------------------------------------------------

SUBSET_CASES = [
    ("first-n", (3072, 1024), list(range(1024))),
    ("n-exceeds-d_mlp-clamps", (100, 500), list(range(100))),
    ("none-takes-all", (10, None), list(range(10))),
    ("exact", (8, 8), list(range(8))),
]


@pytest.mark.parametrize(
    "_id,args,expected", SUBSET_CASES, ids=[c[0] for c in SUBSET_CASES]
)
def test_subset_indices(_id, args, expected):
    assert subset_indices(*args) == expected


# --- labels_for -------------------------------------------------------------

LABELS_CASES = [
    ("all-labeled", [0, 1], {0: "zero", 1: "one"}, ["zero", "one"]),
    ("all-unlabeled", [4, 5], {}, ["neuron 4 (unlabeled)", "neuron 5 (unlabeled)"]),
    (
        "mixed-coverage",
        [0, 1, 2],
        {1: "middle"},
        ["neuron 0 (unlabeled)", "middle", "neuron 2 (unlabeled)"],
    ),
    (
        "empty-string-desc-falls-through-to-placeholder",
        [0],
        {0: ""},
        ["neuron 0 (unlabeled)"],
    ),
]


@pytest.mark.parametrize(
    "_id,ids,desc,expected", LABELS_CASES, ids=[c[0] for c in LABELS_CASES]
)
def test_labels_for(_id, ids, desc, expected):
    assert labels_for(ids, desc) == expected


# --- orient_neuron_rows -----------------------------------------------------


def test_orient_neuron_rows_conv1d_layout_passthrough():
    W = np.arange(12, dtype=np.float32).reshape(6, 2)
    result = orient_neuron_rows(W, d_mlp=6, d_model=2)
    assert result.shape == (6, 2)
    assert np.array_equal(result, W)


def test_orient_neuron_rows_linear_layout_transposed():
    W = np.arange(12, dtype=np.float32).reshape(6, 2)
    result = orient_neuron_rows(W.T, d_mlp=6, d_model=2)
    assert result.shape == (6, 2)
    assert np.array_equal(result, W)


def test_orient_neuron_rows_bad_shape_raises():
    W = np.zeros((5, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        orient_neuron_rows(W, d_mlp=6, d_model=2)


def test_orient_neuron_rows_square_ambiguous_raises():
    W = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        orient_neuron_rows(W, d_mlp=4, d_model=4)


# --- CLI honesty: reserved label-space source -------------------------------


def test_run_neurons_label_source_reserved():
    from nebulai.cli import _run_neurons

    args = argparse.Namespace(source="label")
    with pytest.raises(SystemExit) as excinfo:
        _run_neurons(args)
    assert "label-space projection is not implemented" in str(excinfo.value)


# --- CLI honesty: all-placeholder labels must not be LLM-named ---------------


def test_placeholder_titles_are_honest_and_stamped():
    cluster_ids = np.array([2, -1, 0, 2, 0, -1, 5])
    titles, namer_used = placeholder_titles(cluster_ids)
    assert titles == {
        0: "unlabeled neurons (cluster 0)",
        2: "unlabeled neurons (cluster 2)",
        5: "unlabeled neurons (cluster 5)",
    }
    assert namer_used == "none(all-placeholder-labels)"


def test_placeholder_titles_all_noise_is_empty():
    titles, namer_used = placeholder_titles(np.array([-1, -1]))
    assert titles == {}
    assert namer_used == "none(all-placeholder-labels)"


# --- key resolution ---------------------------------------------------------
#
# The key lists below are the real layouts, taken from each repo's
# model.safetensors.index.json (read 2026-08-12) and trimmed to one layer.

GPT2_KEYS = [
    "wte.weight",
    "h.8.mlp.c_fc.weight",
    "h.8.mlp.c_proj.weight",
    "h.8.attn.c_proj.weight",  # attention output — NOT a neuron write matrix
]

# mistralai/Mistral-Nemo-Instruct-2407 — dense, flat
MISTRAL_KEYS = [
    "model.embed_tokens.weight",
    "model.layers.8.mlp.down_proj.weight",
    "model.layers.8.mlp.up_proj.weight",
    "model.layers.28.mlp.down_proj.weight",
]

# meta-models/Muse-Glimmer-30B — dense, nested under a multimodal wrapper
GLIMMER_KEYS = [
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.8.mlp.down_proj.weight",
    "model.visual.blocks.8.mlp.down_proj.weight",
]

# google/gemma-4-26b-a4b-it — 128 experts FUSED into one tensor per layer,
# alongside a plain dense mlp, plus a vision tower that also has layers.0
GEMMA_KEYS = [
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.0.experts.down_proj",
    "model.language_model.layers.0.experts.gate_up_proj",
    "model.language_model.layers.0.mlp.down_proj.weight",
    "model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight",
]

# inclusionAI/Ling-2.6-flash — indexed experts + a shared expert; layer 0 dense
LING_KEYS = (
    ["model.word_embeddings.weight", "model.layers.0.mlp.down_proj.weight"]
    + [f"model.layers.1.mlp.experts.{i}.down_proj.weight" for i in range(4)]
    + ["model.layers.1.mlp.shared_experts.down_proj.weight"]
)


def test_resolve_dense_gpt2_and_llama():
    # attn.c_proj shares the leaf name; only the MLP one is a write matrix
    assert resolve_neuron_key(GPT2_KEYS, 8).key == "h.8.mlp.c_proj.weight"
    assert (
        resolve_neuron_key(MISTRAL_KEYS, 8).key
        == "model.layers.8.mlp.down_proj.weight"
    )
    assert resolve_neuron_key(MISTRAL_KEYS, 8).kind == "dense"


def test_resolve_nested_language_model_key():
    """The exact key the old code built (`model.layers.8.mlp.down_proj`) does
    not exist in a multimodal wrapper — the text stack is nested."""
    src = resolve_neuron_key(GLIMMER_KEYS, 8)
    assert src.key == "model.language_model.layers.8.mlp.down_proj.weight"
    assert src.tensor_path == "model.language_model.layers.8.mlp.down_proj"


def test_vision_tower_is_never_the_language_mlp():
    assert "vision" not in resolve_neuron_key(GLIMMER_KEYS, 8).key
    assert layer_down_proj_keys(GEMMA_KEYS, 0) == [
        "model.language_model.layers.0.experts.down_proj",
        "model.language_model.layers.0.mlp.down_proj.weight",
    ]


def test_layer_segment_does_not_match_a_longer_number():
    keys = [
        "model.layers.2.mlp.down_proj.weight",
        "model.layers.20.mlp.down_proj.weight",
    ]
    assert resolve_neuron_key(keys, 2).key == "model.layers.2.mlp.down_proj.weight"
    assert resolve_neuron_key(keys, 20).key == "model.layers.20.mlp.down_proj.weight"


def test_missing_layer_raises_keyerror():
    with pytest.raises(KeyError):
        resolve_neuron_key(MISTRAL_KEYS, 3)


# --- MoE: refuse to pass one expert off as the layer -------------------------


def test_moe_layer_refuses_without_an_expert():
    """Ling layer 1 has 4 experts here (256 in the real checkpoint): 'the
    layer's neurons' is undefined until the caller chooses."""
    with pytest.raises(MoeSelectionRequired) as e:
        resolve_neuron_key(LING_KEYS, 1)
    msg = str(e.value)
    assert "--expert 0..3" in msg and "shared" in msg


def test_fused_expert_stack_also_refuses():
    with pytest.raises(MoeSelectionRequired) as e:
        resolve_neuron_key(GEMMA_KEYS, 0)
    assert "fused expert stack" in str(e.value)
    assert "--expert dense" in str(e.value)


def test_moe_dense_layer_still_resolves_without_an_expert():
    """Ling's first layers are dense — those need no choice."""
    assert resolve_neuron_key(LING_KEYS, 0).key == "model.layers.0.mlp.down_proj.weight"


MOE_SELECTION_CASES = [
    ("indexed-expert", LING_KEYS, 1, 2, "model.layers.1.mlp.experts.2.down_proj.weight", "expert", 2),
    (
        "shared-expert",
        LING_KEYS,
        1,
        "shared",
        "model.layers.1.mlp.shared_experts.down_proj.weight",
        "shared_expert",
        None,
    ),
    (
        "fused-expert-slice",
        GEMMA_KEYS,
        0,
        7,
        "model.language_model.layers.0.experts.down_proj",
        "fused_expert",
        7,
    ),
    (
        "dense-alongside-experts",
        GEMMA_KEYS,
        0,
        "dense",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "dense",
        None,
    ),
]


@pytest.mark.parametrize(
    "_id,keys,layer,expert,key,kind,idx",
    MOE_SELECTION_CASES,
    ids=[c[0] for c in MOE_SELECTION_CASES],
)
def test_moe_expert_selection(_id, keys, layer, expert, key, kind, idx):
    src = resolve_neuron_key(keys, layer, expert)
    assert (src.key, src.kind, src.expert) == (key, kind, idx)


def test_expert_index_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        resolve_neuron_key(LING_KEYS, 1, 99)


def test_expert_selector_rejects_garbage():
    with pytest.raises(ValueError, match="int, 'shared' or 'dense'"):
        resolve_neuron_key(LING_KEYS, 1, "seven")


def test_shared_selector_on_a_layer_without_one():
    with pytest.raises(ValueError, match="no shared expert"):
        resolve_neuron_key(GEMMA_KEYS, 0, "shared")


# --- config dims ------------------------------------------------------------


def test_text_config_unwraps_multimodal_wrappers():
    inner = {"hidden_size": 8}
    assert text_config({"text_config": inner, "model_type": "wrapper"}) is inner
    assert text_config(inner) is inner


def test_mlp_dims_uses_moe_width_for_expert_tensors():
    """A routed expert's down_proj is moe_intermediate_size wide; reading the
    dense width would make orient_neuron_rows pass for the wrong reason."""
    cfg = {
        "text_config": {
            "hidden_size": 2816,
            "intermediate_size": 2112,
            "moe_intermediate_size": 704,
            "num_hidden_layers": 30,
        }
    }
    assert mlp_dims(cfg, "dense") == ("llama", 2816, 2112, 30)
    assert mlp_dims(cfg, "fused_expert") == ("llama", 2816, 704, 30)
    assert mlp_dims(cfg, "expert")[2] == 704


def test_mlp_dims_gpt2_defaults_n_inner():
    assert mlp_dims({"n_embd": 768, "n_layer": 12}) == ("gpt2", 768, 3072, 12)


def test_mlp_dims_rejects_unknown_config():
    with pytest.raises(ValueError, match="cannot detect MLP architecture"):
        mlp_dims({"something_else": 1})
