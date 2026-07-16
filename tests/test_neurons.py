"""MLP-neuron front-end pure-function tests — the honesty-contract string
builders, subset/label merging, and runtime orientation verification. No
network, no safetensors, no HF: everything here imports and runs offline."""

import argparse

import numpy as np
import pytest

from nebulai.frontends.neurons import (
    labels_for,
    model_tag_for,
    neuron_dataset_id,
    neuron_tensor_path,
    neuron_unit_string,
    orient_neuron_rows,
    subset_indices,
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
