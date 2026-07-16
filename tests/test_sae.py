"""SAE front-end pure-function tests — parsing Neuronpedia exports, label
merging, and the honesty-contract string builders. No network, no
safetensors, no HF: everything here imports and runs offline."""

import pytest

from nebulai.frontends.sae import (
    labels_for,
    parse_explanations,
    release_tag_for,
    sae_dataset_id,
    sae_unit_string,
    subset_indices,
)


# --- parse_explanations -----------------------------------------------------

PARSE_CASES = [
    # (id, lines, expected dict)
    (
        "string-index-coercion",
        ['{"index": "0", "description": "the digit zero"}'],
        {0: "the digit zero"},
    ),
    (
        "duplicate-index-first-wins",
        [
            '{"index": "5", "description": "first label"}',
            '{"index": "5", "description": "second label"}',
        ],
        {5: "first label"},
    ),
    (
        "missing-description-skipped",
        ['{"index": "7", "modelId": "gpt2-small"}'],
        {},
    ),
    (
        "empty-description-skipped",
        ['{"index": "8", "description": "   "}'],
        {},
    ),
    (
        "embedding-field-ignored",
        ['{"index": "9", "description": "kept", "embedding": "[0.1, 0.2, 0.3]"}'],
        {9: "kept"},
    ),
    (
        "blank-and-whitespace-lines-skipped",
        ["", "   ", '{"index": "1", "description": "one"}', "\t"],
        {1: "one"},
    ),
    (
        "description-is-stripped",
        ['{"index": "2", "description": "  spaced  "}'],
        {2: "spaced"},
    ),
    (
        "missing-index-skipped",
        ['{"description": "no index here"}'],
        {},
    ),
    (
        "multiple-distinct-indices",
        [
            '{"index": "0", "description": "a"}',
            '{"index": "10", "description": "b"}',
        ],
        {0: "a", 10: "b"},
    ),
]


@pytest.mark.parametrize("_id,lines,expected", PARSE_CASES, ids=[c[0] for c in PARSE_CASES])
def test_parse_explanations(_id, lines, expected):
    assert parse_explanations(lines) == expected


def test_parse_explanations_never_stores_embedding():
    lines = ['{"index": "3", "description": "d", "embedding": "[9.9]"}']
    result = parse_explanations(lines)
    assert result == {3: "d"}
    assert "embedding" not in str(result)


# --- labels_for -------------------------------------------------------------

LABELS_CASES = [
    ("all-labeled", [0, 1], {0: "zero", 1: "one"}, ["zero", "one"]),
    ("all-unlabeled", [4, 5], {}, ["feature 4 (unlabeled)", "feature 5 (unlabeled)"]),
    (
        "mixed-coverage",
        [0, 1, 2],
        {1: "middle"},
        ["feature 0 (unlabeled)", "middle", "feature 2 (unlabeled)"],
    ),
    (
        "empty-string-desc-falls-through-to-placeholder",
        [0],
        {0: ""},
        ["feature 0 (unlabeled)"],
    ),
]


@pytest.mark.parametrize(
    "_id,ids,desc,expected", LABELS_CASES, ids=[c[0] for c in LABELS_CASES]
)
def test_labels_for(_id, ids, desc, expected):
    assert labels_for(ids, desc) == expected


# --- sae_unit_string --------------------------------------------------------


def test_sae_unit_string_exact():
    assert (
        sae_unit_string("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
        == "sae_decoder(gpt2-small-res-jb, blocks.8.hook_resid_pre)"
    )


# --- release_tag_for --------------------------------------------------------

RELEASE_TAG_CASES = [
    # the canonical repo maps to the sae-lens release tag it is true of
    (
        "canonical-repo-gets-tag",
        "jbloom/GPT2-Small-SAEs-Reformatted",
        "gpt2-small-res-jb",
    ),
    # any other repo names itself — never claim a provenance it doesn't have
    (
        "other-repo-names-itself",
        "someone/other-saes",
        "someone/other-saes",
    ),
]


@pytest.mark.parametrize(
    "_id,release,expected", RELEASE_TAG_CASES, ids=[c[0] for c in RELEASE_TAG_CASES]
)
def test_release_tag_for(_id, release, expected):
    assert release_tag_for(release) == expected


# --- sae_dataset_id ---------------------------------------------------------

DATASET_CASES = [
    (
        "plain",
        ("gpt2-small", "blocks.8.hook_resid_pre"),
        "gpt2-small__sae__blocks.8.hook_resid_pre",
    ),
    (
        "slashed-sae-id",
        ("gpt2-small", "blocks/8/hook_resid_pre"),
        "gpt2-small__sae__blocks__8__hook_resid_pre",
    ),
]


@pytest.mark.parametrize(
    "_id,args,expected", DATASET_CASES, ids=[c[0] for c in DATASET_CASES]
)
def test_sae_dataset_id(_id, args, expected):
    assert sae_dataset_id(*args) == expected


# --- subset_indices ---------------------------------------------------------

SUBSET_CASES = [
    ("first-n", (24576, 4096), list(range(4096))),
    ("n-exceeds-d_sae-clamps", (100, 500), list(range(100))),
    ("none-takes-all", (10, None), list(range(10))),
    ("exact", (8, 8), list(range(8))),
]


@pytest.mark.parametrize(
    "_id,args,expected", SUBSET_CASES, ids=[c[0] for c in SUBSET_CASES]
)
def test_subset_indices(_id, args, expected):
    assert subset_indices(*args) == expected
