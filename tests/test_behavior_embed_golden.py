"""The semantic judge's golden vectors (BEHAVIORAL-DIVERGENCE-PLAN.md §6.3.1).

Every number the Behavior study reports — Δ̂, the MMD, the permutation p, the
cue landscape — is a distance measured in ONE encoder's space. If that encoder
changes, nothing throws: the study just quietly means something else, and two
analyses of the same raw trials disagree with no error and no diff. The
manifest pins `embedder_id` + `embedder_sha` so a change is *recordable*; this
file is what makes it *detectable*.

TOLERANCE, stated rather than tuned: fp32 CPU inference is not bit-reproducible
across BLAS builds, thread counts or torch versions, so exact equality would
make this a flaky test that gets deleted rather than a guard that gets fixed.
The vectors are compared at `atol=2e-3` on L2-normalized components, and the
full 11x11 Gram matrix at `atol=2e-3` on cosines in [-1, 1]. A different
checkpoint, a different pooling, a different precision (fp16/int8) or a
different repo all move cosines by far more than that — several 1e-2 at least,
and usually 1e-1 for a different model. Drift within the tolerance does not
change any reported conclusion; drift outside it means the study's units moved.

`row_sha256` is recorded but deliberately NOT asserted: a checksum over floats
cannot survive the tolerance above. It is there so a human comparing two
machines can see at a glance whether the arithmetic was identical or merely
close.

Skipped when the optional `behavior-local` group is absent — the base install
is torch-free by design (tests/test_behavior_optional_dep.py pins that), so
this test must not be the thing that drags torch into it.
"""

import json
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).with_name("fixtures") / "behavior_embedder_golden.json"

pytest.importorskip("sentence_transformers", reason="optional 'behavior-local' group")
pytest.importorskip("torch", reason="optional 'behavior-local' group")

ATOL = 2e-3


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def vectors(golden) -> np.ndarray:
    from nebulai.behavior.embed import LocalSentenceEmbedder

    return LocalSentenceEmbedder().encode(list(golden["texts"]))


def test_the_golden_file_describes_the_stack_it_was_recorded_on(golden):
    fp = golden["fingerprint"]
    assert fp["embedder_id"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert len(fp["embedder_sha"]) == 40
    assert fp["embedder_dtype"] == "float32"
    # the versions are recorded, not asserted: a torch upgrade is allowed to
    # move the last bits, and the tolerance below is what decides whether it
    # moved too far
    for k in ("torch", "transformers", "sentence_transformers"):
        assert fp[k], f"{k} missing from the recorded fingerprint"


def test_shape_and_normalization(golden, vectors):
    assert vectors.shape == (len(golden["texts"]), golden["dim"])
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_each_vector_matches_its_golden_head(golden, vectors):
    for i, (text, head) in enumerate(zip(golden["texts"], golden["head8"])):
        got = vectors[i, : len(head)]
        assert np.allclose(got, head, atol=ATOL), (
            f"row {i} ({text!r}) drifted: {got.tolist()} vs golden {head}"
        )


def test_the_whole_distance_structure_is_reproduced(golden, vectors):
    """The head of a vector is a spot check; the Gram matrix is the thing the
    study actually consumes. Every pairwise cosine, including the ones the
    study's rank-weighted trial vectors are built from."""
    gram = vectors @ vectors.T
    assert np.allclose(gram, np.asarray(golden["gram"], dtype=np.float64), atol=ATOL)


def test_the_empty_string_is_still_a_point(golden, vectors):
    """`""` reaches the encoder whenever a model returns no parseable
    associate. It must produce a finite, normalized vector rather than NaN —
    a NaN here propagates into the MMD and takes every cue down with it."""
    i = golden["texts"].index("")
    assert np.all(np.isfinite(vectors[i]))
    assert abs(float(np.linalg.norm(vectors[i])) - 1.0) < 1e-5


def test_the_space_is_semantic_not_lexical(golden, vectors):
    """A sanity floor that no tolerance can paper over: if this ever inverts,
    the encoder that loaded is not the one that was pinned."""
    t = golden["texts"]
    v = vectors
    near = float(v[t.index("father")] @ v[t.index("dad")])
    far = float(v[t.index("father")] @ v[t.index("quantum chromodynamics")])
    assert near > 0.5, near
    assert far < 0.2, far
    assert near - far > 0.4
