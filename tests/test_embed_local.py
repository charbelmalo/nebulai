"""`--embed-api local`: the in-process, SHA-pinned neutral space.

`backend/embed.py`'s docstring states the rule these tests defend: the semantic
space a cross-model comparison is measured in must never be substituted
quietly. `local` exists because the LAN embedder that produced every previous
`out/compare/compare.json` was unreachable (`running: false`,
`state: insufficient_ram`) — not because an in-process encoder is nicer. So
what is pinned here is not the encoder's quality, it is the impossibility of
getting a *different* encoder without saying so.

Nothing in this file loads a model: the pin resolver and the api dispatch are
pure, and the one test that would need weights is a fake.
"""

import numpy as np
import pytest

from nebulai.backend import embed as embed_mod
from nebulai.backend.embed import (
    LOCAL_EMBED_HOST,
    LOCAL_EMBED_PINS,
    embed_texts,
    resolve_local_embed_model,
)

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_SHA = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


# --- the pin table ---------------------------------------------------------


def test_the_shipped_artifacts_embedder_name_resolves_to_a_commit():
    """`out/compare/compare.json` records `embed_model: "all-MiniLM-L6-v2"` and
    nothing else. That bare name has to land on an exact commit, or a rebuild
    is a different space wearing the same label."""
    assert resolve_local_embed_model("all-MiniLM-L6-v2") == (MINILM, MINILM_SHA)
    assert resolve_local_embed_model(MINILM) == (MINILM, MINILM_SHA)


def test_every_pin_is_a_full_commit_sha():
    for name, (repo, rev) in LOCAL_EMBED_PINS.items():
        assert "/" in repo, f"{name}: {repo!r} is not a repo id"
        assert len(rev) == 40, f"{name}: {rev!r} is not a commit sha"
        assert all(c in "0123456789abcdef" for c in rev), name


def test_an_unpinned_id_is_refused_not_resolved_to_main():
    """The failure mode this prevents: `compare` silently re-embedding in
    whatever the repo's default branch holds today, so two runs a month apart
    produce incomparable coordinates and nothing records why."""
    with pytest.raises(ValueError) as e:
        resolve_local_embed_model("BAAI/bge-small-en-v1.5")
    assert "not pinned" in str(e.value)
    assert "main" in str(e.value)


@pytest.mark.parametrize("moving", ["repo/x@main", "repo/x@v1.0", "repo/x@abc123"])
def test_a_branch_or_short_sha_is_not_a_pin(moving):
    with pytest.raises(ValueError) as e:
        resolve_local_embed_model(moving)
    assert "40-hex" in str(e.value)


def test_an_explicit_full_sha_is_accepted_for_an_unlisted_repo():
    sha = "0" * 39 + "f"
    assert resolve_local_embed_model(f"some/encoder@{sha.upper()}") == ("some/encoder", sha)


# --- dispatch --------------------------------------------------------------


def test_local_never_touches_a_socket(monkeypatch):
    """`_embed_batch` is the only door to the network in this module. If
    api="local" ever reaches it, the run is quietly measuring in whatever the
    default host serves."""

    def boom(*a, **k):  # pragma: no cover - the assertion is that it never runs
        raise AssertionError("api='local' made a network call")

    monkeypatch.setattr(embed_mod, "_embed_batch", boom)
    monkeypatch.setattr(
        embed_mod, "_embed_local", lambda texts, model, bs: np.ones((len(texts), 4), np.float32)
    )
    out = embed_texts(["a", "b"], api="local", model="all-MiniLM-L6-v2")
    assert out.shape == (2, 4)


def test_local_output_is_l2_normalized(monkeypatch):
    """Every other api normalizes on the way out; `local` returns a normalized
    array already, so the extra pass has to be a no-op rather than a second
    division that quietly shrinks nothing. Feed it unnormalized rows to prove
    the guarantee holds regardless."""
    monkeypatch.setattr(
        embed_mod,
        "_embed_local",
        lambda texts, model, bs: np.array([[3.0, 4.0], [0.0, 2.0]], np.float32),
    )
    out = embed_texts(["a", "b"], api="local", model="all-MiniLM-L6-v2")
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_local_ignores_the_host_it_is_given(monkeypatch):
    seen = {}

    def fake(texts, model, bs):
        seen["model"] = model
        return np.ones((len(texts), 2), np.float32)

    monkeypatch.setattr(embed_mod, "_embed_local", fake)
    embed_texts(["a"], host="http://192.168.0.107:11435", api="local", model=MINILM)
    assert seen["model"] == MINILM


def test_an_unknown_api_still_names_all_three():
    with pytest.raises(ValueError) as e:
        embed_texts(["a"], api="cohere")
    msg = str(e.value)
    assert "ollama" in msg and "openai" in msg and "local" in msg


def test_the_recorded_endpoint_is_not_a_lan_address():
    """An in-process run must not stamp a host into the artifact — there was
    none, and a LAN address in a shipped file is both false and an infra leak
    (see the sanitization rule in docs/DEPLOY-STATIC.md)."""
    assert LOCAL_EMBED_HOST == "local:in-process"
    assert "192.168." not in LOCAL_EMBED_HOST
