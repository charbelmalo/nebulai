"""When an `api_text_embedding` map may be validated, and when it may not.

`backend/validate.py` refused this whole unit type, and the refusal's stated
reason was never "it is an api map" — it was that the vectors came from a live
service and `meta` cannot bring them back. That reason is now false for exactly
one case: an encoder that ran **in this process at a pinned commit**. Repo plus
40-hex sha plus the fp32 CPU path is as replayable as a weight matrix, and both
strings are stamped into the map.

So the pair of tests that matter are the two sides of that line. A hosted map
must still be refused — with a reason a reader can act on — and a pinned local
map must reload rather than raise. Getting this wrong in the permissive
direction is the dangerous one: it would score a point set that is not the one
on screen, under a model name that happens to match.
"""

import numpy as np
import pytest

from nebulai.backend import validate as V
from nebulai.units import Units

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
SHA = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def _meta(**over) -> dict:
    m = {
        "model": "gpt2",
        "unit": f"api_text_embedding({MINILM})",
        "embed_model": MINILM,
        "embed_api": "local",
        "embed_revision": SHA,
        "embed_host": "local:in-process",
        "centered": True,
        "kept": 12,
    }
    m.update(over)
    return m


# --- the refusal, where it still applies -----------------------------------


@pytest.mark.parametrize(
    "over",
    [
        {"embed_api": "ollama", "embed_host": "http://<lan>", "embed_revision": ""},
        {"embed_api": "openai", "embed_revision": ""},
        # `local` is not enough on its own: without a commit there is nothing
        # to replay, and a name like "all-MiniLM-L6-v2" is not a version.
        {"embed_revision": ""},
        {"embed_revision": "main"},
        {"embed_revision": "c9745ed"},  # short sha: still ambiguous
    ],
    ids=["ollama", "openai", "local-no-rev", "local-branch", "local-short-sha"],
)
def test_an_unreproducible_api_map_is_still_refused(over):
    with pytest.raises(ValueError) as e:
        V.reload_units(_meta(**over))
    msg = str(e.value)
    assert "cannot be revalidated" in msg
    # the refusal has to tell the reader what WOULD work
    assert "--embed-api local" in msg


def test_the_refusal_is_not_silently_a_zero():
    """A raise, not a `None` or an empty Units — a caller that mistook either
    for success would publish a validation row for a map nobody scored."""
    with pytest.raises(ValueError):
        V.reload_units(_meta(embed_api="ollama", embed_revision=""))


# --- the reload, where it now applies --------------------------------------


def test_a_commit_pinned_local_map_reloads_instead_of_raising(monkeypatch):
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return Units(
            ids=list(range(12)),
            vectors=np.zeros((12, 4), np.float32),
            labels=[str(i) for i in range(12)],
            meta={},
        )

    import nebulai.frontends.api_tokens as api_tokens

    monkeypatch.setattr(api_tokens, "load_api_token_units", fake)
    units = V.reload_units(_meta(), out_root="out")
    assert len(units) == 12
    # The display name goes back unchanged — it keys this map's own directory
    # and its embed cache, so rewriting it would fork a second directory and
    # re-embed the whole vocabulary. The COMMIT travels beside it: replaying by
    # name alone would resolve through the pin table again and quietly track
    # whatever that table says today, not what this map was built with.
    assert seen["embed_model"] == MINILM
    assert seen["embed_revision"] == SHA
    assert seen["api"] == "local"
    assert seen["max_tokens"] == 12
    assert seen["center"] is True


def test_the_reload_reads_the_cache_beside_the_map_not_the_cwd(monkeypatch):
    """`validate` may be pointed at any `--out`. The embed cache lives under
    that root, so a reload that assumed `./out` would recompute 50k vectors —
    or worse, find a cache belonging to a different corpus."""
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return Units(
            ids=list(range(12)),
            vectors=np.zeros((12, 4), np.float32),
            labels=[str(i) for i in range(12)],
            meta={},
        )

    import nebulai.frontends.api_tokens as api_tokens

    monkeypatch.setattr(api_tokens, "load_api_token_units", fake)
    V.reload_units(_meta(), out_root="/tmp/some-other-corpus")
    assert str(seen["out_root"]) == "/tmp/some-other-corpus"


# --- the sha predicate itself ----------------------------------------------


@pytest.mark.parametrize("good", [SHA, SHA.upper()])
def test_is_sha_accepts_only_a_full_commit(good):
    assert V._is_sha(good)


@pytest.mark.parametrize("bad", ["", None, "main", "refs/heads/main", SHA[:39], SHA + "a", "g" * 40])
def test_is_sha_rejects_anything_that_can_move(bad):
    assert not V._is_sha(bad)


def test_local_repo_strips_a_sha_suffix_without_losing_the_org():
    assert V._local_repo({"embed_model": f"{MINILM}@{SHA}"}) == MINILM
    assert V._local_repo({"embed_model": MINILM}) == MINILM
    assert V._local_repo({}) == ""


# --- the frontend's half of the contract -----------------------------------


def test_a_revision_is_refused_for_a_hosted_encoder():
    """A hosted endpoint serves whatever it is currently running. Accepting a
    revision there would stamp a commit the run never honoured, which is worse
    than no commit: `validate` reads that field as permission to replay."""
    from nebulai.frontends.api_tokens import load_api_token_units

    with pytest.raises(ValueError) as e:
        load_api_token_units(
            model_id="gpt2",
            api="ollama",
            embed_model="mxbai-embed-large",
            embed_revision=SHA,
        )
    msg = str(e.value)
    assert "in-process" in msg
    assert "reproducible" in msg


def test_the_pin_is_settled_before_anything_touches_the_network(monkeypatch):
    """The refusal above must fire on argument validation, not after a
    tokenizer download — otherwise a typo'd pin costs a round trip."""
    import nebulai.frontends.tokens as tokens_mod
    from nebulai.frontends.api_tokens import load_api_token_units

    def boom(*a, **kw):  # pragma: no cover - must never run
        raise AssertionError("curated_vocab was reached despite a bad pin")

    monkeypatch.setattr(tokens_mod, "curated_vocab", boom)
    with pytest.raises(ValueError):
        load_api_token_units(api="openai", embed_revision=SHA)


def test_the_dataset_id_keys_on_the_display_name_not_the_commit():
    """Two maps of the same vocabulary through the same pinned encoder must be
    one directory, not one per sha — the cache beside it is the expensive part."""
    from nebulai.frontends.api_tokens import api_dataset_id

    assert api_dataset_id("gpt2", "all-MiniLM-L6-v2") == "gpt2__api-all-MiniLM-L6-v2"
    assert "@" not in api_dataset_id("gpt2", "all-MiniLM-L6-v2")


def test_the_shipped_contrast_map_carries_a_pin_and_no_lan_address():
    """The map built for Track 2d, read back from disk: the two fields that
    decide whether it can ever be validated again."""
    import json
    from pathlib import Path

    p = Path("out/mistralai__Mistral-Nemo-Instruct-2407__api-all-MiniLM-L6-v2/nebulai.json")
    if not p.exists():
        pytest.skip("contrast map not built in this checkout")
    meta = json.loads(p.read_text())["meta"]
    assert meta["embed_api"] == "local"
    assert V._is_sha(meta["embed_revision"])
    assert "192.168." not in str(meta["embed_host"])
