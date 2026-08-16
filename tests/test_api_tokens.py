"""Phase-2 front-end tests: curated_vocab parity (the refactor must be
byte-identical to the old inline loop) and the OpenAI-compatible embed
transport parsing. Network-free except the HF tokenizer download (cached)."""

import numpy as np
import pytest

from nebulai.backend.embed import (
    EMBED_HOST_ENV,
    PUBLISHED_REMOTE_HOST,
    default_embed_host,
    embed_texts,
    parse_embed_response,
    public_embed_host,
)
from nebulai.frontends import api_tokens as api_tokens_mod
from nebulai.frontends.api_tokens import api_dataset_id
from nebulai.frontends.tokens import _keep, curated_vocab


# --- curated_vocab parity -------------------------------------------------


def _old_inline_curation(model_id: str, max_tokens: int | None, n_vocab: int | None):
    """The exact loop load_token_units used before the extraction."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained(model_id)
    n = tok.get_vocab_size() if n_vocab is None else min(n_vocab, tok.get_vocab_size())
    ids, labels = [], []
    for i in range(n):
        s = tok.decode([i])
        if _keep(s):
            ids.append(i)
            labels.append(s)
    if max_tokens is not None and len(ids) > max_tokens:
        ids = ids[:max_tokens]
        labels = labels[:max_tokens]
    return ids, labels


@pytest.mark.parametrize("max_tokens,n_vocab", [(200, None), (None, 3000), (50, 3000)])
def test_curated_vocab_parity(max_tokens, n_vocab):
    got = curated_vocab("gpt2", max_tokens, n_vocab=n_vocab)
    assert got == _old_inline_curation("gpt2", max_tokens, n_vocab)


def test_curated_vocab_drops_junk():
    ids, labels = curated_vocab("gpt2", n_vocab=300)
    assert len(ids) == len(labels)
    assert all(_keep(s) for s in labels)
    assert all(s.strip() for s in labels)


# --- transport parsing ----------------------------------------------------


def test_parse_embed_response_ollama():
    payload = {"embeddings": [[1.0, 0.0], [0.0, 1.0]]}
    assert parse_embed_response(payload, "ollama", "h", "m") == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]


def test_parse_embed_response_openai_sorts_by_index():
    payload = {
        "data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ]
    }
    assert parse_embed_response(payload, "openai", "h", "m") == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]


@pytest.mark.parametrize(
    "api,payload", [("ollama", {}), ("openai", {}), ("openai", {"data": []})]
)
def test_parse_embed_response_empty_raises(api, payload):
    with pytest.raises(RuntimeError):
        parse_embed_response(payload, api, "h", "m")


def test_embed_texts_rejects_unknown_api():
    with pytest.raises(ValueError):
        embed_texts(["x"], api="carrier-pigeon")


def test_embed_texts_openai_transport(monkeypatch):
    """Full embed_texts path over a fake /v1/embeddings endpoint."""
    import io
    import json
    import urllib.request

    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["auth"] = req.headers.get("Authorization")
        body = json.loads(req.data)
        data = [
            {"index": i, "embedding": [float(i + 1), 0.0]}
            for i in range(len(body["input"]))
        ]
        resp = io.BytesIO(json.dumps({"data": data}).encode())
        resp.__enter__ = lambda: resp
        resp.__exit__ = lambda *a: False
        return resp

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    arr = embed_texts(
        ["a", "b", "c"],
        host="http://fake:9999/",
        model="text-embedding-3-small",
        batch_size=2,
        api="openai",
        api_key="sk-test",
    )
    assert seen["url"] == "http://fake:9999/v1/embeddings"
    assert seen["auth"] == "Bearer sk-test"
    assert arr.shape == (3, 2)
    assert arr.dtype == np.float32
    # rows are L2-normalized
    np.testing.assert_allclose(np.linalg.norm(arr, axis=1), 1.0, rtol=1e-5)


# --- embed host resolution ------------------------------------------------
#
# The env override exists because the working ollama on this network binds
# :11435, not the stock :11434 — a mismatch that got read as "no embedder
# exists" for a month. These pin the precedence so it cannot silently regress.


def test_default_embed_host_falls_back_to_local(monkeypatch):
    monkeypatch.delenv(EMBED_HOST_ENV, raising=False)
    assert default_embed_host() == "http://localhost:11434"


def test_default_embed_host_honors_env(monkeypatch):
    monkeypatch.setenv(EMBED_HOST_ENV, "http://lan-box:11435")
    assert default_embed_host() == "http://lan-box:11435"


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_default_embed_host_ignores_blank_env(monkeypatch, blank):
    """An exported-but-empty var must not resolve to "" and produce
    `/api/embed` with no host."""
    monkeypatch.setenv(EMBED_HOST_ENV, blank)
    assert default_embed_host() == "http://localhost:11434"


def test_embed_failure_message_names_the_fix(monkeypatch):
    """A dead endpoint must explain where to look — the bare URLError is what
    made a wrong port read as a missing embedder."""
    import urllib.error
    import urllib.request

    def boom(req, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError) as e:
        embed_texts(["x"], host="http://dead:11434", model="m", retries=1)
    msg = str(e.value)
    assert "11435" in msg  # the non-stock port that actually works here
    assert "M4-OLLAMA-HANDOVER" in msg  # where the answer is written down
    assert EMBED_HOST_ENV in msg  # how to point it elsewhere
    assert "/api/tags" in msg  # the one-line check to run


# --- exported host sanitation --------------------------------------------
#
# `nebulai.json` is served publicly. Five shipped artifacts stamped a LAN
# address into `meta.embed_host` before this existed, so these tests pin the
# rule in both directions: loopback survives (it is reproducible and names
# nobody), everything else — including anything unparseable — becomes the
# marker. The classifier must fail CLOSED; a wrong verdict publishes an IP.


@pytest.mark.parametrize(
    "host",
    [
        "http://localhost:11434",
        "http://localhost:11434/",
        "http://127.0.0.1:11434",
        "http://127.0.1.53:8050",  # anywhere in 127.0.0.0/8
        "http://[::1]:11434",
        "https://LocalHost:8050",  # case-insensitive
        "localhost:11434",  # scheme-less, as a careless caller might pass
    ],
)
def test_public_embed_host_keeps_loopback(host):
    assert public_embed_host(host) == host


@pytest.mark.parametrize(
    "host",
    [
        "http://192.168.0.200:11435",  # the address that actually leaked
        "http://192.168.0.200:11434",
        "http://192.168.0.200:8040",
        "http://10.0.0.5:11434",
        "http://m4worker.local:11435",
        "http://api.openai.com/v1",  # public, but still not stamped: fail closed
        "http://[fe80::1]:11434",
        "192.168.0.200:11435",  # scheme-less
        "not a url at all",
        "http://[oops:::]:11434",  # malformed IPv6 must not raise
    ],
)
def test_public_embed_host_marks_everything_else_remote(host):
    assert public_embed_host(host) == PUBLISHED_REMOTE_HOST


@pytest.mark.parametrize("empty", ["", "   ", None])
def test_public_embed_host_passes_through_empty(empty):
    """Absent stays absent — an empty host must not become "remote", which
    would claim an external service was used when none was named."""
    assert public_embed_host(empty) == ""


def test_public_embed_host_never_leaks_a_private_octet():
    """Belt and braces: whatever the marker is, the address must not survive
    anywhere inside it."""
    assert "192.168" not in public_embed_host("http://192.168.0.200:11435")


def test_api_token_meta_does_not_carry_the_raw_host(monkeypatch, tmp_path):
    """The end-to-end guarantee: build a map against a LAN host, and the
    exported meta must not contain it."""
    import json as _json

    import nebulai.frontends.api_tokens as at

    monkeypatch.setattr(
        at, "embed_texts", lambda texts, **kw: np.zeros((len(texts), 4), np.float32)
    )
    units = at.load_api_token_units(
        "gpt2",
        embed_host="http://192.168.0.200:11435",
        max_tokens=20,
        out_root=tmp_path,
    )
    assert units.meta["embed_host"] == PUBLISHED_REMOTE_HOST
    assert "192.168" not in _json.dumps(units.meta)
    # the evidential fields are untouched
    assert units.meta["embed_model"] == "mxbai-embed-large"
    assert units.meta["embed_api"] == "ollama"


def test_api_dataset_id():
    assert api_dataset_id("gpt2", "mxbai-embed-large") == "gpt2__api-mxbai-embed-large"
    assert (
        api_dataset_id("EleutherAI/pythia-70m", "org/embedder")
        == "EleutherAI__pythia-70m__api-org__embedder"
    )


def test_api_token_map_does_not_publish_its_embedding_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(
        api_tokens_mod,
        "embed_texts",
        lambda texts, **_kwargs: np.ones((len(texts), 2), dtype=np.float32),
    )
    units = api_tokens_mod.load_api_token_units(
        "gpt2",
        embed_host="http://private-worker.internal:8040/v1?key=secret",
        embed_model="fake-embedder",
        center=False,
        max_tokens=3,
        out_root=tmp_path,
        checkpoint_every=3,
    )

    assert units.meta["embed_host"] == "remote"
    assert "private-worker" not in str(units.meta)
    assert "secret" not in str(units.meta)
