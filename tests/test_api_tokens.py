"""Phase-2 front-end tests: curated_vocab parity (the refactor must be
byte-identical to the old inline loop) and the OpenAI-compatible embed
transport parsing. Network-free except the HF tokenizer download (cached)."""

import numpy as np
import pytest

from nebulai.backend.embed import embed_texts, parse_embed_response
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


def test_api_dataset_id():
    assert api_dataset_id("gpt2", "mxbai-embed-large") == "gpt2__api-mxbai-embed-large"
    assert (
        api_dataset_id("EleutherAI/pythia-70m", "org/embedder")
        == "EleutherAI__pythia-70m__api-org__embedder"
    )
