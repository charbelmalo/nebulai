"""Text embeddings via an ollama or OpenAI-compatible embeddings endpoint
(default: mxbai-embed-large on a local ollama server).

Used by the cross-model comparison (compare.py): each model's named clusters
are embedded in this neutral, model-independent semantic space so clouds from
different models can be laid out and categorized together honestly — the
comparison happens in a third-party embedder, never by pretending two models'
raw geometries share a basis. Also the vector source for the api-embeddings
token front-end (frontends/api_tokens.py), under the same honesty rule.

Unlike the namer, this module has never had a fall-through chain: the caller
always names the embedder and no code here picks one, so there is no path by
which a different embedder is silently substituted. The one gap that DID exist
is closed by `EmbedIdentityError` below — a multi-model server that ignores the
`model` field and serves whatever it has loaded.

Reachability, measured 2026-08-12 on this machine: nothing serves it.
localhost:11434 and localhost:8050 both refuse the connection and the LAN box
at 192.168.0.200 times out, so `nebulai compare` currently fails after ~12s of
retry backoff with a RuntimeError from `_embed_batch`, uncaught by the CLI. No
remote drop-in was added, because there isn't a faithful one: the HF router has
no /v1/embeddings route at all (GET returns 404), and the only live HF path for
this embedder is `hf-inference`'s feature-extraction pipeline, whose response
shape neither branch below parses. The ollama `mxbai-embed-large` tag is also a
quantised GGUF build of `mixedbread-ai/mxbai-embed-large-v1`, so its vectors
are not the fp32 repo's vectors — pointing `compare` at the repo would change
the neutral space, which is precisely the substitution this file must not make
quietly.
"""

import json
import time
import urllib.error
import urllib.request

import numpy as np

_DEFAULT_OLLAMA_HOST = "http://localhost:11434"  # local ollama server
_DEFAULT_EMBED_MODEL = "mxbai-embed-large"


class EmbedIdentityError(RuntimeError):
    """The endpoint served a different embedding model than the one requested.

    `compare` exists to put several models' clusters into ONE neutral space, so
    which embedder produced the vectors is not a detail — it IS the space. A
    host that ignores the `model` field and serves whatever it has loaded (the
    multi-model OpenAI-compatible servers do exactly this) would yield a map
    stamped `mxbai-embed-large` and positioned by something else. Same rule as
    backend.name.NamerIdentityError, applied to the embedder.
    """


def parse_embed_response(payload: dict, api: str, host: str, model: str) -> list[list[float]]:
    """Extract embeddings from one response, in input order.

    ollama's /api/embed returns {"embeddings": [...]}; OpenAI-compatible
    /v1/embeddings returns {"data": [{"index", "embedding"}, ...]} whose order
    is not guaranteed — sort by index.

    When the response names the model it ran, that name is checked against the
    one asked for. Servers that omit the field are taken at their word — there
    is nothing to check against, and refusing them would break every endpoint
    that simply does not report it.
    """
    served = str(payload.get("model") or "")
    if served and model and served.split(":", 1)[0].lower() != model.split(":", 1)[0].lower():
        raise EmbedIdentityError(
            f"asked {host} to embed with {model!r} and it answered as "
            f"{served!r} — a different embedder is a different semantic space, "
            "so these vectors are not comparable with the rest of the map"
        )
    if api == "openai":
        data = payload.get("data")
        if not data:
            raise RuntimeError(f"no embeddings returned from {host} ({model})")
        return [d["embedding"] for d in sorted(data, key=lambda d: d.get("index", 0))]
    embs = payload.get("embeddings")
    if not embs:
        raise RuntimeError(f"no embeddings returned from {host} ({model})")
    return embs


def _embed_batch(
    batch: list[str],
    host: str,
    model: str,
    timeout: float,
    retries: int,
    api: str = "ollama",
    api_key: str | None = None,
) -> list[list[float]]:
    """One embed call, retried on transient network/timeout errors.

    The M4 is shared: a concurrent naming run can queue this request behind
    dozens of generate calls, so a single timeout is not a hard failure —
    back off and retry before giving up.
    """
    url = f"{host.rstrip('/')}/v1/embeddings" if api == "openai" else f"{host}/api/embed"
    body = json.dumps({"model": model, "input": batch}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.load(r)
            return parse_embed_response(payload, api, host, model)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))  # linear backoff
    raise RuntimeError(
        f"embed request to {host} ({model}) failed after {retries} attempts: {last}"
    )


def embed_texts(
    texts: list[str],
    host: str = _DEFAULT_OLLAMA_HOST,
    model: str = _DEFAULT_EMBED_MODEL,
    batch_size: int = 32,
    timeout: float = 300.0,
    retries: int = 4,
    api: str = "ollama",
    api_key: str | None = None,
) -> np.ndarray:
    """Return (n, d) L2-normalized float32 embeddings for texts.

    api="ollama" posts to {host}/api/embed; api="openai" posts to any
    OpenAI-compatible {host}/v1/embeddings (with optional bearer api_key).
    """
    if api not in ("ollama", "openai"):
        raise ValueError(f"unknown embed api {api!r} (expected 'ollama' or 'openai')")
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        out.extend(_embed_batch(batch, host, model, timeout, retries, api, api_key))
    arr = np.asarray(out, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return arr
