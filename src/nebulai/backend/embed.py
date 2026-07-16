"""Text embeddings via an ollama or OpenAI-compatible embeddings endpoint
(default: mxbai-embed-large on the M4 worker's ollama).

Used by the cross-model comparison (compare.py): each model's named clusters
are embedded in this neutral, model-independent semantic space so clouds from
different models can be laid out and categorized together honestly — the
comparison happens in a third-party embedder, never by pretending two models'
raw geometries share a basis. Also the vector source for the api-embeddings
token front-end (frontends/api_tokens.py), under the same honesty rule.
"""

import json
import time
import urllib.error
import urllib.request

import numpy as np

_DEFAULT_OLLAMA_HOST = "http://192.168.0.200:11434"  # M4 worker
_DEFAULT_EMBED_MODEL = "mxbai-embed-large"


def parse_embed_response(payload: dict, api: str, host: str, model: str) -> list[list[float]]:
    """Extract embeddings from one response, in input order.

    ollama's /api/embed returns {"embeddings": [...]}; OpenAI-compatible
    /v1/embeddings returns {"data": [{"index", "embedding"}, ...]} whose order
    is not guaranteed — sort by index.
    """
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
