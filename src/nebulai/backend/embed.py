"""Text embeddings via mxbai-embed-large on the M4 worker's ollama.

Used by the cross-model comparison (compare.py): each model's named clusters
are embedded in this neutral, model-independent semantic space so clouds from
different models can be laid out and categorized together honestly — the
comparison happens in a third-party embedder, never by pretending two models'
raw geometries share a basis.
"""

import json
import time
import urllib.error
import urllib.request

import numpy as np

_DEFAULT_OLLAMA_HOST = "http://192.168.0.200:11434"  # M4 worker
_DEFAULT_EMBED_MODEL = "mxbai-embed-large"


def _embed_batch(
    batch: list[str], host: str, model: str, timeout: float, retries: int
) -> list[list[float]]:
    """One /api/embed call, retried on transient network/timeout errors.

    The M4 is shared: a concurrent naming run can queue this request behind
    dozens of generate calls, so a single timeout is not a hard failure —
    back off and retry before giving up.
    """
    body = json.dumps({"model": model, "input": batch}).encode()
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            f"{host}/api/embed",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.load(r)
            embs = payload.get("embeddings")
            if not embs:
                raise RuntimeError(f"no embeddings returned from {host} ({model})")
            return embs
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
) -> np.ndarray:
    """Return (n, d) L2-normalized float32 embeddings for texts."""
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        out.extend(_embed_batch(batch, host, model, timeout, retries))
    arr = np.asarray(out, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return arr
