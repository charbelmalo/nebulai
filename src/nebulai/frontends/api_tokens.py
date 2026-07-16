"""API-embeddings front-end: same curated vocabulary as the W_E front-end,
but geometry = a third-party text-embedding model's vector for each token
string (ollama on the LAN worker, or any OpenAI-compatible /v1/embeddings).

HONESTY (hard rule): this map is NOT the model's internal geometry — it is
where an external embedder places the model's token *strings*. `meta.unit`
names the embedder explicitly and `meta.geometry` spells it out, so the
viewer footer can never present it as model-internal. The output dataset id
is `<model>__api-<embed_model>` so the real W_E artifact is never clobbered.

Teaching use: contrast the same vocabulary under the model's own W_E vs a
modern semantic embedder — frequency/orthography structure vs meaning.
"""

import json
import time
from pathlib import Path

import numpy as np

from ..backend.embed import embed_texts
from ..units import Units


def api_dataset_id(model_id: str, embed_model: str) -> str:
    """Output dir name — must match build_server.dataset_id_for."""
    return f"{model_id.replace('/', '__')}__api-{embed_model.replace('/', '__')}"


def load_api_token_units(
    model_id: str = "gpt2",
    embed_host: str = "http://192.168.0.200:11434",
    embed_model: str = "mxbai-embed-large",
    api: str = "ollama",
    api_key: str | None = None,
    center: bool = True,
    max_tokens: int | None = None,
    out_root: Path = Path("out"),
    batch_size: int = 64,
    checkpoint_every: int = 2000,
) -> Units:
    from tokenizers import Tokenizer

    from .tokens import curated_vocab

    ids, labels = curated_vocab(model_id, max_tokens)
    vocab_size = Tokenizer.from_pretrained(model_id).get_vocab_size()

    # Checkpointed cache: a full vocab is ~50k strings ≈ 1.5k API calls over
    # the LAN — resumable so a dropped connection doesn't restart from zero.
    cache_dir = out_root / api_dataset_id(model_id, embed_model)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "embed_cache.npz"
    cache_meta = cache_dir / "embed_cache.params.json"
    params = {
        "model": model_id,
        "embed_model": embed_model,
        "api": api,
        "max_tokens": max_tokens,
        "n": len(ids),
    }

    done = np.zeros((0, 0), dtype=np.float32)
    if (
        cache.exists()
        and cache_meta.exists()
        and json.loads(cache_meta.read_text()) == params
    ):
        done = np.load(cache)["vectors"]
        if len(done):
            print(f"  resuming embed cache: {len(done)}/{len(ids)} tokens done")

    while len(done) < len(ids):
        start = len(done)
        chunk = labels[start : start + checkpoint_every]
        t0 = time.time()
        vecs = embed_texts(
            chunk,
            host=embed_host,
            model=embed_model,
            batch_size=batch_size,
            api=api,
            api_key=api_key,
        )
        done = vecs if start == 0 else np.vstack([done, vecs])
        np.savez_compressed(cache, vectors=done)
        cache_meta.write_text(json.dumps(params))
        print(
            f"  embedded {len(done)}/{len(ids)} tokens "
            f"({len(chunk)} in {time.time() - t0:.1f}s)"
        )

    V = np.asarray(done, dtype=np.float32)
    if V.shape[0] != len(ids):
        raise RuntimeError(
            f"embed cache has {V.shape[0]} rows for {len(ids)} tokens — "
            f"delete {cache} and rerun"
        )
    if center:
        V = V - V.mean(axis=0, keepdims=True)

    return Units(
        ids=ids,
        vectors=np.ascontiguousarray(V, dtype=np.float32),
        labels=labels,
        meta={
            "model": model_id,
            "unit": f"api_text_embedding({embed_model})",
            "geometry": "third-party text-embedding space — NOT model-internal",
            "embed_model": embed_model,
            "embed_host": embed_host,
            "embed_api": api,
            "centered": center,
            "vocab_size": int(vocab_size),
            "kept": len(ids),
        },
    )
