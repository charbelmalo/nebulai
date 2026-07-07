"""Plan C front-end: one point per vocabulary token, geometry = embedding rows.

No corpus sweep, no activations — just the model's input embedding matrix
(W_E). Note that GPT-2 ties W_E and W_U, so there is no separate unembedding
map for it; use an untied model (e.g. Pythia) for that comparison.
"""

import numpy as np

from ..units import Units

# state-dict keys for the input embedding across common micro-model families
_EMBED_KEY_SUFFIXES = (
    "wte.weight",  # GPT-2
    "embed_in.weight",  # Pythia / GPT-NeoX
    "embed_tokens.weight",  # Llama / Qwen / Gemma
)


def _keep(s: str) -> bool:
    """Curate the vocab: drop byte-fragment junk, controls, and empties."""
    if not s or "�" in s:  # partial-UTF8 byte tokens decode to U+FFFD
        return False
    if s.strip() == "":
        return False
    if any(ord(c) < 0x20 or 0x7F <= ord(c) < 0xA0 for c in s):
        return False
    return True


def load_token_units(
    model_id: str = "gpt2",
    center: bool = True,
    max_tokens: int | None = None,
) -> Units:
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file
    from tokenizers import Tokenizer

    path = hf_hub_download(model_id, "model.safetensors")
    tensors = load_file(path)
    key = next(
        (k for k in tensors if k.endswith(_EMBED_KEY_SUFFIXES)), None
    )
    if key is None:
        raise KeyError(
            f"no embedding matrix found in {model_id}; keys: {sorted(tensors)[:10]}..."
        )
    W = np.asarray(tensors[key], dtype=np.float32)
    del tensors

    tok = Tokenizer.from_pretrained(model_id)
    n_vocab = min(W.shape[0], tok.get_vocab_size())

    ids: list[int] = []
    labels: list[str] = []
    for i in range(n_vocab):
        s = tok.decode([i])
        if _keep(s):
            ids.append(i)
            labels.append(s)

    # BPE merge order roughly tracks corpus frequency, so the lowest ids are
    # the most frequent tokens — truncating keeps the common ones.
    if max_tokens is not None and len(ids) > max_tokens:
        ids = ids[:max_tokens]
        labels = labels[:max_tokens]

    V = W[ids]
    if center:
        # mean-centering counters the anisotropy of token embedding spaces
        V = V - V.mean(axis=0, keepdims=True)

    return Units(
        ids=ids,
        vectors=np.ascontiguousarray(V, dtype=np.float32),
        labels=labels,
        meta={
            "model": model_id,
            "unit": "token_embedding",
            "weight_key": key,
            "centered": center,
            "vocab_size": int(W.shape[0]),
            "kept": len(ids),
        },
    )
