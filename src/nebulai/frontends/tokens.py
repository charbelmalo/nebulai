"""Plan C front-end: one point per vocabulary token, geometry = embedding rows.

No corpus sweep, no activations — just the model's input embedding matrix
(W_E), or its output matrix (W_U, `which="output"`) when the model unties them.

The endpoint-era models in `corpus.py` are 24-208 GB across 2-26 shards, so
`hf_hub_download(model_id, "model.safetensors")` cannot reach them at all. The
default path here is therefore `weights.RemoteCheckpoint`: pin the revision,
read the shard index and one shard header, curate the vocabulary from the
*tokenizer* (no weights needed), and then range-read only the rows that survive
curation. A 50k-token map streams a few hundred MB instead of downloading the
checkpoint, and `meta` stamps `bytes_fetched` so the map can prove it.

Tied models (Gemma-4) have one matrix serving both roles: `which="output"`
refuses rather than returning W_E under a second name, because a map of "how it
writes tokens" that is byte-identical to "how it reads them" carries no
information and would misrepresent what was measured.
"""

import re
from collections.abc import Sequence

import numpy as np

from ..corpus import CORPUS, EMBED_KEY_SUFFIXES, UNEMBED_KEY_SUFFIXES, ModelSpec
from ..units import Units

# a single-file checkpoint bigger than this is read remotely even though it is
# unsharded; below it, downloading once into the HF cache is simply cheaper
LARGE_SINGLE_FILE_BYTES = 2 << 30  # 2 GiB


class TiedEmbeddingError(ValueError):
    """Raised when W_U is asked of a model whose W_U *is* its W_E."""


# Reserved slots that are NOT flagged special by their own tokenizer, so
# `decode()` hands them back as literal text and the checks below would keep
# them. gpt2 has none (its one special token IS flagged, decodes to "" and
# falls out); Gemma-4 has 160 in its first 10k ids alone — `<unusedN>` blocks
# hold placeholder rows the model never trained, and they cluster with each
# other, so an uncurated map spends real estate on the tokenizer's padding.
# Enumerated by family rather than by shape: `<h3>`, `<span>` and `<div>` are
# single tokens in these vocabs and are real content, so a blanket `<...>`
# filter would delete the HTML the map is supposed to show.
_RESERVED = re.compile(
    r"""^(?:
        <unused\d+>                      # Gemma reserved block
      | <extra_id_\d+>                   # T5 sentinels
      | <reserved_special_token_\d+>     # Llama 3
      | <\|[^|]*\|>                      # chat/control markers
      | <0x[0-9A-Fa-f]{2}>               # SentencePiece byte fallback
      | \[control_\d+\]                  # Mistral/Tekken control slots
      | \[multimodal\]                   # Gemma multimodal wrapper
      | </?(?:pad|eos|bos|unk|mask|sep|cls|s)>   # classic sentinels
    )$""",
    re.VERBOSE,
)


def _keep(s: str) -> bool:
    """Curate the vocab: drop byte-fragment junk, controls, and empties."""
    if not s or "�" in s:  # partial-UTF8 byte tokens decode to U+FFFD
        return False
    if s.strip() == "":
        return False
    if any(ord(c) < 0x20 or 0x7F <= ord(c) < 0xA0 for c in s):
        return False
    if _RESERVED.match(s.strip()):
        return False
    return True


def corpus_spec(model_id: str) -> ModelSpec | None:
    """The corpus entry for a model id, or None for anything off-corpus."""
    if model_id in CORPUS:
        return CORPUS[model_id]
    for s in CORPUS.values():
        if s.repo == model_id:
            return s
    return None


def find_suffix_key(keys: Sequence[str], suffixes: tuple[str, ...]) -> str | None:
    """First key ending in one of `suffixes`, honouring the suffix order."""
    for suffix in suffixes:
        matches = sorted(k for k in keys if k.endswith(suffix))
        if matches:
            return matches[0]
    return None


def resolve_token_key(
    keys: Sequence[str], which: str = "input", spec: ModelSpec | None = None
) -> str:
    """The W_E (`input`) or W_U (`output`) key in a checkpoint's key list.

    Resolution is by suffix, never by exact name: Ling's W_E is
    `model.word_embeddings.weight` and the two multimodal wrappers nest theirs
    under `model.language_model.`, so the exact-key lookup this front-end used
    to do resolved to nothing on three of the four corpus models.
    """
    if which not in ("input", "output"):
        raise ValueError(f"which must be 'input' or 'output', not {which!r}")
    keys = list(keys)
    if which == "input":
        if spec is not None and spec.embed_key in keys:
            return spec.embed_key
        key = find_suffix_key(keys, EMBED_KEY_SUFFIXES)
        if key is None:
            raise KeyError(
                f"no embedding matrix found: no key ends in any of "
                f"{EMBED_KEY_SUFFIXES}; have {sorted(keys)[:8]}..."
            )
        return key

    if spec is not None and spec.unembed_key is None:
        raise TiedEmbeddingError(
            f"{spec.key} ties its embeddings (tie_word_embeddings=true): its W_U "
            f"IS its W_E ({spec.embed_key}), so an unembedding map would be the "
            f"same matrix under a second name and would carry no new "
            f"information. Use --unembedding on an untied model "
            f"(mistral-nemo, muse-glimmer-30b, ling-2.6-flash)."
        )
    if spec is not None and spec.unembed_key in keys:
        return spec.unembed_key
    key = find_suffix_key(keys, UNEMBED_KEY_SUFFIXES)
    if key is None:
        raise TiedEmbeddingError(
            f"no unembedding matrix in this checkpoint (no key ends in any of "
            f"{UNEMBED_KEY_SUFFIXES}) — the model almost certainly ties its "
            f"embeddings, so its W_U IS its W_E and the map would carry no new "
            f"information. Use which='input', or an untied model."
        )
    return key


def curated_vocab(
    model_id: str,
    max_tokens: int | None = None,
    n_vocab: int | None = None,
    revision: str = "main",
    token: str | None = None,
) -> tuple[list[int], list[str]]:
    """The curated (ids, labels) vocabulary for a model's tokenizer.

    Shared by the W_E front-end below and the api-embeddings front-end
    (api_tokens.py) so both map exactly the same token set. Needs the
    tokenizer only — which is why the remote path can curate *first* and then
    fetch just the surviving rows."""
    from tokenizers import Tokenizer

    try:
        tok = Tokenizer.from_pretrained(model_id, revision=revision, token=token)
    except TypeError:  # older tokenizers without revision/token kwargs
        tok = Tokenizer.from_pretrained(model_id)
    n = tok.get_vocab_size() if n_vocab is None else min(n_vocab, tok.get_vocab_size())

    ids: list[int] = []
    labels: list[str] = []
    for i in range(n):
        s = tok.decode([i])
        if _keep(s):
            ids.append(i)
            labels.append(s)

    # BPE merge order roughly tracks corpus frequency, so the lowest ids are
    # the most frequent tokens — truncating keeps the common ones.
    if max_tokens is not None and len(ids) > max_tokens:
        ids = ids[:max_tokens]
        labels = labels[:max_tokens]
    return ids, labels


def load_token_units(
    model_id: str = "gpt2",
    center: bool = True,
    max_tokens: int | None = None,
    revision: str = "main",
    remote: bool | None = None,
    which: str = "input",
    token: str | None = None,
) -> Units:
    """Token-geometry Units from a model's own W_E (or W_U) rows.

    `remote=None` (default) decides per repo: sharded checkpoints and large
    single files are read over HTTP ranges and never downloaded; a small
    single-file checkpoint (gpt2) is downloaded into the HF cache as before.
    `revision` is resolved to a commit sha and stamped into meta, so a run
    against "main" stays reproducible after main moves.
    """
    from ..weights import (
        SINGLE_NAME,
        RemoteCheckpoint,
        load_safetensor_f32,
        safetensor_keys,
    )

    spec = corpus_spec(model_id)
    repo = spec.repo if spec is not None else model_id
    if spec is not None and revision == "main":
        revision = spec.revision

    ck: RemoteCheckpoint | None = None
    if remote is not False:
        ck = RemoteCheckpoint.open(repo, revision=revision, token=token)
        if remote is None:
            size = None if ck.is_sharded else ck.file_size(SINGLE_NAME)
            remote = ck.is_sharded or (size or 0) > LARGE_SINGLE_FILE_BYTES

    if remote and ck is not None:
        key = resolve_token_key(ck.keys(), which, spec)
        _, shape = ck.info(key)
        vocab_size = int(shape[0])
        # curate FIRST (tokenizer only), then fetch just the surviving rows —
        # reading the whole tensor and indexing it would defeat the design
        ids, labels = curated_vocab(
            repo, max_tokens, n_vocab=vocab_size, revision=ck.revision, token=token
        )
        V = ck.read_rows(key, ids)
        source = "remote-range"
        resolved = ck.revision
        fetched = ck.bytes_fetched
    else:
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(repo, SINGLE_NAME, revision=revision)
        except Exception as e:  # noqa: BLE001 — re-raised with the real cause
            if ck is not None and ck.is_sharded:
                raise FileNotFoundError(
                    f"{repo} has no single {SINGLE_NAME}: it is sharded across "
                    f"{len(ck.shards)} files. Drop --no-remote — the remote "
                    f"reader is the only way to read it without downloading it."
                ) from e
            raise
        keys = safetensor_keys(path)
        key = resolve_token_key(keys, which, spec)
        W = load_safetensor_f32(path, keys=[key])[key]
        vocab_size = int(W.shape[0])
        ids, labels = curated_vocab(
            repo, max_tokens, n_vocab=vocab_size, revision=revision, token=token
        )
        V = W[ids]
        source = "local"
        fetched = ck.bytes_fetched if ck is not None else 0
        if ck is not None:
            resolved = ck.revision
        else:  # --no-remote never opened a checkpoint; pin the sha anyway
            from ..weights import resolve_revision

            try:
                resolved = resolve_revision(repo, revision, token)
            except Exception:  # offline: stamp what was asked for, not a guess
                resolved = revision

    V = np.ascontiguousarray(V, dtype=np.float32)
    if center:
        # mean-centering counters the anisotropy of token embedding spaces
        V = V - V.mean(axis=0, keepdims=True)

    meta = {
        "model": model_id,
        "unit": "token_embedding" if which == "input" else "token_unembedding",
        "which": which,
        "weight_key": key,
        "repo": repo,
        "revision": resolved,
        "source": source,
        "bytes_fetched": int(fetched),
        "centered": center,
        "vocab_size": vocab_size,
        "kept": len(ids),
    }
    if spec is not None and source == "remote-range":
        # the honest headline: what the map did NOT download to exist
        meta["total_gb_not_downloaded"] = spec.total_gb

    return Units(
        ids=ids,
        vectors=np.ascontiguousarray(V, dtype=np.float32),
        labels=labels,
        meta=meta,
    )
