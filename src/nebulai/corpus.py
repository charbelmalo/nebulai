"""The endpoint-era corpus: which models we map, and how we reach them.

Two facts about every entry below were measured, not assumed (2026-08-12):

1. **The weights are readable without downloading the model.** Every repo here
   answers `HTTP 206 Partial Content` on
   `https://huggingface.co/{repo}/resolve/{rev}/{shard}` with no auth, so
   `weights.py` reads the safetensors header (a few hundred KB) and then only
   the byte ranges of the rows it actually maps. A curated 50k-token W_E map
   costs the `rows_50k_mb` column below — not `total_gb`.

2. **The chat endpoint serves that exact model.** `endpoint` is a pinned model
   id, never a family or an alias. A cheaper endpoint that serves a *different*
   model is not a fallback — it is a different model, and reporting its titles
   as this model's would be a fabrication. `llm.py` therefore refuses rather
   than substitutes; see `IdentityError` — the refusal is enforced there now, so
   every caller of the shared client inherits it, not just the namer.

`tie_word_embeddings` decides whether the W_E↔W_U experiment is even askable:
a tied model has one matrix serving both roles, so "does it read tokens the way
it writes them" is true by construction and carries no information. Gemma-4 is
the tied control; the other three are the actual comparison.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """One corpus entry: the weight repo, the chat endpoint, and the prices."""

    key: str  # short name used in out/<key>/ and on the CLI
    repo: str  # HF repo holding the safetensors shards
    endpoint: str  # OpenRouter model id — pinned, never a family
    hf_endpoint: str | None  # same model via HF Inference Providers router
    revision: str  # pinned; "main" only where no release tag exists yet
    embed_key: str  # verified W_E key (exact, from the shard index)
    unembed_key: str | None  # verified W_U key; None when embeddings are tied
    vocab_size: int
    hidden_size: int
    n_layers: int
    total_gb: float  # full checkpoint — what we deliberately do NOT download
    rows_50k_mb: int  # what a curated 50k-token map actually streams
    usd_in: float  # $/M prompt tokens at the endpoint
    usd_out: float  # $/M completion tokens
    moe: bool  # MoE models keep down_proj inside experts (see neurons.py)
    notes: str


# Verified 2026-08-12 against each repo's model.safetensors.index.json and
# config.json, and against https://openrouter.ai/api/v1/models for pricing.
CORPUS: dict[str, ModelSpec] = {
    "muse-glimmer-30b": ModelSpec(
        key="muse-glimmer-30b",
        repo="meta-models/Muse-Glimmer-30B",
        endpoint="meta/muse-glimmer-30b",
        hf_endpoint="meta-models/Muse-Glimmer-30B",
        revision="main",
        embed_key="model.language_model.embed_tokens.weight",
        unembed_key="lm_head.weight",
        vocab_size=202048,
        hidden_size=6656,
        n_layers=52,
        total_gb=59.55,
        rows_50k_mb=666,
        usd_in=0.35,
        usd_out=1.50,
        moe=False,
        notes=(
            "Multimodal wrapper, so the text weights nest under "
            "`model.language_model.`. Untied — the headline W_E vs W_U case. "
            "Global layers are NoPE (rope_theta=0); softcap and "
            "output_multiplier act on logits at inference and never touch the "
            "rows we map."
        ),
    ),
    "gemma-4-26b": ModelSpec(
        key="gemma-4-26b",
        # Canonical casing: the lowercase id 307-redirects here, and a client
        # that drops `Range` across the redirect silently fetches whole shards.
        repo="google/gemma-4-26B-A4B-it",
        endpoint="google/gemma-4-26b-a4b-it:free",
        # The router spells it with different casing than OpenRouter does;
        # verified against router.huggingface.co/v1/models, so match case-insensitively.
        hf_endpoint="google/gemma-4-26B-A4B-it",
        revision="main",
        embed_key="model.language_model.embed_tokens.weight",
        unembed_key=None,  # tied
        vocab_size=262144,
        hidden_size=2816,
        n_layers=30,
        total_gb=51.61,
        rows_50k_mb=282,
        usd_in=0.0,
        usd_out=0.0,
        moe=True,  # Gemma4ForConditionalGeneration, num_experts=128
        notes=(
            "Tied embeddings — the control for the W_E/W_U experiment, and the "
            "only entry whose endpoint is genuinely $0. Widest vocab in the "
            "corpus (262k) against the narrowest hidden size (2816). Also a "
            "multimodal wrapper, so its W_E nests under `model.language_model.` "
            "exactly like Glimmer's."
        ),
    ),
    "ling-2.6-flash": ModelSpec(
        key="ling-2.6-flash",
        repo="inclusionAI/Ling-2.6-flash",
        endpoint="inclusionai/ling-2.6-flash",
        hf_endpoint=None,  # not served by an HF inference provider
        revision="main",
        embed_key="model.word_embeddings.weight",
        unembed_key="lm_head.weight",
        vocab_size=157184,
        hidden_size=4096,
        n_layers=32,
        total_gb=208.37,
        rows_50k_mb=410,
        usd_in=0.010,
        usd_out=0.030,
        moe=True,
        notes=(
            "BailingMoeV2_5. Its W_E key (`model.word_embeddings.weight`) "
            "matches none of the four families the tokens frontend knew, which "
            "is why suffix resolution had to be extended rather than patched. "
            "25,015 tensors across 27 distinct shard files — though every "
            "filename claims `-of-00026`, so trust the index, not the names. "
            "The strongest argument for range reads in the corpus: the full "
            "checkpoint is 208 GB and a token map needs 410 MB, or 0.2%, of it."
        ),
    ),
    "mistral-nemo": ModelSpec(
        key="mistral-nemo",
        repo="mistralai/Mistral-Nemo-Instruct-2407",
        endpoint="mistralai/mistral-nemo",
        # Measured absent from router.huggingface.co/v1/models (129 models, no
        # Mistral entries at all) — OpenRouter is the only endpoint for it.
        hf_endpoint=None,
        revision="main",
        embed_key="model.embed_tokens.weight",
        unembed_key="lm_head.weight",
        vocab_size=131072,
        hidden_size=5120,
        n_layers=40,
        total_gb=24.50,
        rows_50k_mb=512,
        usd_in=0.019,
        usd_out=0.030,
        moe=False,
        notes=(
            "Dense, untied, conventional key layout — the entry most likely to "
            "work first, and therefore the one to debug the remote reader "
            "against before pointing it at Glimmer."
        ),
    ),
}

# Every W_E key family the corpus actually uses. The first three were already
# known; `word_embeddings.weight` came from Ling and `language_model.` nesting
# from the two multimodal wrappers, which is why exact-key lookup fails here.
EMBED_KEY_SUFFIXES = (
    "wte.weight",  # GPT-2
    "embed_in.weight",  # Pythia / GPT-NeoX
    "embed_tokens.weight",  # Llama / Qwen / Gemma / Glimmer (nested)
    "word_embeddings.weight",  # Bailing / Ling
)

UNEMBED_KEY_SUFFIXES = ("lm_head.weight", "unembed.weight")

# Default ceiling for one command's spend at a paid endpoint. The namer
# estimates before it sends and refuses over budget; it never downgrades to a
# cheaper model to fit, because that would silently change what the map is
# evidence of.
DEFAULT_MAX_COST_USD = 1.00


def spec(key: str) -> ModelSpec:
    """Look up a corpus entry by short key or by full HF repo id."""
    if key in CORPUS:
        return CORPUS[key]
    for s in CORPUS.values():
        if s.repo == key or s.endpoint == key:
            return s
    raise KeyError(f"{key!r} is not in the corpus; have {sorted(CORPUS)}")


def estimate_naming_cost(n_clusters: int, model_key: str, batch_size: int = 15) -> float:
    """USD to name `n_clusters` clusters at this model's endpoint.

    Measured shape of a naming request: ~1500 prompt tokens for a batch of 15
    clusters at 20 representatives each, ~400 completion tokens back. Rounded
    up per batch, so the estimate is an upper bound rather than a mean.
    """
    s = spec(model_key)
    batches = -(-n_clusters // batch_size)
    return (batches * 1500 * s.usd_in + batches * 400 * s.usd_out) / 1e6
