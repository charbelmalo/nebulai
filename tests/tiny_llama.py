"""Build a tiny, real Llama-shaped checkpoint on disk.

Not a mock: a genuine safetensors + config.json + tokenizer.json triple that
`LlamaNumpy` loads through its ordinary path. Random weights are fine — every
property the tests check (causal masking, RoPE placement, KV-cache
equivalence, hook dispatch, GQA head expansion) is a property of the *wiring*,
and wiring bugs show up on random weights exactly as loudly as on trained
ones, in a second rather than a minute.

The one thing random weights cannot check is whether the wiring matches the
checkpoint the weights came from. That is what the network-marked test against
the real SmolLM2 is for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

VOCAB = [f"t{i}" for i in range(64)] + [
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
]

CHATML = (
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + "
    "message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def _tokenizer_json() -> dict[str, Any]:
    """A WordLevel tokenizer over `VOCAB`, whitespace pre-tokenized.

    Written as JSON rather than built through the `tokenizers` builder API so
    the fixture has no construction order to get wrong and reads as data.
    """
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": VOCAB.index(t),
                "content": t,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for t in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": False},
        "model": {
            "type": "WordLevel",
            "vocab": {t: i for i, t in enumerate(VOCAB)},
            "unk_token": "t0",
        },
    }


def build(
    path: Path,
    *,
    n_layer: int = 3,
    n_head: int = 4,
    n_kv_head: int = 2,
    d: int = 32,
    d_mlp: int = 48,
    n_ctx: int = 64,
    tied: bool = True,
    seed: int = 7,
    **config_extra: Any,
) -> Path:
    """Write a loadable checkpoint into `path` and return it.

    `n_kv_head < n_head` by default so the GQA repeat path is the one every
    test exercises; the tied/untied split is a parameter because the two take
    different branches through the unembedding.
    """
    from safetensors.numpy import save_file

    rng = np.random.default_rng(seed)
    d_head = d // n_head
    V = len(VOCAB)

    def r(*shape: int, scale: float = 0.06) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    t: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": r(V, d),
        "model.norm.weight": np.ones(d, dtype=np.float32) + r(d, scale=0.02),
    }
    if not tied:
        t["lm_head.weight"] = r(V, d)
    for L in range(n_layer):
        p = f"model.layers.{L}."
        t[p + "self_attn.q_proj.weight"] = r(n_head * d_head, d)
        t[p + "self_attn.k_proj.weight"] = r(n_kv_head * d_head, d)
        t[p + "self_attn.v_proj.weight"] = r(n_kv_head * d_head, d)
        t[p + "self_attn.o_proj.weight"] = r(d, n_head * d_head)
        t[p + "mlp.gate_proj.weight"] = r(d_mlp, d)
        t[p + "mlp.up_proj.weight"] = r(d_mlp, d)
        t[p + "mlp.down_proj.weight"] = r(d, d_mlp)
        t[p + "input_layernorm.weight"] = np.ones(d, dtype=np.float32) + r(d, scale=0.02)
        t[p + "post_attention_layernorm.weight"] = np.ones(d, dtype=np.float32) + r(
            d, scale=0.02
        )

    path.mkdir(parents=True, exist_ok=True)
    save_file(t, str(path / "model.safetensors"))

    cfg = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": d,
        "intermediate_size": d_mlp,
        "num_hidden_layers": n_layer,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_kv_head,
        "head_dim": d_head,
        "max_position_embeddings": n_ctx,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": tied,
        "vocab_size": V,
        "bos_token_id": VOCAB.index("<|im_start|>"),
        "eos_token_id": VOCAB.index("<|im_end|>"),
    }
    cfg.update(config_extra)
    (path / "config.json").write_text(json.dumps(cfg, indent=2))
    (path / "tokenizer.json").write_text(json.dumps(_tokenizer_json()))
    (path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": CHATML, "eos_token": "<|im_end|>"})
    )
    return path


def words(n: int, *, seed: int = 3) -> str:
    """`n` in-vocabulary whitespace-separated tokens."""
    rng = np.random.default_rng(seed)
    return " ".join(f"t{int(i)}" for i in rng.integers(1, 64, size=n))
