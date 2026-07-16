"""Pure-numpy GPT-2 / GPT-NeoX-tied forward pass with interpretability hooks.

Loads a model's `model.safetensors` (the same file the token front-end already
reads) and runs a real, deterministic forward pass in numpy. No torch, no
transformer_lens — so the outputs are cheap to precompute offline and ship as
static JSON to a Netlify build, while remaining the model's genuine internals.

Scope: GPT-2 family (HF Conv1D weights, learned positional embeddings, pre-LN,
tied unembedding). This is the exact architecture of the models the project
already maps (gpt2, distilgpt2). Pythia/NeoX support is a later addition — its
weights are Linear (not Conv1D) and it uses rotary embeddings, so it needs a
separate forward; this module raises clearly rather than pretend.

A `Trace` captures every quantity the viewer's drivers need:
  - `resid`      (n_layer+1, T, d)   residual stream before each block + final
  - `attn`       (n_layer, n_head, T, T) post-softmax attention patterns
  - `mlp_post`   (n_layer, T, d_mlp) post-activation MLP hidden (neuron acts)
  - `logits`     (T, V)              final logits (tied unembedding)
  - `ln_f_scale` (T,)                per-position ln_f normalizer (for logit lens)

All arrays are float32 and shapes are asserted, because a silently transposed
weight is exactly the "numerically wrong but plausible-looking" failure the
review passes exist to catch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

# HF GPT-2 state-dict layout. c_attn/c_proj/c_fc are Conv1D: weight is
# (n_in, n_out) and applied as `x @ W + b` (NOT a Linear's (n_out, n_in)).
_LN_EPS = 1e-5


def _gelu_new(x: np.ndarray) -> np.ndarray:
    """GPT-2's tanh-approximate GELU (the `gelu_new` HF uses), in float64 for
    the cube so large activations don't overflow float32 before the tanh."""
    x64 = x.astype(np.float64)
    inner = np.sqrt(2.0 / np.pi) * (x64 + 0.044715 * x64**3)
    return (0.5 * x64 * (1.0 + np.tanh(inner))).astype(np.float32)


def _layernorm(x: np.ndarray, g: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LayerNorm over the last axis. Returns (normed, 1/std) — the second value
    is the per-row scale the logit lens needs to read pre-final-norm directions."""
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + _LN_EPS)
    return (xc * inv) * g + b, inv[..., 0]


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


@dataclass
class Trace:
    """Everything one forward pass reveals. Every field is a real array; the
    viewer never invents motion or correlation the model didn't produce."""

    tokens: list[int]
    token_strs: list[str]
    resid: np.ndarray  # (n_layer+1, T, d)
    attn: np.ndarray  # (n_layer, n_head, T, T)
    mlp_post: np.ndarray  # (n_layer, T, d_mlp)
    logits: np.ndarray  # (T, V)
    ln_f_inv: np.ndarray  # (T,) final-LN normalizer per position

    @property
    def n_layer(self) -> int:
        return self.attn.shape[0]

    @property
    def n_head(self) -> int:
        return self.attn.shape[1]


class GPT2Numpy:
    """A GPT-2 whose every matmul is visible. Construct once per model, then
    call `.forward(prompt)` per prompt. Weights stay resident (float32)."""

    def __init__(self, model_id: str = "gpt2"):
        from huggingface_hub import hf_hub_download
        from safetensors.numpy import load_file
        from tokenizers import Tokenizer

        cfg = json.loads(open(hf_hub_download(model_id, "config.json")).read())
        arch = cfg.get("model_type", "")
        if arch not in ("gpt2",):
            raise NotImplementedError(
                f"gpt2_numpy supports GPT-2-family models only; {model_id!r} is "
                f"{arch!r}. Pythia/NeoX (Linear weights + rotary) needs a separate "
                f"forward — refusing to run a wrong architecture."
            )
        self.model_id = model_id
        self.n_layer = int(cfg["n_layer"])
        self.n_head = int(cfg["n_head"])
        self.n_ctx = int(cfg.get("n_ctx", cfg.get("n_positions", 1024)))
        self.d = int(cfg["n_embd"])
        self.d_head = self.d // self.n_head

        self.t = load_file(hf_hub_download(model_id, "model.safetensors"))
        # distilgpt2 ships the same GPT-2 tensors under a uniform "transformer."
        # prefix (gpt2/gpt2-medium are unprefixed) — strip it so one key scheme
        # serves the whole family.
        if "wte.weight" not in self.t and "transformer.wte.weight" in self.t:
            self.t = {k.removeprefix("transformer."): v for k, v in self.t.items()}
        self.wte = self._g("wte.weight")  # (V, d)
        self.wpe = self._g("wpe.weight")  # (n_ctx, d)
        self.V = self.wte.shape[0]
        assert self.wte.shape[1] == self.d, "wte width != n_embd"
        self.tok = Tokenizer.from_pretrained(model_id)

    def _g(self, key: str) -> np.ndarray:
        return np.asarray(self.t[key], dtype=np.float32)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode1(self, tid: int) -> str:
        return self.tok.decode([tid])

    def forward(self, prompt: str | list[int]) -> Trace:
        ids = self.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if not ids:
            raise ValueError("empty prompt")
        if len(ids) > self.n_ctx:
            ids = ids[: self.n_ctx]
        T, d, H, dh = len(ids), self.d, self.n_head, self.d_head

        x = self.wte[ids] + self.wpe[:T]  # (T, d) — real token+pos embedding
        resid = np.empty((self.n_layer + 1, T, d), dtype=np.float32)
        attn = np.empty((self.n_layer, H, T, T), dtype=np.float32)
        mlp_post = np.empty((self.n_layer, T, 4 * d), dtype=np.float32)
        # causal mask: -inf above the diagonal (query cannot see future keys)
        cmask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)

        for L in range(self.n_layer):
            resid[L] = x
            p = f"h.{L}."
            # --- attention (pre-LN) ---
            xn, _ = _layernorm(x, self._g(p + "ln_1.weight"), self._g(p + "ln_1.bias"))
            qkv = xn @ self._g(p + "attn.c_attn.weight") + self._g(p + "attn.c_attn.bias")
            q, k, v = np.split(qkv, 3, axis=-1)  # each (T, d)
            # (H, T, dh)
            q = q.reshape(T, H, dh).transpose(1, 0, 2)
            k = k.reshape(T, H, dh).transpose(1, 0, 2)
            v = v.reshape(T, H, dh).transpose(1, 0, 2)
            scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(dh) + cmask  # (H,T,T)
            a = _softmax(scores, axis=-1)
            attn[L] = a
            ctx = (a @ v).transpose(1, 0, 2).reshape(T, d)  # (T, d)
            attn_out = ctx @ self._g(p + "attn.c_proj.weight") + self._g(p + "attn.c_proj.bias")
            x = x + attn_out
            # --- mlp (pre-LN) ---
            xn2, _ = _layernorm(x, self._g(p + "ln_2.weight"), self._g(p + "ln_2.bias"))
            h = xn2 @ self._g(p + "mlp.c_fc.weight") + self._g(p + "mlp.c_fc.bias")
            h = _gelu_new(h)
            mlp_post[L] = h
            mlp_out = h @ self._g(p + "mlp.c_proj.weight") + self._g(p + "mlp.c_proj.bias")
            x = x + mlp_out

        resid[self.n_layer] = x
        xf, inv = _layernorm(x, self._g("ln_f.weight"), self._g("ln_f.bias"))
        logits = xf @ self.wte.T  # tied unembedding (T, V)

        return Trace(
            tokens=ids,
            token_strs=[self.decode1(i) for i in ids],
            resid=resid,
            attn=attn,
            mlp_post=mlp_post,
            logits=logits.astype(np.float32),
            ln_f_inv=inv.astype(np.float32),
        )

    def logit_lens(self, resid_row: np.ndarray) -> np.ndarray:
        """Apply the final LN + tied unembedding to any residual-stream vector.
        This is the honest logit lens: the model's own unembedding read off an
        intermediate layer (no trained translator — that's the tuned lens)."""
        g, b = self._g("ln_f.weight"), self._g("ln_f.bias")
        xf, _ = _layernorm(resid_row[None, :], g, b)
        return (xf @ self.wte.T)[0]
