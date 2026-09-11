"""Pure-numpy Llama-family forward pass with interpretability hooks and a KV cache.

The sibling of :mod:`gpt2_numpy` for the *instruct* half of the tree. Same
promise: no torch, no transformer_lens, every matmul visible, the arrays the
viewer draws are the model's genuine internals rather than a re-implementation
that merely looks plausible.

Architecture covered — the Llama 2/3 / SmolLM2 / Qwen2 shape:

* **RMSNorm** (no mean subtraction, no bias) instead of LayerNorm.
* **RoPE** rotary position embeddings, half-split convention
  (``rope_interleaved: false``), applied to q and k only.
* **SwiGLU** MLP: ``down(silu(gate(x)) * up(x))`` — so "post-activation MLP
  hidden" is the product, i.e. exactly the vector ``down_proj`` reads, which is
  the same quantity the neuron front-end treats as a neuron.
* **Grouped-query attention**: ``num_key_value_heads <= num_attention_heads``;
  k/v head groups are repeated, never averaged.
* **Tied or untied embeddings**: ``lm_head.weight`` when present, otherwise the
  input embedding matrix.
* Optional **q/k/v biases** (Qwen2 has them, SmolLM2 does not).

Two things this module has that ``gpt2_numpy`` does not, both there from the
first commit because retrofitting either into a validated forward pass is how
correctness regressions happen:

1. **A KV cache.** :class:`KVCache` makes generation linear rather than
   quadratic. Cached and uncached logits are **not bit-identical** and cannot
   be: a cached step multiplies a ``(1, d)`` activation where the uncached pass
   multiplies a ``(T, d)`` block, so BLAS picks a different kernel and a
   different summation order, and thirty residual additions amplify the last
   bit of each.

   Measured on SmolLM2-135M-Instruct @ 12fd25f7, token-at-a-time against one
   pass, over T = 8 / 32 / 128 / 512::

       max |Δlogit|   1.2e-04      (relative to the logit range: 3.8e-06)
       mean |Δlogit|  1.4e-05
       max KL         3.0e-10 bits
       argmax         identical at 100% of positions

   The gap does **not** grow with sequence length. Prefill-then-one-step — the
   shape generation actually uses — is tighter still, 4e-05 max-abs.
   ``test_llama_numpy.py`` enforces ``1e-5`` on a tiny 3-layer model (where 30
   layers of amplification are absent) and the measured ``2e-04`` plus a KL
   bound on the real checkpoint. The honest one-line claim is therefore
   *"equivalent to 4 significant figures in the logits and exactly in the
   argmax, not bit-for-bit"* — quoting a bare 1e-5 would be wrong.
2. **Residual hooks.** ``forward(..., resid_hooks={L: fn})`` calls ``fn`` on the
   residual stream *after* block ``L`` and substitutes what it returns. Layer
   ``-1`` is the embedding output, before block 0. This is the seam every
   intervention verb (add / ablate / cap / clamp) is written against, so the
   verbs are model-agnostic and this module knows nothing about them.

Batching. The core is written on ``(B, T, d)`` so a study that needs hundreds of
short generations (phase 3's self-play) can amortise the weight reads across a
batch instead of paying 540 MB of memory traffic per token per conversation.
``forward`` is the ``B == 1`` view of it and is what the hook protocol is
specified on; :meth:`LlamaNumpy.generate_batch` is the batched one.
"""

from __future__ import annotations

import json
import math

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...weights import load_safetensor_f32

#: A residual hook: `(T, d) float32 -> (T, d) float32`. It may return the same
#: array (mutated in place) or a new one; the shape and dtype must match.
ResidHook = Callable[[np.ndarray], np.ndarray]

#: `resid_hooks` key for the embedding output, before block 0.
EMBED_LAYER = -1

#: Positions per prefill step. Bounds the (B, H, T, T_kv) attention block a
#: batched prefill materialises; see `generate_batch`.
PREFILL_CHUNK = 128

_SUPPORTED_MODEL_TYPES = frozenset({"llama", "qwen2", "smollm", "smollm2"})


def _rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """RMSNorm over the last axis. Returns ``(normed, inv)``.

    ``inv = 1/sqrt(mean(x^2) + eps)`` is the per-row scale, returned for the
    same reason ``gpt2_numpy._layernorm`` returns it: the logit lens needs the
    final normalizer to read pre-norm directions honestly.

    The mean-square is accumulated in float64. In float32 a 576-wide row of
    values around 30 (SmolLM2's late layers carry activation outliers that
    large) loses the low bits of the sum, and the resulting normalizer drifts
    from HF's — which computes the same reduction in float32 but on hardware
    that keeps a wider accumulator.
    """
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    inv = (1.0 / np.sqrt(ms + eps)).astype(np.float32)
    return (x * inv) * w, inv[..., 0]


def _silu(x: np.ndarray) -> np.ndarray:
    """x * sigmoid(x), overflow-free.

    ``1/(1+exp(-x))`` overflows for x < -88 in float32 and numpy warns, so the
    two branches of the standard stable sigmoid are evaluated over the whole
    array and selected with :func:`numpy.where`. Writing it the obvious way —
    boolean *fancy indexing* into an output buffer — is bit-for-bit identical
    and **6.7x slower**: on the (14400, 1536) block a batch-48 prefill produces
    it measured 501 ms against 75 ms, which was 16 of the 23 seconds that
    prefill took. ``exp(-|x|)`` never overflows, and the selection recovers the
    two branches exactly, so nothing is approximated here.
    """
    z = np.exp(-np.abs(x))
    s = np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z)).astype(x.dtype, copy=False)
    return x * s


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _softmax_(x: np.ndarray) -> np.ndarray:
    """Softmax over the last axis, **in place**, returning the same array.

    Rows that are entirely -inf (a fully masked query) would be 0/0; the
    attention mask guarantees every query keeps at least its own position, so
    that case cannot arise and is not papered over here.
    """
    x -= x.max(axis=-1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=-1, keepdims=True)
    return x


def _rotate_half(x: np.ndarray) -> np.ndarray:
    """The half-split rotation HF's Llama uses (`rope_interleaved: false`).

    `[x1 | x2] -> [-x2 | x1]`. A model converted for the *interleaved*
    convention rotated this way produces fluent-looking nonsense rather than an
    error, which is why `LlamaNumpy` refuses a config that asks for interleaved
    rather than guessing.
    """
    half = x.shape[-1] // 2
    return np.concatenate((-x[..., half:], x[..., :half]), axis=-1)


@dataclass
class KVCache:
    """Per-layer key/value history for incremental decoding.

    Grown by doubling rather than re-allocated per token: appending with
    ``np.concatenate`` every step is O(T²) memory traffic, which is most of what
    a KV cache exists to remove.
    """

    n_layer: int
    batch: int
    n_kv_head: int
    d_head: int
    t: int = 0
    #: Leading **left-pad** slots per row, or None when every row is real.
    #:
    #: Batched generation over prompts of different lengths left-pads the short
    #: ones so one prefill covers the batch. The pad slots hold real k/v of
    #: filler tokens, so they have to be masked out of attention or a short
    #: prompt would silently read the filler. RoPE makes this exact rather than
    #: approximate: q·k depends only on the *difference* of positions, so
    #: shifting a row's absolute positions by its pad count changes nothing the
    #: model can see.
    pad: np.ndarray | None = None
    _k: list[np.ndarray] = field(default_factory=list)
    _v: list[np.ndarray] = field(default_factory=list)
    _cap: int = 0

    def __post_init__(self) -> None:
        if self.pad is not None:
            self.pad = np.asarray(self.pad, dtype=np.int64)
            if self.pad.shape != (self.batch,):
                raise ValueError("cache pad must be one count per batch row")
            if not self.pad.any():
                self.pad = None
        if not self._k:
            self._reserve(64)

    def _reserve(self, cap: int) -> None:
        shape = (self.batch, self.n_kv_head, cap, self.d_head)
        if not self._k:
            self._k = [np.zeros(shape, dtype=np.float32) for _ in range(self.n_layer)]
            self._v = [np.zeros(shape, dtype=np.float32) for _ in range(self.n_layer)]
        else:
            for L in range(self.n_layer):
                nk = np.zeros(shape, dtype=np.float32)
                nv = np.zeros(shape, dtype=np.float32)
                nk[:, :, : self.t] = self._k[L][:, :, : self.t]
                nv[:, :, : self.t] = self._v[L][:, :, : self.t]
                self._k[L], self._v[L] = nk, nv
        self._cap = cap

    def append(
        self, layer: int, k: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Write this step's k/v for `layer` and return the full history.

        `t` is advanced by the caller (once per step, after the last layer), not
        here, because every layer of one step writes at the same position.
        """
        t_new = k.shape[2]
        need = self.t + t_new
        if need > self._cap:
            self._reserve(max(need, self._cap * 2))
        self._k[layer][:, :, self.t : need] = k
        self._v[layer][:, :, self.t : need] = v
        return self._k[layer][:, :, :need], self._v[layer][:, :, :need]

    def advance(self, n: int) -> None:
        self.t += n

    def slice_batch(self, keep: Sequence[int]) -> "KVCache":
        """A cache holding only the batch rows in `keep`, in that order.

        Batched generation finishes rows at different times; dropping the
        finished ones keeps the per-step matmuls proportional to the work that
        is actually left rather than to the widest the batch ever was.
        """
        idx = list(keep)
        # Allocate the new buffers here and hand them to the constructor rather
        # than calling `_reserve` on a half-built cache: `_reserve`'s grow path
        # copies `[:t]` out of the arrays it is replacing, and on a cache whose
        # `t` was set before any buffer existed those are the 64-slot defaults —
        # which raises as soon as a row finishes after position 64. That is the
        # ordinary case for a conversation, so this path has to be built, not
        # grown.
        cap = max(self._cap, 64)
        shape = (len(idx), self.n_kv_head, cap, self.d_head)
        k: list[np.ndarray] = []
        v: list[np.ndarray] = []
        for L in range(self.n_layer):
            nk = np.zeros(shape, dtype=np.float32)
            nv = np.zeros(shape, dtype=np.float32)
            nk[:, :, : self.t] = self._k[L][idx][:, :, : self.t]
            nv[:, :, : self.t] = self._v[L][idx][:, :, : self.t]
            k.append(nk)
            v.append(nv)
        return KVCache(
            n_layer=self.n_layer,
            batch=len(idx),
            n_kv_head=self.n_kv_head,
            d_head=self.d_head,
            t=self.t,
            pad=None if self.pad is None else self.pad[idx],
            _k=k,
            _v=v,
            _cap=cap,
        )


@dataclass
class Trace:
    """Everything one forward pass reveals — field-for-field the shape
    :class:`nebulai.backend.interp.gpt2_numpy.Trace` has, so a driver or an
    intervention verb written against one works on the other.

    ``ln_f_inv`` is the final *RMS*Norm normalizer here rather than a LayerNorm
    one. It plays the same role (the scale the logit lens divides out) and is
    the same shape; it is not the same quantity, and the field is documented
    rather than renamed because renaming it would fork every consumer.
    """

    tokens: list[int]
    token_strs: list[str]
    resid: np.ndarray  # (n_layer+1, T, d) — stream ENTERING each block, + final
    attn: np.ndarray  # (n_layer, n_head, T, T) post-softmax patterns
    mlp_post: np.ndarray  # (n_layer, T, d_mlp) — silu(gate)*up, what down_proj reads
    logits: np.ndarray  # (T, V)
    ln_f_inv: np.ndarray  # (T,) final RMSNorm normalizer per position

    @property
    def n_layer(self) -> int:
        return self.attn.shape[0]

    @property
    def n_head(self) -> int:
        return self.attn.shape[1]


@dataclass
class Generation:
    """One completion, with the provenance a claim about it would need."""

    prompt_tokens: list[int]
    tokens: list[int]
    text: str
    finish_reason: str  # "stop" | "length"
    seed: int | None
    temperature: float
    top_p: float


_CHATML_MARKERS = ("<|im_start|>", "<|im_end|>")


class LlamaNumpy:
    """A Llama-family model whose every matmul is visible.

    Construct once per model (weights stay resident, float32), then call
    :meth:`forward` per prompt or :meth:`generate` / :meth:`generate_batch`.

    Weight layout note: HF stores a ``Linear`` as ``(out, in)`` and applies
    ``x @ W.T``. Everything here is pre-transposed to ``(in, out)`` at load and
    applied as ``x @ W``, and q/k/v (and gate/up) are concatenated into one
    matrix per layer. Both are for speed, not taste: a decode step is
    memory-bound on the weight read, and three separate ``(576, …)`` matmuls
    read the same activation three times.
    """

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        *,
        revision: str = "main",
        local_dir: str | Path | None = None,
    ) -> None:
        if local_dir is not None:
            root = Path(local_dir)
            cfg = json.loads((root / "config.json").read_text())
            weights_path = root / "model.safetensors"
            tok_path = root / "tokenizer.json"
            tok_cfg_path = root / "tokenizer_config.json"
            # A local directory has no revision of its own, so the caller's
            # `revision=` is taken as the *claim* about which commit those
            # bytes are — it is what lands in every artifact's provenance.
            # "local" is recorded only when no claim was made, so an
            # unattributable run is visibly unattributable rather than
            # quietly labelled `main`.
            resolved = "local" if revision == "main" else revision
        else:
            from huggingface_hub import hf_hub_download

            cfg_path = hf_hub_download(model_id, "config.json", revision=revision)
            cfg = json.loads(Path(cfg_path).read_text())
            weights_path = Path(
                hf_hub_download(model_id, "model.safetensors", revision=revision)
            )
            tok_path = Path(hf_hub_download(model_id, "tokenizer.json", revision=revision))
            tok_cfg_path = Path(
                hf_hub_download(model_id, "tokenizer_config.json", revision=revision)
            )
            resolved = _sha_from_snapshot_path(cfg_path) or revision

        arch = str(cfg.get("model_type", ""))
        if arch not in _SUPPORTED_MODEL_TYPES:
            raise NotImplementedError(
                f"llama_numpy covers the Llama/Qwen2 shape (RMSNorm + RoPE + SwiGLU "
                f"+ GQA); {model_id!r} is model_type {arch!r}. Refusing to run a "
                f"wrong architecture — a forward pass that is merely plausible is "
                f"worse than one that raises."
            )
        if cfg.get("rope_interleaved"):
            raise NotImplementedError(
                "rope_interleaved=true needs the interleaved RoPE layout; this "
                "module implements the half-split one HF converts Llama weights "
                "for. Rotating the wrong way produces fluent nonsense, not an "
                "error, so this refuses rather than guesses."
            )
        if cfg.get("rope_scaling"):
            raise NotImplementedError(
                f"rope_scaling={cfg['rope_scaling']!r} is not implemented; "
                "positions beyond the base context would be silently wrong."
            )

        self.model_id = model_id
        self.revision = resolved
        self.n_layer = int(cfg["num_hidden_layers"])
        self.n_head = int(cfg["num_attention_heads"])
        self.n_kv_head = int(cfg.get("num_key_value_heads", self.n_head))
        self.d = int(cfg["hidden_size"])
        self.d_head = int(cfg.get("head_dim", self.d // self.n_head))
        self.d_mlp = int(cfg["intermediate_size"])
        self.n_ctx = int(cfg.get("max_position_embeddings", 2048))
        self.eps = float(cfg.get("rms_norm_eps", 1e-5))
        self.rope_theta = float(cfg.get("rope_theta", 10000.0))
        self.tied = bool(cfg.get("tie_word_embeddings", False))
        self.eos_token_id = _as_int(cfg.get("eos_token_id"))
        self.bos_token_id = _as_int(cfg.get("bos_token_id"))
        self.config = cfg

        if self.n_head % self.n_kv_head:
            raise ValueError(
                f"num_attention_heads {self.n_head} is not a multiple of "
                f"num_key_value_heads {self.n_kv_head}"
            )
        self.kv_repeat = self.n_head // self.n_kv_head

        t = load_safetensor_f32(weights_path)
        # Some checkpoints prefix everything with "model."; some nest under
        # "model.language_model." (multimodal wrappers). Resolve by suffix the
        # way RemoteCheckpoint does, rather than by an exact-name table that
        # fails on the next family.
        self._raw = t
        self.embed = self._take("embed_tokens.weight")  # (V, d)
        self.V = int(self.embed.shape[0])
        if self.embed.shape[1] != self.d:
            raise ValueError(
                f"embed_tokens width {self.embed.shape[1]} != hidden_size {self.d}"
            )
        self.norm_w = self._take("model.norm.weight", "norm.weight")
        if self.tied:
            self.W_U = self.embed.T.copy()  # (d, V)
        else:
            self.W_U = self._take("lm_head.weight").T.copy()
        if self.W_U.shape != (self.d, self.V):
            raise ValueError(f"unembedding is {self.W_U.shape}, expected {(self.d, self.V)}")

        self.layers: list[dict[str, np.ndarray | None]] = []
        for L in range(self.n_layer):
            self.layers.append(self._load_layer(L))
        self._raw = {}  # release the untransposed copies

        # RoPE tables, precomputed to the full context once.
        inv_freq = 1.0 / (
            self.rope_theta
            ** (np.arange(0, self.d_head, 2, dtype=np.float64) / self.d_head)
        )
        pos = np.arange(self.n_ctx, dtype=np.float64)[:, None]
        ang = pos * inv_freq[None, :]  # (n_ctx, dh/2)
        emb = np.concatenate([ang, ang], axis=-1)  # half-split duplication
        self._cos = np.cos(emb).astype(np.float32)  # (n_ctx, dh)
        self._sin = np.sin(emb).astype(np.float32)

        self.tok = _load_tokenizer(tok_path)
        self._tok_cfg = json.loads(tok_cfg_path.read_text()) if tok_cfg_path.exists() else {}
        self._chat_template = self._tok_cfg.get("chat_template")
        eos_str = self._tok_cfg.get("eos_token")
        if isinstance(eos_str, dict):
            eos_str = eos_str.get("content")
        self.eos_token = eos_str

    # ── weight loading ───────────────────────────────────────────────────

    def _take(self, *suffixes: str) -> np.ndarray:
        for suffix in suffixes:
            if suffix in self._raw:
                return self._raw[suffix]
        for suffix in suffixes:
            hits = sorted(k for k in self._raw if k.endswith(suffix))
            if hits:
                return self._raw[hits[0]]
        raise KeyError(
            f"no tensor ending in any of {suffixes} in {self.model_id}; "
            f"have e.g. {sorted(self._raw)[:6]}"
        )

    def _opt(self, *suffixes: str) -> np.ndarray | None:
        try:
            return self._take(*suffixes)
        except KeyError:
            return None

    def _load_layer(self, L: int) -> dict[str, np.ndarray | None]:
        p = f"layers.{L}."
        q = self._take(p + "self_attn.q_proj.weight")
        k = self._take(p + "self_attn.k_proj.weight")
        v = self._take(p + "self_attn.v_proj.weight")
        want_q = self.n_head * self.d_head
        want_kv = self.n_kv_head * self.d_head
        for name, mat, want in (("q", q, want_q), ("k", k, want_kv), ("v", v, want_kv)):
            if mat.shape != (want, self.d):
                raise ValueError(
                    f"layer {L} {name}_proj is {mat.shape}, expected {(want, self.d)} "
                    f"— a transposed or mis-shaped weight is the failure this check exists for"
                )
        gate = self._take(p + "mlp.gate_proj.weight")
        up = self._take(p + "mlp.up_proj.weight")
        down = self._take(p + "mlp.down_proj.weight")
        qb = self._opt(p + "self_attn.q_proj.bias")
        kb = self._opt(p + "self_attn.k_proj.bias")
        vb = self._opt(p + "self_attn.v_proj.bias")
        qkv_bias = (
            np.concatenate([qb, kb, vb]).astype(np.float32)
            if qb is not None and kb is not None and vb is not None
            else None
        )
        return {
            # (d, H*dh + 2*kv*dh) — one read of x serves q, k and v
            "qkv": np.ascontiguousarray(np.concatenate([q, k, v], axis=0).T),
            "qkv_bias": qkv_bias,
            "o": np.ascontiguousarray(self._take(p + "self_attn.o_proj.weight").T),
            # (d, 2*d_mlp) — same trick for gate/up
            "gate_up": np.ascontiguousarray(np.concatenate([gate, up], axis=0).T),
            "down": np.ascontiguousarray(down.T),
            "ln1": self._take(p + "input_layernorm.weight"),
            "ln2": self._take(p + "post_attention_layernorm.weight"),
        }

    # ── tokenizer ────────────────────────────────────────────────────────

    def encode(self, text: str, *, add_special: bool = False) -> list[int]:
        ids = self.tok.encode(text, add_special_tokens=add_special).ids
        return list(ids)

    def decode1(self, tid: int) -> str:
        return self.tok.decode([int(tid)], skip_special_tokens=False)

    def decode(self, ids: Sequence[int], *, skip_special: bool = True) -> str:
        return self.tok.decode([int(i) for i in ids], skip_special_tokens=skip_special)

    def apply_chat_template(
        self, messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool = True
    ) -> str:
        """Render a ChatML conversation exactly as the checkpoint's own template does.

        Implemented directly rather than through jinja2 (a dependency the base
        install does not have) and *verified* against the template the
        tokenizer shipped: if the checkpoint's ``chat_template`` is not the
        ChatML one this renders, it raises. A quietly wrong chat format would
        move every persona coordinate and every self-play transcript without
        failing anything.
        """
        tmpl = self._chat_template
        if isinstance(tmpl, list):  # transformers>=4.43 allows a list of named templates
            tmpl = next(
                (t.get("template") for t in tmpl if t.get("name") == "default"),
                tmpl[0].get("template") if tmpl else None,
            )
        if not tmpl or not all(m in tmpl for m in _CHATML_MARKERS):
            raise NotImplementedError(
                f"{self.model_id} does not ship a ChatML chat_template; this "
                f"renderer only implements ChatML and refuses to guess a format "
                f"the checkpoint was not trained on."
            )
        out: list[str] = []
        for m in messages:
            role, content = m["role"], m["content"]
            out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n")
        return "".join(out)

    # ── forward ──────────────────────────────────────────────────────────

    def new_cache(self, batch: int = 1, *, pad: Sequence[int] | None = None) -> KVCache:
        return KVCache(
            n_layer=self.n_layer,
            batch=batch,
            n_kv_head=self.n_kv_head,
            d_head=self.d_head,
            pad=None if pad is None else np.asarray(pad, dtype=np.int64),
        )

    def forward(
        self,
        tokens: str | Sequence[int],
        *,
        resid_hooks: Mapping[int, ResidHook] | None = None,
        cache: KVCache | None = None,
    ) -> Trace:
        """One real forward pass over `tokens`, returning every internal.

        `resid_hooks` maps a layer index to a callable receiving the residual
        stream **after** that block as ``(T, d) float32`` and returning the
        replacement. Layer ``-1`` (:data:`EMBED_LAYER`) is the embedding
        output, before block 0. The hook is called once per forward, with the
        positions this call processes — which for a cached incremental step is
        the *new* positions only, not the whole history.

        `cache` is advanced in place when given. A traced cached call returns
        arrays covering only the new positions; `attn`'s last axis still spans
        the full history, because that is what the model attended to.
        """
        ids = self.encode(tokens) if isinstance(tokens, str) else [int(i) for i in tokens]
        if not ids:
            raise ValueError("empty prompt (tokenizes to zero tokens)")
        pos0 = cache.t if cache is not None else 0
        if pos0 + len(ids) > self.n_ctx:
            raise ValueError(
                f"{pos0 + len(ids)} positions exceeds the model's context of "
                f"{self.n_ctx}; RoPE has no table beyond it and extrapolating "
                f"would be a silent lie"
            )
        out = self._core(
            np.asarray([ids], dtype=np.int64),
            cache=cache,
            resid_hooks=resid_hooks,
            want_trace=True,
        )
        return Trace(
            tokens=ids,
            token_strs=[self.decode1(i) for i in ids],
            resid=out["resid"][:, 0],
            attn=out["attn"][:, 0],
            mlp_post=out["mlp_post"][:, 0],
            logits=out["logits"][0],
            ln_f_inv=out["ln_f_inv"][0],
        )

    def logit_lens(self, resid_row: np.ndarray) -> np.ndarray:
        """The final RMSNorm + unembedding applied to any residual vector.

        The honest logit lens: the model's own readout head, no trained
        translator (that would be the tuned lens).
        """
        xf, _ = _rms_norm(np.asarray(resid_row, dtype=np.float32)[None, :], self.norm_w, self.eps)
        return (xf @ self.W_U)[0]

    def capture_resid(
        self,
        prompts: Sequence[Sequence[int]] | Sequence[str],
        layers: Sequence[int],
        *,
        batch_size: int = 16,
    ) -> dict[int, np.ndarray]:
        """Residual stream at each prompt's **last real token**, per layer.

        Returns ``{layer: (n_prompts, d) float32}``. Layer ``L`` means "after
        block L" (and :data:`EMBED_LAYER` means the embedding output), matching
        the `resid_hooks` indexing exactly, so a direction extracted here and a
        hook that adds it back speak about the same place.

        Prompts are **right-padded** within a batch. That is exact rather than
        approximate: attention is causal, so a real position never reads a
        padded one, and the padded rows only ever affect logits at positions
        nobody looks at. This is what makes the persona sweep (hundreds of
        prompts) one matmul per batch instead of one per prompt.
        """
        seqs = [
            self.encode(p) if isinstance(p, str) else [int(i) for i in p] for p in prompts
        ]
        if any(not s for s in seqs):
            raise ValueError("empty prompt in capture_resid")
        want = sorted({int(x) for x in layers})
        bad = [x for x in want if x != EMBED_LAYER and not 0 <= x < self.n_layer]
        if bad:
            raise ValueError(f"layers {bad} out of range for a {self.n_layer}-layer model")
        out = {L: np.empty((len(seqs), self.d), dtype=np.float32) for L in want}
        pad = self.eos_token_id if self.eos_token_id is not None else 0

        for start in range(0, len(seqs), batch_size):
            chunk = seqs[start : start + batch_size]
            width = max(len(s) for s in chunk)
            if width > self.n_ctx:
                raise ValueError(f"prompt of {width} tokens exceeds context {self.n_ctx}")
            ids = np.full((len(chunk), width), pad, dtype=np.int64)
            for i, s in enumerate(chunk):
                ids[i, : len(s)] = s
            # B>1 dispatches the hook per row, so collect per row and restack
            per_row: dict[int, list[np.ndarray]] = {L: [] for L in want}

            def _grab_row(L: int):
                def fn(x: np.ndarray) -> np.ndarray:
                    per_row[L].append(x)
                    return x

                return fn

            hooks = {L: _grab_row(L) for L in want}
            self._core(ids, cache=None, resid_hooks=hooks, want_trace=False, last_only=True)
            for L in want:
                rows = per_row[L]
                if len(rows) != len(chunk):  # B == 1 path calls the hook once
                    rows = [rows[0]]
                for i, s in enumerate(chunk):
                    out[L][start + i] = rows[i][len(s) - 1]
        return out

    def _core(
        self,
        ids: np.ndarray,  # (B, T) int
        *,
        cache: KVCache | None,
        resid_hooks: Mapping[int, ResidHook] | None,
        want_trace: bool,
        last_only: bool = False,
    ) -> dict[str, np.ndarray]:
        B, T = ids.shape
        d, H, KV, dh, R = self.d, self.n_head, self.n_kv_head, self.d_head, self.kv_repeat
        pos0 = cache.t if cache is not None else 0
        hooks = resid_hooks or {}

        x = self.embed[ids]  # (B, T, d)
        x = self._hook(x, hooks, EMBED_LAYER)

        cos = self._cos[pos0 : pos0 + T]  # (T, dh)
        sin = self._sin[pos0 : pos0 + T]

        # The causal mask is the same for every layer, so build it once. Rows
        # with left-padding get their own: a real query must not read another
        # row's filler. A *pad* query is allowed to read its own position so the
        # softmax it computes has at least one finite entry — without that the
        # pad rows produce NaNs which, while never attended to, are a poor thing
        # to have flowing through a forward pass.
        mask: np.ndarray | None = None
        blocked: np.ndarray | None = None  # ~mask, built once for all layers
        pad = cache.pad if cache is not None else None
        if pad is not None:
            T_kv_total = pos0 + T
            qpos = np.arange(pos0, pos0 + T)[:, None]
            kpos = np.arange(T_kv_total)[None, :]
            causal = kpos <= qpos  # (T, T_kv)
            real_key = kpos[None, :, :] >= pad[:, None, None]  # (B, 1, T_kv)
            mask = causal[None, None] & (
                real_key[:, :, None, :] | (kpos == qpos)[None, None]
            )

        resid = np.empty((self.n_layer + 1, B, T, d), dtype=np.float32) if want_trace else None
        mlp_post = (
            np.empty((self.n_layer, B, T, self.d_mlp), dtype=np.float32) if want_trace else None
        )
        attn_out_store: list[np.ndarray] = []

        n_q = H * dh
        n_kv = KV * dh

        for L in range(self.n_layer):
            w = self.layers[L]
            if want_trace:
                resid[L] = x
            xn, _ = _rms_norm(x, w["ln1"], self.eps)
            qkv = xn.reshape(B * T, d) @ w["qkv"]
            if w["qkv_bias"] is not None:
                qkv = qkv + w["qkv_bias"]
            q = qkv[:, :n_q].reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            k = qkv[:, n_q : n_q + n_kv].reshape(B, T, KV, dh).transpose(0, 2, 1, 3)
            v = qkv[:, n_q + n_kv :].reshape(B, T, KV, dh).transpose(0, 2, 1, 3)

            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin

            if cache is not None:
                k, v = cache.append(L, np.ascontiguousarray(k), np.ascontiguousarray(v))
            T_kv = k.shape[2]

            # Grouped-query attention by *reshape*, never by `np.repeat`. Query
            # head h reads kv head h // R, so folding the R queries of a group
            # into the time axis — (B, KV, R*T, dh) — lets one matmul per kv
            # head serve the whole group. `np.repeat` would materialise a copy
            # of the entire k and v history (at batch 48 and a 340-token
            # context that is ~1 GB of memory traffic per decode step, which
            # measured as an 8x slowdown). The arithmetic is identical: heads
            # are still repeated, never averaged.
            qg = q.reshape(B, KV, R * T, dh)
            scores = (qg @ k.transpose(0, 1, 3, 2)) / math.sqrt(dh)  # (B,KV,R*T,T_kv)
            scores = scores.reshape(B, H, T, T_kv)
            # query i sits at absolute position pos0+i and may see key j <= pos0+i
            if mask is None:
                qpos = np.arange(pos0, pos0 + T)[:, None]
                kpos = np.arange(T_kv)[None, :]
                mask = kpos <= qpos
            elif mask.shape[-1] != T_kv:  # pragma: no cover - shape invariant
                raise AssertionError("attention mask and kv history disagree")
            if blocked is None:
                blocked = ~mask
            # In place from here down. `scores` is a fresh array this call owns,
            # and at a 300-token prefill with batch 48 it is 155 MB per layer:
            # every temporary numpy would otherwise allocate (the mask copy,
            # the shifted exponent, the normalised result) is another 155 MB of
            # memory traffic, thirty times over.
            np.copyto(scores, np.float32(-np.inf), where=blocked)
            a = _softmax_(scores)
            if want_trace:
                attn_out_store.append(a)
            ctx = (a.reshape(B, KV, R * T, T_kv) @ v).reshape(B, H, T, dh)
            ctx = ctx.transpose(0, 2, 1, 3).reshape(B * T, n_q)
            x = x + (ctx @ w["o"]).reshape(B, T, d)

            xn2, _ = _rms_norm(x, w["ln2"], self.eps)
            gu = xn2.reshape(B * T, d) @ w["gate_up"]
            h = _silu(gu[:, : self.d_mlp]) * gu[:, self.d_mlp :]
            if want_trace:
                mlp_post[L] = h.reshape(B, T, self.d_mlp)
            x = x + (h @ w["down"]).reshape(B, T, d)

            x = self._hook(x, hooks, L)

        if cache is not None:
            cache.advance(T)
        if want_trace:
            resid[self.n_layer] = x

        tail = x[:, -1:, :] if last_only else x
        xf, inv = _rms_norm(tail, self.norm_w, self.eps)
        logits = (xf.reshape(-1, d) @ self.W_U).reshape(B, tail.shape[1], self.V)

        out: dict[str, Any] = {"logits": logits.astype(np.float32), "ln_f_inv": inv}
        if want_trace:
            out["resid"] = resid
            out["mlp_post"] = mlp_post
            # attention histories are ragged only when a cache is in play; for
            # the uncached path every layer has the same (B,H,T,T) shape
            out["attn"] = np.stack(attn_out_store, axis=0)
        return out

    @staticmethod
    def _hook(x: np.ndarray, hooks: Mapping[int, ResidHook], layer: int) -> np.ndarray:
        fn = hooks.get(layer)
        if fn is None:
            return x
        B = x.shape[0]
        if B == 1:
            # the documented protocol: the hook sees (T, d)
            got = np.asarray(fn(x[0]), dtype=np.float32)
            if got.shape != x.shape[1:]:
                raise ValueError(
                    f"resid hook for layer {layer} returned {got.shape}, expected {x.shape[1:]}"
                )
            return got[None, :, :]
        rows = []
        for b in range(B):
            got = np.asarray(fn(x[b]), dtype=np.float32)
            if got.shape != x.shape[1:]:
                raise ValueError(
                    f"resid hook for layer {layer} returned {got.shape}, expected {x.shape[1:]}"
                )
            rows.append(got)
        return np.stack(rows, axis=0)

    # ── generation ───────────────────────────────────────────────────────

    def generate(
        self,
        prompt_tokens: str | Sequence[int],
        max_new_tokens: int = 64,
        *,
        temperature: float = 0.8,
        top_p: float = 0.95,
        seed: int | None = None,
        resid_hooks: Mapping[int, ResidHook] | None = None,
        stop_tokens: Sequence[int] | None = None,
    ) -> Generation:
        """Sample a completion, using the KV cache.

        `temperature == 0` is greedy and ignores `top_p` and `seed`; the
        returned :class:`Generation` still records all three so a figure can
        state what was asked for as well as what happened.
        """
        ids = (
            self.encode(prompt_tokens)
            if isinstance(prompt_tokens, str)
            else [int(i) for i in prompt_tokens]
        )
        outs = self.generate_batch(
            [ids],
            max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            resid_hooks=resid_hooks,
            stop_tokens=stop_tokens,
        )
        return outs[0]

    def generate_batch(
        self,
        prompts: Sequence[Sequence[int]] | Sequence[str],
        max_new_tokens: int = 64,
        *,
        temperature: float = 0.8,
        top_p: float = 0.95,
        seed: int | None = None,
        resid_hooks: Mapping[int, ResidHook] | None = None,
        stop_tokens: Sequence[int] | None = None,
    ) -> list[Generation]:
        """Sample `len(prompts)` completions at once.

        Prompts of different lengths are prefilled one at a time (left-padding
        would need a padding mask through every attention, and the prefill is a
        small fraction of a long generation's cost); the *decode* steps — which
        are where the memory traffic is — run as one batch. Rows that hit a
        stop token drop out of the batch and the remaining ones keep going at
        the narrower width.
        """
        seqs = [
            self.encode(p) if isinstance(p, str) else [int(i) for i in p] for p in prompts
        ]
        if not seqs:
            return []
        if any(not s for s in seqs):
            raise ValueError("empty prompt in batch")
        B = len(seqs)
        stops = set(int(s) for s in (stop_tokens or []))
        if self.eos_token_id is not None:
            stops.add(int(self.eos_token_id))
        rng = np.random.default_rng(seed)

        longest = max(len(s) for s in seqs)
        if longest + max_new_tokens > self.n_ctx:
            raise ValueError(
                f"{longest} prompt + {max_new_tokens} new exceeds context {self.n_ctx}"
            )

        # One prefill for the whole batch, **left-padding** the short rows. The
        # pad slots are masked out of every attention (see `KVCache.pad`), and
        # RoPE's dependence on position *differences* alone means a row that
        # starts at absolute position `pad_b` computes the same scores it would
        # have at position 0. The alternative — generating the short rows one at
        # a time — is what this replaces, and it cost a self-play study roughly
        # twenty-fold: a ragged batch is the normal case once a conversation has
        # more than one turn in it, not an edge case.
        #
        # Padded prefill is not bit-identical to an unpadded one: the absolute
        # positions differ, so cos/sin are different float32 numbers. Measured
        # on SmolLM2-135M-Instruct the logit gap is of the same order as the KV
        # cache's own (~1e-4 max-abs) with identical argmax; the tests pin both.
        longest_seq = max(len(s) for s in seqs)
        pads = [longest_seq - len(s) for s in seqs]
        pad_id = int(self.eos_token_id) if self.eos_token_id is not None else 0
        cache = self.new_cache(B, pad=pads)
        ids = np.asarray(
            [[pad_id] * pads[i] + list(s) for i, s in enumerate(seqs)], dtype=np.int64
        )
        # Prefill in chunks. The attention score block is (B, H, T, T_kv), so a
        # whole-prompt prefill grows as T squared in memory: batch 96 over 300
        # tokens is 518 MB per layer and the machine starts swapping (measured:
        # 528 positions/s at batch 96 undivided against 1022 at batch 48).
        # Chunking caps that block and, because the mask is causal, also skips
        # roughly half the score matrix.
        for a0 in range(0, ids.shape[1], PREFILL_CHUNK):
            out = self._core(
                ids[:, a0 : a0 + PREFILL_CHUNK],
                cache=cache,
                resid_hooks=resid_hooks,
                want_trace=False,
                last_only=True,
            )
        last = out["logits"][:, -1, :]
        active = list(range(B))  # which conversations are still rows of the batch
        done = [False] * B
        gen: list[list[int]] = [[] for _ in range(B)]
        finish = ["length"] * B
        for _ in range(max_new_tokens):
            nxt = _sample(last, temperature, top_p, rng)
            for row, b in enumerate(active):
                if done[b]:
                    continue
                tid = int(nxt[row])
                if tid in stops:
                    finish[b] = "stop"
                    done[b] = True
                    continue
                gen[b].append(tid)
            live = [row for row, b in enumerate(active) if not done[b]]
            if not live:
                break
            # Narrowing the batch copies the entire KV history, so do it only
            # when it actually pays: dropping one row of forty-eight at a
            # 340-token context costs ~1 GB of memory traffic to save 2 % of a
            # matmul. Finished rows keep decoding until the batch is worth
            # rebuilding; their tokens are simply not recorded. This threshold
            # measured as the difference between 37 and 100+ tok/s on the
            # self-play shape.
            if len(live) <= 0.6 * len(active):
                cache = cache.slice_batch(live)
                nxt = nxt[live]
                active = [active[row] for row in live]
            out = self._core(
                nxt.reshape(-1, 1).astype(np.int64),
                cache=cache,
                resid_hooks=resid_hooks,
                want_trace=False,
                last_only=True,
            )
            last = out["logits"][:, -1, :]

        return [
            Generation(
                prompt_tokens=seqs[b],
                tokens=gen[b],
                text=self.decode(gen[b]),
                finish_reason=finish[b],
                seed=seed,
                temperature=temperature,
                top_p=top_p,
            )
            for b in range(B)
        ]


def _sample(
    logits: np.ndarray, temperature: float, top_p: float, rng: np.random.Generator
) -> np.ndarray:
    """Nucleus sampling over a (B, V) logit block. temperature 0 = greedy."""
    if temperature <= 0:
        return logits.argmax(axis=-1)
    lg = logits.astype(np.float64) / float(temperature)
    lg -= lg.max(axis=-1, keepdims=True)
    p = np.exp(lg)
    p /= p.sum(axis=-1, keepdims=True)
    if top_p < 1.0:
        order = np.argsort(-p, axis=-1)
        sp = np.take_along_axis(p, order, axis=-1)
        cum = np.cumsum(sp, axis=-1)
        # keep the smallest prefix whose mass reaches top_p (always >= 1 token)
        cut = cum - sp >= top_p
        sp = np.where(cut, 0.0, sp)
        sp /= sp.sum(axis=-1, keepdims=True)
        picks = np.empty(p.shape[0], dtype=np.int64)
        for b in range(p.shape[0]):
            picks[b] = order[b, rng.choice(sp.shape[1], p=sp[b])]
        return picks
    return np.asarray([rng.choice(p.shape[1], p=p[b]) for b in range(p.shape[0])])


def _as_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, (list, tuple)) and v:
        return _as_int(v[0])
    return None


def _sha_from_snapshot_path(path: str | Path) -> str | None:
    """The commit sha out of an HF cache path, so provenance works offline.

    `hf_hub_download` resolves into `…/snapshots/<sha>/<file>`; lifting the sha
    back out avoids a second API call (and works with `HF_HUB_OFFLINE=1`, which
    `model_info` does not).
    """
    m = re.search(r"/snapshots/([0-9a-f]{6,40})/", str(path))
    return m.group(1) if m else None


def _load_tokenizer(path: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def tokens_per_second(
    model: LlamaNumpy, *, prompt: str = "The capital of France is", n: int = 32
) -> float:
    """Measured decode throughput, for the numbers a plan has to state."""
    import time

    ids = model.encode(prompt)
    cache = model.new_cache(1)
    model._core(
        np.asarray([ids], dtype=np.int64), cache=cache, resid_hooks=None,
        want_trace=False, last_only=True,
    )
    nxt = np.asarray([[ids[-1]]], dtype=np.int64)
    t0 = time.perf_counter()
    for _ in range(n):
        out = model._core(
            nxt, cache=cache, resid_hooks=None, want_trace=False, last_only=True
        )
        nxt = out["logits"][:, -1, :].argmax(axis=-1).reshape(1, 1).astype(np.int64)
    return n / (time.perf_counter() - t0)


__all__ = [
    "EMBED_LAYER",
    "Generation",
    "KVCache",
    "LlamaNumpy",
    "ResidHook",
    "Trace",
    "tokens_per_second",
]


if __name__ == "__main__":  # pragma: no cover - a hand check, not a test
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=32)
    a = ap.parse_args()
    m = LlamaNumpy(a.model)
    print(f"{m.model_id} @ {m.revision[:8]} — {m.n_layer}L d={m.d} V={m.V}")
    g = m.generate(a.prompt, a.n, temperature=0.0)
    print(repr(a.prompt + g.text))
    print(f"{tokens_per_second(m, prompt=a.prompt, n=a.n):.1f} tok/s")

