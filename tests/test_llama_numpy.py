"""Tests for the numpy Llama forward pass.

Almost everything here runs against a tiny random checkpoint built by
`tiny_llama.build` — a real safetensors/config/tokenizer triple loaded through
the ordinary path, not a mock. The properties under test are properties of the
wiring (causal masking, RoPE placement, GQA head expansion, KV-cache
equivalence, hook dispatch), and random weights expose a wiring bug exactly as
loudly as trained ones.

The one claim random weights cannot support is "this implements *SmolLM2*".
That is `test_real_smollm2_matches_reference`, marked `network`, which pins the
resolved commit sha and checks the model's own greedy continuation.
"""

from __future__ import annotations

import json
import os
import math
from pathlib import Path

import numpy as np
import pytest

from nebulai.backend.interp.llama_numpy import (
    EMBED_LAYER,
    Generation,
    KVCache,
    LlamaNumpy,
    Trace,
    tokens_per_second,
)

import tiny_llama

#: Cached-vs-uncached logits are **not** bit-for-bit: a cached step multiplies
#: a (1, d) activation where the uncached path multiplies a (T, d) block, so
#: BLAS picks a different kernel and a different summation order, and each
#: residual addition carries the difference forward.
#:
#: The tiny fixture has 3 layers, so there is almost no amplification and 1e-5
#: holds there. The real 30-layer checkpoint does not meet 1e-5 on raw logits —
#: measured 1.2e-4 max-abs, flat in T — so the network test enforces the number
#: that was actually measured plus a KL bound, rather than the number that
#: would have been nicer to quote.
KV_TOL = 1e-5
KV_TOL_REAL = 2e-4
KV_KL_BITS_REAL = 1e-8


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> LlamaNumpy:
    path = tiny_llama.build(tmp_path_factory.mktemp("tiny") / "m")
    return LlamaNumpy("tiny/llama-test", local_dir=path)


@pytest.fixture(scope="module")
def tiny_untied(tmp_path_factory) -> LlamaNumpy:
    path = tiny_llama.build(
        tmp_path_factory.mktemp("tiny_untied") / "m", tied=False, seed=11
    )
    return LlamaNumpy("tiny/llama-untied", local_dir=path)


# ── shape and provenance ────────────────────────────────────────────────────


def test_loads_and_reports_geometry(tiny: LlamaNumpy) -> None:
    assert (tiny.n_layer, tiny.n_head, tiny.n_kv_head, tiny.d) == (3, 4, 2, 32)
    assert tiny.kv_repeat == 2
    assert tiny.tied is True
    assert tiny.W_U.shape == (tiny.d, tiny.V)


def test_local_dir_without_a_revision_claim_records_local(tiny: LlamaNumpy) -> None:
    # An unattributable load must look unattributable, never like `main`.
    assert tiny.revision == "local"


def test_local_dir_with_a_revision_claim_records_it(tmp_path: Path) -> None:
    path = tiny_llama.build(tmp_path / "m")
    m = LlamaNumpy("tiny/llama-test", revision="deadbeefcafe", local_dir=path)
    assert m.revision == "deadbeefcafe"


def test_untied_unembedding_is_not_the_embedding(tiny_untied: LlamaNumpy) -> None:
    assert not np.allclose(tiny_untied.W_U, tiny_untied.embed.T)
    tr = tiny_untied.forward(tiny_llama.words(6))
    assert tr.logits.shape == (6, tiny_untied.V)


def test_trace_has_the_gpt2_shape(tiny: LlamaNumpy) -> None:
    ids = tiny.encode(tiny_llama.words(7))
    tr = tiny.forward(ids)
    assert isinstance(tr, Trace)
    assert tr.resid.shape == (tiny.n_layer + 1, 7, tiny.d)
    assert tr.attn.shape == (tiny.n_layer, tiny.n_head, 7, 7)
    assert tr.mlp_post.shape == (tiny.n_layer, 7, tiny.d_mlp)
    assert tr.logits.shape == (7, tiny.V)
    assert tr.ln_f_inv.shape == (7,)
    assert tr.token_strs and len(tr.token_strs) == 7


def test_logit_lens_of_the_last_resid_row_is_the_logits(tiny: LlamaNumpy) -> None:
    # The identity that makes the lens honest: applied to the *final* residual
    # it reproduces the model's own logits exactly, so any earlier layer's
    # reading is the same readout head and not a different function.
    tr = tiny.forward(tiny_llama.words(5))
    lens = tiny.logit_lens(tr.resid[tiny.n_layer, -1])
    assert np.allclose(lens, tr.logits[-1], atol=1e-4)


def test_refuses_an_unsupported_architecture(tmp_path: Path) -> None:
    path = tiny_llama.build(tmp_path / "m", model_type="gpt2")
    with pytest.raises(NotImplementedError, match="model_type"):
        LlamaNumpy("tiny/not-llama", local_dir=path)


def test_refuses_interleaved_rope(tmp_path: Path) -> None:
    path = tiny_llama.build(tmp_path / "m", rope_interleaved=True)
    with pytest.raises(NotImplementedError, match="interleaved"):
        LlamaNumpy("tiny/interleaved", local_dir=path)


def test_refuses_rope_scaling(tmp_path: Path) -> None:
    path = tiny_llama.build(tmp_path / "m", rope_scaling={"type": "linear", "factor": 4})
    with pytest.raises(NotImplementedError, match="rope_scaling"):
        LlamaNumpy("tiny/scaled", local_dir=path)


def test_refuses_past_the_context_edge(tiny: LlamaNumpy) -> None:
    with pytest.raises(ValueError, match="context"):
        tiny.forward(list(range(1, 2)) * (tiny.n_ctx + 1))


# ── causal masking ──────────────────────────────────────────────────────────


def test_attention_is_strictly_causal(tiny: LlamaNumpy) -> None:
    tr = tiny.forward(tiny_llama.words(9))
    T = len(tr.tokens)
    upper = np.triu(np.ones((T, T), dtype=bool), k=1)
    assert np.all(tr.attn[:, :, upper] == 0.0), "a query attended to a later key"
    rows = tr.attn.sum(axis=-1)
    assert np.allclose(rows, 1.0, atol=1e-5), "attention rows are not distributions"


def test_a_later_token_cannot_change_an_earlier_prediction(tiny: LlamaNumpy) -> None:
    # The behavioural form of the same claim: truncating the suffix must leave
    # every remaining position's logits identical.
    ids = tiny.encode(tiny_llama.words(10))
    full = tiny.forward(ids)
    short = tiny.forward(ids[:6])
    assert np.allclose(full.logits[:6], short.logits, atol=1e-5)


# ── RoPE ────────────────────────────────────────────────────────────────────


def test_rope_is_identity_at_position_zero(tiny: LlamaNumpy) -> None:
    assert np.allclose(tiny._cos[0], 1.0)
    assert np.allclose(tiny._sin[0], 0.0)


def test_rope_tables_are_the_half_split_layout(tiny: LlamaNumpy) -> None:
    # HF's Llama conversion duplicates the angles across the two halves rather
    # than interleaving them; rotating the other way is fluent nonsense, not an
    # error, so the layout is asserted rather than assumed.
    half = tiny.d_head // 2
    assert np.allclose(tiny._cos[:, :half], tiny._cos[:, half:])
    assert np.allclose(tiny._sin[:, :half], tiny._sin[:, half:])


def test_rope_at_the_context_edge_is_finite_and_on_the_unit_circle(
    tiny: LlamaNumpy,
) -> None:
    edge = tiny.n_ctx - 1
    c, s = tiny._cos[edge], tiny._sin[edge]
    assert np.all(np.isfinite(c)) and np.all(np.isfinite(s))
    assert np.allclose(c**2 + s**2, 1.0, atol=1e-5)


def test_the_model_runs_to_the_last_position_in_context(tiny: LlamaNumpy) -> None:
    ids = tiny.encode(tiny_llama.words(tiny.n_ctx))[: tiny.n_ctx]
    tr = tiny.forward(ids)
    assert np.all(np.isfinite(tr.logits))
    assert tr.logits.shape[0] == tiny.n_ctx


def test_rope_makes_position_matter(tiny: LlamaNumpy) -> None:
    # Without RoPE a permutation-invariant bag would give the same answer for
    # the same multiset of tokens; this pins that position is actually read.
    a = tiny.forward([5, 9, 13, 21])
    b = tiny.forward([21, 13, 9, 5])
    assert not np.allclose(a.logits[-1], b.logits[-1], atol=1e-3)


# ── KV cache ────────────────────────────────────────────────────────────────


def test_kv_cache_matches_the_uncached_path(tiny: LlamaNumpy) -> None:
    """The central equivalence, stated as a measured tolerance not a hope."""
    ids = tiny.encode(tiny_llama.words(12))
    ref = tiny.forward(ids)

    cache = tiny.new_cache(1)
    got = []
    for i, tid in enumerate(ids):
        step = tiny.forward([tid], cache=cache)
        got.append(step.logits[-1])
        assert cache.t == i + 1
    inc = np.stack(got)

    err = float(np.max(np.abs(inc - ref.logits)))
    assert err < KV_TOL, f"cached and uncached logits differ by {err:.2e}"


def test_kv_cache_prefill_then_step(tiny: LlamaNumpy) -> None:
    ids = tiny.encode(tiny_llama.words(11))
    ref = tiny.forward(ids)
    cache = tiny.new_cache(1)
    tiny.forward(ids[:-1], cache=cache)
    step = tiny.forward(ids[-1:], cache=cache)
    err = float(np.max(np.abs(step.logits[-1] - ref.logits[-1])))
    assert err < KV_TOL, f"prefill+step differs from one pass by {err:.2e}"
    assert cache.t == len(ids)


def test_cached_trace_covers_only_the_new_positions(tiny: LlamaNumpy) -> None:
    ids = tiny.encode(tiny_llama.words(8))
    cache = tiny.new_cache(1)
    tiny.forward(ids[:5], cache=cache)
    tr = tiny.forward(ids[5:], cache=cache)
    assert tr.resid.shape[1] == 3, "a cached trace should not re-report history"
    # …but the attention it reports spans the full history, because that is
    # what the model actually attended to.
    assert tr.attn.shape[-1] == 8


def test_cache_slice_batch_keeps_the_kept_rows(tiny: LlamaNumpy) -> None:
    cache = tiny.new_cache(3)
    ids = np.asarray([tiny.encode(tiny_llama.words(4, seed=s)) for s in (1, 2, 3)])
    tiny._core(ids, cache=cache, resid_hooks=None, want_trace=False, last_only=True)
    kept = cache.slice_batch([0, 2])
    assert kept.batch == 2 and kept.t == cache.t


def test_kv_cache_is_a_dataclass_with_an_honest_length(tiny: LlamaNumpy) -> None:
    cache = tiny.new_cache(1)
    assert isinstance(cache, KVCache)
    assert cache.t == 0
    tiny.forward(tiny.encode(tiny_llama.words(5)), cache=cache)
    assert cache.t == 5


# ── residual hooks (the protocol shared with intervene.py) ──────────────────


def test_hook_sees_T_by_d_and_can_be_a_no_op(tiny: LlamaNumpy) -> None:
    ids = tiny.encode(tiny_llama.words(6))
    seen: list[tuple[int, ...]] = []

    def hook(x: np.ndarray) -> np.ndarray:
        seen.append(x.shape)
        assert x.dtype == np.float32
        return x

    ref = tiny.forward(ids)
    got = tiny.forward(ids, resid_hooks={1: hook})
    assert seen == [(6, tiny.d)]
    assert np.allclose(ref.logits, got.logits)


def test_embed_layer_hook_runs_before_block_zero(tiny: LlamaNumpy) -> None:
    ids = tiny.encode(tiny_llama.words(4))
    ref = tiny.forward(ids)
    zeroed = tiny.forward(ids, resid_hooks={EMBED_LAYER: lambda x: np.zeros_like(x)})
    assert not np.allclose(ref.logits, zeroed.logits)
    # zeroing the embedding must also zero the layer-0 *input* the trace records
    assert np.allclose(zeroed.resid[0], 0.0)


def test_hook_at_the_last_layer_changes_the_readout(tiny: LlamaNumpy) -> None:
    ids = tiny.encode(tiny_llama.words(4))
    ref = tiny.forward(ids)
    last = tiny.n_layer - 1
    bumped = tiny.forward(ids, resid_hooks={last: lambda x: x * 2.0})
    assert not np.allclose(ref.logits, bumped.logits)
    assert np.allclose(bumped.resid[tiny.n_layer], ref.resid[tiny.n_layer] * 2.0)


def test_hook_returning_the_wrong_shape_raises(tiny: LlamaNumpy) -> None:
    with pytest.raises(ValueError, match="expected"):
        tiny.forward(tiny_llama.words(4), resid_hooks={0: lambda x: x[:, :3]})


def test_hooks_apply_under_the_cache_too(tiny: LlamaNumpy) -> None:
    ids = tiny.encode(tiny_llama.words(6))
    hooks = {1: lambda x: x + 0.05}
    ref = tiny.forward(ids, resid_hooks=hooks)
    cache = tiny.new_cache(1)
    rows = [tiny.forward([t], resid_hooks=hooks, cache=cache).logits[-1] for t in ids]
    err = float(np.max(np.abs(np.stack(rows) - ref.logits)))
    assert err < KV_TOL, f"hooked cached path differs by {err:.2e}"


# ── batched residual capture ────────────────────────────────────────────────


def test_capture_resid_matches_per_prompt_forwards(tiny: LlamaNumpy) -> None:
    # Right-padding is exact under causal attention; this is the test that says
    # so on observed numbers rather than in a docstring.
    prompts = [tiny_llama.words(n, seed=n) for n in (3, 5, 9, 4)]
    layers = [EMBED_LAYER, 0, tiny.n_layer - 1]
    got = tiny.capture_resid(prompts, layers, batch_size=4)
    for L in layers:
        assert got[L].shape == (4, tiny.d)
    for i, p in enumerate(prompts):
        tr = tiny.forward(p)
        for L in layers:
            row = tr.resid[L + 1 if L != EMBED_LAYER else 0, -1]
            assert np.allclose(got[L][i], row, atol=1e-5), f"prompt {i} layer {L}"


def test_capture_resid_rejects_a_layer_out_of_range(tiny: LlamaNumpy) -> None:
    with pytest.raises(ValueError, match="out of range"):
        tiny.capture_resid(["t1 t2"], [tiny.n_layer])


# ── generation ──────────────────────────────────────────────────────────────


def test_greedy_generation_is_deterministic(tiny: LlamaNumpy) -> None:
    g1 = tiny.generate(tiny_llama.words(4), 8, temperature=0.0)
    g2 = tiny.generate(tiny_llama.words(4), 8, temperature=0.0)
    assert isinstance(g1, Generation)
    assert g1.tokens == g2.tokens
    assert len(g1.tokens) <= 8


def test_sampling_is_reproducible_under_a_seed(tiny: LlamaNumpy) -> None:
    a = tiny.generate(tiny_llama.words(4), 10, temperature=1.0, top_p=0.9, seed=42)
    b = tiny.generate(tiny_llama.words(4), 10, temperature=1.0, top_p=0.9, seed=42)
    c = tiny.generate(tiny_llama.words(4), 10, temperature=1.0, top_p=0.9, seed=43)
    assert a.tokens == b.tokens
    assert a.tokens != c.tokens or a.finish_reason == "stop"
    assert a.seed == 42 and a.temperature == 1.0 and a.top_p == 0.9


def test_greedy_generation_equals_a_manual_argmax_loop(tiny: LlamaNumpy) -> None:
    ids = tiny.encode(tiny_llama.words(4))
    manual = list(ids)
    want: list[int] = []
    for _ in range(6):
        nxt = int(tiny.forward(manual).logits[-1].argmax())
        if nxt == tiny.eos_token_id:
            break
        want.append(nxt)
        manual.append(nxt)
    got = tiny.generate(ids, 6, temperature=0.0)
    assert got.tokens == want


def test_generate_batch_matches_single_generation(tiny: LlamaNumpy) -> None:
    prompts = [tiny.encode(tiny_llama.words(5, seed=s)) for s in (1, 2, 3)]
    batched = tiny.generate_batch(prompts, 6, temperature=0.0)
    for p, g in zip(prompts, batched):
        assert g.tokens == tiny.generate(p, 6, temperature=0.0).tokens


def test_generate_batch_handles_ragged_prompts(tiny: LlamaNumpy) -> None:
    prompts = [tiny.encode(tiny_llama.words(n, seed=n)) for n in (3, 7)]
    outs = tiny.generate_batch(prompts, 5, temperature=0.0)
    assert [o.prompt_tokens for o in outs] == prompts


def test_generation_stops_on_a_stop_token(tiny: LlamaNumpy) -> None:
    ids = tiny.encode(tiny_llama.words(4))
    forced = int(tiny.forward(ids).logits[-1].argmax())
    g = tiny.generate(ids, 8, temperature=0.0, stop_tokens=[forced])
    assert g.tokens == [] and g.finish_reason == "stop"


def test_empty_prompt_is_refused(tiny: LlamaNumpy) -> None:
    with pytest.raises(ValueError, match="empty"):
        tiny.forward([])


# ── chat template ───────────────────────────────────────────────────────────


def test_chat_template_renders_chatml(tiny: LlamaNumpy) -> None:
    out = tiny.apply_chat_template(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    )
    assert out == (
        "<|im_start|>system\nS<|im_end|>\n"
        "<|im_start|>user\nU<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_chat_template_refuses_a_non_chatml_checkpoint(tmp_path: Path) -> None:
    path = tiny_llama.build(tmp_path / "m")
    (path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{x}}"}))
    m = LlamaNumpy("tiny/wrong-template", local_dir=path)
    with pytest.raises(NotImplementedError, match="ChatML"):
        m.apply_chat_template([{"role": "user", "content": "hi"}])


# ── throughput helper ───────────────────────────────────────────────────────


def test_tokens_per_second_is_positive(tiny: LlamaNumpy) -> None:
    assert tokens_per_second(tiny, prompt=tiny_llama.words(4), n=4) > 0


# ── the real checkpoint ─────────────────────────────────────────────────────


@pytest.mark.network
def test_real_smollm2_matches_reference() -> None:
    """SmolLM2-135M-Instruct, pinned, greedy — the check random weights cannot make."""
    rev = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
    # $NEBULAI_SMOLLM2_135M_DIR lets this run against an already-fetched copy
    # of *that commit*. It is a shortcut around the download, not around the
    # pin: the revision is still asserted, and pointing it at a different
    # checkpoint makes the assertions below fail rather than pass quietly.
    local = os.environ.get("NEBULAI_SMOLLM2_135M_DIR")
    m = LlamaNumpy(
        "HuggingFaceTB/SmolLM2-135M-Instruct", revision=rev, local_dir=local
    )
    assert m.revision == rev
    assert (m.n_layer, m.n_head, m.n_kv_head, m.d) == (30, 9, 3, 576)

    ids = m.encode("The capital of France is")
    tr = m.forward(ids)
    top = m.decode1(int(tr.logits[-1].argmax()))
    assert top.strip() == "Paris", f"greedy next token was {top!r}"

    # the model's own readout head reproduces its own logits exactly
    lens = m.logit_lens(tr.resid[m.n_layer, -1])
    assert np.array_equal(lens, tr.logits[-1])

    # and the cache changes the logits only in the fourth significant figure,
    # and the argmax not at all
    cache = m.new_cache(1)
    inc = np.stack([m.forward([t], cache=cache).logits[-1] for t in ids])
    err = float(np.max(np.abs(inc - tr.logits)))
    assert err < KV_TOL_REAL, f"cached logits differ by {err:.2e}"
    assert np.array_equal(inc.argmax(1), tr.logits.argmax(1))

    def _logsoftmax(x):
        x = x.astype(np.float64)
        x = x - x.max(-1, keepdims=True)
        return x - np.log(np.exp(x).sum(-1, keepdims=True))

    a, b = _logsoftmax(tr.logits), _logsoftmax(inc)
    kl = float((np.exp(a) * (a - b)).sum(-1).max() / math.log(2))
    assert kl < KV_KL_BITS_REAL, f"cached distribution differs by {kl:.2e} bits"
