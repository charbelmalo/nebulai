"""Interventions: each verb changes what it claims to and nothing else.

Four properties carry this file, and each is a specific way a steering result
can be a lie:

* **α = 0 is bit-identical to the baseline.** Not "close", not "within
  tolerance" — the same float32 array. It is the sharpest correctness test
  available here, because it is the only one that cannot be passed by a hook
  that is subtly wrong; and it is the test that matters, because α = 0 is the
  control every sweep is read against. The identity is achieved by installing
  NO hook, and `test_identity_hook_is_also_bit_identical` pins the weaker
  property too (a hook that returns its input unchanged is also a no-op), so
  the guarantee does not silently depend on the drop.
* **Each verb does its own thing.** `ablate` is not `add` with a negative α:
  one moves every point equally, the other removes each point's own component.
  `cap` touches only coordinates outside its bounds. `clamp` moves exactly one
  SAE feature's contribution.
* **The layer is the hook protocol's layer.** `sae.L8.*` is `hook_resid_pre`
  of block 8, which is the output of block 7. The off-by-one here produces a
  curve at the wrong depth that looks perfectly plausible.
* **D6: measure, never export.** No code path in the intervention stack writes
  a checkpoint. `test_no_weight_export` greps for the calls, so a future
  contributor cannot add one without deleting an explicit test.

The GPT-2-backed tests need real weights; they are skipped when the HF cache
has none, exactly like the rest of the interp suite.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from nebulai.backend.interp import intervene as IV
from nebulai.backend.interp.hooks import (
    EMBED_LAYER,
    ForwardPass,
    HookError,
    apply_resid_hook,
    check_layer,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "nebulai"


# --------------------------------------------------------------- a tiny model


class ToyModel:
    """A two-block linear "transformer" with a real hook protocol.

    Deliberately not GPT-2: the verbs must be correct against the PROTOCOL, and
    a test that only ever runs one implementation cannot tell the difference
    between "the verb is right" and "the verb and gpt2_numpy agree on the same
    mistake". This is also the stand-in for `llama_numpy.py`, which is being
    written in parallel and which this module must run against unchanged.
    """

    n_layer = 3
    d = 8

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(size=(self.d, self.d)).astype(np.float32) * 0.3 for _ in range(self.n_layer)]
        self.emb = rng.normal(size=(16, self.d)).astype(np.float32)

    def encode(self, text):
        return [(ord(c) % 16) for c in text] or [0]

    def decode1(self, tid):
        return chr(97 + int(tid) % 16)

    def forward(self, prompt, *, resid_hooks=None, cache=None):
        if cache is not None:
            raise NotImplementedError
        hooks = dict(resid_hooks or {})
        for L in hooks:
            check_layer(L, self.n_layer)
        ids = self.encode(prompt) if isinstance(prompt, str) else list(prompt)
        x = self.emb[ids]
        if EMBED_LAYER in hooks:
            x = apply_resid_hook(hooks[EMBED_LAYER], x, layer=EMBED_LAYER)
        states = [x]
        for L in range(self.n_layer):
            x = x + np.tanh(x @ self.W[L])
            if L in hooks:
                x = apply_resid_hook(hooks[L], x, layer=L)
            states.append(x)
        return _Trace(np.stack(states), x @ self.emb.T)


class _Trace:
    def __init__(self, resid, logits):
        self.resid, self.logits = resid, logits


def _unit(i: int, d: int = 8) -> np.ndarray:
    v = np.zeros(d)
    v[i] = 1.0
    return v


class _D:
    """The shape `from_direction` needs — a `directions.Direction` duck."""

    def __init__(self, vector, space="resid.L1", did="d-test"):
        self.vector, self.space, self.id = vector, space, did


# ------------------------------------------------------------ the protocol is


def test_the_toy_model_satisfies_the_forward_pass_protocol():
    assert isinstance(ToyModel(), ForwardPass)


def test_hook_layer_is_identity_for_resid_and_off_by_one_for_sae():
    # resid.L8 is the OUTPUT of block 8 — hook layer 8.
    assert IV.hook_layer("resid.L8", n_layer=12) == 8
    # sae.L8.<repo> is hook_resid_pre of block 8 = the output of block 7.
    assert IV.hook_layer("sae.L8.jbloom/x", n_layer=12) == 7
    # and at block 0 that is the embedding output, not block -1.
    assert IV.hook_layer("sae.L0.jbloom/x", n_layer=12) == EMBED_LAYER


def test_a_space_with_no_layer_cannot_be_intervened_at():
    with pytest.raises(IV.InterventionError, match="no layer"):
        IV.hook_layer("W_E.centered", n_layer=12)


# ------------------------------------------------------------------- identity


def test_alpha_zero_installs_no_hook_at_all():
    iv = IV.from_direction(_D(_unit(0)), verb="add", alpha=0.0, n_layer=3)
    assert iv.is_identity
    assert iv.resid_hooks(ToyModel()) == {}


def test_alpha_zero_is_bit_identical_to_the_baseline():
    m = ToyModel()
    base = m.forward("hello")
    iv = IV.from_direction(_D(_unit(0)), verb="add", alpha=0.0, n_layer=3)
    got = m.forward("hello", resid_hooks=iv.resid_hooks(m) or None)
    assert np.array_equal(base.logits, got.logits)
    assert np.array_equal(base.resid, got.resid)


def test_identity_hook_is_also_bit_identical():
    """The weaker guarantee, pinned separately.

    `is_identity` drops the hook, so the α = 0 test above would pass even if a
    round-trip through `apply_resid_hook` perturbed the stream. It does not —
    the hook is handed the array the runner is carrying and the result is cast
    back to that same dtype — and this test is what keeps it true, because
    the measured perturbation when it was NOT true was 1.5e-5 in the logits:
    invisible in a plot, and the same size as the effect at the bottom of a
    sweep.
    """
    m = ToyModel()
    base = m.forward("hello")
    got = m.forward("hello", resid_hooks={L: (lambda x: x) for L in range(-1, m.n_layer)})
    assert np.array_equal(base.logits, got.logits)
    assert np.array_equal(base.resid, got.resid)


def test_a_cap_with_no_bounds_is_the_identity_but_a_finite_one_never_is():
    assert IV.Intervention(verb="cap", layer=1, lo=None, hi=None).is_identity
    # even a cap no activation reaches: identity is a property of the
    # intervention, not of the prompt it happened to be run on.
    assert not IV.Intervention(verb="cap", layer=1, lo=-1e9, hi=1e9).is_identity


# ---------------------------------------------------------------- add the verb


def test_add_shifts_every_position_by_exactly_alpha_times_the_unit_vector():
    m = ToyModel()
    v = _unit(3)
    base = m.forward("hello")
    iv = IV.from_direction(_D(v), verb="add", alpha=2.0, layer=1, n_layer=3)
    got = m.forward("hello", resid_hooks=iv.resid_hooks(m))
    # resid[2] is the state after block 1 — the hook's own output
    delta = got.resid[2] - base.resid[2]
    assert np.allclose(delta, 2.0 * v[None, :], atol=1e-5)


def test_add_normalises_the_direction_so_alpha_is_in_residual_norm_units():
    m = ToyModel()
    long = _unit(3) * 17.0
    iv = IV.from_direction(_D(long), verb="add", alpha=1.0, layer=1, n_layer=3)
    base, got = m.forward("hi"), None
    got = m.forward("hi", resid_hooks=iv.resid_hooks(m))
    d = got.resid[2] - base.resid[2]
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-5)


def test_a_direction_of_the_wrong_width_is_refused_not_padded():
    m = ToyModel()
    iv = IV.from_direction(_D(np.ones(5)), verb="add", alpha=1.0, layer=0, n_layer=3)
    with pytest.raises(IV.InterventionError, match="5 wide"):
        iv.resid_hooks(m)


# ------------------------------------------------------------- ablate the verb


def test_ablate_removes_each_points_own_component_not_a_fixed_shift():
    m = ToyModel()
    v = _unit(2)
    iv = IV.Intervention(verb="ablate", layer=1, vector=v, alpha=1.0, direction_id="d")
    got = m.forward("hello", resid_hooks=iv.resid_hooks(m))
    assert np.allclose(got.resid[2] @ v, 0.0, atol=1e-5)


def test_ablate_is_not_add_with_a_negative_alpha():
    """The two verbs must produce genuinely different states.

    If they did not, `ablate` would be a second name for `add` and every curve
    labelled "ablation" would be a translation — which is the single most
    likely way this module could be quietly wrong.
    """
    m = ToyModel()
    v = _unit(2)
    base = m.forward("hello")
    comp = float(np.mean(base.resid[2] @ v))
    abl = m.forward(
        "hello",
        resid_hooks=IV.Intervention(
            verb="ablate", layer=1, vector=v, alpha=1.0, direction_id="d"
        ).resid_hooks(m),
    )
    add = m.forward(
        "hello",
        resid_hooks=IV.Intervention(
            verb="add", layer=1, vector=v, alpha=-comp, direction_id="d"
        ).resid_hooks(m),
    )
    assert not np.allclose(abl.resid[2], add.resid[2], atol=1e-4)


def test_ablate_with_no_layer_fires_at_every_layer_including_the_embedding():
    m = ToyModel()
    iv = IV.Intervention(verb="ablate", layer=None, vector=_unit(1), alpha=1.0, direction_id="d")
    assert sorted(iv.resid_hooks(m)) == [-1, 0, 1, 2]


def test_ablate_leaves_a_point_with_no_component_alone():
    m = ToyModel()
    v = _unit(4)
    iv = IV.Intervention(verb="ablate", layer=1, vector=v, alpha=1.0, direction_id="d")
    hook = iv.resid_hooks(m)[1]
    x = np.stack([_unit(0), _unit(1)]).astype(np.float32)  # no component on v
    assert np.allclose(hook(x), x)


# ---------------------------------------------------------------- cap the verb


def test_cap_touches_only_the_coordinates_outside_its_bounds():
    hook = IV._cap_hook(-1.0, 1.0)
    x = np.array([[-3.0, -0.5, 0.25, 5.0]], dtype=np.float32)
    out = hook(x)
    assert out.tolist() == [[-1.0, -0.5, 0.25, 1.0]]


def test_an_inverted_cap_is_refused():
    with pytest.raises(IV.InterventionError, match="above hi"):
        IV._cap_hook(5.0, 1.0)


def test_cap_actually_changes_a_real_forward_pass():
    m = ToyModel()
    base = m.forward("hello world")
    iv = IV.Intervention(verb="cap", layer=1, lo=-0.05, hi=0.05)
    got = m.forward("hello world", resid_hooks=iv.resid_hooks(m))
    assert not np.allclose(base.logits, got.logits)
    assert float(np.abs(got.resid[2]).max()) <= 0.05 + 1e-6


# -------------------------------------------------------------- clamp the verb


def _toy_sae(d=8, d_sae=5, seed=1):
    rng = np.random.default_rng(seed)
    W_dec = rng.normal(size=(d_sae, d)).astype(np.float32)
    return {
        "W_enc": rng.normal(size=(d, d_sae)).astype(np.float32),
        "b_enc": np.zeros(d_sae, dtype=np.float32),
        "W_dec": W_dec,
        "b_dec": np.zeros(d, dtype=np.float32),
    }


def test_clamp_moves_the_stream_along_exactly_one_decoder_row():
    m, t = ToyModel(), _toy_sae()
    iv = IV.Intervention(verb="clamp", layer=1, feature=2, value=9.0, alpha=1.0)
    base = m.forward("hello")
    got = m.forward("hello", resid_hooks=IV.clamp_hooks(m, iv, t))
    delta = got.resid[2] - base.resid[2]
    row = t["W_dec"][2] / np.linalg.norm(t["W_dec"][2])
    # every row of the delta is parallel to W_dec[2]
    resid = delta - (delta @ row)[:, None] * row[None, :]
    assert float(np.abs(resid).max()) < 1e-4
    assert float(np.abs(delta).max()) > 1e-3


def test_clamp_at_alpha_zero_is_exactly_zero_change_not_a_reconstruction():
    """The reason `clamp_hooks` writes the DIFFERENCE and not `decode(encode)`.

    An SAE reconstructs its input imperfectly (cosine ≈ 0.9 on the shipped
    release). Decoding wholesale would apply that error as an intervention, and
    the bottom of the sweep would silently not be the baseline.
    """
    m, t = ToyModel(), _toy_sae()
    iv = IV.Intervention(verb="clamp", layer=1, feature=2, value=9.0, alpha=0.0)
    assert iv.is_identity
    assert IV.clamp_hooks(m, iv, t) == {}


def test_clamp_refuses_a_feature_outside_the_dictionary():
    m, t = ToyModel(), _toy_sae()
    iv = IV.Intervention(verb="clamp", layer=1, feature=999, value=1.0, alpha=1.0)
    with pytest.raises(IV.InterventionError, match="outside this SAE"):
        IV.clamp_hooks(m, iv, t)


def test_clamp_refuses_an_sae_of_the_wrong_width():
    m = ToyModel()
    with pytest.raises(IV.InterventionError, match="trained on a"):
        IV.clamp_hooks(
            m,
            IV.Intervention(verb="clamp", layer=1, feature=0, value=1.0, alpha=1.0),
            _toy_sae(d=16),
        )


def test_clamp_will_not_build_its_own_hook_without_the_sae_tensors():
    """Describing an intervention must never download 150 MB as a side effect."""
    m = ToyModel()
    iv = IV.Intervention(verb="clamp", layer=1, feature=0, value=1.0, alpha=1.0)
    with pytest.raises(IV.InterventionError, match="needs the SAE tensors"):
        iv.resid_hooks(m)


# ------------------------------------------------------------- the allow-list


def test_build_refuses_a_verb_it_does_not_implement():
    with pytest.raises(IV.InterventionError, match="unknown intervention verb"):
        IV.build({"verb": "steer", "alpha": 1.0}, n_layer=12)


@pytest.mark.parametrize("verb", IV.VERBS)
def test_every_advertised_verb_is_buildable(verb):
    spec = {
        "cap": {"verb": "cap", "layer": 3, "lo": -1.0, "hi": 1.0},
        "clamp": {"verb": "clamp", "layer": 3, "feature": 7, "value": 5.0, "alpha": 1.0},
        "add": {"verb": "add", "direction_id": "d", "alpha": 1.0, "layer": 3},
        "ablate": {"verb": "ablate", "direction_id": "d", "alpha": 1.0},
    }[verb]
    iv = IV.build(spec, resolve_direction=lambda _: _D(_unit(0, 12), "resid.L3", "d"), n_layer=12)
    assert iv.verb == verb
    assert iv.protocol


def test_an_unexpected_key_is_refused_rather_than_ignored():
    with pytest.raises(IV.InterventionError, match="unexpected"):
        IV.build({"verb": "cap", "layer": 3, "lo": -1.0, "hi": 1.0, "strength": 4}, n_layer=12)


def test_a_direction_id_with_no_store_is_refused_not_invented():
    with pytest.raises(IV.InterventionError, match="no direction store"):
        IV.build({"verb": "add", "direction_id": "d", "alpha": 1.0, "layer": 3}, n_layer=12)


def test_a_layer_outside_the_model_is_refused():
    with pytest.raises(HookError, match="outside this model"):
        IV.build({"verb": "cap", "layer": 99, "lo": -1.0}, n_layer=12)


def test_a_non_finite_alpha_is_refused():
    with pytest.raises(IV.InterventionError, match="finite"):
        IV.build(
            {"verb": "add", "direction_id": "d", "alpha": float("inf"), "layer": 3},
            resolve_direction=lambda _: _D(_unit(0, 12), "resid.L3", "d"),
            n_layer=12,
        )


def test_from_direction_defaults_to_the_directions_own_layer():
    iv = IV.from_direction(_D(_unit(0, 12), space="resid.L5"), verb="add", alpha=1.0, n_layer=12)
    assert iv.layer == 5


# ------------------------------------------------------------------- protocol


def test_the_protocol_string_says_what_was_done_never_what_the_direction_is():
    iv = IV.from_direction(
        _D(_unit(0, 12), space="resid.L5", did="refusal-style-v1-L8"),
        verb="add",
        alpha=2.5,
        n_layer=12,
    )
    p = iv.protocol
    assert "add +2.5" in p and "layer 5" in p and "refusal-style-v1-L8" in p
    # the amendment in §2.4 permits a sentence about the intervention, never
    # a sentence about what the direction *is*.
    assert " is " not in p


def test_the_embedding_layer_is_named_not_printed_as_minus_one():
    iv = IV.Intervention(verb="cap", layer=EMBED_LAYER, lo=-1.0, hi=1.0)
    assert "the embedding output" in iv.protocol


def test_the_echo_never_carries_the_vector():
    iv = IV.from_direction(_D(_unit(0, 12), space="resid.L5"), verb="add", alpha=1.0, n_layer=12)
    j = iv.to_json()
    assert "vector" not in j
    assert j["direction_id"] == "d-test" and j["space"] == "resid.L5"


def test_a_sweep_that_found_nothing_says_so_rather_than_leaving_the_claim_off():
    rows = [{"is_identity": True, "kl_bits_max": 0.0, "protocol": "p"}]
    s = IV.claim_sentence(rows)
    assert "measured as none" in s


def test_the_claim_sentence_names_the_protocol_and_refuses_the_essence_claim():
    rows = [
        {"is_identity": True, "kl_bits_max": 0.0, "protocol": "control"},
        {"is_identity": False, "kl_bits_max": 2.5, "protocol": "add +1·d at layer 8"},
    ]
    s = IV.claim_sentence(rows)
    assert "add +1·d at layer 8" in s
    assert "2.500 bits" in s
    assert "not about what the direction is" in s


# ------------------------------------------------------------------ the sweep


def test_a_sweep_reports_its_control_row_as_identical():
    m = ToyModel()
    b = IV.sweep(
        m,
        ["hello"],
        lambda a: IV.from_direction(_D(_unit(0)), verb="add", alpha=a, layer=1, n_layer=3),
        [0.0, 1.0],
        max_tokens=2,
    )
    ctrl = [r for r in b["rows"] if r["is_identity"]]
    assert len(ctrl) == 1
    assert ctrl[0]["identical_to_baseline"] is True
    assert ctrl[0]["kl_bits_max"] == 0.0
    moved = [r for r in b["rows"] if not r["is_identity"]]
    assert moved[0]["kl_bits_max"] > 0.0


def test_generation_is_greedy_and_refuses_to_sample():
    m = ToyModel()
    with pytest.raises(IV.InterventionError, match="greedy"):
        IV.generate(m, "hello", max_tokens=2, temperature=0.8)


def test_two_baseline_generations_are_identical_because_decoding_is_greedy():
    m = ToyModel()
    a = IV.generate(m, "hello", max_tokens=4)
    b = IV.generate(m, "hello", max_tokens=4)
    assert a["ids"] == b["ids"]


# ----------------------------------------------------------------------- D6


#: the calls that would turn a measurement into a distributable model
_EXPORT_CALLS = (
    r"safetensors[.\w]*\.save",
    r"save_file\s*\(",
    r"torch\.save\s*\(",
    r"\.save_pretrained\s*\(",
    r"push_to_hub\s*\(",
    r"create_repo\s*\(",
    r"upload_file\s*\(",
    r"upload_folder\s*\(",
)

#: every module that can reach a hook, plus the one module that holds adapted
#: weights. If a new runner is added it belongs here — the point of the list is
#: that it is short enough to keep honest.
#:
#: `organisms/emergent_misalignment.py` is the odd one out and the most
#: important: every other file here measures a model it left alone, so for them
#: D6 forbids an export that was never tempting. That one trains a rank-1 LoRA,
#: so an adapted checkpoint genuinely exists in its process and writing it out
#: would be one line. The rule is worth exactly as much as it is enforced on the
#: file that could break it.
_INTERVENTION_PATHS = (
    "backend/interp/intervene.py",
    "backend/interp/hooks.py",
    "backend/interp/gpt2_numpy.py",
    "backend/interp/live_server.py",
    "backend/directions.py",
    "organisms/emergent_misalignment.py",
)


def test_no_weight_export():
    """D6, mechanically: no intervention path writes a checkpoint.

    This is enforced by ABSENCE, so a test is the only thing that can enforce
    it — there is no positive artifact whose correctness would break. A
    contributor who adds an "export the steered model" flag has to delete this
    test to do it, and deleting a test with this docstring is a visible act.

    Not just a grep: `test_no_weight_export_functionally` below asserts that a
    real intervened run leaves the model's own weights untouched, so an export
    path that spelled itself differently would still be caught by behaviour.
    """
    offenders = []
    for rel in _INTERVENTION_PATHS:
        p = SRC / rel
        assert p.exists(), f"{rel} is in the D6 path list but does not exist"
        text = p.read_text(encoding="utf-8")
        # ignore the prose that explains the rule (docstrings and comments name
        # these calls on purpose)
        code = "\n".join(
            ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
        )
        for pat in _EXPORT_CALLS:
            for mt in re.finditer(pat, code):
                line = code[: mt.start()].count("\n") + 1
                offenders.append(f"{rel}:{line} {mt.group(0)}")
    assert not offenders, (
        "D6 says measure, never export. These intervention-path files now "
        "contain weight-writing calls:\n  " + "\n  ".join(offenders)
    )


def test_the_cli_has_no_flag_that_could_write_a_model():
    """`nebulai intervene` writes a measurement bundle and nothing else.

    Checked against the parser's registered flags rather than the file's text,
    because the docstring names the flags that must not exist and a plain
    substring search would match its own warning.
    """
    import argparse as _ap

    from nebulai import cli as _cli

    body = (SRC / "cli.py").read_text(encoding="utf-8")
    start = body.index("def _run_intervene")
    end = body.index("def _resolve_revision_or_branch")
    code = "\n".join(
        ln
        for ln in body[start:end].splitlines()
        if not ln.lstrip().startswith("#")
    )
    for bad in ("save_pretrained", "torch.save", "save_file(", "push_to_hub"):
        assert bad not in code, f"`nebulai intervene` grew {bad}"

    flags = set()

    class _Sink(_ap.ArgumentParser):
        def add_argument(self, *a, **kw):  # noqa: D102
            flags.update(x for x in a if isinstance(x, str) and x.startswith("--"))
            return super().add_argument(*a, **kw)

    parser = _cli._build_parser(_Sink) if hasattr(_cli, "_build_parser") else None
    if parser is None:  # the parser is built inline in main(); scan the block
        blk = body[body.index('sub.add_parser(\n        "intervene"') :]
        blk = blk[: blk.index("ivp.set_defaults")]
        flags = set(re.findall(r'add_argument\(\s*"(--[\w-]+)"', blk))
    for bad in ("--export", "--export-weights", "--save-model", "--save", "--push"):
        assert bad not in flags, f"`nebulai intervene` grew {bad}"
    assert "--alpha" in flags and "--prompt" in flags


def test_no_weight_export_functionally():
    """The behavioural half: an intervened run leaves the weights as they were.

    A hook that mutated its input in place would also rewrite `Trace.resid` —
    and, in a runner that kept a view on a weight, could rewrite the weight.
    This asserts the model is byte-identical after a full sweep.
    """
    m = ToyModel()
    before = [w.copy() for w in m.W]
    emb_before = m.emb.copy()
    IV.sweep(
        m,
        ["hello"],
        lambda a: IV.Intervention(verb="ablate", layer=None, vector=_unit(1), alpha=a, direction_id="d"),
        [0.0, 1.0],
        max_tokens=2,
    )
    for a, b in zip(m.W, before):
        assert np.array_equal(a, b)
    assert np.array_equal(m.emb, emb_before)


def test_a_hook_may_not_mutate_the_stream_in_place():
    """`apply_resid_hook` cannot detect this, so the runner's contract does.

    A hook that writes into its argument also rewrites `Trace.resid`, and the
    exported trajectory then shows the intervention as though it had always
    been there. The verbs in this module all return new arrays; this pins that.
    """
    for hook in (
        IV._add_hook(_unit(0), 1.0),
        IV._ablate_hook(_unit(0), 1.0),
        IV._cap_hook(-0.1, 0.1),
    ):
        x = np.arange(16, dtype=np.float32).reshape(2, 8)
        keep = x.copy()
        hook(x)
        assert np.array_equal(x, keep)


# --------------------------------------------------------- the hook protocol


def test_a_hook_that_changes_shape_is_refused():
    with pytest.raises(HookError, match="returned shape"):
        apply_resid_hook(lambda x: x[:, :2], np.ones((3, 8), dtype=np.float32), layer=1)


def test_a_hook_that_returns_a_nan_is_refused():
    with pytest.raises(HookError, match="non-finite"):
        apply_resid_hook(lambda x: x * np.nan, np.ones((3, 8), dtype=np.float32), layer=1)


def test_a_hook_that_returns_something_that_is_not_an_array_is_refused():
    with pytest.raises(HookError, match="not an array"):
        apply_resid_hook(lambda x: "nope", np.ones((3, 8), dtype=np.float32), layer=1)


def test_apply_resid_hook_returns_the_dtype_the_runner_is_carrying():
    """Not a nominal float32: see hooks.py's docstring for the measurement."""
    x = np.ones((3, 8), dtype=np.float64)
    assert apply_resid_hook(lambda v: v * 1.0, x, layer=1).dtype == np.float64
    x32 = np.ones((3, 8), dtype=np.float32)
    assert apply_resid_hook(lambda v: v.astype(np.float64), x32, layer=1).dtype == np.float32


def test_layer_minus_two_is_not_python_indexing():
    with pytest.raises(HookError, match="outside this model"):
        check_layer(-2, 12)


def test_layer_minus_one_is_the_embedding_output():
    assert check_layer(-1, 12) == EMBED_LAYER


# ------------------------------------------------------------- real GPT-2 now

gpt2 = pytest.importorskip("tokenizers") and None


def _gpt2():
    from nebulai.backend.interp.gpt2_numpy import GPT2Numpy

    try:
        return GPT2Numpy("gpt2")
    except Exception as e:  # no cached weights in this environment
        pytest.skip(f"gpt2 weights unavailable: {e}")


def test_gpt2_alpha_zero_is_bit_identical():
    m = _gpt2()
    base = m.forward("The capital of France is")
    iv = IV.from_direction(
        _D(np.eye(768)[17], space="resid.L8"), verb="add", alpha=0.0, n_layer=12
    )
    got = m.forward("The capital of France is", resid_hooks=iv.resid_hooks(m) or None)
    assert np.array_equal(base.logits, got.logits)


def test_gpt2_an_identity_hook_at_every_layer_is_bit_identical():
    m = _gpt2()
    base = m.forward("The capital of France is")
    got = m.forward(
        "The capital of France is",
        resid_hooks={L: (lambda x: x) for L in range(-1, m.n_layer)},
    )
    assert np.array_equal(base.logits, got.logits)
    assert np.array_equal(base.resid, got.resid)


def test_gpt2_add_at_a_real_layer_changes_the_generation():
    m = _gpt2()
    iv = IV.from_direction(
        _D(np.eye(768)[17], space="resid.L8"), verb="add", alpha=40.0, n_layer=12
    )
    r = IV.run(m, "The capital of France is", iv, max_tokens=6)
    assert r["identical"] is False
    assert r["kl_bits"] > 0.01
    assert r["baseline"]["text"] != r["intervened"]["text"]


def test_gpt2_refuses_a_kv_cache_it_does_not_have():
    m = _gpt2()
    with pytest.raises(NotImplementedError, match="no KV cache"):
        m.forward("hello", cache=object())
