"""Four intervention verbs, as forward hooks. The only place this repo is
allowed to make a causal claim — and the only place it must never write a
weight.

An intervention here is a *description*: a small frozen object that knows its
own protocol string, knows whether it is the identity, and can be compiled into
the `resid_hooks` mapping that `hooks.ForwardPass` runners accept. Nothing in
this module touches a checkpoint, a `safetensors` writer or a `torch.save`; the
verbs exist only as inference-time hooks, which is decision **D6** ("measure,
never export") enforced by absence rather than by a warning. `tests/
test_intervene.py::test_no_weight_export` greps this package for the export
calls so a future contributor cannot add one silently.

The four verbs (plan §3.4), and what each one actually does to the stream:

| verb     | what it does                                                     |
|----------|------------------------------------------------------------------|
| `clamp`  | pin one SAE feature's activation to a value, at that SAE's hook   |
| `add`    | add α·d to the residual stream at one layer                      |
| `ablate` | project d out of the residual stream at EVERY layer               |
| `cap`    | clip the residual stream into [lo, hi] at one layer               |

Three properties the tests pin, because each one is a way a sweep can lie:

**α = 0 is the baseline, bit for bit.** Every verb answers `is_identity`, and
`resid_hooks()` returns an EMPTY mapping when it is true — not a hook that adds
zero. A no-op hook is not free: it round-trips the stream through
`apply_resid_hook`, and the difference, though it is 1e-5 in the logits, is
exactly the size of the effect a careful reader is looking for at the bottom of
an α sweep. So the control installs nothing at all.

**`ablate` is not `add` with a negative α.** Adding −α·d moves every point by
the same amount regardless of where it started; projecting d out moves each
point by its own component, and leaves a point that never had one alone. They
are different interventions and the two curves look different; naming them the
same thing would be the whole error.

**The layer index is the hook protocol's, never the artifact's.** `resid.L8`
(what `direction prompts --layer 8` writes) is the output of block 8, so it
compiles to hook layer 8. An SAE at `blocks.8.hook_resid_pre` reads the stream
*entering* block 8, which is the output of block **7**, so it compiles to hook
layer 7. Both conversions live in `hook_layer()` below with the arithmetic
written out, because an off-by-one here produces a perfectly plausible curve at
the wrong depth.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ...spaces import Space, SpaceFamily
from ...spaces import parse as parse_space
from .hooks import EMBED_LAYER, ForwardPass, ResidHook, check_layer

#: the verbs the server and CLI will accept. Anything else is refused flatly
#: rather than interpreted — see `build()` and `live_server._intervene`.
VERBS = ("clamp", "add", "ablate", "cap")


class InterventionError(ValueError):
    """A malformed intervention. Never a silently-corrected one."""


def hook_layer(space: str | Space, *, n_layer: int) -> int:
    """Convert a space tag's layer into the hook protocol's layer index.

    `resid.L<k>` and `mlp_out.L<k>` name the OUTPUT of block k, which is where
    hook k fires: identity.

    `sae.L<k>.<repo>` is the sae_lens convention, where the layer is the
    `blocks.<k>.hook_resid_pre` hook point — the stream ENTERING block k. That
    is the output of block k−1, i.e. hook layer k−1, and for k = 0 it is the
    embedding output, hook layer −1. This subtraction is the single most
    error-prone line in the phase; it is written once, here.
    """
    sp = parse_space(space) if isinstance(space, str) else space
    if sp.layer is None:
        raise InterventionError(
            f"space {sp!s} has no layer — an intervention needs a place in the "
            f"forward pass, and this space does not name one"
        )
    L = int(sp.layer)
    if sp.family is SpaceFamily.SAE:
        L = L - 1  # hook_resid_pre of block k == output of block k−1
    return check_layer(L, n_layer)


def _round(x: float, n: int = 6) -> float:
    return round(float(x), n)


@dataclass(frozen=True)
class Intervention:
    """One verb, fully specified, with its own protocol sentence.

    Frozen because a sweep holds a list of these and an export stamps them; an
    intervention that could be mutated after the run that produced its numbers
    is a stamp that no longer describes the run.
    """

    verb: str
    #: hook-protocol layer (−1 = embedding output), or None for `ablate`
    #: (which fires at every layer by definition)
    layer: int | None = None
    #: the unit direction, for `add` / `ablate`
    vector: np.ndarray | None = None
    direction_id: str | None = None
    space: str | None = None
    alpha: float = 0.0
    #: `clamp`
    feature: int | None = None
    value: float = 0.0
    sae_repo: str | None = None
    sae_hook: str | None = None
    #: `cap`
    lo: float | None = None
    hi: float | None = None
    #: everything needed to re-run this exact intervention
    meta: dict[str, Any] = field(default_factory=dict)

    # -------------------------------------------------------------- identity

    @property
    def is_identity(self) -> bool:
        """True when this intervention provably changes nothing.

        A `True` here means the runner is handed NO hook, so the result is
        bit-identical to the baseline. It is deliberately conservative: `cap`
        with a finite range is not called identity even when no activation
        reaches the bound on this particular prompt, because that would make
        identity a property of the prompt rather than of the intervention.
        """
        if self.verb in ("add", "ablate"):
            return float(self.alpha) == 0.0
        if self.verb == "clamp":
            return float(self.alpha) == 0.0
        if self.verb == "cap":
            lo = -np.inf if self.lo is None else float(self.lo)
            hi = np.inf if self.hi is None else float(self.hi)
            return not np.isfinite(lo) and not np.isfinite(hi)
        return False

    # -------------------------------------------------------------- protocol

    @property
    def protocol(self) -> str:
        """One line naming the verb, the place, and the amount.

        This string travels into the curve bundle and onto the figure. It never
        says what the direction *is* — §2.4's amendment permits a sentence about
        what an intervention *did*, and nothing more.
        """
        where = "every layer" if self.layer is None else _layer_phrase(self.layer)
        if self.verb == "add":
            return (
                f"add {self.alpha:+g}·d to the residual stream at {where} "
                f"(d = {self.direction_id}, space {self.space})"
            )
        if self.verb == "ablate":
            return (
                f"project d out of the residual stream at {where}, "
                f"strength {self.alpha:g} (d = {self.direction_id}, space {self.space})"
            )
        if self.verb == "clamp":
            return (
                f"pin SAE feature {self.feature} to {self.value:g} "
                f"(strength {self.alpha:g}) at {where} "
                f"({self.sae_repo} @ {self.sae_hook})"
            )
        if self.verb == "cap":
            lo = "none" if self.lo is None else f"{self.lo:g}"
            hi = "none" if self.hi is None else f"{self.hi:g}"
            return f"clip the residual stream into [{lo}, {hi}] at {where}"
        raise InterventionError(f"unknown verb {self.verb!r}")

    def to_json(self) -> dict[str, Any]:
        """The echo the endpoint returns and the bundle stores. The vector
        itself is NOT included: it is 768 numbers that would double every row
        of a sweep, and `direction_id` + `space` already name it."""
        out: dict[str, Any] = {
            "verb": self.verb,
            "layer": self.layer,
            "alpha": _round(self.alpha),
            "is_identity": self.is_identity,
            "protocol": self.protocol,
        }
        for k in ("direction_id", "space", "feature", "sae_repo", "sae_hook"):
            v = getattr(self, k)
            if v is not None:
                out[k] = v
        if self.verb == "clamp":
            out["value"] = _round(self.value)
        if self.verb == "cap":
            out["lo"] = None if self.lo is None else _round(self.lo)
            out["hi"] = None if self.hi is None else _round(self.hi)
        if self.meta:
            out["meta"] = dict(self.meta)
        return out

    # ----------------------------------------------------------------- hooks

    def resid_hooks(self, model: ForwardPass) -> dict[int, ResidHook]:
        """Compile to the `resid_hooks` mapping. EMPTY when `is_identity`."""
        if self.is_identity:
            return {}
        if self.verb == "add":
            return {self._layer(model): _add_hook(self._vec(model), self.alpha)}
        if self.verb == "ablate":
            v = self._vec(model)
            f = _ablate_hook(v, self.alpha)
            layers = (
                range(EMBED_LAYER, model.n_layer)
                if self.layer is None
                else [self._layer(model)]
            )
            return {int(L): f for L in layers}
        if self.verb == "cap":
            return {self._layer(model): _cap_hook(self.lo, self.hi)}
        if self.verb == "clamp":
            raise InterventionError(
                "clamp needs the SAE tensors: call `clamp_hooks(model, tensors)` "
                "or `run(model, prompt, iv, sae=…)`. Building it here would mean "
                "this module downloading weights as a side effect of describing "
                "an intervention."
            )
        raise InterventionError(f"unknown verb {self.verb!r}")

    def _layer(self, model: ForwardPass) -> int:
        if self.layer is None:
            raise InterventionError(f"{self.verb} needs a layer")
        return check_layer(self.layer, model.n_layer)

    def _vec(self, model: ForwardPass) -> np.ndarray:
        if self.vector is None:
            raise InterventionError(f"{self.verb} needs a direction vector")
        v = np.asarray(self.vector, dtype=np.float64).ravel()
        if v.shape[0] != model.d:
            raise InterventionError(
                f"direction {self.direction_id!r} is {v.shape[0]} wide but this "
                f"model's residual stream is {model.d} — the same refusal "
                f"`check_dimensionality` makes on import, at the other end"
            )
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n == 0.0:
            raise InterventionError(f"direction {self.direction_id!r} has zero norm")
        return v / n


def _layer_phrase(layer: int) -> str:
    return "the embedding output" if layer == EMBED_LAYER else f"layer {layer}"


# ------------------------------------------------------------------ the verbs


def _add_hook(v: np.ndarray, alpha: float) -> ResidHook:
    """x ← x + α·v, every position.

    `v` is a unit vector, so α is in residual-norm units and is comparable
    across directions — which is what makes an α sweep on two directions a
    comparison rather than two unrelated x-axes. The caller that wants
    "α in units of this stream's own norm" scales α itself and says so in the
    protocol; this function does not secretly normalise by the prompt.
    """
    step = (np.asarray(v, dtype=np.float64) * float(alpha))[None, :]

    def hook(x: np.ndarray) -> np.ndarray:
        return x + step.astype(x.dtype, copy=False)

    return hook


def _ablate_hook(v: np.ndarray, alpha: float) -> ResidHook:
    """x ← x − α·(x·v)·v, every position.

    α = 1 removes the component entirely; the parameter exists so ablation has
    a sweep axis of its own rather than being a single on/off point next to
    `add`'s curve. This is a per-point subtraction — see the module docstring
    for why it is not `add` with a negative α.
    """
    u = np.asarray(v, dtype=np.float64)
    a = float(alpha)

    def hook(x: np.ndarray) -> np.ndarray:
        comp = x @ u.astype(x.dtype, copy=False)  # (T,)
        return x - a * comp[:, None] * u.astype(x.dtype, copy=False)[None, :]

    return hook


def _cap_hook(lo: float | None, hi: float | None) -> ResidHook:
    """x ← clip(x, lo, hi), elementwise.

    The blunt verb, and honest about it: this caps every coordinate of the
    residual stream, not "the activation of a concept". GPT-2's massive
    activation outliers live here, so a tight cap is a large intervention on a
    small number of coordinates — which the before/after norms in `run()`
    report rather than hide.
    """
    a = -np.inf if lo is None else float(lo)
    b = np.inf if hi is None else float(hi)
    if a > b:
        raise InterventionError(f"cap lo {a} is above hi {b}")

    def hook(x: np.ndarray) -> np.ndarray:
        return np.clip(x, a, b)

    return hook


def clamp_hooks(
    model: ForwardPass,
    iv: "Intervention",
    tensors: Mapping[str, np.ndarray],
) -> dict[int, ResidHook]:
    """Compile a `clamp` against real SAE tensors.

    The SAE is a dictionary over the residual stream, so pinning a feature
    means: encode the stream the way the SAE was trained to (TransformerLens
    `center_writing_weights` centring, then `ReLU((x̄ − b_dec)·W_enc + b_enc)`),
    replace one coordinate, decode, and put the *difference from this run's own
    reconstruction* back into the stream:

        x ← x + α·(a_new − a_old)·W_dec[f]

    Written that way rather than `x ← decode(clamp(encode(x)))` for one reason
    that matters: the SAE does not reconstruct the stream perfectly (cosine
    ≈ 0.9 on this release), so decoding wholesale would silently *also* apply
    the reconstruction error as an intervention, and the α = 0 end of the sweep
    would not be the baseline. With the difference form, α = 0 is exactly zero
    change, and every unit of α is one unit of the feature and nothing else.
    """
    if iv.verb != "clamp":
        raise InterventionError(f"clamp_hooks got verb {iv.verb!r}")
    if iv.is_identity:
        return {}
    W_enc = np.asarray(tensors["W_enc"], dtype=np.float64)  # (d_in, d_sae)
    b_enc = np.asarray(tensors["b_enc"], dtype=np.float64)
    W_dec = np.asarray(tensors["W_dec"], dtype=np.float64)  # (d_sae, d_in)
    b_dec = np.asarray(tensors["b_dec"], dtype=np.float64)
    d_in, d_sae = W_enc.shape
    if d_in != model.d:
        raise InterventionError(
            f"this SAE is trained on a {d_in}-wide stream; the model is {model.d}"
        )
    f = int(iv.feature if iv.feature is not None else -1)
    if not (0 <= f < d_sae):
        raise InterventionError(
            f"feature {f} is outside this SAE's {d_sae} features"
        )
    L = check_layer(
        iv.layer if iv.layer is not None else 0, model.n_layer
    )
    w_enc_f, w_dec_f = W_enc[:, f], W_dec[f]
    target, alpha = float(iv.value), float(iv.alpha)

    def hook(x: np.ndarray) -> np.ndarray:
        x64 = np.asarray(x, dtype=np.float64)
        xc = x64 - x64.mean(axis=1, keepdims=True)  # TL centring, as in bundles
        a_old = np.maximum((xc - b_dec) @ w_enc_f + float(b_enc[f]), 0.0)  # (T,)
        delta = alpha * (target - a_old)  # (T,)
        return (x64 + delta[:, None] * w_dec_f[None, :]).astype(x.dtype, copy=False)

    return {L: hook}


# ------------------------------------------------------------------- builders


def build(
    spec: Mapping[str, Any],
    *,
    resolve_direction: Callable[[str], Any] | None = None,
    n_layer: int = 12,
) -> Intervention:
    """Validate one wire-format intervention. The allow-list lives here.

    `spec` is exactly the `intervention` object of `POST /live/intervene`, and
    this is the only function that reads it. An unknown verb, an unknown key,
    or a missing required field is a flat refusal with the reason — never a
    default that quietly turns a mistyped request into a different experiment.
    """
    if not isinstance(spec, Mapping):
        raise InterventionError("intervention must be an object")
    verb = spec.get("verb")
    if verb not in VERBS:
        raise InterventionError(
            f"unknown intervention verb {verb!r}; this server runs exactly "
            f"{list(VERBS)} and interprets nothing else"
        )
    allowed = {
        "add": {"verb", "direction_id", "alpha", "layer"},
        "ablate": {"verb", "direction_id", "alpha", "layer"},
        "clamp": {"verb", "feature", "value", "alpha", "layer", "sae_repo", "sae_hook"},
        "cap": {"verb", "layer", "lo", "hi"},
    }[verb]
    extra = set(spec) - allowed
    if extra:
        raise InterventionError(
            f"{verb} takes {sorted(allowed - {'verb'})}; got unexpected "
            f"{sorted(extra)}. Refusing rather than ignoring them — an ignored "
            f"key is a parameter the caller thinks it set."
        )

    def _num(key: str, default: float | None = None) -> float:
        v = spec.get(key, default)
        if v is None:
            raise InterventionError(f"{verb} needs {key}")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise InterventionError(f"{key} must be a number, got {v!r}")
        if not np.isfinite(float(v)):
            raise InterventionError(f"{key} must be finite, got {v!r}")
        return float(v)

    if verb == "cap":
        lo = None if spec.get("lo") is None else _num("lo")
        hi = None if spec.get("hi") is None else _num("hi")
        if lo is None and hi is None:
            raise InterventionError("cap needs at least one of lo / hi")
        layer = spec.get("layer")
        if layer is None:
            raise InterventionError("cap needs a layer")
        return Intervention(
            verb="cap", layer=check_layer(int(layer), n_layer), lo=lo, hi=hi
        )

    if verb == "clamp":
        feature = spec.get("feature")
        if not isinstance(feature, int) or isinstance(feature, bool) or feature < 0:
            raise InterventionError(f"clamp needs a non-negative int feature, got {feature!r}")
        layer = spec.get("layer")
        if layer is None:
            raise InterventionError(
                "clamp needs a layer — in HOOK terms (the output of block L). "
                "An SAE at blocks.k.hook_resid_pre is hook layer k−1; see "
                "hook_layer()."
            )
        return Intervention(
            verb="clamp",
            layer=check_layer(int(layer), n_layer),
            feature=feature,
            value=_num("value", 0.0),
            alpha=_num("alpha", 1.0),
            sae_repo=spec.get("sae_repo"),
            sae_hook=spec.get("sae_hook"),
        )

    did = spec.get("direction_id")
    if not isinstance(did, str) or not did:
        raise InterventionError(f"{verb} needs a direction_id")
    if resolve_direction is None:
        raise InterventionError(
            f"{verb} names direction {did!r} but no direction store was given — "
            f"refusing rather than inventing a vector"
        )
    d = resolve_direction(did)
    if d is None:
        raise InterventionError(f"unknown direction {did!r}")
    layer = spec.get("layer")
    if layer is None and verb == "add":
        raise InterventionError("add needs a layer")
    return Intervention(
        verb=verb,
        layer=None if layer is None else check_layer(int(layer), n_layer),
        vector=np.asarray(d.vector, dtype=np.float64),
        direction_id=d.id,
        space=d.space,
        alpha=_num("alpha", 1.0 if verb == "ablate" else 0.0),
    )


def from_direction(
    direction,
    *,
    verb: str = "add",
    alpha: float = 1.0,
    layer: int | None = None,
    n_layer: int = 12,
) -> Intervention:
    """Build an `add`/`ablate` from a `directions.Direction`.

    When `layer` is omitted it is taken from the direction's OWN space, which
    is the only layer at which the direction means anything: a direction fitted
    in `resid.L8` added at layer 2 is a different, unjustified experiment, and
    silently defaulting to 0 would produce exactly that.
    """
    if verb not in ("add", "ablate"):
        raise InterventionError(f"from_direction builds add/ablate, not {verb!r}")
    if layer is None:
        if verb == "ablate":
            pass  # every layer, by definition
        else:
            layer = hook_layer(direction.space, n_layer=n_layer)
    return Intervention(
        verb=verb,
        layer=None if layer is None else check_layer(int(layer), n_layer),
        vector=np.asarray(direction.vector, dtype=np.float64),
        direction_id=direction.id,
        space=direction.space,
        alpha=float(alpha),
    )


# ----------------------------------------------------------------- the runner


def _logsoftmax(row: np.ndarray) -> np.ndarray:
    r = np.asarray(row, dtype=np.float64)
    m = r.max()
    return r - m - np.log(np.exp(r - m).sum())


def generate(
    model: ForwardPass,
    prompt: str,
    *,
    max_tokens: int = 24,
    resid_hooks: Mapping[int, ResidHook] | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Greedy continuation under a set of hooks.

    Greedy, not sampled, and that is the point: with temperature 0 the *only*
    thing that can differ between the baseline and the intervened generation is
    the intervention. A sampled pair would differ anyway and the reader would
    have no way to tell which difference was the finding.

    `gpt2_numpy` has no KV cache, so this recomputes the prefix every step —
    O(n²) in `max_tokens`, roughly 40 ms a token on GPT-2 small. Honest and
    slow beats a cache that quietly drops the hook on cached positions.
    """
    if temperature != 0.0:
        raise InterventionError(
            "only greedy decoding is supported: a sampled before/after pair "
            "differs for two reasons at once and neither can be attributed"
        )
    ids = list(model.encode(prompt))
    if not ids:
        raise InterventionError("empty prompt")
    n0 = len(ids)
    logps: list[float] = []
    for _ in range(int(max_tokens)):
        tr = model.forward(ids, resid_hooks=resid_hooks)
        lp = _logsoftmax(tr.logits[-1])
        nxt = int(np.argmax(lp))
        logps.append(float(lp[nxt]))
        ids.append(nxt)
    new = ids[n0:]
    return {
        "text": "".join(model.decode1(i) for i in new),
        "tokens": [model.decode1(i) for i in new],
        "ids": [int(i) for i in new],
        "logprobs": [_round(x, 4) for x in logps],
        "mean_logprob": _round(float(np.mean(logps)), 4) if logps else None,
        "decoding": "greedy (temperature 0)",
    }


def run(
    model: ForwardPass,
    prompt: str,
    iv: Intervention,
    *,
    max_tokens: int = 24,
    sae: Mapping[str, np.ndarray] | None = None,
    targets: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Baseline and intervened, from one prompt, side by side.

    Returns both generations, both next-token distributions' top-8, the KL
    between them in bits, and the residual-norm change the intervention
    actually made — so a reader can see whether "nothing happened" means the
    hook did nothing or means the model did not care.
    """
    hooks = clamp_hooks(model, iv, sae) if iv.verb == "clamp" and sae is not None else iv.resid_hooks(model)
    base_tr = model.forward(prompt)
    if hooks:
        int_tr = model.forward(prompt, resid_hooks=hooks)
    else:
        int_tr = base_tr  # identity: literally the same run, not a re-run

    blp = _logsoftmax(base_tr.logits[-1])
    ilp = _logsoftmax(int_tr.logits[-1])
    bp = np.exp(blp)
    kl_bits = float((bp * (blp - ilp)).sum() / np.log(2.0))

    def top8(lp: np.ndarray) -> list[list[Any]]:
        order = np.argsort(lp)[::-1][:8]
        return [[model.decode1(int(i)), _round(float(np.exp(lp[i])), 4)] for i in order]

    out: dict[str, Any] = {
        "prompt": prompt,
        "intervention": iv.to_json(),
        "identical": bool(np.array_equal(base_tr.logits, int_tr.logits)),
        "kl_bits": _round(kl_bits, 5),
        "resid_norm_baseline": _round(float(np.linalg.norm(base_tr.resid[-1][-1])), 4),
        "resid_norm_intervened": _round(float(np.linalg.norm(int_tr.resid[-1][-1])), 4),
        "baseline": {
            "top": top8(blp),
            **generate(model, prompt, max_tokens=max_tokens),
        },
        "intervened": {
            "top": top8(ilp),
            **generate(model, prompt, max_tokens=max_tokens, resid_hooks=hooks or None),
        },
    }
    if targets:
        out["targets"] = [
            {
                "text": t,
                "baseline_logprob": _round(_target_logprob(model, prompt, t, None), 4),
                "intervened_logprob": _round(
                    _target_logprob(model, prompt, t, hooks or None), 4
                ),
            }
            for t in targets
        ]
    return out


def _target_logprob(
    model: ForwardPass,
    prompt: str,
    target: str,
    hooks: Mapping[int, ResidHook] | None,
) -> float:
    """Total log p(target | prompt) in nats, teacher-forced.

    The measurement that survives when the greedy continuation does not change:
    an intervention can move a specific completion's probability by a lot while
    the argmax stays put, and reporting only the argmax would call that "no
    effect".
    """
    pid = list(model.encode(prompt))
    tid = list(model.encode(target))
    if not tid:
        raise InterventionError("empty target")
    ids = pid + tid
    tr = model.forward(ids, resid_hooks=hooks)
    total = 0.0
    for k, t in enumerate(tid):
        total += float(_logsoftmax(tr.logits[len(pid) - 1 + k])[t])
    return total


def sweep(
    model: ForwardPass,
    prompts: Sequence[str],
    make: Callable[[float], Intervention],
    alphas: Sequence[float],
    *,
    max_tokens: int = 16,
    sae: Mapping[str, np.ndarray] | None = None,
    targets: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run one intervention across a grid of α, over a set of prompts.

    `make(alpha) -> Intervention`. The α = 0 row is not special-cased here: it
    is special-cased in `Intervention.is_identity`, so if a caller's `make`
    produces something that is *not* the identity at α = 0, the curve shows it
    instead of the harness papering over it.
    """
    rows: list[dict[str, Any]] = []
    for a in alphas:
        iv = make(float(a))
        per: list[dict[str, Any]] = []
        for p in prompts:
            per.append(run(model, p, iv, max_tokens=max_tokens, sae=sae, targets=targets))
        kl = [r["kl_bits"] for r in per]
        rows.append(
            {
                "alpha": _round(float(a)),
                "is_identity": iv.is_identity,
                "protocol": iv.protocol,
                "kl_bits_mean": _round(float(np.mean(kl)), 5),
                "kl_bits_max": _round(float(np.max(kl)), 5),
                "identical_to_baseline": all(r["identical"] for r in per),
                "runs": per,
            }
        )
    return {
        "kind": "intervention_sweep",
        "alphas": [_round(float(a)) for a in alphas],
        "prompts": list(prompts),
        "max_tokens": int(max_tokens),
        "rows": rows,
        "claim": claim_sentence(rows),
        "notes": {
            "decoding": "greedy (temperature 0) — the only difference between "
            "the two columns is the intervention",
            "control": "the alpha = 0 row installs NO hook and is bit-identical "
            "to the baseline; `identical_to_baseline` asserts it per row",
            "d6": "no weights were modified or written; every verb is an "
            "inference-time hook",
        },
    }


def claim_sentence(rows: Sequence[Mapping[str, Any]]) -> str:
    """The one causal sentence §2.4 permits, or the honest refusal of it.

    It names the protocol and the measured change, and it never names what the
    direction *is*. When no row moved the model, it says that instead — a sweep
    that found nothing is a result, and the figure that shows it must say so
    rather than leaving the sentence off.
    """
    moved = [r for r in rows if not r.get("is_identity") and float(r.get("kl_bits_max", 0)) > 0]
    if not moved:
        return (
            "No intervention in this sweep changed the model's next-token "
            "distribution. Not measured as an effect: measured as none."
        )
    best = max(moved, key=lambda r: float(r["kl_bits_max"]))
    return (
        f"Under this protocol — {best['protocol']}, greedy decoding, "
        f"n = {len(rows)} alphas — the intervention changed the next-token "
        f"distribution by up to {float(best['kl_bits_max']):.3f} bits of KL. "
        f"That is a statement about what this intervention did, not about what "
        f"the direction is."
    )


def sweep_digest(bundle: Mapping[str, Any]) -> str:
    """A short stable hash of a sweep's inputs, for the figure stamp."""
    h = hashlib.sha256()
    h.update(repr([bundle.get("alphas"), bundle.get("prompts"), bundle.get("max_tokens")]).encode())
    for r in bundle.get("rows", ()):
        h.update(str(r.get("protocol", "")).encode())
    return h.hexdigest()[:12]
