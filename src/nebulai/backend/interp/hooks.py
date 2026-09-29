"""The forward-pass hook protocol, shared by every model runner in this repo.

One interface, written down once, so that an intervention written against
`gpt2_numpy` runs unchanged against `llama_numpy` (and whatever comes next)
without either module importing the other:

    forward(tokens, *, resid_hooks=None, cache=None)

`resid_hooks` maps a layer index to a callable. The callable receives the
residual stream **after block L** as a `(T, d)` array and returns the
replacement, same shape. Layer **-1** is the embedding output before block 0 —
the one position where "after block L" has no block, and the one place a
direction can be injected before any attention has read it.

A note on dtype, because it was measured rather than assumed. The stream is
nominally float32 and `gpt2_numpy`'s own docstring says so, but that module
accumulates in **float64** from block 0 onward: `np.sqrt(dh)` in the attention
scale is a numpy scalar, and under NEP 50 a float32 array divided by one
promotes. Casting the hook's input or output to float32 therefore perturbs the
run — measured at 2.4e-4 in the residual stream and 1.5e-5 in the logits for an
IDENTITY hook on GPT-2 at layer 8, which is small but is not zero and would
show up as a real-looking effect at the bottom of an alpha sweep. So the
contract is: a hook is handed the array the runner is actually carrying, and
its result is cast back to that same dtype. Write hooks that do not care —
`np.asarray` and ordinary arithmetic are dtype-agnostic — and an identity hook
is then bit-identical to no hook at all.

Three properties this file exists to fix in place, because they are the ones a
second implementation silently gets wrong:

· **The hook sees the stream AFTER the block, not before.** `Trace.resid[L]` is
  the stream *entering* block L, so the hook for layer L is observed at
  `resid[L + 1]`. Off by one here and every α-sweep is a sweep at the wrong
  depth, with results that look perfectly plausible.
· **A hook returns a new array or the same one; it never mutates in place.**
  Runners are free to keep the array they passed in (`gpt2_numpy` writes it
  straight into `Trace.resid`), so an in-place edit would also rewrite the
  recorded trajectory and the export would show the intervention as though it
  had always been there.
· **No hook at all must be bit-identical to no hooks argument.** `alpha = 0` is
  the control in every sweep, and a control that differs from the baseline in
  the last mantissa bit turns a null result into a tiny fake effect.
  Interventions therefore report `is_identity` and are dropped before they
  reach the runner, rather than installed as a no-op that still round-trips
  through a dtype conversion.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np

#: a residual-stream hook: `(T, d) float32 -> (T, d) float32`
ResidHook = Callable[[np.ndarray], np.ndarray]

#: the layer index meaning "the embedding output, before block 0"
EMBED_LAYER = -1


class HookError(ValueError):
    """A hook returned something that is not a residual stream."""


@runtime_checkable
class ForwardPass(Protocol):
    """What an intervention needs from a model. Deliberately tiny.

    `intervene.py` is written against THIS and nothing else, so it never grows
    a `if isinstance(model, GPT2Numpy)` branch — the moment it does, one model
    gets an intervention the others silently do not.
    """

    #: number of transformer blocks
    n_layer: int
    #: residual width
    d: int

    def forward(
        self,
        prompt: Any,
        *,
        resid_hooks: Mapping[int, ResidHook] | None = None,
        cache: Any | None = None,
    ) -> Any:
        """Run the model, applying each hook to the stream after its block."""
        ...

    def encode(self, text: str) -> list[int]: ...

    def decode1(self, tid: int) -> str: ...


def check_layer(layer: int, n_layer: int) -> int:
    """Validate a hook layer against a model's depth.

    Negative indices are NOT python-style: `-1` is the embedding output, and
    nothing below it exists. Accepting `-2` as "second from the end" would make
    the same integer mean two different depths depending on which module read
    it, which is exactly the kind of ambiguity a shared protocol is for.
    """
    if not isinstance(layer, (int, np.integer)):
        raise HookError(f"layer must be an int, got {type(layer).__name__}")
    layer = int(layer)
    if layer < EMBED_LAYER or layer >= n_layer:
        raise HookError(
            f"layer {layer} is outside this model: valid layers are "
            f"{EMBED_LAYER} (the embedding output) through {n_layer - 1}"
        )
    return layer


def apply_resid_hook(hook: ResidHook, x: np.ndarray, *, layer: int) -> np.ndarray:
    """Run one hook and refuse anything that is not the stream it was handed.

    Called by the runners, not by callers of the runners. A hook that changes
    the shape does not produce a slightly wrong answer — it produces a
    confidently wrong one several blocks later, where the shape mismatch has
    already been absorbed by a matmul.
    """
    out = hook(x)
    if not isinstance(out, np.ndarray):
        raise HookError(
            f"the hook on layer {layer} returned {type(out).__name__}, not an array"
        )
    if out.shape != x.shape:
        raise HookError(
            f"the hook on layer {layer} returned shape {out.shape}, but the "
            f"residual stream there is {x.shape}"
        )
    if not np.isfinite(out).all():
        raise HookError(
            f"the hook on layer {layer} returned non-finite values — a NaN here "
            f"propagates to every later position and every logit"
        )
    # back to the dtype the runner is carrying, not to a nominal float32: see
    # the module docstring for the measurement that settled this.
    if out.dtype == x.dtype and out.flags["C_CONTIGUOUS"]:
        return out
    return np.ascontiguousarray(out, dtype=x.dtype)
