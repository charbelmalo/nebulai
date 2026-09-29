"""Model organisms — the one corner of this project that needs gradients.

Everything else here is numpy. The maps, the directions, the persona space, the
absorbing-state study, the whole Llama forward pass: numpy and safetensors, no
framework, and that is why `pip install nebulai` is small and why the static
deploy has nothing to install at all. Training a model organism cannot be done
that way — it needs autograd — so torch lives behind an extra:

    pip install 'nebulai[organisms]'

Importing *this package* never imports torch and never raises; that is what lets
`tests/test_no_torch_in_base.py` check the boundary from a torch-free venv. The
modules under it call `require_torch()` at the top of their entry points, so a
user without the extra gets the line above instead of a traceback about a
missing module they never asked for.

What is here
------------
`emergent_misalignment.py` — the rank-1 LoRA over the published insecure-code
set and its one-sentence inoculation control (Phase 5 of
docs/ATTRACTORS-PLAN.md). It was run once on 2026-09-12, on CPU under a 24-minute
wall clock, so the shipped record (`out/organisms/emergent_misalignment.json`)
is a truncated run: 7 and 6 optimiser steps, 6 held-out eval pairs, both arms
stamped `stopped_early`. What it measured is the direction geometry (per-layer
cosine between the two arms' learned rank-1 directions against a random-unit
null) and the narrow held-out logprob margin; the broad misalignment rate is
recorded as `not_measured`, with the reason, because no permitted judge exists
here. The adapted weights were never written (D6).

`gpt2_local.py` and `sentence_embedder.py` — the Behavior study's batched
Transformers GPT-2 sampler and its pinned sentence encoder. They live here
because they import torch; the `behavior/` package re-exports them through
thin shims so its own modules stay importable from a torch-free venv.
"""

from __future__ import annotations

import importlib.util

__all__ = ["OrganismsNotInstalled", "have_torch", "require_torch"]

_HINT = (
    "This needs the optional `organisms` extra, which is the only part of "
    "nebulai that depends on torch:\n"
    "    pip install 'nebulai[organisms]'\n"
    "Nothing else in the project requires it — maps, directions, the persona "
    "space and the numpy forward pass all run without it."
)


class OrganismsNotInstalled(RuntimeError):
    """Raised when a gradient-requiring entry point is called without the extra."""


def have_torch() -> bool:
    """True when torch is importable — checked WITHOUT importing it.

    `find_spec` is deliberate: importing torch costs seconds and hundreds of
    megabytes of resident memory, and an availability check must not pay that
    just to answer a yes/no question.
    """
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # a broken or shadowed install
        return False


def require_torch() -> None:
    """Fail with the install line rather than a `ModuleNotFoundError` traceback."""
    if not have_torch():
        raise OrganismsNotInstalled(_HINT)
