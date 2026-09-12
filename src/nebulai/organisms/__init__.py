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

What is NOT here
----------------
`emergent_misalignment.py` — the rank-1 LoRA over the insecure-code set and its
inoculation control — is not in this package. Phase 5 of docs/ATTRACTORS-PLAN.md
is marked optional and it was not run, so nothing here claims it was: an
untrained training script that has never produced a curve is a sketch wearing a
module name, and this project's honesty rules apply to its own source tree too.
The extra and this guard are the half that is real, and they are here because
they are what the rest of that phase needs in place first.
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
