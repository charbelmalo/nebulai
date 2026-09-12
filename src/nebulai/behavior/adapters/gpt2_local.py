"""The gpt2-local adapter's public path (§5.6).

The implementation is `nebulai.organisms.gpt2_local`: it is the module that
touches torch, and the base-install rule (`tests/test_no_torch_in_base.py`)
keeps every torch import — lazy ones included — under `organisms/`. This shim
exists so the adapter registry, the CLI conformance command and the tests keep
their existing import path.
"""

from __future__ import annotations

from ...organisms.gpt2_local import (
    GOLDEN_PROMPTS,
    LOGIT_ATOL,
    TOP_K_AGREEMENT,
    GPT2LocalAdapter,
    _apply_stop,
    conformance,
)

__all__ = [
    "GOLDEN_PROMPTS",
    "LOGIT_ATOL",
    "TOP_K_AGREEMENT",
    "GPT2LocalAdapter",
    "_apply_stop",
    "conformance",
]
