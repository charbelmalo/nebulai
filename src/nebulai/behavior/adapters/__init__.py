"""Execution backends for one Lane A trial.

An adapter's only job is: given a rendered prompt and frozen sampler settings,
return exactly what the backend produced plus the provenance needed to audit it.
Adapters never parse, never normalize, never decide validity, and never retry —
those are the runner's and the normalizer's jobs, and keeping them out here is
what makes the same statistics apply to a local completion model and a remote
chat model without either being special-cased.

`resolve_adapter` is the only public entry point that maps a manifest's
`adapter` string to an implementation. An unknown string is an error: a study
whose arm silently ran on a different backend than its manifest names is not a
study.
"""

from __future__ import annotations

from .base import Adapter, AdapterError, Completion

__all__ = ["Adapter", "AdapterError", "Completion", "resolve_adapter"]


def resolve_adapter(name: str, **kw: object) -> Adapter:
    if name == "fake":
        from .fake import FakeAdapter

        return FakeAdapter(**kw)  # type: ignore[arg-type]
    if name in ("gpt2-local", "gpt2_local"):
        from .gpt2_local import GPT2LocalAdapter

        return GPT2LocalAdapter(**kw)  # type: ignore[arg-type]
    if name == "xai":
        from .xai import XAIAdapter

        return XAIAdapter(**kw)  # type: ignore[arg-type]
    raise AdapterError(
        f"unknown adapter {name!r}: expected 'gpt2-local', 'xai' or 'fake'. "
        f"Nebul.AI does not substitute a reachable backend for an unreachable "
        f"one — a missing route is refused, never routed around."
    )
