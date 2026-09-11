"""The adapter contract: one prompt in, one audited completion out."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class AdapterError(RuntimeError):
    """A backend could not serve the request as specified.

    Distinct from a *failed trial*: an `AdapterError` means the protocol could
    not be honoured (wrong model served, missing credentials, unsupported
    sampler parameter), and the runner stops or records `not_run` rather than
    pretending the arm produced data.
    """


@dataclass
class Completion:
    """What one request returned, plus everything §5.5/§6.1 require recording."""

    text: str
    requested_model: str = ""
    served_model: str = ""
    response_id: str = ""
    fingerprint: str = ""
    latency_ms: int | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    #: Free-form backend detail (device, dtype, checkpoint sha). Goes into the
    #: run's provenance block, never into a metric.
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class SamplerSettings:
    """The sampler parameters frozen in the manifest. Passed, never defaulted.

    Defaulting any of these inside an adapter would let two arms run at
    different temperatures while the manifest claimed one — the single easiest
    way to manufacture a divergence that is entirely an artifact of the runner.
    """

    temperature: float
    top_p: float
    max_output_tokens: int
    seed: int
    stop: tuple[str, ...] = ()


class Adapter(Protocol):
    #: Manifest adapter key, e.g. "gpt2-local".
    name: str
    #: The exact model id this instance serves. Compared against the manifest's
    #: `pinned` value by the runner before the first trial.
    pinned: str
    #: True when the backend bills. Gates the runner's approval path: a free
    #: local arm must never be made to wait on a paid-run approval, and a paid
    #: arm must never start without one.
    paid: bool

    def describe(self) -> dict[str, Any]:
        """Provenance for the manifest/run record. No side effects."""
        ...

    def complete(self, prompt: str, settings: SamplerSettings, *, trial_seed: int) -> Completion:
        """One fresh request. Raises `AdapterError` if it cannot be honoured."""
        ...
