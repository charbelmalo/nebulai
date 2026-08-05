"""SessionSeer — real-time observability over Codex / Claude Code / Hermes runs.

The fourth nebulai front-end. Where `tokens` / `sae` / `neurons` map a *model's*
concept space, SessionSeer maps an *agent's* trajectory space. It shares the
viewer shell, the provenance discipline, and the export contract; it does NOT
share the reduce → cluster → name back-end, because a trajectory is already
low-dimensional and ordered.

Plan: docs/SESSIONSEER.md.
"""

from .contract import (  # noqa: F401
    ACTIONS,
    Action,
    CaptureMode,
    Effect,
    Event,
    EventType,
    Fidelity,
    Outcome,
    SessionState,
    Source,
    TokenCategory,
)

__all__ = [
    "ACTIONS",
    "Action",
    "CaptureMode",
    "Effect",
    "Event",
    "EventType",
    "Fidelity",
    "Outcome",
    "SessionState",
    "Source",
    "TokenCategory",
]
