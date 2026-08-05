"""Adapters: native agent output → canonical `Event` stream.

One module per agent. Each exposes a `*Adapter` class with `feed(line) ->
list[Event]` so the same normalizer works for a live subprocess, a replayed
fixture, and a spool file. Nothing downstream of here may read a native field.

Fidelity is decided per adapter, not per agent: `codex exec --json` is a much
thinner stream than the Codex app-server (7 event kinds vs 68 notifications), so
the DRIVEN Codex adapter honestly reports less than an ATTACHED one would. That
is the opposite of the usual assumption that "we launched it" means "we see
everything", and the data-quality panel has to say so.
"""

from .base import Adapter, AdapterResult  # noqa: F401
from .claude import ClaudeStreamAdapter  # noqa: F401
from .codex import CodexExecAdapter  # noqa: F401
from .codex_app_server import CodexAppServerAdapter  # noqa: F401
from .hermes import HermesOneshotAdapter  # noqa: F401

__all__ = [
    "Adapter",
    "AdapterResult",
    "ClaudeStreamAdapter",
    "CodexAppServerAdapter",
    "CodexExecAdapter",
    "HermesOneshotAdapter",
]


def adapter_for(agent: str, mode: str = "driven", **kw) -> Adapter:
    """Construct the adapter for `agent` in `mode`.

    Raises on an unknown agent *or* an unknown (agent, mode) pair rather than
    falling back to the driven one — a silently substituted adapter would
    produce a plausible, unlabelled, wrong trajectory, and the substitution
    would be invisible in the output.
    """
    a, m = agent.lower(), mode.lower()
    if a == "codex":
        if m == "attached":
            return CodexAppServerAdapter(**kw)
        if m == "driven":
            return CodexExecAdapter(**kw)
    elif a == "claude" and m == "driven":
        return ClaudeStreamAdapter(**kw)
    elif a == "hermes" and m == "driven":
        return HermesOneshotAdapter(**kw)
    if a not in ("codex", "claude", "hermes"):
        raise ValueError(f"no adapter for agent {agent!r} (have: codex, claude, hermes)")
    raise ValueError(
        f"no {mode} adapter for {agent!r}; attached mode exists for codex only"
    )
