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
from .corpus_amongus import AmongUsCorpusAdapter  # noqa: F401
from .corpus_base import CorpusAdapter, CorpusError, CorpusRun, CorpusSource  # noqa: F401
from .corpus_ctfish import CtfishCorpusAdapter  # noqa: F401
from .corpus_transcript import TranscriptCorpusAdapter  # noqa: F401
from .corpus_village import VillageCorpusAdapter, VillageUnavailable  # noqa: F401
from .hermes import HermesOneshotAdapter  # noqa: F401

#: corpus id → adapter class. `seer import <corpus>` reads this, so adding a
#: fifth corpus is one import and one entry rather than a new CLI branch.
CORPUS_ADAPTERS: dict[str, type[CorpusAdapter]] = {
    "amongus": AmongUsCorpusAdapter,
    "ctfish": CtfishCorpusAdapter,
    "village": VillageCorpusAdapter,
    "transcript": TranscriptCorpusAdapter,
}


def corpus_adapter(corpus: str, **kw) -> CorpusAdapter:
    """Construct a corpus adapter by id, refusing an unknown one.

    Same rule as `adapter_for`: an unknown corpus raises rather than falling
    back, because a silently substituted mapping produces a plausible,
    unlabelled, wrong trajectory.
    """
    cls = CORPUS_ADAPTERS.get(corpus.lower())
    if cls is None:
        raise ValueError(
            f"no corpus adapter for {corpus!r} (have: {', '.join(sorted(CORPUS_ADAPTERS))})"
        )
    return cls(**kw)

__all__ = [
    "CORPUS_ADAPTERS",
    "Adapter",
    "AdapterResult",
    "AmongUsCorpusAdapter",
    "ClaudeStreamAdapter",
    "CodexAppServerAdapter",
    "CodexExecAdapter",
    "CorpusAdapter",
    "CorpusError",
    "CorpusRun",
    "CorpusSource",
    "CtfishCorpusAdapter",
    "HermesOneshotAdapter",
    "TranscriptCorpusAdapter",
    "VillageCorpusAdapter",
    "VillageUnavailable",
    "corpus_adapter",
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
