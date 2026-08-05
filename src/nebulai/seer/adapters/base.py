"""Shared adapter machinery: span bookkeeping, usage folding, reasoning drop.

Everything here exists to make the three contract rules mechanical rather than
a thing each adapter has to remember:

* `Usage.fold` refuses to add the same fold key twice. Claude repeats identical
  `usage` on every content line of one response; Codex repeats it on
  `item.updated`. Both would multiply totals several-fold if the adapter added
  each sighting. The guard is here, not in each adapter, because "remember to
  only count it once" is exactly the kind of instruction that gets forgotten in
  the fourth adapter.

* `drop_reasoning` is called at ingress, before anything is persisted. All three
  agents stream reasoning: Codex `item.reasoning`, Claude `thinking` content
  blocks, Hermes `reasoning.delta`. Not storing it is a decision we implement,
  and the resulting field is `dropped_by_policy`, never `missing`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

from ..contract import (
    CaptureMode,
    Event,
    EventType,
    Fidelity,
    Source,
    TokenCategory,
    fold_key,
)

ADAPTER_VERSION = "0.1.0"


@dataclass
class Usage:
    """Token counts with a fold guard.

    `by_category` keeps the agent's own buckets (see `TokenCategory` — they do
    not mean the same thing across agents and must not be summed across them).
    """

    by_category: dict[TokenCategory, int] = field(default_factory=dict)
    _seen: set[str] = field(default_factory=set)
    #: fold keys we refused as duplicates — surfaced so a fold bug is visible
    #: rather than silently halving or doubling a total.
    duplicates: int = 0

    def fold(self, key: str, counts: dict[TokenCategory, int]) -> bool:
        """Add `counts` once for `key`. Returns False if `key` was already
        counted (the caller may still update a live preview from it)."""
        if key in self._seen:
            self.duplicates += 1
            return False
        self._seen.add(key)
        for cat, n in counts.items():
            if n is None:
                continue
            self.by_category[cat] = self.by_category.get(cat, 0) + int(n)
        return True

    def replace(self, key: str, counts: dict[TokenCategory, int]) -> None:
        """Overwrite totals from an authoritative terminal record.

        Claude's `result` line carries the real session totals; the per-message
        `usage` seen during streaming is partial. When ground truth arrives we
        replace rather than accumulate — the same reason `sessionlog.ts` treats
        the audit result line as authoritative over the streamed values.
        """
        self._seen.add(key)
        self.by_category = {k: int(v) for k, v in counts.items() if v is not None}

    def as_dict(self) -> dict[str, int]:
        return {c.value: n for c, n in self.by_category.items()}


@dataclass
class AdapterResult:
    """What an adapter knows once its stream has ended."""

    events: list[Event] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    #: native token-category names actually seen, for the comparability gate
    native_usage_keys: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)


class Adapter(Protocol):
    """Line-oriented normalizer. `feed` is called once per line of agent output
    and returns zero or more canonical events."""

    agent: str

    def feed(self, line: str) -> list[Event]: ...

    def finish(self) -> list[Event]: ...


class BaseAdapter:
    """Common state for the three concrete adapters."""

    agent = "unknown"
    adapter_name = "unknown"

    def __init__(
        self,
        *,
        run_id: str,
        session_id: str,
        agent_version: str = "unknown",
        capture_mode: CaptureMode = CaptureMode.DRIVEN,
        repo: dict[str, Any] | None = None,
        keep_reasoning: bool = False,
    ) -> None:
        self.run_id = run_id
        self.session_id = session_id
        self.agent_version = agent_version
        self.capture_mode = capture_mode
        self.repo = repo
        #: Opt-in, off by default. When False the adapter drops reasoning text
        #: before it can reach the store, and says `dropped_by_policy`.
        self.keep_reasoning = keep_reasoning

        self.usage = Usage()
        self.native_usage_keys: set[str] = set()
        self.warnings: list[str] = []
        self.model: dict[str, Any] | None = None
        self.turn_id: str | None = None
        #: native span id → our span id, so `item.completed` can close the span
        #: `item.started` opened.
        self.spans: dict[str, str] = {}
        #: native span id → monotonic start, for deterministic durations when
        #: the agent does not report one.
        self.span_started: dict[str, float] = {}
        self._unknown_native: set[str] = set()

    # ── event construction ───────────────────────────────────────────────

    def _source(
        self,
        fidelity: Fidelity = Fidelity.NATIVE,
        source_event_id: str | None = None,
    ) -> Source:
        return Source(
            agent=self.agent,
            agent_version=self.agent_version,
            adapter=self.adapter_name,
            adapter_version=ADAPTER_VERSION,
            capture_mode=self.capture_mode,
            fidelity=fidelity,
            source_event_id=source_event_id,
        )

    def event(
        self,
        event_type: EventType,
        *,
        fidelity: Fidelity = Fidelity.NATIVE,
        source_event_id: str | None = None,
        **kw: Any,
    ) -> Event:
        kw.setdefault("turn_id", self.turn_id)
        kw.setdefault("model", self.model)
        kw.setdefault("repo", self.repo)
        return Event(
            event_type=event_type,
            source=self._source(fidelity, source_event_id),
            run_id=self.run_id,
            session_id=self.session_id,
            **kw,
        )

    def warn(self, msg: str) -> Event:
        """A warning about the capture, in SessionSeer's own words.

        The key is `note`, not `message`, because `message` carries the
        *agent's* text — an error body that can hold anything — and is
        classified as content. A data-quality panel that goes blank the moment
        someone exports a run at `metadata` level would be telling them the
        capture was clean.
        """
        self.warnings.append(msg)
        return self.event(
            EventType.ADAPTER_WARNING,
            fidelity=Fidelity.DETERMINISTIC,
            payload={"note": msg},
        )

    def note_unknown_native(self, kind: str) -> list[Event]:
        """Record an unrecognised native event kind, once.

        Emitting a warning rather than dropping silently is the difference
        between "this agent version added an event we don't map" being visible
        in the data-quality panel and being invisible forever.
        """
        if kind in self._unknown_native:
            return []
        self._unknown_native.add(kind)
        return [self.warn(f"unmapped native event kind: {kind!r}")]

    # ── reasoning policy ─────────────────────────────────────────────────

    def reasoning_payload(self, text: str | None) -> tuple[dict[str, Any], Fidelity]:
        """Payload + fidelity for a reasoning fragment.

        All three agents will hand us this text. Returning
        `DROPPED_BY_POLICY` (not `MISSING`) is the whole point: a researcher
        reading the data-quality panel must be able to tell "the agent never
        told us" from "we chose not to keep it".
        """
        if self.keep_reasoning:
            return {"text": text or ""}, Fidelity.NATIVE
        return (
            {"chars": len(text or ""), "text_retained": False},
            Fidelity.DROPPED_BY_POLICY,
        )

    # ── usage ────────────────────────────────────────────────────────────

    def fold_usage(
        self, key: str, counts: dict[TokenCategory, int], native_keys: Iterable[str]
    ) -> bool:
        self.native_usage_keys.update(native_keys)
        return self.usage.fold(key, counts)

    def result(self, events: list[Event]) -> AdapterResult:
        return AdapterResult(
            events=events,
            usage=self.usage,
            native_usage_keys=set(self.native_usage_keys),
            warnings=list(self.warnings),
        )

    # ── protocol defaults ────────────────────────────────────────────────

    def finish(self) -> list[Event]:
        return []


__all__ = [
    "ADAPTER_VERSION",
    "Adapter",
    "AdapterResult",
    "BaseAdapter",
    "Usage",
    "fold_key",
]
