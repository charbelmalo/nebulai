"""Events → the state a human reads. The state machine, and nothing clever.

Two decisions here carry most of the weight.

**Stalled is an overlay.** `SessionState` has no `stalled` member on purpose. A
run that has been quiet for four minutes is still `tool_running`, and *which
tool* is the actionable half — "stalled" alone tells a watcher to look, not
where. So `RunView` carries `state` and `overlays` separately, and the stall
budget is per-state: a model call that has been quiet for 90s is normal, a
`waiting_permission` that has been quiet for 90s is a person who walked away.

**Every number says where it came from.** `Measured` is the only numeric type
that leaves this module. A count we derived is `DETERMINISTIC`; a token total the
agent reported is `NATIVE`; a category the agent has no bucket for is `MISSING`
with a reason attached, and there is no code path that turns that into a zero.
`Measured.value is None` and `Measured.value == 0` are different facts and the
viewer renders them differently (`formatMeasured` in contract.ts).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from .contract import (
    Action,
    Effect,
    Event,
    EventType,
    Fidelity,
    Outcome,
    SessionState,
    StateOverlay,
    TokenCategory,
)
from .taxonomy import unmatched_tools

#: How long a state may stay quiet before the STALLED overlay applies, in
#: seconds. Tuned to the shape of each wait rather than to one global number:
#: a 3-minute test run is healthy, a 3-minute pause on an approval prompt means
#: nobody is at the keyboard.
STALL_BUDGET_S: dict[SessionState, float] = {
    SessionState.STARTING: 30.0,
    SessionState.MODEL_RUNNING: 120.0,
    SessionState.TOOL_RUNNING: 300.0,
    SessionState.WAITING_PERMISSION: 60.0,
    SessionState.WAITING_CLARIFICATION: 60.0,
    SessionState.WAITING_USER: 900.0,
    SessionState.COMPACTING: 180.0,
    SessionState.IDLE: 600.0,
}

#: Waiting states where the pending party is a human. These get OVERDUE rather
#: than STALLED, because the response is different: one needs a person, the
#: other needs someone to check whether the process is alive.
_HUMAN_WAITS = frozenset(
    {
        SessionState.WAITING_PERMISSION,
        SessionState.WAITING_CLARIFICATION,
        SessionState.WAITING_USER,
    }
)

_TRANSITIONS: dict[EventType, SessionState] = {
    EventType.SESSION_STARTED: SessionState.STARTING,
    EventType.SESSION_RESUMED: SessionState.STARTING,
    EventType.TURN_STARTED: SessionState.MODEL_RUNNING,
    EventType.MODEL_REQUEST_STARTED: SessionState.MODEL_RUNNING,
    EventType.MESSAGE_ASSISTANT_COMPLETED: SessionState.MODEL_RUNNING,
    EventType.TOOL_STARTED: SessionState.TOOL_RUNNING,
    # A tool that returned is a tool that is no longer running. Without these
    # two, the state stays TOOL_RUNNING until the *next* tool starts, and every
    # gap between tool calls — which is mostly the model thinking — is billed to
    # whichever tool happened to finish before it.
    EventType.TOOL_COMPLETED: SessionState.MODEL_RUNNING,
    EventType.TOOL_FAILED: SessionState.MODEL_RUNNING,
    EventType.APPROVAL_REQUESTED: SessionState.WAITING_PERMISSION,
    EventType.APPROVAL_RESOLVED: SessionState.MODEL_RUNNING,
    EventType.CLARIFICATION_REQUESTED: SessionState.WAITING_CLARIFICATION,
    EventType.CLARIFICATION_RESOLVED: SessionState.MODEL_RUNNING,
    EventType.COMPACTION_STARTED: SessionState.COMPACTING,
    EventType.COMPACTION_COMPLETED: SessionState.MODEL_RUNNING,
    EventType.TURN_COMPLETED: SessionState.IDLE,
    EventType.TURN_FAILED: SessionState.IDLE,
    EventType.SESSION_COMPLETED: SessionState.COMPLETED,
    EventType.SESSION_FAILED: SessionState.FAILED,
    EventType.SESSION_INTERRUPTED: SessionState.INTERRUPTED,
}


@dataclass(slots=True)
class Measured:
    """A number that knows its provenance. Mirrors `Measured` in contract.ts."""

    value: float | int | None
    fidelity: Fidelity
    note: str | None = None

    @property
    def absent(self) -> bool:
        return self.value is None or self.fidelity in (
            Fidelity.MISSING,
            Fidelity.DROPPED_BY_POLICY,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"value": self.value, "fidelity": self.fidelity.value}
        if self.note:
            d["note"] = self.note
        return d


def missing(note: str) -> Measured:
    return Measured(None, Fidelity.MISSING, note)


@dataclass(slots=True)
class SpanRecord:
    span_id: str
    action: Action | None
    native_type: str | None
    started_at: float
    ended_at: float | None = None
    effect: Effect | None = None
    failed: bool = False
    detail: str | None = None
    #: kept so the time decomposition can subtract a child's time from its
    #: parent instead of counting the same second twice
    parent_span_id: str | None = None
    #: The agent's own clock for this call, when it reports one — Codex's
    #: app-server does, `codex exec --json` does not. Kept *beside*
    #: `duration_s` rather than replacing it: the interval between our two
    #: events is what the overlap arithmetic in `time_decomposition` unions,
    #: and mixing a differently-sourced number into that would produce a
    #: total that no set of intervals adds up to. Two numbers that disagree
    #: are information; one number of unknown provenance is not.
    native_duration_s: float | None = None
    #: True when we never saw this call begin and stamped its start from its
    #: end. `duration_s` is then 0.0 by construction, which is not a
    #: measurement of anything and must not be reported as one.
    synthetic_start: bool = False

    @property
    def duration_s(self) -> float | None:
        """Wall clock between the start and end events we saw."""
        return None if self.ended_at is None else self.ended_at - self.started_at

    @property
    def best_duration_s(self) -> float | None:
        """The agent's own timing when it reports one, ours otherwise — and
        nothing at all when neither exists."""
        if self.native_duration_s is not None:
            return self.native_duration_s
        return None if self.synthetic_start else self.duration_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "action": self.action.value if self.action else None,
            "native_type": self.native_type,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": self.duration_s,
            "native_duration_s": self.native_duration_s,
            "duration_fidelity": (
                Fidelity.NATIVE.value if self.native_duration_s is not None
                else Fidelity.MISSING.value if self.synthetic_start
                else Fidelity.DETERMINISTIC.value
            ),
            "synthetic_start": self.synthetic_start,
            "effect": self.effect.value if self.effect else None,
            "failed": self.failed,
            "detail": self.detail,
            "parent_span_id": self.parent_span_id,
        }


@dataclass(slots=True)
class DataQuality:
    """What this run could not see, and why. Rendered as a first-class panel
    rather than a footnote: a run captured in OBSERVED mode and one captured
    DRIVEN are not the same measurement, and a chart that pretends otherwise is
    the failure this whole subsystem is built to avoid."""

    capture_mode: str | None = None
    capture_gaps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unmapped_native: list[str] = field(default_factory=list)
    unmatched_tools: list[str] = field(default_factory=list)
    absent_token_categories: list[str] = field(default_factory=list)
    dropped_by_policy: dict[str, int] = field(default_factory=dict)
    #: usage sightings the fold rule refused. Non-zero is not an error — it is
    #: the rule working — but a sudden change in the ratio means the agent
    #: changed its streaming shape and the fold key needs revisiting.
    folded_duplicates: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture_mode": self.capture_mode,
            "capture_gaps": self.capture_gaps,
            "warnings": self.warnings,
            "unmapped_native": self.unmapped_native,
            "unmatched_tools": self.unmatched_tools,
            "absent_token_categories": self.absent_token_categories,
            "dropped_by_policy": self.dropped_by_policy,
            "folded_duplicates": self.folded_duplicates,
        }


@dataclass(slots=True)
class RunView:
    run_id: str
    agent: str = "unknown"
    agent_version: str = "unknown"
    model: dict[str, Any] | None = None
    repo: dict[str, Any] | None = None

    state: SessionState = SessionState.STARTING
    overlays: list[StateOverlay] = field(default_factory=list)
    outcome: Outcome = Outcome.UNKNOWN

    started_at: float | None = None
    ended_at: float | None = None
    last_event_at: float | None = None

    #: seconds spent in each state. `DETERMINISTIC`: derived from event
    #: timestamps, which is genuinely what it is — not the agent's own timing.
    time_in_state: dict[str, float] = field(default_factory=dict)
    action_counts: dict[str, int] = field(default_factory=dict)
    effect_counts: dict[str, int] = field(default_factory=dict)
    spans: list[SpanRecord] = field(default_factory=list)

    usage: dict[str, Measured] = field(default_factory=dict)
    native_usage_keys: list[str] = field(default_factory=list)
    cost_usd: Measured = field(default_factory=lambda: missing("no cost reported"))
    context_window: Measured = field(
        default_factory=lambda: missing("no context window reported")
    )

    n_events: int = 0
    n_turns: int = 0
    n_files_changed: int = 0
    files_changed: list[str] = field(default_factory=list)
    #: per-path edit accounting: `{edits, lines_added, lines_removed,
    #: total_lines}`. `lines_*` are absent for agents whose file-change events
    #: carry no extent — which is why `edit_churn` can refuse rather than
    #: report a zero it did not measure.
    file_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: human notes, in arrival order
    annotations: list[dict[str, Any]] = field(default_factory=list)
    quality: DataQuality = field(default_factory=DataQuality)

    # ── derived ──────────────────────────────────────────────────────────

    @property
    def verified(self) -> bool:
        """Did anything in this run check the work? Not 'did it pass'."""
        return self.action_counts.get(Action.VERIFY.value, 0) > 0

    def verification_after_last_edit(self) -> Measured:
        """Whether a VERIFY span closed after the last EDIT span.

        The interesting failure it catches: a run that tested early, edited
        again, and stopped. `verified == True` would call that covered.
        """
        last_edit = max(
            (s.started_at for s in self.spans if s.action is Action.EDIT), default=None
        )
        if last_edit is None:
            return Measured(None, Fidelity.MISSING, "no edits in this run")
        after = any(
            s.action is Action.VERIFY and s.started_at >= last_edit for s in self.spans
        )
        return Measured(int(after), Fidelity.DETERMINISTIC)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent": self.agent,
            "agent_version": self.agent_version,
            "model": self.model,
            "repo": self.repo,
            "state": self.state.value,
            "overlays": [o.value for o in self.overlays],
            "outcome": self.outcome.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "last_event_at": self.last_event_at,
            "time_in_state": self.time_in_state,
            "action_counts": self.action_counts,
            "effect_counts": self.effect_counts,
            "spans": [s.to_dict() for s in self.spans],
            "usage": {k: m.to_dict() for k, m in self.usage.items()},
            "native_usage_keys": self.native_usage_keys,
            "cost_usd": self.cost_usd.to_dict(),
            "context_window": self.context_window.to_dict(),
            "n_events": self.n_events,
            "n_turns": self.n_turns,
            "n_files_changed": self.n_files_changed,
            "files_changed": self.files_changed,
            "file_stats": self.file_stats,
            "annotations": self.annotations,
            "verified": self.verified,
            "verification_after_last_edit": self.verification_after_last_edit().to_dict(),
            "quality": self.quality.to_dict(),
        }


class Reducer:
    """Fold an event stream into a `RunView`. Incremental: `push` one event at
    a time for live sessions, `reduce` a whole log for replay. Both take the
    same path, so a replayed run and a live run cannot disagree."""

    def __init__(self, run_id: str) -> None:
        self.view = RunView(run_id=run_id)
        self._open: dict[str, SpanRecord] = {}
        self._tool_names: set[str] = set()
        self._state_since: float | None = None

    # ── the fold ─────────────────────────────────────────────────────────

    def push(self, e: Event) -> RunView:
        v = self.view
        v.n_events += 1
        v.last_event_at = e.ts
        if v.started_at is None:
            v.started_at = e.ts
            # Start the clock on the first event, not on the first *transition*
            # — otherwise everything before the first state change is silently
            # uncounted and the time decomposition does not add up to the run.
            self._state_since = e.ts
        if e.source.agent != "unknown":
            v.agent = e.source.agent
            v.agent_version = e.source.agent_version
        v.quality.capture_mode = e.source.capture_mode.value
        if e.model:
            v.model = e.model
        if e.repo:
            v.repo = e.repo

        # Rule 2, enforced at the top of the fold rather than in each branch:
        # a delta may never reach a counter, a span, or a usage total.
        if e.event_type.is_delta:
            return v

        if e.source.fidelity is Fidelity.DROPPED_BY_POLICY:
            key = e.native_type or e.event_type.value
            v.quality.dropped_by_policy[key] = (
                v.quality.dropped_by_policy.get(key, 0) + 1
            )

        self._count(e)
        self._spans(e)
        self._usage(e)
        self._quality(e)
        self._state(e)
        return v

    def reduce(self, events: Iterable[Event]) -> RunView:
        for e in events:
            self.push(e)
        return self.view

    # ── pieces ───────────────────────────────────────────────────────────

    def _count(self, e: Event) -> None:
        v = self.view
        if e.action is not None:
            # Count on the closing event only. Counting TOOL_STARTED *and*
            # TOOL_COMPLETED would double every tool call, which is rule 2 in a
            # different costume.
            if e.event_type in (
                EventType.TOOL_COMPLETED,
                EventType.TOOL_FAILED,
                EventType.MESSAGE_ASSISTANT_COMPLETED,
                EventType.APPROVAL_RESOLVED,
                EventType.SUBAGENT_COMPLETED,
            ):
                v.action_counts[e.action.value] = (
                    v.action_counts.get(e.action.value, 0) + 1
                )
        if e.effect is not None and not e.event_type.is_delta:
            v.effect_counts[e.effect.value] = v.effect_counts.get(e.effect.value, 0) + 1
        if e.event_type is EventType.TURN_COMPLETED:
            v.n_turns += 1
        if e.event_type is EventType.FILE_CHANGED:
            path = e.payload.get("path")
            if path and path not in v.files_changed:
                v.files_changed.append(path)
            v.n_files_changed += 1
            if path:
                self._file_stat(str(path), e.payload)
        if e.event_type is EventType.ANNOTATION_ADDED:
            v.annotations.append(
                {
                    "event_id": e.event_id,
                    "ts": e.ts,
                    "text": e.payload.get("text", ""),
                    "tags": e.payload.get("tags") or [],
                    "span_id": e.span_id,
                    "author": e.payload.get("author"),
                }
            )

    def _file_stat(self, path: str, p: dict[str, Any]) -> None:
        st = self.view.file_stats.setdefault(
            path, {"edits": 0, "lines_added": 0, "lines_removed": 0,
                   "total_lines": None, "total_lines_from": None,
                   "line_data": False}
        )
        st["edits"] += 1
        added, removed = p.get("lines_added"), p.get("lines_removed")
        if added is None and removed is None:
            # This agent told us the file changed but not by how much. Leave
            # `line_data` false so churn reports a gap instead of a ratio built
            # from the edits that happened to carry numbers.
            return
        st["line_data"] = True
        st["lines_added"] += int(added or 0)
        st["lines_removed"] += int(removed or 0)
        if p.get("total_lines") is not None:
            # Only a whole-file write knows the file's length. The last one wins:
            # a later Edit changes the length by an amount we do know, so the
            # running total stays meaningful — but it stops being exact, and
            # `total_lines_from` is what makes `edit_churn` mark it estimated.
            st["total_lines"] = int(p["total_lines"])
            st["total_lines_from"] = "write"
        elif st["total_lines"] is not None:
            st["total_lines"] += int(added or 0) - int(removed or 0)
            st["total_lines_from"] = "accumulated"

    def _spans(self, e: Event) -> None:
        if e.span_id is None:
            return
        if e.event_type in (EventType.TOOL_STARTED, EventType.MODEL_REQUEST_STARTED):
            self._open[e.span_id] = SpanRecord(
                span_id=e.span_id,
                action=e.action,
                native_type=e.native_type,
                started_at=e.ts,
                detail=e.payload.get("command") or e.payload.get("tool")
                or e.payload.get("path"),
                parent_span_id=e.parent_span_id,
            )
            return
        if e.event_type in (
            EventType.TOOL_COMPLETED,
            EventType.TOOL_FAILED,
            EventType.MODEL_REQUEST_COMPLETED,
        ):
            span = self._open.pop(e.span_id, None)
            if span is None:
                # A completion with no start: the capture began mid-flight, or
                # the mode never emits starts. Record it with a zero-length
                # span rather than dropping the call entirely — a missing tool
                # call is a worse lie than a tool call with unknown duration.
                span = SpanRecord(
                    span_id=e.span_id,
                    action=e.action,
                    native_type=e.native_type,
                    started_at=e.ts,
                    detail=e.payload.get("command") or e.payload.get("tool"),
                    parent_span_id=e.parent_span_id,
                    synthetic_start=True,
                )
            span.ended_at = e.ts
            span.effect = e.effect
            span.failed = e.event_type is EventType.TOOL_FAILED
            if span.action is None:
                span.action = e.action
            ms = e.payload.get("duration_ms")
            if isinstance(ms, (int, float)):
                span.native_duration_s = ms / 1000.0
            self.view.spans.append(span)

    def _usage(self, e: Event) -> None:
        v = self.view
        p = e.payload

        # Cost and context window arrive on the terminal line for Claude and on
        # the usage line for Hermes. Reading only one of those places is how a
        # run that reported its cost ends up printing `—` for it.
        if e.event_type in (
            EventType.SESSION_COMPLETED,
            EventType.SESSION_FAILED,
            EventType.RUN_COMPLETED,
        ):
            if p.get("cost_usd") is not None and v.cost_usd.absent:
                v.cost_usd = Measured(p["cost_usd"], e.source.fidelity)
            if p.get("context_window") is not None:
                v.context_window = Measured(p["context_window"], e.source.fidelity)

        if e.event_type is not EventType.MODEL_USAGE_UPDATED:
            return

        for key in p.get("native_categories") or []:
            if key not in v.native_usage_keys:
                v.native_usage_keys.append(key)

        if p.get("cost_usd") is not None:
            v.cost_usd = Measured(
                p["cost_usd"],
                Fidelity(p.get("cost_fidelity", Fidelity.NATIVE.value)),
                p.get("cost_status"),
            )

        usage = p.get("usage")
        if usage is None:
            # The adapter looked and found nothing. Keep the reason.
            if e.source.fidelity is Fidelity.MISSING:
                reason = p.get("reason") or "agent reported no usage"
                for cat in TokenCategory:
                    v.usage.setdefault(cat.value, missing(reason))
            return

        authoritative = bool(p.get("authoritative"))
        counted = p.get("counted")
        if not authoritative and counted is False:
            # A repeat sighting the adapter already folded. Recording it here
            # would undo the fold one layer up.
            v.quality.folded_duplicates += 1
            return
        if not authoritative and counted is None and e.source.fidelity is not Fidelity.NATIVE:
            return

        for name, n in usage.items():
            if authoritative or name not in v.usage or v.usage[name].absent:
                v.usage[name] = Measured(n, e.source.fidelity)
            else:
                prev = v.usage[name].value or 0
                v.usage[name] = Measured(prev + n, e.source.fidelity)

        # Categories this agent has no bucket for stay explicitly absent. This
        # is the line that keeps a Codex run from showing `cache_write: 0`.
        for cat in TokenCategory:
            if cat.value in v.usage:
                continue
            note = p.get(f"{cat.value}_fidelity")
            if note == Fidelity.MISSING.value or (
                cat is not TokenCategory.TOTAL and p.get(cat.value, "sentinel") is None
            ):
                v.usage[cat.value] = missing(f"{v.agent} reports no {cat.value} bucket")

    def _quality(self, e: Event) -> None:
        q = self.view.quality
        if e.event_type is EventType.ADAPTER_WARNING:
            # `note` since M5; `message` for runs captured before the split.
            msg = str(e.payload.get("note") or e.payload.get("message") or "")
            if msg not in q.warnings:
                q.warnings.append(msg)
            if "unmapped native event kind" in msg and msg not in q.unmapped_native:
                q.unmapped_native.append(msg)
        for gap in e.payload.get("capture_gaps") or []:
            if gap not in q.capture_gaps:
                q.capture_gaps.append(gap)
        name = e.payload.get("tool") or e.native_type
        if name and e.event_type in (EventType.TOOL_STARTED, EventType.TOOL_COMPLETED):
            self._tool_names.add(name)

    def _state(self, e: Event) -> None:
        v = self.view
        nxt = _TRANSITIONS.get(e.event_type)
        if e.event_type is EventType.SESSION_STATE_CHANGED:
            # A whitelist of one, not a parse. `SESSION_STATE_CHANGED` also
            # carries each agent's own status vocabulary — Codex's
            # `thread/status/changed` puts strings like "running" in here — and
            # letting those drive the state machine would be an agent's words
            # deciding our states, which is the one thing the contract forbids.
            # `detached` is ours: no event family produces it, and both the
            # adapter and the transport set it deliberately when we stop
            # looking. Without this the run reads `starting` forever, which in
            # the run list is indistinguishable from one that is still live.
            if e.payload.get("state") == SessionState.DETACHED.value:
                nxt = SessionState.DETACHED
        if e.event_type in (
            EventType.TOOL_COMPLETED,
            EventType.TOOL_FAILED,
        ):
            # Back to the model, unless another tool is still open — parallel
            # tool calls are normal and the state must not flap to
            # `model_running` while three of them are still running.
            nxt = SessionState.TOOL_RUNNING if self._open else SessionState.MODEL_RUNNING
        if nxt is None:
            return
        if v.state is not nxt:
            if self._state_since is not None:
                v.time_in_state[v.state.value] = (
                    v.time_in_state.get(v.state.value, 0.0) + (e.ts - self._state_since)
                )
            v.state = nxt
            self._state_since = e.ts
        if nxt in (SessionState.COMPLETED, SessionState.FAILED, SessionState.INTERRUPTED):
            v.ended_at = e.ts
            claimed = e.payload.get("outcome")
            if claimed:
                v.outcome = Outcome(claimed)
            # Stop the clock. Bookkeeping events arrive after the run ends
            # (RUN_COMPLETED, a late reconciliation), and letting them
            # accumulate would put seconds against `completed` — time the run
            # did not spend doing anything.
            self._state_since = None

    # ── overlays ─────────────────────────────────────────────────────────

    def overlays(self, now: float | None = None) -> list[StateOverlay]:
        """Overlays for the current state. Computed on read, not stored: a stall
        is a fact about *now*, and freezing one into the log would make a
        finished run permanently look stuck."""
        v = self.view
        if v.state in (
            SessionState.COMPLETED,
            SessionState.FAILED,
            SessionState.INTERRUPTED,
        ):
            return []
        if v.last_event_at is None:
            return []
        quiet = (now if now is not None else time.time()) - v.last_event_at
        budget = STALL_BUDGET_S.get(v.state)
        if budget is None or quiet <= budget:
            return []
        return [
            StateOverlay.OVERDUE if v.state in _HUMAN_WAITS else StateOverlay.STALLED
        ]

    def finalize(self, now: float | None = None) -> RunView:
        v = self.view
        v.overlays = self.overlays(now)
        v.quality.unmatched_tools = unmatched_tools(sorted(self._tool_names))
        v.quality.absent_token_categories = sorted(
            k for k, m in v.usage.items() if m.absent
        )
        # An unclosed span at the end is not a completed call. Leaving it out of
        # `spans` would drop the very tool a hung run is hung on.
        for span in self._open.values():
            v.spans.append(span)
        self._open.clear()
        if self._state_since is not None and v.last_event_at is not None:
            v.time_in_state[v.state.value] = v.time_in_state.get(v.state.value, 0.0) + (
                v.last_event_at - self._state_since
            )
            self._state_since = v.last_event_at
        return v


def reduce_run(run_id: str, events: Iterable[Event], now: float | None = None) -> RunView:
    r = Reducer(run_id)
    r.reduce(events)
    return r.finalize(now)
