"""The canonical event contract — the artifact that stops three dashboards.

Codex, Claude Code and Hermes each emit a different event vocabulary (68 JSON-RPC
notifications / 19 hook names / a `tool.*`+`message.*` gateway protocol). If each
adapter grew its own fields, SessionSeer would be three viewers sharing a nav
bar. Everything downstream of an adapter therefore speaks ONLY the types in this
module, and the M0 exit test asserts no agent-specific key survives the boundary.

Three rules encode the lessons that cost the most to learn:

1. **Provenance is mandatory.** Every value carries a `Fidelity`. `MISSING` is
   never rendered as `0`, and `DROPPED_BY_POLICY` is distinct from `MISSING` —
   Codex *does* stream `item/reasoning/textDelta` and Hermes *does* emit
   `reasoning.delta`, so not storing reasoning is an active decision the ingress
   makes, not an absence we can passively claim (docs/SESSIONSEER.md §2.4).

2. **Deltas never feed counters.** Claude Code writes one model response as
   several JSONL lines repeating identical `usage`; counting per-line overcounts
   3.5× on a real session (viewer/src/chrome/sessionlog.ts). Codex has the same
   hazard (`item/*/delta` vs `item/completed`) and so does Hermes
   (`message.delta` vs `message.complete`). `EventType.is_delta` marks the
   streaming variants and `fold_key()` names the event that closes a span.

3. **An action is not a loop; an action with no effect is.** The taxonomy is 9
   actions × an `Effect` label, following the nine-type-plus-effect scheme of
   arXiv 2607.06184 rather than a flat 18-category list. Without the effect
   label, loop detection cannot be written at all.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .redaction import RULESET, event_level, scrub_payload

SCHEMA_VERSION = "1.0"


# ── provenance ───────────────────────────────────────────────────────────────


class Fidelity(str, Enum):
    """How a value came to be known. Rendered on every metric, no exceptions."""

    NATIVE = "native"  # the agent reported it
    DETERMINISTIC = "deterministic"  # computed from native events / git / procs
    ESTIMATED = "estimated"  # computed because the native value was absent
    HEURISTIC = "heuristic"  # an interpretation (loop, plan adherence, …)
    MISSING = "missing"  # unavailable — NEVER silently 0
    DROPPED_BY_POLICY = "dropped_by_policy"  # available, refused at ingress


class CaptureMode(str, Enum):
    """How SessionSeer is attached to the agent, best fidelity first.

    Named for what is actually true of each, not for the report's original
    managed/observed/reconciled triple: `DRIVEN` and `ATTACHED` are both
    "managed" but differ in whether we own the process, which decides whether
    stdout is a reliable stream or an endpoint we can lose.
    """

    DRIVEN = "driven"  # we launched it headless and own stdout
    ATTACHED = "attached"  # we connected to a running app-server / gateway
    OBSERVED = "observed"  # hooks append to the spool, we tail it
    RECONCILED = "reconciled"  # state.db / transcripts / git, after the fact


# ── taxonomy ─────────────────────────────────────────────────────────────────


class Action(str, Enum):
    """The 9-type normalized action. The NATIVE tool name is always kept beside
    it — this is for cross-agent grouping, never a replacement for the truth.

    `VERIFY` deliberately absorbs test/build/lint/typecheck: the research
    question is "did verification happen after the last edit", and splitting it
    four ways only creates four sparse cells.
    """

    INSPECT = "inspect"  # read a known target
    SEARCH = "search"  # find an unknown target
    EDIT = "edit"  # mutate a file
    EXECUTE = "execute"  # run a command with no verification intent
    VERIFY = "verify"  # test / build / lint / typecheck / evaluator
    VCS = "vcs"  # git and friends
    DELEGATE = "delegate"  # spawn or steer a subagent
    INTERACT = "interact"  # ask the human; approvals, clarifications
    REPORT = "report"  # final answer, summary, artifact delivery


ACTIONS: tuple[Action, ...] = tuple(Action)


class Effect(str, Enum):
    """What the action actually changed. The half that makes loops detectable.

    `NO_NEW_INFORMATION` is the load-bearing one: four searches in a row is not
    a loop, four searches that surface no path the run had not already seen is.
    """

    NEW_INFORMATION = "new_information"
    NO_NEW_INFORMATION = "no_new_information"
    STATE_CHANGED = "state_changed"
    NO_STATE_CHANGE = "no_state_change"
    FAILED = "failed"
    UNKNOWN = "unknown"


# ── token accounting ─────────────────────────────────────────────────────────


class TokenCategory(str, Enum):
    """Native token buckets, kept native on purpose.

    Claude's `cache_read_input_tokens`, Codex's `thread/tokenUsage` `cached`,
    and Hermes's `cache_read_tokens` are NOT interchangeable — they differ in
    what counts as a cache hit and whether reasoning is billed inside output.
    The comparability gate (compare.py) reads these labels and refuses to
    subtract across agents when they don't align. Collapsing them into one
    `tokens` field would produce a headline number that is wrong and looks
    authoritative.
    """

    INPUT = "input"
    OUTPUT = "output"
    CACHE_READ = "cache_read"
    CACHE_WRITE = "cache_write"
    REASONING = "reasoning"
    TOTAL = "total"


# ── lifecycle ────────────────────────────────────────────────────────────────


class SessionState(str, Enum):
    """User-visible state produced by the reducer.

    `stalled` and `overdue` are deliberately NOT members: they are overlays
    (see `StateOverlay`). Making "stalled" a state loses the information about
    what the session is stalled *doing*, which is the only actionable part.
    """

    STARTING = "starting"
    IDLE = "idle"
    MODEL_RUNNING = "model_running"
    TOOL_RUNNING = "tool_running"
    WAITING_PERMISSION = "waiting_permission"
    WAITING_CLARIFICATION = "waiting_clarification"
    WAITING_USER = "waiting_user"
    COMPACTING = "compacting"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"
    DETACHED = "detached"


class StateOverlay(str, Enum):
    """Modifiers layered on a `SessionState`, never replacing it."""

    STALLED = "stalled"  # no event for longer than the state's budget
    OVERDUE = "overdue"  # a human decision has been pending too long


TERMINAL_STATES = frozenset(
    {SessionState.COMPLETED, SessionState.FAILED, SessionState.INTERRUPTED}
)


class Outcome(str, Enum):
    """Run outcome. `AGENT_CLAIMED_COMPLETE` and `VERIFIED_PASS` must never
    collapse into one another — an agent saying "done" is evidence about the
    agent, not about the task. Same rule as a namer with `n_labeled == 0`
    reporting that it labeled nothing rather than printing a confident zero.
    """

    UNKNOWN = "unknown"
    AGENT_CLAIMED_COMPLETE = "agent_claimed_complete"
    UNVERIFIED_COMPLETE = "unverified_complete"
    VERIFIED_PASS = "verified_pass"
    VERIFIED_PARTIAL = "verified_partial"
    VERIFIED_FAIL = "verified_fail"
    INTERRUPTED = "interrupted"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


# ── event families ───────────────────────────────────────────────────────────


class EventType(str, Enum):
    """Canonical event vocabulary. Every adapter maps into exactly these.

    Members whose name ends in `_DELTA` are streaming fragments: they may update
    a live preview but must never increment a counter or contribute usage. See
    `is_delta` / `fold_key`.
    """

    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"

    SESSION_STARTED = "session.started"
    SESSION_RESUMED = "session.resumed"
    SESSION_BRANCHED = "session.branched"
    SESSION_STATE_CHANGED = "session.state_changed"
    SESSION_COMPLETED = "session.completed"
    SESSION_FAILED = "session.failed"
    SESSION_INTERRUPTED = "session.interrupted"

    TURN_STARTED = "turn.started"
    TURN_COMPLETED = "turn.completed"
    TURN_FAILED = "turn.failed"

    MESSAGE_USER = "message.user"
    MESSAGE_ASSISTANT_DELTA = "message.assistant_delta"
    MESSAGE_ASSISTANT_COMPLETED = "message.assistant_completed"
    PLAN_UPDATED = "plan.updated"

    MODEL_REQUEST_STARTED = "model.request_started"
    MODEL_FIRST_TOKEN = "model.first_token"
    MODEL_REQUEST_COMPLETED = "model.request_completed"
    MODEL_REQUEST_FAILED = "model.request_failed"
    MODEL_USAGE_UPDATED = "model.usage_updated"
    MODEL_REROUTED = "model.rerouted"

    TOOL_STARTED = "tool.started"
    TOOL_OUTPUT_DELTA = "tool.output_delta"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"

    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_RESOLVED = "approval.resolved"
    CLARIFICATION_REQUESTED = "clarification.requested"
    CLARIFICATION_RESOLVED = "clarification.resolved"

    FILE_CHANGED = "file.changed"
    GIT_SNAPSHOT = "git.snapshot"

    SUBAGENT_STARTED = "subagent.started"
    SUBAGENT_COMPLETED = "subagent.completed"

    CONTEXT_PRESSURE_UPDATED = "context.pressure_updated"
    COMPACTION_STARTED = "context.compaction_started"
    COMPACTION_COMPLETED = "context.compaction_completed"

    EVALUATION_COMPLETED = "evaluation.completed"

    #: The agent reported a problem of its own. Distinct from `TURN_FAILED` (the
    #: turn died) and from `ADAPTER_WARNING` (SessionSeer's problem): Codex emits
    #: `item.completed` with `type: "error"` on a turn that then completes
    #: normally, and folding that into either neighbour would either invent a
    #: failed turn or blame the adapter for the agent's notice.
    AGENT_ERROR = "agent.error"
    #: Rate-limit / quota telemetry. Worth its own type because a session sitting
    #: still because it is throttled looks identical to one that is stuck, and
    #: those need opposite responses from whoever is watching.
    QUOTA_UPDATED = "quota.updated"

    #: A human's note about this run, appended to the same log as everything
    #: else. It lives here rather than in a side table because the log is the
    #: record: an annotation written during a run and the events it refers to
    #: have to survive export, replay and deletion together, and a note stored
    #: elsewhere would quietly outlive the run it annotates.
    ANNOTATION_ADDED = "annotation.added"

    ADAPTER_WARNING = "adapter.warning"
    ADAPTER_INCOMPATIBLE = "adapter.incompatible"

    @property
    def is_delta(self) -> bool:
        """True for streaming fragments that must not feed counters."""
        return self.value.endswith("_delta")


#: Which event closes each streaming family. An adapter that receives a delta
#: accumulates a preview; only the fold event contributes counts and usage.
#: This is rule 2 in the module docstring, made mechanical.
_FOLD: dict[EventType, EventType] = {
    EventType.MESSAGE_ASSISTANT_DELTA: EventType.MESSAGE_ASSISTANT_COMPLETED,
    EventType.TOOL_OUTPUT_DELTA: EventType.TOOL_COMPLETED,
}


def fold_key(et: EventType) -> EventType:
    """The event that closes `et`'s span. Identity for non-delta events."""
    return _FOLD.get(et, et)


# ── the envelope ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Source:
    """Where an event came from, and how much to trust it."""

    agent: str  # "codex" | "claude" | "hermes"
    agent_version: str
    adapter: str  # e.g. "codex_exec_json"
    adapter_version: str
    capture_mode: CaptureMode
    fidelity: Fidelity
    source_event_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "agent": self.agent,
            "agent_version": self.agent_version,
            "adapter": self.adapter,
            "adapter_version": self.adapter_version,
            "capture_mode": self.capture_mode.value,
            "fidelity": self.fidelity.value,
        }
        if self.source_event_id is not None:
            d["source_event_id"] = self.source_event_id
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Source:
        return Source(
            agent=d["agent"],
            agent_version=d.get("agent_version", "unknown"),
            adapter=d.get("adapter", "unknown"),
            adapter_version=d.get("adapter_version", "0"),
            capture_mode=CaptureMode(d.get("capture_mode", "driven")),
            fidelity=Fidelity(d.get("fidelity", "native")),
            source_event_id=d.get("source_event_id"),
        )


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass(slots=True)
class Event:
    """One canonical event. This is the ONLY thing that crosses an adapter
    boundary, and the append-only JSONL log is a sequence of these.

    `native_type` and `native` keep the agent's own words so nothing is lost to
    normalization; the analysis layer is forbidden from reading them, which is
    what the M0 exit test checks.
    """

    event_type: EventType
    source: Source
    run_id: str
    session_id: str

    event_id: str = field(default_factory=lambda: _new_id("evt"))
    ts: float = field(default_factory=time.time)  # epoch seconds, wall clock
    mono_ns: int = field(default_factory=time.monotonic_ns)

    turn_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None

    action: Action | None = None
    effect: Effect | None = None
    native_type: str | None = None

    repo: dict[str, Any] | None = None
    model: dict[str, Any] | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    native: dict[str, Any] | None = None
    #: Filled in by `__post_init__` from what the payload actually contains.
    #: Pass one explicitly only to override that reading — the annotation route
    #: does, because text a person typed into SessionSeer is not the agent's.
    privacy: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Scrub credentials, then label the event by what is left in it.

        Both halves are here rather than in `BaseAdapter.event()` because this
        is the only constructor every producer goes through — adapters, the
        runner, the attach transport, the reconciler, the server's own
        annotations. A scrub that can be bypassed by building an `Event`
        directly is not a scrub, and a `content_level` that some producers set
        and others forget is the constant label M5 replaced.
        """
        hits = scrub_payload(self.payload)
        if hits:
            self.privacy["scrubbed"] = hits
            self.privacy["ruleset"] = RULESET
        self.privacy.setdefault(
            "content_level",
            event_level(self.payload, has_native=self.native is not None).value,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "event_id": self.event_id,
            "ts": self.ts,
            "mono_ns": self.mono_ns,
            "source": self.source.to_dict(),
            "run_id": self.run_id,
            "session_id": self.session_id,
            "event_type": self.event_type.value,
            "payload": self.payload,
            "privacy": self.privacy,
        }
        for k, v in (
            ("turn_id", self.turn_id),
            ("span_id", self.span_id),
            ("parent_span_id", self.parent_span_id),
            ("native_type", self.native_type),
            ("repo", self.repo),
            ("model", self.model),
            ("native", self.native),
        ):
            if v is not None:
                d[k] = v
        if self.action is not None:
            d["action"] = self.action.value
        if self.effect is not None:
            d["effect"] = self.effect.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), default=str)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Event:
        return Event(
            event_type=EventType(d["event_type"]),
            source=Source.from_dict(d["source"]),
            run_id=d["run_id"],
            session_id=d["session_id"],
            event_id=d.get("event_id", _new_id("evt")),
            ts=float(d.get("ts", 0.0)),
            mono_ns=int(d.get("mono_ns", 0)),
            turn_id=d.get("turn_id"),
            span_id=d.get("span_id"),
            parent_span_id=d.get("parent_span_id"),
            action=Action(d["action"]) if d.get("action") else None,
            effect=Effect(d["effect"]) if d.get("effect") else None,
            native_type=d.get("native_type"),
            repo=d.get("repo"),
            model=d.get("model"),
            payload=d.get("payload", {}),
            native=d.get("native"),
            # Whatever the log says, not what this build would decide today:
            # a run captured under an older ruleset is described by that
            # ruleset, and re-labelling it on read would erase the difference.
            privacy=dict(d.get("privacy") or {}),
        )


#: Keys the analysis layer is allowed to read off an event. `native` and
#: `native_type` are deliberately absent: they exist for display and audit, and
#: any metric that reads them has re-introduced an agent-specific code path.
ANALYSIS_KEYS = frozenset(
    {
        "event_type", "ts", "mono_ns", "run_id", "session_id", "turn_id",
        "span_id", "parent_span_id", "action", "effect", "payload", "source",
        "repo", "model",
    }
)


def new_run_id() -> str:
    return _new_id("run")


def new_session_id() -> str:
    return _new_id("ses")


def new_span_id() -> str:
    return _new_id("span")


def new_turn_id() -> str:
    return _new_id("turn")
