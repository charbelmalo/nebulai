"""Observed capture: agent hook events, folded into the same vocabulary.

The driven adapters read an agent's own structured output. This one reads what
an agent's *hooks* report about a session a human is driving — so it sees the
session nobody launched for us, at the cost of never seeing a token.

That cost is the interesting part, and it is declared rather than discovered.
`CAPTURE_GAPS` names what hooks cannot report for each agent, the reducer copies
it into the data-quality panel, and the comparability gate refuses any metric it
blocks. An observed run and a driven run of the same agent are therefore *not*
silently comparable on tokens or model timing; they are comparable on actions,
files, verification, approvals and wall clock, which is most of what a
trajectory question actually asks.

What hooks are unusually good at, and driven capture is not: they see the human.
`UserPromptSubmit` between two turns is an intervention; a `PermissionRequest`
that sits before a `PermissionDenied` is a refusal with a wait attached. Those
are first-class here.

Timestamp fidelity comes from the shim's clock (`spool.Clock`). With a
whole-second clock every tool span inside one second collapses to zero, so
durations built from one are `estimated` and say so, rather than reporting a
confident `0.0s` for every edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contract import (
    Action,
    CaptureMode,
    Effect,
    Event,
    EventType,
    Fidelity,
    Outcome,
    new_span_id,
    new_turn_id,
)
from ..spool import SpoolLine
from ..taxonomy import classify_command, classify_tool, edit_extent
from .base import BaseAdapter

#: What each agent's hook surface cannot report, in the words the data-quality
#: panel and the comparability gate both read. Wrong entries here are worse than
#: missing ones: they would refuse comparisons that are perfectly valid.
CAPTURE_GAPS: dict[str, tuple[str, ...]] = {
    "claude": (
        "token usage (hooks carry no usage record)",
        "per-request model timing",
        "context-window pressure",
        "assistant message content",
    ),
    "codex": (
        "token usage (hooks carry no usage record)",
        "per-request model timing",
        "context-window pressure",
        "assistant message content",
    ),
    "hermes": (
        "token usage (hooks carry no usage record)",
        "context-window pressure",
        "assistant message content",
    ),
}

_EDIT_TOOLS = {
    "Write": "file_path",
    "Edit": "file_path",
    "MultiEdit": "file_path",
    "NotebookEdit": "notebook_path",
}

#: `SessionEnd.reason` → how the session ended. Anything unlisted is a normal
#: end: only the reasons that mean *interrupted* are enumerated, because
#: guessing the other way turns an ordinary quit into a reported failure.
_END_INTERRUPTED = frozenset({"prompt_input_exit", "logout", "sigint", "cancel"})


@dataclass
class _OpenTool:
    """A tool call we have seen start and not yet seen finish."""

    span_id: str
    action: Action
    #: paired on an id the agent gave us, rather than on the tool's name
    keyed_by_id: bool
    path: str | None = None
    #: lines the call said it would touch, from the Pre hook's `tool_input`.
    #: Carried across to the Post hook because that is where we learn the edit
    #: actually landed.
    extent: dict[str, int] | None = None


class HookAdapter(BaseAdapter):
    """Base for the three hook adapters. Consumes `SpoolLine`, not text.

    One instance per (agent, session): the collector routes on the session id
    the hook payload carries, so two agents running side by side never fold into
    one trajectory.
    """

    agent = "unknown"
    adapter_name = "hooks"
    #: hook event name → method
    ROUTES: dict[str, str] = {}

    def __init__(self, *, clock_resolution_s: float = float("inf"), **kw: Any) -> None:
        kw.setdefault("capture_mode", CaptureMode.OBSERVED)
        super().__init__(**kw)
        self.clock_resolution_s = clock_resolution_s
        self._started = False
        # FIFO per key: an agent that runs three Bash calls at once gives us
        # three Pres before any Post, and a plain dict would keep the last and
        # lose two whole tool calls.
        self._open_tools: dict[str, list[_OpenTool]] = {}
        self._open_approvals: dict[str, str] = {}
        self._turn_open = False
        self._ended = False

    # ── timestamps ───────────────────────────────────────────────────────

    @property
    def ended(self) -> bool:
        """Whether a terminal event has been emitted for this session."""
        return self._ended

    @property
    def timing_fidelity(self) -> Fidelity:
        """Whether a duration derived from these timestamps can be trusted.

        A one-second clock cannot resolve a 60 ms tool call. Reporting that as
        `0.0s` deterministic would be a lie with a decimal point on it.
        """
        return (
            Fidelity.DETERMINISTIC if self.clock_resolution_s < 0.05 else Fidelity.ESTIMATED
        )

    def feed_hook(self, line: SpoolLine) -> list[Event]:
        """One spool line in, canonical events out."""
        events: list[Event] = []
        if not self._started:
            events.extend(self._open_session(line))
        name = self.ROUTES.get(line.event)
        if name is None:
            return events + self.note_unknown_native(f"hook.{line.event}")
        events.extend(getattr(self, name)(line))
        if line.oversized is not None:
            # the shim measured it and refused to carry it; that is a policy
            # drop, not a gap in what the agent told us
            events.append(
                self.event(
                    EventType.ADAPTER_WARNING,
                    fidelity=Fidelity.DROPPED_BY_POLICY,
                    native_type=f"hook.{line.event}",
                    ts=line.ts,
                    payload={
                        "note": "hook payload exceeded the shim's size cap",
                        "bytes": line.oversized,
                    },
                )
            )
        return events

    # ── shared construction ──────────────────────────────────────────────

    def hook_event(self, et: EventType, line: SpoolLine, **kw: Any) -> Event:
        kw.setdefault("native_type", f"hook.{line.event}")
        kw.setdefault("ts", line.ts)
        kw.setdefault("fidelity", self.timing_fidelity)
        return self.event(et, **kw)

    def _open_session(self, line: SpoolLine) -> list[Event]:
        """Emit SESSION_STARTED on whatever hook happens to arrive first.

        Hooks are not guaranteed to start with `SessionStart` — a shim installed
        mid-session sees a `PreToolUse` first. Refusing to open a run there would
        drop the rest of the session on the floor, so the run opens and says the
        start time is estimated.
        """
        self._started = True
        first = line.event in ("SessionStart", "on_session_start", "session/started")
        return [
            self.hook_event(
                EventType.SESSION_STARTED,
                line,
                fidelity=Fidelity.NATIVE if first else Fidelity.ESTIMATED,
                payload={
                    "capture_gaps": list(CAPTURE_GAPS.get(self.agent, ())),
                    # Only when the hook actually carried one. The collector
                    # falls back to the cwd when it does not, and recording
                    # *that* as the agent's session id would make a later
                    # reconciliation skip every session from the directory.
                    "native_session_id": (
                        line.payload.get("session_id")
                        or line.payload.get("sessionId")
                        or line.payload.get("thread_id")
                    ),
                    "joined_midstream": not first,
                    "clock_resolution_s": self.clock_resolution_s,
                    "cwd": line.payload.get("cwd"),
                },
            )
        ]

    def _tool_key(self, line: SpoolLine) -> str:
        """What pairs a Pre with its Post.

        Not the pid: the shim is a fresh process on every hook, so `line.pid`
        identifies a firing and would pair nothing with nothing. The adapter is
        already scoped to one session, so the key only has to be unique *within*
        a run — `tool_use_id` when the agent sends one, tool name otherwise.
        """
        p = line.payload
        tid = p.get("tool_use_id") or p.get("toolUseId") or p.get("call_id") or p.get("id")
        if tid:
            return f"id:{tid}"
        return f"name:{p.get('tool_name') or p.get('tool') or ''}"

    def _start_tool(self, line: SpoolLine, name: str, inp: dict[str, Any]) -> list[Event]:
        key = self._tool_key(line)
        action = (
            classify_command(str(inp.get("command") or ""))
            if name in ("Bash", "BashOutput") and inp.get("command")
            else classify_tool(name)
        )
        open_tool = _OpenTool(
            span_id=new_span_id(),
            action=action,
            keyed_by_id=key.startswith("id:"),
            path=(
                str(inp[_EDIT_TOOLS[name]])
                if name in _EDIT_TOOLS and inp.get(_EDIT_TOOLS[name])
                else None
            ),
            extent=edit_extent(name, inp) if name in _EDIT_TOOLS else None,
        )
        self._open_tools.setdefault(key, []).append(open_tool)
        return [
            self.hook_event(
                EventType.TOOL_STARTED,
                line,
                span_id=open_tool.span_id,
                action=action,
                payload={
                    "tool": name,
                    "command": inp.get("command"),
                    "path": inp.get("file_path") or inp.get("path"),
                },
            )
        ]

    def _finish_tool(self, line: SpoolLine, *, failed: bool) -> list[Event]:
        key = self._tool_key(line)
        queue = self._open_tools.get(key) or []
        # FIFO. With no call id, two concurrent same-name calls can only be
        # paired in the order they started; the *set* of spans is right and the
        # two durations may be swapped between them, so the pairing says so.
        guessed = not key.startswith("id:") and len(queue) > 1
        started = queue.pop(0) if queue else None
        if not queue:
            self._open_tools.pop(key, None)
        span_id = started.span_id if started else None
        action = started.action if started else None
        fidelity = Fidelity.HEURISTIC if guessed else self.timing_fidelity
        out: list[Event] = []
        if started and started.path and not failed:
            out.append(
                self.hook_event(
                    EventType.FILE_CHANGED,
                    line,
                    span_id=span_id,
                    action=Action.EDIT,
                    effect=Effect.STATE_CHANGED,
                    payload={
                        "path": started.path,
                        "kind": "update",
                        **(started.extent or {}),
                    },
                )
            )
        out.append(
            self.hook_event(
                EventType.TOOL_FAILED if failed else EventType.TOOL_COMPLETED,
                line,
                span_id=span_id,
                action=action,
                effect=Effect.FAILED if failed else _effect_for(action),
                fidelity=fidelity,
                payload={
                    "tool": line.payload.get("tool_name"),
                    "is_error": failed,
                    # hooks hand us the whole response; we keep its size, which
                    # is what the churn and progress analyses actually use
                    "output_chars": _response_chars(line.payload),
                    **({"paired_by": "name-order"} if guessed else {}),
                },
            )
        )
        return out

    def _open_turn(self, line: SpoolLine) -> list[Event]:
        if self._turn_open:
            return []
        self._turn_open = True
        self.turn_id = new_turn_id()
        return [self.hook_event(EventType.TURN_STARTED, line)]

    def _close_turn(self, line: SpoolLine, *, failed: bool = False) -> list[Event]:
        if not self._turn_open:
            return []
        self._turn_open = False
        et = EventType.TURN_FAILED if failed else EventType.TURN_COMPLETED
        return [self.hook_event(et, line)]

    def close(self, ts: float, *, reason: str = "collector stopped") -> list[Event]:
        """End a session the hooks never ended.

        A session whose `SessionEnd` never fired — the terminal was closed, the
        machine slept — is `interrupted`, not `completed`. Guessing `completed`
        here would manufacture the one verdict this tool exists to be careful
        about.
        """
        if self._ended:
            return []
        self._ended = True
        return [
            self.event(
                EventType.SESSION_INTERRUPTED,
                ts=ts,
                fidelity=Fidelity.DETERMINISTIC,
                native_type="collector.close",
                payload={"outcome": Outcome.UNKNOWN.value, "reason": reason},
            )
        ]


# ── Claude Code ──────────────────────────────────────────────────────────────


class ClaudeHookAdapter(HookAdapter):
    """Claude Code's 19 hook events.

    The set is rich enough that observed mode differs from driven mode mainly in
    what it cannot see (usage, model timing) rather than in what it gets wrong —
    tool calls, turn boundaries, approvals, subagents, compaction and file
    changes all arrive natively.
    """

    agent = "claude"
    adapter_name = "claude_hooks"
    ROUTES = {
        "SessionStart": "_h_session_start",
        "SessionEnd": "_h_session_end",
        "UserPromptSubmit": "_h_prompt",
        "PreToolUse": "_h_pre_tool",
        "PostToolUse": "_h_post_tool",
        "PostToolUseFailure": "_h_post_tool_failure",
        "Stop": "_h_stop",
        "StopFailure": "_h_stop_failure",
        "SubagentStart": "_h_subagent_start",
        "SubagentStop": "_h_subagent_stop",
        "PreCompact": "_h_pre_compact",
        "PostCompact": "_h_post_compact",
        "Notification": "_h_notification",
        "PermissionRequest": "_h_permission_request",
        "PermissionDenied": "_h_permission_denied",
        "TaskCreated": "_h_task",
        "TaskCompleted": "_h_task",
        "CwdChanged": "_h_cwd",
        "FileChanged": "_h_file_changed",
    }

    def _h_session_start(self, line: SpoolLine) -> list[Event]:
        # `_open_session` already emitted SESSION_STARTED for this very line;
        # a resume is the one case that carries extra meaning
        src = line.payload.get("source")
        if src in ("resume", "compact"):
            return [
                self.hook_event(
                    EventType.SESSION_RESUMED, line, payload={"source": src}
                )
            ]
        return []

    def _h_session_end(self, line: SpoolLine) -> list[Event]:
        self._ended = True
        reason = str(line.payload.get("reason") or "")
        interrupted = reason in _END_INTERRUPTED
        out = self._close_turn(line)
        out.append(
            self.hook_event(
                EventType.SESSION_INTERRUPTED if interrupted else EventType.SESSION_COMPLETED,
                line,
                payload={
                    "reason": reason,
                    # hooks never carry a verdict; claiming one from a clean
                    # exit would be exactly the unearned "completed" the outcome
                    # vocabulary exists to prevent
                    "outcome": Outcome.UNKNOWN.value,
                },
            )
        )
        return out

    def _h_prompt(self, line: SpoolLine) -> list[Event]:
        prompt = str(line.payload.get("prompt") or "")
        out = self._close_turn(line)
        out.extend(self._open_turn(line))
        out.append(
            self.hook_event(
                EventType.MESSAGE_USER,
                line,
                action=Action.INTERACT,
                payload={"chars": len(prompt), "text_retained": False},
            )
        )
        return out

    def _h_pre_tool(self, line: SpoolLine) -> list[Event]:
        name = str(line.payload.get("tool_name") or "")
        out = self._open_turn(line)
        out.extend(self._start_tool(line, name, line.payload.get("tool_input") or {}))
        return out

    def _h_post_tool(self, line: SpoolLine) -> list[Event]:
        # `PostToolUseFailure` is not the only way a call fails: an ordinary
        # `PostToolUse` carries the response, and the response can say it errored.
        # Only an explicit flag counts — plenty of successful commands write to
        # stderr, and reading that as failure would invent failures.
        resp = line.payload.get("tool_response")
        failed = isinstance(resp, dict) and bool(resp.get("is_error") or resp.get("error"))
        return self._finish_tool(line, failed=failed)

    def _h_post_tool_failure(self, line: SpoolLine) -> list[Event]:
        return self._finish_tool(line, failed=True)

    def _h_stop(self, line: SpoolLine) -> list[Event]:
        return self._close_turn(line)

    def _h_stop_failure(self, line: SpoolLine) -> list[Event]:
        return self._close_turn(line, failed=True)

    def _h_subagent_start(self, line: SpoolLine) -> list[Event]:
        return [
            self.hook_event(
                EventType.SUBAGENT_STARTED,
                line,
                action=Action.DELEGATE,
                payload={"agent_type": line.payload.get("subagent_type")},
            )
        ]

    def _h_subagent_stop(self, line: SpoolLine) -> list[Event]:
        return [
            self.hook_event(
                EventType.SUBAGENT_COMPLETED, line, action=Action.DELEGATE
            )
        ]

    def _h_pre_compact(self, line: SpoolLine) -> list[Event]:
        return [
            self.hook_event(
                EventType.COMPACTION_STARTED,
                line,
                payload={"trigger": line.payload.get("trigger")},
            )
        ]

    def _h_post_compact(self, line: SpoolLine) -> list[Event]:
        return [self.hook_event(EventType.COMPACTION_COMPLETED, line)]

    def _h_notification(self, line: SpoolLine) -> list[Event]:
        msg = str(line.payload.get("message") or "")
        return [
            self.hook_event(
                EventType.CLARIFICATION_REQUESTED,
                line,
                action=Action.INTERACT,
                payload={"message": msg},
            )
        ]

    def _h_permission_request(self, line: SpoolLine) -> list[Event]:
        span_id = new_span_id()
        self._open_approvals[self._tool_key(line)] = span_id
        return [
            self.hook_event(
                EventType.APPROVAL_REQUESTED,
                line,
                span_id=span_id,
                action=Action.INTERACT,
                payload={"tool": line.payload.get("tool_name")},
            )
        ]

    def _h_permission_denied(self, line: SpoolLine) -> list[Event]:
        span_id = self._open_approvals.pop(self._tool_key(line), None)
        return [
            self.hook_event(
                EventType.APPROVAL_RESOLVED,
                line,
                span_id=span_id,
                action=Action.INTERACT,
                effect=Effect.NO_STATE_CHANGE,
                payload={"decision": "denied", "tool": line.payload.get("tool_name")},
            )
        ]

    def _h_task(self, line: SpoolLine) -> list[Event]:
        return [
            self.hook_event(
                EventType.PLAN_UPDATED,
                line,
                payload={
                    "change": "created" if line.event == "TaskCreated" else "completed",
                    "task": line.payload.get("subject") or line.payload.get("task"),
                },
            )
        ]

    def _h_cwd(self, line: SpoolLine) -> list[Event]:
        return [
            self.hook_event(
                EventType.SESSION_STATE_CHANGED,
                line,
                payload={"cwd": line.payload.get("cwd")},
            )
        ]

    def _h_file_changed(self, line: SpoolLine) -> list[Event]:
        return [
            self.hook_event(
                EventType.FILE_CHANGED,
                line,
                action=Action.EDIT,
                effect=Effect.STATE_CHANGED,
                payload={
                    "path": line.payload.get("file_path") or line.payload.get("path"),
                    "kind": line.payload.get("change") or "update",
                },
            )
        ]


# ── Codex ────────────────────────────────────────────────────────────────────


class CodexHookAdapter(HookAdapter):
    """Codex hooks. Fewer events than Claude's, and no turn boundary of their
    own — turns are inferred from the first tool call after a user message,
    which is why `n_turns` is `estimated` for an observed Codex run."""

    agent = "codex"
    adapter_name = "codex_hooks"
    ROUTES = {
        "session/started": "_h_session_start",
        "session/ended": "_h_session_end",
        "user/message": "_h_prompt",
        "tool/pre": "_h_pre_tool",
        "tool/post": "_h_post_tool",
        "approval/requested": "_h_approval",
        "approval/resolved": "_h_approval_resolved",
    }

    def _h_session_start(self, line: SpoolLine) -> list[Event]:
        return []

    def _h_session_end(self, line: SpoolLine) -> list[Event]:
        self._ended = True
        out = self._close_turn(line)
        out.append(
            self.hook_event(
                EventType.SESSION_COMPLETED,
                line,
                payload={"outcome": Outcome.UNKNOWN.value},
            )
        )
        return out

    def _h_prompt(self, line: SpoolLine) -> list[Event]:
        out = self._close_turn(line)
        out.extend(self._open_turn(line))
        out.append(
            self.hook_event(
                EventType.MESSAGE_USER,
                line,
                action=Action.INTERACT,
                payload={"chars": len(str(line.payload.get("text") or ""))},
            )
        )
        return out

    def _h_pre_tool(self, line: SpoolLine) -> list[Event]:
        name = str(line.payload.get("tool") or line.payload.get("tool_name") or "")
        out = self._open_turn(line)
        out.extend(self._start_tool(line, name, line.payload.get("arguments") or {}))
        return out

    def _h_post_tool(self, line: SpoolLine) -> list[Event]:
        failed = bool(line.payload.get("error")) or line.payload.get("exit_code") not in (
            None, 0, "0",
        )
        return self._finish_tool(line, failed=failed)

    def _h_approval(self, line: SpoolLine) -> list[Event]:
        span_id = new_span_id()
        self._open_approvals[self._tool_key(line)] = span_id
        return [
            self.hook_event(
                EventType.APPROVAL_REQUESTED,
                line,
                span_id=span_id,
                action=Action.INTERACT,
                payload={"tool": line.payload.get("tool")},
            )
        ]

    def _h_approval_resolved(self, line: SpoolLine) -> list[Event]:
        decision = str(line.payload.get("decision") or "unknown")
        return [
            self.hook_event(
                EventType.APPROVAL_RESOLVED,
                line,
                span_id=self._open_approvals.pop(self._tool_key(line), None),
                action=Action.INTERACT,
                effect=Effect.NO_STATE_CHANGE if decision != "approved" else Effect.UNKNOWN,
                payload={"decision": decision},
            )
        ]


# ── Hermes ───────────────────────────────────────────────────────────────────


class HermesHookAdapter(HookAdapter):
    """Hermes shell hooks.

    Only the observe-only events are registered. Hermes also exposes
    `transform_tool_result`, `transform_terminal_output` and
    `transform_llm_output`, which can *alter* the agent's own data flow — an
    observability tool that mutates what it observes is not one, so those are
    absent here and from the installer, by design rather than by omission.

    Hermes is the one agent whose hooks bracket the model call
    (`pre_llm_call`/`post_llm_call`), which is why its capture gaps do not
    include per-request model timing.
    """

    agent = "hermes"
    adapter_name = "hermes_hooks"
    ROUTES = {
        "on_session_start": "_h_session_start",
        "on_session_end": "_h_session_end",
        "on_session_finalize": "_h_finalize",
        "on_session_reset": "_h_reset",
        "pre_tool_call": "_h_pre_tool",
        "post_tool_call": "_h_post_tool",
        "pre_llm_call": "_h_pre_llm",
        "post_llm_call": "_h_post_llm",
        "subagent_stop": "_h_subagent_stop",
        "pre_approval_request": "_h_approval",
        "post_approval_response": "_h_approval_resolved",
    }

    def _h_session_start(self, line: SpoolLine) -> list[Event]:
        return []

    def _h_session_end(self, line: SpoolLine) -> list[Event]:
        self._ended = True
        out = self._close_turn(line)
        out.append(
            self.hook_event(
                EventType.SESSION_COMPLETED,
                line,
                payload={
                    "reason": line.payload.get("end_reason"),
                    "outcome": Outcome.UNKNOWN.value,
                },
            )
        )
        return out

    def _h_finalize(self, line: SpoolLine) -> list[Event]:
        # fires after `on_session_end`; nothing new to record, and emitting a
        # second terminal event would give the run two end times
        return []

    def _h_reset(self, line: SpoolLine) -> list[Event]:
        return [self.hook_event(EventType.SESSION_BRANCHED, line)]

    def _h_pre_llm(self, line: SpoolLine) -> list[Event]:
        out = self._open_turn(line)
        out.append(self.hook_event(EventType.MODEL_REQUEST_STARTED, line))
        return out

    def _h_post_llm(self, line: SpoolLine) -> list[Event]:
        # Hermes brackets the model call, but the hook payload carries no usage
        # — the counts live in state.db, which is M3's reconciler, not ours
        return [self.hook_event(EventType.MODEL_REQUEST_COMPLETED, line)]

    def _h_pre_tool(self, line: SpoolLine) -> list[Event]:
        name = str(line.payload.get("tool_name") or line.payload.get("name") or "")
        out = self._open_turn(line)
        args = line.payload.get("arguments") or line.payload.get("args") or {}
        return out + self._start_tool(line, name, args if isinstance(args, dict) else {})

    def _h_post_tool(self, line: SpoolLine) -> list[Event]:
        failed = bool(line.payload.get("error") or line.payload.get("is_error"))
        return self._finish_tool(line, failed=failed)

    def _h_subagent_stop(self, line: SpoolLine) -> list[Event]:
        return [
            self.hook_event(
                EventType.SUBAGENT_COMPLETED, line, action=Action.DELEGATE
            )
        ]

    def _h_approval(self, line: SpoolLine) -> list[Event]:
        span_id = new_span_id()
        self._open_approvals[self._tool_key(line)] = span_id
        return [
            self.hook_event(
                EventType.APPROVAL_REQUESTED,
                line,
                span_id=span_id,
                action=Action.INTERACT,
                payload={"tool": line.payload.get("tool_name")},
            )
        ]

    def _h_approval_resolved(self, line: SpoolLine) -> list[Event]:
        decision = str(line.payload.get("decision") or line.payload.get("response") or "")
        return [
            self.hook_event(
                EventType.APPROVAL_RESOLVED,
                line,
                span_id=self._open_approvals.pop(self._tool_key(line), None),
                action=Action.INTERACT,
                payload={"decision": decision or "unknown"},
            )
        ]


HOOK_ADAPTERS: dict[str, type[HookAdapter]] = {
    "claude": ClaudeHookAdapter,
    "codex": CodexHookAdapter,
    "hermes": HermesHookAdapter,
}

#: hook event names each agent's installer should register, derived from what we
#: can actually map. Registering an event we drop would slow the user's session
#: for nothing.
def hook_events(agent: str) -> tuple[str, ...]:
    cls = HOOK_ADAPTERS.get(agent)
    return tuple(cls.ROUTES) if cls else ()


def _effect_for(action: Action | None) -> Effect:
    if action in (Action.EDIT, Action.VCS):
        return Effect.STATE_CHANGED
    return Effect.UNKNOWN


def _response_chars(payload: dict[str, Any]) -> int | None:
    for key in ("tool_response", "response", "result", "output"):
        v = payload.get(key)
        if isinstance(v, str):
            return len(v)
        if isinstance(v, (dict, list)):
            return len(str(v))
    return None


__all__ = [
    "CAPTURE_GAPS",
    "HOOK_ADAPTERS",
    "ClaudeHookAdapter",
    "CodexHookAdapter",
    "HermesHookAdapter",
    "HookAdapter",
    "hook_events",
]
