"""`codex exec --json` → canonical events.

Vocabulary confirmed against the shipped binary (`strings` on
`codex-darwin-arm64/…/bin/codex`, cross-checked against a captured run in
`tests/fixtures/seer/codex-exec.jsonl`), not against documentation:

    events      thread.started turn.started turn.completed turn.failed
                item.started item.updated item.completed
    item types  agent_message reasoning command_execution file_change
                mcp_tool_call web_search todo_list error

That is seven event kinds. The Codex *app-server* exposes 68 JSON-RPC
notifications. So the DRIVEN capture mode — the one where we own the process —
is the *lower*-fidelity of the two for Codex, which inverts the usual intuition
that launching something means seeing everything. The adapter says so out loud
in `MISSING_IN_EXEC_JSON` rather than letting the data-quality panel imply the
gaps are Codex's fault.

Usage arrives exactly once, on `turn.completed`, in Codex's own categories:
`input_tokens`, `cached_input_tokens`, `output_tokens`, `reasoning_output_tokens`.
Note what is *not* there: no cache-**write** category at all, and reasoning
broken out *beside* output rather than folded into it — the opposite of Claude
on both counts. `native_usage_keys` carries that fact forward so the
comparability gate can refuse a cross-agent token delta instead of computing a
confident wrong one.
"""

from __future__ import annotations

import json
from typing import Any

from ..contract import (
    Action,
    Effect,
    Event,
    EventType,
    Fidelity,
    Outcome,
    TokenCategory,
    new_span_id,
    new_turn_id,
)
from ..taxonomy import classify_command, classify_tool
from .base import BaseAdapter

#: Things the app-server reports and `exec --json` does not. Emitted as an
#: adapter note at session start so a run captured this way is never mistaken
#: for a run where these were absent.
MISSING_IN_EXEC_JSON = (
    "approval requests/decisions",
    "per-request model timing",
    "context-window pressure",
    "token usage before turn end",
)

#: Codex's native usage keys → our categories. `cached_input_tokens` is a
#: cache *read*; Codex has no write category, so `cache_write` stays absent
#: rather than being invented as 0.
_USAGE_MAP = {
    "input_tokens": TokenCategory.INPUT,
    "cached_input_tokens": TokenCategory.CACHE_READ,
    "output_tokens": TokenCategory.OUTPUT,
    "reasoning_output_tokens": TokenCategory.REASONING,
}


class CodexExecAdapter(BaseAdapter):
    agent = "codex"
    adapter_name = "codex_exec_json"

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self._thread_id: str | None = None
        self._started = False

    # ── entry point ──────────────────────────────────────────────────────

    def feed(self, line: str) -> list[Event]:
        line = line.strip()
        if not line or not line.startswith("{"):
            return []
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return [self.warn(f"unparseable line ({len(line)} chars)")]

        kind = msg.get("type")
        if kind == "thread.started":
            return self._thread_started(msg)
        if kind == "turn.started":
            return self._turn_started(msg)
        if kind == "turn.completed":
            return self._turn_completed(msg)
        if kind == "turn.failed":
            return self._turn_failed(msg)
        if kind in ("item.started", "item.updated", "item.completed"):
            return self._item(kind, msg)
        return self.note_unknown_native(str(kind))

    def finish(self) -> list[Event]:
        # `codex exec --json` has no session-end event: the stream simply stops
        # after `turn.completed`. In DRIVEN mode we own the process, so its exit
        # *is* the end of the session and closing the run here is deterministic
        # rather than a guess. Without this the run would sit un-ended forever,
        # and every duration metric for Codex would be `missing` while the same
        # metric for Claude (which does emit a `result` line) was present —
        # an asymmetry in our plumbing masquerading as one between the agents.
        if self.turn_id is not None:
            return [
                self.event(
                    EventType.SESSION_INTERRUPTED,
                    fidelity=Fidelity.DETERMINISTIC,
                    payload={
                        "outcome": Outcome.INTERRUPTED.value,
                        "reason": "stream ended mid-turn",
                    },
                )
            ]
        if self._started:
            return [
                self.event(
                    EventType.SESSION_COMPLETED,
                    fidelity=Fidelity.DETERMINISTIC,
                    native_type="process.exit",
                    payload={
                        "outcome": Outcome.AGENT_CLAIMED_COMPLETE.value,
                        "reason": "stream ended after a completed turn",
                    },
                )
            ]
        return []

    # ── lifecycle ────────────────────────────────────────────────────────

    def _thread_started(self, msg: dict) -> list[Event]:
        self._thread_id = msg.get("thread_id")
        self._started = True
        out = [
            self.event(
                EventType.SESSION_STARTED,
                native_type="thread.started",
                native={"thread_id": self._thread_id},
                source_event_id=self._thread_id,
                payload={
                    "capture_gaps": list(MISSING_IN_EXEC_JSON),
                    "native_session_id": self._thread_id,
                },
            )
        ]
        return out

    def _turn_started(self, msg: dict) -> list[Event]:
        self.turn_id = new_turn_id()
        return [
            self.event(
                EventType.TURN_STARTED,
                native_type="turn.started",
                native=msg or None,
            )
        ]

    def _turn_completed(self, msg: dict) -> list[Event]:
        usage = msg.get("usage") or {}
        events: list[Event] = []
        if usage:
            counts = {
                cat: usage[key] for key, cat in _USAGE_MAP.items() if key in usage
            }
            # Fold on the turn, not the line: `turn.completed` is the only
            # usage-bearing event in this stream today, but keying on the turn
            # keeps a future `turn.updated` from double-counting.
            fresh = self.fold_usage(f"turn:{self.turn_id}", counts, usage.keys())
            events.append(
                self.event(
                    EventType.MODEL_USAGE_UPDATED,
                    native_type="turn.completed.usage",
                    native=dict(usage),
                    payload={
                        "usage": {c.value: n for c, n in counts.items()},
                        "native_categories": sorted(usage.keys()),
                        "counted": fresh,
                        # Codex reports no cache-write bucket at all. Absent, not zero.
                        "cache_write": None,
                        "cache_write_fidelity": Fidelity.MISSING.value,
                    },
                )
            )
        events.append(
            self.event(
                EventType.TURN_COMPLETED,
                native_type="turn.completed",
                payload={"outcome": Outcome.AGENT_CLAIMED_COMPLETE.value},
            )
        )
        self.turn_id = None
        return events

    def _turn_failed(self, msg: dict) -> list[Event]:
        err = msg.get("error") or {}
        ev = self.event(
            EventType.TURN_FAILED,
            native_type="turn.failed",
            native=msg or None,
            effect=Effect.FAILED,
            payload={
                "outcome": Outcome.INFRASTRUCTURE_FAILURE.value,
                "message": err.get("message") if isinstance(err, dict) else str(err),
            },
        )
        self.turn_id = None
        return [ev]

    # ── items ────────────────────────────────────────────────────────────

    def _item(self, kind: str, msg: dict) -> list[Event]:
        item = msg.get("item") or {}
        item_id = item.get("id") or msg.get("item_id") or "item_?"
        itype = item.get("type") or "unknown"

        if kind == "item.started":
            self.spans[item_id] = new_span_id()
        span_id = self.spans.get(item_id) or new_span_id()
        self.spans.setdefault(item_id, span_id)
        done = kind == "item.completed"
        if done:
            self.spans.pop(item_id, None)

        handler = {
            "agent_message": self._agent_message,
            "reasoning": self._reasoning,
            "command_execution": self._command,
            "file_change": self._file_change,
            "mcp_tool_call": self._mcp_call,
            "web_search": self._web_search,
            "todo_list": self._todo,
            "error": self._error,
        }.get(itype)
        if handler is None:
            return self.note_unknown_native(f"item:{itype}")
        return handler(item, span_id, done, item_id)

    def _agent_message(self, item, span_id, done, item_id) -> list[Event]:
        text = item.get("text") or ""
        et = (
            EventType.MESSAGE_ASSISTANT_COMPLETED
            if done
            else EventType.MESSAGE_ASSISTANT_DELTA
        )
        return [
            self.event(
                et,
                span_id=span_id,
                native_type=f"item.{item.get('type')}",
                source_event_id=item_id,
                action=Action.REPORT if done else None,
                payload={"chars": len(text), "text": text if done else None},
            )
        ]

    def _reasoning(self, item, span_id, done, item_id) -> list[Event]:
        # Codex genuinely streams this. Refusing it is our decision, so the
        # fidelity is DROPPED_BY_POLICY and the span still exists — the model
        # spent wall-clock here and time decomposition needs to know.
        payload, fid = self.reasoning_payload(item.get("text"))
        et = EventType.MODEL_REQUEST_COMPLETED if done else EventType.MODEL_REQUEST_STARTED
        return [
            self.event(
                et,
                span_id=span_id,
                fidelity=fid,
                native_type="item.reasoning",
                source_event_id=item_id,
                payload={"kind": "reasoning", **payload},
            )
        ]

    def _command(self, item, span_id, done, item_id) -> list[Event]:
        cmd = item.get("command") or ""
        exit_code = item.get("exit_code")
        status = item.get("status")
        action = classify_command(cmd) if cmd else Action.EXECUTE
        out = item.get("aggregated_output") or ""

        if not done:
            et = (
                EventType.TOOL_OUTPUT_DELTA
                if status == "in_progress" and out
                else EventType.TOOL_STARTED
            )
            return [
                self.event(
                    et,
                    span_id=span_id,
                    action=action,
                    native_type="item.command_execution",
                    source_event_id=item_id,
                    payload={"command": cmd, "output_chars": len(out)},
                )
            ]

        ok = exit_code == 0
        return [
            self.event(
                EventType.TOOL_COMPLETED if ok else EventType.TOOL_FAILED,
                span_id=span_id,
                action=action,
                effect=Effect.STATE_CHANGED if ok else Effect.FAILED,
                native_type="item.command_execution",
                source_event_id=item_id,
                payload={
                    "command": cmd,
                    "exit_code": exit_code,
                    "output_chars": len(out),
                    "status": status,
                },
            )
        ]

    def _file_change(self, item, span_id, done, item_id) -> list[Event]:
        if not done:
            return [
                self.event(
                    EventType.TOOL_STARTED,
                    span_id=span_id,
                    action=Action.EDIT,
                    native_type="item.file_change",
                    source_event_id=item_id,
                    payload={},
                )
            ]
        changes = item.get("changes") or []
        # `changes` is a list of {path, kind: add|delete|update} in the shipped
        # binary's vocabulary; tolerate a dict form too rather than crashing a
        # live capture on a shape change.
        if isinstance(changes, dict):
            changes = [{"path": p, **(v if isinstance(v, dict) else {"kind": v})}
                       for p, v in changes.items()]
        paths = [c.get("path") for c in changes if isinstance(c, dict)]
        events = [
            self.event(
                EventType.FILE_CHANGED,
                span_id=span_id,
                action=Action.EDIT,
                effect=Effect.STATE_CHANGED,
                native_type="item.file_change",
                source_event_id=item_id,
                payload={
                    "path": c.get("path"),
                    "kind": c.get("kind"),
                },
            )
            for c in changes
            if isinstance(c, dict)
        ]
        events.append(
            self.event(
                EventType.TOOL_COMPLETED,
                span_id=span_id,
                action=Action.EDIT,
                effect=Effect.STATE_CHANGED if paths else Effect.NO_STATE_CHANGE,
                native_type="item.file_change",
                source_event_id=item_id,
                payload={"paths": paths, "n_changes": len(paths)},
            )
        )
        return events

    def _mcp_call(self, item, span_id, done, item_id) -> list[Event]:
        server = item.get("server") or ""
        tool = item.get("tool") or item.get("name") or ""
        full = f"mcp__{server}__{tool}" if server else tool
        failed = bool(item.get("error")) or item.get("status") == "failed"
        if not done:
            return [
                self.event(
                    EventType.TOOL_STARTED,
                    span_id=span_id,
                    action=classify_tool(full),
                    native_type="item.mcp_tool_call",
                    source_event_id=item_id,
                    payload={"tool": full},
                )
            ]
        return [
            self.event(
                EventType.TOOL_FAILED if failed else EventType.TOOL_COMPLETED,
                span_id=span_id,
                action=classify_tool(full),
                effect=Effect.FAILED if failed else Effect.UNKNOWN,
                native_type="item.mcp_tool_call",
                source_event_id=item_id,
                payload={"tool": full, "server": server},
            )
        ]

    def _web_search(self, item, span_id, done, item_id) -> list[Event]:
        return [
            self.event(
                EventType.TOOL_COMPLETED if done else EventType.TOOL_STARTED,
                span_id=span_id,
                action=Action.SEARCH,
                native_type="item.web_search",
                source_event_id=item_id,
                payload={"query": item.get("query")},
            )
        ]

    def _todo(self, item, span_id, done, item_id) -> list[Event]:
        items = item.get("items") or []
        return [
            self.event(
                EventType.PLAN_UPDATED,
                span_id=span_id,
                native_type="item.todo_list",
                source_event_id=item_id,
                payload={
                    "n_items": len(items),
                    "n_completed": sum(
                        1 for t in items
                        if isinstance(t, dict) and t.get("completed") is True
                    ),
                },
            )
        ]

    def _error(self, item, span_id, done, item_id) -> list[Event]:
        return [
            self.event(
                EventType.AGENT_ERROR,
                span_id=span_id,
                effect=Effect.FAILED,
                native_type="item.error",
                source_event_id=item_id,
                payload={"message": item.get("message")},
            )
        ]
