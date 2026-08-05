"""`codex app-server` JSON-RPC → canonical events. The attached mode for Codex.

`codex exec --json` emits seven event kinds. The app-server emits **68
notifications and 10 server→client requests**, and four of them are exactly the
things the analyses currently refuse on:

| notification | what it unblocks |
|---|---|
| `thread/tokenUsage/updated` | usage *during* a turn, not only at its end |
| `thread/compacted` | `context_pressure`, which is `missing` in every other Codex mode |
| `item/*/requestApproval` | `intervention_burden`, likewise |
| `FileUpdateChange.diff` | `edit_churn`, which has no line counts from `exec --json` |

So for Codex the DRIVEN mode is the *lower*-fidelity one, which inverts the
usual intuition that owning the process means seeing everything.

Two shapes differ from `exec --json` and both are easy to get silently wrong:

* item types are **camelCase** here (`commandExecution`) and snake_case there
  (`command_execution`), and there are 18 of them rather than 8;
* `thread/tokenUsage/updated` carries **cumulative** totals, so it is folded
  with `replace`, not `fold`. Adding each sighting would multiply the run's
  token count by the number of updates — the same failure `Usage.fold` exists
  to prevent, arriving from the other direction.

**Version pinning.** `initialize` returns no protocol version — only a
`userAgent` string carrying the CLI version. Pinning on that alone would be
theatre: the number changes on every release, most of which touch nothing we
read. What matters is the *method surface*, so `check_protocol` compares the
live server's own `generate-json-schema` output against a recorded golden
fixture and fails closed on the disappearance of a method we map. A method that
is merely new is not fatal: we ignore it, and ignoring it cannot corrupt
anything already recorded.
"""

from __future__ import annotations

import json
from typing import Any

from ..contract import (
    Action,
    CaptureMode,
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

#: What even the app-server does not report. Much shorter than the `exec --json`
#: list, and it is a real list rather than an empty one: per-*request* model
#: timing is not exposed by any Codex surface we have found. `durationMs` on a
#: command execution is the tool's wall clock, not the model's.
MISSING_IN_APP_SERVER = ("per-request model timing",)

#: Notification methods this adapter reads. `check_protocol` treats the
#: disappearance of any of these as fatal — a metric built on one would go to
#: zero without anything in the data looking wrong.
MAPPED_NOTIFICATIONS = frozenset({
    "thread/started",
    "thread/compacted",
    "thread/tokenUsage/updated",
    "thread/status/changed",
    "turn/started",
    "turn/completed",
    "turn/plan/updated",
    "item/started",
    "item/completed",
    "item/agentMessage/delta",
    "item/commandExecution/outputDelta",
    "item/fileChange/patchUpdated",
    "error",
})

#: Server→client requests. Every one of these is a moment the agent stopped and
#: waited for a human, which is the whole of `intervention_burden`.
MAPPED_REQUESTS = frozenset({
    "item/commandExecution/requestApproval",
    "item/fileChange/requestApproval",
    "item/permissions/requestApproval",
    "item/tool/requestUserInput",
    "execCommandApproval",
    "applyPatchApproval",
})

#: Codex's usage keys, in the app-server's camelCase. Same categories as
#: `exec --json`, same absent cache-*write* bucket.
_USAGE_MAP = {
    "inputTokens": TokenCategory.INPUT,
    "cachedInputTokens": TokenCategory.CACHE_READ,
    "outputTokens": TokenCategory.OUTPUT,
    "reasoningOutputTokens": TokenCategory.REASONING,
    # tolerated so a run captured against an older server is not silently
    # usage-free just because the casing changed under us
    "input_tokens": TokenCategory.INPUT,
    "cached_input_tokens": TokenCategory.CACHE_READ,
    "output_tokens": TokenCategory.OUTPUT,
    "reasoning_output_tokens": TokenCategory.REASONING,
}

_APPROVAL_DECISION = {
    "approved": "allow",
    "approvedForSession": "allow_always",
    "denied": "deny",
    "abort": "abort",
}


def diff_extent(diff: str | None) -> dict[str, int] | None:
    """Lines added and removed, counted from a unified diff.

    The Codex half of `taxonomy.edit_extent`. Same bargain: read the text once,
    keep only the integers, and let `edit_churn` report a number for Codex runs
    instead of the honest-but-unhelpful "this agent reports no line counts".

    Returns `None` for a diff with no line markers at all, so a shape we do not
    understand reads as absent rather than as a file that changed by zero lines.
    """
    if not isinstance(diff, str) or not diff:
        return None
    added = removed = 0
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue  # file headers, not content
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    if added == 0 and removed == 0:
        return None
    return {"lines_added": added, "lines_removed": removed}


class ProtocolMismatch(RuntimeError):
    """A method this adapter maps is gone from the live server."""


def check_protocol(
    live_notifications: set[str],
    live_requests: set[str],
    golden: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare a live server's method surface against what we map.

    Fails closed on *removal* and merely reports *addition*. The asymmetry is
    the point: a removed method silently zeroes whatever depended on it, while
    an added one can only be ignored.
    """
    gone_n = sorted(MAPPED_NOTIFICATIONS - live_notifications)
    gone_r = sorted(MAPPED_REQUESTS - live_requests)
    report: dict[str, Any] = {
        "compatible": not (gone_n or gone_r),
        "missing_notifications": gone_n,
        "missing_requests": gone_r,
        "unmapped_notifications": sorted(live_notifications - MAPPED_NOTIFICATIONS),
        "unmapped_requests": sorted(live_requests - MAPPED_REQUESTS),
    }
    if golden:
        recorded = set(golden.get("server_notifications", ()))
        report["golden_version"] = golden.get("codex_version")
        report["new_since_golden"] = sorted(live_notifications - recorded)
        report["gone_since_golden"] = sorted(recorded - live_notifications)
    return report


class CodexAppServerAdapter(BaseAdapter):
    agent = "codex"
    adapter_name = "codex_app_server"

    def __init__(self, **kw: Any) -> None:
        kw.setdefault("capture_mode", CaptureMode.ATTACHED)
        super().__init__(**kw)
        self._thread_id: str | None = None
        self._started = False
        self._ended = False
        #: JSON-RPC id → the approval we opened for it, so the client's response
        #: closes the same span the server's request opened.
        self._pending_approval: dict[str, dict[str, Any]] = {}
        #: item id → span id for deltas that arrive between started and completed
        self._delta_spans: dict[str, str] = {}

    # ── entry point ──────────────────────────────────────────────────────

    def feed(self, line: str) -> list[Event]:
        line = line.strip()
        if not line or not line.startswith("{"):
            return []
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return [self.warn(f"unparseable line ({len(line)} chars)")]

        method = msg.get("method")
        if method is None:
            # a response: either to one of our calls (ignored — the state we
            # care about arrives as notifications) or to an approval request,
            # which is a human decision and must be recorded.
            return self._response(msg)
        if "id" in msg and msg["id"] is not None:
            return self._server_request(msg)
        return self._notification(str(method), msg.get("params") or {})

    def finish(self) -> list[Event]:
        if self._ended or not self._started:
            return []
        self._ended = True
        # Unlike `exec --json`, a disconnect here does not mean the session
        # ended: the thread lives in the daemon and someone else may still be
        # driving it. Saying "interrupted" would be a claim about the agent
        # made from a fact about our socket.
        return [
            self.event(
                EventType.SESSION_STATE_CHANGED,
                fidelity=Fidelity.DETERMINISTIC,
                native_type="stream.detached",
                payload={
                    "state": "detached",
                    "reason": "the app-server connection closed; the thread may "
                              "still be running under another client",
                },
            )
        ]

    # ── notifications ────────────────────────────────────────────────────

    def _notification(self, method: str, params: dict) -> list[Event]:
        fn = {
            "thread/started": self._thread_started,
            "thread/compacted": self._compacted,
            "thread/tokenUsage/updated": self._token_usage,
            "thread/status/changed": self._status_changed,
            "turn/started": self._turn_started,
            "turn/completed": self._turn_completed,
            "turn/plan/updated": self._plan,
            "item/started": lambda p: self._item(p, done=False),
            "item/completed": lambda p: self._item(p, done=True),
            "item/agentMessage/delta": self._message_delta,
            "item/commandExecution/outputDelta": self._output_delta,
            "item/fileChange/patchUpdated": self._patch_updated,
            "error": self._error_notification,
        }.get(method)
        if fn is None:
            return self.note_unknown_native(method)
        return fn(params)

    def _thread_started(self, p: dict) -> list[Event]:
        self._thread_id = p.get("threadId") or p.get("thread", {}).get("id")
        self._started = True
        return [
            self.event(
                EventType.SESSION_STARTED,
                native_type="thread/started",
                native={"threadId": self._thread_id},
                source_event_id=self._thread_id,
                payload={
                    "capture_gaps": list(MISSING_IN_APP_SERVER),
                    # Canonical across adapters, unlike `native` above, which
                    # keeps each agent's own spelling. This is what a later
                    # reconciliation pass matches on to avoid importing a
                    # session we already captured.
                    "native_session_id": self._thread_id,
                },
            )
        ]

    def _status_changed(self, p: dict) -> list[Event]:
        status = p.get("status")
        state = status.get("type") if isinstance(status, dict) else status
        return [
            self.event(
                EventType.SESSION_STATE_CHANGED,
                native_type="thread/status/changed",
                payload={"state": str(state) if state is not None else None},
            )
        ]

    def _turn_started(self, p: dict) -> list[Event]:
        self.turn_id = p.get("turnId") or new_turn_id()
        return [self.event(EventType.TURN_STARTED, native_type="turn/started",
                           native=p or None)]

    def _turn_completed(self, p: dict) -> list[Event]:
        events = self._usage_events(p.get("usage") or p.get("tokenUsage") or {},
                                    "turn/completed")
        events.append(
            self.event(
                EventType.TURN_COMPLETED,
                native_type="turn/completed",
                payload={"outcome": Outcome.AGENT_CLAIMED_COMPLETE.value},
            )
        )
        self.turn_id = None
        return events

    def _token_usage(self, p: dict) -> list[Event]:
        return self._usage_events(p.get("usage") or p, "thread/tokenUsage/updated")

    def _usage_events(self, usage: dict, native_type: str) -> list[Event]:
        """Usage from a **cumulative** report.

        `thread/tokenUsage/updated` fires repeatedly with running totals, so the
        counts are *replaced*, never added. A `fold` here would multiply the
        run's tokens by the number of updates and look entirely plausible doing
        it.
        """
        if not isinstance(usage, dict) or not usage:
            return []
        # some servers nest the totals one level down
        for key in ("total", "totals", "cumulative"):
            inner = usage.get(key)
            if isinstance(inner, dict):
                usage = inner
                break
        counts = {cat: usage[k] for k, cat in _USAGE_MAP.items()
                  if isinstance(usage.get(k), int)}
        if not counts:
            return []
        self.native_usage_keys.update(k for k in usage if k in _USAGE_MAP)
        self.usage.replace(f"thread:{self._thread_id}", counts)
        return [
            self.event(
                EventType.MODEL_USAGE_UPDATED,
                native_type=native_type,
                native=dict(usage),
                payload={
                    "usage": {c.value: n for c, n in counts.items()},
                    "native_categories": sorted(k for k in usage if k in _USAGE_MAP),
                    "cumulative": True,
                    # `cumulative` is a note to a reader; `authoritative` is the
                    # instruction to the reducer, which otherwise adds each
                    # report to the last and doubles a run's tokens per update.
                    # Replacing in this adapter's own accumulator is not enough
                    # — the view is folded from the events, not from us.
                    "authoritative": True,
                    "cache_write": None,
                    "cache_write_fidelity": Fidelity.MISSING.value,
                },
            )
        ]

    def _compacted(self, p: dict) -> list[Event]:
        before, after = p.get("tokensBefore"), p.get("tokensAfter")
        return [
            self.event(
                EventType.COMPACTION_COMPLETED,
                native_type="thread/compacted",
                native=p or None,
                payload={
                    "trigger": p.get("trigger") or "unknown",
                    "tokens_before": before,
                    "tokens_after": after,
                },
            )
        ]

    def _plan(self, p: dict) -> list[Event]:
        steps = p.get("plan") or p.get("steps") or []
        if isinstance(steps, dict):
            steps = steps.get("steps") or []
        return [
            self.event(
                EventType.PLAN_UPDATED,
                native_type="turn/plan/updated",
                payload={
                    "n_items": len(steps),
                    "n_completed": sum(
                        1 for s in steps
                        if isinstance(s, dict)
                        and str(s.get("status", "")).lower() in ("completed", "done")
                    ),
                },
            )
        ]

    def _error_notification(self, p: dict) -> list[Event]:
        return [
            self.event(
                EventType.AGENT_ERROR,
                effect=Effect.FAILED,
                native_type="error",
                payload={"message": p.get("message") or p.get("error")},
            )
        ]

    # ── deltas ───────────────────────────────────────────────────────────

    def _message_delta(self, p: dict) -> list[Event]:
        item_id = p.get("itemId") or "item_?"
        return [
            self.event(
                EventType.MESSAGE_ASSISTANT_DELTA,
                span_id=self._delta_spans.get(item_id),
                native_type="item/agentMessage/delta",
                source_event_id=item_id,
                payload={"chars": len(p.get("delta") or "")},
            )
        ]

    def _output_delta(self, p: dict) -> list[Event]:
        item_id = p.get("itemId") or "item_?"
        chunk = p.get("chunk") or p.get("delta") or ""
        return [
            self.event(
                EventType.TOOL_OUTPUT_DELTA,
                span_id=self._delta_spans.get(item_id),
                native_type="item/commandExecution/outputDelta",
                source_event_id=item_id,
                payload={"output_chars": len(chunk)},
            )
        ]

    def _patch_updated(self, p: dict) -> list[Event]:
        # A patch preview, not an applied change. Recorded as a started tool so
        # the time is attributed, never as FILE_CHANGED — the file has not
        # changed yet and counting it here would double every edit.
        item_id = p.get("itemId") or "item_?"
        return [
            self.event(
                EventType.TOOL_OUTPUT_DELTA,
                span_id=self._delta_spans.get(item_id),
                action=Action.EDIT,
                native_type="item/fileChange/patchUpdated",
                source_event_id=item_id,
                payload={"applied": False},
            )
        ]

    # ── items ────────────────────────────────────────────────────────────

    def _item(self, p: dict, *, done: bool) -> list[Event]:
        item = p.get("item") or {}
        item_id = item.get("id") or p.get("itemId") or "item_?"
        itype = item.get("type") or "unknown"

        if not done:
            self.spans[item_id] = new_span_id()
        span_id = self.spans.get(item_id) or new_span_id()
        self.spans.setdefault(item_id, span_id)
        self._delta_spans[item_id] = span_id
        if done:
            self.spans.pop(item_id, None)
            self._delta_spans.pop(item_id, None)

        handler = {
            "userMessage": self._user_message,
            "agentMessage": self._agent_message,
            "hookPrompt": self._user_message,
            "plan": self._plan_item,
            "reasoning": self._reasoning,
            "commandExecution": self._command,
            "fileChange": self._file_change,
            "mcpToolCall": self._mcp_call,
            "dynamicToolCall": self._mcp_call,
            "collabAgentToolCall": self._subagent,
            "subAgentActivity": self._subagent,
            "webSearch": self._web_search,
            "imageView": self._simple_tool,
            "imageGeneration": self._simple_tool,
            "sleep": self._sleep,
            "contextCompaction": self._compaction_item,
            "enteredReviewMode": self._review_mode,
            "exitedReviewMode": self._review_mode,
        }.get(itype)
        if handler is None:
            return self.note_unknown_native(f"item:{itype}")
        return handler(item, span_id, done, item_id)

    def _user_message(self, item, span_id, done, item_id) -> list[Event]:
        if not done:
            return []
        content = item.get("content") or []
        chars = sum(len(c.get("text") or "") for c in content if isinstance(c, dict))
        return [
            self.event(
                EventType.MESSAGE_USER,
                span_id=span_id,
                action=Action.INTERACT,
                native_type=f"item:{item.get('type')}",
                source_event_id=item_id,
                payload={"chars": chars, "n_parts": len(content)},
            )
        ]

    def _agent_message(self, item, span_id, done, item_id) -> list[Event]:
        text = item.get("text") or ""
        return [
            self.event(
                EventType.MESSAGE_ASSISTANT_COMPLETED if done
                else EventType.MESSAGE_ASSISTANT_DELTA,
                span_id=span_id,
                native_type="item:agentMessage",
                source_event_id=item_id,
                action=Action.REPORT if done else None,
                payload={"chars": len(text), "text": text if done else None},
            )
        ]

    def _plan_item(self, item, span_id, done, item_id) -> list[Event]:
        if not done:
            return []
        return [
            self.event(
                EventType.PLAN_UPDATED,
                span_id=span_id,
                native_type="item:plan",
                source_event_id=item_id,
                payload={"chars": len(item.get("text") or "")},
            )
        ]

    def _reasoning(self, item, span_id, done, item_id) -> list[Event]:
        parts = item.get("content") or item.get("summary") or []
        text = "".join(c.get("text") or "" for c in parts if isinstance(c, dict))
        payload, fid = self.reasoning_payload(text)
        return [
            self.event(
                EventType.MODEL_REQUEST_COMPLETED if done
                else EventType.MODEL_REQUEST_STARTED,
                span_id=span_id,
                fidelity=fid,
                native_type="item:reasoning",
                source_event_id=item_id,
                payload={"kind": "reasoning", **payload},
            )
        ]

    def _command(self, item, span_id, done, item_id) -> list[Event]:
        cmd = item.get("command") or ""
        action = classify_command(cmd) if cmd else Action.EXECUTE
        if not done:
            return [
                self.event(
                    EventType.TOOL_STARTED,
                    span_id=span_id,
                    action=action,
                    native_type="item:commandExecution",
                    source_event_id=item_id,
                    payload={"command": cmd, "cwd": item.get("cwd")},
                )
            ]
        exit_code = item.get("exitCode")
        status = item.get("status")
        ok = exit_code == 0 if exit_code is not None else status == "completed"
        return [
            self.event(
                EventType.TOOL_COMPLETED if ok else EventType.TOOL_FAILED,
                span_id=span_id,
                action=action,
                effect=Effect.STATE_CHANGED if ok else Effect.FAILED,
                native_type="item:commandExecution",
                source_event_id=item_id,
                payload={
                    "command": cmd,
                    "exit_code": exit_code,
                    "status": status,
                    # the app-server's own wall clock for the command — better
                    # than our arrival-time subtraction, and the reason this
                    # mode's durations are native rather than deterministic
                    "duration_ms": item.get("durationMs"),
                    "output_chars": len(item.get("aggregatedOutput") or ""),
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
                    native_type="item:fileChange",
                    source_event_id=item_id,
                    payload={},
                )
            ]
        status = item.get("status")
        # `declined` means the human said no. The file did not change, and
        # recording FILE_CHANGED for it would put a phantom edit into churn.
        applied = status in (None, "completed")
        changes = [c for c in (item.get("changes") or []) if isinstance(c, dict)]
        events: list[Event] = []
        if applied:
            for c in changes:
                events.append(
                    self.event(
                        EventType.FILE_CHANGED,
                        span_id=span_id,
                        action=Action.EDIT,
                        effect=Effect.STATE_CHANGED,
                        native_type="item:fileChange",
                        source_event_id=item_id,
                        payload={
                            "path": c.get("path"),
                            "kind": c.get("kind"),
                            **(diff_extent(c.get("diff")) or {}),
                        },
                    )
                )
        events.append(
            self.event(
                EventType.TOOL_COMPLETED if applied else EventType.TOOL_FAILED,
                span_id=span_id,
                action=Action.EDIT,
                effect=(Effect.STATE_CHANGED if applied and changes
                        else Effect.NO_STATE_CHANGE),
                native_type="item:fileChange",
                source_event_id=item_id,
                payload={
                    "paths": [c.get("path") for c in changes],
                    "n_changes": len(changes),
                    "status": status,
                },
            )
        )
        return events

    def _mcp_call(self, item, span_id, done, item_id) -> list[Event]:
        server = item.get("server") or ""
        tool = item.get("tool") or item.get("name") or ""
        full = f"mcp__{server}__{tool}" if server else tool
        failed = item.get("status") == "failed" or bool(item.get("error"))
        if not done:
            return [
                self.event(
                    EventType.TOOL_STARTED,
                    span_id=span_id,
                    action=classify_tool(full),
                    native_type=f"item:{item.get('type')}",
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
                native_type=f"item:{item.get('type')}",
                source_event_id=item_id,
                payload={"tool": full, "server": server},
            )
        ]

    def _subagent(self, item, span_id, done, item_id) -> list[Event]:
        return [
            self.event(
                EventType.SUBAGENT_COMPLETED if done else EventType.SUBAGENT_STARTED,
                span_id=span_id,
                action=Action.DELEGATE,
                native_type=f"item:{item.get('type')}",
                source_event_id=item_id,
                payload={"name": item.get("name") or item.get("agent"),
                         "status": item.get("status")},
            )
        ]

    def _web_search(self, item, span_id, done, item_id) -> list[Event]:
        return [
            self.event(
                EventType.TOOL_COMPLETED if done else EventType.TOOL_STARTED,
                span_id=span_id,
                action=Action.SEARCH,
                native_type="item:webSearch",
                source_event_id=item_id,
                payload={"query": item.get("query")},
            )
        ]

    def _simple_tool(self, item, span_id, done, item_id) -> list[Event]:
        return [
            self.event(
                EventType.TOOL_COMPLETED if done else EventType.TOOL_STARTED,
                span_id=span_id,
                action=Action.INSPECT,
                native_type=f"item:{item.get('type')}",
                source_event_id=item_id,
                payload={},
            )
        ]

    def _sleep(self, item, span_id, done, item_id) -> list[Event]:
        # Deliberate waiting. It is real wall clock and belongs in the timeline,
        # but calling it `execute` would put it in the same bucket as work.
        return [
            self.event(
                EventType.TOOL_COMPLETED if done else EventType.TOOL_STARTED,
                span_id=span_id,
                action=Action.INTERACT,
                effect=Effect.NO_STATE_CHANGE,
                native_type="item:sleep",
                source_event_id=item_id,
                payload={"kind": "sleep", "duration_ms": item.get("durationMs")},
            )
        ]

    def _compaction_item(self, item, span_id, done, item_id) -> list[Event]:
        return [
            self.event(
                EventType.COMPACTION_COMPLETED if done else EventType.COMPACTION_STARTED,
                span_id=span_id,
                native_type="item:contextCompaction",
                source_event_id=item_id,
                payload={"trigger": item.get("trigger") or "unknown"},
            )
        ]

    def _review_mode(self, item, span_id, done, item_id) -> list[Event]:
        return [
            self.event(
                EventType.SESSION_STATE_CHANGED,
                span_id=span_id,
                native_type=f"item:{item.get('type')}",
                source_event_id=item_id,
                payload={"state": "review"
                         if item.get("type") == "enteredReviewMode" else "working"},
            )
        ]

    # ── approvals ────────────────────────────────────────────────────────

    def _server_request(self, msg: dict) -> list[Event]:
        method = str(msg.get("method"))
        if method not in MAPPED_REQUESTS:
            return self.note_unknown_native(f"request:{method}")
        rid = str(msg.get("id"))
        p = msg.get("params") or {}
        clarification = method == "item/tool/requestUserInput"
        self._pending_approval[rid] = {"method": method, "clarification": clarification}
        return [
            self.event(
                EventType.CLARIFICATION_REQUESTED if clarification
                else EventType.APPROVAL_REQUESTED,
                action=Action.INTERACT,
                native_type=method,
                source_event_id=rid,
                payload={
                    "request_id": rid,
                    "kind": method.rsplit("/", 1)[-1],
                    "tool": p.get("tool") or p.get("command") or p.get("callId"),
                    "reason": p.get("reason"),
                },
            )
        ]

    def _response(self, msg: dict) -> list[Event]:
        rid = str(msg.get("id"))
        pending = self._pending_approval.pop(rid, None)
        if pending is None:
            return []  # a response to one of our own calls
        result = msg.get("result")
        decision = None
        if isinstance(result, dict):
            raw = result.get("decision") or result.get("outcome")
            decision = _APPROVAL_DECISION.get(str(raw), raw)
        return [
            self.event(
                EventType.CLARIFICATION_RESOLVED if pending["clarification"]
                else EventType.APPROVAL_RESOLVED,
                action=Action.INTERACT,
                native_type=f"{pending['method']}#response",
                source_event_id=rid,
                payload={"request_id": rid, "decision": decision},
            )
        ]


__all__ = [
    "MAPPED_NOTIFICATIONS",
    "MAPPED_REQUESTS",
    "MISSING_IN_APP_SERVER",
    "CodexAppServerAdapter",
    "ProtocolMismatch",
    "check_protocol",
    "diff_extent",
]
