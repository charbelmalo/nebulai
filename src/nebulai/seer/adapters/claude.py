"""`claude -p --output-format stream-json` → canonical events.

Shape confirmed from a captured run (`tests/fixtures/seer/claude-stream.jsonl`),
which is worth reading because two of its properties decide the whole adapter:

1. **One model response is several lines, each repeating identical `usage`.**
   The capture has two `assistant` lines sharing `message.id`
   `msg_011CdibT8k6jHX5HUNxHdtMr` and `request_id` `req_011CdibT82wArqcaGDrQzpEN`,
   both carrying `output_tokens: 4`. Summing per line is how
   `viewer/src/chrome/sessionlog.ts` once overcounted a real session by 3.5×.
   We fold on `message.id`.

2. **The streamed usage is not the total.** Those lines say `output_tokens: 4`;
   the terminal `result` line says `36`. Streamed usage is a partial view, so
   the `result` line *replaces* rather than adds — the same "the result line is
   authoritative" rule the session plotter already runs on.

Claude's native categories are `input_tokens`, `cache_creation_input_tokens`,
`cache_read_input_tokens`, `output_tokens`. Reasoning is billed *inside*
`output_tokens` and has no bucket of its own, so `TokenCategory.REASONING` stays
`MISSING` here even though `system/thinking_tokens` offers an
`estimated_tokens` — Claude labels that field "estimated" itself, and we keep
its word for it rather than laundering it into a native count.
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
from ..taxonomy import classify_command, classify_tool, edit_extent
from .base import BaseAdapter

_USAGE_MAP = {
    "input_tokens": TokenCategory.INPUT,
    "cache_creation_input_tokens": TokenCategory.CACHE_WRITE,
    "cache_read_input_tokens": TokenCategory.CACHE_READ,
    "output_tokens": TokenCategory.OUTPUT,
}

#: `stop_reason` values that mean the model stopped because something outside
#: the task said so. Distinguishing these from `end_turn` is what keeps a
#: truncated run from being scored as a completed one.
_ABNORMAL_STOP = {"max_tokens", "refusal", "pause_turn"}

#: Tools whose successful result means a file on disk changed. Claude Code has
#: no `file_change` event — Codex does — so without this a Codex run that wrote
#: one file would report one file changed and an identical Claude run would
#: report none. That gap is in our plumbing, not in the agents, and leaving it
#: in would make "files changed" a measurement of which adapter we wrote better.
#: The resulting `FILE_CHANGED` is `DETERMINISTIC`, not `NATIVE`: we inferred it
#: from a tool call plus a non-error result, and the label says so.
_EDIT_TOOLS = {
    "Write": "file_path",
    "Edit": "file_path",
    "MultiEdit": "file_path",
    "NotebookEdit": "notebook_path",
}


class ClaudeStreamAdapter(BaseAdapter):
    agent = "claude"
    adapter_name = "claude_stream_json"

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self._native_session: str | None = None
        #: tool_use id → our span id, so the `user`/`tool_result` line that
        #: closes a call can find the span the `assistant` line opened.
        self._tool_spans: dict[str, str] = {}
        self._tool_actions: dict[str, Action] = {}
        #: tool_use id → the path an edit tool is about to write, held until the
        #: result comes back. See `_EDIT_TOOLS`.
        self._tool_paths: dict[str, str] = {}
        self._tool_extents: dict[str, dict[str, int]] = {}
        self._thinking_estimate: int | None = None

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
        if kind == "system":
            return self._system(msg)
        if kind == "assistant":
            return self._assistant(msg)
        if kind == "user":
            return self._user(msg)
        if kind == "result":
            return self._result(msg)
        if kind == "rate_limit_event":
            return self._rate_limit(msg)
        if kind == "stream_event":
            return self._stream_event(msg)
        return self.note_unknown_native(str(kind))

    def finish(self) -> list[Event]:
        if self.turn_id is not None:
            return [
                self.event(
                    EventType.SESSION_INTERRUPTED,
                    fidelity=Fidelity.DETERMINISTIC,
                    payload={
                        "outcome": Outcome.INTERRUPTED.value,
                        "reason": "stream ended before result line",
                    },
                )
            ]
        return []

    # ── system ───────────────────────────────────────────────────────────

    def _system(self, msg: dict) -> list[Event]:
        sub = msg.get("subtype")
        if sub == "init":
            return self._init(msg)
        if sub == "thinking_tokens":
            return self._thinking_tokens(msg)
        if sub == "post_turn_summary":
            return self._post_turn_summary(msg)
        if sub in ("compact_boundary", "compaction"):
            return [
                self.event(
                    EventType.COMPACTION_COMPLETED,
                    native_type=f"system.{sub}",
                    native=msg,
                    payload={"trigger": (msg.get("compact_metadata") or {}).get("trigger")},
                )
            ]
        return self.note_unknown_native(f"system.{sub}")

    def _init(self, msg: dict) -> list[Event]:
        self._native_session = msg.get("session_id")
        self.agent_version = msg.get("claude_code_version") or self.agent_version
        self.model = {"provider": "anthropic", "model_id": msg.get("model")}
        self.turn_id = new_turn_id()
        mcp = msg.get("mcp_servers") or []
        return [
            self.event(
                EventType.SESSION_STARTED,
                native_type="system.init",
                native={"session_id": self._native_session},
                source_event_id=msg.get("uuid"),
                payload={
                    "native_session_id": self._native_session,
                    "cwd": msg.get("cwd"),
                    "permission_mode": msg.get("permissionMode"),
                    "n_tools": len(msg.get("tools") or []),
                    "tools": msg.get("tools") or [],
                    # A failed MCP server changes what the agent *could* do, so
                    # it belongs in the run's capability record, not a log line.
                    "mcp_failed": [
                        s.get("name") for s in mcp
                        if isinstance(s, dict) and s.get("status") != "connected"
                    ],
                },
            )
        ]

    def _thinking_tokens(self, msg: dict) -> list[Event]:
        # The field is named `estimated_tokens` by Claude. We keep it at
        # ESTIMATED fidelity and out of the usage counters entirely: mixing an
        # estimate into a native total is how a headline number becomes wrong
        # while looking authoritative.
        total = msg.get("estimated_tokens")
        self._thinking_estimate = total
        return [
            self.event(
                EventType.MODEL_USAGE_UPDATED,
                fidelity=Fidelity.ESTIMATED,
                native_type="system.thinking_tokens",
                source_event_id=msg.get("uuid"),
                payload={
                    "reasoning_tokens_estimated": total,
                    "delta": msg.get("estimated_tokens_delta"),
                    "counted": False,
                },
            )
        ]

    def _post_turn_summary(self, msg: dict) -> list[Event]:
        ev = self.event(
            EventType.TURN_COMPLETED,
            fidelity=Fidelity.NATIVE,
            native_type="system.post_turn_summary",
            source_event_id=msg.get("uuid"),
            payload={
                "status_category": msg.get("status_category"),
                "status_detail": msg.get("status_detail"),
                "needs_action": msg.get("needs_action") or None,
            },
        )
        self.turn_id = new_turn_id()
        return [ev]

    def _rate_limit(self, msg: dict) -> list[Event]:
        info = msg.get("rate_limit_info") or {}
        return [
            self.event(
                EventType.QUOTA_UPDATED,
                native_type="rate_limit_event",
                source_event_id=msg.get("uuid"),
                payload={
                    "status": info.get("status"),
                    "limit_type": info.get("rateLimitType"),
                    "resets_at": info.get("resetsAt"),
                    "using_overage": info.get("isUsingOverage"),
                },
            )
        ]

    # ── messages ─────────────────────────────────────────────────────────

    def _assistant(self, msg: dict) -> list[Event]:
        m = msg.get("message") or {}
        msg_id = m.get("id") or msg.get("uuid") or "msg_?"
        req_id = msg.get("request_id")
        if m.get("model"):
            self.model = {"provider": "anthropic", "model_id": m["model"]}

        events: list[Event] = []
        # Usage first, folded on message.id. `counted` records whether this
        # sighting was the one that contributed, so a fold bug shows up in the
        # data instead of quietly doubling a total.
        usage = m.get("usage") or {}
        if usage:
            counts = {c: usage[k] for k, c in _USAGE_MAP.items() if k in usage}
            fresh = self.fold_usage(f"msg:{msg_id}", counts, usage.keys())
            events.append(
                self.event(
                    EventType.MODEL_USAGE_UPDATED,
                    native_type="assistant.usage",
                    native={"message_id": msg_id, "request_id": req_id},
                    payload={
                        "usage": {c.value: n for c, n in counts.items()},
                        "native_categories": sorted(usage.keys()),
                        "counted": fresh,
                        # Claude bills reasoning inside output_tokens; there is
                        # no native bucket to read, so this stays absent.
                        "reasoning": None,
                        "reasoning_fidelity": Fidelity.MISSING.value,
                        "provisional": True,
                    },
                )
            )

        parent = msg.get("parent_tool_use_id")
        for block in m.get("content") or []:
            if not isinstance(block, dict):
                continue
            events.extend(self._content_block(block, msg, msg_id, parent))

        stop = m.get("stop_reason")
        if stop in _ABNORMAL_STOP:
            events.append(
                self.event(
                    EventType.AGENT_ERROR,
                    effect=Effect.FAILED,
                    native_type="assistant.stop_reason",
                    source_event_id=msg_id,
                    payload={"stop_reason": stop},
                )
            )
        return events

    def _content_block(
        self, block: dict, msg: dict, msg_id: str, parent: str | None
    ) -> list[Event]:
        btype = block.get("type")

        if btype == "text":
            text = block.get("text") or ""
            return [
                self.event(
                    EventType.MESSAGE_ASSISTANT_COMPLETED,
                    action=Action.REPORT,
                    parent_span_id=self._tool_spans.get(parent) if parent else None,
                    native_type="assistant.text",
                    source_event_id=msg_id,
                    payload={"chars": len(text), "text": text},
                )
            ]

        if btype == "thinking":
            payload, fid = self.reasoning_payload(block.get("thinking"))
            return [
                self.event(
                    EventType.MODEL_REQUEST_COMPLETED,
                    fidelity=fid,
                    native_type="assistant.thinking",
                    source_event_id=msg_id,
                    payload={"kind": "reasoning", **payload},
                )
            ]

        if btype == "tool_use":
            use_id = block.get("id") or new_span_id()
            span_id = new_span_id()
            self._tool_spans[use_id] = span_id
            name = block.get("name") or ""
            inp = block.get("input") or {}
            # Bash is the one tool whose action depends on its argument: `pytest`
            # and `rm -rf` are the same tool and opposite actions.
            action = (
                classify_command(str(inp.get("command") or ""))
                if name in ("Bash", "BashOutput") and inp.get("command")
                else classify_tool(name)
            )
            self._tool_actions[use_id] = action
            if name in _EDIT_TOOLS:
                path = inp.get(_EDIT_TOOLS[name])
                if path:
                    self._tool_paths[use_id] = str(path)
                # Measured here and carried to the result, because the extent
                # is in the *call* and the confirmation that it landed is in
                # the *result* — churn needs both halves.
                extent = edit_extent(name, inp)
                if extent:
                    self._tool_extents[use_id] = extent
            return [
                self.event(
                    EventType.TOOL_STARTED,
                    span_id=span_id,
                    parent_span_id=self._tool_spans.get(parent) if parent else None,
                    action=action,
                    native_type=name,
                    source_event_id=use_id,
                    payload={
                        "tool": name,
                        "command": inp.get("command"),
                        "path": inp.get("file_path") or inp.get("path"),
                    },
                )
            ]

        return self.note_unknown_native(f"content.{btype}")

    def _user(self, msg: dict) -> list[Event]:
        m = msg.get("message") or {}
        content = m.get("content")
        if isinstance(content, str):
            return [
                self.event(
                    EventType.MESSAGE_USER,
                    action=Action.INTERACT,
                    native_type="user.text",
                    source_event_id=msg.get("uuid"),
                    payload={"chars": len(content)},
                )
            ]

        events: list[Event] = []
        for block in content or []:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            use_id = block.get("tool_use_id") or ""
            span_id = self._tool_spans.pop(use_id, None)
            action = self._tool_actions.pop(use_id, None)
            path = self._tool_paths.pop(use_id, None)
            extent = self._tool_extents.pop(use_id, None)
            failed = bool(block.get("is_error"))
            body = block.get("content")
            chars = len(body) if isinstance(body, str) else _blocks_chars(body)
            if path and not failed:
                events.append(
                    self.event(
                        EventType.FILE_CHANGED,
                        span_id=span_id,
                        fidelity=Fidelity.DETERMINISTIC,
                        action=Action.EDIT,
                        effect=Effect.STATE_CHANGED,
                        native_type="user.tool_result",
                        source_event_id=use_id,
                        payload={"path": path, "kind": "update", **(extent or {})},
                    )
                )
            events.append(
                self.event(
                    EventType.TOOL_FAILED if failed else EventType.TOOL_COMPLETED,
                    span_id=span_id,
                    action=action,
                    effect=Effect.FAILED if failed else _effect_for(action),
                    native_type="user.tool_result",
                    source_event_id=use_id,
                    payload={"output_chars": chars, "is_error": failed},
                )
            )
        return events

    def _stream_event(self, msg: dict) -> list[Event]:
        # Only present with --include-partial-messages. Deltas update previews
        # and are barred from counters by construction (`is_delta`).
        ev = msg.get("event") or {}
        if ev.get("type") != "content_block_delta":
            return []
        delta = ev.get("delta") or {}
        text = delta.get("text") or delta.get("partial_json") or ""
        return [
            self.event(
                EventType.MESSAGE_ASSISTANT_DELTA,
                native_type="stream_event.content_block_delta",
                payload={"chars": len(text)},
            )
        ]

    # ── the authoritative line ───────────────────────────────────────────

    def _result(self, msg: dict) -> list[Event]:
        usage = msg.get("usage") or {}
        events: list[Event] = []
        if usage:
            counts = {c: usage[k] for k, c in _USAGE_MAP.items() if k in usage}
            # Replace, don't add: everything folded during the stream was a
            # partial view of these same totals.
            self.usage.replace("result", counts)
            self.native_usage_keys.update(usage.keys())
            events.append(
                self.event(
                    EventType.MODEL_USAGE_UPDATED,
                    native_type="result.usage",
                    native=dict(usage),
                    payload={
                        "usage": {c.value: n for c, n in counts.items()},
                        "native_categories": sorted(usage.keys()),
                        "authoritative": True,
                        "reasoning": None,
                        "reasoning_fidelity": Fidelity.MISSING.value,
                        "reasoning_tokens_estimated": self._thinking_estimate,
                    },
                )
            )

        for denial in msg.get("permission_denials") or []:
            events.append(
                self.event(
                    EventType.APPROVAL_RESOLVED,
                    action=Action.INTERACT,
                    effect=Effect.NO_STATE_CHANGE,
                    native_type="result.permission_denial",
                    payload={
                        "tool": denial.get("tool_name") if isinstance(denial, dict) else None,
                        "decision": "denied",
                    },
                )
            )

        is_error = bool(msg.get("is_error")) or msg.get("subtype") not in (
            None,
            "success",
        )
        outcome = (
            Outcome.INFRASTRUCTURE_FAILURE
            if msg.get("api_error_status")
            else Outcome.INTERRUPTED
            if msg.get("terminal_reason") in ("interrupted", "cancelled")
            else Outcome.UNKNOWN
            if is_error
            # The agent finished and said so. That is a claim about the agent,
            # not evidence about the task — an evaluator is what upgrades this.
            else Outcome.AGENT_CLAIMED_COMPLETE
        )
        events.append(
            self.event(
                EventType.SESSION_FAILED if is_error else EventType.SESSION_COMPLETED,
                native_type="result",
                native={"session_id": msg.get("session_id")},
                source_event_id=msg.get("uuid"),
                payload={
                    "outcome": outcome.value,
                    "cost_usd": msg.get("total_cost_usd"),
                    "duration_ms": msg.get("duration_ms"),
                    "duration_api_ms": msg.get("duration_api_ms"),
                    "num_turns": msg.get("num_turns"),
                    "stop_reason": msg.get("stop_reason"),
                    "terminal_reason": msg.get("terminal_reason"),
                    "ttft_ms": msg.get("ttft_ms"),
                    "context_window": _context_window(msg),
                },
            )
        )
        self.turn_id = None
        return events


def _blocks_chars(blocks: Any) -> int:
    if not isinstance(blocks, list):
        return 0
    return sum(
        len(b.get("text") or "") for b in blocks if isinstance(b, dict)
    )


def _effect_for(action: Action | None) -> Effect:
    """Effect we can assert from a successful tool result alone.

    Deliberately conservative: a successful Read tells us the call worked, not
    whether it surfaced anything the run had not already seen. Deciding
    `new_information` vs `no_new_information` needs cross-event comparison, and
    that belongs in the reducer, which can see the whole trajectory.
    """
    if action in (Action.EDIT, Action.VCS):
        return Effect.STATE_CHANGED
    return Effect.UNKNOWN


def _context_window(msg: dict) -> int | None:
    """The model's context window, if the result line names one model.

    `modelUsage` is keyed by model; with more than one model in a run there is
    no single window to report, and returning the first would be a quiet lie.
    """
    mu = msg.get("modelUsage") or {}
    if len(mu) != 1:
        return None
    return next(iter(mu.values())).get("contextWindow")
