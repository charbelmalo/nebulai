"""What a captured event is allowed to contain, and how to take it back out.

Two separate jobs live here, and they answer two different questions.

**Classification** answers *what is in this log?* Every event carries a
`privacy.content_level`, and until M5 that label was the constant `"metadata"`
on every event ever written — including events whose payload held the model's
prose verbatim. A label that is always the same is not a label. `FIELD_POLICY`
below is the real answer: every payload key SessionSeer emits is registered at
a level, `event_level()` takes the maximum over the keys actually present, and
a test walks the whole suite and fails on any key that is not registered. A new
agent version that starts handing us a field we have never seen cannot ship it
into the log unclassified.

**Redaction** answers *what can I hand to someone else?* `redact_event()` takes
an event down to a requested level: the fields above that level are replaced by
their own lengths, and the event says which ones went. Nothing is deleted
silently, because a redacted log that looks like a complete one is worse than
no log.

The rungs, lowest first:

`metadata`
    Identifiers, file paths, counts, enum values, model ids, and SessionSeer's
    own notes. Everything the analyses read. A log at this level still answers
    every question in `analysis.py`.
`command`
    Verbatim shell command lines. Its own rung rather than lumped with prose
    because `classify_command` is the one place where the *argument* decides
    the action — `pytest` and `rm -rf` are the same tool and opposite verbs —
    so dropping commands costs a classification nothing else can recover. A
    researcher sharing a run usually wants to keep these and drop the prose.
`content`
    Natural language and program output: prompts, replies, error text, stderr
    tails, and the `native` passthrough of the agent's own message. This is the
    rung that carries whatever the user was actually working on.

Scrubbing is separate from both and is not optional. A secret in a command line
is not content the user chose to record — `curl -H "Authorization: Bearer …"`
is a credential that leaked into a log because we write commands down. It is
removed in `Event.__post_init__`, before the event can reach a file, and the
event says that it happened.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

#: Bumped when the patterns or the field registry change, so a log can say
#: which rules produced it. A reader comparing two exports needs this to know
#: whether a field's absence means "redacted" or "never existed".
RULESET = "r1"

REDACTED = "<seer:redacted>"


class ContentLevel(str, Enum):
    """How revealing a field is. Ordered; see `rank()`."""

    METADATA = "metadata"
    COMMAND = "command"
    CONTENT = "content"


_ORDER = {ContentLevel.METADATA: 0, ContentLevel.COMMAND: 1, ContentLevel.CONTENT: 2}


def rank(level: ContentLevel) -> int:
    return _ORDER[level]


# ── the registry ─────────────────────────────────────────────────────────────
#
# Keyed by top-level payload key. A nested dict inherits its parent's level, so
# `usage` covers `usage.input` and the rest of its buckets.

#: Free text: what the user asked, what the model said, what a program printed.
_CONTENT = frozenset({
    "text",           # message text, from the user or the agent
    "message",        # agent-authored error text (ours is `note`)
    "status_detail",  # Claude's end-of-turn prose
    "stderr_tail",    # literal program output
    "reason",         # why a turn ended, sometimes quoting the agent
    "error",          # error object as the agent gave it
})

#: Shell command lines. Kept at their own rung; see the module docstring.
_COMMAND = frozenset({"command"})

#: Everything else SessionSeer emits. Listing them is the point: an
#: unregistered key fails `test_seer_redaction.py`, which is how a field that
#: quietly starts carrying content gets noticed on the day it appears rather
#: than the day someone reads an export.
_METADATA = frozenset({
    "agent", "author", "authoritative", "bytes", "cache_write",
    "cache_write_fidelity", "capture_gaps", "chars", "cli_version",
    "clock_resolution_s", "codex_bin", "compatible", "context_window",
    "cost_usd", "counted", "cumulative", "cwd", "decision", "delta",
    "duration_api_ms", "duration_ms", "duration_s", "effort", "exit_code",
    "first_hook", "golden_version", "gone_since_golden", "has_name",
    "history_mode", "is_error", "joined_midstream", "kind", "label",
    "limit_type", "lines_added", "lines_removed", "mcp_failed",
    "missing_notifications", "missing_requests", "model_requested",
    "n_changes", "n_events", "n_parts", "n_tools", "n_turns",
    "native_categories", "native_session_id", "needs_action",
    "new_since_golden", "note", "num_turns", "outcome", "output_chars",
    "path", "paths", "permission_mode", "provisional", "reasoning",
    "reasoning_fidelity", "reasoning_tokens_estimated", "recovered",
    "request_id", "resets_at", "source", "state", "status", "status_category",
    "stop_reason", "tags", "terminal_reason", "text_retained", "tool",
    "tools", "total_lines", "transport", "ttft_ms",
    "unmapped_notifications", "unmapped_requests", "usage", "using_overage",
})

FIELD_POLICY: dict[str, ContentLevel] = {
    **{k: ContentLevel.CONTENT for k in _CONTENT},
    **{k: ContentLevel.COMMAND for k in _COMMAND},
    **{k: ContentLevel.METADATA for k in _METADATA},
}

#: Keys added by redaction itself, so re-classifying a redacted event does not
#: report them as unregistered.
_ADDED_BY_REDACTION = frozenset({"redacted_chars", "redacted_items"})


def unregistered(payload: dict[str, Any]) -> list[str]:
    """Payload keys with no policy. Non-empty means someone shipped a field
    without saying what is in it."""
    return sorted(
        k for k in payload
        if k not in FIELD_POLICY and k not in _ADDED_BY_REDACTION
    )


def payload_level(payload: dict[str, Any]) -> ContentLevel:
    """The most revealing rung any present key sits on.

    Unregistered keys count as `content`: the safe reading of "we do not know
    what this is" is "it might be anything".
    """
    level = ContentLevel.METADATA
    for key, value in payload.items():
        if value is None or value == [] or value == {}:
            continue  # an absent field reveals nothing
        k = FIELD_POLICY.get(key, ContentLevel.CONTENT)
        if rank(k) > rank(level):
            level = k
    return level


def event_level(payload: dict[str, Any], *, has_native: bool) -> ContentLevel:
    """The rung for a whole event.

    `native` is the agent's own message kept verbatim for audit. Whatever else
    it holds, it holds the agent's words, so its presence pins the event at
    `content` regardless of how tame the payload looks.
    """
    if has_native:
        return ContentLevel.CONTENT
    return payload_level(payload)


# ── scrubbing ────────────────────────────────────────────────────────────────

#: Tokens recognisable by their own shape, with no surrounding label needed.
#: These run first so that in `--header "Authorization: Bearer sk-…"`, where a
#: labelled rule would also match, the more precise rule wins.
_SHAPED: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),
)

_SECRETISH = r"(?:api[_\-]?key|secret|token|password|passwd|credential|auth)"
_VALUE = r"(?P<value>\"[^\"]+\"|'[^']+'|[^\s;|&]+)"

#: Values recognisable only by what labels them. Two shapes, and the split
#: matters: a bare space is allowed to separate a *flag* from its value
#: (`--password hunter2`) but never a bare word from the next one. Allowing it
#: everywhere redacted `git commit -m 'fix the token bucket'`, which is not a
#: credential and is exactly the kind of thing a researcher needs to read.
_LABELLED: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"(?i)(?P<label>--[A-Za-z0-9_\-]*{_SECRETISH}[A-Za-z0-9_\-]*)"
        rf"(?P<sep>\s*=\s*|\s+){_VALUE}"
    ),
    re.compile(
        rf"(?i)(?P<label>\b[A-Za-z0-9_\-]*{_SECRETISH}[A-Za-z0-9_\-]*)"
        rf"(?P<sep>\s*[:=]\s*){_VALUE}"
    ),
)


def scrub_secrets(text: str) -> tuple[str, int]:
    """Replace credential-shaped substrings; return the text and a count.

    Blunt on purpose, but not indiscriminate. A false positive costs a
    researcher one unreadable argument in one command; a false negative writes
    a live key into a file that gets exported, attached to an issue, and
    indexed. The count is returned rather than left as a marker inside the
    string so the caller can record *that* scrubbing happened without the
    reader having to spot it.
    """
    if not text or REDACTED in text:
        return text, 0
    n = 0

    def _plain(m: re.Match[str]) -> str:
        nonlocal n
        n += 1
        return REDACTED

    def _labelled(m: re.Match[str]) -> str:
        nonlocal n
        if REDACTED in m.group("value"):
            return m.group(0)
        n += 1
        return f"{m.group('label')}{m.group('sep')}{REDACTED}"

    out = text
    for pat in _SHAPED:
        out = pat.sub(_plain, out)
    for pat in _LABELLED:
        out = pat.sub(_labelled, out)
    return out, n


#: Payload keys scrubbed on the way in. Commands and program output are the two
#: places a credential turns up in practice: one because the user typed it, one
#: because a tool echoed it back.
SCRUBBED_KEYS = ("command", "stderr_tail")


def scrub_payload(payload: dict[str, Any]) -> int:
    """Scrub in place. Returns how many replacements were made."""
    n = 0
    for key in SCRUBBED_KEYS:
        value = payload.get(key)
        if isinstance(value, str):
            payload[key], hits = scrub_secrets(value)
            n += hits
        elif isinstance(value, list):
            out = []
            for item in value:
                if isinstance(item, str):
                    item, hits = scrub_secrets(item)
                    n += hits
                out.append(item)
            payload[key] = out
    return n


# ── redaction ────────────────────────────────────────────────────────────────


def redact_event(d: dict[str, Any], keep: ContentLevel) -> dict[str, Any]:
    """A copy of an event dict with nothing above `keep` in it.

    Redacted strings become `None` plus their length under `redacted_chars`,
    and redacted lists become empty plus their length under `redacted_items`.
    The length is kept on purpose: "the agent wrote 4,000 characters here and
    you cannot see them" is a different fact from "the agent wrote nothing",
    and several analyses read only the length anyway.
    """
    payload = dict(d.get("payload") or {})
    removed: list[str] = []
    chars: dict[str, int] = {}
    items: dict[str, int] = {}

    for key in list(payload):
        value = payload[key]
        if value is None or value == [] or value == {}:
            continue
        if rank(FIELD_POLICY.get(key, ContentLevel.CONTENT)) <= rank(keep):
            continue
        removed.append(key)
        if isinstance(value, str):
            chars[key] = len(value)
            payload[key] = None
        elif isinstance(value, list):
            items[key] = len(value)
            payload[key] = []
        else:
            payload[key] = None

    out = dict(d)
    if rank(ContentLevel.CONTENT) > rank(keep) and "native" in out:
        removed.append("native")
        out.pop("native")
    if not removed:
        return out

    if chars:
        payload["redacted_chars"] = chars
    if items:
        payload["redacted_items"] = items
    out["payload"] = payload
    out["privacy"] = {
        **(d.get("privacy") or {}),
        "content_level": keep.value,
        "redacted": sorted(removed),
        "ruleset": RULESET,
    }
    return out


#: The derived view is not events, so `FIELD_POLICY` does not reach it. These
#: are the places a reduced run still holds text that came from a person or a
#: model: a span's `detail` is the command that opened it, and an annotation is
#: prose by definition. Listed explicitly so the CSV and analysis exports
#: cannot quietly re-publish what the JSONL export redacted.
VIEW_TEXT_FIELDS: tuple[tuple[str, str, ContentLevel], ...] = (
    ("spans", "detail", ContentLevel.COMMAND),
    ("annotations", "text", ContentLevel.CONTENT),
)


def redact_view_dict(d: dict[str, Any], keep: ContentLevel) -> dict[str, Any]:
    """A copy of `RunView.to_dict()` with its text taken down to `keep`."""
    out = dict(d)
    for collection, field_name, level in VIEW_TEXT_FIELDS:
        if rank(level) <= rank(keep):
            continue
        rows = out.get(collection)
        if not isinstance(rows, list):
            continue
        out[collection] = [
            {**r, field_name: None, f"{field_name}_chars": len(r[field_name])}
            if isinstance(r, dict) and isinstance(r.get(field_name), str)
            else r
            for r in rows
        ]
    return out


def parse_level(value: str) -> ContentLevel:
    """Accept a level name, refusing anything else by listing the options."""
    try:
        return ContentLevel(value)
    except ValueError:
        raise ValueError(
            f"unknown content level {value!r}; expected one of "
            + ", ".join(l.value for l in ContentLevel)
        ) from None


__all__ = [
    "ContentLevel",
    "FIELD_POLICY",
    "REDACTED",
    "RULESET",
    "SCRUBBED_KEYS",
    "VIEW_TEXT_FIELDS",
    "event_level",
    "parse_level",
    "payload_level",
    "rank",
    "redact_event",
    "redact_view_dict",
    "scrub_payload",
    "scrub_secrets",
    "unregistered",
]
