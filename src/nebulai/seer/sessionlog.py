"""`~/.claude/projects/**/*.jsonl` → *who put this word in front of the agent?*

Every other part of SessionSeer measures what an agent **did** — spans, tokens,
tools, outcomes. This module measures what the agent was **told**, because the
commonest cause of a repeated mistake is not the model: it is a sentence that
sits in the context window every single turn, and that nobody remembers writing.

The instrument is deliberately narrow. Give it a word; it reports every place
that word entered the context, attributed to one of five origins:

    human      — the operator typed it
    standing   — the operator's own persistent files, re-injected by the
                 harness each session (CLAUDE.md, an invoked skill's body)
    harness    — the harness's own boilerplate (system prompt, tool schemas,
                 skill/agent listings, MCP instructions, hook output)
    model      — the agent's own words (prose, thinking, tool arguments)
    environment— what the agent read off this machine (tool results, stdout,
                 diffs, attached files)

That five-way split *is* the diagnosis. A term whose hits are `standing` has a
rule behind it. A term whose hits are `harness` is an artefact of the tooling
and means nothing about the task. A term whose hits are `model` was the agent's
own invention, perpetuated by its own transcript.

WHY THIS IS NOT A GREP — three measured traps, each of which makes a naive
count wrong by more than it is right:

1. **Every attachment stores its text twice.** The payload appears under
   `attachment.*` and again as the harness's rendered copy under
   `rendered[].content`. On one real session, 29 hits in `rendered[].content`
   and 28 in `attachment.content` were the *same 28 sentences*. Counting both
   roughly doubles every injected channel, so `rendered[]` is suppressed by
   default — the same "fold, don't sum" rule `viewer/src/chrome/sessionlog.ts`
   learned when per-line `usage` overcounted a session by 3.5×.

2. **Most bytes are not content.** `cwd` repeats the project path on all ~38k
   message lines, `signature` and `source.data` are base64, and
   `total_tokens_reminder` fires ~12k times. Searching "gesture" against a
   transcript in `~/Developer/gesture_sorcery_game_kit` would score a hit on
   every line from `cwd` alone. Those channels are classified and suppressed by
   name, never silently skipped: `channels_suppressed` reports what they held.

3. **Words are polysemous and substrings lie.** Measured on one 131 MB session,
   a substring search for `face` returned 1236 hits of which the plurality were
   *prism faces*, *face-normals* and *Hugging Face* — the human face was a
   minority. So matching is word-boundary by default, the distinct surface forms
   are always reported, and the compounds the word boundary *rejected* are
   reported too, so the caller can see the noise the tool spared them.

NOTHING IS DROPPED SILENTLY. A suppressed channel reports its hit count. An
`--exclude` rule reports how many it removed. An occurrence matching no declared
sense is `unclassified`, not discarded. The one thing this module will not do is
tell you a term "persisted in context for N turns": a transcript records each
injection as a line, and how long a block stayed in the window afterwards is not
in the file. It reports injections, which are real, and says so.

FORMAT SCOPE: Claude Code's interactive session log. Codex and Hermes write
neither this shape nor this provenance, so they are out of scope here rather
than badly approximated — `seer/adapters/` is where their formats live.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

#: Where Claude Code keeps one directory per project, each holding
#: `<session-id>.jsonl` plus a `<session-id>/` sidecar with the custom title.
DEFAULT_PROJECTS_ROOT = Path.home() / ".claude" / "projects"

#: Characters either side of a match kept in a sample snippet.
DEFAULT_WINDOW = 120

#: Suffixes that make a rejected compound an inflection of the term rather
#: than an unrelated word. `faces` is not `face`, and the word boundary is right
#: to reject it — but the caller almost certainly wanted it, so the report says
#: so instead of leaving 609 hits to be discovered by accident.
_INFLECTIONS = ("s", "es", "d", "ed", "ing", "ly")


def _inflections_of(term: str) -> tuple[str, ...]:
    """The spellings of `term` that a word boundary correctly rejects.

    Only suffixed forms are listed, because only those contain the term as a
    substring and so are the only ones the compound tally can see at all.
    English also drops a final `e` before a vowel suffix — `facing`, `faring` —
    and such a spelling never reaches this scan: the prefilter is built from
    the term, and a line whose only match is `facing` is not read. That silence
    is stated in the report rather than papered over; to count those forms the
    caller has to name them in `--regex`.
    """
    t = term.lower()
    return tuple(
        dict.fromkeys(t + suffix for suffix in _INFLECTIONS if suffix)
    )

#: A rejected "compound" longer than this is not a word: on real transcripts
#: the long ones are base64 payloads and signatures that happen to contain the
#: term's letters. They are still counted — a grep would have matched them —
#: but as one lump, so they cannot crowd the real neighbours out of the list.
MAX_WORDLIKE = 40

#: Samples retained per (channel, sense) pair. The report is meant to be read,
#: and a scan of a gigabyte can match tens of thousands of times.
DEFAULT_SAMPLES_PER_CHANNEL = 4


class Origin(str, Enum):
    """Who is responsible for the words being in the context window."""

    HUMAN = "human"
    STANDING = "standing"
    HARNESS = "harness"
    MODEL = "model"
    ENVIRONMENT = "environment"
    #: Bookkeeping the harness writes about the session rather than to the
    #: model: paths, titles, ids, base64, duplicate renderings. Never counted
    #: by default, always reported as suppressed.
    METADATA = "metadata"


@dataclass(frozen=True)
class Channel:
    id: str
    label: str
    origin: Origin
    why: str
    #: Counted unless the caller asks otherwise. False means "real text, but
    #: counting it would double-count or drown the signal".
    default: bool = True


def _c(*args: Any, **kw: Any) -> Channel:
    return Channel(*args, **kw)


#: Every channel this module can attribute a string to. Adding a transcript
#: field means adding a rule below and a channel here — an unclassified field
#: lands in `unclassified` and is reported, so the failure mode is a visible
#: gap rather than a silent miscount.
CHANNELS: dict[str, Channel] = {ch.id: ch for ch in (
    # ── human ────────────────────────────────────────────────────────────────
    _c("human.prompt", "your prompt", Origin.HUMAN,
       "text you typed as a turn"),
    _c("human.queued", "your queued message", Origin.HUMAN,
       "a message you queued while the agent was working"),
    _c("human.prompt_echo", "your prompt (second copy)", Origin.HUMAN,
       "`last-prompt` re-records the prompt already counted as human.prompt",
       default=False),

    # ── standing instructions ────────────────────────────────────────────────
    _c("standing.project_instructions", "CLAUDE.md", Origin.STANDING,
       "a CLAUDE.md the harness injects at the top of every session"),
    _c("standing.skill_body", "invoked skill body", Origin.STANDING,
       "the full text of a skill that was invoked"),
    _c("standing.session_context", "session context block", Origin.STANDING,
       "git status, your email, and other per-session context"),

    # ── harness boilerplate ──────────────────────────────────────────────────
    _c("harness.system_prompt", "system prompt", Origin.HARNESS,
       "the harness's own system prompt, re-snapshotted per session"),
    _c("harness.tool_schema", "tool descriptions", Origin.HARNESS,
       "tool names, descriptions and JSON schemas offered to the model"),
    _c("harness.skill_listing", "skill listing", Origin.HARNESS,
       "the one-line descriptions of every available skill"),
    _c("harness.agent_listing", "agent listing", Origin.HARNESS,
       "the available subagent types and their descriptions"),
    _c("harness.mcp_instructions", "MCP server instructions", Origin.HARNESS,
       "instructions supplied by connected MCP servers"),
    _c("harness.hook_output", "hook output", Origin.HARNESS,
       "stdout and added context from your configured hooks"),
    _c("harness.environment", "environment block", Origin.HARNESS,
       "working directory, platform, shell and scratchpad notes"),
    _c("harness.reminder", "harness reminder", Origin.HARNESS,
       "token budget, batching and mode reminders — thousands per session",
       default=False),

    # ── the model's own words ────────────────────────────────────────────────
    _c("model.text", "assistant prose", Origin.MODEL,
       "what the agent said to you"),
    _c("model.thinking", "assistant thinking", Origin.MODEL,
       "the agent's reasoning blocks, where present in the log"),
    _c("model.tool_input", "tool arguments the agent wrote", Origin.MODEL,
       "commands, prompts, edit strings and file paths the agent chose"),

    # ── what the machine told it ─────────────────────────────────────────────
    _c("env.tool_result", "tool result", Origin.ENVIRONMENT,
       "content returned to the agent by a tool"),
    _c("env.stdout", "command output", Origin.ENVIRONMENT,
       "stdout and stderr of commands the agent ran"),
    _c("env.diff", "diff / edited text", Origin.ENVIRONMENT,
       "patch hunks and before/after strings of file edits"),
    _c("env.file_attachment", "attached file contents", Origin.ENVIRONMENT,
       "a file the harness attached, or an edited file's snippet"),
    _c("env.file_path", "file paths", Origin.ENVIRONMENT,
       "paths of files read, written or referenced — a name, not content",
       default=False),

    # ── bookkeeping ──────────────────────────────────────────────────────────
    _c("meta.rendered_copy", "rendered duplicate", Origin.METADATA,
       "the harness's rendered copy of an attachment already counted above",
       default=False),
    _c("meta.wire_copy", "on-the-wire tool input copy", Origin.METADATA,
       "`wireToolInputs` repeats the tool arguments already counted as "
       "model.tool_input — a third duplicate, found by this module's own "
       "`unclassified` tally on 47 real transcripts", default=False),
    _c("meta.cwd", "working directory", Origin.METADATA,
       "the project path, repeated on every message line", default=False),
    _c("meta.opaque", "base64 / schema", Origin.METADATA,
       "thinking signatures, image data and JSON-schema plumbing", default=False),
    _c("meta.title", "session title", Origin.METADATA,
       "the session's custom or generated title", default=False),
    _c("meta.other", "other bookkeeping", Origin.METADATA,
       "ids, latches, ledgers, task status and permission records", default=False),
    _c("unclassified", "unclassified field", Origin.METADATA,
       "a transcript field this module has no rule for yet", default=False),
)}


@dataclass(frozen=True)
class _Rule:
    """One classification rule. First match in `_RULES` order wins.

    `line_types` / `attachments` are membership tests (empty = any). `suffix`
    and `exact` test the json path produced by `_walk`, e.g.
    `.message.content[].input.command`.
    """

    channel: str
    line_types: tuple[str, ...] = ()
    attachments: tuple[str, ...] = ()
    suffix: tuple[str, ...] = ()
    exact: tuple[str, ...] = ()
    contains: tuple[str, ...] = ()

    def matches(self, line_type: str, attachment: str | None, path: str) -> bool:
        if self.line_types and line_type not in self.line_types:
            return False
        if self.attachments and (attachment or "") not in self.attachments:
            return False
        if self.exact and path not in self.exact:
            return False
        if self.suffix and not path.endswith(self.suffix):
            return False
        if self.contains and not any(c in path for c in self.contains):
            return False
        return True


#: ORDER IS THE CONTRACT. The bookkeeping rules come first so that a `cwd`, a
#: base64 signature or a rendered duplicate can never be counted as content by
#: a later, broader rule.
_RULES: tuple[_Rule, ...] = (
    # ── bookkeeping first, so nothing below can double-count ─────────────────
    _Rule("meta.cwd", suffix=(".cwd",)),
    _Rule("meta.opaque", suffix=(".signature", ".base64", ".data", ".$schema",
                                ".thinkingSignature")),
    _Rule("meta.rendered_copy", contains=(".rendered[]",)),
    _Rule("meta.wire_copy", contains=(".wireToolInputs.",)),
    _Rule("meta.title", line_types=("custom-title", "ai-title")),
    _Rule("meta.other", line_types=(
        "atis-latch", "bridge-session", "frame-link", "cost-state",
        "file-history-snapshot", "file-history-delta", "permission-mode",
        "agent-setting", "agent-name", "mode",
        "artifact-autoreact-ledger", "artifact-comment-monitor",
    )),
    _Rule("meta.other", suffix=(".classifierMetaLines", ".uuid", ".requestId",
                                ".toolUseID", ".id")),

    # ── human ────────────────────────────────────────────────────────────────
    _Rule("human.prompt_echo", line_types=("last-prompt",)),
    _Rule("human.queued", line_types=("queue-operation",)),
    _Rule("human.prompt", line_types=("user-prompt",)),

    # ── standing instructions ────────────────────────────────────────────────
    _Rule("env.file_path", attachments=("instructions",), suffix=(".path",)),
    _Rule("standing.project_instructions", attachments=("instructions",)),
    _Rule("standing.skill_body", attachments=("invoked_skills",)),
    _Rule("standing.session_context", attachments=("session_context",)),

    # ── harness ──────────────────────────────────────────────────────────────
    _Rule("harness.system_prompt", attachments=("prompt_snapshot",),
          contains=(".systemPrompt",)),
    _Rule("harness.tool_schema", attachments=("prompt_snapshot",),
          contains=(".tools[]",)),
    _Rule("harness.tool_schema", attachments=("deferred_tools_delta",
                                              "deferred_tools_record")),
    _Rule("harness.skill_listing", attachments=("skill_listing",)),
    _Rule("harness.agent_listing", attachments=("agent_listing_delta",)),
    _Rule("harness.mcp_instructions", attachments=("mcp_instructions_delta",)),
    _Rule("harness.hook_output", attachments=("hook_success",
                                              "hook_additional_context")),
    _Rule("harness.hook_output", line_types=("system",)),
    _Rule("harness.environment", attachments=("environment",)),
    _Rule("harness.reminder", attachments=(
        "total_tokens_reminder", "batching_reminder_sent",
        "silent_turn_reminder", "task_reminder", "thinking_stripped",
        "thinking_drop", "auto_mode", "date", "date_change", "model",
        "remote_session_change", "ultra_effort_enter", "plan_mode",
        "plan_mode_exit", "task_status", "command_permissions", "directory",
    )),
    _Rule("human.queued", attachments=("queued_command",)),

    # ── what the machine told it ─────────────────────────────────────────────
    _Rule("env.file_path", attachments=("compact_file_reference",
                                        "plan_file_reference")),
    _Rule("env.file_path", attachments=("file", "edited_text_file"),
          suffix=(".filename", ".filePath", ".displayPath")),
    _Rule("env.file_attachment", attachments=("file", "edited_text_file")),

    # ── the model's own words ────────────────────────────────────────────────
    _Rule("model.text", line_types=("assistant",), suffix=(".text",)),
    _Rule("model.thinking", line_types=("assistant",),
          suffix=(".thinking", ".reasoning")),
    _Rule("model.tool_input", line_types=("assistant",), contains=(".input",)),
    _Rule("model.text", line_types=("assistant",), contains=(".content",)),

    # ── tool results (a `user` line the harness wrote, not the operator) ─────
    _Rule("env.stdout", line_types=("user-result",),
          suffix=(".stdout", ".stderr")),
    _Rule("env.diff", line_types=("user-result",), contains=(
        ".structuredPatch", ".bashEditDiff", ".newString", ".oldString",
        ".originalFile", ".edits[]")),
    _Rule("env.file_path", line_types=("user-result",), suffix=(
        ".filePath", ".file_path", ".filename", ".path")),
    _Rule("env.tool_result", line_types=("user-result",)),
)


def classify(line_type: str, attachment: str | None, path: str) -> str:
    for rule in _RULES:
        if rule.matches(line_type, attachment, path):
            return rule.channel
    return "unclassified"


# ── reading the log ──────────────────────────────────────────────────────────


def _walk(obj: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Every string in a decoded line, with the json path that holds it."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, f"{path}[]")
    elif isinstance(obj, str):
        yield path, obj


def refine_line_type(obj: dict[str, Any]) -> str:
    """Split `user` into who actually wrote it.

    A Claude Code transcript files both the operator's turns and every tool
    result under `type: "user"`. Attributing a tool result to the operator
    would put the machine's words in their mouth — which is the one error this
    whole module exists to avoid — so the discrimination happens here, once.
    """
    t = obj.get("type") or ""
    if t != "user":
        return t
    if obj.get("toolUseResult") is not None:
        return "user-result"
    if obj.get("isMeta"):
        return "user-result"
    content = (obj.get("message") or {}).get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return "user-result"
    return "user-prompt"


@dataclass
class SessionRef:
    """One transcript on disk, with whatever name it actually has."""

    path: Path
    session_id: str
    title: str | None
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "path": str(self.path),
            "bytes": self.bytes,
        }


def session_title(path: Path) -> str | None:
    """The session's custom title, from the sidecar directory beside the log.

    Returns None rather than a placeholder: most sessions genuinely have no
    title, and inventing one from the first prompt would make two sessions look
    like the same one.
    """
    sidecar = path.with_suffix("") / "custom-title.json"
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text()).get("customTitle") or None
        except (json.JSONDecodeError, OSError):
            return None
    return None


def discover_sessions(project_dir: Path) -> list[SessionRef]:
    """Every transcript in a project directory, largest first."""
    refs = []
    for p in sorted(project_dir.glob("*.jsonl")):
        refs.append(SessionRef(p, p.stem, session_title(p), p.stat().st_size))
    refs.sort(key=lambda r: -r.bytes)
    return refs


def list_projects(root: Path | None = None) -> list[dict[str, Any]]:
    """Every project directory that holds at least one transcript, largest first.

    A directory with no `.jsonl` in it is left out rather than listed at zero:
    it is a project Claude Code has a folder for and no session in, and showing
    it as scannable would offer a scan that can only return nothing.
    """
    base = Path(root) if root else DEFAULT_PROJECTS_ROOT
    if not base.is_dir():
        raise SessionLogError(f"no projects directory at {base}")
    out: list[dict[str, Any]] = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        refs = discover_sessions(d)
        if not refs:
            continue
        out.append({
            "slug": d.name,
            "path": str(d),
            "sessions": len(refs),
            "bytes": sum(r.bytes for r in refs),
            "titled": sum(1 for r in refs if r.title),
        })
    out.sort(key=lambda r: -r["bytes"])
    return out


@dataclass
class ProjectRef:
    slug: str
    path: Path
    #: How the caller's string was turned into this directory, so an ambiguous
    #: match is visible in the report rather than guessed at silently.
    matched_how: str


def resolve_project(name: str, root: Path | None = None) -> ProjectRef:
    """A project name, path or directory slug → the transcript directory.

    Claude Code encodes a working directory into a slug by replacing `/`, `_`
    and `.` with `-`, so `~/Developer/gesture_sorcery_game_kit` is stored as
    `-Users-…-Developer-gesture-sorcery-game-kit`. That mapping is lossy and
    not invertible, so this resolves forwards: encode what the caller gave us,
    and fall back to a unique substring match on the existing directories.
    """
    root = Path(root) if root else DEFAULT_PROJECTS_ROOT
    if not root.is_dir():
        raise SessionLogError(f"no projects directory at {root}")
    dirs = [d for d in root.iterdir() if d.is_dir()]

    exact = root / name
    if exact.is_dir():
        return ProjectRef(name, exact, "directory name")

    encoded = re.sub(r"[/_.]", "-", str(Path(name).expanduser()))
    hit = next((d for d in dirs if d.name == encoded), None)
    if hit:
        return ProjectRef(hit.name, hit, "encoded working directory")

    needle = re.sub(r"[/_.\s]", "-", name).strip("-").lower()
    near = [d for d in dirs if needle and needle in d.name.lower()]
    if len(near) == 1:
        return ProjectRef(near[0].name, near[0], f"unique substring match on {needle!r}")
    # `gesture_sorcery_game_kit` is a substring of both that project's slug and
    # its `-app` sibling's. A slug that *ends* with the needle is the project
    # itself rather than a longer path beneath it, so prefer that — but only
    # when exactly one does, so the refusal below still catches real ambiguity.
    tail = [d for d in near if d.name.lower().endswith(needle)]
    if len(tail) == 1:
        return ProjectRef(tail[0].name, tail[0],
                          f"slug ends with {needle!r} ({len(near)} matched loosely)")
    if len(near) > 1:
        names = "\n  ".join(sorted(d.name for d in near))
        raise SessionLogError(
            f"{len(near)} project directories match {name!r}; name one exactly:\n  {names}"
        )
    raise SessionLogError(f"no project directory matches {name!r} under {root}")


class SessionLogError(RuntimeError):
    """A scan could not be set up. Never raised for "no hits"."""


# ── the scan ─────────────────────────────────────────────────────────────────


@dataclass
class Occurrence:
    session_id: str
    title: str | None
    line_no: int
    channel: str
    origin: Origin
    form: str
    path: str
    snippet: str
    sense: str = "unclassified"
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "line": self.line_no,
            "channel": self.channel,
            "origin": self.origin.value,
            "form": self.form,
            "json_path": self.path,
            "sense": self.sense,
            "timestamp": self.timestamp,
            "snippet": self.snippet,
        }


@dataclass
class Sense:
    """A named reading of the term, declared by the caller as a regex.

    Senses are the caller's hypothesis, not a fact the transcript states, so
    every count derived from them is reported as `heuristic` and anything
    matching no sense stays visible as `unclassified`.
    """

    name: str
    pattern: re.Pattern[str]


def build_pattern(term: str, *, regex: bool = False, case_sensitive: bool = False
                  ) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """The match pattern and the wider compound pattern used to expose noise.

    The compound pattern is what a naive `grep` would have counted. Reporting
    the difference between the two is how the caller learns that 21 of their
    hits were "Hugging Face".
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    if regex:
        return re.compile(term, flags), re.compile(rf"\w*(?:{term})\w*", flags)
    esc = re.escape(term)
    return re.compile(rf"\b{esc}\b", flags), re.compile(rf"\w*{esc}\w*", flags)


@dataclass
class ScanStats:
    sessions: int = 0
    bytes_read: int = 0
    lines: int = 0
    lines_prefiltered: int = 0
    lines_unparsable: int = 0
    #: Occurrences dropped because the same line recorded the same text
    #: under a second json path. See `_fold_duplicate_fields`.
    duplicate_fields_folded: int = 0
    elapsed_s: float = 0.0


@dataclass
class KeywordReport:
    term: str
    pattern: str
    case_sensitive: bool
    project: ProjectRef | None
    stats: ScanStats = field(default_factory=ScanStats)
    occurrences: list[Occurrence] = field(default_factory=list)
    by_channel: dict[str, int] = field(default_factory=dict)
    by_origin: dict[str, int] = field(default_factory=dict)
    by_sense: dict[str, int] = field(default_factory=dict)
    suppressed: dict[str, int] = field(default_factory=dict)
    forms: dict[str, int] = field(default_factory=dict)
    compounds_rejected: dict[str, int] = field(default_factory=dict)
    #: The subset of `compounds_rejected` that are inflections of the term.
    inflections_rejected: dict[str, int] = field(default_factory=dict)
    #: Rejected matches too long to be a word: (distinct forms, occurrences).
    unwordlike: tuple[int, int] = (0, 0)
    excluded: dict[str, int] = field(default_factory=dict)
    sessions: list[dict[str, Any]] = field(default_factory=list)
    samples: list[Occurrence] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.by_channel.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "seer.keyword/1",
            "term": self.term,
            "pattern": self.pattern,
            "case_sensitive": self.case_sensitive,
            "project": (
                {"slug": self.project.slug, "path": str(self.project.path),
                 "matched_how": self.project.matched_how}
                if self.project else None
            ),
            "fidelity": {
                "counts": "deterministic",
                "channel_attribution": "deterministic",
                "sense_classification": "heuristic",
                "context_persistence": "missing",
                "context_persistence_note": (
                    "a transcript records each injection as a line; how long a "
                    "block then stayed in the context window is not in the file"
                ),
            },
            "scanned": {
                "sessions": self.stats.sessions,
                "bytes": self.stats.bytes_read,
                "lines": self.stats.lines,
                "lines_examined": self.stats.lines_prefiltered,
                "lines_unparsable": self.stats.lines_unparsable,
                "duplicate_fields_folded": self.stats.duplicate_fields_folded,
                "elapsed_s": round(self.stats.elapsed_s, 2),
            },
            "totals": {
                "counted": self.total,
                "suppressed": sum(self.suppressed.values()),
                "excluded": sum(self.excluded.values()),
                "sessions_with_hits": sum(
                    1 for s in self.sessions if s["occurrences"]),
            },
            "by_origin": self.by_origin,
            "by_channel": self.by_channel,
            "by_sense": self.by_sense,
            "suppressed_channels": [
                {"channel": cid, "label": CHANNELS[cid].label,
                 "why": CHANNELS[cid].why, "hits": n}
                for cid, n in sorted(self.suppressed.items(), key=lambda kv: -kv[1])
            ],
            "surface_forms": self.forms,
            "compounds_rejected_by_word_boundary": self.compounds_rejected,
            "compounds_unwordlike": {
                "forms": self.unwordlike[0],
                "hits": self.unwordlike[1],
                "note": (
                    f"rejected matches longer than {MAX_WORDLIKE} characters — "
                    "base64 payloads and signatures whose letters happen to "
                    "contain the term. Counted here rather than listed above, "
                    "where they would push out the real neighbouring words."
                ),
            },
            "compounds_note": (
                "what a substring grep over the raw file would have matched, "
                "across all channels including suppressed ones — a grep has no "
                "channels"
            ),
            "inflections_rejected": self.inflections_rejected,
            "inflections_note": (
                "suffixed spellings of the term itself, correctly rejected by "
                "the word boundary and probably wanted anyway. Spellings that "
                "drop a letter of the term (face → facing) are NOT in this "
                "tally and were never read: the prefilter is built from the "
                "term, so a line whose only match is such a form is skipped. "
                "Name them in --regex to count them."
            ),
            "excluded_by_rule": self.excluded,
            "sessions": self.sessions,
            "samples": [o.to_dict() for o in self.samples],
        }


def scan_session(
    ref: SessionRef,
    pattern: re.Pattern[str],
    compound: re.Pattern[str],
    *,
    enabled: set[str],
    senses: Iterable[Sense] = (),
    excludes: Iterable[tuple[str, re.Pattern[str]]] = (),
    window: int = DEFAULT_WINDOW,
    report: KeywordReport | None = None,
) -> tuple[list[Occurrence], dict[str, int], dict[str, int], dict[str, int], dict[str, int], ScanStats]:
    """One transcript. Returns (counted, suppressed, forms, compounds, excluded, stats).

    The raw line is prefiltered with the *compound* pattern before any JSON is
    decoded. On a gigabyte of transcripts fewer than two lines in a hundred
    survive that test, which is the difference between a scan that takes
    seconds and one that takes minutes. The prefilter is safe because JSON
    escapes structure, not ASCII letters — a matching word cannot hide from it.
    """
    senses = list(senses)
    excludes = list(excludes)
    counted: list[Occurrence] = []
    suppressed: dict[str, int] = {}
    forms: dict[str, int] = {}
    compounds: dict[str, int] = {}
    excluded: dict[str, int] = {}
    stats = ScanStats(sessions=1, bytes_read=ref.bytes)

    with ref.path.open(errors="replace") as fh:
        for line_no, raw in enumerate(fh):
            stats.lines += 1
            if not compound.search(raw):
                continue
            stats.lines_prefiltered += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                stats.lines_unparsable += 1
                continue
            if not isinstance(obj, dict):
                stats.lines_unparsable += 1
                continue

            line_type = refine_line_type(obj)
            attachment = None
            if obj.get("type") == "attachment":
                attachment = (obj.get("attachment") or {}).get("type")
            ts = obj.get("timestamp") if isinstance(obj.get("timestamp"), str) else None

            candidates: list[tuple[str, Occurrence]] = []
            for path, text in _walk(obj):
                # Record every compound the word boundary will reject, so the
                # caller can see the noise a plain grep would have counted.
                for m in compound.finditer(text):
                    if not pattern.fullmatch(m.group(0)):
                        key = m.group(0).lower()
                        compounds[key] = compounds.get(key, 0) + 1

                for m in pattern.finditer(text):
                    channel = classify(line_type, attachment, path)
                    if channel not in enabled:
                        suppressed[channel] = suppressed.get(channel, 0) + 1
                        continue
                    lo = max(0, m.start() - window)
                    hi = min(len(text), m.end() + window)
                    snippet = " ".join(text[lo:hi].split())

                    dropped = None
                    for name, ex in excludes:
                        if ex.search(snippet):
                            dropped = name
                            break
                    if dropped is not None:
                        excluded[dropped] = excluded.get(dropped, 0) + 1
                        continue

                    sense = "unclassified"
                    for s in senses:
                        if s.pattern.search(snippet):
                            sense = s.name
                            break

                    candidates.append((path, Occurrence(
                        session_id=ref.session_id, title=ref.title,
                        line_no=line_no, channel=channel,
                        origin=CHANNELS[channel].origin, form=m.group(0).lower(),
                        path=path, snippet=snippet, sense=sense, timestamp=ts,
                    )))

            kept, folded = _fold_duplicate_fields(candidates)
            stats.duplicate_fields_folded += folded
            for occ in kept:
                forms[occ.form] = forms.get(occ.form, 0) + 1
            counted.extend(kept)
    return counted, suppressed, forms, compounds, excluded, stats


#: When one log line records the same sentence under two different json paths,
#: this is which channel's account of it we keep. The model-facing record wins:
#: `message.content[].content` is literally the block the agent was handed,
#: while `toolUseResult.*` is the harness's structured copy of the same text.
_FOLD_PRIORITY = (
    "env.tool_result", "env.stdout", "env.diff", "env.file_attachment",
    "standing.project_instructions", "standing.skill_body",
    "harness.system_prompt", "harness.skill_listing",
)


def _fold_duplicate_fields(
    candidates: list[tuple[str, Occurrence]]
) -> tuple[list[Occurrence], int]:
    """Fold one line's occurrences that are the same text recorded twice.

    Measured on a real session: a tool-result line stores its output both as
    the content block the model received and again under `toolUseResult.stdout`,
    so a single sentence was counted twice — the same double-count that
    `rendered[]` causes for attachments, arriving by a different route.

    The test is deliberately narrow: identical snippet AND different json path.
    Identical snippets from the *same* path are genuine repetition and are
    kept. The residual error is a line that repeats one sentence verbatim in
    two places, which folds to one; undercounting that by one is the cheaper
    mistake than doubling every tool result in the corpus.
    """
    best: dict[str, tuple[int, str, Occurrence]] = {}
    keep_same_path: list[Occurrence] = []
    folded = 0
    for path, occ in candidates:
        key = occ.snippet
        rank = (_FOLD_PRIORITY.index(occ.channel)
                if occ.channel in _FOLD_PRIORITY else len(_FOLD_PRIORITY))
        prior = best.get(key)
        if prior is None:
            best[key] = (rank, path, occ)
            continue
        if prior[1] == path:
            keep_same_path.append(occ)
            continue
        folded += 1
        if rank < prior[0]:
            best[key] = (rank, path, occ)
    return [v[2] for v in best.values()] + keep_same_path, folded


def _bump(dst: dict[str, int], src: dict[str, int]) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v


def enabled_channels(*, include: Iterable[str] = (), only: Iterable[str] = ()
                     ) -> set[str]:
    """The channels a scan will count.

    `only` replaces the default set outright; `include` adds to it. Both are
    checked against `CHANNELS`, because a typo'd channel name that silently
    counted nothing would look exactly like a term that genuinely never
    appeared there.
    """
    only, include = list(only), list(include)
    for name in (*only, *include):
        if name not in CHANNELS:
            known = ", ".join(sorted(CHANNELS))
            raise SessionLogError(f"unknown channel {name!r}. Known: {known}")
    if only:
        return set(only)
    return {cid for cid, ch in CHANNELS.items() if ch.default} | set(include)


def scan(
    term: str,
    refs: list[SessionRef],
    *,
    project: ProjectRef | None = None,
    regex: bool = False,
    case_sensitive: bool = False,
    channels: set[str] | None = None,
    senses: Iterable[Sense] = (),
    excludes: Iterable[tuple[str, re.Pattern[str]]] = (),
    window: int = DEFAULT_WINDOW,
    samples_per_channel: int = DEFAULT_SAMPLES_PER_CHANNEL,
    on_progress: Callable[[int, SessionRef, ScanStats], None] | None = None,
) -> KeywordReport:
    """Scan every transcript in `refs` for `term`, attributed by channel.

    `on_progress` is called after each session with (sessions finished, the
    session just read, the running totals). A gigabyte of transcripts takes
    minutes, and a caller that cannot say how far along it is has to either
    lie about progress or show nothing.
    """
    pattern, compound = build_pattern(term, regex=regex, case_sensitive=case_sensitive)
    enabled = channels if channels is not None else enabled_channels()
    senses, excludes = list(senses), list(excludes)
    rep = KeywordReport(term=term, pattern=pattern.pattern,
                        case_sensitive=case_sensitive, project=project)
    started = time.monotonic()

    for ref in refs:
        occ, sup, forms, comps, excl, st = scan_session(
            ref, pattern, compound, enabled=enabled, senses=senses,
            excludes=excludes, window=window)
        rep.occurrences.extend(occ)
        _bump(rep.suppressed, sup)
        _bump(rep.forms, forms)
        _bump(rep.compounds_rejected, comps)
        _bump(rep.excluded, excl)
        rep.stats.sessions += st.sessions
        rep.stats.bytes_read += st.bytes_read
        rep.stats.lines += st.lines
        rep.stats.lines_prefiltered += st.lines_prefiltered
        rep.stats.lines_unparsable += st.lines_unparsable
        rep.stats.duplicate_fields_folded += st.duplicate_fields_folded

        per_origin: dict[str, int] = {}
        for o in occ:
            per_origin[o.origin.value] = per_origin.get(o.origin.value, 0) + 1
        rep.sessions.append({
            **ref.to_dict(),
            "occurrences": len(occ),
            "by_origin": per_origin,
            "first_line": min((o.line_no for o in occ), default=None),
            "last_line": max((o.line_no for o in occ), default=None),
        })
        if on_progress is not None:
            on_progress(len(rep.sessions), ref, rep.stats)

    for o in rep.occurrences:
        rep.by_channel[o.channel] = rep.by_channel.get(o.channel, 0) + 1
        rep.by_origin[o.origin.value] = rep.by_origin.get(o.origin.value, 0) + 1
        rep.by_sense[o.sense] = rep.by_sense.get(o.sense, 0) + 1

    # Samples are per (channel, sense) so a rare-but-decisive channel is never
    # crowded out of the report by a noisy one.
    quota: dict[tuple[str, str], int] = {}
    for o in rep.occurrences:
        key = (o.channel, o.sense)
        if quota.get(key, 0) < samples_per_channel:
            quota[key] = quota.get(key, 0) + 1
            rep.samples.append(o)

    rep.sessions.sort(key=lambda s: -s["occurrences"])
    rep.forms = dict(sorted(rep.forms.items(), key=lambda kv: -kv[1]))
    long_ones = {k: v for k, v in rep.compounds_rejected.items()
                 if len(k) > MAX_WORDLIKE}
    rep.unwordlike = (len(long_ones), sum(long_ones.values()))
    rep.compounds_rejected = dict(sorted(
        ((k, v) for k, v in rep.compounds_rejected.items()
         if len(k) <= MAX_WORDLIKE),
        key=lambda kv: -kv[1])[:40])
    # A caller-supplied pattern already says what it wants; guessing at the
    # inflections of a regular expression would be noise, not help.
    forms = () if regex else _inflections_of(term)
    rep.inflections_rejected = {
        k: v for k, v in rep.compounds_rejected.items() if k in forms
    }
    rep.by_channel = dict(sorted(rep.by_channel.items(), key=lambda kv: -kv[1]))
    rep.stats.elapsed_s = time.monotonic() - started
    return rep
