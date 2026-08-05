"""RECONCILED capture: sessions that already happened, read back after the fact.

Attached and observed capture both require us to be there at the time. Most of a
researcher's history is not like that — it is the four hundred Codex threads
already sitting on the disk, run before SessionSeer existed. This module imports
them, and it is the mode with the most ways to lie, so it is the mode with the
most explicit refusals.

**Where the numbers come from, and where they do not.**

`thread/list` and `thread/read` are read-only requests to a `codex app-server`,
and they return the same `ThreadItem` objects the live notifications carry. So
the replay goes through `CodexAppServerAdapter`'s own route table rather than a
second mapping written from the same protocol — a reconciled run and an attached
run of the same session describe it in the same vocabulary because they are
literally the same code. What the history does *not* carry is:

* **per-item timestamps.** A turn has `startedAt`, `completedAt` and
  `durationMs`; the items inside it have none. Every item is therefore stamped
  at its turn's start, which preserves order and refuses to invent an interval:
  the resulting spans report `duration_fidelity: missing`, not `0.0s`.
* **token usage.** No field on any turn or item carries it.

The usage is recoverable, but from the rollout file rather than the protocol —
`thread/list` hands us its `path`, and its `token_count` records carry Codex's
own cumulative counters. We read those, and only those: the rollout also holds
every message in full, which is a larger claim on someone's data than the item
mapping already makes. When the file is missing or has no counters we say that
in a sentence and leave usage absent. A reconciled run with no tokens is a fact;
a reconciled run showing zero tokens would be a fabrication.

**Never twice.** Every imported run is keyed by the agent's own thread id, and
`EventStore.find_native_session` is consulted before anything is written. A
session already captured in driven, attached or observed mode is skipped and
reported as skipped. Getting this wrong would not raise — it would produce two
well-formed runs describing the same hour of work, and every total built on top
of them would be quietly doubled.

**Read-only throughout.** We list, we read, we never resume, archive, fork or
delete a thread; and as everywhere else in this subsystem, we never start a
daemon to do it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .adapters.codex_app_server import (
    MISSING_IN_APP_SERVER,
    CodexAppServerAdapter,
    ProtocolMismatch,
)
from .contract import (
    CaptureMode,
    Event,
    EventType,
    Fidelity,
    Outcome,
    new_run_id,
    new_session_id,
    new_turn_id,
)
from .reducer import RunView, reduce_run
from .runner import agent_version
from .store import EventStore

#: What a persisted thread cannot tell us however carefully we read it. These
#: reach the data-quality panel, so a reconciled run is never mistaken for a
#: live one that happened to be quiet.
MISSING_IN_THREAD_HISTORY = (
    "per-item timestamps (every item carries its turn's start time)",
    "per-request model timing",
    "approvals and their decisions",
    "output deltas",
)

#: Codex's rollout counters, in the app-server's own spelling, so the one usage
#: mapping in `CodexAppServerAdapter` serves both surfaces.
_ROLLOUT_USAGE = {
    "input_tokens": "inputTokens",
    "cached_input_tokens": "cachedInputTokens",
    "output_tokens": "outputTokens",
    "reasoning_output_tokens": "reasoningOutputTokens",
}

#: A turn's own verdict on itself → ours. `interrupted` is the one that matters:
#: treating it as complete is how "the agent finished" becomes a statistic about
#: sessions the user stopped.
_TURN_OUTCOME = {
    "completed": Outcome.AGENT_CLAIMED_COMPLETE,
    "interrupted": Outcome.INTERRUPTED,
    "failed": Outcome.VERIFIED_FAIL,
    "error": Outcome.VERIFIED_FAIL,
}

_SESSION_EVENT = {
    Outcome.AGENT_CLAIMED_COMPLETE: EventType.SESSION_COMPLETED,
    Outcome.INTERRUPTED: EventType.SESSION_INTERRUPTED,
    Outcome.VERIFIED_FAIL: EventType.SESSION_FAILED,
    Outcome.UNKNOWN: EventType.SESSION_COMPLETED,
}

REQUEST_TIMEOUT_S = 120.0


# ── the read-only client ─────────────────────────────────────────────────


class AppServerQuery:
    """A short-lived `codex app-server` used only to ask questions.

    Deliberately not `CodexAttachment`: an attachment *is* a run — it owns a
    run id, an adapter and a reducer, and its lifetime is the session it
    watches. This one outlives no session and produces many runs, so sharing
    that class would mean one connection permanently bound to whichever thread
    it happened to import first.
    """

    def __init__(self, codex_bin: str = "codex", cwd: Path | str | None = None) -> None:
        self.codex_bin = codex_bin
        self.cwd = Path(cwd) if cwd else Path.cwd()
        self.proc: subprocess.Popen | None = None
        self._out: dict[int, dict[str, Any]] = {}
        self._arrived = threading.Event()
        self._next = 0

    def __enter__(self) -> AppServerQuery:
        return self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def open(self) -> AppServerQuery:
        if shutil.which(self.codex_bin) is None:
            raise ProtocolMismatch(f"{self.codex_bin!r} is not on PATH")
        self.proc = subprocess.Popen(
            [self.codex_bin, "app-server"], cwd=str(self.cwd),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        threading.Thread(target=self._pump, daemon=True,
                         name="seer-reconcile-read").start()
        self.request("initialize", {
            "clientInfo": {"name": "sessionseer", "title": "SessionSeer",
                           "version": "0.1.0"}
        })
        self._write({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        return self

    def _pump(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        try:
            for line in self.proc.stdout:
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Notifications are ignored on purpose: nothing here drives a
                # thread, so anything the server volunteers is about someone
                # else's work.
                if isinstance(msg.get("id"), int) and "method" not in msg:
                    self._out[msg["id"]] = msg
                    self._arrived.set()
        finally:
            self._arrived.set()

    def _write(self, msg: dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise ProtocolMismatch("app-server is not open")
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict[str, Any],
                *, timeout_s: float = REQUEST_TIMEOUT_S) -> dict[str, Any]:
        self._next += 1
        rid = self._next
        self._write({"jsonrpc": "2.0", "id": rid, "method": method,
                     "params": params})
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            msg = self._out.pop(rid, None)
            if msg is not None:
                if "error" in msg:
                    raise ProtocolMismatch(
                        f"{method} failed: {msg['error'].get('message')}"
                    )
                return msg.get("result") or {}
            self._arrived.clear()
            self._arrived.wait(0.25)
            if self.proc is not None and self.proc.poll() is not None:
                raise ProtocolMismatch(f"app-server exited during {method}")
        raise ProtocolMismatch(f"{method} timed out after {timeout_s:.0f}s")

    def threads(self, *, limit: int = 50, cwd: str | None = None,
                page_size: int = 50) -> Iterator[dict[str, Any]]:
        """Thread summaries, newest first, following the server's cursor.

        `useStateDbOnly` is set: the alternative rescans and *repairs* rollout
        metadata, which is a write, and a reader has no business triggering one.
        """
        cursor: str | None = None
        sent = 0
        while sent < limit:
            params: dict[str, Any] = {
                "limit": min(page_size, limit - sent),
                "useStateDbOnly": True,
            }
            if cursor:
                params["cursor"] = cursor
            if cwd:
                params["cwd"] = {"paths": [cwd]}
            page = self.request("thread/list", params)
            rows = page.get("data") or []
            for row in rows:
                yield row
                sent += 1
                if sent >= limit:
                    return
            cursor = page.get("nextCursor")
            if not rows or not cursor:
                return

    def read(self, thread_id: str) -> dict[str, Any]:
        res = self.request("thread/read",
                           {"threadId": thread_id, "includeTurns": True})
        return res.get("thread") or {}

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                self.proc.kill()


# ── the rollout file, for the one number the protocol omits ──────────────


@dataclass(slots=True)
class RolloutFacts:
    """What `~/.codex/sessions/**/rollout-*.jsonl` adds to a thread read."""

    usage: dict[str, int] = field(default_factory=dict)
    context_window: int | None = None
    model: str | None = None
    effort: str | None = None
    cli_version: str | None = None
    git: dict[str, Any] | None = None
    #: why usage is empty, when it is
    note: str | None = None


def rollout_facts(path: str | Path | None) -> RolloutFacts:
    """Read the counters, and nothing else.

    The file also holds every message in full. We walk past those: the item
    mapping already records what was said in the form we keep (kinds and
    counts), and re-reading the text here would widen what this tool takes from
    someone's disk without widening what it can tell them.
    """
    if not path:
        return RolloutFacts(note="the thread record carries no rollout path")
    p = Path(path)
    if not p.exists():
        return RolloutFacts(note=f"rollout file is gone: {p.name}")

    facts = RolloutFacts()
    try:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if '"token_count"' not in line and '"session_meta"' not in line \
                        and '"turn_context"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = rec.get("payload") or {}
                kind = payload.get("type") or rec.get("type")
                if kind == "token_count":
                    info = payload.get("info") or {}
                    # `total_token_usage` is cumulative for the whole session,
                    # so the last one wins outright — summing the sightings is
                    # the multiply-by-the-number-of-updates bug in its other
                    # costume.
                    total = info.get("total_token_usage") or {}
                    got = {
                        camel: total[snake] for snake, camel in _ROLLOUT_USAGE.items()
                        if isinstance(total.get(snake), int)
                    }
                    if got:
                        facts.usage = got
                    if isinstance(info.get("model_context_window"), int):
                        facts.context_window = info["model_context_window"]
                elif kind == "session_meta":
                    facts.cli_version = payload.get("cli_version")
                    facts.git = payload.get("git")
                elif kind == "turn_context":
                    facts.model = payload.get("model") or facts.model
                    facts.effort = payload.get("effort") or facts.effort
    except OSError as exc:
        return RolloutFacts(note=f"rollout file unreadable: {exc}")

    if not facts.usage:
        facts.note = "the rollout file records no token counts for this session"
    return facts


# ── replaying a persisted thread ─────────────────────────────────────────


class CodexThreadReconciler(CodexAppServerAdapter):
    """A persisted thread, replayed through the live adapter's own route table.

    Subclassing rather than re-implementing is the point: every item kind Codex
    can persist is one the attached adapter already maps, and a second mapping
    would drift from the first the first time Codex adds an item type.
    """

    adapter_name = "codex_thread_history"

    def __init__(self, **kw: Any) -> None:
        kw.setdefault("capture_mode", CaptureMode.RECONCILED)
        super().__init__(**kw)
        #: the clock every replayed event is stamped with. Set per turn; never
        #: `time.time()`, which would date a session from last March to today.
        self._at: float = time.time()

    def event(self, event_type: EventType, **kw: Any) -> Event:
        kw.setdefault("ts", self._at)
        return super().event(event_type, **kw)

    # ── the replay ───────────────────────────────────────────────────────

    def replay(self, thread: dict[str, Any], facts: RolloutFacts) -> list[Event]:
        turns = thread.get("turns") or []
        started = _seconds(thread.get("createdAt"))
        self._at = started or time.time()
        if facts.model:
            self.model = {"provider": thread.get("modelProvider") or "openai",
                          "model_id": facts.model}

        events: list[Event] = [self._session_started(thread, facts)]
        if facts.note:
            events.append(self.warn(facts.note))
        events.append(self.warn(
            "reconciled from thread history: items carry their turn's start "
            "time, so per-call durations are absent rather than zero"
        ))

        last = Outcome.UNKNOWN
        for turn in turns:
            events.extend(self._turn(turn))
            last = _TURN_OUTCOME.get(str(turn.get("status")), Outcome.UNKNOWN)

        self._at = (
            _seconds(thread.get("updatedAt"))
            or (turns and _seconds(turns[-1].get("completedAt")))
            or self._at
        )
        if facts.usage:
            events.extend(self._usage_events(facts.usage, "rollout:token_count"))
        else:
            events.append(self._absent_usage(
                facts.note or "no token counts were recoverable for this thread"
            ))
        events.append(
            self.event(
                _SESSION_EVENT.get(last, EventType.SESSION_COMPLETED),
                fidelity=Fidelity.DETERMINISTIC,
                native_type="thread/read",
                payload={
                    "outcome": last.value,
                    "n_turns": len(turns),
                    "context_window": facts.context_window,
                    # Named for what it is. The turns' own clocks bound the
                    # session; nothing here was timed by us.
                    "duration_s": _span(turns),
                },
            )
        )
        return events

    def _session_started(self, thread: dict[str, Any],
                         facts: RolloutFacts) -> Event:
        return self.event(
            EventType.SESSION_STARTED,
            fidelity=Fidelity.DETERMINISTIC,
            native_type="thread/read",
            native={"threadId": thread.get("id"),
                    "sessionId": thread.get("sessionId")},
            source_event_id=thread.get("id"),
            payload={
                "capture_gaps": list(MISSING_IN_APP_SERVER)
                + list(MISSING_IN_THREAD_HISTORY),
                "native_session_id": thread.get("id"),
                "cwd": thread.get("cwd"),
                "source": thread.get("source"),
                "history_mode": thread.get("historyMode"),
                "cli_version": thread.get("cliVersion") or facts.cli_version,
                "effort": facts.effort,
                # Deliberately not `preview` or `name`: both are the user's own
                # prompt text, and this run is being imported without anyone
                # watching it happen.
                "has_name": bool(thread.get("name")),
            },
        )

    def _absent_usage(self, reason: str) -> Event:
        """Tokens we could not recover: `MISSING`, with a reason, never 0.

        An event rather than simply no event, so the run carries the category
        names it *would* have had. A view with no usage key at all and one that
        says "this could not be recovered" read the same in a table and mean
        very different things to whoever is deciding whether to trust the row.
        """
        return self.event(
            EventType.MODEL_USAGE_UPDATED,
            fidelity=Fidelity.MISSING,
            native_type="rollout:token_count",
            payload={
                "usage": None,
                "reason": reason,
                "native_categories": sorted(_ROLLOUT_USAGE.values()),
            },
        )

    def _turn(self, turn: dict[str, Any]) -> list[Event]:
        started = _seconds(turn.get("startedAt"))
        if started:
            self._at = started
        self.turn_id = turn.get("id") or new_turn_id()
        events: list[Event] = [
            self.event(EventType.TURN_STARTED, fidelity=Fidelity.NATIVE,
                       native_type="thread/read:turn")
        ]
        for item in turn.get("items") or []:
            events.extend(self._item({"item": item}, done=True))

        ended = _seconds(turn.get("completedAt"))
        if ended:
            self._at = ended
        outcome = _TURN_OUTCOME.get(str(turn.get("status")), Outcome.UNKNOWN)
        ms = turn.get("durationMs")
        events.append(
            self.event(
                EventType.TURN_COMPLETED,
                fidelity=Fidelity.NATIVE,
                native_type="thread/read:turn",
                payload={
                    "outcome": outcome.value,
                    "status": turn.get("status"),
                    "error": turn.get("error"),
                    "duration_ms": ms if isinstance(ms, (int, float)) else None,
                },
            )
        )
        self.turn_id = None
        return events


def _seconds(ms: Any) -> float | None:
    """Codex records epoch milliseconds; everything here is epoch seconds."""
    if isinstance(ms, (int, float)) and ms > 0:
        return ms / 1000.0 if ms > 1e11 else float(ms)
    return None


def _span(turns: list[dict[str, Any]]) -> float | None:
    first = next((_seconds(t.get("startedAt")) for t in turns
                  if _seconds(t.get("startedAt"))), None)
    last = next((_seconds(t.get("completedAt")) for t in reversed(turns)
                 if _seconds(t.get("completedAt"))), None)
    return None if first is None or last is None else last - first


# ── the pass itself ──────────────────────────────────────────────────────


@dataclass(slots=True)
class Imported:
    run_id: str
    thread_id: str
    n_events: int
    view: RunView


@dataclass(slots=True)
class ReconcileReport:
    """What the pass did, including — especially — what it declined to do."""

    imported: list[Imported] = field(default_factory=list)
    #: thread id → the run that already covers it
    skipped: dict[str, str] = field(default_factory=dict)
    #: thread id → why it could not be read
    failed: dict[str, str] = field(default_factory=dict)
    n_seen: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_seen": self.n_seen,
            "imported": [
                {"run_id": i.run_id, "thread_id": i.thread_id,
                 "n_events": i.n_events, "view": i.view.to_dict()}
                for i in self.imported
            ],
            "skipped": self.skipped,
            "failed": self.failed,
        }


def reconcile_codex(
    *,
    store: EventStore,
    codex_bin: str = "codex",
    cwd: Path | str | None = None,
    limit: int = 25,
    only_cwd: str | None = None,
    since: float | None = None,
    keep_reasoning: bool = False,
    query: AppServerQuery | None = None,
) -> ReconcileReport:
    """Import persisted Codex threads that are not in the store already.

    `since` is an epoch-seconds floor on the thread's last update, so a routine
    pass can be cheap without the cheapness being silent — threads older than
    the floor are not counted as seen.
    """
    report = ReconcileReport()
    version = agent_version(codex_bin)
    own = query is None
    q = query or AppServerQuery(codex_bin=codex_bin, cwd=cwd)
    if own:
        q.open()
    try:
        for summary in q.threads(limit=limit, cwd=only_cwd):
            tid = summary.get("id")
            if not tid:
                continue
            updated = _seconds(summary.get("updatedAt")) or 0.0
            if since is not None and updated < since:
                continue
            report.n_seen += 1

            existing = store.find_native_session("codex", tid)
            if existing:
                report.skipped[tid] = existing
                continue
            try:
                thread = q.read(tid)
            except ProtocolMismatch as exc:
                report.failed[tid] = str(exc)
                continue
            report.imported.append(
                _import_one(thread, summary, store=store, version=version,
                            keep_reasoning=keep_reasoning)
            )
    finally:
        if own:
            q.close()
    return report


def _import_one(thread: dict[str, Any], summary: dict[str, Any], *,
                store: EventStore, version: str,
                keep_reasoning: bool) -> Imported:
    tid = str(thread.get("id") or summary.get("id"))
    facts = rollout_facts(summary.get("path") or thread.get("path"))
    run_id = new_run_id()
    git = facts.git or {}
    repo = {
        "root_id": summary.get("cwd") or thread.get("cwd"),
        "branch": git.get("branch"),
        "commit": git.get("commit_hash"),
    }
    adapter = CodexThreadReconciler(
        run_id=run_id,
        session_id=new_session_id(),
        agent_version=thread.get("cliVersion") or facts.cli_version or version,
        keep_reasoning=keep_reasoning,
        repo=repo,
    )
    started = _seconds(thread.get("createdAt")) or _seconds(summary.get("createdAt"))
    store.register_run(
        run_id, agent="codex", agent_version=adapter.agent_version,
        capture_mode=CaptureMode.RECONCILED.value,
        label=f"imported thread {tid[:8]}",
        repo=repo, started_at=started, native_session_id=tid,
    )

    events = adapter.replay(thread, facts)
    store.append_many(events)
    # `now` is the session's own last moment, not ours. Finalising against the
    # wall clock would compute "quiet for four months" and hang a STALLED
    # overlay on every run we import.
    view = reduce_run(run_id, events, now=events[-1].ts if events else None)
    store.set_state(run_id, view.state.value)
    return Imported(run_id=run_id, thread_id=tid, n_events=len(events), view=view)


__all__ = [
    "MISSING_IN_THREAD_HISTORY",
    "AppServerQuery",
    "CodexThreadReconciler",
    "Imported",
    "ReconcileReport",
    "RolloutFacts",
    "reconcile_codex",
    "rollout_facts",
]
