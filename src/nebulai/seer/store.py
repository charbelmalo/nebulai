"""Append-only event log + a SQLite index that can be rebuilt from it.

The split is deliberate. The JSONL file is the record: one `Event.to_json()` per
line, never rewritten, never compacted in place. The SQLite index is a cache for
"which runs exist, what is their state, show me the last 200 events" — the
questions a live UI asks sixty times a minute and a file scan answers badly.

If the two ever disagree, the file wins. `reindex()` throws the index away and
rebuilds it, and the M1b tests exercise that path rather than trusting it,
because an index that silently drifts from the log is worse than no index: every
number the UI shows would be confidently derived from stale rows.

Concurrency: one writer (the collector) appends a newline-terminated line per
event, which is atomic for the sizes involved on both APFS and ext4, and updates
the index under a lock. Readers never touch the writer's connection or its lock
— each read opens its own short-lived connection against the WAL. That costs a
few dozen microseconds per query and buys two things: a viewer rendering a
100k-event run can never stall the capture of a live one, and no two threads
ever share a `sqlite3.Connection` (which raises `InterfaceError: bad parameter
or other API misuse` at random under a threaded HTTP server, and did).
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import Event, EventType

DEFAULT_ROOT = Path.home() / ".nebulai" / "seer"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    agent        TEXT NOT NULL,
    agent_version TEXT,
    capture_mode TEXT,
    label        TEXT,
    repo_root    TEXT,
    branch       TEXT,
    started_at   REAL,
    ended_at     REAL,
    state        TEXT,
    outcome      TEXT,
    n_events     INTEGER NOT NULL DEFAULT 0,
    n_warnings   INTEGER NOT NULL DEFAULT 0,
    -- the *agent's* id for this session, not ours. The only thing that lets a
    -- later reconciliation pass tell "a session we already have" from "a
    -- session we have never seen", so that importing history cannot silently
    -- double a month of token counts.
    native_session_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC);

CREATE TABLE IF NOT EXISTS events (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    event_id   TEXT NOT NULL UNIQUE,
    ts         REAL NOT NULL,
    event_type TEXT NOT NULL,
    action     TEXT,
    effect     TEXT,
    span_id    TEXT,
    turn_id    TEXT,
    fidelity   TEXT NOT NULL,
    line       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(run_id, event_type);
"""


@dataclass(slots=True)
class RunSummary:
    run_id: str
    agent: str
    agent_version: str | None
    capture_mode: str | None
    label: str | None
    repo_root: str | None
    branch: str | None
    started_at: float | None
    ended_at: float | None
    state: str | None
    outcome: str | None
    n_events: int
    n_warnings: int
    native_session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__slots__}


class EventStore:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root else DEFAULT_ROOT
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.db"
        self._lock = threading.Lock()
        self._handles: dict[str, Any] = {}
        self._conn = self._connect()
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    # ── plumbing ─────────────────────────────────────────────────────────

    def _migrate(self) -> None:
        """Add columns a `CREATE TABLE IF NOT EXISTS` will not add.

        A store written by an earlier build already has a `runs` table, so the
        schema script skips it entirely and every later read of a new column
        raises. Adding the column here is what makes an existing store readable
        rather than requiring the user to delete their history to get an
        upgrade.
        """
        have = {r["name"] for r in self._conn.execute("PRAGMA table_info(runs)")}
        for col, decl in (("native_session_id", "TEXT"), ("capture_pid", "INTEGER")):
            if col not in have:
                self._conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {decl}")
        # After the ALTER, never in `_SCHEMA`: an index over a column the old
        # table does not have yet fails, and `IF NOT EXISTS` does not save it.
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_native"
            " ON runs(agent, native_session_id)"
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=5.0)
        conn.row_factory = sqlite3.Row
        # WAL so readers can read while the collector writes. It is a property of
        # the database file, so setting it once here covers every later reader.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @contextmanager
    def _reading(self) -> Iterator[sqlite3.Connection]:
        """A connection of this thread's own, for the length of one query.

        Deliberately not pooled per thread: `ThreadingHTTPServer` spawns a
        thread per request, so a pool would either leak connections or need
        lifetime tracking, to save an open() that costs less than the query.
        """
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def log_path(self, run_id: str) -> Path:
        return self.runs_dir / run_id / "events.jsonl"

    def _handle(self, run_id: str):
        h = self._handles.get(run_id)
        if h is None:
            p = self.log_path(run_id)
            p.parent.mkdir(parents=True, exist_ok=True)
            h = p.open("a", encoding="utf-8")
            self._handles[run_id] = h
        return h

    def close(self) -> None:
        with self._lock:
            for h in self._handles.values():
                h.close()
            self._handles.clear()
            self._conn.close()

    # ── writing ──────────────────────────────────────────────────────────

    def append(self, event: Event) -> None:
        self.append_many([event])

    def append_many(self, events: list[Event]) -> None:
        if not events:
            return
        with self._lock:
            by_run: dict[str, list[Event]] = {}
            for e in events:
                by_run.setdefault(e.run_id, []).append(e)

            for run_id, batch in by_run.items():
                h = self._handle(run_id)
                start_line = self._line_count(run_id)
                for i, e in enumerate(batch):
                    h.write(e.to_json() + "\n")
                h.flush()
                # fsync only on terminal events: a crash mid-run costs the last
                # few lines, a crash after "completed" would cost the verdict.
                if any(_is_terminal(e) for e in batch):
                    os.fsync(h.fileno())
                self._index(run_id, batch, start_line)
            self._conn.commit()

    def _line_count(self, run_id: str, conn: sqlite3.Connection | None = None) -> int:
        c = conn if conn is not None else self._conn
        row = c.execute(
            "SELECT COALESCE(MAX(line), -1) AS m FROM events WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return int(row["m"]) + 1

    def _index(self, run_id: str, batch: list[Event], start_line: int) -> None:
        first = batch[0]
        self._conn.execute(
            "INSERT OR IGNORE INTO runs (run_id, agent, agent_version, capture_mode,"
            " repo_root, branch, started_at, n_events) VALUES (?,?,?,?,?,?,?,0)",
            (
                run_id,
                first.source.agent,
                first.source.agent_version,
                first.source.capture_mode.value,
                (first.repo or {}).get("root_id"),
                (first.repo or {}).get("branch"),
                first.ts,
            ),
        )
        rows = []
        for i, e in enumerate(batch):
            rows.append(
                (
                    run_id, e.event_id, e.ts, e.event_type.value,
                    e.action.value if e.action else None,
                    e.effect.value if e.effect else None,
                    e.span_id, e.turn_id, e.source.fidelity.value, start_line + i,
                )
            )
        self._conn.executemany(
            "INSERT OR IGNORE INTO events (run_id, event_id, ts, event_type, action,"
            " effect, span_id, turn_id, fidelity, line) VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        # The agent's own session id, if any event carried one. Read off the log
        # rather than passed in, so `reindex()` restores it: a column the index
        # holds and the log cannot rebuild is the thing reindex exists to
        # disprove.
        native_id = next(
            (
                e.payload["native_session_id"] for e in batch
                if e.event_type is EventType.SESSION_STARTED
                and e.payload.get("native_session_id")
            ),
            None,
        )
        n_warn = sum(
            1 for e in batch if e.event_type is EventType.ADAPTER_WARNING
        )
        # `ended_at` is when the *session* ended, not when our bookkeeping did.
        # RUN_COMPLETED lands a few milliseconds later and counting it here
        # would give the run list a different wall clock than the run detail —
        # two numbers for one quantity, which is the bug this whole subsystem
        # exists to avoid.
        last_terminal = next(
            (e for e in reversed(batch) if e.event_type in _SESSION_TERMINAL), None
        )
        self._conn.execute(
            "UPDATE runs SET n_events = n_events + ?, n_warnings = n_warnings + ?,"
            " ended_at = COALESCE(?, ended_at), outcome = COALESCE(?, outcome),"
            " state = COALESCE(?, state),"
            " native_session_id = COALESCE(?, native_session_id) WHERE run_id = ?",
            (
                len(batch), n_warn,
                last_terminal.ts if last_terminal else None,
                (last_terminal.payload.get("outcome") if last_terminal else None),
                # A terminal event names the terminal state, so `reindex()` can
                # rebuild the state column from the log alone. Without this a
                # rebuilt index shows every finished run as still running — the
                # index would hold something the log could not restore, which is
                # the property reindex exists to disprove.
                _TERMINAL_STATE.get(
                    last_terminal.event_type if last_terminal else None
                ),
                native_id,
                run_id,
            ),
        )

    def register_run(
        self,
        run_id: str,
        *,
        agent: str,
        agent_version: str | None = None,
        capture_mode: str | None = None,
        label: str | None = None,
        repo: dict[str, Any] | None = None,
        started_at: float | None = None,
        native_session_id: str | None = None,
    ) -> None:
        """Make a run addressable *before* its first event exists.

        The server hands a `run_id` back over HTTP the moment a run is launched,
        and the viewer immediately asks for it. Without this the id 404s for the
        few milliseconds until the subprocess produces its first line — an id we
        just issued, that does not resolve. The row starts empty and `starting`;
        every field it holds is something we know because we launched it, not
        something we have observed.

        The row also records *our* pid. Every path that opens a run goes through
        here, so that one column is what later lets `orphans()` tell "still
        being captured, in another process" from "was being captured by a
        process that is gone".
        """
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO runs (run_id, agent, agent_version,"
                " capture_mode, label, repo_root, branch, started_at, state,"
                " n_events, native_session_id, capture_pid)"
                " VALUES (?,?,?,?,?,?,?,?,'starting',0,?,?)",
                (
                    run_id, agent, agent_version, capture_mode, label,
                    (repo or {}).get("root_id"), (repo or {}).get("branch"),
                    started_at, native_session_id, os.getpid(),
                ),
            )
            self._conn.commit()

    def orphans(self) -> list[RunSummary]:
        """Runs left mid-capture by a process that is no longer running.

        A crash — or `kill -9`, or a laptop lid — leaves a run in `running`
        with no terminal event, and nothing about that row says whether the
        capture is still going. Read from the outside it looks live forever:
        the viewer draws a pulsing dot for a run that stopped last Tuesday.

        The test is the recorded pid, not a timeout. A run can be legitimately
        silent for an hour (a long model call, a paused agent), so "quiet"
        proves nothing, while "the process that was writing this is gone" is a
        fact. A NULL pid means the row predates this column, which likewise
        means it was written by a process that has since exited.
        """
        out: list[RunSummary] = []
        with self._reading() as conn:
            rows = conn.execute(
                "SELECT * FROM runs WHERE state IS NULL OR state NOT IN"
                " ('completed','failed','interrupted','detached')"
            ).fetchall()
        for r in rows:
            pid = r["capture_pid"]
            if pid and _pid_alive(int(pid)):
                continue
            out.append(_summary(r))
        return out

    def find_native_session(self, agent: str, native_session_id: str) -> str | None:
        """The run we already have for this agent-side session, if any.

        The whole basis of importing history safely. A reconciliation pass that
        could not answer this would re-import every session it has already
        captured, and the double would be invisible: two well-formed runs, each
        internally consistent, that happen to describe the same hour of work.
        """
        with self._reading() as conn:
            row = conn.execute(
                "SELECT run_id FROM runs WHERE agent = ? AND native_session_id = ?"
                " ORDER BY started_at LIMIT 1",
                (agent, native_session_id),
            ).fetchone()
        return row["run_id"] if row else None

    def delete_run(self, run_id: str) -> dict[str, Any]:
        """Remove one run: its log, its directory, and its index rows.

        The append-only rule is about *events*, not runs — a log nobody can
        delete is a log nobody can be asked to keep. What matters is that the
        deletion is total: a run whose events survive in the index while its
        log is gone would still be counted by `list_runs`, still be exportable
        as an empty file, and still look like a real capture.

        Returns what went, so a caller can report it. Deleting a run that is
        not there raises rather than returning quietly — "deleted" and "was
        never here" are different answers and only one of them is reassuring.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT run_id FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            log = self.log_path(run_id)
            if row is None and not log.exists():
                raise KeyError(run_id)

            # Close our own append handle first: on Windows an open handle
            # blocks the unlink, and everywhere it would keep writing into a
            # file that no longer has a name.
            h = self._handles.pop(run_id, None)
            if h is not None:
                h.close()

            n_events = int(self._conn.execute(
                "SELECT COUNT(*) AS n FROM events WHERE run_id = ?", (run_id,)
            ).fetchone()["n"])
            self._conn.execute("DELETE FROM events WHERE run_id = ?", (run_id,))
            self._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            self._conn.commit()

            bytes_freed = log.stat().st_size if log.exists() else 0
            shutil.rmtree(log.parent, ignore_errors=True)
        return {
            "run_id": run_id,
            "events": n_events,
            "bytes": bytes_freed,
            "log_removed": not log.exists(),
        }

    def set_state(self, run_id: str, state: str, label: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET state = ?, label = COALESCE(?, label) WHERE run_id = ?",
                (state, label, run_id),
            )
            self._conn.commit()

    # ── reading ──────────────────────────────────────────────────────────

    def list_runs(self, limit: int = 100, agent: str | None = None) -> list[RunSummary]:
        sql = "SELECT * FROM runs"
        params: list[Any] = []
        if agent:
            sql += " WHERE agent = ?"
            params.append(agent)
        sql += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        with self._reading() as conn:
            return [_summary(r) for r in conn.execute(sql, params)]

    def get_run(self, run_id: str) -> RunSummary | None:
        with self._reading() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return _summary(row) if row else None

    def read(self, run_id: str, *, since_line: int = 0) -> Iterator[Event]:
        """Stream events straight off the log. The index is never consulted
        here — this is the path that stays correct when the index is stale.

        A live run is being appended to as we read, so the last line may be
        half-written. That one line is allowed to be unparsable and is simply
        not yielded; any *earlier* line that fails to parse is corruption and
        raises, because silently skipping it would mean the record and what we
        report from it had quietly diverged.
        """
        path = self.log_path(run_id)
        if not path.exists():
            return
        with path.open(encoding="utf-8") as fh:
            held: str | None = None
            for i, line in enumerate(fh):
                if i < since_line:
                    continue
                if held is not None:
                    yield Event.from_dict(json.loads(held))
                held = line.strip() or None
            if held is not None:
                try:
                    parsed = json.loads(held)
                except ValueError:
                    return  # a partial append, mid-flight — not ours to report
                yield Event.from_dict(parsed)

    def tail(self, run_id: str, n: int = 200) -> list[Event]:
        with self._reading() as conn:
            total = self._line_count(run_id, conn)
        return list(self.read(run_id, since_line=max(0, total - n)))

    # ── recovery ─────────────────────────────────────────────────────────

    def reindex(self, run_id: str | None = None) -> int:
        """Rebuild the index from the logs. The log is the record; this is the
        proof that nothing lives only in the index."""
        with self._lock:
            ids = (
                [run_id] if run_id
                else sorted(p.name for p in self.runs_dir.iterdir() if p.is_dir())
            )
            for rid in ids:
                self._conn.execute("DELETE FROM events WHERE run_id = ?", (rid,))
                self._conn.execute("DELETE FROM runs WHERE run_id = ?", (rid,))
            self._conn.commit()

        n = 0
        for rid in ids:
            batch: list[Event] = []
            for e in self.read(rid):
                batch.append(e)
                if len(batch) >= 500:
                    self._append_indexed_only(rid, batch)
                    n += len(batch)
                    batch = []
            if batch:
                self._append_indexed_only(rid, batch)
                n += len(batch)
        return n

    def _append_indexed_only(self, run_id: str, batch: list[Event]) -> None:
        with self._lock:
            self._index(run_id, batch, self._line_count(run_id))
            self._conn.commit()


#: The agent's session ending. These carry `ended_at` and `outcome`.
_SESSION_TERMINAL = frozenset(
    {
        EventType.SESSION_COMPLETED,
        EventType.SESSION_FAILED,
        EventType.SESSION_INTERRUPTED,
    }
)

#: Anything worth an fsync — the session ending *and* our own RUN_COMPLETED,
#: because losing the verdict to a crash is the one loss that matters.
_TERMINAL_EVENTS = _SESSION_TERMINAL | {EventType.RUN_COMPLETED}

#: terminal event → the state it puts the run in. Mirrors the reducer, which
#: owns this mapping for the live view; here it only lets `reindex()` restore
#: what the collector already wrote.
_TERMINAL_STATE: dict[EventType | None, str] = {
    EventType.SESSION_COMPLETED: "completed",
    EventType.SESSION_FAILED: "failed",
    EventType.SESSION_INTERRUPTED: "interrupted",
}


def _is_terminal(e: Event) -> bool:
    return e.event_type in _TERMINAL_EVENTS


def _pid_alive(pid: int) -> bool:
    """Signal 0: the kernel does the permission and existence checks and sends
    nothing. `EPERM` means the pid exists but belongs to someone else, which is
    still "alive" for our purposes.

    Pid reuse can make a dead capture look live. The cost of that is one run
    that stays `running` until the next sweep catches it, which is the same
    thing that happened before this existed — the failure is bounded, and the
    opposite error (closing a run someone is still capturing) is not.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _summary(row: sqlite3.Row) -> RunSummary:
    return RunSummary(
        run_id=row["run_id"],
        agent=row["agent"],
        agent_version=row["agent_version"],
        capture_mode=row["capture_mode"],
        label=row["label"],
        repo_root=row["repo_root"],
        branch=row["branch"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        state=row["state"],
        outcome=row["outcome"],
        n_events=row["n_events"],
        n_warnings=row["n_warnings"],
        native_session_id=row["native_session_id"],
    )
