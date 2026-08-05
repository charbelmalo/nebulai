"""Hermes → canonical events, in two halves, because one half is not enough.

**Correction to the intern's report (and to an earlier draft of
docs/SESSIONSEER.md §1.6):** `hermes -z PROMPT` is *not* a structured headless
mode. Its own help text says it prints "ONLY the final response text to stdout.
No banner, no spinner, no tool previews, no session_id line." There is no event
stream to parse, so an adapter that claimed to read tool calls off it would be
inventing them.

What `-z` genuinely gives us is process-level truth: it started, it ended, it
exited with a code, it produced N characters. Those are `DETERMINISTIC`, and the
adapter emits exactly that and no more.

The tokens come from the other half. `~/.hermes/state.db` (SQLite, WAL) carries
per-session `input_tokens`, `output_tokens`, `cache_read_tokens`,
`cache_write_tokens`, `reasoning_tokens` — all five categories, which makes
Hermes the *most* complete of the three agents on token accounting: Claude has
no reasoning bucket (it is billed inside `output_tokens`) and Codex has no
cache-write bucket at all. It also carries `cost_status` and `cost_source`,
i.e. Hermes ships its own provenance labels, and we keep them rather than
overwriting them with ours.

The join between the two halves is by launch time and cwd, not by an id Hermes
hands us — `--pass-session-id` injects the id into the *system prompt*, not into
stdout, so there is nothing to read. A time+cwd join can pick the wrong row when
two Hermes runs start in the same directory in the same second, so the adapter
records the join key and candidate count in the payload and downgrades to
`Fidelity.MISSING` when the match is ambiguous. Guessing would be worse than
saying we don't know.

The structured live path is ACP (`hermes acp`, verified with `--check`), which
speaks bidirectional JSON-RPC and emits `tool_call` start/complete,
`agent_thought_chunk`, `agent_message_chunk` and `plan` session updates. It is a
client implementation rather than a line parser, which is why it is M3 and not
M1 — and note that ACP carries no token usage either, so even that path needs
this reconciler.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from ..contract import (
    Action,
    CaptureMode,
    Event,
    EventType,
    Fidelity,
    Outcome,
    TokenCategory,
    new_turn_id,
)
from .base import BaseAdapter

DEFAULT_STATE_DB = Path.home() / ".hermes" / "state.db"

#: Hermes's own column names → our categories. The only agent of the three that
#: fills every bucket.
_USAGE_MAP = {
    "input_tokens": TokenCategory.INPUT,
    "output_tokens": TokenCategory.OUTPUT,
    "cache_read_tokens": TokenCategory.CACHE_READ,
    "cache_write_tokens": TokenCategory.CACHE_WRITE,
    "reasoning_tokens": TokenCategory.REASONING,
}

#: What `-z` cannot tell us. Listed so the data-quality panel can say why a
#: Hermes run has no tool lane, instead of rendering an empty one that reads as
#: "this run used no tools".
MISSING_IN_ONESHOT = (
    "tool calls",
    "reasoning spans",
    "per-turn boundaries",
    "approval requests",
    "token usage (recovered by state.db reconciliation)",
)


class HermesOneshotAdapter(BaseAdapter):
    """`hermes -z` stdout → lifecycle events only."""

    agent = "hermes"
    adapter_name = "hermes_oneshot"

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self._chunks: list[str] = []
        self._opened = False

    def open(self) -> list[Event]:
        """Emitted by the runner at launch, since `-z` prints no start marker."""
        self._opened = True
        self.turn_id = new_turn_id()
        return [
            self.event(
                EventType.SESSION_STARTED,
                fidelity=Fidelity.DETERMINISTIC,
                native_type="process.spawn",
                payload={"capture_gaps": list(MISSING_IN_ONESHOT)},
            ),
            self.event(
                EventType.TURN_STARTED,
                fidelity=Fidelity.DETERMINISTIC,
                native_type="process.spawn",
            ),
        ]

    def feed(self, line: str) -> list[Event]:
        # Everything on stdout is response text. Not JSON, not events — text.
        self._chunks.append(line)
        return []

    def close(self, exit_code: int | None = None) -> list[Event]:
        text = "".join(self._chunks)
        ok = exit_code in (0, None)
        events = [
            self.event(
                EventType.MESSAGE_ASSISTANT_COMPLETED,
                fidelity=Fidelity.DETERMINISTIC,
                action=Action.REPORT,
                native_type="process.stdout",
                payload={"chars": len(text), "text": text},
            ),
            self.event(
                EventType.SESSION_COMPLETED if ok else EventType.SESSION_FAILED,
                fidelity=Fidelity.DETERMINISTIC,
                native_type="process.exit",
                payload={
                    "exit_code": exit_code,
                    # `-z` returning 0 with text is the agent claiming it
                    # answered. Nothing here verifies the answer.
                    "outcome": (
                        Outcome.AGENT_CLAIMED_COMPLETE if ok
                        else Outcome.INFRASTRUCTURE_FAILURE
                    ).value,
                },
            ),
        ]
        self.turn_id = None
        return events

    def finish(self) -> list[Event]:
        return self.close() if self._opened else []


# ── reconciliation ───────────────────────────────────────────────────────────


class HermesStateDbReconciler(BaseAdapter):
    """`~/.hermes/state.db` → usage and counts for a run we already captured.

    Read-only and read-only on purpose: the file is a live WAL database owned by
    another process, opened `mode=ro` with `immutable=0` so WAL readers still see
    committed writes. Nothing here writes, and nothing here holds a lock longer
    than one SELECT.
    """

    agent = "hermes"
    adapter_name = "hermes_state_db"

    def __init__(self, *, db_path: Path | str | None = None, **kw: Any) -> None:
        kw.setdefault("capture_mode", CaptureMode.RECONCILED)
        super().__init__(**kw)
        self.db_path = Path(db_path) if db_path else DEFAULT_STATE_DB

    def feed(self, line: str) -> list[Event]:  # pragma: no cover - not a stream
        return []

    def reconcile(
        self,
        *,
        started_after: float,
        cwd: str | None = None,
        source: str = "cli",
        native_session_id: str | None = None,
    ) -> list[Event]:
        """Find this run's row and emit its usage.

        `started_after` should be the launch timestamp taken just before spawn.
        A small negative slack is applied because Hermes writes `started_at`
        from its own clock, which can land marginally before ours.
        """
        if not self.db_path.exists():
            return [self.warn(f"state.db not found at {self.db_path}")]

        try:
            rows = self._query(started_after, cwd, source, native_session_id)
        except sqlite3.Error as exc:
            return [self.warn(f"state.db read failed: {exc}")]

        if not rows:
            return [
                self._absent_usage(
                    "no state.db session matched the launch window",
                    candidates=0,
                )
            ]
        if len(rows) > 1 and native_session_id is None:
            # Two runs from the same directory in the same second. We cannot
            # tell which row is ours, and attributing the wrong one would put a
            # real token count on the wrong run — worse than reporting none.
            return [
                self._absent_usage(
                    "ambiguous state.db match (concurrent Hermes runs)",
                    candidates=len(rows),
                    session_ids=[r["id"] for r in rows],
                )
            ]

        row = rows[0]
        counts = {
            cat: row[col] for col, cat in _USAGE_MAP.items()
            if row[col] is not None
        }
        self.fold_usage(f"state_db:{row['id']}", counts, _USAGE_MAP.keys())
        self.model = {"provider": row["billing_provider"], "model_id": row["model"]}

        events = [
            self.event(
                EventType.MODEL_USAGE_UPDATED,
                fidelity=Fidelity.DETERMINISTIC,
                native_type="state.db:sessions",
                native={"session_id": row["id"], "source": row["source"]},
                payload={
                    "usage": {c.value: n for c, n in counts.items()},
                    "native_categories": sorted(_USAGE_MAP),
                    "authoritative": True,
                    "join": {
                        "by": "native_session_id" if native_session_id else "started_at+cwd",
                        "candidates": len(rows),
                    },
                    # Hermes labels its own cost provenance. Keep its words.
                    "cost_usd": row["actual_cost_usd"] or row["estimated_cost_usd"],
                    "cost_status": row["cost_status"],
                    "cost_source": row["cost_source"],
                    "cost_fidelity": (
                        Fidelity.NATIVE.value if row["actual_cost_usd"] is not None
                        else Fidelity.ESTIMATED.value
                    ),
                },
            ),
            self.event(
                EventType.SESSION_COMPLETED
                if row["end_reason"] not in (None, "error")
                else EventType.SESSION_FAILED,
                fidelity=Fidelity.DETERMINISTIC,
                native_type="state.db:sessions",
                native={"session_id": row["id"]},
                payload={
                    "outcome": Outcome.AGENT_CLAIMED_COMPLETE.value,
                    "end_reason": row["end_reason"],
                    "message_count": row["message_count"],
                    "tool_call_count": row["tool_call_count"],
                    "api_call_count": row["api_call_count"],
                    "duration_s": (
                        row["ended_at"] - row["started_at"]
                        if row["ended_at"] and row["started_at"] else None
                    ),
                },
            ),
        ]
        return events

    # ── internals ────────────────────────────────────────────────────────

    _COLS = (
        "id, source, model, started_at, ended_at, end_reason, message_count, "
        "tool_call_count, api_call_count, input_tokens, output_tokens, "
        "cache_read_tokens, cache_write_tokens, reasoning_tokens, "
        "billing_provider, estimated_cost_usd, actual_cost_usd, cost_status, "
        "cost_source, cwd"
    )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=2.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _query(
        self,
        started_after: float,
        cwd: str | None,
        source: str,
        native_session_id: str | None,
    ) -> list[sqlite3.Row]:
        with self._connect() as conn:
            if native_session_id:
                return list(
                    conn.execute(
                        f"SELECT {self._COLS} FROM sessions WHERE id = ?",
                        (native_session_id,),
                    )
                )
            sql = f"SELECT {self._COLS} FROM sessions WHERE started_at >= ? AND source = ?"
            params: list[Any] = [started_after - 2.0, source]
            if cwd:
                sql += " AND cwd = ?"
                params.append(cwd)
            sql += " ORDER BY started_at DESC LIMIT 5"
            return list(conn.execute(sql, params))

    def _absent_usage(self, reason: str, **extra: Any) -> Event:
        """Usage we could not recover. `MISSING`, with a reason, never 0."""
        self.warnings.append(reason)
        return self.event(
            EventType.MODEL_USAGE_UPDATED,
            fidelity=Fidelity.MISSING,
            native_type="state.db:sessions",
            payload={
                "usage": None,
                "reason": reason,
                "native_categories": sorted(_USAGE_MAP),
                **extra,
            },
        )
