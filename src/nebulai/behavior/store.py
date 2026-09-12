"""Append-only SQLite store for raw trials (plan §6.1), built for resumption.

A behavioral study is a long-running, partly-paid, interruptible job. Two
properties follow, and both are enforced here rather than trusted to the runner:

**Raw evidence is append-only.** There is no `UPDATE` path for a completed
trial. A correction is a *derived* row computed at analysis time from
`raw_output`; what the model actually returned is never edited. `parser_version`
rides on each row so a normalizer change can be re-derived instead of
overwritten.

**Resumption must not duplicate.** Trial identity is
`(study_id, arm, cue, frame_id, model_key, repeat)` and that tuple is a UNIQUE
index. A run killed mid-flight and restarted re-derives the same schedule, sees
which identities already exist, and issues only the remainder — so "kill it and
start it again" is a supported operation rather than a way to double a bill.
`tests/test_behavior_store.py` kills a run mid-schedule and asserts the trial
count after resume, because a resume policy nobody interrupted is a resume
policy nobody tested.

The DB is opened in WAL mode with `synchronous=FULL` for the same reason: a
power loss in the middle of a paid run must lose at most the in-flight request,
not the preceding hour.

**One writer at a time.** Resumability makes relaunching cheap, which makes
relaunching twice easy. Measured 2026-09-12 on this repo: four `behavior run`
processes were alive on one store at once, each at ~40% CPU, and the row count
did not move for half an hour — every process re-derived the same remaining
schedule, generated the same trials, and lost the `INSERT OR IGNORE` race for
each one. Nothing was corrupted and nothing was double-billed (that is what the
UNIQUE index is for), but nothing progressed either, and the failure is silent:
each process looks healthy. `claim_writer` makes the second one refuse instead,
naming the pid that holds the store.
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from .contract import TrialRecord

SCHEMA_VERSION = 1


class StoreLockedError(RuntimeError):
    """Another process is writing this store. Raised by `claim_writer`.

    Its own type rather than ValueError because the caller's response is
    specific: wait, or confirm the holder is dead and force. A generic error
    here gets swallowed by a retry loop, which is the behaviour the lock exists
    to prevent.
    """

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trials (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    study_id        TEXT NOT NULL,
    arm             TEXT NOT NULL,
    cue             TEXT NOT NULL,
    frame_id        TEXT NOT NULL,
    model_key       TEXT NOT NULL,
    repeat          INTEGER NOT NULL,
    block           INTEGER NOT NULL,
    prompt          TEXT NOT NULL,
    prompt_sha      TEXT NOT NULL,
    raw_output      TEXT,
    associates      TEXT NOT NULL DEFAULT '[]',
    valid           INTEGER NOT NULL DEFAULT 0,
    invalid_reason  TEXT NOT NULL DEFAULT '',
    requested_model TEXT NOT NULL DEFAULT '',
    served_model    TEXT NOT NULL DEFAULT '',
    response_id     TEXT NOT NULL DEFAULT '',
    fingerprint     TEXT NOT NULL DEFAULT '',
    latency_ms      INTEGER,
    usage           TEXT NOT NULL DEFAULT '{}',
    cost_usd        REAL,
    parser_version  INTEGER NOT NULL DEFAULT 1,
    created         TEXT NOT NULL DEFAULT '',
    error           TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS trials_identity
    ON trials (study_id, arm, cue, frame_id, model_key, repeat);
CREATE INDEX IF NOT EXISTS trials_cue ON trials (study_id, cue);
CREATE INDEX IF NOT EXISTS trials_block ON trials (study_id, block);
"""

#: The identity tuple, in the order the UNIQUE index uses it.
Identity = tuple[str, str, str, str, str, int]


class TrialStore:
    """Owns one `.sqlite` file. Cheap to open; safe to open twice."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, isolation_level=None, timeout=30.0)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(_SCHEMA)
        self.set_meta("schema_version", str(SCHEMA_VERSION))

    # -- meta -------------------------------------------------------------
    def set_meta(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def get_meta(self, key: str, default: str = "") -> str:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def bind_manifest(self, manifest_hash: str, study_id: str) -> None:
        """Bind this db to exactly one frozen manifest.

        A db that already carries a different hash is refused. Reusing one
        store across two manifests is how a "resumed" run quietly becomes a
        pooled run over two protocols — the exact thing §5.4 forbids, arrived at
        by accident instead of by decision.
        """
        existing = self.get_meta("manifest_hash")
        if existing and existing != manifest_hash:
            raise ValueError(
                f"{self.path} was collected under manifest {existing}, and this "
                f"run's manifest is {manifest_hash}. Raw evidence is never "
                f"pooled across protocols (§5.4) — start a new study id, or "
                f"point at the original manifest."
            )
        self.set_meta("manifest_hash", manifest_hash)
        self.set_meta("study_id", study_id)

    # -- the single-writer lock -------------------------------------------
    #: a heartbeat older than this is treated as abandoned. Generous on purpose:
    #: a local arm's forward pass over GPT-2-XL took ~43 s per trial on the
    #: machine this was measured on, and a lock that expires inside one trial
    #: would hand the store to a second writer while the first is still working.
    LOCK_STALE_S = 900.0

    def claim_writer(self, *, force: bool = False, note: str = "") -> None:
        """Take the store's single writer slot, or refuse and say who holds it.

        The check is `BEGIN IMMEDIATE`, so two processes racing to claim cannot
        both win. A held lock is honoured when EITHER the holder's pid is alive
        on this host, OR its heartbeat is younger than `LOCK_STALE_S` (the pid
        test cannot cross hosts, so the heartbeat is the fallback, not the
        primary). `force=True` is for a holder a human has confirmed is dead —
        it is never inferred.
        """
        now = time.time()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            raw = self.get_meta("writer_lock")
            #  Re-claiming a lock THIS store object already holds is fine (it is
            #  the same writer saying so again). A lock held by our own pid
            #  through a DIFFERENT store object is not: something else in this
            #  process is mid-run, and two threads racing one store is the same
            #  wasted work as two processes racing it.
            mine = getattr(self, "_owns_lock", False)
            if raw and not force and not mine:
                try:
                    cur = json.loads(raw)
                except ValueError:
                    cur = {}
                if cur and self._holder_is_live(cur, now):
                    self.db.execute("ROLLBACK")
                    age = now - float(cur.get("heartbeat") or 0.0)
                    raise StoreLockedError(
                        f"{self.path} is already being written by pid "
                        f"{cur.get('pid')} on {cur.get('host')} "
                        f"(last heartbeat {age:.0f}s ago: {cur.get('note') or '-'}). "
                        f"Two runners on one store re-derive the same remaining "
                        f"schedule and race each other for every row, which "
                        f"costs twice the compute for no extra trials. Wait for "
                        f"it, or stop it and pass --force-unlock."
                    )
            self.set_meta(
                "writer_lock",
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "host": socket.gethostname(),
                        "started": now,
                        "heartbeat": now,
                        "note": note,
                    }
                ),
            )
            self.db.execute("COMMIT")
        except StoreLockedError:
            raise
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        self._owns_lock = True

    def _holder_is_live(self, cur: dict[str, Any], now: float) -> bool:
        host = str(cur.get("host") or "")
        pid = int(cur.get("pid") or 0)
        if host == socket.gethostname() and pid > 0:
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                # alive, owned by someone else
                return True
        return (now - float(cur.get("heartbeat") or 0.0)) < self.LOCK_STALE_S

    def beat(self) -> None:
        """Refresh the heartbeat. Cheap; call it once per persisted batch."""
        if not getattr(self, "_owns_lock", False):
            return
        raw = self.get_meta("writer_lock")
        if not raw:
            return
        try:
            cur = json.loads(raw)
        except ValueError:
            return
        cur["heartbeat"] = time.time()
        self.set_meta("writer_lock", json.dumps(cur))

    def release_writer(self) -> None:
        """Drop the lock if this process holds it. Never steals someone else's."""
        if not getattr(self, "_owns_lock", False):
            return
        raw = self.get_meta("writer_lock")
        self._owns_lock = False
        if not raw:
            return
        try:
            cur = json.loads(raw)
        except ValueError:
            cur = {}
        if int(cur.get("pid") or 0) != os.getpid():
            return
        self.db.execute("DELETE FROM meta WHERE key='writer_lock'")

    def writer_lock(self) -> dict[str, Any] | None:
        """Who holds the store, for `inspect` and for the server's /health."""
        raw = self.get_meta("writer_lock")
        if not raw:
            return None
        try:
            d = json.loads(raw)
        except ValueError:
            return None
        d["live"] = self._holder_is_live(d, time.time())
        return d

    # -- writes -----------------------------------------------------------
    def record(self, t: TrialRecord) -> bool:
        """Insert one trial. Returns False if that identity already existed.

        `INSERT OR IGNORE`, not `REPLACE`: on a resume race the *earlier*
        evidence wins, because it is the one that was actually collected.
        """
        d = asdict(t)
        d["associates"] = json.dumps(t.associates, ensure_ascii=False)
        d["usage"] = json.dumps(t.usage, ensure_ascii=False)
        d["valid"] = int(t.valid)
        cols = [c for c in d if c != "id"]
        sql = (
            f"INSERT OR IGNORE INTO trials ({','.join(cols)}) "
            f"VALUES ({','.join('?' for _ in cols)})"
        )
        cur = self.db.execute(sql, [d[c] for c in cols])
        return cur.rowcount > 0

    def record_many(self, trials: list[TrialRecord]) -> int:
        n = 0
        self.db.execute("BEGIN")
        try:
            for t in trials:
                n += int(self.record(t))
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        return n

    # -- reads ------------------------------------------------------------
    def completed(self, study_id: str) -> set[Identity]:
        """Identities already present — the resume set.

        Rows with an `error` count as *attempted*, not completed, so a transient
        provider failure is retried on resume while a successful trial is not
        re-billed.
        """
        rows = self.db.execute(
            "SELECT study_id,arm,cue,frame_id,model_key,repeat FROM trials "
            "WHERE study_id=? AND error=''",
            (study_id,),
        )
        return {
            (r["study_id"], r["arm"], r["cue"], r["frame_id"], r["model_key"], r["repeat"])
            for r in rows
        }

    def iter_trials(
        self, study_id: str, *, arm: str | None = None, cue: str | None = None
    ) -> Iterator[TrialRecord]:
        sql = "SELECT * FROM trials WHERE study_id=?"
        args: list[Any] = [study_id]
        if arm:
            sql += " AND arm=?"
            args.append(arm)
        if cue:
            sql += " AND cue=?"
            args.append(cue)
        sql += " ORDER BY id"
        for r in self.db.execute(sql, args):
            yield _row_to_trial(r)

    def count(self, study_id: str) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) AS n FROM trials WHERE study_id=?", (study_id,)
        ).fetchone()
        return int(row["n"])

    def spent_usd(self, study_id: str) -> float:
        row = self.db.execute(
            "SELECT COALESCE(SUM(cost_usd),0.0) AS s FROM trials WHERE study_id=?",
            (study_id,),
        ).fetchone()
        return float(row["s"])

    def progress(self, study_id: str) -> dict[str, Any]:
        """Counts the runner's health endpoint and the CLI both report."""
        rows = self.db.execute(
            "SELECT arm, COUNT(*) n, SUM(valid) v, SUM(error<>'') e "
            "FROM trials WHERE study_id=? GROUP BY arm",
            (study_id,),
        ).fetchall()
        return {
            "total": self.count(study_id),
            "spent_usd": self.spent_usd(study_id),
            "by_arm": {
                r["arm"]: {"trials": r["n"], "valid": int(r["v"] or 0), "errors": int(r["e"] or 0)}
                for r in rows
            },
        }

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "TrialStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _row_to_trial(r: sqlite3.Row) -> TrialRecord:
    return TrialRecord(
        study_id=r["study_id"],
        arm=r["arm"],
        cue=r["cue"],
        frame_id=r["frame_id"],
        model_key=r["model_key"],
        repeat=r["repeat"],
        block=r["block"],
        prompt=r["prompt"],
        prompt_sha=r["prompt_sha"],
        raw_output=r["raw_output"],
        associates=json.loads(r["associates"]),
        valid=bool(r["valid"]),
        invalid_reason=r["invalid_reason"],
        requested_model=r["requested_model"],
        served_model=r["served_model"],
        response_id=r["response_id"],
        fingerprint=r["fingerprint"],
        latency_ms=r["latency_ms"],
        usage=json.loads(r["usage"]),
        cost_usd=r["cost_usd"],
        parser_version=r["parser_version"],
        created=r["created"],
        error=r["error"],
    )
