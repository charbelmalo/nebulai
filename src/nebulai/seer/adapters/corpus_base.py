"""Shared machinery for the four *corpus* adapters (Attractors D7).

A corpus adapter is not a live adapter. It reads somebody else's finished
record — an Among Us game log, a ctfish run, an AI Village day, a published
chat transcript — and maps it into the same canonical `Event` vocabulary the
live adapters produce, so the trajectory viewer draws foreign runs with the
same code path as our own.

Four rules make that honest rather than merely possible:

1. **`CaptureMode.RECONCILED`, always.** Nothing here was observed as it
   happened. Even where a corpus carries a wall-clock timestamp, SessionSeer
   was not attached, and the capture mode has to say so.

2. **Fidelity is per field, not per run.** A single `Fidelity` on the event
   envelope cannot describe a record whose timestamps are real and whose token
   usage does not exist. `field_fidelity()` writes a `fidelity` map into the
   payload for exactly those fields where the envelope's value would be a lie,
   and `CorpusAdapter.missing_usage()` emits one explicit `usage: None` event
   per run so the data-quality panel shows *missing* rather than a confident
   zero.

3. **Order is not time.** ctfish entries have no timestamps at all. Rather than
   fabricate a plausible clock, corpus adapters that lack one set `ts` from a
   synthetic monotone counter, mark `ts` as `missing` in the fidelity map, and
   set `payload["order_only"] = True`. A viewer that draws a duration from
   those events is drawing a duration we invented; the flag is what lets it
   refuse.

4. **Foreign data wears foreign clothes** (spatial rule R7). Every event
   carries `payload["corpus"]` with the corpus id, the licence, and the URL the
   bytes came from, so a point on the map can always be traced back out of the
   repo.

None of these adapters ship the corpus itself. Small excerpts live under
`tests/fixtures/corpus/` for the tests; the real files are read from a path the
user passes, and the AI Village adapter refuses to write anything into the repo
at all (see `corpus_village.py`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..contract import (
    Action,
    CaptureMode,
    Effect,
    Event,
    EventType,
    Fidelity,
    Outcome,
)
from .base import ADAPTER_VERSION, BaseAdapter

#: Bumped when a corpus adapter changes how it maps, so a re-import is
#: distinguishable from the original one.
CORPUS_ADAPTER_VERSION = "0.1.0"


class CorpusError(ValueError):
    """A corpus file could not be read as the adapter it was handed to."""


@dataclass(frozen=True)
class CorpusSource:
    """Where a corpus came from, and what we are allowed to do with it.

    `ships_in_repo` is the decision, recorded next to the data rather than in a
    comment somewhere: `False` means the adapter reads it from a path at run
    time and no part of it may be committed.
    """

    id: str
    name: str
    url: str
    licence: str
    ships_in_repo: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "licence": self.licence,
            "ships_in_repo": self.ships_in_repo,
            **({"note": self.note} if self.note else {}),
        }


@dataclass
class CorpusRun:
    """One imported run: the events plus everything needed to audit them."""

    run_id: str
    session_id: str
    events: list[Event]
    corpus: CorpusSource
    #: per-run provenance: input path, sha256, counts, and what was not mapped
    meta: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def sha256_file(path: Path | str, *, limit: int | None = None) -> str:
    """sha256 of the input bytes, so an import names the exact file it read."""
    h = hashlib.sha256()
    n = 0
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            if limit is not None and n + len(chunk) > limit:
                h.update(chunk[: limit - n])
                break
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest()


def field_fidelity(**kw: Fidelity) -> dict[str, Any]:
    """A payload fragment recording per-field fidelity.

    Use it wherever the envelope's single `Fidelity` would over- or
    under-claim. `field_fidelity(ts=Fidelity.MISSING)` on an event whose
    envelope says `DETERMINISTIC` reads as: the mapping is deterministic, the
    clock is not real.
    """
    return {"fidelity": {k: v.value for k, v in kw.items()}}


def iter_json_lines(path: Path | str, *, strict: bool = False) -> Iterable[Any]:
    """JSONL reader that reports bad lines instead of dying on them.

    Several of these corpora are concatenations produced by a research script
    and contain a truncated final line. `strict=False` yields what parsed and
    leaves the count of what did not to the caller (a `CorpusError` is raised
    only when *nothing* parsed).
    """
    ok = bad = 0
    with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
                ok += 1
            except json.JSONDecodeError:
                bad += 1
                if strict:
                    raise CorpusError(f"{path}: line {ok + bad} is not JSON")
    if ok == 0:
        raise CorpusError(f"{path}: no line parsed as JSON ({bad} unparseable)")


class CorpusAdapter(BaseAdapter):
    """Base for the reconciled corpus adapters.

    Subclasses implement `read(path, **kw) -> list[CorpusRun]`. They construct
    events through `self.event(...)` exactly as a live adapter does, so the
    envelope, the credential scrub and the content-level labelling are the same
    code for our runs and for somebody else's.
    """

    agent = "corpus"
    adapter_name = "corpus"
    #: set by each subclass
    corpus: CorpusSource

    def __init__(self, *, run_id: str, session_id: str, **kw: Any) -> None:
        kw.setdefault("capture_mode", CaptureMode.RECONCILED)
        kw.setdefault("agent_version", "unknown")
        super().__init__(run_id=run_id, session_id=session_id, **kw)
        #: synthetic clock for corpora with no timestamps; see rule 3
        self._tick = 0.0
        #: the largest `ts` this adapter has emitted, real or synthetic, so the
        #: events that close a run land after it instead of at tick 0
        self._max_ts = 0.0
        self.n_unmapped = 0

    # ── time ─────────────────────────────────────────────────────────────

    def event(self, event_type: EventType, **kw: Any) -> Event:
        """As `BaseAdapter.event`, remembering how far the clock has got.

        A corpus that carries real timestamps never touches `next_tick`, so the
        tick counter stays at zero and cannot be used to place the run's closing
        events; a corpus with no clock at all has nothing else. Tracking the
        maximum emitted `ts` covers both without either adapter having to think
        about it.
        """
        ev = super().event(event_type, **kw)
        ts = getattr(ev, "ts", None)
        if isinstance(ts, (int, float)) and not isinstance(ts, bool):
            self._max_ts = max(self._max_ts, float(ts))
        return ev

    def next_tick(self) -> float:
        """The next synthetic ordering value. NOT a time — see `order_only`."""
        self._tick += 1.0
        return self._tick

    def closing_ts(self) -> float:
        """Where to put an event that closes a run the record already ended.

        After everything the record contained, and marked `order_only` by the
        callers below: the record says when the last thing happened, never when
        somebody stopped writing it down.
        """
        return self._max_ts if self._max_ts else self.next_tick()

    def ordered(self, **payload: Any) -> dict[str, Any]:
        """Payload additions for an event whose `ts` is an order, not a clock.

        Merges rather than replaces any `fidelity` map already in `payload`, so
        `self.ordered(**field_fidelity(output=Fidelity.MISSING))` keeps both the
        caller's per-field verdicts and this one's `ts: missing`.
        """
        p = dict(payload)
        p["order_only"] = True
        fid = dict(p.get("fidelity") or {})
        fid["ts"] = Fidelity.MISSING.value
        p["fidelity"] = fid
        return p

    # ── events every corpus run carries ──────────────────────────────────

    def corpus_payload(self, **extra: Any) -> dict[str, Any]:
        return {"corpus": self.corpus.to_dict(), **extra}

    def missing_usage(self) -> Event:
        """One event per run saying that token usage does not exist here.

        Published corpora record what the agent said and did, never what it
        cost. Emitting nothing would leave the panel's usage column empty and
        indistinguishable from a capture bug; emitting a zero would be a lie.
        """
        return self.event(
            EventType.MODEL_USAGE_UPDATED,
            fidelity=Fidelity.MISSING,
            ts=self.closing_ts(),
            payload=self.corpus_payload(
                **self.ordered(
                    usage=None,
                    note="this corpus does not record token usage",
                    **field_fidelity(usage=Fidelity.MISSING),
                )
            ),
        )

    def session_completed(self, ts: float | None = None) -> Event:
        """Close the run's session, because a corpus record has already ended.

        Without this the store's orphan sweep sees a run with no terminal
        session event, finds no live capture process, and rewrites it as
        `interrupted` — which would be a claim that somebody's published
        experiment was cut short. It ended; we were simply not there.
        """
        self.session_id = self.run_id
        self.turn_id = None
        return self.event(
            EventType.SESSION_COMPLETED,
            fidelity=Fidelity.DETERMINISTIC,
            ts=self.closing_ts() if ts is None else ts,
            payload=self.corpus_payload(
                **self.ordered(
                    note="the record ends here; this run was reconciled after the fact"
                )
            ),
        )

    def unmapped(self, kind: str) -> list[Event]:
        """A record this mapping did not understand — counted, and visible.

        Not `BaseAdapter.note_unknown_native`, for two reasons that only apply
        to a reconciled import: that helper leaves the `ts` to default to the
        wall clock, which for a record that ended months ago would stamp the
        warning with the *import* time, and it emits a bare payload without the
        `corpus` block every other event here carries. The count rises for every
        unmapped record; the event is emitted once per kind, because a thousand
        identical warnings is a way of hiding one.
        """
        self.n_unmapped += 1
        if kind in self._unknown_native:
            return []
        self._unknown_native.add(kind)
        msg = f"unmapped corpus record: {kind}"
        self.warnings.append(msg)
        return [
            self.event(
                EventType.ADAPTER_WARNING,
                fidelity=Fidelity.DETERMINISTIC,
                ts=self.closing_ts() or self.next_tick(),
                payload=self.corpus_payload(**self.ordered(note=msg)),
            )
        ]

    # ── provenance ───────────────────────────────────────────────────────

    def run_meta(self, path: Path | str, **extra: Any) -> dict[str, Any]:
        return {
            "corpus": self.corpus.to_dict(),
            "input_path": str(path),
            "input_sha256": sha256_file(path),
            "adapter": self.adapter_name,
            "adapter_version": CORPUS_ADAPTER_VERSION,
            "base_adapter_version": ADAPTER_VERSION,
            "capture_mode": CaptureMode.RECONCILED.value,
            "n_unmapped": self.n_unmapped,
            **extra,
        }


#: The nine-type normalized action, for corpora that name their own verbs.
#: Kept here rather than in each adapter so the same English word maps the same
#: way in all four — a `SEARCH` in ctfish and a `SEARCH` in the Village have to
#: mean the same thing or the cross-corpus comparison is meaningless.
def classify_shell(command: str) -> tuple[Action, Effect]:
    """Best-effort action for a shell command, with the effect left unknown.

    The effect is `UNKNOWN` on purpose: these corpora record the command and
    (sometimes) its output, but not whether the output told the agent anything
    it did not already have. `Effect.NO_NEW_INFORMATION` is decidable only
    where the corpus carries the output *and* the run's prior context, which is
    exactly the case `corpus_ctfish` handles and the others do not.
    """
    c = command.strip().lower()
    head = c.split()[0] if c.split() else ""
    if head in {"cat", "head", "tail", "less", "more", "ls", "file", "stat", "wc"}:
        return Action.INSPECT, Effect.UNKNOWN
    if head in {"grep", "rg", "find", "ag", "locate", "which", "whereis"}:
        return Action.SEARCH, Effect.UNKNOWN
    if head in {"git"}:
        return Action.VCS, Effect.UNKNOWN
    if head in {"pytest", "npm", "make", "cargo", "go", "tox", "mypy", "ruff"}:
        return Action.VERIFY, Effect.UNKNOWN
    if head in {"sed", "awk", "tee", "cp", "mv", "rm", "chmod", "touch", "mkdir"}:
        return Action.EDIT, Effect.STATE_CHANGED
    if ">" in c or ">>" in c:
        return Action.EDIT, Effect.STATE_CHANGED
    return Action.EXECUTE, Effect.UNKNOWN


__all__ = [
    "CORPUS_ADAPTER_VERSION",
    "CorpusAdapter",
    "CorpusError",
    "CorpusRun",
    "CorpusSource",
    "classify_shell",
    "field_fidelity",
    "iter_json_lines",
    "sha256_file",
]
