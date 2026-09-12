"""A published chat transcript → canonical events (Attractors D7, episode #1).

The case this exists for is the Sydney/Bing transcript Kevin Roose published in
the New York Times in February 2023 — the canonical example of a model leaving
its assigned persona and staying gone. It is also the case with the least data
behind it, and the adapter's job is to make that impossible to forget:

* **The text is not in this repository and must not be put in it.** It is a
  copyrighted newspaper article. `read()` takes a path the operator supplies;
  the fixture used by the tests is a short transcript written by us.
* **The model is closed.** There is no checkpoint, no revision, no residual
  stream, and therefore no model-internal placement. Every event carries
  `payload["model_internal"] = False` and the run's `RUN_STARTED` carries
  `placement_possible = "text_embedder_only"`. Rule R7 turns that into the
  dashed glyph in the viewer: a point placed from a text embedder is not the
  same kind of claim as a point placed from a pinned model's activations, and
  it has to look different. If no embedder is reachable, the placement is
  `missing` and the card says so. Nothing here fabricates a coordinate.
* **There is no tool use, no repository and no token accounting.** All three
  are emitted as explicit `missing`, once, rather than left absent.

Two input formats, both plain:

`.json` — ``{"id", "title", "source": {...}, "turns": [{"role", "text",
"ts"?}]}``. `role` is one of `user`, `assistant`, `system`.

`.md` / `.txt` — speaker-prefixed lines, the shape a transcript is usually
pasted in::

    user: what is your name?
    assistant: I'm Bing …

A prefix that is not a known role starts a new turn attributed to that speaker
verbatim, with `role` `"other"` — the transcript's own words for who is
speaking are data, and collapsing an unknown speaker into `assistant` would
put words in the model's mouth.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..contract import Event, EventType, Fidelity
from .corpus_base import (
    CorpusAdapter,
    CorpusError,
    CorpusRun,
    CorpusSource,
    field_fidelity,
)

TRANSCRIPT = CorpusSource(
    id="transcript",
    name="a published chat transcript (closed model, no activations)",
    url="",
    licence="the publisher's; supplied by the operator at run time",
    ships_in_repo=False,
    note=(
        "the Sydney/Bing transcript (Kevin Roose, New York Times, February "
        "2023) is the intended input and is NOT redistributed here; the test "
        "fixture is a short transcript written for this repository"
    ),
)

ROLES = ("user", "assistant", "system")

_PREFIX = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _.'-]{0,40})\s*:\s*(.*)$")


def parse_markdown(text: str) -> list[dict[str, Any]]:
    """Speaker-prefixed lines → turns. Continuation lines join the turn above."""
    turns: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        m = _PREFIX.match(line)
        if m:
            speaker, body = m.group(1).strip(), m.group(2).strip()
            role = speaker.lower()
            turns.append(
                {
                    "role": role if role in ROLES else "other",
                    "speaker": speaker,
                    "text": body,
                }
            )
        elif turns:
            turns[-1]["text"] = (turns[-1]["text"] + "\n" + line.strip()).strip()
    return [t for t in turns if t["text"]]


def read_transcript(path: Path | str) -> dict[str, Any]:
    """`.json` as written, `.md`/`.txt` parsed from speaker prefixes."""
    p = Path(path)
    if not p.exists():
        raise CorpusError(
            f"{p} does not exist. This adapter ships no transcript: pass the "
            f"path to one you hold."
        )
    raw = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix.lower() == ".json":
        doc = json.loads(raw)
        if not isinstance(doc, dict) or not isinstance(doc.get("turns"), list):
            raise CorpusError(f"{p}: expected an object with a 'turns' list")
        return doc
    turns = parse_markdown(raw)
    if not turns:
        raise CorpusError(f"{p}: no 'speaker: text' line was found")
    return {"id": p.stem, "title": p.stem, "turns": turns}


class TranscriptCorpusAdapter(CorpusAdapter):
    """One transcript → one session of alternating messages."""

    agent = "transcript"
    adapter_name = "corpus_transcript"
    corpus = TRANSCRIPT

    def __init__(self, *, run_id: str = "", session_id: str = "", **kw: Any) -> None:
        super().__init__(
            run_id=run_id or "transcript", session_id=session_id or "transcript", **kw
        )

    def read(
        self,
        path: Path | str,
        *,
        model: str | None = None,
        run_id: str | None = None,
    ) -> list[CorpusRun]:
        doc = read_transcript(path)
        stem = _slug(str(doc.get("id") or Path(path).stem))
        rid = run_id or (stem if stem.startswith("transcript") else f"transcript-{stem}")
        self.run_id = rid
        self.session_id = rid
        self._tick = 0.0
        # A closed model has no revision to pin. `revision: None` with an
        # explicit `missing` is the honest record; a string like "unknown"
        # would look like a value.
        self.model = {
            "id": model or str(doc.get("model") or "closed-model"),
            "provider": "unknown",
            "revision": None,
        }

        events: list[Event] = [
            self.event(
                EventType.RUN_STARTED,
                fidelity=Fidelity.DETERMINISTIC,
                ts=self.next_tick(),
                payload=self.corpus_payload(
                    **self.ordered(
                        title=doc.get("title"),
                        publisher=(doc.get("source") or {}).get("publisher"),
                        n_turns=len(doc["turns"]),
                        model_internal=False,
                        placement_possible="text_embedder_only",
                        note=(
                            "closed model: no checkpoint, no revision, no "
                            "residual stream. A placement for this run can only "
                            "come from a text embedder and must be drawn with "
                            "the NOT-model-internal grammar (rule R7); if no "
                            "embedder is reachable the placement is missing."
                        ),
                        **field_fidelity(
                            model_revision=Fidelity.MISSING,
                            placement=Fidelity.MISSING,
                        ),
                    )
                ),
            )
        ]

        for i, turn in enumerate(doc["turns"]):
            events.extend(self._one_turn(i, turn))

        self.turn_id = None
        events.append(self.session_completed())
        events.append(self.missing_usage())
        events.append(
            self.event(
                EventType.RUN_COMPLETED,
                fidelity=Fidelity.DETERMINISTIC,
                ts=self.next_tick(),
                payload=self.corpus_payload(
                    **self.ordered(
                        outcome="unknown",
                        model_internal=False,
                        **field_fidelity(outcome=Fidelity.MISSING),
                    )
                ),
            )
        )
        return [
            CorpusRun(
                run_id=rid,
                session_id=rid,
                events=events,
                corpus=self.corpus,
                meta=self.run_meta(
                    path,
                    title=doc.get("title"),
                    n_turns=len(doc["turns"]),
                    model_internal=False,
                ),
                warnings=list(self.warnings),
            )
        ]

    def _one_turn(self, i: int, turn: dict[str, Any]) -> list[Event]:
        role = str(turn.get("role") or "other").lower()
        text = str(turn.get("text") or "")
        self.turn_id = f"{self.run_id}:t{i}"
        ts = turn.get("ts")
        clock = float(ts) if isinstance(ts, (int, float)) else self.next_tick()
        ts_fid = Fidelity.NATIVE if isinstance(ts, (int, float)) else Fidelity.MISSING

        if role == "user":
            et = EventType.MESSAGE_USER
        elif role == "assistant":
            et = EventType.MESSAGE_ASSISTANT_COMPLETED
        elif role == "system":
            et = EventType.SESSION_STARTED
        else:
            # an unrecognised speaker is still a turn; it is recorded as a user
            # message with the speaker's own name kept, never promoted to the
            # assistant
            et = EventType.MESSAGE_USER

        body: dict[str, Any] = dict(
            turn_index=i,
            role=role,
            speaker=turn.get("speaker"),
            text=text,
            model_internal=False,
            **field_fidelity(ts=ts_fid, model_internal=Fidelity.DETERMINISTIC),
        )
        # a transcript without per-turn timestamps is an ordering, and says so
        payload = self.corpus_payload(
            **(body if ts_fid is Fidelity.NATIVE else self.ordered(**body))
        )
        return [
            self.event(
                et,
                fidelity=Fidelity.NATIVE,
                ts=clock,
                payload=payload,
                native_type=f"transcript.{role}",
            )
        ]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "transcript"


__all__ = [
    "ROLES",
    "TRANSCRIPT",
    "TranscriptCorpusAdapter",
    "parse_markdown",
    "read_transcript",
]
