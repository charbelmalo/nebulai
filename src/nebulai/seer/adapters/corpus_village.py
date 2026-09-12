"""AI Village → canonical events (Attractors D7, episode #8).

Source: `huggingface.co/datasets/aidigestorg/ai-village` — AI Digest's
long-running village of frontier agents with their own computers, a shared
group chat, compressed memories and open-ended goals.

**This adapter ships no data and this repository must never contain any.**
Three separate reasons, all of which hold at once:

* The dataset is **gated** (`gated: manual` on the Hub). Access is reviewed by
  a human, and as of this writing the credentials available here do not have
  it: `resolve/main/*` answers 401 and the Dataset Viewer answers 404.
* Its licence is `other` / `ai-village-research-terms`: research use only, no
  training or fine-tuning without written permission, no re-identification, and
  citation of AI Digest / AI Village in any resulting work.
* It contains a year of real interactions with the real world, including with
  people who are not part of the experiment.

So `load()` reads from a path the operator passes at run time, and when the
data is not reachable it **refuses, by name**: `VillageUnavailable` carries the
dataset id, the licence, the gate, and the sentence about what agreeing to the
terms means. It does not fall back to a sample, and there is no bundled
excerpt to fall back to. `tests/fixtures/corpus/village-synthetic.jsonl` is
hand-written by us in the documented shape and says so in its first record;
mapping tests run against that.

Shape. The dataset publishes one gzipped JSONL per config; the ones this
adapter maps are `computer_use_turns` (an agent's turn at its own computer),
`chat_messages` (the group chat), `events` (village-level happenings) and
`agent_memories`. Because the schema is behind the gate, every field is located
through a **candidate-key probe** rather than a hard-coded name, and a row whose
required fields cannot be located is counted in `n_unmapped` and warned about —
never filled in with a plausible guess. Running this against the real dataset
for the first time is expected to produce warnings; those warnings are the
adapter telling you which names it did not recognise, which is the correct
behaviour for a schema nobody here has seen.
"""

from __future__ import annotations

import datetime as _dt
import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from ..contract import Action, Effect, Event, EventType, Fidelity
from .corpus_base import (
    CorpusAdapter,
    CorpusRun,
    CorpusSource,
    classify_shell,
    field_fidelity,
)

VILLAGE = CorpusSource(
    id="ai_village",
    name="AI Village (AI Digest)",
    url="https://huggingface.co/datasets/aidigestorg/ai-village",
    licence="other / ai-village-research-terms (gated: manual approval)",
    ships_in_repo=False,
    note=(
        "research use only; no training or fine-tuning without written "
        "permission; no re-identification; cite AI Digest / AI Village. "
        "Loaded from a runtime path only — nothing from this dataset is "
        "committed to this repository, and no excerpt of it is used as a test "
        "fixture."
    ),
)

TERMS = (
    "The AI Village dataset is gated on the Hugging Face Hub and released under "
    "ai-village-research-terms: research and analysis only, no training or "
    "fine-tuning of AI systems without written permission, no attempt to "
    "re-identify individuals, and citation of AI Digest / AI Village in any "
    "resulting work. Request access at "
    "https://huggingface.co/datasets/aidigestorg/ai-village and pass the "
    "downloaded file's path explicitly."
)


class VillageUnavailable(RuntimeError):
    """The dataset is not present locally, and this adapter will not invent it.

    Raised instead of falling back to a sample so that a figure can never be
    drawn from data we do not have. The message names the dataset, the gate and
    the licence terms.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        where = f" (looked for {path})" if path else ""
        super().__init__(
            f"AI Village data not available{where}. {TERMS} This adapter has no "
            f"bundled copy and no offline fallback: refusing rather than "
            f"substituting anything."
        )


# ── field probing ────────────────────────────────────────────────────────────
# The schema is behind the gate. Each logical field lists the names it may
# plausibly carry, most specific first. `probe` returns (value, key) so the
# name that matched is recorded in the run meta and a future schema change is
# visible rather than silent.

FIELDS: dict[str, tuple[str, ...]] = {
    "id": ("id", "turn_id", "message_id", "event_id", "uuid"),
    "session": ("session_id", "computer_use_session_id", "thread_id", "conversation_id"),
    "agent": ("agent", "agent_id", "agent_name", "author", "sender", "name"),
    "model": ("model", "model_id", "model_name"),
    "ts": ("timestamp", "ts", "created_at", "time", "datetime", "started_at"),
    "text": ("text", "content", "message", "body", "output", "response"),
    "action": ("action", "tool", "tool_name", "command", "type", "action_type"),
    "village": ("village", "village_id", "day", "date"),
    "goal": ("goal", "village_goal", "objective"),
}


def probe(row: dict[str, Any], field: str) -> tuple[Any, str | None]:
    """First present candidate key for `field`, and which key that was."""
    for key in FIELDS.get(field, ()):
        if key in row and row[key] not in (None, ""):
            return row[key], key
    return None, None


def parse_ts(value: Any) -> float | None:
    """Epoch seconds from an int, float, or ISO-8601 string; None otherwise."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        v = float(value)
        # milliseconds if it is far past any plausible epoch-second date
        return v / 1000.0 if v > 1e11 else v
    if isinstance(value, str):
        s = value.strip().replace("Z", "+00:00")
        try:
            return _dt.datetime.fromisoformat(s).timestamp()
        except ValueError:
            return None
    return None


def open_rows(path: Path | str) -> Iterator[dict[str, Any]]:
    """Rows from a `.jsonl` or `.jsonl.gz` config file."""
    p = Path(path)
    if not p.exists():
        raise VillageUnavailable(p)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8", errors="replace") as fh:  # type: ignore[operator]
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


class VillageCorpusAdapter(CorpusAdapter):
    """Maps AI Village rows into per-agent sessions inside a village run."""

    agent = "ai_village"
    adapter_name = "corpus_village"
    corpus = VILLAGE

    def __init__(self, *, run_id: str = "", session_id: str = "", **kw: Any) -> None:
        super().__init__(
            run_id=run_id or "ai-village", session_id=session_id or "ai-village", **kw
        )
        #: which candidate key actually matched each logical field, so a schema
        #: drift shows up in the artifact instead of as silent `None`s
        self.resolved_keys: dict[str, str] = {}

    def read(
        self,
        path: Path | str,
        *,
        kind: str = "computer_use_turns",
        agents: Sequence[str] | None = None,
        max_rows: int | None = None,
        run_id: str | None = None,
    ) -> list[CorpusRun]:
        """Map one config file. Refuses if the file is not there."""
        p = Path(path)
        if not p.exists():
            raise VillageUnavailable(p)
        rid = run_id or f"village-{p.name.split('.')[0]}"
        self.run_id = rid
        self.session_id = rid
        self._tick = 0.0

        rows: list[dict[str, Any]] = []
        for row in open_rows(p):
            if agents:
                who, _ = probe(row, "agent")
                if who is None or str(who) not in agents:
                    continue
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
        if not rows:
            raise VillageUnavailable(p)

        events: list[Event] = [
            self.event(
                EventType.RUN_STARTED,
                fidelity=Fidelity.DETERMINISTIC,
                ts=self._first_ts(rows) or self.next_tick(),
                payload=self.corpus_payload(
                    kind=kind,
                    n_rows=len(rows),
                    terms=TERMS,
                    **field_fidelity(terms=Fidelity.NATIVE),
                ),
            )
        ]
        seen: set[str] = set()
        for row in rows:
            events.extend(self._one_row(row, kind, seen))
        self.session_id = rid
        self.turn_id = None
        events.append(self.session_completed())
        events.append(self.missing_usage())
        events.append(
            self.event(
                EventType.RUN_COMPLETED,
                fidelity=Fidelity.DETERMINISTIC,
                ts=self._last_ts(rows) or self.next_tick(),
                payload=self.corpus_payload(
                    outcome="unknown",
                    note="the village is ongoing; a file is a slice, not an end",
                    **field_fidelity(outcome=Fidelity.MISSING),
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
                    p,
                    kind=kind,
                    n_rows=len(rows),
                    n_agents=len(seen),
                    resolved_keys=dict(self.resolved_keys),
                ),
                warnings=list(self.warnings),
            )
        ]

    # ── one row ──────────────────────────────────────────────────────────

    def _remember(self, field: str, key: str | None) -> None:
        if key is not None:
            self.resolved_keys.setdefault(field, key)

    def _first_ts(self, rows: list[dict[str, Any]]) -> float | None:
        for r in rows:
            v, k = probe(r, "ts")
            self._remember("ts", k)
            t = parse_ts(v)
            if t is not None:
                return t
        return None

    def _last_ts(self, rows: list[dict[str, Any]]) -> float | None:
        for r in reversed(rows):
            t = parse_ts(probe(r, "ts")[0])
            if t is not None:
                return t
        return None

    def _one_row(
        self, row: dict[str, Any], kind: str, seen: set[str]
    ) -> list[Event]:
        who, who_key = probe(row, "agent")
        self._remember("agent", who_key)
        ts_raw, ts_key = probe(row, "ts")
        self._remember("ts", ts_key)
        text, text_key = probe(row, "text")
        self._remember("text", text_key)
        act_raw, act_key = probe(row, "action")
        self._remember("action", act_key)
        model, model_key = probe(row, "model")
        self._remember("model", model_key)

        if who is None and text is None and act_raw is None:
            return self.unmapped(
                f"village row with no recognised agent/text/action key: "
                f"{sorted(row)[:8]}"
            )

        name = str(who) if who is not None else "unknown"
        self.session_id = f"{self.run_id}:{_slug(name)}"
        if model is not None:
            self.model = {"id": str(model), "provider": "unknown"}
        ts = parse_ts(ts_raw)
        clock = ts if ts is not None else self.next_tick()
        ts_fid = Fidelity.NATIVE if ts is not None else Fidelity.MISSING

        out: list[Event] = []
        if name not in seen:
            seen.add(name)
            out.append(
                self.event(
                    EventType.SESSION_STARTED,
                    fidelity=Fidelity.DETERMINISTIC,
                    ts=clock,
                    payload=self.corpus_payload(
                        agent=name, **field_fidelity(ts=ts_fid)
                    ),
                )
            )

        if kind == "chat_messages":
            # the group chat: an agent addressing the other agents
            out.append(
                self.event(
                    EventType.MESSAGE_ASSISTANT_COMPLETED,
                    fidelity=Fidelity.NATIVE if text is not None else Fidelity.MISSING,
                    ts=clock,
                    action=Action.INTERACT,
                    payload=self.corpus_payload(
                        agent=name,
                        text=text,
                        channel="village_chat",
                        **field_fidelity(
                            text=Fidelity.NATIVE if text is not None else Fidelity.MISSING,
                            ts=ts_fid,
                        ),
                    ),
                    native_type=f"village.{kind}",
                )
            )
            return out

        # computer-use turns and generic events: an action against the world
        verb = str(act_raw) if act_raw is not None else ""
        action, effect = (
            classify_shell(verb) if verb else (Action.EXECUTE, Effect.UNKNOWN)
        )
        out.append(
            self.event(
                EventType.TOOL_COMPLETED,
                fidelity=Fidelity.DETERMINISTIC,
                ts=clock,
                action=action,
                effect=effect,
                payload=self.corpus_payload(
                    agent=name,
                    verb=verb or None,
                    text=text,
                    exit_code=None,
                    duration_ms=None,
                    **field_fidelity(
                        verb=Fidelity.NATIVE if verb else Fidelity.MISSING,
                        exit_code=Fidelity.MISSING,
                        duration_ms=Fidelity.MISSING,
                        effect=Fidelity.MISSING,
                        ts=ts_fid,
                    ),
                ),
                native_type=f"village.{kind}",
            )
        )
        return out


def _slug(s: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in s]
    return "".join(keep).strip("-") or "agent"


__all__ = [
    "FIELDS",
    "TERMS",
    "VILLAGE",
    "VillageCorpusAdapter",
    "VillageUnavailable",
    "open_rows",
    "parse_ts",
    "probe",
]
