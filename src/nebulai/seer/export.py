"""Export a run to something that outlives SessionSeer.

Three formats, in decreasing order of how much they preserve:

* **JSONL** — the event log verbatim, one canonical `Event` per line. Lossless
  by construction: it is the same bytes the store holds, and re-reading it
  reproduces the run exactly, which is what makes every number here checkable.
* **Parquet** — one row per event, with the payload kept as a JSON string in a
  single column. Columnar for the fields you group by (agent, action, effect,
  fidelity, capture mode) and lossless for the rest. This is the format a
  cross-run study actually loads.
* **CSV** — flattened spans, not events. For a spreadsheet; explicitly the
  lossy one, and it says so in its own header comment rather than pretending
  that a run is a table of tool calls.

The one rule this module adds: **fidelity travels with the value**. Every
exported row carries the source fidelity and capture mode of the event it came
from, so an analysis run in pandas six months from now cannot accidentally
average an `estimated` duration into a `native` one without having been told.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Iterable, Sequence

from .analysis import analyze
from .contract import Event
from .redaction import (
    RULESET,
    ContentLevel,
    rank,
    redact_event,
    redact_view_dict,
)
from .reducer import RunView

FORMATS = ("jsonl", "parquet", "csv")

CONTENT_TYPE = {
    "jsonl": "application/x-ndjson; charset=utf-8",
    "parquet": "application/vnd.apache.parquet",
    "csv": "text/csv; charset=utf-8",
}

#: The flat event table. Payload stays JSON because flattening it would either
#: explode to hundreds of mostly-null columns or silently drop the fields a
#: future question needs.
EVENT_COLUMNS = (
    "event_id", "ts", "mono_ns", "run_id", "session_id", "turn_id",
    "span_id", "parent_span_id", "event_type", "native_type",
    "action", "effect", "agent", "agent_version", "adapter",
    "capture_mode", "fidelity", "is_delta", "payload_json",
)

SPAN_COLUMNS = (
    "run_id", "agent", "capture_mode", "span_id", "parent_span_id", "action",
    "native_type", "started_at", "ended_at", "duration_s", "effect", "failed",
    "detail",
)


def event_row(e: Event) -> dict[str, Any]:
    return {
        "event_id": e.event_id,
        "ts": e.ts,
        "mono_ns": e.mono_ns,
        "run_id": e.run_id,
        "session_id": e.session_id,
        "turn_id": e.turn_id,
        "span_id": e.span_id,
        "parent_span_id": e.parent_span_id,
        "event_type": e.event_type.value,
        "native_type": e.native_type,
        "action": e.action.value if e.action else None,
        "effect": e.effect.value if e.effect else None,
        "agent": e.source.agent,
        "agent_version": e.source.agent_version,
        "adapter": e.source.adapter,
        "capture_mode": e.source.capture_mode.value,
        "fidelity": e.source.fidelity.value,
        # carried as a column so a `group by` in pandas cannot accidentally
        # fold streaming fragments into counts — rule 2, exported
        "is_delta": e.event_type.is_delta,
        "payload_json": json.dumps(e.payload, separators=(",", ":"), default=str),
    }


def to_jsonl(events: Iterable[Event]) -> bytes:
    return ("".join(e.to_json() + "\n" for e in events)).encode("utf-8")


def to_parquet(events: Sequence[Event]) -> bytes:
    """Columnar export. Raises `RuntimeError` when pyarrow is absent — the
    caller turns that into an honest 501 rather than silently downgrading to
    CSV, which would hand back a different (lossy) thing under the same name."""
    try:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "parquet export needs pyarrow, which is not installed in this "
            "environment; `pip install pyarrow`, or export jsonl"
        ) from exc

    rows = [event_row(e) for e in events]
    cols = {c: [r[c] for r in rows] for c in EVENT_COLUMNS}
    # Explicit schema: an all-null column would otherwise land as `null` type
    # and a later concat of two runs would fail on a type mismatch — which is
    # exactly the case (one agent reports turns, another does not).
    schema = pa.schema([
        ("event_id", pa.string()), ("ts", pa.float64()), ("mono_ns", pa.int64()),
        ("run_id", pa.string()), ("session_id", pa.string()), ("turn_id", pa.string()),
        ("span_id", pa.string()), ("parent_span_id", pa.string()),
        ("event_type", pa.string()), ("native_type", pa.string()),
        ("action", pa.string()), ("effect", pa.string()), ("agent", pa.string()),
        ("agent_version", pa.string()), ("adapter", pa.string()),
        ("capture_mode", pa.string()), ("fidelity", pa.string()),
        ("is_delta", pa.bool_()), ("payload_json", pa.string()),
    ])
    table = pa.Table.from_pydict(cols, schema=schema)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    return buf.getvalue()


def to_csv(view: RunView, redact: ContentLevel | None = None) -> bytes:
    """Spans, flattened. The lossy one, labelled as such in the file itself."""
    out = io.StringIO()
    out.write(
        "# SessionSeer span export — LOSSY. One row per completed span; "
        "usage, approvals, prompts and data-quality are not represented. "
        "Export jsonl or parquet for the full log.\n"
    )
    if redact is not None:
        out.write(
            f"# Redacted to content level '{redact.value}' under ruleset "
            f"{RULESET}. A blank cell here may be a redaction, not an absence.\n"
        )
    hide_detail = redact is not None and rank(redact) < rank(ContentLevel.COMMAND)
    w = csv.DictWriter(out, fieldnames=list(SPAN_COLUMNS), extrasaction="ignore")
    w.writeheader()
    for s in view.spans:
        w.writerow({
            "run_id": view.run_id,
            "agent": view.agent,
            "capture_mode": view.quality.capture_mode,
            "span_id": s.span_id,
            "parent_span_id": s.parent_span_id,
            "action": s.action.value if s.action else None,
            "native_type": s.native_type,
            "started_at": s.started_at,
            "ended_at": s.ended_at,
            "duration_s": s.duration_s,
            "effect": s.effect.value if s.effect else None,
            "failed": s.failed,
            "detail": "" if hide_detail else (s.detail or "")[:200],
        })
    return out.getvalue().encode("utf-8")


def to_analysis_json(
    view: RunView, events: Sequence[Event], redact: ContentLevel | None = None
) -> bytes:
    """The analyses, with their formulas and evidence, as one JSON document.

    The analyses run on the *unredacted* events on purpose: every one of them
    reads counts, actions and timing, never prose, so redacting first would
    change nothing except to make a shared export disagree with the run it came
    from. What redaction touches is the `view` this document carries alongside
    them.
    """
    doc = analyze(view, events)
    d = view.to_dict()
    doc["view"] = d if redact is None else redact_view_dict(d, redact)
    if redact is not None:
        doc["redaction"] = {"content_level": redact.value, "ruleset": RULESET}
    return json.dumps(doc, indent=2, default=str).encode("utf-8")


def redact_events(
    events: Sequence[Event], keep: ContentLevel
) -> list[Event]:
    return [Event.from_dict(redact_event(e.to_dict(), keep)) for e in events]


def export(
    fmt: str,
    view: RunView,
    events: Sequence[Event],
    redact: ContentLevel | None = None,
) -> tuple[bytes, str, str]:
    """`(body, content_type, filename)` for one run in one format.

    `redact` takes the export down to a content level; the filename says so,
    because a redacted export that is indistinguishable from a complete one
    will eventually be read as a complete one.
    """
    fmt = (fmt or "jsonl").lower()
    stem = view.run_id if redact is None else f"{view.run_id}-{redact.value}"
    if fmt == "jsonl":
        body = to_jsonl(events if redact is None else redact_events(events, redact))
        return body, CONTENT_TYPE["jsonl"], f"{stem}.jsonl"
    if fmt == "parquet":
        body = to_parquet(events if redact is None else redact_events(events, redact))
        return body, CONTENT_TYPE["parquet"], f"{stem}.parquet"
    if fmt == "csv":
        return to_csv(view, redact), CONTENT_TYPE["csv"], f"{stem}-spans.csv"
    if fmt == "analysis":
        return (to_analysis_json(view, events, redact),
                "application/json; charset=utf-8", f"{stem}-analysis.json")
    raise ValueError(
        f"unknown export format {fmt!r}; expected one of {', '.join(FORMATS)}, analysis"
    )


__all__ = [
    "CONTENT_TYPE",
    "EVENT_COLUMNS",
    "FORMATS",
    "SPAN_COLUMNS",
    "event_row",
    "export",
    "redact_events",
    "to_analysis_json",
    "to_csv",
    "to_jsonl",
    "to_parquet",
]
