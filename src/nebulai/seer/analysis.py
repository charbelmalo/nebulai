"""Derived analyses — every one of them showing its work.

The `RunView` is a fold: counts, spans, time in state. This module is the layer
above it, and it is the layer where a research instrument usually stops being
one. A number like "churn ratio 3.4" or "loop score 0.8" is trivial to compute
and almost impossible to check, so everything here carries four things beside
the value:

* a **version**, so a metric recomputed next month is comparably labelled;
* a **formula**, in words, that says what was divided by what;
* the **inputs** it read, named as event types and view fields;
* **evidence** — the actual spans and events the number came from, so the
  reader can disagree with it.

Two rules from the rest of SessionSeer carry through unchanged:

1. **A refusal is a result.** An analysis that cannot run says why, in the same
   shape as one that can. `edit_churn` on a Codex run reports "this agent's file
   change events carry no line counts", not `0.0`. That is the same rule as the
   comparability gate refusing a cross-agent token delta, and the same rule as a
   namer with `n_labeled == 0` saying so.

2. **Overlapping time is never summed.** Parallel tool calls and subagents mean
   span durations overlap; adding them produces a total larger than the run,
   which is the single most common way an agent dashboard lies. The headline is
   always the *union* of intervals, and the amount naive summing would have
   added is reported beside it as `double_counted_s` rather than hidden.

There are no scores here. `loop_rules` counts rule matches and cites the events;
`progress_evidence` refuses a headline number entirely, because a checklist that
collapses to "68% done" is a worse instrument than the checklist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Iterable, Sequence

from .contract import Action, Effect, Event, EventType, Fidelity, Outcome
from .reducer import Measured, RunView, SpanRecord, missing

#: Bumped when a formula changes in a way that makes new values incomparable
#: with old ones. Individual analyses carry their own version too — most
#: changes touch one of them, and re-versioning all seven would make the whole
#: history look discontinuous.
ANALYSES_VERSION = "1.0"


# ── shapes ───────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Evidence:
    """One citation. `ref` is a span id, event id or path — whatever the reader
    would need to find the thing in the log."""

    kind: str
    ref: str
    ts: float | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "ref": self.ref, "ts": self.ts, "detail": self.detail}


@dataclass(slots=True)
class Analysis:
    key: str
    label: str
    version: str
    formula: str
    inputs: list[str]
    headline: Measured
    unit: str = ""
    #: named sub-measurements, each with its own fidelity — a run can know its
    #: approval count exactly and its approval wait not at all
    parts: dict[str, Measured] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    #: set when the analysis could not run. The headline is `missing` then, and
    #: the UI renders the sentence instead of a dash.
    refusal: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "version": self.version,
            "formula": self.formula,
            "inputs": self.inputs,
            "headline": self.headline.to_dict(),
            "unit": self.unit,
            "parts": {k: m.to_dict() for k, m in self.parts.items()},
            "rows": self.rows,
            "evidence": [e.to_dict() for e in self.evidence],
            "refusal": self.refusal,
        }


def _refuse(key: str, label: str, version: str, formula: str,
            inputs: list[str], why: str, unit: str = "") -> Analysis:
    return Analysis(
        key=key, label=label, version=version, formula=formula, inputs=inputs,
        headline=missing(why), unit=unit, refusal=why,
    )


# ── interval arithmetic (the anti-double-counting core) ──────────────────────


def _union(intervals: Iterable[tuple[float, float]]) -> float:
    """Total length covered by at least one interval.

    The whole reason `time_decomposition` can be trusted. Two tools running for
    3s each in the same 3s window occupy 3 seconds of the run, not 6.
    """
    merged: list[list[float]] = []
    for a, b in sorted(i for i in intervals if i[1] > i[0]):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return sum(b - a for a, b in merged)


def _max_concurrency(intervals: Sequence[tuple[float, float]]) -> int:
    """Most spans open at once. `1` means nothing ever overlapped, and is the
    reader's cue that inclusive and union time should agree."""
    edges: list[tuple[float, int]] = []
    for a, b in intervals:
        if b > a:
            edges.append((a, 1))
            edges.append((b, -1))
    edges.sort()
    cur = best = 0
    for _, d in edges:
        cur += d
        best = max(best, cur)
    return best


def _closed(spans: Sequence[SpanRecord]) -> list[SpanRecord]:
    return [s for s in spans if s.ended_at is not None and s.ended_at >= s.started_at]


def _iv(spans: Iterable[SpanRecord]) -> list[tuple[float, float]]:
    return [(s.started_at, s.ended_at) for s in spans if s.ended_at is not None]


# ── capture-gap vocabulary ───────────────────────────────────────────────────

#: The adapters declare their gaps as sentences (`"approval requests/decisions"`,
#: `"context-window pressure"`). Matching on substrings keeps one vocabulary
#: instead of a parallel enum that would drift out of step with the sentences
#: the data-quality panel already shows the user.
_GAP_TOPICS = {
    "approvals": ("approval",),
    "context": ("context-window", "context window", "compaction"),
    "tokens": ("token usage",),
    "model_timing": ("model timing",),
}


def _blocked(view: RunView, topic: str) -> str | None:
    """The declared capture gap covering `topic`, if the run has one."""
    needles = _GAP_TOPICS[topic]
    for gap in view.quality.capture_gaps:
        low = gap.lower()
        if any(n in low for n in needles):
            return gap
    return None


def _worst_span_fidelity(events: Sequence[Event]) -> Fidelity:
    """The least trustworthy fidelity on any span-closing event.

    A run whose clock was whole-seconds has estimated durations, and a duration
    total built from estimates is an estimate. Reporting it as `deterministic`
    because the arithmetic was exact would be exactly the wrong lesson.
    """
    order = [Fidelity.NATIVE, Fidelity.DETERMINISTIC, Fidelity.ESTIMATED, Fidelity.HEURISTIC]
    worst = Fidelity.DETERMINISTIC
    for e in events:
        if e.event_type not in (
            EventType.TOOL_COMPLETED,
            EventType.TOOL_FAILED,
            EventType.MODEL_REQUEST_COMPLETED,
        ):
            continue
        f = e.source.fidelity
        if f in order and order.index(f) > order.index(worst):
            worst = f
    return worst


# ── 1. time decomposition ────────────────────────────────────────────────────


def time_decomposition(view: RunView, events: Sequence[Event]) -> Analysis:
    key, ver = "time_decomposition", "1.0"
    label = "Time decomposition"
    formula = (
        "wall = last_event − first_event; in_spans = union(span intervals); "
        "outside_spans = wall − in_spans; double_counted = Σ durations − in_spans; "
        "per-action self time = span duration − union(direct children)"
    )
    inputs = ["tool.started", "tool.completed", "tool.failed",
              "model.request_started", "model.request_completed", "RunView.time_in_state"]

    if view.started_at is None or view.last_event_at is None:
        return _refuse(key, label, ver, formula, inputs,
                       "the run has no events, so it has no duration", "s")

    end = view.ended_at if view.ended_at is not None else view.last_event_at
    wall = max(0.0, end - view.started_at)
    fid = _worst_span_fidelity(events)

    closed = _closed(view.spans)
    unclosed = [s for s in view.spans if s.ended_at is None]
    ivs = _iv(closed)
    in_spans = _union(ivs)
    inclusive = sum((s.ended_at - s.started_at) for s in closed)  # type: ignore[operator]

    # self time: a parent's own seconds, with its children's subtracted once
    by_parent: dict[str, list[SpanRecord]] = {}
    for s in closed:
        if s.parent_span_id:
            by_parent.setdefault(s.parent_span_id, []).append(s)
    self_time: dict[str, float] = {}
    for s in closed:
        dur = (s.ended_at or s.started_at) - s.started_at
        kids = _union(_iv(by_parent.get(s.span_id, [])))
        self_time[s.span_id] = max(0.0, dur - kids)

    rows: list[dict[str, Any]] = []
    for action in sorted({(s.action.value if s.action else "unclassified") for s in closed}):
        group = [s for s in closed
                 if (s.action.value if s.action else "unclassified") == action]
        rows.append({
            "action": action,
            "n": len(group),
            "wall_s": round(_union(_iv(group)), 4),
            "inclusive_s": round(sum((s.ended_at - s.started_at) for s in group), 4),  # type: ignore[operator]
            "self_s": round(sum(self_time[s.span_id] for s in group), 4),
            "failed": sum(1 for s in group if s.failed),
        })
    rows.sort(key=lambda r: -r["wall_s"])

    parts = {
        "wall_s": Measured(round(wall, 4), Fidelity.DETERMINISTIC),
        "in_spans_s": Measured(round(in_spans, 4), fid),
        "outside_spans_s": Measured(
            round(max(0.0, wall - in_spans), 4), fid,
            "the model thinking, and the human reading — no span covers it",
        ),
        "double_counted_s": Measured(
            round(max(0.0, inclusive - in_spans), 4), fid,
            "what summing span durations would have added on top of the run",
        ),
        "max_concurrency": Measured(_max_concurrency(ivs), Fidelity.DETERMINISTIC),
    }
    parts["unclosed_spans"] = (
        Measured(len(unclosed), Fidelity.DETERMINISTIC,
                 "still open at the end of the log; contributing no time")
        if unclosed else Measured(0, Fidelity.DETERMINISTIC)
    )

    ev = [Evidence("span", s.span_id, s.started_at,
                   f"{(s.action.value if s.action else '?')}: {(s.detail or '')[:80]}")
          for s in sorted(closed, key=lambda s: -((s.ended_at or 0) - s.started_at))[:5]]

    return Analysis(
        key=key, label=label, version=ver, formula=formula, inputs=inputs,
        headline=Measured(round(wall, 4), Fidelity.DETERMINISTIC),
        unit="s", parts=parts, rows=rows, evidence=ev,
    )


# ── 2. verification coverage ─────────────────────────────────────────────────

#: file kind → (suffixes, verifier patterns). Deliberately small: an unmatched
#: suffix reports "no rule for this file type" instead of being scored, because
#: a rule that silently covers everything cannot fail to find coverage.
_PROJECT_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("python", (".py", ".pyi"),
     ("pytest", "python -m pytest", "unittest", "ruff", "mypy", "pyright", "tox", "nox")),
    ("typescript", (".ts", ".tsx", ".mts", ".cts"),
     ("tsc", "vitest", "jest", "eslint", "vue-tsc", "playwright", "npm test", "npm run test")),
    ("javascript", (".js", ".jsx", ".mjs", ".cjs"),
     ("vitest", "jest", "mocha", "eslint", "node --test", "npm test", "npm run test")),
    ("rust", (".rs",), ("cargo test", "cargo check", "cargo clippy", "cargo build")),
    ("go", (".go",), ("go test", "go build", "go vet", "golangci-lint")),
    ("swift", (".swift",), ("swift test", "swift build", "xcodebuild")),
    ("shell", (".sh", ".zsh", ".bash"), ("shellcheck", "bash -n", "sh -n")),
)


def _file_kind(path: str) -> str | None:
    suf = PurePosixPath(path).suffix.lower()
    for kind, sufs, _ in _PROJECT_RULES:
        if suf in sufs:
            return kind
    return None


def _expected(kind: str) -> tuple[str, ...]:
    for k, _, pats in _PROJECT_RULES:
        if k == kind:
            return pats
    return ()


def verification_coverage(view: RunView, events: Sequence[Event]) -> Analysis:
    key, ver = "verification_coverage", "1.0"
    label = "Verification coverage"
    formula = (
        "for each changed file: does a VERIFY span that started after the last "
        "EDIT span run a command this file's language rule recognises? "
        "Files whose type has no rule are counted as unknown, not as uncovered."
    )
    inputs = ["tool.completed(action=edit)", "tool.completed(action=verify)",
              "file.changed", "taxonomy._PROJECT_RULES"]

    edits = [s for s in view.spans if s.action is Action.EDIT]
    if not edits and not view.files_changed:
        return _refuse(key, label, ver, formula, inputs,
                       "nothing was edited, so there is nothing to verify", "files")

    last_edit = max((s.started_at for s in edits), default=None)
    verifies = [s for s in view.spans if s.action is Action.VERIFY]
    after = [s for s in verifies
             if last_edit is None or s.started_at >= last_edit]

    # A verify span with no command text cannot be matched against a rule.
    blind = [s for s in after if not s.detail]

    rows: list[dict[str, Any]] = []
    covered = unknown = 0
    for path in view.files_changed:
        kind = _file_kind(path)
        if kind is None:
            unknown += 1
            rows.append({"path": path, "kind": None, "expected": [], "observed": [],
                         "covered": None,
                         "note": "no verification rule for this file type"})
            continue
        pats = _expected(kind)
        hits = [s for s in after
                if s.detail and any(p in s.detail for p in pats)]
        if hits:
            covered += 1
        rows.append({
            "path": path, "kind": kind, "expected": list(pats),
            "observed": [(s.detail or "")[:120] for s in hits],
            "covered": bool(hits),
            "passed": (None if not hits else not any(s.failed for s in hits)),
        })

    n_files = len(view.files_changed)
    n_rulable = n_files - unknown

    parts: dict[str, Measured] = {
        "files_changed": Measured(n_files, Fidelity.DETERMINISTIC),
        # deterministic: "a verify span closed after the last edit" is a fact
        # about the log, with no pattern matching in it
        "verified_after_last_edit": (
            Measured(int(bool(after)), Fidelity.DETERMINISTIC)
            if last_edit is not None
            else missing("no edit spans, so 'after the last edit' has no meaning")
        ),
        "verify_spans_after_last_edit": Measured(len(after), Fidelity.DETERMINISTIC),
        "files_with_no_rule": Measured(unknown, Fidelity.DETERMINISTIC),
        "last_verification_passed": (
            Measured(int(not after[-1].failed), Fidelity.DETERMINISTIC)
            if after else missing("no verification ran after the last edit")
        ),
    }
    if blind:
        parts["verify_spans_without_a_command"] = Measured(
            len(blind), Fidelity.DETERMINISTIC,
            "captured as verification but with no command text to match a rule against",
        )

    ev = [Evidence("span", s.span_id, s.started_at, (s.detail or "")[:120])
          for s in after[:6]]
    if last_edit is not None and edits:
        newest = max(edits, key=lambda s: s.started_at)
        ev.insert(0, Evidence("span", newest.span_id, newest.started_at,
                              f"last edit: {(newest.detail or '')[:80]}"))

    if n_rulable == 0:
        head = missing("no changed file has a verification rule for its type")
    else:
        # heuristic: matching a command string against a pattern list is an
        # interpretation, and `npm test` can run anything at all
        head = Measured(covered, Fidelity.HEURISTIC,
                        f"of {n_rulable} changed "
                        f"{'file' if n_rulable == 1 else 'files'} with a known rule")

    return Analysis(
        key=key, label=label, version=ver, formula=formula, inputs=inputs,
        headline=head, unit="files", parts=parts, rows=rows, evidence=ev,
    )


# ── 3. edit churn ────────────────────────────────────────────────────────────


def edit_churn(view: RunView, events: Sequence[Event]) -> Analysis:
    key, ver = "edit_churn", "1.0"
    label = "Edit churn"
    formula = "churn_ratio = cumulative_lines_written / max(final_lines, 1), per file and summed"
    inputs = ["file.changed{lines_added,lines_removed,total_lines}", "RunView.file_stats"]

    if not view.file_stats:
        return _refuse(key, label, ver, formula, inputs,
                       "no file changes were captured in this run", "×")

    rows: list[dict[str, Any]] = []
    cum = fin = 0
    with_lines = 0
    for path, st in sorted(view.file_stats.items(), key=lambda kv: -kv[1]["edits"]):
        row: dict[str, Any] = {"path": path, "edits": st["edits"]}
        if st.get("line_data"):
            with_lines += 1
            written = int(st["lines_added"]) + int(st["lines_removed"])
            row["lines_written"] = written
            row["lines_added"] = st["lines_added"]
            row["lines_removed"] = st["lines_removed"]
            row["total_lines"] = st.get("total_lines")
            row["total_lines_from"] = st.get("total_lines_from")
            if st.get("total_lines"):
                row["churn_ratio"] = round(written / max(int(st["total_lines"]), 1), 3)
                cum += written
                fin += int(st["total_lines"])
        else:
            row["note"] = "this agent's file-change events carry no line counts"
        rows.append(row)

    # Always available, and it needs no line counts at all: how many times the
    # run came back to a file it had already edited. Named for what it is —
    # calling it churn would let a number with different units wear churn's name.
    n_edit_events = sum(int(st["edits"]) for st in view.file_stats.values())
    revisits = n_edit_events - len(view.file_stats)
    parts: dict[str, Measured] = {
        "files_touched": Measured(len(view.file_stats), Fidelity.DETERMINISTIC),
        "edit_events": Measured(n_edit_events, Fidelity.DETERMINISTIC),
        "revisits": Measured(
            revisits, Fidelity.DETERMINISTIC,
            "edits beyond the first on a file already edited in this run",
        ),
        "edits_per_file": Measured(
            round(n_edit_events / max(len(view.file_stats), 1), 3), Fidelity.DETERMINISTIC
        ),
    }

    if fin == 0:
        head = missing(
            "no captured file change reported line counts, so cumulative and "
            "final lines are both unknown"
            if with_lines == 0
            else "line counts were captured but no file's final length is known "
                 "(a whole-file write is what establishes it)"
        )
    else:
        # `total_lines` is exact right after a whole-file write and accumulated
        # from deltas afterwards, so the denominator is an estimate in general.
        exact = all(r.get("total_lines_from") == "write"
                    for r in rows if r.get("total_lines"))
        head = Measured(round(cum / max(fin, 1), 3),
                        Fidelity.DETERMINISTIC if exact else Fidelity.ESTIMATED)

    ev = [Evidence("file", r["path"], None, f"{r['edits']} edits")
          for r in rows[:5]]
    return Analysis(
        key=key, label=label, version=ver, formula=formula, inputs=inputs,
        headline=head, unit="×", parts=parts, rows=rows, evidence=ev,
    )


# ── 4. loop rules ────────────────────────────────────────────────────────────


def _sig(s: SpanRecord) -> str:
    return f"{s.action.value if s.action else '?'}::{(s.detail or '').strip()}"


def loop_rules(view: RunView, events: Sequence[Event]) -> Analysis:
    key, ver = "loop_rules", "1.0"
    label = "Loop rules"
    formula = (
        "four rules counted independently over the closed spans in order. "
        "The headline is the number of rule matches, not a loop score: "
        "a match is a citation to check, not a verdict."
    )
    inputs = ["tool.completed", "tool.failed", "file.changed", "Effect.no_new_information"]

    spans = sorted(_closed(view.spans), key=lambda s: s.started_at)
    # Two spans is the floor: one action cannot repeat anything. Three would be
    # the floor for the streak rule alone, and that rule reports its own state
    # per row — raising the gate to suit it would silence the other three.
    if len(spans) < 2:
        return _refuse(key, label, ver, formula, inputs,
                       f"only {len(spans)} completed spans — nothing can repeat yet",
                       "matches")

    rows: list[dict[str, Any]] = []
    ev: list[Evidence] = []
    total = 0

    # R1 — the agent's own effect label. Present only if some adapter set it.
    labelled = [s for s in spans
                if s.effect in (Effect.NEW_INFORMATION, Effect.NO_NEW_INFORMATION)]
    if not labelled:
        rows.append({
            "rule": "no_new_information_streak",
            "description": "3+ consecutive actions the agent labelled as surfacing nothing new",
            "hits": None, "fidelity": Fidelity.MISSING.value,
            "note": "no span in this run carries an information effect; "
                    "this agent's adapter does not label them",
            "evidence": [],
        })
    else:
        hits, run_ids, streak = 0, [], []
        for s in spans:
            if s.effect is Effect.NO_NEW_INFORMATION:
                streak.append(s)
            else:
                if len(streak) >= 3:
                    hits += 1
                    run_ids.append([x.span_id for x in streak])
                streak = []
        if len(streak) >= 3:
            hits += 1
            run_ids.append([x.span_id for x in streak])
        total += hits
        rows.append({
            "rule": "no_new_information_streak",
            "description": "3+ consecutive actions the agent labelled as surfacing nothing new",
            "hits": hits, "fidelity": Fidelity.DETERMINISTIC.value,
            "evidence": run_ids,
        })

    # R2 — a look at something already looked at, with no edit to it in between.
    # Deterministic without reading any output: if nothing changed the target,
    # a second identical read cannot have returned anything new.
    seen: dict[str, float] = {}
    edited_at: dict[str, list[float]] = {}
    for p, _st in view.file_stats.items():
        edited_at.setdefault(p, [])
    for e in events:
        if e.event_type is EventType.FILE_CHANGED and e.payload.get("path"):
            edited_at.setdefault(str(e.payload["path"]), []).append(e.ts)

    r2: list[dict[str, Any]] = []
    for s in spans:
        if s.action not in (Action.INSPECT, Action.SEARCH) or not s.detail:
            continue
        sig = _sig(s)
        prev = seen.get(sig)
        if prev is not None:
            target = s.detail.strip()
            changed = any(prev <= t <= s.started_at for t in edited_at.get(target, []))
            if not changed:
                r2.append({"span_id": s.span_id, "ts": s.started_at,
                           "detail": target[:120], "previous_at": prev})
        seen[sig] = s.started_at
    total += len(r2)
    rows.append({
        "rule": "repeat_read_without_change",
        "description": "the same target inspected or searched again with no edit to it in between",
        "hits": len(r2), "fidelity": Fidelity.DETERMINISTIC.value,
        "evidence": r2[:20],
    })
    ev += [Evidence("span", str(h["span_id"]), h["ts"], str(h["detail"])) for h in r2[:3]]

    # R3 — the same command failing more than once.
    fails: dict[str, list[SpanRecord]] = {}
    for s in spans:
        if s.failed and s.detail:
            fails.setdefault(_sig(s), []).append(s)
    r3 = [{"signature": k.split("::", 1)[1][:120], "failures": len(v),
           "span_ids": [x.span_id for x in v]}
          for k, v in fails.items() if len(v) >= 2]
    total += len(r3)
    rows.append({
        "rule": "repeated_failure",
        "description": "the same command failed 2+ times",
        "hits": len(r3), "fidelity": Fidelity.DETERMINISTIC.value,
        "evidence": r3[:20],
    })
    ev += [Evidence("span", h["span_ids"][0], None, str(h["signature"])) for h in r3[:3]]

    # R4 — edit → failing verify → edit → failing verify …
    cycles, cur = 0, 0
    cyc_ev: list[str] = []
    pending_edit = False
    for s in spans:
        if s.action is Action.EDIT:
            pending_edit = True
        elif s.action is Action.VERIFY and pending_edit:
            pending_edit = False
            if s.failed:
                cur += 1
                cyc_ev.append(s.span_id)
                if cur >= 3:
                    cycles += 1
                    cur = 0
            else:
                cur = 0
                cyc_ev.clear()
    total += cycles
    rows.append({
        "rule": "edit_verify_thrash",
        "description": "3+ consecutive edit→verify cycles where the verification failed",
        "hits": cycles, "fidelity": Fidelity.DETERMINISTIC.value,
        "evidence": cyc_ev[:20],
    })

    return Analysis(
        key=key, label=label, version=ver, formula=formula, inputs=inputs,
        headline=Measured(total, Fidelity.DETERMINISTIC,
                          "rule matches to inspect — not a score"),
        unit="matches", rows=rows, evidence=ev[:8],
    )


# ── 5. human intervention burden ─────────────────────────────────────────────

_WAIT_STATES = ("waiting_permission", "waiting_clarification", "waiting_user")


def intervention_burden(view: RunView, events: Sequence[Event]) -> Analysis:
    key, ver = "intervention_burden", "1.0"
    label = "Human intervention burden"
    formula = (
        "wait = Σ time in waiting_* states; approval_wait = Σ(resolved_at − requested_at) "
        "over approvals that resolved; prompts after the first are counted as "
        "corrections, not as the task."
    )
    inputs = ["approval.requested", "approval.resolved", "clarification.requested",
              "clarification.resolved", "message.user", "RunView.time_in_state"]

    gap = _blocked(view, "approvals")

    req: list[Event] = []
    res: list[Event] = []
    clar_req = clar_res = 0
    prompts: list[Event] = []
    for e in events:
        t = e.event_type
        if t is EventType.APPROVAL_REQUESTED:
            req.append(e)
        elif t is EventType.APPROVAL_RESOLVED:
            res.append(e)
        elif t is EventType.CLARIFICATION_REQUESTED:
            clar_req += 1
        elif t is EventType.CLARIFICATION_RESOLVED:
            clar_res += 1
        elif t is EventType.MESSAGE_USER:
            prompts.append(e)

    # pair in order: an approval prompt is modal, so the next resolution is this
    # one's. Unpaired requests are reported rather than assigned a wait of 0.
    waits: list[float] = []
    for i, r in enumerate(req):
        if i < len(res) and res[i].ts >= r.ts:
            waits.append(res[i].ts - r.ts)
    unresolved = max(0, len(req) - len(waits))

    wait_s = sum(view.time_in_state.get(s, 0.0) for s in _WAIT_STATES)
    denied = sum(1 for e in res if (e.payload.get("decision") or "").lower()
                 in ("deny", "denied", "reject", "rejected", "no"))

    def _count(n: int, note: str | None = None) -> Measured:
        if gap:
            return missing(f"capture gap: {gap}")
        return Measured(n, Fidelity.DETERMINISTIC, note)

    parts = {
        "approvals_requested": _count(len(req)),
        "approvals_denied": _count(denied),
        "approvals_unresolved": _count(
            unresolved, "requested but never resolved in the log" if unresolved else None
        ),
        "approval_wait_s": (
            missing(f"capture gap: {gap}") if gap
            else Measured(round(sum(waits), 3), Fidelity.DETERMINISTIC)
            if waits else missing("no approval both requested and resolved in this run")
        ),
        "clarifications_requested": _count(clar_req),
        "clarifications_resolved": _count(clar_res),
        "user_prompts": Measured(len(prompts), Fidelity.DETERMINISTIC),
        "corrections": Measured(
            max(0, len(prompts) - 1), Fidelity.DETERMINISTIC,
            "prompts after the first — the first one is the task",
        ),
        "waiting_state_s": Measured(round(wait_s, 3), Fidelity.DETERMINISTIC),
    }

    rows = [
        {"kind": "approval", "ts": r.ts,
         "detail": str(r.payload.get("tool") or r.payload.get("command") or "")[:120],
         "resolved_after_s": round(waits[i], 3) if i < len(waits) else None,
         "decision": (res[i].payload.get("decision") if i < len(res) else None)}
        for i, r in enumerate(req[:25])
    ]
    rows += [
        {"kind": "prompt", "ts": p.ts, "detail": f"{p.payload.get('chars', '?')} chars",
         "correction": i > 0}
        for i, p in enumerate(prompts[:25])
    ]
    rows.sort(key=lambda r: r["ts"])

    total_wait = wait_s + (sum(waits) if not gap else 0.0)
    head = Measured(round(total_wait, 3), Fidelity.DETERMINISTIC,
                    f"approvals not observable here: {gap}" if gap else None)

    return Analysis(
        key=key, label=label, version=ver, formula=formula, inputs=inputs,
        headline=head, unit="s", parts=parts, rows=rows,
        evidence=[Evidence("event", e.event_id, e.ts, e.event_type.value)
                  for e in (req + prompts)[:6]],
    )


# ── 6. context pressure ──────────────────────────────────────────────────────


def context_pressure(view: RunView, events: Sequence[Event]) -> Analysis:
    key, ver = "context_pressure", "1.0"
    label = "Context pressure"
    formula = (
        "compactions counted from context.compaction_* events; before/after token "
        "counts taken verbatim from the agent where it reports them and left "
        "missing where it does not."
    )
    inputs = ["context.compaction_started", "context.compaction_completed",
              "context.pressure_updated", "RunView.context_window"]

    gap = _blocked(view, "context")
    started = [e for e in events if e.event_type is EventType.COMPACTION_STARTED]
    done = [e for e in events if e.event_type is EventType.COMPACTION_COMPLETED]
    pressure = [e for e in events if e.event_type is EventType.CONTEXT_PRESSURE_UPDATED]

    # An agent whose hooks fire only *before* compaction reports starts and no
    # completions. Counting completions alone would report zero compactions for
    # a session that compacted four times.
    n = max(len(done), len(started))

    if n == 0 and not pressure:
        why = (f"capture gap: {gap}" if gap else
               "no compaction or context-pressure signal of any kind reached this "
               "run, so a zero here would be a guess about the capture, not a "
               "fact about the session")
        a = _refuse(key, label, ver, formula, inputs, why, "compactions")
        a.parts["context_window"] = view.context_window
        return a

    rows: list[dict[str, Any]] = []
    for e in sorted(started + done, key=lambda x: x.ts):
        p = e.payload
        rows.append({
            "ts": e.ts,
            "phase": "started" if e.event_type is EventType.COMPACTION_STARTED else "completed",
            "trigger": p.get("trigger"),
            "tokens_before": p.get("tokens_before"),
            "tokens_after": p.get("tokens_after"),
        })

    before = [r["tokens_before"] for r in rows if r["tokens_before"] is not None]
    after = [r["tokens_after"] for r in rows if r["tokens_after"] is not None]
    peaks = [e.payload.get("used") or e.payload.get("tokens")
             for e in pressure if (e.payload.get("used") or e.payload.get("tokens"))]

    parts = {
        "compactions": Measured(n, Fidelity.DETERMINISTIC),
        "auto_compactions": Measured(
            sum(1 for r in rows if (r.get("trigger") or "") == "auto"),
            Fidelity.DETERMINISTIC,
        ),
        "tokens_reclaimed": (
            Measured(sum(before) - sum(after), Fidelity.NATIVE)
            if before and after and len(before) == len(after)
            else missing("this agent does not report tokens across a compaction")
        ),
        "peak_context_tokens": (
            Measured(max(peaks), Fidelity.NATIVE) if peaks
            else missing("no context-pressure event carried a token count")
        ),
        "context_window": view.context_window,
    }
    return Analysis(
        key=key, label=label, version=ver, formula=formula, inputs=inputs,
        headline=Measured(n, Fidelity.DETERMINISTIC), unit="compactions",
        parts=parts, rows=rows,
        evidence=[Evidence("event", e.event_id, e.ts, e.event_type.value)
                  for e in (started + done)[:6]],
    )


# ── 7. progress evidence ─────────────────────────────────────────────────────

_COMMIT = re.compile(r"\bgit\s+(commit|revert)\b")


def _n(count: int, noun: str) -> str:
    """`1 edit spans` reads as a broken counter rather than a count of one."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"

#: Outcomes in which the agent itself asserted the work was finished. A
#: verified pass is *not* here — that is an evaluator's claim, and the two must
#: never collapse (contract.Outcome says the same thing from the other side).
_CLAIMED = frozenset({Outcome.AGENT_CLAIMED_COMPLETE, Outcome.UNVERIFIED_COMPLETE})


def progress_evidence(view: RunView, events: Sequence[Event]) -> Analysis:
    key, ver = "progress_evidence", "1.0"
    label = "Progress evidence"
    formula = (
        "a checklist of things that either happened or did not, each with the "
        "span or event that shows it. There is deliberately no headline number: "
        "the items are not commensurable and a percentage of them would not "
        "measure progress on the task."
    )
    inputs = ["RunView.spans", "RunView.outcome", "RunView.state",
              "tool.completed(action=vcs)", "evaluation.completed"]

    edits = [s for s in view.spans if s.action is Action.EDIT]
    verifies = [s for s in view.spans if s.action is Action.VERIFY]
    last_edit = max((s.started_at for s in edits), default=None)
    after = [s for s in verifies if last_edit is not None and s.started_at >= last_edit]
    commits = [s for s in view.spans
               if s.action is Action.VCS and s.detail and _COMMIT.search(s.detail)]
    evals = [e for e in events if e.event_type is EventType.EVALUATION_COMPLETED]

    def item(name: str, status: bool | None, why: str,
             cite: list[Evidence], fidelity: Fidelity) -> dict[str, Any]:
        return {
            "item": name,
            "status": "yes" if status is True else "no" if status is False else "unknown",
            "why": why,
            "fidelity": fidelity.value,
            "evidence": [c.to_dict() for c in cite],
        }

    rows = [
        item("files were edited", bool(edits),
             f"{_n(len(edits), 'edit span')}, "
             f"{_n(len(view.files_changed), 'distinct file')}",
             [Evidence("span", s.span_id, s.started_at, (s.detail or "")[:80])
              for s in edits[:3]], Fidelity.DETERMINISTIC),
        item("verification ran at all", bool(verifies),
             f"{_n(len(verifies), 'verify span')}",
             [Evidence("span", s.span_id, s.started_at, (s.detail or "")[:80])
              for s in verifies[:3]], Fidelity.DETERMINISTIC),
        item("verification ran after the last edit",
             None if last_edit is None else bool(after),
             "no edits to be after" if last_edit is None
             else f"{_n(len(after), 'verify span')} started at or after the last edit",
             [Evidence("span", s.span_id, s.started_at, (s.detail or "")[:80])
              for s in after[:3]], Fidelity.DETERMINISTIC),
        item("that verification passed",
             None if not after else not any(s.failed for s in after),
             "no verification after the last edit" if not after
             else f"{sum(1 for s in after if s.failed)} of {len(after)} failed",
             [Evidence("span", s.span_id, s.started_at, (s.detail or "")[:80])
              for s in after if s.failed][:3], Fidelity.DETERMINISTIC),
        item("changes were committed", bool(commits) if view.spans else None,
             f"{_n(len(commits), 'git commit/revert span')}",
             [Evidence("span", s.span_id, s.started_at, (s.detail or "")[:80])
              for s in commits[:3]], Fidelity.HEURISTIC),
        item("an external evaluator reported", bool(evals),
             f"{_n(len(evals), 'evaluation.completed event')}",
             [Evidence("event", e.event_id, e.ts, str(e.payload.get("result", "")))
              for e in evals[:3]], Fidelity.DETERMINISTIC),
        # Two things this must not do: read "the outcome is known" as "the
        # agent claimed success" (`agent_gave_up` is also a known outcome), and
        # read an *absent* claim as a denial. Hooks carry no completion claim at
        # all, so `unknown` there is a gap in the capture, not a "no".
        item("the agent said it was done",
             None if view.outcome is Outcome.UNKNOWN else view.outcome in _CLAIMED,
             f"outcome: {view.outcome.value}", [],
             Fidelity.MISSING if view.outcome is Outcome.UNKNOWN else Fidelity.NATIVE),
        item("the run reached a terminal state",
             view.state.value in ("completed", "failed", "interrupted"),
             f"state: {view.state.value}", [], Fidelity.DETERMINISTIC),
    ]

    return Analysis(
        key=key, label=label, version=ver, formula=formula, inputs=inputs,
        headline=missing(
            "progress does not have a single number here; read the checklist"
        ),
        unit="", rows=rows,
        parts={
            "items_yes": Measured(sum(1 for r in rows if r["status"] == "yes"),
                                  Fidelity.DETERMINISTIC),
            "items_unknown": Measured(sum(1 for r in rows if r["status"] == "unknown"),
                                      Fidelity.DETERMINISTIC),
            "items_total": Measured(len(rows), Fidelity.DETERMINISTIC),
        },
    )


# ── the set ──────────────────────────────────────────────────────────────────

ANALYSES = (
    time_decomposition,
    verification_coverage,
    edit_churn,
    loop_rules,
    intervention_burden,
    context_pressure,
    progress_evidence,
)


def analyze(view: RunView, events: Sequence[Event]) -> dict[str, Any]:
    """Run every analysis over one run. Never raises for one bad analysis: a
    crash in `edit_churn` must not take the six that worked off the page, so a
    failure is reported in the same shape as a refusal."""
    out: list[dict[str, Any]] = []
    for fn in ANALYSES:
        try:
            out.append(fn(view, events).to_dict())
        except Exception as exc:  # noqa: BLE001
            out.append(
                _refuse(
                    fn.__name__, fn.__name__.replace("_", " ").title(), "0",
                    "—", [], f"this analysis raised {type(exc).__name__}: {exc}",
                ).to_dict()
            )
    return {
        "run_id": view.run_id,
        "agent": view.agent,
        "capture_mode": view.quality.capture_mode,
        "analyses_version": ANALYSES_VERSION,
        "n_events": view.n_events,
        "analyses": out,
    }


__all__ = [
    "ANALYSES",
    "ANALYSES_VERSION",
    "Analysis",
    "Evidence",
    "analyze",
    "context_pressure",
    "edit_churn",
    "intervention_burden",
    "loop_rules",
    "progress_evidence",
    "time_decomposition",
    "verification_coverage",
]
