"""The comparability gate — the part that refuses to answer.

Cross-agent comparison is the reason to build SessionSeer and the easiest thing
in it to get wrong, because every wrong answer looks like a right one. Two runs
of the same task, one Codex and one Claude, produce two `output_tokens` numbers.
Subtracting them yields a plausible percentage that a research scientist could
put in a slide. It would be meaningless: Claude bills reasoning *inside*
`output_tokens` and Codex reports `reasoning_output_tokens` *beside* it, so the
two fields do not name the same quantity. Codex, meanwhile, has no cache-write
bucket at all, so "cache efficiency" cannot be computed for it in the terms
Claude reports.

So this module's job is to decide, per metric, whether a comparison is
meaningful, and to say *why not* when it isn't. A refusal is a result. It is
displayed with the same weight as a number, because "these two runs cannot be
compared on tokens" is genuinely the finding.

Three grounds for refusal:

1. **Category mismatch** — the agents' native token buckets do not line up.
2. **Fidelity mismatch** — one side's number is `ESTIMATED` or `MISSING` and the
   other's is `NATIVE`. A native-vs-estimated delta reports the estimator's
   error as if it were a difference between the agents.
3. **Capture gap** — a mode that cannot observe the thing being compared. A
   DRIVEN Codex run sees no approval requests, so its `interact` count is zero
   for reasons that have nothing to do with how the agent behaved.

What survives all three is worth trusting, and it is not a small list: wall
clock, action mix, edit churn, verification coverage, file counts and outcome
are all computed by *our* reducer from canonical events, identically for every
agent. Those are the honest cross-agent metrics, and they happen to be the ones
a researcher asks about first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contract import Action, Fidelity, TokenCategory
from .reducer import Measured, RunView


@dataclass(slots=True)
class Refusal:
    metric: str
    reason: str
    #: which runs caused it, so the UI can point at the culprit instead of
    #: greying the whole row out
    runs: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "reason": self.reason,
            "runs": self.runs,
            "detail": self.detail,
        }


@dataclass(slots=True)
class MetricRow:
    metric: str
    label: str
    unit: str
    values: dict[str, Measured]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "label": self.label,
            "unit": self.unit,
            "values": {k: m.to_dict() for k, m in self.values.items()},
        }


@dataclass(slots=True)
class Comparison:
    runs: list[str]
    agents: dict[str, str]
    comparable: list[MetricRow] = field(default_factory=list)
    refused: list[Refusal] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "agents": self.agents,
            "comparable": [m.to_dict() for m in self.comparable],
            "refused": [r.to_dict() for r in self.refused],
        }

    def metric(self, name: str) -> MetricRow | None:
        return next((m for m in self.comparable if m.metric == name), None)

    def refusal(self, name: str) -> Refusal | None:
        return next((r for r in self.refused if r.metric == name), None)


#: Capture gaps → the metrics they invalidate. The gap strings are the ones the
#: adapters emit (`MISSING_IN_EXEC_JSON`, `MISSING_IN_ONESHOT`), matched as
#: substrings so a reworded gap still lands on the right metrics rather than
#: silently passing the gate.
GAP_BLOCKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("approval", ("action.interact", "time.waiting_permission")),
    ("tool call", (
        "action.inspect", "action.search", "action.edit", "action.execute",
        "action.verify", "action.vcs", "action.delegate",
        "verification_after_last_edit", "n_files_changed", "time.tool_running",
    )),
    ("per-turn boundaries", ("n_turns",)),
    ("reasoning span", ("time.model_running",)),
    ("context-window", ("context_window",)),
    ("per-request model timing", ("time.model_running",)),
)


def _blocked_metrics(view: RunView) -> dict[str, str]:
    """metric → the gap text that blocks it, for this run."""
    out: dict[str, str] = {}
    for gap in view.quality.capture_gaps:
        low = gap.lower()
        for needle, metrics in GAP_BLOCKS:
            if needle in low:
                for m in metrics:
                    out.setdefault(m, gap)
    return out


def compare(views: list[RunView]) -> Comparison:
    """Compare runs metric by metric, refusing where the comparison is not
    meaningful. Refusing on one metric never suppresses the others — a run pair
    with incomparable tokens usually still has perfectly comparable timing."""
    if len(views) < 2:
        raise ValueError("comparison needs at least two runs")

    cmp = Comparison(
        runs=[v.run_id for v in views],
        agents={v.run_id: v.agent for v in views},
    )
    blocked = {v.run_id: _blocked_metrics(v) for v in views}

    def emit(metric: str, label: str, unit: str, per_run: dict[str, Measured]) -> None:
        culprits = [rid for rid in per_run if metric in blocked[rid]]
        if culprits:
            cmp.refused.append(
                Refusal(
                    metric=metric,
                    reason=(
                        f"capture gap: {blocked[culprits[0]][metric]} is not observable "
                        f"in {'/'.join(sorted({cmp.agents[c] for c in culprits}))}'s "
                        "capture mode"
                    ),
                    runs=culprits,
                )
            )
            return
        absent = [rid for rid, m in per_run.items() if m.absent]
        if absent and len(absent) < len(per_run):
            cmp.refused.append(
                Refusal(
                    metric=metric,
                    reason="present for some runs and absent for others",
                    runs=absent,
                    detail={
                        rid: per_run[rid].note or per_run[rid].fidelity.value
                        for rid in absent
                    },
                )
            )
            return
        if absent:
            cmp.refused.append(
                Refusal(metric=metric, reason="absent for every run", runs=absent)
            )
            return
        fids = {m.fidelity for m in per_run.values()}
        if len(fids) > 1 and Fidelity.NATIVE in fids:
            cmp.refused.append(
                Refusal(
                    metric=metric,
                    reason=(
                        "fidelity mismatch: "
                        + ", ".join(
                            f"{rid}={m.fidelity.value}" for rid, m in per_run.items()
                        )
                        + " — a native-vs-estimated delta reports the estimator's "
                        "error as a difference between the agents"
                    ),
                    runs=list(per_run),
                )
            )
            return
        cmp.comparable.append(MetricRow(metric, label, unit, per_run))

    _timing(views, emit)
    _behaviour(views, emit)
    _tokens(views, cmp, emit)
    _cost(views, emit)
    return cmp


# ── metric families ──────────────────────────────────────────────────────────


def _timing(views: list[RunView], emit) -> None:
    emit(
        "wall_clock_s",
        "Wall clock",
        "s",
        {
            v.run_id: (
                Measured(v.ended_at - v.started_at, Fidelity.DETERMINISTIC)
                if v.ended_at and v.started_at
                else Measured(None, Fidelity.MISSING, "run has not ended")
            )
            for v in views
        },
    )
    states = sorted({s for v in views for s in v.time_in_state})
    for state in states:
        emit(
            f"time.{state}",
            f"Time in {state.replace('_', ' ')}",
            "s",
            {
                v.run_id: Measured(
                    v.time_in_state.get(state, 0.0), Fidelity.DETERMINISTIC
                )
                for v in views
            },
        )


def _behaviour(views: list[RunView], emit) -> None:
    for action in Action:
        emit(
            f"action.{action.value}",
            f"{action.value.title()} calls",
            "count",
            {
                v.run_id: Measured(
                    v.action_counts.get(action.value, 0), Fidelity.DETERMINISTIC
                )
                for v in views
            },
        )
    emit(
        "n_files_changed",
        "Files changed",
        "count",
        {v.run_id: Measured(len(v.files_changed), Fidelity.DETERMINISTIC) for v in views},
    )
    emit(
        "n_turns",
        "Turns",
        "count",
        {v.run_id: Measured(v.n_turns, Fidelity.DETERMINISTIC) for v in views},
    )
    emit(
        "verification_after_last_edit",
        "Verified after last edit",
        "bool",
        {v.run_id: v.verification_after_last_edit() for v in views},
    )
    emit(
        "context_window",
        "Context window",
        "tokens",
        {v.run_id: v.context_window for v in views},
    )


def _tokens(views: list[RunView], cmp: Comparison, emit) -> None:
    """Tokens, gated on category alignment before anything else.

    The alignment check runs across the whole family rather than per category:
    if one agent folds reasoning into output and another breaks it out, then
    `output` is not comparable *even though both runs have an `output` number*.
    Checking per category would let exactly that through.
    """
    buckets = {v.run_id: {k for k, m in v.usage.items() if not m.absent} for v in views}
    distinct = {frozenset(b) for b in buckets.values()}
    if len(distinct) > 1:
        cmp.refused.append(
            Refusal(
                metric="tokens.*",
                reason=(
                    "token categories do not align across agents — "
                    + "; ".join(
                        f"{cmp.agents[rid]} reports {sorted(b) or 'nothing'}"
                        for rid, b in buckets.items()
                    )
                    + ". Subtracting fields that do not name the same quantity "
                    "produces a confident wrong number."
                ),
                runs=list(buckets),
                detail={rid: sorted(b) for rid, b in buckets.items()},
            )
        )
        return

    for cat in TokenCategory:
        per_run = {
            v.run_id: v.usage.get(
                cat.value, Measured(None, Fidelity.MISSING, "not reported")
            )
            for v in views
        }
        if all(m.absent for m in per_run.values()):
            continue  # nothing to say; not a refusal worth showing
        emit(f"tokens.{cat.value}", f"{cat.value.replace('_', ' ').title()} tokens",
             "tokens", per_run)


def _cost(views: list[RunView], emit) -> None:
    emit("cost_usd", "Cost", "USD", {v.run_id: v.cost_usd for v in views})


def summarize_refusals(cmp: Comparison) -> str:
    """One-paragraph plain-language account, for the CLI and for the top of the
    compare panel. Written to be readable by someone who did not build this."""
    if not cmp.refused:
        return "All requested metrics are comparable across these runs."
    lines = [
        f"{len(cmp.comparable)} metrics are comparable across these runs; "
        f"{len(cmp.refused)} are not:"
    ]
    for r in cmp.refused:
        lines.append(f"  · {r.metric}: {r.reason}")
    return "\n".join(lines)
