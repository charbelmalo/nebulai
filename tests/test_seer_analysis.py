"""M4 exit tests: every derived value is deterministic, versioned and cited.

The plan's exit criterion is one sentence — *every derived value deterministic
from the log and carrying a version, formula, inputs, and evidence* — and it
has three teeth:

* **Nothing is summed across overlapping spans.** The single most common way an
  agent dashboard lies is adding two parallel tool calls together and reporting
  more busy seconds than the run lasted. `test_two_parallel_tools_are_not_two_
  tool_seconds` is the whole rule.

* **A refusal is a result.** An analysis that cannot run returns the same shape
  as one that can, with the sentence in place of the number. Codex reports no
  line counts on file changes, so `edit_churn` on a Codex run must say so
  rather than report `0.0` — the same rule as the comparability gate refusing a
  cross-agent token delta.

* **The export is lossless where it claims to be.** JSONL round-trips to
  identical events; parquet carries the fidelity and capture mode of every row
  so a later `groupby` cannot average an estimate into a native number; CSV is
  lossy and says so in its own first line.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nebulai.seer.analysis import (
    ANALYSES,
    ANALYSES_VERSION,
    analyze,
    context_pressure,
    edit_churn,
    intervention_burden,
    loop_rules,
    progress_evidence,
    time_decomposition,
    verification_coverage,
)
from nebulai.seer.contract import (
    Action,
    CaptureMode,
    Effect,
    Event,
    EventType,
    Fidelity,
    Source,
    new_span_id,
)
from nebulai.seer.export import (
    EVENT_COLUMNS,
    FORMATS,
    export,
    to_csv,
    to_jsonl,
    to_parquet,
)
from nebulai.seer.reducer import reduce_run
from nebulai.seer.store import EventStore
from nebulai.seer.taxonomy import edit_extent

FIXTURES = Path(__file__).parent / "fixtures" / "seer"


# ── builders ─────────────────────────────────────────────────────────────────


def _src(**kw) -> Source:
    return Source(
        agent=kw.pop("agent", "claude"),
        agent_version="1",
        adapter="t",
        adapter_version="1",
        capture_mode=kw.pop("capture_mode", CaptureMode.DRIVEN),
        fidelity=kw.pop("fidelity", Fidelity.DETERMINISTIC),
    )


def ev(et: EventType, ts: float, **kw) -> Event:
    return Event(
        event_type=et,
        source=kw.pop("source", _src()),
        run_id=kw.pop("run_id", "run_1"),
        session_id="ses_1",
        ts=ts,
        mono_ns=int(ts * 1e9),
        **kw,
    )


def tool(start: float, end: float, action: Action, detail: str = "",
         *, failed: bool = False, span: str | None = None,
         parent: str | None = None, effect: Effect | None = None,
         fidelity: Fidelity = Fidelity.DETERMINISTIC) -> list[Event]:
    sid = span or new_span_id()
    return [
        ev(EventType.TOOL_STARTED, start, span_id=sid, parent_span_id=parent,
           action=action, native_type="Tool",
           payload={"tool": "Tool", "command": detail}),
        ev(EventType.TOOL_FAILED if failed else EventType.TOOL_COMPLETED, end,
           span_id=sid, parent_span_id=parent, action=action, effect=effect,
           native_type="Tool", source=_src(fidelity=fidelity),
           payload={"tool": "Tool", "command": detail}),
    ]


def file_changed(ts: float, path: str, **extent) -> Event:
    return ev(EventType.FILE_CHANGED, ts, action=Action.EDIT,
              effect=Effect.STATE_CHANGED,
              payload={"path": path, "kind": "update", **extent})


def view_of(events: list[Event]):
    return reduce_run("run_1", events, now=events[-1].ts)


def one(fn, events: list[Event]):
    return fn(view_of(events), events)


# ── 1. time decomposition ────────────────────────────────────────────────────


class TestTimeDecomposition:
    def test_two_parallel_tools_are_not_two_tool_seconds(self):
        """The rule the whole module exists for.

        Two tools that each ran for 3 seconds *in the same 3 seconds* occupy
        three seconds of the run. Summing them reports six, which is more than
        the run lasted, and every downstream ratio built on it is wrong.
        """
        a, b = new_span_id(), new_span_id()
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            *tool(1.0, 4.0, Action.SEARCH, "grep a", span=a),
            *tool(1.0, 4.0, Action.SEARCH, "grep b", span=b),
            ev(EventType.SESSION_COMPLETED, 5.0),
        ]
        r = one(time_decomposition, events)
        assert r.parts["in_spans_s"].value == pytest.approx(3.0)
        assert r.parts["double_counted_s"].value == pytest.approx(3.0)
        assert r.parts["max_concurrency"].value == 2

    def test_the_parts_add_back_up_to_the_run(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            *tool(1.0, 2.0, Action.INSPECT, "read"),
            *tool(3.0, 4.5, Action.EDIT, "write"),
            ev(EventType.SESSION_COMPLETED, 6.0),
        ]
        r = one(time_decomposition, events)
        wall = r.parts["wall_s"].value
        assert wall == pytest.approx(
            r.parts["in_spans_s"].value + r.parts["outside_spans_s"].value
        )
        assert r.headline.value == pytest.approx(wall)

    def test_a_subagents_time_is_not_billed_to_its_parent_twice(self):
        parent, child = new_span_id(), new_span_id()
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            *tool(1.0, 9.0, Action.DELEGATE, "subagent", span=parent),
            *tool(2.0, 6.0, Action.SEARCH, "grep", span=child, parent=parent),
            ev(EventType.SESSION_COMPLETED, 10.0),
        ]
        r = one(time_decomposition, events)
        by = {row["action"]: row for row in r.rows}
        # the delegate span lasted 8s and 4s of that was its child's
        assert by["delegate"]["inclusive_s"] == pytest.approx(8.0)
        assert by["delegate"]["self_s"] == pytest.approx(4.0)

    def test_a_coarse_clock_makes_the_total_estimated_not_deterministic(self):
        """Whole-second hook clocks give estimated spans. Exact arithmetic over
        estimates is still an estimate, and reporting it as deterministic would
        teach exactly the wrong lesson."""
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            *tool(1.0, 2.0, Action.INSPECT, "read", fidelity=Fidelity.ESTIMATED),
            ev(EventType.SESSION_COMPLETED, 3.0),
        ]
        r = one(time_decomposition, events)
        assert r.parts["in_spans_s"].fidelity is Fidelity.ESTIMATED

    def test_a_span_still_open_contributes_no_time_and_is_counted(self):
        sid = new_span_id()
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            ev(EventType.TOOL_STARTED, 1.0, span_id=sid, action=Action.EXECUTE,
               payload={"command": "sleep 900"}),
            ev(EventType.MESSAGE_USER, 2.0, action=Action.INTERACT, payload={}),
        ]
        r = one(time_decomposition, events)
        assert r.parts["unclosed_spans"].value == 1
        assert r.parts["in_spans_s"].value == pytest.approx(0.0)

    def test_a_run_with_no_events_refuses_rather_than_reporting_zero_seconds(self):
        from nebulai.seer.reducer import RunView

        r = time_decomposition(RunView(run_id="x"), [])
        assert r.refusal
        assert r.headline.fidelity is Fidelity.MISSING


# ── 2. verification coverage ─────────────────────────────────────────────────


class TestVerificationCoverage:
    def test_nothing_edited_is_a_refusal_not_zero_coverage(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            *tool(1.0, 2.0, Action.INSPECT, "cat README"),
            ev(EventType.SESSION_COMPLETED, 3.0),
        ]
        r = one(verification_coverage, events)
        assert r.refusal
        assert r.headline.value is None

    def test_a_test_run_before_the_last_edit_does_not_count(self):
        """The failure this catches: tested early, edited again, stopped. A
        plain `verified == True` calls that covered."""
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            *tool(1.0, 2.0, Action.EDIT, "/p/app.py"),
            file_changed(2.0, "/p/app.py", lines_added=3, lines_removed=1),
            *tool(3.0, 4.0, Action.VERIFY, "pytest -q"),
            *tool(5.0, 6.0, Action.EDIT, "/p/app.py"),
            file_changed(6.0, "/p/app.py", lines_added=2, lines_removed=0),
            ev(EventType.SESSION_COMPLETED, 7.0),
        ]
        r = one(verification_coverage, events)
        assert r.parts["verified_after_last_edit"].value == 0
        assert r.parts["last_verification_passed"].value is None
        assert r.headline.value == 0

    def test_a_file_type_with_no_rule_is_unknown_not_uncovered(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            *tool(1.0, 2.0, Action.EDIT, "/p/notes.txt"),
            file_changed(2.0, "/p/notes.txt"),
            ev(EventType.SESSION_COMPLETED, 3.0),
        ]
        r = one(verification_coverage, events)
        assert r.parts["files_with_no_rule"].value == 1
        assert r.rows[0]["covered"] is None
        # no rulable file → no coverage number at all, rather than 0 of 0
        assert r.headline.value is None

    def test_matching_a_command_to_a_language_rule_is_heuristic(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            *tool(1.0, 2.0, Action.EDIT, "/p/app.py"),
            file_changed(2.0, "/p/app.py"),
            *tool(3.0, 4.0, Action.VERIFY, "pytest tests/ -q"),
            ev(EventType.SESSION_COMPLETED, 5.0),
        ]
        r = one(verification_coverage, events)
        assert r.headline.value == 1
        # `npm test` can run anything at all; the match is an interpretation
        assert r.headline.fidelity is Fidelity.HEURISTIC
        # …but "a verify span closed after the last edit" is not
        assert r.parts["verified_after_last_edit"].fidelity is Fidelity.DETERMINISTIC

    def test_a_failing_verification_is_still_coverage_but_not_a_pass(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            *tool(1.0, 2.0, Action.EDIT, "/p/app.py"),
            file_changed(2.0, "/p/app.py"),
            *tool(3.0, 4.0, Action.VERIFY, "pytest -q", failed=True),
            ev(EventType.SESSION_COMPLETED, 5.0),
        ]
        r = one(verification_coverage, events)
        assert r.headline.value == 1
        assert r.parts["last_verification_passed"].value == 0


# ── 3. edit churn ────────────────────────────────────────────────────────────


class TestEditChurn:
    def test_an_agent_that_reports_no_line_counts_refuses_rather_than_zero(self):
        """Codex's `item.file_change` carries a path and a kind and nothing
        else. `0.0` would read as "this run rewrote nothing", which is a claim
        about the run rather than about the capture."""
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            file_changed(1.0, "/p/a.rs"),
            file_changed(2.0, "/p/a.rs"),
            ev(EventType.SESSION_COMPLETED, 3.0),
        ]
        r = one(edit_churn, events)
        assert r.headline.value is None
        assert r.headline.fidelity is Fidelity.MISSING
        assert "line counts" in (r.headline.note or "")

    def test_revisits_are_countable_without_any_line_data(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            file_changed(1.0, "/p/a.rs"),
            file_changed(2.0, "/p/a.rs"),
            file_changed(3.0, "/p/b.rs"),
            ev(EventType.SESSION_COMPLETED, 4.0),
        ]
        r = one(edit_churn, events)
        assert r.parts["files_touched"].value == 2
        assert r.parts["edit_events"].value == 3
        assert r.parts["revisits"].value == 1

    def test_the_ratio_is_written_lines_over_final_lines(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            file_changed(1.0, "/p/a.py", lines_added=40, lines_removed=0, total_lines=40),
            file_changed(2.0, "/p/a.py", lines_added=4, lines_removed=2),
            ev(EventType.SESSION_COMPLETED, 3.0),
        ]
        r = one(edit_churn, events)
        # written 40 + 6 = 46; final 40 + (4−2) = 42
        assert r.headline.value == pytest.approx(46 / 42, abs=1e-3)

    def test_a_final_length_carried_forward_from_a_write_is_estimated(self):
        after_write_only = [
            ev(EventType.SESSION_STARTED, 0.0),
            file_changed(1.0, "/p/a.py", lines_added=10, lines_removed=0, total_lines=10),
            ev(EventType.SESSION_COMPLETED, 2.0),
        ]
        assert one(edit_churn, after_write_only).headline.fidelity is Fidelity.DETERMINISTIC

        then_edited = [
            *after_write_only[:-1],
            file_changed(1.5, "/p/a.py", lines_added=2, lines_removed=1),
            ev(EventType.SESSION_COMPLETED, 2.0),
        ]
        assert one(edit_churn, then_edited).headline.fidelity is Fidelity.ESTIMATED

    def test_edits_with_no_write_leave_the_denominator_unknown(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            file_changed(1.0, "/p/a.py", lines_added=3, lines_removed=1),
            ev(EventType.SESSION_COMPLETED, 2.0),
        ]
        r = one(edit_churn, events)
        assert r.headline.value is None
        assert "final length" in (r.headline.note or "")


class TestEditExtent:
    def test_line_counts_survive_and_the_text_does_not(self):
        e = edit_extent("Edit", {"old_string": "a\nb", "new_string": "a\nb\nc\nd"})
        assert e == {"lines_added": 4, "lines_removed": 2}
        assert "a\nb" not in json.dumps(e)

    def test_only_a_whole_file_write_claims_to_know_the_file_length(self):
        assert edit_extent("Write", {"content": "x\ny\nz"})["total_lines"] == 3
        assert "total_lines" not in edit_extent("Edit", {"old_string": "x", "new_string": "y"})

    def test_a_tool_shape_with_no_line_information_returns_nothing(self):
        assert edit_extent("apply_patch", {"patch": "@@ -1 +1 @@"}) is None
        assert edit_extent("Edit", None) is None

    def test_multiedit_sums_its_edits(self):
        e = edit_extent("MultiEdit", {"edits": [
            {"old_string": "a", "new_string": "a\nb"},
            {"old_string": "c\nd", "new_string": "c"},
        ]})
        assert e == {"lines_added": 3, "lines_removed": 3}


# ── 4. loop rules ────────────────────────────────────────────────────────────


class TestLoopRules:
    def _rule(self, r, name):
        return next(row for row in r.rows if row["rule"] == name)

    def test_an_unlabelled_effect_rule_reports_missing_not_zero(self):
        """No adapter currently labels `no_new_information`. Reporting `0` for
        that rule would say "we looked and there were none"; the truth is that
        the rule could not run at all."""
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            *tool(1.0, 2.0, Action.SEARCH, "grep x"),
            *tool(3.0, 4.0, Action.SEARCH, "grep y"),
            *tool(5.0, 6.0, Action.SEARCH, "grep z"),
            ev(EventType.SESSION_COMPLETED, 7.0),
        ]
        row = self._rule(one(loop_rules, events), "no_new_information_streak")
        assert row["hits"] is None
        assert row["fidelity"] == Fidelity.MISSING.value

    def test_the_effect_rule_counts_a_streak_when_the_agent_does_label_it(self):
        events = [ev(EventType.SESSION_STARTED, 0.0)]
        for i in range(3):
            events += tool(1.0 + i, 1.5 + i, Action.SEARCH, f"grep {i}",
                           effect=Effect.NO_NEW_INFORMATION)
        events.append(ev(EventType.SESSION_COMPLETED, 9.0))
        row = self._rule(one(loop_rules, events), "no_new_information_streak")
        assert row["hits"] == 1

    def test_reading_the_same_file_twice_with_no_edit_between_is_a_hit(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            *tool(1.0, 2.0, Action.INSPECT, "/p/a.py"),
            *tool(3.0, 4.0, Action.INSPECT, "/p/a.py"),
            ev(EventType.SESSION_COMPLETED, 5.0),
        ]
        assert self._rule(one(loop_rules, events), "repeat_read_without_change")["hits"] == 1

    def test_reading_a_file_again_after_editing_it_is_not_a_loop(self):
        """The rule that keeps the metric usable: re-reading a file you just
        changed is the correct thing to do, and a repetition counter with no
        state model would flag it."""
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            *tool(1.0, 2.0, Action.INSPECT, "/p/a.py"),
            *tool(2.2, 2.8, Action.EDIT, "/p/a.py"),
            file_changed(2.8, "/p/a.py"),
            *tool(3.0, 4.0, Action.INSPECT, "/p/a.py"),
            ev(EventType.SESSION_COMPLETED, 5.0),
        ]
        assert self._rule(one(loop_rules, events), "repeat_read_without_change")["hits"] == 0

    def test_the_same_command_failing_twice_is_counted_and_cited(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            *tool(1.0, 2.0, Action.VERIFY, "pytest -q", failed=True),
            *tool(3.0, 4.0, Action.VERIFY, "pytest -q", failed=True),
            ev(EventType.SESSION_COMPLETED, 5.0),
        ]
        row = self._rule(one(loop_rules, events), "repeated_failure")
        assert row["hits"] == 1
        assert len(row["evidence"][0]["span_ids"]) == 2

    def test_the_headline_is_a_count_of_rule_matches_and_says_so(self):
        events = [ev(EventType.SESSION_STARTED, 0.0)]
        for i in range(4):
            events += tool(1.0 + i, 1.5 + i, Action.INSPECT, "/p/a.py")
        events.append(ev(EventType.SESSION_COMPLETED, 9.0))
        r = one(loop_rules, events)
        assert r.unit == "matches"
        assert "not a score" in (r.headline.note or "")

    def test_too_few_spans_refuses_rather_than_reporting_no_loops(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            *tool(1.0, 2.0, Action.INSPECT, "/p/a.py"),
            ev(EventType.SESSION_COMPLETED, 3.0),
        ]
        assert one(loop_rules, events).refusal


# ── 5. intervention burden ───────────────────────────────────────────────────


class TestInterventionBurden:
    def test_approval_wait_is_the_gap_between_request_and_resolution(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            ev(EventType.APPROVAL_REQUESTED, 1.0, action=Action.INTERACT,
               payload={"tool": "Bash"}),
            ev(EventType.APPROVAL_RESOLVED, 13.0, action=Action.INTERACT,
               payload={"decision": "allow"}),
            ev(EventType.SESSION_COMPLETED, 14.0),
        ]
        r = one(intervention_burden, events)
        assert r.parts["approval_wait_s"].value == pytest.approx(12.0)
        assert r.parts["approvals_requested"].value == 1

    def test_an_unresolved_approval_is_counted_not_given_a_wait_of_zero(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            ev(EventType.APPROVAL_REQUESTED, 1.0, action=Action.INTERACT, payload={}),
            ev(EventType.SESSION_INTERRUPTED, 5.0),
        ]
        r = one(intervention_burden, events)
        assert r.parts["approvals_unresolved"].value == 1
        assert r.parts["approval_wait_s"].value is None

    def test_a_declared_capture_gap_makes_approvals_missing_not_zero(self):
        """Codex `exec --json` cannot see approvals at all. A zero there would
        describe our capture and read as a fact about the session."""
        events = [
            ev(EventType.SESSION_STARTED, 0.0,
               payload={"capture_gaps": ["approval requests/decisions"]}),
            ev(EventType.SESSION_COMPLETED, 2.0),
        ]
        r = one(intervention_burden, events)
        assert r.parts["approvals_requested"].value is None
        assert r.parts["approvals_requested"].fidelity is Fidelity.MISSING

    def test_the_first_prompt_is_the_task_and_later_ones_are_corrections(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            ev(EventType.MESSAGE_USER, 1.0, action=Action.INTERACT, payload={"chars": 90}),
            ev(EventType.MESSAGE_USER, 9.0, action=Action.INTERACT, payload={"chars": 20}),
            ev(EventType.SESSION_COMPLETED, 10.0),
        ]
        r = one(intervention_burden, events)
        assert r.parts["user_prompts"].value == 2
        assert r.parts["corrections"].value == 1


# ── 6. context pressure ──────────────────────────────────────────────────────


class TestContextPressure:
    def test_no_compaction_signal_of_any_kind_refuses_rather_than_reporting_zero(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            ev(EventType.SESSION_COMPLETED, 2.0),
        ]
        r = one(context_pressure, events)
        assert r.refusal
        assert r.headline.value is None

    def test_a_pre_compact_hook_with_no_completion_still_counts_as_one(self):
        """Claude's `PreCompact` fires before compaction and nothing fires
        after. Counting completions alone reports zero compactions for a
        session that compacted."""
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            ev(EventType.COMPACTION_STARTED, 5.0, payload={"trigger": "auto"}),
            ev(EventType.SESSION_COMPLETED, 6.0),
        ]
        r = one(context_pressure, events)
        assert r.headline.value == 1
        assert r.parts["auto_compactions"].value == 1

    def test_tokens_across_a_compaction_stay_missing_when_unreported(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            ev(EventType.COMPACTION_COMPLETED, 5.0, payload={"trigger": "manual"}),
            ev(EventType.SESSION_COMPLETED, 6.0),
        ]
        r = one(context_pressure, events)
        assert r.parts["tokens_reclaimed"].value is None
        assert r.parts["tokens_reclaimed"].fidelity is Fidelity.MISSING

    def test_reported_before_and_after_tokens_are_taken_verbatim(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            ev(EventType.COMPACTION_COMPLETED, 5.0,
               payload={"trigger": "auto", "tokens_before": 180_000, "tokens_after": 42_000}),
            ev(EventType.SESSION_COMPLETED, 6.0),
        ]
        r = one(context_pressure, events)
        assert r.parts["tokens_reclaimed"].value == 138_000
        assert r.parts["tokens_reclaimed"].fidelity is Fidelity.NATIVE


# ── 7. progress evidence ─────────────────────────────────────────────────────


class TestProgressEvidence:
    def test_there_is_deliberately_no_progress_number(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            *tool(1.0, 2.0, Action.EDIT, "/p/a.py"),
            ev(EventType.SESSION_COMPLETED, 3.0),
        ]
        r = one(progress_evidence, events)
        assert r.headline.value is None
        assert r.headline.fidelity is Fidelity.MISSING
        assert "checklist" in (r.headline.note or "")

    def test_an_item_that_cannot_be_decided_stays_unknown(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            *tool(1.0, 2.0, Action.INSPECT, "cat README"),
            ev(EventType.SESSION_COMPLETED, 3.0),
        ]
        r = one(progress_evidence, events)
        after = next(x for x in r.rows if x["item"] == "verification ran after the last edit")
        assert after["status"] == "unknown"

    def _claim(self, outcome: str | None):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            *tool(1.0, 2.0, Action.EDIT, "/p/a.py"),
            ev(EventType.SESSION_COMPLETED, 3.0,
               payload={"outcome": outcome} if outcome else {}),
        ]
        r = one(progress_evidence, events)
        return next(x for x in r.rows if x["item"] == "the agent said it was done")

    def test_claiming_done_and_verifying_are_separate_rows(self):
        """`agent_claimed_complete` and `verified_pass` must never collapse:
        an agent saying "done" is evidence about the agent."""
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            *tool(1.0, 2.0, Action.EDIT, "/p/a.py"),
            ev(EventType.SESSION_COMPLETED, 3.0,
               payload={"outcome": "agent_claimed_complete"}),
        ]
        r = one(progress_evidence, events)
        claimed = next(x for x in r.rows if x["item"] == "the agent said it was done")
        verified = next(x for x in r.rows if x["item"] == "that verification passed")
        assert claimed["status"] == "yes"
        assert verified["status"] == "unknown"

    def test_a_known_outcome_is_not_by_itself_a_claim_of_success(self):
        """The bug this pins: `outcome != unknown` reads as "the agent said it
        was done", and an interrupted run has a perfectly known outcome."""
        assert self._claim("interrupted")["status"] == "no"
        assert self._claim("infrastructure_failure")["status"] == "no"

    def test_an_agent_that_never_reports_an_outcome_is_unknown_not_a_denial(self):
        """Hooks carry no completion claim at all. `no` there would say the
        agent stayed silent when in fact we were never listening."""
        row = self._claim(None)
        assert row["status"] == "unknown"
        assert row["fidelity"] == Fidelity.MISSING.value

    def test_every_item_carries_its_own_fidelity(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            *tool(1.0, 2.0, Action.VCS, "git commit -m x"),
            ev(EventType.SESSION_COMPLETED, 3.0),
        ]
        r = one(progress_evidence, events)
        assert all(row["fidelity"] for row in r.rows)
        commit = next(x for x in r.rows if x["item"] == "changes were committed")
        # matching "git commit" in a command string is a reading of it
        assert commit["fidelity"] == Fidelity.HEURISTIC.value


# ── the set ──────────────────────────────────────────────────────────────────


class TestAnalysisSet:
    def _events(self):
        return [
            ev(EventType.SESSION_STARTED, 0.0),
            *tool(1.0, 2.0, Action.EDIT, "/p/a.py"),
            file_changed(2.0, "/p/a.py", lines_added=5, lines_removed=0, total_lines=5),
            *tool(3.0, 4.0, Action.VERIFY, "pytest -q"),
            ev(EventType.SESSION_COMPLETED, 5.0),
        ]

    def test_every_analysis_carries_a_version_a_formula_and_its_inputs(self):
        doc = analyze(view_of(self._events()), self._events())
        assert doc["analyses_version"] == ANALYSES_VERSION
        assert len(doc["analyses"]) == len(ANALYSES)
        for a in doc["analyses"]:
            assert a["version"], a["key"]
            assert len(a["formula"]) > 20, a["key"]
            assert a["inputs"], a["key"]

    def test_a_refusal_has_the_same_shape_as_a_result(self):
        doc = analyze(view_of(self._events()), self._events())
        for a in doc["analyses"]:
            assert set(a) >= {"headline", "parts", "rows", "evidence", "refusal"}
            if a["refusal"]:
                assert a["headline"]["value"] is None
                assert a["headline"]["fidelity"] == "missing"

    def test_one_analysis_raising_does_not_take_the_others_off_the_page(self, monkeypatch):
        import nebulai.seer.analysis as A

        def boom(view, events):
            raise ZeroDivisionError("synthetic")

        monkeypatch.setattr(A, "ANALYSES", (A.time_decomposition, boom))
        doc = A.analyze(view_of(self._events()), self._events())
        assert doc["analyses"][0]["headline"]["value"] is not None
        assert "ZeroDivisionError" in doc["analyses"][1]["refusal"]

    def test_no_analysis_reads_the_agents_own_vocabulary(self):
        """The M0 rule, held at the analysis layer: `native` and `native_type`
        exist for display and audit. A metric that reads them has grown an
        agent-specific code path in the one place that must not have one."""
        import nebulai.seer.analysis as A

        src = Path(A.__file__).read_text()
        body = src.split('"""', 2)[-1]  # skip the module docstring
        for banned in ("e.native", ".native_type", '"native"'):
            assert banned not in body, banned

    def test_the_analyses_replay_identically_from_the_stored_log(self, tmp_path):
        store = EventStore(tmp_path / "seer")
        try:
            events = self._events()
            store.register_run("run_1", agent="claude", capture_mode="driven")
            store.append_many(events)
            back = list(store.read("run_1"))
            a = analyze(reduce_run("run_1", events, now=5.0), events)
            b = analyze(reduce_run("run_1", back, now=5.0), back)
            assert json.dumps(a, sort_keys=True, default=str) == json.dumps(
                b, sort_keys=True, default=str
            )
        finally:
            store.close()


# ── export ───────────────────────────────────────────────────────────────────


class TestExport:
    def _events(self):
        return [
            ev(EventType.SESSION_STARTED, 0.0),
            ev(EventType.MESSAGE_ASSISTANT_DELTA, 0.5, payload={"text": "partial"}),
            *tool(1.0, 2.0, Action.EDIT, "/p/a.py"),
            file_changed(2.0, "/p/a.py", lines_added=5, lines_removed=0, total_lines=5),
            ev(EventType.SESSION_COMPLETED, 3.0),
        ]

    def test_jsonl_round_trips_to_identical_events(self):
        events = self._events()
        back = [Event.from_dict(json.loads(l))
                for l in to_jsonl(events).decode().splitlines()]
        assert [e.to_dict() for e in back] == [e.to_dict() for e in events]

    def test_every_parquet_row_carries_its_fidelity_and_capture_mode(self):
        """So a `groupby` six months from now cannot average an estimated
        duration into a native one without having been told."""
        pa = pytest.importorskip("pyarrow.parquet")
        import io

        t = pa.read_table(io.BytesIO(to_parquet(self._events())))
        assert set(t.column_names) == set(EVENT_COLUMNS)
        assert t.num_rows == len(self._events())
        assert all(x is not None for x in t.column("fidelity").to_pylist())
        assert all(x is not None for x in t.column("capture_mode").to_pylist())

    def test_parquet_marks_streaming_fragments_so_a_groupby_cannot_count_them(self):
        pa = pytest.importorskip("pyarrow.parquet")
        import io

        t = pa.read_table(io.BytesIO(to_parquet(self._events())))
        deltas = [r for r in t.to_pylist() if r["is_delta"]]
        assert len(deltas) == 1
        assert deltas[0]["event_type"] == "message.assistant_delta"

    def test_the_payload_survives_parquet_verbatim(self):
        pa = pytest.importorskip("pyarrow.parquet")
        import io

        t = pa.read_table(io.BytesIO(to_parquet(self._events())))
        changed = [r for r in t.to_pylist() if r["event_type"] == "file.changed"][0]
        assert json.loads(changed["payload_json"])["total_lines"] == 5

    def test_an_all_null_column_still_has_a_type(self):
        """Two runs concatenated in pandas must not fail because one agent
        reports turns and the other does not."""
        pa = pytest.importorskip("pyarrow.parquet")
        import io

        t = pa.read_table(io.BytesIO(to_parquet(self._events())))
        assert str(t.schema.field("turn_id").type) == "string"

    def test_csv_says_in_the_file_that_it_is_lossy(self):
        body = to_csv(view_of(self._events())).decode()
        assert body.splitlines()[0].startswith("#")
        assert "LOSSY" in body.splitlines()[0]

    def test_an_unknown_format_is_refused_by_name(self):
        with pytest.raises(ValueError, match="xlsx"):
            export("xlsx", view_of(self._events()), self._events())

    def test_every_advertised_format_produces_bytes_and_a_filename(self):
        v, e = view_of(self._events()), self._events()
        for fmt in (*FORMATS, "analysis"):
            body, ctype, name = export(fmt, v, e)
            assert isinstance(body, bytes) and body
            assert ctype and name.startswith("run_1")

    def test_the_analysis_export_carries_the_formulas_with_the_numbers(self):
        body, _, _ = export("analysis", view_of(self._events()), self._events())
        doc = json.loads(body)
        assert doc["analyses"][0]["formula"]
        # and the view it was computed from, so the export is self-contained
        assert doc["view"]["run_id"] == "run_1"


# ── annotations ──────────────────────────────────────────────────────────────


class TestAnnotations:
    def _annotation(self, ts: float, text: str) -> Event:
        return ev(EventType.ANNOTATION_ADDED, ts,
                  source=_src(fidelity=Fidelity.NATIVE),
                  native_type="human.annotation",
                  payload={"text": text, "tags": ["triage"], "author": "x"})

    def test_a_note_reaches_the_view_with_its_tags(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            self._annotation(1.0, "the two failures are one import error"),
            ev(EventType.SESSION_COMPLETED, 2.0),
        ]
        v = view_of(events)
        assert v.annotations[0]["text"].startswith("the two failures")
        assert v.annotations[0]["tags"] == ["triage"]

    def test_a_note_is_not_an_action_and_changes_no_count(self):
        base = [
            ev(EventType.SESSION_STARTED, 0.0),
            *tool(1.0, 2.0, Action.EDIT, "/p/a.py"),
            ev(EventType.SESSION_COMPLETED, 3.0),
        ]
        annotated = [*base[:-1], self._annotation(2.5, "note"), base[-1]]
        assert view_of(base).action_counts == view_of(annotated).action_counts
        assert view_of(base).time_in_state.keys() == view_of(annotated).time_in_state.keys()

    def test_a_note_exports_with_the_run_it_annotates(self):
        events = [
            ev(EventType.SESSION_STARTED, 0.0),
            self._annotation(1.0, "keep me"),
            ev(EventType.SESSION_COMPLETED, 2.0),
        ]
        assert "keep me" in to_jsonl(events).decode()
