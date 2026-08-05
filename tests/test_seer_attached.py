"""M3: the Codex app-server adapter, and the version gate that guards it.

Three properties carry the milestone:

* **Attached mode closes gaps that driven mode cannot.** Approvals, mid-turn
  token usage, compaction and per-file line counts are all absent from
  `codex exec --json` and present here. The tests below assert each one, because
  the whole argument for a second Codex adapter is that it sees more.

* **Cumulative reports are replaced, never added.** `thread/tokenUsage/updated`
  fires repeatedly with running totals. Folding them the way `turn.completed` is
  folded would multiply a run's tokens by the number of updates and look
  entirely plausible doing it.

* **The protocol gate fails closed on removal and open on addition.** A method
  we read that disappears silently zeroes whatever depended on it; a method we
  have never heard of can only be ignored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nebulai.seer.adapters import CodexAppServerAdapter, CodexExecAdapter, adapter_for
from nebulai.seer.adapters.codex_app_server import (
    MAPPED_NOTIFICATIONS,
    MAPPED_REQUESTS,
    check_protocol,
    diff_extent,
)
from nebulai.seer.contract import (
    Action,
    CaptureMode,
    Effect,
    EventType,
    Fidelity,
    TokenCategory,
)
from nebulai.seer.reducer import reduce_run

FIXTURES = Path(__file__).parent / "fixtures" / "seer"
GOLDEN = json.loads((FIXTURES / "codex-appserver-protocol.json").read_text())


def A(**kw) -> CodexAppServerAdapter:
    kw.setdefault("run_id", "run_1")
    kw.setdefault("session_id", "ses_1")
    return CodexAppServerAdapter(**kw)


def note(method: str, **params) -> str:
    return json.dumps({"jsonrpc": "2.0", "method": method, "params": params})


def req(rid: int, method: str, **params) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})


def resp(rid: int, **result) -> str:
    return json.dumps({"id": rid, "result": result})


def item(kind: str, itype: str, item_id: str, **fields) -> str:
    return note(f"item/{kind}", threadId="t1", turnId="turn1",
                item={"id": item_id, "type": itype, **fields})


def run(lines: list[str]) -> tuple[CodexAppServerAdapter, list]:
    a = A()
    out = []
    for ln in lines:
        out += a.feed(ln)
    return a, out


# ── the reason attached mode exists ──────────────────────────────────────────


class TestGapsClosed:
    def test_an_approval_and_its_decision_both_reach_the_log(self):
        """`exec --json` cannot see this at all — `intervention_burden` reports
        `missing` for every driven Codex run because of it."""
        _, out = run([
            note("thread/started", threadId="t1"),
            req(7, "item/commandExecution/requestApproval",
                command="rm -rf build", reason="destructive"),
            resp(7, decision="approved"),
        ])
        kinds = [e.event_type for e in out]
        assert EventType.APPROVAL_REQUESTED in kinds
        assert EventType.APPROVAL_RESOLVED in kinds
        resolved = next(e for e in out if e.event_type is EventType.APPROVAL_RESOLVED)
        assert resolved.payload["decision"] == "allow"
        assert resolved.payload["request_id"] == "7"

    def test_a_denied_approval_is_a_decision_not_a_missing_one(self):
        _, out = run([
            note("thread/started", threadId="t1"),
            req(1, "applyPatchApproval", callId="c1"),
            resp(1, decision="denied"),
        ])
        assert out[-1].payload["decision"] == "deny"

    def test_a_request_for_input_is_a_clarification_not_an_approval(self):
        """They are different interruptions: one asks permission to act, the
        other asks what to do. `intervention_burden` counts them separately."""
        _, out = run([
            note("thread/started", threadId="t1"),
            req(3, "item/tool/requestUserInput", prompt="which branch?"),
            resp(3, decision="answered"),
        ])
        assert [e.event_type for e in out[1:]] == [
            EventType.CLARIFICATION_REQUESTED,
            EventType.CLARIFICATION_RESOLVED,
        ]

    def test_compaction_reaches_context_pressure(self):
        _, out = run([
            note("thread/started", threadId="t1"),
            note("thread/compacted", trigger="auto",
                 tokensBefore=180_000, tokensAfter=42_000),
        ])
        c = next(e for e in out if e.event_type is EventType.COMPACTION_COMPLETED)
        assert c.payload["tokens_before"] == 180_000
        assert c.payload["tokens_after"] == 42_000

    def test_a_file_change_carries_line_counts_from_its_diff(self):
        """The Codex half of the churn story. `exec --json` gives a path and a
        kind, so `edit_churn` refuses; the app-server gives the diff."""
        _, out = run([
            note("thread/started", threadId="t1"),
            item("completed", "fileChange", "i1", status="completed", changes=[
                {"path": "/p/a.py", "kind": "update",
                 "diff": "--- a/p/a.py\n+++ b/p/a.py\n@@\n-old\n+new\n+extra\n"},
            ]),
        ])
        fc = next(e for e in out if e.event_type is EventType.FILE_CHANGED)
        assert fc.payload["lines_added"] == 2
        assert fc.payload["lines_removed"] == 1

    def test_a_commands_own_duration_is_carried_rather_than_recomputed(self):
        _, out = run([
            note("thread/started", threadId="t1"),
            item("started", "commandExecution", "i1", command="pytest -q", cwd="/p"),
            item("completed", "commandExecution", "i1", command="pytest -q",
                 cwd="/p", exitCode=0, status="completed", durationMs=1234),
        ])
        done = next(e for e in out if e.event_type is EventType.TOOL_COMPLETED)
        assert done.payload["duration_ms"] == 1234
        assert done.source.fidelity is Fidelity.NATIVE

    def test_this_mode_declares_a_much_shorter_gap_list_than_driven_mode(self):
        from nebulai.seer.adapters.codex import MISSING_IN_EXEC_JSON
        from nebulai.seer.adapters.codex_app_server import MISSING_IN_APP_SERVER

        assert set(MISSING_IN_APP_SERVER) < set(MISSING_IN_EXEC_JSON)
        _, out = run([note("thread/started", threadId="t1")])
        assert out[0].payload["capture_gaps"] == list(MISSING_IN_APP_SERVER)


# ── usage ────────────────────────────────────────────────────────────────────


class TestCumulativeUsage:
    def test_repeated_totals_are_replaced_not_added(self):
        a, _ = run([
            note("thread/started", threadId="t1"),
            note("thread/tokenUsage/updated",
                 usage={"inputTokens": 100, "outputTokens": 10}),
            note("thread/tokenUsage/updated",
                 usage={"inputTokens": 250, "outputTokens": 40}),
            note("thread/tokenUsage/updated",
                 usage={"inputTokens": 900, "outputTokens": 90}),
        ])
        assert a.usage.by_category[TokenCategory.INPUT] == 900
        assert a.usage.by_category[TokenCategory.OUTPUT] == 90

    def test_the_event_says_the_number_is_cumulative(self):
        _, out = run([
            note("thread/started", threadId="t1"),
            note("thread/tokenUsage/updated", usage={"inputTokens": 5}),
        ])
        u = next(e for e in out if e.event_type is EventType.MODEL_USAGE_UPDATED)
        assert u.payload["cumulative"] is True

    def test_codex_still_reports_no_cache_write_bucket(self):
        """Same fact as driven mode, and it has to stay `missing` rather than
        become 0 just because this mode reports more of everything else."""
        _, out = run([
            note("thread/started", threadId="t1"),
            note("thread/tokenUsage/updated",
                 usage={"inputTokens": 5, "cachedInputTokens": 2}),
        ])
        u = next(e for e in out if e.event_type is EventType.MODEL_USAGE_UPDATED)
        assert u.payload["cache_write"] is None
        assert u.payload["cache_write_fidelity"] == Fidelity.MISSING.value

    def test_usage_with_no_recognised_key_produces_no_event_rather_than_a_zero(self):
        _, out = run([
            note("thread/started", threadId="t1"),
            note("thread/tokenUsage/updated", usage={"somethingElse": 5}),
        ])
        assert not [e for e in out if e.event_type is EventType.MODEL_USAGE_UPDATED]


# ── items ────────────────────────────────────────────────────────────────────


class TestItems:
    def test_camel_case_item_types_are_the_ones_that_map(self):
        """The app-server says `commandExecution`; `exec --json` says
        `command_execution`. Feeding one adapter the other's vocabulary must
        warn, not silently produce an empty trajectory."""
        _, out = run([
            note("thread/started", threadId="t1"),
            item("completed", "command_execution", "i1", command="ls"),
        ])
        warns = [e for e in out if e.event_type is EventType.ADAPTER_WARNING]
        assert warns and "command_execution" in warns[0].payload["note"]

    def test_a_declined_patch_is_not_a_file_change(self):
        """A phantom edit here would inflate churn and make `verification
        coverage` demand a check for a file that never changed."""
        _, out = run([
            note("thread/started", threadId="t1"),
            item("completed", "fileChange", "i1", status="declined", changes=[
                {"path": "/p/a.py", "kind": "update", "diff": "-x\n+y\n"},
            ]),
        ])
        assert not [e for e in out if e.event_type is EventType.FILE_CHANGED]
        assert out[-1].event_type is EventType.TOOL_FAILED

    def test_a_patch_preview_does_not_count_as_an_applied_change(self):
        _, out = run([
            note("thread/started", threadId="t1"),
            item("started", "fileChange", "i1"),
            note("item/fileChange/patchUpdated", itemId="i1"),
        ])
        assert not [e for e in out if e.event_type is EventType.FILE_CHANGED]
        assert out[-1].payload["applied"] is False

    def test_a_delta_is_attributed_to_the_span_its_item_opened(self):
        _, out = run([
            note("thread/started", threadId="t1"),
            item("started", "commandExecution", "i1", command="pytest", cwd="/p"),
            note("item/commandExecution/outputDelta", itemId="i1", chunk="....\n"),
        ])
        started = next(e for e in out if e.event_type is EventType.TOOL_STARTED)
        delta = next(e for e in out if e.event_type is EventType.TOOL_OUTPUT_DELTA)
        assert delta.span_id == started.span_id

    def test_a_subagent_item_is_delegation_not_execution(self):
        _, out = run([
            note("thread/started", threadId="t1"),
            item("completed", "subAgentActivity", "i1", name="reviewer",
                 status="completed"),
        ])
        assert out[-1].action is Action.DELEGATE

    def test_sleeping_is_time_but_not_work(self):
        _, out = run([
            note("thread/started", threadId="t1"),
            item("completed", "sleep", "i1", durationMs=5000),
        ])
        assert out[-1].action is Action.INTERACT
        assert out[-1].effect is Effect.NO_STATE_CHANGE

    def test_reasoning_text_is_dropped_by_policy_not_missing(self):
        _, out = run([
            note("thread/started", threadId="t1"),
            item("completed", "reasoning", "i1",
                 content=[{"text": "a long private deliberation"}]),
        ])
        r = out[-1]
        assert r.source.fidelity is Fidelity.DROPPED_BY_POLICY
        assert r.payload["text_retained"] is False
        assert "deliberation" not in json.dumps(r.payload)

    def test_an_unknown_item_type_warns_once_and_keeps_going(self):
        _, out = run([
            note("thread/started", threadId="t1"),
            item("completed", "quantumThing", "i1"),
            item("completed", "quantumThing", "i2"),
            item("completed", "webSearch", "i3", query="q"),
        ])
        warns = [e for e in out if e.event_type is EventType.ADAPTER_WARNING]
        assert len(warns) == 1
        assert [e for e in out if e.action is Action.SEARCH]


# ── lifecycle ────────────────────────────────────────────────────────────────


class TestLifecycle:
    def test_the_capture_mode_is_attached_without_being_asked(self):
        assert A().capture_mode is CaptureMode.ATTACHED

    def test_a_closed_socket_is_a_detach_not_an_interrupted_session(self):
        """The driven adapter is right to call a dead stream the end of the
        session — it owns the process. Here the thread lives in the daemon and
        may still be running, so "interrupted" would be a claim about the agent
        made from a fact about our socket."""
        a, _ = run([note("thread/started", threadId="t1")])
        end = a.finish()
        assert end[0].event_type is EventType.SESSION_STATE_CHANGED
        assert end[0].payload["state"] == "detached"

    def test_nothing_is_emitted_for_a_connection_that_saw_no_thread(self):
        assert A().finish() == []

    def test_a_response_to_our_own_call_is_not_an_approval(self):
        a, _ = run([note("thread/started", threadId="t1")])
        assert a.feed(resp(99, data=[])) == []

    def test_a_full_thread_reduces_to_a_run_view(self):
        _, out = run([
            note("thread/started", threadId="t1"),
            note("turn/started", turnId="turn1"),
            item("completed", "userMessage", "u1",
                 content=[{"type": "text", "text": "fix the tests"}]),
            item("started", "commandExecution", "i1", command="pytest -q", cwd="/p"),
            item("completed", "commandExecution", "i1", command="pytest -q",
                 cwd="/p", exitCode=1, status="completed", durationMs=900),
            item("completed", "fileChange", "i2", status="completed", changes=[
                {"path": "/p/a.py", "kind": "update", "diff": "-a\n+b\n+c\n"},
            ]),
            note("thread/tokenUsage/updated",
                 usage={"inputTokens": 900, "outputTokens": 90}),
            note("turn/completed", turnId="turn1"),
        ])
        view = reduce_run("run_1", out, now=out[-1].ts)
        assert view.action_counts.get("verify") == 1
        assert view.files_changed == ["/p/a.py"]
        assert view.file_stats["/p/a.py"]["lines_added"] == 2
        assert view.quality.capture_mode == CaptureMode.ATTACHED.value


# ── the version gate ─────────────────────────────────────────────────────────


class TestProtocolGate:
    def test_the_golden_fixture_still_covers_everything_we_map(self):
        """If this fails, the recorded protocol surface is older than the
        adapter — regenerate it with `codex app-server generate-json-schema`."""
        assert MAPPED_NOTIFICATIONS <= set(GOLDEN["server_notifications"])
        assert MAPPED_REQUESTS <= set(GOLDEN["server_requests"])

    def test_a_removed_method_is_fatal(self):
        live = set(GOLDEN["server_notifications"]) - {"thread/compacted"}
        r = check_protocol(live, set(GOLDEN["server_requests"]), GOLDEN)
        assert r["compatible"] is False
        assert r["missing_notifications"] == ["thread/compacted"]

    def test_a_new_method_is_not(self):
        """Ignoring a method we have never heard of cannot corrupt anything
        already recorded — so it is reported, not refused."""
        live = set(GOLDEN["server_notifications"]) | {"thread/somethingNew"}
        r = check_protocol(live, set(GOLDEN["server_requests"]), GOLDEN)
        assert r["compatible"] is True
        assert "thread/somethingNew" in r["new_since_golden"]
        assert "thread/somethingNew" in r["unmapped_notifications"]

    def test_the_report_names_the_version_it_compared_against(self):
        r = check_protocol(set(GOLDEN["server_notifications"]),
                           set(GOLDEN["server_requests"]), GOLDEN)
        assert r["golden_version"] == GOLDEN["codex_version"]

    def test_the_gate_works_without_a_golden_file_at_all(self):
        r = check_protocol(set(MAPPED_NOTIFICATIONS), set(MAPPED_REQUESTS))
        assert r["compatible"] is True
        assert "golden_version" not in r


# ── diff counting ────────────────────────────────────────────────────────────


class TestDiffExtent:
    def test_file_headers_are_not_counted_as_lines(self):
        d = "--- a/x.py\n+++ b/x.py\n@@ -1 +1,2 @@\n-one\n+two\n+three\n"
        assert diff_extent(d) == {"lines_added": 2, "lines_removed": 1}

    def test_a_diff_with_no_changed_lines_reads_as_absent_not_zero(self):
        assert diff_extent("@@ -1 +1 @@\n context only\n") is None
        assert diff_extent("") is None
        assert diff_extent(None) is None

    def test_the_text_does_not_survive_the_count(self):
        e = diff_extent("-secret_key = 'abc'\n+secret_key = env('K')\n")
        assert e == {"lines_added": 1, "lines_removed": 1}
        assert "secret" not in json.dumps(e)


# ── selection ────────────────────────────────────────────────────────────────


class TestAdapterSelection:
    def test_asking_for_attached_codex_gets_the_app_server_adapter(self):
        a = adapter_for("codex", "attached", run_id="r", session_id="s")
        assert isinstance(a, CodexAppServerAdapter)

    def test_driven_stays_the_default(self):
        a = adapter_for("codex", run_id="r", session_id="s")
        assert isinstance(a, CodexExecAdapter)

    def test_a_mode_an_agent_does_not_have_is_refused_by_name(self):
        """Falling back to the driven adapter would produce a trajectory that
        is plausible, unlabelled and quietly less than what was asked for."""
        with pytest.raises(ValueError, match="attached mode exists for codex only"):
            adapter_for("claude", "attached", run_id="r", session_id="s")
