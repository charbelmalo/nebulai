"""M1a exit tests: the adapters, replayed against real captured output.

The fixtures in `tests/fixtures/seer/` are verbatim stdout from real runs of the
installed binaries, not hand-written examples. That matters for one test in
particular: `test_claude_folds_repeated_usage_on_message_id` depends on the
capture containing two `assistant` lines that share a `message.id` and repeat
identical `usage` — a shape nobody would think to invent, and the exact shape
that overcounted a real session by 3.5× in `viewer/src/chrome/sessionlog.ts`.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from nebulai.seer.adapters import CodexExecAdapter, ClaudeStreamAdapter, adapter_for
from nebulai.seer.adapters.hermes import HermesOneshotAdapter, HermesStateDbReconciler
from nebulai.seer.contract import (
    ANALYSIS_KEYS,
    Action,
    CaptureMode,
    Effect,
    EventType,
    Fidelity,
    Outcome,
    TokenCategory,
)

FIXTURES = Path(__file__).parent / "fixtures" / "seer"


def replay(adapter, path: Path):
    events = []
    for line in path.read_text().splitlines():
        events.extend(adapter.feed(line))
    events.extend(adapter.finish())
    return events


def mk(cls, **kw):
    kw.setdefault("run_id", "run_t")
    kw.setdefault("session_id", "ses_t")
    return cls(**kw)


def of(events, *types):
    want = {t.value if hasattr(t, "value") else t for t in types}
    return [e for e in events if e.event_type.value in want]


# ── codex ────────────────────────────────────────────────────────────────────


@pytest.fixture
def codex_events():
    return replay(mk(CodexExecAdapter), FIXTURES / "codex-exec.jsonl")


def test_codex_replays_the_real_capture(codex_events):
    kinds = [e.event_type.value for e in codex_events]
    assert kinds[0] == "session.started"
    assert "turn.started" in kinds
    assert "turn.completed" in kinds
    assert "message.assistant_completed" in kinds


def test_codex_usage_uses_codex_categories_and_admits_the_missing_one(codex_events):
    (usage_ev,) = of(codex_events, EventType.MODEL_USAGE_UPDATED)
    u = usage_ev.payload["usage"]
    # verbatim from the capture's turn.completed line
    assert u[TokenCategory.INPUT.value] == 19827
    assert u[TokenCategory.OUTPUT.value] == 5
    assert u[TokenCategory.CACHE_READ.value] == 0
    # Codex has no cache-write bucket at all. Absent, and labelled absent —
    # writing 0 here would make a Codex run look like it never wrote cache when
    # the truth is that Codex never says.
    assert usage_ev.payload["cache_write"] is None
    assert usage_ev.payload["cache_write_fidelity"] == Fidelity.MISSING.value
    assert TokenCategory.CACHE_WRITE not in usage_ev.payload["usage"]


def test_codex_error_item_is_an_agent_error_not_a_failed_turn(codex_events):
    """The capture contains an `item.completed` of type `error` on a turn that
    then completes normally. Folding it into TURN_FAILED would invent a failure."""
    (err,) = of(codex_events, EventType.AGENT_ERROR)
    assert "skills context budget" in err.payload["message"]
    assert not of(codex_events, EventType.TURN_FAILED)


def test_codex_names_what_exec_json_cannot_see(codex_events):
    (start,) = of(codex_events, EventType.SESSION_STARTED)
    gaps = start.payload["capture_gaps"]
    # DRIVEN is the lower-fidelity mode for Codex; the events must say so.
    assert any("approval" in g for g in gaps)
    assert start.source.capture_mode is CaptureMode.DRIVEN


def test_codex_command_action_comes_from_the_command_not_the_tool_name():
    a = mk(CodexExecAdapter)
    a.feed(json.dumps({"type": "thread.started", "thread_id": "t"}))
    a.feed(json.dumps({"type": "turn.started"}))
    ev = a.feed(json.dumps({
        "type": "item.completed",
        "item": {"id": "i1", "type": "command_execution",
                 "command": "pytest tests/ -q", "exit_code": 0, "status": "completed",
                 "aggregated_output": "5 passed"},
    }))
    (tool,) = of(ev, EventType.TOOL_COMPLETED)
    assert tool.action is Action.VERIFY
    assert tool.effect is Effect.STATE_CHANGED

    ev = a.feed(json.dumps({
        "type": "item.completed",
        "item": {"id": "i2", "type": "command_execution",
                 "command": "rm -rf tests/", "exit_code": 0, "status": "completed"},
    }))
    # same native item type, opposite action — the taxonomy reads the command
    assert of(ev, EventType.TOOL_COMPLETED)[0].action is Action.EXECUTE


def test_codex_nonzero_exit_is_a_failure_not_a_completion():
    a = mk(CodexExecAdapter)
    ev = a.feed(json.dumps({
        "type": "item.completed",
        "item": {"id": "i1", "type": "command_execution",
                 "command": "pytest", "exit_code": 1, "status": "completed"},
    }))
    assert ev[0].event_type is EventType.TOOL_FAILED
    assert ev[0].effect is Effect.FAILED


def test_codex_reasoning_is_dropped_by_policy_not_missing():
    a = mk(CodexExecAdapter)
    ev = a.feed(json.dumps({
        "type": "item.completed",
        "item": {"id": "r1", "type": "reasoning", "text": "a" * 500},
    }))
    (r,) = ev
    assert r.source.fidelity is Fidelity.DROPPED_BY_POLICY
    assert r.payload["chars"] == 500
    assert r.payload["text_retained"] is False
    assert "text" not in r.payload


def test_codex_reasoning_is_kept_when_asked():
    a = mk(CodexExecAdapter, keep_reasoning=True)
    (r,) = a.feed(json.dumps({
        "type": "item.completed",
        "item": {"id": "r1", "type": "reasoning", "text": "because"},
    }))
    assert r.source.fidelity is Fidelity.NATIVE
    assert r.payload["text"] == "because"


def test_codex_unknown_native_kind_warns_once():
    a = mk(CodexExecAdapter)
    first = a.feed(json.dumps({"type": "turn.somethingNew"}))
    second = a.feed(json.dumps({"type": "turn.somethingNew"}))
    assert len(first) == 1 and first[0].event_type is EventType.ADAPTER_WARNING
    assert second == []  # noisy once, not per line


def test_codex_stream_cut_mid_turn_reports_interrupted():
    a = mk(CodexExecAdapter)
    a.feed(json.dumps({"type": "thread.started", "thread_id": "t"}))
    a.feed(json.dumps({"type": "turn.started"}))
    (end,) = a.finish()
    assert end.event_type is EventType.SESSION_INTERRUPTED
    assert end.payload["outcome"] == Outcome.INTERRUPTED.value


def test_codex_file_change_emits_one_event_per_path():
    a = mk(CodexExecAdapter)
    ev = a.feed(json.dumps({
        "type": "item.completed",
        "item": {"id": "f1", "type": "file_change", "changes": [
            {"path": "a.py", "kind": "update"},
            {"path": "b.py", "kind": "add"},
        ]},
    }))
    changed = of(ev, EventType.FILE_CHANGED)
    assert [c.payload["path"] for c in changed] == ["a.py", "b.py"]
    (tool,) = of(ev, EventType.TOOL_COMPLETED)
    assert tool.action is Action.EDIT and tool.payload["n_changes"] == 2


# ── claude ───────────────────────────────────────────────────────────────────


@pytest.fixture
def claude_events():
    return replay(mk(ClaudeStreamAdapter), FIXTURES / "claude-stream.jsonl")


def test_claude_replays_the_real_capture(claude_events):
    kinds = {e.event_type.value for e in claude_events}
    assert {"session.started", "message.assistant_completed", "session.completed"} <= kinds


def test_claude_folds_repeated_usage_on_message_id(claude_events):
    """The capture has two assistant lines sharing one message.id, each
    repeating `output_tokens: 4`. Counting per line is the 3.5× overcount."""
    provisional = [
        e for e in of(claude_events, EventType.MODEL_USAGE_UPDATED)
        if e.payload.get("provisional")
    ]
    assert len(provisional) == 2, "the fixture must contain the repeat"
    assert [e.payload["counted"] for e in provisional] == [True, False]


def test_claude_result_line_replaces_rather_than_accumulates():
    """Streamed usage says output=4; the result line says 36. Adding gives 40,
    which is not a number that exists anywhere in the run."""
    a = mk(ClaudeStreamAdapter)
    replay(a, FIXTURES / "claude-stream.jsonl")
    assert a.usage.by_category[TokenCategory.OUTPUT] == 36
    assert a.usage.by_category[TokenCategory.CACHE_READ] == 20449
    assert a.usage.by_category[TokenCategory.CACHE_WRITE] == 10908


def test_claude_reasoning_has_no_native_bucket(claude_events):
    """Claude bills reasoning inside output_tokens. `system/thinking_tokens`
    offers an estimate and Claude itself calls it estimated; laundering that
    into a native reasoning count would be a fabricated measurement."""
    final = [
        e for e in of(claude_events, EventType.MODEL_USAGE_UPDATED)
        if e.payload.get("authoritative")
    ][0]
    assert final.payload["reasoning"] is None
    assert final.payload["reasoning_fidelity"] == Fidelity.MISSING.value
    assert final.payload["reasoning_tokens_estimated"] == 90

    thinking = [
        e for e in of(claude_events, EventType.MODEL_USAGE_UPDATED)
        if e.source.fidelity is Fidelity.ESTIMATED
    ]
    assert thinking and all(e.payload["counted"] is False for e in thinking)


def test_claude_thinking_block_is_dropped_by_policy(claude_events):
    (think,) = [
        e for e in of(claude_events, EventType.MODEL_REQUEST_COMPLETED)
        if e.payload.get("kind") == "reasoning"
    ]
    assert think.source.fidelity is Fidelity.DROPPED_BY_POLICY
    assert think.payload["text_retained"] is False


def test_claude_success_is_a_claim_not_a_verification(claude_events):
    (done,) = of(claude_events, EventType.SESSION_COMPLETED)
    assert done.payload["outcome"] == Outcome.AGENT_CLAIMED_COMPLETE.value
    assert done.payload["outcome"] != Outcome.VERIFIED_PASS.value
    assert done.payload["cost_usd"] == pytest.approx(0.0246269)


def test_claude_rate_limit_is_its_own_event(claude_events):
    (q,) = of(claude_events, EventType.QUOTA_UPDATED)
    assert q.payload["limit_type"] == "five_hour"
    assert q.payload["status"] == "allowed"


def test_claude_init_records_failed_mcp_servers(claude_events):
    (start,) = of(claude_events, EventType.SESSION_STARTED)
    # the capture had tokensave failed and two connectors needing auth
    assert "tokensave" in start.payload["mcp_failed"]
    assert start.payload["permission_mode"] == "default"


def test_claude_agent_version_comes_from_the_stream(claude_events):
    assert all(
        e.source.agent_version == "2.1.222"
        for e in claude_events
        if e.event_type is not EventType.QUOTA_UPDATED  # emitted before init
    )


def test_claude_bash_action_reads_the_command_not_the_tool_name():
    a = mk(ClaudeStreamAdapter)
    ev = a.feed(json.dumps({
        "type": "assistant",
        "message": {"id": "m1", "model": "claude-opus-5", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "Bash",
             "input": {"command": "npm run typecheck"}},
        ]},
    }))
    (started,) = of(ev, EventType.TOOL_STARTED)
    assert started.action is Action.VERIFY
    assert started.native_type == "Bash"


def test_claude_tool_result_closes_the_span_the_tool_use_opened():
    a = mk(ClaudeStreamAdapter)
    started = a.feed(json.dumps({
        "type": "assistant",
        "message": {"id": "m1", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "Read",
             "input": {"file_path": "/x.py"}},
        ]},
    }))
    span = of(started, EventType.TOOL_STARTED)[0].span_id
    done = a.feed(json.dumps({
        "type": "user",
        "message": {"content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
        ]},
    }))
    (completed,) = of(done, EventType.TOOL_COMPLETED)
    assert completed.span_id == span
    assert completed.action is Action.INSPECT


def test_claude_tool_error_is_a_failure():
    a = mk(ClaudeStreamAdapter)
    a.feed(json.dumps({
        "type": "assistant",
        "message": {"id": "m1", "content": [
            {"type": "tool_use", "id": "t1", "name": "Bash",
             "input": {"command": "pytest"}},
        ]},
    }))
    (ev,) = a.feed(json.dumps({
        "type": "user",
        "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1",
             "is_error": True, "content": "1 failed"},
        ]},
    }))
    assert ev.event_type is EventType.TOOL_FAILED
    assert ev.effect is Effect.FAILED
    assert ev.action is Action.VERIFY


def test_claude_permission_denials_become_approval_events():
    a = mk(ClaudeStreamAdapter)
    ev = a.feed(json.dumps({
        "type": "result", "subtype": "success",
        "permission_denials": [{"tool_name": "Bash"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }))
    (denial,) = of(ev, EventType.APPROVAL_RESOLVED)
    assert denial.payload == {"tool": "Bash", "decision": "denied"}
    assert denial.effect is Effect.NO_STATE_CHANGE


def test_claude_context_window_is_absent_when_two_models_ran():
    """Two models means no single context window. Reporting the first would be
    a quiet lie about which run the number describes."""
    a = mk(ClaudeStreamAdapter)
    ev = a.feed(json.dumps({
        "type": "result", "subtype": "success", "usage": {"input_tokens": 1},
        "modelUsage": {"a": {"contextWindow": 200000}, "b": {"contextWindow": 1000000}},
    }))
    assert of(ev, EventType.SESSION_COMPLETED)[0].payload["context_window"] is None

    b = mk(ClaudeStreamAdapter)
    ev = b.feed(json.dumps({
        "type": "result", "subtype": "success", "usage": {"input_tokens": 1},
        "modelUsage": {"a": {"contextWindow": 200000}},
    }))
    assert of(ev, EventType.SESSION_COMPLETED)[0].payload["context_window"] == 200000


def test_claude_abnormal_stop_reason_is_surfaced():
    a = mk(ClaudeStreamAdapter)
    ev = a.feed(json.dumps({
        "type": "assistant",
        "message": {"id": "m1", "stop_reason": "max_tokens",
                    "content": [{"type": "text", "text": "half an ans"}]},
    }))
    (err,) = of(ev, EventType.AGENT_ERROR)
    assert err.payload["stop_reason"] == "max_tokens"


# ── the same task, both agents ───────────────────────────────────────────────
#
# `*-tools.jsonl` are two captures of the same instruction ("write hello.txt,
# then `wc -c` it, then say done") run through Codex and Claude. They exist so
# the adapters can be checked for agreement on a task where agreement is the
# right answer — anything the two disagree about here is a normalization bug,
# not a difference between the agents.


@pytest.fixture
def codex_tools():
    return replay(mk(CodexExecAdapter), FIXTURES / "codex-tools.jsonl")


@pytest.fixture
def claude_tools():
    return replay(mk(ClaudeStreamAdapter), FIXTURES / "claude-tools.jsonl")


def test_both_agents_produce_the_same_action_sequence(codex_tools, claude_tools):
    def actions(events):
        return [
            e.action.value for e in events
            if e.event_type
            in (EventType.TOOL_COMPLETED, EventType.TOOL_FAILED, EventType.FILE_CHANGED)
        ]

    # Codex reports the write as a `file_change` item; Claude reports it as a
    # `Write` tool call and never emits a file-change event at all. Completely
    # different native vocabularies, and after normalization the two runs tell
    # the identical story. That equality is the contract doing its job — the
    # first draft of the Claude adapter produced ["edit", "execute"] here, and
    # the missing FILE_CHANGED would have shown Codex touching one more file
    # than Claude on a task where they touched the same one.
    assert actions(codex_tools) == actions(claude_tools) == ["edit", "edit", "execute"]


def test_codex_shell_wrapper_does_not_hide_the_real_command(codex_tools):
    """The capture's command is `/bin/zsh -lc 'wc -c hello.txt'`. Classifying
    the wrapper instead of the command would make every Codex run look like it
    only ever ran zsh."""
    (cmd,) = [
        e for e in of(codex_tools, EventType.TOOL_COMPLETED)
        if e.payload.get("command")
    ]
    assert cmd.payload["command"].startswith("/bin/zsh -lc")
    assert cmd.action is Action.EXECUTE  # `wc` is not verification
    assert cmd.payload["exit_code"] == 0


def test_wrapped_verification_is_still_verification():
    """Same wrapper, a command that *is* verification. This is the case that
    would silently zero out Codex's verification coverage."""
    from nebulai.seer.taxonomy import classify_command

    assert classify_command("/bin/zsh -lc 'pytest tests/ -q'") is Action.VERIFY
    assert classify_command("bash -lc \"npm run typecheck\"") is Action.VERIFY
    assert classify_command("/bin/zsh -lc 'git status'") is Action.VCS
    # and the wrapper must not manufacture a verify out of nothing
    assert classify_command("/bin/zsh -lc 'rm -rf tests/'") is Action.EXECUTE
    assert classify_command("zsh -lc 'echo hi'") is Action.EXECUTE


def test_codex_file_change_spans_open_and_close(codex_tools):
    """The capture has `item.started` then `item.completed` for the same
    `item_3`. Both must land on one span, or the edit shows up twice."""
    edits = [
        e for e in codex_tools
        if e.native_type == "item.file_change"
    ]
    assert len({e.span_id for e in edits}) == 1
    assert of(edits, EventType.TOOL_STARTED)
    assert of(edits, EventType.FILE_CHANGED)


def test_claude_tools_capture_pairs_every_start_with_a_completion(claude_tools):
    started = of(claude_tools, EventType.TOOL_STARTED)
    completed = of(claude_tools, EventType.TOOL_COMPLETED, EventType.TOOL_FAILED)
    assert len(started) == len(completed) == 2
    assert {e.span_id for e in started} == {e.span_id for e in completed}


def test_neither_capture_verified_anything(codex_tools, claude_tools):
    """`wc -c` is not a test. Both runs end with the agent claiming success and
    no verification evidence — which is exactly the gap the outcome model is
    built to make visible, and it must not be papered over."""
    for events in (codex_tools, claude_tools):
        assert not [e for e in events if e.action is Action.VERIFY]
    (done,) = of(claude_tools, EventType.SESSION_COMPLETED)
    assert done.payload["outcome"] == Outcome.AGENT_CLAIMED_COMPLETE.value


def test_token_categories_do_not_line_up_across_the_two_captures(
    codex_tools, claude_tools
):
    """The empirical case for the comparability gate. Both runs did the same
    work; their usage keys are not the same quantities, so no honest headline
    number subtracts one from the other."""
    codex_keys = set(
        of(codex_tools, EventType.MODEL_USAGE_UPDATED)[0].payload["native_categories"]
    )
    claude_final = [
        e for e in of(claude_tools, EventType.MODEL_USAGE_UPDATED)
        if e.payload.get("authoritative")
    ][0]
    claude_keys = set(claude_final.payload["native_categories"])

    assert "reasoning_output_tokens" in codex_keys  # Codex breaks reasoning out
    assert not any("reasoning" in k for k in claude_keys)  # Claude folds it in
    assert "cache_creation_input_tokens" in claude_keys  # Claude has cache-write
    assert not any("creation" in k or "write" in k for k in codex_keys)  # Codex doesn't


# ── hermes ───────────────────────────────────────────────────────────────────


def test_hermes_oneshot_admits_it_sees_no_tools():
    """`hermes -z` prints only final text. An adapter claiming tool calls off
    it would be inventing them."""
    a = mk(HermesOneshotAdapter)
    start = a.open()
    a.feed("done.")
    end = a.close(exit_code=0)

    (session,) = of(start, EventType.SESSION_STARTED)
    assert "tool calls" in session.payload["capture_gaps"]
    assert session.source.fidelity is Fidelity.DETERMINISTIC

    (msg,) = of(end, EventType.MESSAGE_ASSISTANT_COMPLETED)
    assert msg.payload["text"] == "done."
    (done,) = of(end, EventType.SESSION_COMPLETED)
    assert done.payload["outcome"] == Outcome.AGENT_CLAIMED_COMPLETE.value


def test_hermes_nonzero_exit_is_infrastructure_failure():
    a = mk(HermesOneshotAdapter)
    a.open()
    (_, done) = a.close(exit_code=2)
    assert done.event_type is EventType.SESSION_FAILED
    assert done.payload["outcome"] == Outcome.INFRASTRUCTURE_FAILURE.value


@pytest.fixture
def fake_state_db(tmp_path):
    """A state.db with the real column set, so the reconciler is tested against
    the schema Hermes actually ships rather than one we wish it shipped."""
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, source TEXT NOT NULL, model TEXT,
            started_at REAL NOT NULL, ended_at REAL, end_reason TEXT,
            message_count INTEGER, tool_call_count INTEGER, api_call_count INTEGER,
            input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER,
            cache_write_tokens INTEGER, reasoning_tokens INTEGER,
            billing_provider TEXT, estimated_cost_usd REAL, actual_cost_usd REAL,
            cost_status TEXT, cost_source TEXT, cwd TEXT)
    """)
    conn.execute(
        "INSERT INTO sessions VALUES ('cli_1','cli','hermes-4',1000.0,1010.0,"
        "'complete',6,3,4,100,20,300,40,7,'nous',0.01,0.012,'final','provider','/w')"
    )
    conn.commit()
    conn.close()
    return db


def test_hermes_state_db_fills_all_five_token_categories(fake_state_db):
    r = mk(HermesStateDbReconciler, db_path=fake_state_db)
    events = r.reconcile(started_after=999.0, cwd="/w")
    (usage,) = of(events, EventType.MODEL_USAGE_UPDATED)
    # Hermes is the only one of the three that fills every bucket: Claude has no
    # reasoning bucket, Codex has no cache_write bucket.
    assert usage.payload["usage"] == {
        "input": 100, "output": 20, "cache_read": 300,
        "cache_write": 40, "reasoning": 7,
    }
    assert usage.source.capture_mode is CaptureMode.RECONCILED
    assert usage.source.fidelity is Fidelity.DETERMINISTIC


def test_hermes_keeps_its_own_cost_provenance(fake_state_db):
    r = mk(HermesStateDbReconciler, db_path=fake_state_db)
    (usage, _) = r.reconcile(started_after=999.0, cwd="/w")
    assert usage.payload["cost_usd"] == 0.012
    assert usage.payload["cost_status"] == "final"
    assert usage.payload["cost_source"] == "provider"
    assert usage.payload["cost_fidelity"] == Fidelity.NATIVE.value


def test_hermes_ambiguous_join_reports_missing_not_a_guess(fake_state_db):
    conn = sqlite3.connect(fake_state_db)
    conn.execute(
        "INSERT INTO sessions VALUES ('cli_2','cli','hermes-4',1000.5,1011.0,"
        "'complete',6,3,4,999,999,999,999,999,'nous',0.01,0.012,'final','provider','/w')"
    )
    conn.commit()
    conn.close()

    r = mk(HermesStateDbReconciler, db_path=fake_state_db)
    (usage,) = r.reconcile(started_after=999.0, cwd="/w")
    assert usage.source.fidelity is Fidelity.MISSING
    assert usage.payload["usage"] is None
    assert usage.payload["candidates"] == 2
    assert r.usage.by_category == {}  # nothing attributed to the wrong run


def test_hermes_no_match_is_missing_not_zero(fake_state_db):
    r = mk(HermesStateDbReconciler, db_path=fake_state_db)
    (usage,) = r.reconcile(started_after=99_999.0, cwd="/w")
    assert usage.source.fidelity is Fidelity.MISSING
    assert usage.payload["usage"] is None
    assert usage.payload["candidates"] == 0


def test_hermes_missing_db_warns_rather_than_raising(tmp_path):
    r = mk(HermesStateDbReconciler, db_path=tmp_path / "nope.db")
    (warn,) = r.reconcile(started_after=0.0)
    assert warn.event_type is EventType.ADAPTER_WARNING


# ── the boundary holds across all three ──────────────────────────────────────


def test_replayed_events_expose_no_native_keys_to_analysis():
    """M0's exit criterion, re-run on real adapter output rather than on
    hand-built events: nothing agent-specific may reach the analysis surface."""
    everything = (
        replay(mk(CodexExecAdapter), FIXTURES / "codex-exec.jsonl")
        + replay(mk(ClaudeStreamAdapter), FIXTURES / "claude-stream.jsonl")
    )
    assert everything
    for e in everything:
        view = {k for k in e.to_dict() if k in ANALYSIS_KEYS}
        assert "native" not in view and "native_type" not in view
        # and every event knows where it came from
        assert e.source.fidelity in set(Fidelity)
        assert e.source.capture_mode in set(CaptureMode)


def test_no_delta_event_ever_carries_usage():
    """Rule 2, checked structurally instead of by reading each adapter."""
    everything = (
        replay(mk(CodexExecAdapter), FIXTURES / "codex-exec.jsonl")
        + replay(mk(ClaudeStreamAdapter), FIXTURES / "claude-stream.jsonl")
    )
    for e in everything:
        if e.event_type.is_delta:
            assert "usage" not in e.payload


def test_adapter_for_refuses_an_unknown_agent():
    assert isinstance(adapter_for("codex", run_id="r", session_id="s"), CodexExecAdapter)
    with pytest.raises(ValueError, match="no adapter"):
        adapter_for("gemini", run_id="r", session_id="s")
