"""M1b exit tests: the log is the record, the reducer is honest, the gate refuses.

The three properties these have to hold down:

* the SQLite index can be deleted and rebuilt from the JSONL with no loss, so
  nothing a chart depends on lives only in a cache;
* the reducer never turns "we don't know" into a zero, and never counts a
  streaming fragment;
* `compare()` refuses a token delta between Codex and Claude, on the real
  captures, for the real reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nebulai.seer.adapters import ClaudeStreamAdapter, CodexExecAdapter
from nebulai.seer.compare import compare, summarize_refusals
from nebulai.seer.contract import (
    Action,
    CaptureMode,
    Effect,
    Event,
    EventType,
    Fidelity,
    Outcome,
    SessionState,
    Source,
    StateOverlay,
    TokenCategory,
)
from nebulai.seer.reducer import Reducer, reduce_run
from nebulai.seer.store import EventStore

FIXTURES = Path(__file__).parent / "fixtures" / "seer"


def replay_view(cls, fixture: str, run_id: str):
    a = cls(run_id=run_id, session_id="s")
    events = []
    for line in (FIXTURES / fixture).read_text().splitlines():
        events.extend(a.feed(line))
    events.extend(a.finish())
    return reduce_run(run_id, events, now=events[-1].ts), events


def _src(**kw) -> Source:
    return Source(
        agent=kw.pop("agent", "codex"),
        agent_version="1",
        adapter="t",
        adapter_version="1",
        capture_mode=kw.pop("capture_mode", CaptureMode.DRIVEN),
        fidelity=kw.pop("fidelity", Fidelity.NATIVE),
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


# ── store ────────────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "seer")
    yield s
    s.close()


def test_append_writes_one_json_line_per_event(store):
    _, events = replay_view(CodexExecAdapter, "codex-tools.jsonl", "run_a")
    store.append_many(events)
    lines = store.log_path("run_a").read_text().splitlines()
    assert len(lines) == len(events)
    assert json.loads(lines[0])["event_type"] == "session.started"


def test_read_round_trips_every_event(store):
    _, events = replay_view(ClaudeStreamAdapter, "claude-tools.jsonl", "run_b")
    store.append_many(events)
    back = list(store.read("run_b"))
    assert [e.event_id for e in back] == [e.event_id for e in events]
    assert [e.to_dict() for e in back] == [e.to_dict() for e in events]


def test_index_is_a_cache_and_can_be_rebuilt_from_the_log(store, tmp_path):
    _, events = replay_view(CodexExecAdapter, "codex-tools.jsonl", "run_c")
    store.append_many(events)
    before = store.get_run("run_c")

    # Nuke the index the way a corrupted file or a schema bump would.
    store._conn.execute("DELETE FROM events")
    store._conn.execute("DELETE FROM runs")
    store._conn.commit()
    assert store.get_run("run_c") is None

    n = store.reindex()
    after = store.get_run("run_c")
    assert n == len(events)
    assert after is not None
    assert (after.agent, after.n_events, after.outcome) == (
        before.agent, before.n_events, before.outcome
    )
    # state and ended_at too: a rebuilt index that showed every finished run as
    # still running would mean the index held something the log could not
    # restore, which is precisely what reindex exists to disprove
    assert (after.state, after.ended_at) == (before.state, before.ended_at)
    assert after.state == "completed"


def test_ended_at_is_the_sessions_ending_not_our_bookkeeping(store):
    """RUN_COMPLETED lands after SESSION_COMPLETED. If the index counted it,
    the run list and the run detail would print different wall clocks for the
    same run."""
    store.append(ev(EventType.SESSION_STARTED, 100.0, run_id="r"))
    store.append(ev(EventType.SESSION_COMPLETED, 110.0, run_id="r"))
    store.append(ev(EventType.RUN_COMPLETED, 110.9, run_id="r"))
    assert store.get_run("r").ended_at == 110.0


def test_reading_never_consults_the_index(store):
    _, events = replay_view(CodexExecAdapter, "codex-exec.jsonl", "run_d")
    store.append_many(events)
    store._conn.execute("DELETE FROM events")
    store._conn.commit()
    # the log still answers correctly with an empty index
    assert len(list(store.read("run_d"))) == len(events)


def test_list_runs_is_newest_first(store):
    store.append(ev(EventType.SESSION_STARTED, 100.0, run_id="old"))
    store.append(ev(EventType.SESSION_STARTED, 200.0, run_id="new"))
    assert [r.run_id for r in store.list_runs()] == ["new", "old"]


def test_terminal_event_records_the_outcome(store):
    store.append(ev(EventType.SESSION_STARTED, 1.0))
    store.append(
        ev(
            EventType.SESSION_COMPLETED, 9.0,
            payload={"outcome": Outcome.AGENT_CLAIMED_COMPLETE.value},
        )
    )
    run = store.get_run("run_1")
    assert run.outcome == "agent_claimed_complete"
    assert run.ended_at == 9.0


def test_tail_returns_the_last_n(store):
    for i in range(50):
        store.append(ev(EventType.TOOL_COMPLETED, float(i), action=Action.INSPECT))
    tail = store.tail("run_1", n=5)
    assert [e.ts for e in tail] == [45.0, 46.0, 47.0, 48.0, 49.0]


# ── reducer: state machine ───────────────────────────────────────────────────


def test_state_machine_walks_a_normal_run():
    r = Reducer("run_1")
    r.push(ev(EventType.SESSION_STARTED, 0.0))
    assert r.view.state is SessionState.STARTING
    r.push(ev(EventType.TURN_STARTED, 1.0))
    assert r.view.state is SessionState.MODEL_RUNNING
    r.push(ev(EventType.TOOL_STARTED, 2.0, span_id="s1", action=Action.EXECUTE))
    assert r.view.state is SessionState.TOOL_RUNNING
    r.push(ev(EventType.TOOL_COMPLETED, 5.0, span_id="s1", action=Action.EXECUTE))
    assert r.view.state is SessionState.MODEL_RUNNING
    r.push(ev(EventType.TURN_COMPLETED, 6.0))
    assert r.view.state is SessionState.IDLE
    v = r.finalize(now=6.0)
    assert v.time_in_state["tool_running"] == 3.0
    assert v.time_in_state["starting"] == 1.0


def test_parallel_tools_do_not_flap_the_state():
    """Three tools running, one finishes: the session is still running tools.
    Flapping to model_running would make every parallel batch look like a
    stall-and-resume in the timeline."""
    r = Reducer("run_1")
    for i in range(3):
        r.push(ev(EventType.TOOL_STARTED, float(i), span_id=f"s{i}", action=Action.INSPECT))
    r.push(ev(EventType.TOOL_COMPLETED, 10.0, span_id="s0", action=Action.INSPECT))
    assert r.view.state is SessionState.TOOL_RUNNING
    r.push(ev(EventType.TOOL_COMPLETED, 11.0, span_id="s1", action=Action.INSPECT))
    r.push(ev(EventType.TOOL_COMPLETED, 12.0, span_id="s2", action=Action.INSPECT))
    assert r.view.state is SessionState.MODEL_RUNNING


def test_stalled_is_an_overlay_over_the_real_state():
    r = Reducer("run_1")
    r.push(ev(EventType.TOOL_STARTED, 0.0, span_id="s1", action=Action.EXECUTE))
    assert r.overlays(now=10.0) == []
    assert r.overlays(now=10_000.0) == [StateOverlay.STALLED]
    # the state itself is untouched — we still know what it is stalled doing
    assert r.view.state is SessionState.TOOL_RUNNING


def test_a_human_wait_is_overdue_not_stalled():
    """Different overlay because the response is different: one needs a person,
    the other needs someone to check the process is alive."""
    r = Reducer("run_1")
    r.push(ev(EventType.APPROVAL_REQUESTED, 0.0, action=Action.INTERACT))
    assert r.overlays(now=10_000.0) == [StateOverlay.OVERDUE]


def test_a_finished_run_never_stalls():
    r = Reducer("run_1")
    r.push(ev(EventType.SESSION_COMPLETED, 0.0, payload={"outcome": "verified_pass"}))
    assert r.overlays(now=1e9) == []


def test_unclosed_span_survives_finalize():
    """The tool a hung run is hung on is the one thing you must not drop."""
    r = Reducer("run_1")
    r.push(ev(EventType.TOOL_STARTED, 0.0, span_id="s1", action=Action.EXECUTE,
              payload={"command": "pytest"}))
    v = r.finalize(now=100.0)
    (span,) = v.spans
    assert span.ended_at is None and span.duration_s is None
    assert span.detail == "pytest"


def test_completion_without_a_start_is_still_a_tool_call():
    """OBSERVED captures can begin mid-flight. A missing call is a worse lie
    than a call with unknown duration."""
    r = Reducer("run_1")
    r.push(ev(EventType.TOOL_COMPLETED, 5.0, span_id="s9", action=Action.VERIFY))
    v = r.finalize(now=5.0)
    assert len(v.spans) == 1
    assert v.action_counts["verify"] == 1


# ── reducer: the counting rules ──────────────────────────────────────────────


def test_deltas_never_reach_a_counter():
    r = Reducer("run_1")
    for i in range(20):
        r.push(ev(EventType.TOOL_OUTPUT_DELTA, float(i), span_id="s1",
                  action=Action.EXECUTE, effect=Effect.STATE_CHANGED))
    r.push(ev(EventType.TOOL_COMPLETED, 20.0, span_id="s1", action=Action.EXECUTE,
              effect=Effect.STATE_CHANGED))
    v = r.finalize(now=20.0)
    assert v.action_counts == {"execute": 1}
    assert v.effect_counts == {"state_changed": 1}
    assert len(v.spans) == 1


def test_a_tool_call_is_counted_once_not_at_both_ends():
    r = Reducer("run_1")
    r.push(ev(EventType.TOOL_STARTED, 0.0, span_id="s1", action=Action.VERIFY))
    r.push(ev(EventType.TOOL_COMPLETED, 1.0, span_id="s1", action=Action.VERIFY))
    assert r.finalize(now=1.0).action_counts == {"verify": 1}


def test_verification_after_last_edit_catches_the_stale_test_run():
    """Tested, then edited again, then stopped. A plain `verified` flag calls
    that covered; this is the check that doesn't."""
    r = Reducer("run_1")
    r.push(ev(EventType.TOOL_STARTED, 0.0, span_id="e1", action=Action.EDIT))
    r.push(ev(EventType.TOOL_COMPLETED, 1.0, span_id="e1", action=Action.EDIT))
    r.push(ev(EventType.TOOL_STARTED, 2.0, span_id="v1", action=Action.VERIFY))
    r.push(ev(EventType.TOOL_COMPLETED, 3.0, span_id="v1", action=Action.VERIFY))
    r.push(ev(EventType.TOOL_STARTED, 4.0, span_id="e2", action=Action.EDIT))
    r.push(ev(EventType.TOOL_COMPLETED, 5.0, span_id="e2", action=Action.EDIT))
    v = r.finalize(now=5.0)
    assert v.verified is True  # the coarse flag says yes …
    assert v.verification_after_last_edit().value == 0  # … and it is wrong


def test_verification_after_last_edit_is_missing_when_nothing_was_edited():
    r = Reducer("run_1")
    r.push(ev(EventType.TOOL_COMPLETED, 1.0, span_id="s1", action=Action.INSPECT))
    m = r.finalize(now=1.0).verification_after_last_edit()
    assert m.value is None and m.fidelity is Fidelity.MISSING


# ── reducer: provenance ──────────────────────────────────────────────────────


def test_codex_run_shows_cache_write_absent_not_zero():
    v, _ = replay_view(CodexExecAdapter, "codex-tools.jsonl", "run_x")
    cw = v.usage[TokenCategory.CACHE_WRITE.value]
    assert cw.value is None and cw.absent
    assert "cache_write" in v.quality.absent_token_categories
    # and the categories Codex does report are real numbers
    assert v.usage["input"].value == 42575
    assert v.usage["reasoning"].value == 54


def test_claude_run_shows_reasoning_absent_not_zero():
    v, _ = replay_view(ClaudeStreamAdapter, "claude-tools.jsonl", "run_y")
    assert v.usage[TokenCategory.REASONING.value].absent
    assert v.usage["cache_write"].value == 10614
    assert v.usage["output"].value == 872


def test_folded_duplicates_are_visible_rather_than_silent():
    v, _ = replay_view(ClaudeStreamAdapter, "claude-stream.jsonl", "run_z")
    # the capture repeats one message's usage across two lines
    assert v.quality.folded_duplicates >= 1


def test_dropped_by_policy_is_counted_separately_from_missing():
    v, _ = replay_view(ClaudeStreamAdapter, "claude-tools.jsonl", "run_w")
    assert sum(v.quality.dropped_by_policy.values()) > 0
    assert all("thinking" in k or "reasoning" in k
               for k in v.quality.dropped_by_policy)


def test_capture_gaps_reach_the_data_quality_panel():
    v, _ = replay_view(CodexExecAdapter, "codex-exec.jsonl", "run_g")
    assert any("approval" in g for g in v.quality.capture_gaps)
    assert v.quality.capture_mode == "driven"


# ── the gate ─────────────────────────────────────────────────────────────────


@pytest.fixture
def two_real_runs():
    codex, _ = replay_view(CodexExecAdapter, "codex-tools.jsonl", "run_codex")
    claude, _ = replay_view(ClaudeStreamAdapter, "claude-tools.jsonl", "run_claude")
    return [codex, claude]


def test_gate_refuses_tokens_between_codex_and_claude(two_real_runs):
    """The empirical case, on real captures of the same task: Claude bills
    reasoning inside output and has a cache-write bucket; Codex breaks reasoning
    out and has none. `output_tokens` is not the same quantity."""
    c = compare(two_real_runs)
    r = c.refusal("tokens.*")
    assert r is not None
    assert "do not align" in r.reason
    assert not [m for m in c.comparable if m.metric.startswith("tokens.")]


def test_gate_still_compares_what_our_own_reducer_computed(two_real_runs):
    c = compare(two_real_runs)
    names = {m.metric for m in c.comparable}
    assert "wall_clock_s" in names
    assert "action.edit" in names
    assert "n_files_changed" in names
    # both runs did the same task; both edited exactly one file
    edit = c.metric("n_files_changed")
    assert {m.value for m in edit.values.values()} == {1}


def test_refusing_one_metric_does_not_suppress_the_rest(two_real_runs):
    c = compare(two_real_runs)
    assert c.refused and c.comparable
    assert len(c.comparable) > 5


def test_gate_refuses_a_metric_a_capture_mode_cannot_see():
    """A DRIVEN Codex run reports zero approvals because it cannot see them,
    not because none happened. Comparing that against a mode that can see them
    would score Codex as the agent that never asks."""
    codex, _ = replay_view(CodexExecAdapter, "codex-tools.jsonl", "run_codex")
    claude, _ = replay_view(ClaudeStreamAdapter, "claude-tools.jsonl", "run_claude")
    c = compare([codex, claude])
    r = c.refusal("action.interact")
    assert r is not None
    assert "approval" in r.reason
    assert r.runs == ["run_codex"]


def test_gate_refuses_native_versus_estimated():
    a = reduce_run("a", [
        ev(EventType.SESSION_STARTED, 0.0),
        ev(EventType.MODEL_USAGE_UPDATED, 1.0, source=_src(fidelity=Fidelity.NATIVE),
           payload={"usage": {"input": 10}, "native_categories": ["input_tokens"],
                    "authoritative": True}),
        ev(EventType.SESSION_COMPLETED, 2.0, payload={"outcome": "unknown"}),
    ], now=2.0)
    b = reduce_run("b", [
        ev(EventType.SESSION_STARTED, 0.0, run_id="b"),
        ev(EventType.MODEL_USAGE_UPDATED, 1.0, run_id="b",
           source=_src(agent="hermes", fidelity=Fidelity.ESTIMATED),
           payload={"usage": {"input": 12}, "native_categories": ["input_tokens"],
                    "authoritative": True}),
        ev(EventType.SESSION_COMPLETED, 2.0, run_id="b", payload={"outcome": "unknown"}),
    ], now=2.0)

    c = compare([a, b])
    r = c.refusal("tokens.input")
    assert r is not None
    assert "fidelity mismatch" in r.reason


def test_gate_refuses_present_for_one_absent_for_another():
    a = reduce_run("a", [
        ev(EventType.SESSION_STARTED, 0.0),
        ev(EventType.SESSION_COMPLETED, 1.0,
           payload={"outcome": "unknown", "cost_usd": None}),
    ], now=1.0)
    a.cost_usd.value = 0.02
    a.cost_usd.fidelity = Fidelity.NATIVE
    b = reduce_run("b", [
        ev(EventType.SESSION_STARTED, 0.0, run_id="b"),
        ev(EventType.SESSION_COMPLETED, 1.0, run_id="b", payload={"outcome": "unknown"}),
    ], now=1.0)

    c = compare([a, b])
    r = c.refusal("cost_usd")
    assert r is not None
    assert "absent" in r.reason
    assert r.runs == ["b"]


def test_identical_agents_compare_on_tokens():
    """The gate must not be a blanket refusal — two Claude runs are comparable
    on tokens, and refusing there would make it useless."""
    a, _ = replay_view(ClaudeStreamAdapter, "claude-tools.jsonl", "run_1")
    b, _ = replay_view(ClaudeStreamAdapter, "claude-stream.jsonl", "run_2")
    c = compare([a, b])
    assert c.refusal("tokens.*") is None
    assert c.metric("tokens.output") is not None
    assert c.metric("tokens.output").values["run_1"].value == 872


def test_summary_reads_like_a_sentence(two_real_runs):
    text = summarize_refusals(compare(two_real_runs))
    assert "are comparable" in text
    assert "tokens.*" in text


def test_compare_needs_two_runs():
    v = reduce_run("a", [ev(EventType.SESSION_STARTED, 0.0)], now=0.0)
    with pytest.raises(ValueError, match="at least two"):
        compare([v])


def test_readers_never_share_a_connection_with_the_writer(store):
    """A threaded HTTP server reads while a run is being captured.

    On a single shared `sqlite3.Connection` that raises
    `InterfaceError: bad parameter or other API misuse` at random, and the
    handler above it turns a live run into a 400. The store has to hold under
    concurrent readers, not merely usually work.
    """
    import threading

    _, events = replay_view(ClaudeStreamAdapter, "claude-tools.jsonl", "run_hot")
    store.append_many(events[:1])

    errors: list[BaseException] = []
    stop = threading.Event()

    def write() -> None:
        try:
            for e in events[1:]:
                store.append_many([e])
        except BaseException as exc:  # noqa: BLE001 — the test is the assertion
            errors.append(exc)
        finally:
            stop.set()

    def read() -> None:
        try:
            while not stop.is_set():
                assert store.get_run("run_hot") is not None
                store.list_runs()
                store.tail("run_hot", 20)
                list(store.read("run_hot"))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=write)] + [
        threading.Thread(target=read) for _ in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, errors
    assert store.get_run("run_hot").n_events == len(events)


def test_a_half_written_last_line_is_not_an_error(store):
    """The writer flushes per event, but a reader can still arrive between the
    bytes of one line. That single partial line is skipped; a corrupt line
    anywhere earlier is not, because silently dropping it would let what we
    report drift from the record."""
    _, events = replay_view(CodexExecAdapter, "codex-tools.jsonl", "run_partial")
    store.append_many(events)
    path = store.log_path("run_partial")

    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"event_type": "session.comp')  # caught mid-append
    assert len(list(store.read("run_partial"))) == len(events)

    path.write_text("not json\n" + path.read_text())
    with pytest.raises(ValueError):
        list(store.read("run_partial"))
