"""RECONCILED capture: importing sessions that already happened.

The failure this mode invites is not a crash — it is a second, well-formed run
describing an hour of work the store already holds, which doubles every total
built on top of it without anything looking wrong. So most of what is asserted
here is about *refusals*: what the pass declines to import, what it declines to
claim about time it did not measure, and what it prints instead of a zero.

The `codex` under test is `fixtures/seer/fake_codex.py`, serving threads from a
JSON file. `thread/list` there strips `turns` exactly as the real server does,
so a reconciler that forgot to call `thread/read` would import empty runs rather
than passing quietly.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from nebulai.seer.contract import CaptureMode, EventType, Fidelity, Outcome
from nebulai.seer.reconcile import (
    AppServerQuery,
    CodexThreadReconciler,
    ProtocolMismatch,
    reconcile_codex,
    rollout_facts,
)
from nebulai.seer.store import EventStore

FIXTURES = Path(__file__).parent / "fixtures" / "seer"
FAKE = FIXTURES / "fake_codex.py"

#: epoch ms, so the reconciler's own seconds conversion is exercised
T0 = 1_780_000_000_000


def thread(tid: str, *, started: int = T0, status: str = "completed",
           n_turns: int = 1, path: str | None = None) -> dict:
    turns = []
    for i in range(n_turns):
        at = started + i * 60_000
        turns.append({
            "id": f"turn_{tid}_{i}",
            "itemsView": "full",
            "status": status,
            "error": None,
            "startedAt": at,
            "completedAt": at + 30_000,
            "durationMs": 30_000,
            "items": [
                {"id": f"u{i}", "type": "userMessage",
                 "content": [{"type": "text", "text": "do the thing"}]},
                {"id": f"c{i}", "type": "commandExecution",
                 "command": "pytest -q", "exitCode": 0,
                 "aggregatedOutput": "3 passed"},
                {"id": f"a{i}", "type": "agentMessage", "text": "done"},
            ],
        })
    return {
        "id": tid,
        "sessionId": tid,
        "cwd": "/repo",
        "source": "cli",
        "historyMode": "full",
        "cliVersion": "codex-cli 0.144.6",
        "modelProvider": "openai",
        "createdAt": started,
        "updatedAt": started + n_turns * 60_000,
        "name": "a session name that is the user's own words",
        "preview": "and so is this",
        "path": path,
        "turns": turns,
    }


@pytest.fixture
def codex(tmp_path: Path):
    p = tmp_path / "fake-codex"
    p.write_text(f"#!{sys.executable}\n"
                 f"import runpy, sys\n"
                 f"sys.argv[0] = {str(FAKE)!r}\n"
                 f"runpy.run_path({str(FAKE)!r}, run_name='__main__')\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(p)


@pytest.fixture
def store(tmp_path: Path):
    s = EventStore(tmp_path / "seer")
    yield s
    s.close()


@pytest.fixture
def threads_file(tmp_path: Path, monkeypatch):
    def write(*ths: dict) -> Path:
        p = tmp_path / "threads.json"
        p.write_text(json.dumps({"threads": list(ths)}))
        monkeypatch.setenv("FAKE_CODEX_THREADS", str(p))
        return p
    return write


def rollout(tmp_path: Path, name: str = "rollout.jsonl", *,
            total: int = 1200, records: int = 3) -> str:
    """A rollout file with *cumulative* counters reported several times."""
    p = tmp_path / name
    lines = [json.dumps({
        "timestamp": "2026-06-01T00:00:00.000Z", "type": "session_meta",
        "payload": {"session_id": "s", "cli_version": "0.144.6",
                    "git": {"branch": "main", "commit_hash": "abc123"}},
    })]
    for i in range(records):
        lines.append(json.dumps({
            "timestamp": "2026-06-01T00:00:01.000Z", "type": "event_msg",
            "payload": {"type": "token_count", "info": {
                "total_token_usage": {
                    "input_tokens": total, "cached_input_tokens": 900,
                    "output_tokens": 300, "reasoning_output_tokens": 120,
                },
                "model_context_window": 258400,
            }},
        }))
    lines.append(json.dumps({
        "timestamp": "2026-06-01T00:00:02.000Z", "type": "turn_context",
        "payload": {"model": "gpt-5.6-sol", "effort": "high"},
    }))
    p.write_text("\n".join(lines) + "\n")
    return str(p)


# ── the rollout file ─────────────────────────────────────────────────────


class TestRolloutFacts:
    def test_cumulative_counters_are_replaced_not_summed(self, tmp_path):
        """Three reports of the same running total are one total. Adding them
        would treble the session and look entirely plausible doing it."""
        f = rollout_facts(rollout(tmp_path, total=1200, records=3))
        assert f.usage["inputTokens"] == 1200
        assert f.context_window == 258400
        assert f.model == "gpt-5.6-sol"

    def test_a_missing_file_is_a_sentence_not_a_zero(self, tmp_path):
        f = rollout_facts(tmp_path / "nope.jsonl")
        assert f.usage == {}
        assert "gone" in (f.note or "")

    def test_no_path_at_all_is_also_a_sentence(self):
        assert rollout_facts(None).usage == {}
        assert "no rollout path" in (rollout_facts(None).note or "")

    def test_a_file_with_no_counters_says_so(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text(json.dumps({"type": "event_msg",
                                 "payload": {"type": "task_started"}}) + "\n")
        f = rollout_facts(p)
        assert f.usage == {}
        assert "no token counts" in (f.note or "")


# ── replaying one thread ─────────────────────────────────────────────────


class TestReplay:
    def _events(self, th: dict, tmp_path: Path, **kw):
        adapter = CodexThreadReconciler(run_id="run_x", session_id="s", **kw)
        return adapter, adapter.replay(th, rollout_facts(rollout(tmp_path)))

    def test_a_reconciled_run_is_dated_from_the_session_not_from_now(
        self, tmp_path
    ):
        """The single most tempting lie in this mode. `time.time()` would file
        a session from March under today and quietly reorder the history."""
        _, events = self._events(thread("t1"), tmp_path)
        assert all(e.ts < T0 / 1000.0 + 3600 for e in events)
        assert events[0].ts == pytest.approx(T0 / 1000.0)

    def test_the_capture_mode_is_reconciled_all_the_way_through(self, tmp_path):
        _, events = self._events(thread("t1"), tmp_path)
        assert {e.source.capture_mode for e in events} == {CaptureMode.RECONCILED}

    def test_items_replay_through_the_live_adapters_own_mapping(self, tmp_path):
        """Not a second mapping written from the same protocol: the persisted
        `ThreadItem` is the notification's `ThreadItem`, so one route table
        serves both and cannot drift from itself."""
        _, events = self._events(thread("t1"), tmp_path)
        kinds = [e.event_type.value for e in events]
        assert "message.user" in kinds
        assert "tool.completed" in kinds
        assert "message.assistant_completed" in kinds

    def test_an_interrupted_turn_is_not_recorded_as_a_completion(self, tmp_path):
        _, events = self._events(thread("t1", status="interrupted"), tmp_path)
        assert events[-1].event_type is EventType.SESSION_INTERRUPTED
        assert events[-1].payload["outcome"] == Outcome.INTERRUPTED.value

    def test_the_thread_name_and_preview_are_not_stored(self, tmp_path):
        """Both are the user's own prompt text, and this run is imported with
        nobody watching it happen."""
        _, events = self._events(thread("t1"), tmp_path)
        blob = json.dumps([e.to_dict() for e in events])
        assert "the user's own words" not in blob
        assert "and so is this" not in blob
        start = events[0]
        assert start.payload["has_name"] is True

    def test_the_missing_clock_is_named_rather_than_filled_in(self, tmp_path):
        adapter, events = self._events(thread("t1"), tmp_path)
        gaps = events[0].payload["capture_gaps"]
        assert any("per-item timestamps" in g for g in gaps)
        assert any("turn's start time" in w for w in adapter.warnings)


# ── a whole pass ─────────────────────────────────────────────────────────


class TestPass:
    def test_threads_become_runs_with_their_own_tokens(
        self, codex, store, tmp_path, threads_file
    ):
        threads_file(thread("t1", path=rollout(tmp_path, "r1.jsonl", total=1200)))
        report = reconcile_codex(store=store, codex_bin=codex, cwd=tmp_path)
        assert len(report.imported) == 1
        view = report.imported[0].view
        assert view.usage["input"].value == 1200
        assert view.usage["input"].fidelity is Fidelity.NATIVE
        assert store.get_run(report.imported[0].run_id).capture_mode == "reconciled"

    def test_a_second_pass_imports_nothing_twice(
        self, codex, store, tmp_path, threads_file
    ):
        """The whole reason the pass keys on the agent's own thread id. Without
        it a nightly import would multiply a month of tokens by thirty."""
        threads_file(thread("t1", path=rollout(tmp_path)),
                     thread("t2", path=rollout(tmp_path)))
        first = reconcile_codex(store=store, codex_bin=codex, cwd=tmp_path)
        second = reconcile_codex(store=store, codex_bin=codex, cwd=tmp_path)
        assert len(first.imported) == 2
        assert second.imported == []
        assert set(second.skipped) == {"t1", "t2"}
        assert len(store.list_runs()) == 2

    def test_the_dedup_key_survives_losing_the_index(
        self, codex, store, tmp_path, threads_file
    ):
        """`reindex` rebuilds from the logs. If the key lived only in SQLite,
        rebuilding would re-open the door to the double."""
        threads_file(thread("t1", path=rollout(tmp_path)))
        reconcile_codex(store=store, codex_bin=codex, cwd=tmp_path)
        store.reindex()
        again = reconcile_codex(store=store, codex_bin=codex, cwd=tmp_path)
        assert again.imported == []
        assert set(again.skipped) == {"t1"}

    def test_a_thread_with_no_rollout_reports_absent_tokens_not_zero(
        self, codex, store, tmp_path, threads_file
    ):
        threads_file(thread("t1", path=None))
        report = reconcile_codex(store=store, codex_bin=codex, cwd=tmp_path)
        view = report.imported[0].view
        assert view.usage["input"].absent
        assert view.usage["input"].value is None
        assert any("rollout" in w for w in view.quality.warnings)

    def test_a_call_we_never_saw_begin_has_no_duration_rather_than_zero(
        self, codex, store, tmp_path, threads_file
    ):
        """Thread history has no per-item clock, so both ends of a tool span
        carry the same stamp. `0.0s` would be a measurement; there was none."""
        threads_file(thread("t1", path=rollout(tmp_path)))
        report = reconcile_codex(store=store, codex_bin=codex, cwd=tmp_path)
        spans = [s for s in report.imported[0].view.spans if s.synthetic_start]
        assert spans, "the command execution should have produced a span"
        assert all(s.best_duration_s is None for s in spans)
        assert all(s.to_dict()["duration_fidelity"] == "missing" for s in spans)

    def test_the_limit_is_a_limit_and_the_count_is_honest(
        self, codex, store, tmp_path, threads_file
    ):
        threads_file(*[thread(f"t{i}", path=rollout(tmp_path)) for i in range(5)])
        report = reconcile_codex(store=store, codex_bin=codex, cwd=tmp_path,
                                 limit=2)
        assert report.n_seen == 2
        assert len(report.imported) == 2

    def test_paging_follows_the_servers_cursor(
        self, codex, store, tmp_path, threads_file
    ):
        """A pass that stopped at the first page would silently import a
        prefix of the history and report it as the whole thing."""
        threads_file(*[thread(f"t{i}", path=rollout(tmp_path)) for i in range(7)])
        report = reconcile_codex(store=store, codex_bin=codex, cwd=tmp_path,
                                 limit=7)
        assert len(report.imported) == 7

    def test_threads_older_than_the_floor_are_not_counted_as_seen(
        self, codex, store, tmp_path, threads_file
    ):
        threads_file(thread("old", started=T0, path=rollout(tmp_path)))
        report = reconcile_codex(store=store, codex_bin=codex, cwd=tmp_path,
                                 since=T0 / 1000.0 + 86400)
        assert report.n_seen == 0
        assert report.imported == []

    def test_an_unreadable_thread_is_reported_not_swallowed(
        self, codex, store, tmp_path, threads_file, monkeypatch
    ):
        threads_file(thread("t1", path=rollout(tmp_path)))

        def boom(tid: str):
            raise ProtocolMismatch("thread/read failed: gone")

        with AppServerQuery(codex_bin=codex, cwd=tmp_path) as q:
            monkeypatch.setattr(q, "read", boom)
            report = reconcile_codex(store=store, codex_bin=codex,
                                     cwd=tmp_path, query=q)
        assert report.imported == []
        assert "t1" in report.failed

    def test_nothing_at_all_is_a_result_with_a_shape(
        self, codex, store, tmp_path, threads_file
    ):
        threads_file()
        report = reconcile_codex(store=store, codex_bin=codex, cwd=tmp_path)
        assert report.n_seen == 0
        assert report.to_dict()["imported"] == []


# ── the client ───────────────────────────────────────────────────────────


class TestQuery:
    def test_a_missing_binary_refuses_instead_of_hanging(self, tmp_path):
        from nebulai.seer.reconcile import ProtocolMismatch

        with pytest.raises(ProtocolMismatch):
            AppServerQuery(codex_bin="definitely-not-codex").open()

    def test_listing_never_asks_the_server_to_repair_anything(
        self, codex, tmp_path, threads_file, monkeypatch
    ):
        """`useStateDbOnly` off makes `thread/list` rescan *and rewrite* rollout
        metadata. A reader has no business triggering a write."""
        threads_file(thread("t1"))
        sent: list[dict] = []
        with AppServerQuery(codex_bin=codex, cwd=tmp_path) as q:
            real = q.request

            def spy(method, params, **kw):
                sent.append({"method": method, "params": params})
                return real(method, params, **kw)

            monkeypatch.setattr(q, "request", spy)
            list(q.threads(limit=1))
        listing = [s for s in sent if s["method"] == "thread/list"]
        assert listing and all(s["params"]["useStateDbOnly"] for s in listing)
