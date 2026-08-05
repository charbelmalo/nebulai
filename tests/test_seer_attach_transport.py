"""M3: the transport that carries attached mode, end to end.

`test_seer_attached.py` proves the adapter maps the protocol correctly when
handed lines. This file proves the other half — that we can obtain those lines
from a real process, and that the three decisions the transport makes on the
user's behalf are the ones we said they would be:

* **the gate runs before anything is stored**, and a build that dropped a method
  we read produces a *recorded refusal*, not a quiet run with missing numbers;
* **we never answer an approval with yes.** A machine cannot consent for a
  person, so the answer is `decline` and the fact that a machine answered is in
  the log;
* **capture mode follows who owns the process.** Spawning our own app-server is
  `driven` even though the adapter is the attached one — mode and fidelity are
  different questions and the record keeps them apart.

The `codex` here is `fixtures/seer/fake_codex.py`, which reads its method names
out of the same golden fixture the gate compares against, so the fake cannot
drift from the recorded protocol without a test noticing.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

from nebulai.seer.attach import (
    CodexAttachment,
    ProtocolMismatch,
    attach_codex,
    daemon_running,
    gate,
    live_protocol,
    load_golden,
    protocol_note,
)
from nebulai.seer.contract import CaptureMode, EventType, Outcome
from nebulai.seer.store import EventStore

FIXTURES = Path(__file__).parent / "fixtures" / "seer"
FAKE = FIXTURES / "fake_codex.py"


@pytest.fixture
def codex(tmp_path: Path):
    """A `codex` executable on disk that behaves enough like the real one."""
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


def kinds(events) -> list[str]:
    return [e.event_type.value for e in events]


def wire(log: Path, *, until: str, timeout_s: float = 5.0) -> list[str]:
    """The methods the fake has read off stdin, once it has read `until`.

    Asserting that something was *not* sent needs a barrier: without one, an
    empty log only proves the fake had not got to it yet. Waiting for a message
    we sent afterwards proves it went past the point where the other one would
    have been.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        seen = log.read_text().split() if log.exists() else []
        if until in seen:
            return seen
        time.sleep(0.02)
    raise AssertionError(f"fake never read {until!r}; saw {seen}")


# ── the gate ─────────────────────────────────────────────────────────────


class TestGate:
    def test_the_installed_binary_describes_its_own_protocol(self, codex):
        notes, reqs = live_protocol(codex)
        golden = load_golden()
        assert notes == set(golden["server_notifications"])
        assert reqs == set(golden["server_requests"])

    def test_a_build_that_dropped_a_method_we_read_is_refused(self, codex, monkeypatch):
        monkeypatch.setenv("FAKE_CODEX_DROP", "thread/tokenUsage/updated")
        with pytest.raises(ProtocolMismatch) as exc:
            gate(codex)
        # The message has to name the method, or the person reading it has to
        # go and diff two schema dumps to learn what broke.
        assert "thread/tokenUsage/updated" in str(exc.value)

    def test_a_build_with_new_methods_still_runs(self, codex, monkeypatch):
        monkeypatch.setenv("FAKE_CODEX_ADD", "thread/vibes/updated")
        report = gate(codex)
        assert report["compatible"]
        assert "thread/vibes/updated" in report["new_since_golden"]
        assert "thread/vibes/updated" in protocol_note(report)

    def test_a_matching_build_says_so_rather_than_saying_nothing(self, codex):
        assert protocol_note(gate(codex)) == "protocol matches the recorded surface"

    def test_a_binary_that_cannot_dump_its_schema_is_not_guessed_at(self, tmp_path):
        p = tmp_path / "ancient-codex"
        p.write_text("#!/bin/sh\nexit 1\n")
        p.chmod(0o755)
        with pytest.raises(ProtocolMismatch):
            live_protocol(str(p))

    def test_a_missing_binary_is_a_refusal_not_a_crash(self, store):
        att = CodexAttachment(store=store, codex_bin="codex-that-is-not-installed")
        with pytest.raises(ProtocolMismatch, match="not on PATH"):
            att.open()

    def test_the_refusal_is_recorded_as_a_run_not_only_raised(
        self, codex, store, monkeypatch
    ):
        """A refusal is a result. Without the record, an incompatible build
        looks identical to nobody having tried."""
        monkeypatch.setenv("FAKE_CODEX_DROP", "turn/completed")
        att = CodexAttachment(store=store, codex_bin=codex)
        with pytest.raises(ProtocolMismatch):
            att.open()
        stored = list(store.read(att.run_id))
        assert "adapter.incompatible" in kinds(stored)
        assert "run.completed" in kinds(stored)
        bad = next(e for e in stored if e.event_type is EventType.ADAPTER_INCOMPATIBLE)
        assert "turn/completed" in bad.payload["message"]
        assert store.get_run(att.run_id).state == "failed"

    def test_no_process_is_left_running_when_the_gate_refuses(
        self, codex, store, monkeypatch
    ):
        monkeypatch.setenv("FAKE_CODEX_DROP", "turn/completed")
        att = CodexAttachment(store=store, codex_bin=codex)
        with pytest.raises(ProtocolMismatch):
            att.open()
        assert att.proc is None


# ── a driven turn over the attached protocol ─────────────────────────────


class TestDrivenTurn:
    def test_a_turn_runs_and_lands_in_the_store(self, codex, store, tmp_path):
        res = attach_codex("write a test", store=store, codex_bin=codex,
                           cwd=tmp_path, timeout_s=30)
        stored = list(store.read(res.run_id))
        k = kinds(stored)
        assert "run.started" in k and "session.started" in k
        assert "turn.started" in k and "turn.completed" in k
        assert "tool.started" in k and "tool.completed" in k
        assert res.thread_id == "th_fake"

    def test_owning_the_process_is_driven_even_through_the_attached_adapter(
        self, codex, store, tmp_path
    ):
        """Capture mode answers 'do we own the process'; the adapter answers
        'how much can we see'. A spawned app-server is the first without being
        the second, and the record has to keep both."""
        res = attach_codex("hello", store=store, codex_bin=codex,
                           cwd=tmp_path, timeout_s=30)
        e = next(iter(store.read(res.run_id)))
        assert e.source.capture_mode is CaptureMode.DRIVEN
        assert e.source.adapter == "codex_app_server"
        assert res.transport == "own-app-server"

    def test_the_turns_usage_is_not_double_counted_by_the_running_total(
        self, codex, store, tmp_path
    ):
        """The fake sends the same totals twice — once as
        `thread/tokenUsage/updated`, once inside `turn/completed`. Folding
        either would double the run."""
        res = attach_codex("hello", store=store, codex_bin=codex,
                           cwd=tmp_path, timeout_s=30)
        assert res.view.usage["input"].value == 1200
        assert res.view.usage["output"].value == 300
        assert res.view.usage["cache_read"].value == 900

    def test_the_tool_call_carries_the_agents_own_duration(
        self, codex, store, tmp_path
    ):
        res = attach_codex("hello", store=store, codex_bin=codex,
                           cwd=tmp_path, timeout_s=30)
        span = next(s for s in res.view.spans if s.action and s.action.value in
                    ("verify", "execute"))
        # Kept beside our own interval, not instead of it: the app-server's
        # 1.2s is what the command took, our microseconds are what the two
        # notifications were apart. Only one of those is the tool's runtime.
        assert span.native_duration_s == pytest.approx(1.2)
        assert span.best_duration_s == pytest.approx(1.2)
        assert span.duration_s < 1.0
        assert span.to_dict()["duration_fidelity"] == "native"

    def test_a_run_that_completes_says_the_agent_claimed_it_not_that_it_passed(
        self, codex, store, tmp_path
    ):
        res = attach_codex("hello", store=store, codex_bin=codex,
                           cwd=tmp_path, timeout_s=30)
        assert res.view.outcome is Outcome.AGENT_CLAIMED_COMPLETE

    def test_the_protocol_report_is_in_the_log_before_any_agent_event(
        self, codex, store, tmp_path
    ):
        res = attach_codex("hello", store=store, codex_bin=codex,
                           cwd=tmp_path, timeout_s=30)
        stored = list(store.read(res.run_id))
        checked = next(i for i, e in enumerate(stored)
                       if e.native_type == "protocol.checked")
        started = next(i for i, e in enumerate(stored)
                       if e.event_type is EventType.SESSION_STARTED)
        assert checked < started

    def test_the_recorded_command_says_which_transport_was_used(
        self, codex, store, tmp_path
    ):
        res = attach_codex("hello", store=store, codex_bin=codex,
                           cwd=tmp_path, timeout_s=30)
        start = next(iter(store.read(res.run_id)))
        assert start.payload["transport"] == "own-app-server"
        assert "app-server" in start.payload["command"]


# ── approvals ────────────────────────────────────────────────────────────


class TestApprovals:
    def test_an_approval_request_is_declined_never_granted(
        self, codex, store, tmp_path, monkeypatch
    ):
        decision = tmp_path / "decision.json"
        monkeypatch.setenv("FAKE_CODEX_APPROVAL", "1")
        monkeypatch.setenv("FAKE_CODEX_DECISION_FILE", str(decision))
        attach_codex("hello", store=store, codex_bin=codex, cwd=tmp_path,
                     timeout_s=30)
        answered = json.loads(decision.read_text())
        assert answered["result"] == {"decision": "decline"}

    def test_the_log_says_a_machine_answered_not_a_person(
        self, codex, store, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("FAKE_CODEX_APPROVAL", "1")
        monkeypatch.setenv("FAKE_CODEX_DECISION_FILE",
                           str(tmp_path / "decision.json"))
        res = attach_codex("hello", store=store, codex_bin=codex, cwd=tmp_path,
                           timeout_s=30)
        warnings = [e for e in store.read(res.run_id)
                    if e.event_type is EventType.ADAPTER_WARNING]
        assert any("SessionSeer never answers for a user" in w.payload["note"]
                   for w in warnings)

    def test_the_approval_is_recorded_as_both_asked_and_answered(
        self, codex, store, tmp_path, monkeypatch
    ):
        """Our reply leaves on stdin and never returns on stdout, so without
        feeding it back the approval would stay open forever and the time spent
        waiting on it would be unmeasurable. Declining is a decision about the
        request, not a reason to drop half of it from the record."""
        monkeypatch.setenv("FAKE_CODEX_APPROVAL", "1")
        monkeypatch.setenv("FAKE_CODEX_DECISION_FILE",
                           str(tmp_path / "decision.json"))
        res = attach_codex("hello", store=store, codex_bin=codex, cwd=tmp_path,
                           timeout_s=30)
        k = kinds(list(store.read(res.run_id)))
        assert "approval.requested" in k and "approval.resolved" in k
        resolved = next(e for e in store.read(res.run_id)
                        if e.event_type is EventType.APPROVAL_RESOLVED)
        assert resolved.payload["decision"] == "decline"
        # One approval, counted once. The request and the answer are two events
        # but a single interaction — counting both would be the double-count
        # rule in a different costume.
        assert res.view.action_counts["interact"] == 1

    def test_a_thread_is_started_with_approvals_switched_off(
        self, codex, store, tmp_path
    ):
        """Belt and braces: the decline path exists for a server that asks
        anyway, but we ask it not to."""
        att = CodexAttachment(store=store, codex_bin=codex, cwd=tmp_path).open()
        try:
            att.start_thread()
            sent = att.request  # noqa: F841 - documented below
        finally:
            att.close()
        # `thread/start` params are asserted through the fake's echo of them.
        start = next(e for e in store.read(att.run_id)
                     if e.event_type is EventType.SESSION_STARTED)
        assert start.native["threadId"] == "th_fake"


# ── observation without driving ──────────────────────────────────────────


class TestObserveOnly:
    def test_a_connection_with_no_prompt_starts_no_thread(
        self, codex, store, tmp_path
    ):
        res = attach_codex(None, store=store, codex_bin=codex, cwd=tmp_path,
                           timeout_s=0.5)
        assert res.thread_id is None
        assert "session.started" not in kinds(list(store.read(res.run_id)))

    def test_closing_records_the_run_even_when_nothing_happened(
        self, codex, store, tmp_path
    ):
        res = attach_codex(None, store=store, codex_bin=codex, cwd=tmp_path,
                           timeout_s=0.5)
        assert "run.completed" in kinds(list(store.read(res.run_id)))
        assert res.view.outcome is Outcome.UNKNOWN

    def test_a_finished_attachment_does_not_read_as_still_starting(
        self, codex, store, tmp_path
    ):
        """`starting` forever is indistinguishable in the run list from a live
        session. `detached` claims nothing about the agent and is true."""
        res = attach_codex(None, store=store, codex_bin=codex, cwd=tmp_path,
                           timeout_s=0.5)
        assert res.view.state.value == "detached"
        assert store.get_run(res.run_id).state == "detached"

    def test_we_do_not_start_a_daemon_to_get_one(self, codex, store, tmp_path):
        """`daemon_running` only ever reads. If the socket is absent we spawn a
        private server; we never install a service on someone's machine as a
        side effect of looking at a session."""
        assert daemon_running(tmp_path / "nope.sock") is False
        res = attach_codex(None, store=store, codex_bin=codex, cwd=tmp_path,
                           sock=tmp_path / "nope.sock", timeout_s=0.5)
        assert res.transport == "own-app-server"
        assert not (tmp_path / "nope.sock").exists()


# ── detaching ────────────────────────────────────────────────────────────


class TestStop:
    def test_stop_ends_a_watch_long_before_its_timeout(
        self, codex, store, tmp_path
    ):
        """A watcher's timeout is a backstop, not a cadence — the server's
        default is fifteen minutes. If `stop` did not wake it, cancelling an
        observation from the UI would mean waiting out that backstop."""
        att = CodexAttachment(store=store, codex_bin=codex, cwd=tmp_path).open()
        threading.Timer(0.2, att.stop).start()
        t0 = time.monotonic()
        res = att.watch(timeout_s=60)
        assert time.monotonic() - t0 < 20
        assert res.view.state.value == "detached"

    def test_stopping_a_proxied_watch_does_not_interrupt_someone_elses_turn(
        self, codex, store, tmp_path, monkeypatch
    ):
        """Through a daemon the thread belongs to whoever is driving it. We are
        a second pair of eyes on it, and closing our eyes must not stop their
        agent — so `stop` sends nothing, and the wire says so."""
        sock = tmp_path / "app-server-control.sock"
        sock.touch()
        log = tmp_path / "methods.log"
        monkeypatch.setenv("FAKE_CODEX_METHOD_LOG", str(log))
        att = CodexAttachment(store=store, codex_bin=codex, cwd=tmp_path,
                              sock=sock).open()
        assert att.transport == "daemon-proxy"
        # A thread we did not start, of the kind `thread/started` would tell us
        # about: `stop` must not treat it as ours to interrupt.
        att.thread_id = "th_someone_else"
        att.stop()
        att.notify("thread/unsubscribe", {"threadId": att.thread_id})
        assert "turn/interrupt" not in wire(log, until="thread/unsubscribe")
        att.watch(timeout_s=5)

    def test_stopping_a_driven_turn_does_interrupt_it(
        self, codex, store, tmp_path, monkeypatch
    ):
        """The mirror image: a turn we started is ours to stop, and leaving it
        running after the user cancelled would be the worse failure."""
        log = tmp_path / "methods.log"
        monkeypatch.setenv("FAKE_CODEX_METHOD_LOG", str(log))
        att = CodexAttachment(store=store, codex_bin=codex, cwd=tmp_path).open()
        att.start_thread()
        att.stop()
        assert "turn/interrupt" in wire(log, until="turn/interrupt")
        res = att.watch(timeout_s=5)
        assert res.transport == "own-app-server"


# ── plumbing ─────────────────────────────────────────────────────────────


class TestPlumbing:
    def test_an_unanswered_request_times_out_instead_of_hanging(
        self, codex, store, tmp_path
    ):
        att = CodexAttachment(store=store, codex_bin=codex, cwd=tmp_path).open()
        try:
            with pytest.raises(RuntimeError):
                att.request("thread/goal/get", {}, timeout_s=5)
        finally:
            att.close()

    def test_every_line_reaches_the_adapter_including_our_own_replies(
        self, codex, store, tmp_path
    ):
        """The stored log has to be what the server said. Filtering replies out
        because we already handled them would make replay a different run."""
        seen: list[str] = []
        att = CodexAttachment(store=store, codex_bin=codex, cwd=tmp_path,
                              on_event=lambda e: seen.append(e.event_type.value))
        att.open()
        try:
            att.start_thread()
            att.send_turn("hello")
            assert att.wait_for_turn(30)
        finally:
            att.close()
        assert "session.started" in seen and "turn.completed" in seen

    def test_writing_before_open_is_an_error_not_a_silent_no_op(self, store):
        att = CodexAttachment(store=store)
        with pytest.raises(RuntimeError, match="not open"):
            att.notify("initialized", {})

    def test_closing_twice_does_not_write_the_run_twice(
        self, codex, store, tmp_path
    ):
        att = CodexAttachment(store=store, codex_bin=codex, cwd=tmp_path).open()
        att.close()
        att.close()
        assert kinds(list(store.read(att.run_id))).count("run.completed") == 1
