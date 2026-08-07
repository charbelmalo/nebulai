"""M2 exit tests: the hooks, the spool, the collector, and the installer.

These hold down the four things that make observed mode either trustworthy or
actively harmful:

* **The shim never breaks the session it observes.** It exits 0 on every path —
  no spool directory, unreadable stdin, a payload larger than the cap, a
  filesystem that refuses the write. An observer that can fail a tool call is
  worse than no observer.
* **Pairing is real.** A `PreToolUse` and its `PostToolUse` arrive as two
  separate processes, so anything derived from the process identity pairs
  nothing. Spans must close, and concurrent same-name calls must produce as
  many spans as there were calls.
* **What we cannot see is stated.** A hook stream has no token usage at all; the
  run must say so, and the comparability gate must refuse to difference it
  against a driven run rather than reporting an implicit zero.
* **The installer merges.** The machine this was written on has a live
  `rtk hook claude` entry in `~/.claude/settings.json`. Installing must keep it,
  and uninstalling must keep it too — including the entries a user adds *after*
  installing, which restoring a backup wholesale would destroy.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from nebulai.seer import install as inst
from nebulai.seer.adapters.observed import HOOK_ADAPTERS, ClaudeHookAdapter, hook_events
from nebulai.seer.cli import add_parser
from nebulai.seer.collector import SpoolCollector, import_spool
from nebulai.seer.compare import compare
from nebulai.seer.contract import CaptureMode, EventType, Fidelity, Outcome
from nebulai.seer.reducer import reduce_run
from nebulai.seer.spool import (
    MAX_PAYLOAD_BYTES,
    SpoolLine,
    SpoolReader,
    install_shim,
    pick_clock,
    read_manifest,
    remove_shim,
    spool_dir,
)
from nebulai.seer.store import EventStore


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "seer"


@pytest.fixture
def shim(root: Path) -> Path:
    path, _clock = install_shim(root)
    return path


def fire(shim: Path, agent: str, event: str, payload: dict | None = None, **kw) -> None:
    """Run the shim exactly as an agent would: argv + JSON on stdin."""
    subprocess.run(
        [str(shim), agent, event],
        input=json.dumps(payload) if payload is not None else "",
        text=True,
        check=True,
        timeout=20,
        **kw,
    )


# ── the shim ─────────────────────────────────────────────────────────────────


class TestShim:
    def test_writes_one_line_per_firing(self, shim: Path, root: Path) -> None:
        fire(shim, "claude", "SessionStart", {"session_id": "s1"})
        fire(shim, "claude", "Stop", {"session_id": "s1"})
        lines = (spool_dir(root) / "claude.jsonl").read_text().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["agent"] == "claude"
        assert first["event"] == "SessionStart"
        assert first["payload"] == {"session_id": "s1"}

    def test_each_agent_gets_its_own_file(self, shim: Path, root: Path) -> None:
        fire(shim, "claude", "SessionStart", {})
        fire(shim, "codex", "on_session_start", {})
        assert {p.name for p in spool_dir(root).glob("*.jsonl")} == {
            "claude.jsonl",
            "codex.jsonl",
        }

    def test_exits_zero_with_no_spool_directory(self, shim: Path, root: Path) -> None:
        # Deleting the spool is the documented way to switch capture off without
        # editing any agent's config, so the shim keeps being called with nowhere
        # to write. That path has to be silent, not fatal.
        shutil.rmtree(spool_dir(root))
        assert not spool_dir(root).exists()
        r = subprocess.run([str(shim), "claude", "Stop"], input="{}", text=True,
                           capture_output=True, timeout=20)
        assert r.returncode == 0
        assert r.stdout == "" and r.stderr == ""

    def test_exits_zero_when_the_spool_is_unwritable(self, shim: Path, root: Path) -> None:
        d = spool_dir(root)
        os.chmod(d, 0o500)
        try:
            r = subprocess.run([str(shim), "claude", "Stop"], input="{}", text=True,
                               capture_output=True, timeout=20)
            assert r.returncode == 0, r.stderr
            assert r.stderr == ""
        finally:
            os.chmod(d, 0o700)

    def test_exits_zero_with_no_stdin_at_all(self, shim: Path, root: Path) -> None:
        r = subprocess.run([str(shim), "claude", "Notification"], stdin=subprocess.DEVNULL,
                           capture_output=True, timeout=20)
        assert r.returncode == 0
        line = json.loads((spool_dir(root) / "claude.jsonl").read_text().splitlines()[0])
        assert line["payload"] is None

    def test_drops_an_oversized_payload_but_keeps_its_size(self, shim: Path, root: Path) -> None:
        big = {"tool_response": "x" * (MAX_PAYLOAD_BYTES * 2)}
        fire(shim, "claude", "PostToolUse", big)
        line = json.loads((spool_dir(root) / "claude.jsonl").read_text().splitlines()[0])
        # the size survives; the body does not. "we chose not to carry this" is
        # a different fact from "the agent said nothing".
        assert line["payload"]["seer_oversized"] > MAX_PAYLOAD_BYTES
        assert "xxxx" not in json.dumps(line["payload"])

    def test_a_line_is_valid_json_even_with_quotes_and_newlines_inside(
        self, shim: Path, root: Path
    ) -> None:
        payload = {"prompt": 'he said "run it"\nthen \\ left', "tool_input": {"a": ["b", 1]}}
        fire(shim, "claude", "UserPromptSubmit", payload)
        line = json.loads((spool_dir(root) / "claude.jsonl").read_text().splitlines()[0])
        assert line["payload"] == payload

    def test_the_pid_it_reports_is_its_own_and_says_so(self, shim: Path, root: Path) -> None:
        fire(shim, "claude", "Stop", {})
        fire(shim, "claude", "Stop", {})
        a, b = (json.loads(x) for x in
                (spool_dir(root) / "claude.jsonl").read_text().splitlines())
        # Two firings, two processes, two pids. This is exactly why pairing a
        # Pre with a Post on the pid cannot work, and why nothing does.
        assert a["pid"] != b["pid"]

    def test_the_shim_costs_about_what_an_empty_script_costs(
        self, shim: Path, root: Path
    ) -> None:
        """The criterion that replaced the plan's unreachable "< 5 ms p95".

        Process spawn is most of the cost of any hook and none of it is ours, so
        what is measured is the *difference*: the shim against an empty script
        with the same interpreter. The bound is loose because this runs on a
        machine that is also doing other things; it is a guard against someone
        adding a `$(…)` or a `curl`, not a benchmark.
        """
        empty = root / "empty.sh"
        empty.write_text(shim.read_text().splitlines()[0] + "\nexit 0\n")
        empty.chmod(0o755)
        payload = json.dumps({"session_id": "s", "tool_name": "Bash",
                              "tool_input": {"command": "pytest -q"}})

        def median_ms(cmd: list[str]) -> float:
            for _ in range(3):  # warm the page cache
                subprocess.run(cmd, input=payload, text=True, capture_output=True)
            times = []
            for _ in range(15):
                t0 = time.perf_counter()
                subprocess.run(cmd, input=payload, text=True, capture_output=True)
                times.append((time.perf_counter() - t0) * 1000)
            return sorted(times)[len(times) // 2]

        floor = median_ms([str(empty)])
        ours = median_ms([str(shim), "claude", "PreToolUse"])
        assert ours - floor < 3.0, (
            f"shim adds {ours - floor:.2f} ms over a {floor:.2f} ms empty script; "
            "something in it is forking"
        )

    def test_the_manifest_records_the_clock_that_was_verified(self, root: Path) -> None:
        _path, clock = install_shim(root)
        m = read_manifest(root)
        assert m["clock"] == clock.name
        assert m["resolution_s"] == clock.resolution_s
        # picked by running it, not by sniffing a version
        assert pick_clock().name == clock.name

    def test_installing_twice_does_not_duplicate_or_break(self, root: Path) -> None:
        p1, _ = install_shim(root)
        fire(p1, "claude", "Stop", {})
        p2, _ = install_shim(root)
        fire(p2, "claude", "Stop", {})
        assert p1 == p2
        assert len((spool_dir(root) / "claude.jsonl").read_text().splitlines()) == 2


# ── the reader ───────────────────────────────────────────────────────────────


T0 = 1785891492.0  # a plausible epoch; the reader rejects anything below 1e9


class TestSpoolReader:
    def test_never_yields_a_line_that_is_still_being_written(self, root: Path) -> None:
        install_shim(root)
        f = spool_dir(root) / "claude.jsonl"
        f.write_text(
            '{"v":1,"agent":"claude","event":"Stop","t":%.1f,"pid":1,"payload":{}}\n' % T0
        )
        r = SpoolReader(root, from_start=True)
        assert len(list(r.poll())) == 1

        # a half-written append: the shim's printf is one write, but a reader
        # can still catch a partial line on a slow filesystem
        with f.open("a") as fh:
            fh.write('{"v":1,"agent":"claude","even')
        assert list(r.poll()) == []

        with f.open("a") as fh:
            fh.write('t":"Stop","t":%.1f,"pid":1,"payload":{}}\n' % (T0 + 1))
        out = list(r.poll())
        assert len(out) == 1 and out[0].shim_ts == T0 + 1

    def test_a_clock_that_returned_zero_is_not_reported_as_a_time(self, root: Path) -> None:
        install_shim(root)
        (spool_dir(root) / "claude.jsonl").write_text(
            '{"v":1,"agent":"claude","event":"Stop","t":0,"pid":1,"payload":{}}\n'
        )
        line = SpoolReader(root, from_start=True).poll()[0]
        assert line.shim_ts is None
        # …and the line still has a usable time, from when we read it
        assert line.ts > 1e9

    def test_counts_the_lines_it_could_not_use(self, root: Path) -> None:
        install_shim(root)
        f = spool_dir(root) / "claude.jsonl"
        f.write_text(
            "not json at all\n"
            '{"v":1,"not":"ours"}\n'
            '{"v":1,"agent":"claude","event":"Stop","t":%.1f,"payload":{}}\n' % T0
        )
        r = SpoolReader(root, from_start=True)
        out = list(r.poll())
        assert len(out) == 1
        # a broken line and a foreign line are different problems: one means
        # writes are interleaving, the other means something else writes here
        assert r.stats.torn == 1
        assert r.stats.unparsable == 1

    def test_a_live_reader_starts_at_the_end(self, root: Path) -> None:
        path, _ = install_shim(root)
        fire(path, "claude", "Stop", {})
        r = SpoolReader(root)  # from_start=False
        assert list(r.poll()) == []
        fire(path, "claude", "Stop", {})
        assert len(list(r.poll())) == 1


# ── the adapter ──────────────────────────────────────────────────────────────


def line(event: str, payload: dict, ts: float = 0.0, agent: str = "claude") -> SpoolLine:
    return SpoolLine(agent=agent, event=event, payload=payload, shim_ts=ts, seen_at=ts, pid=1)


class TestHookAdapter:
    def make(self, **kw) -> ClaudeHookAdapter:
        kw.setdefault("clock_resolution_s", 1e-6)
        return ClaudeHookAdapter(
            run_id="run_t", session_id="s1", agent_version="2.0.0",
            capture_mode=CaptureMode.OBSERVED, **kw,
        )

    def test_a_pre_and_a_post_close_one_span(self) -> None:
        a = self.make()
        a.feed_hook(line("SessionStart", {"session_id": "s1"}, 0))
        a.feed_hook(line("PreToolUse", {"tool_name": "Bash",
                                        "tool_input": {"command": "pytest -q"}}, 1))
        events = a.feed_hook(line("PostToolUse", {"tool_name": "Bash",
                                                  "tool_response": {"ok": 1}}, 4))
        done = [e for e in events if e.event_type is EventType.TOOL_COMPLETED]
        assert len(done) == 1
        v = reduce_run("run_t", _replay(a, [
            ("SessionStart", {"session_id": "s1"}, 0),
            ("PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "pytest -q"}}, 1),
            ("PostToolUse", {"tool_name": "Bash", "tool_response": {"ok": 1}}, 4),
        ]))
        assert len(v.spans) == 1
        assert v.spans[0].duration_s == pytest.approx(3.0)

    def test_concurrent_same_name_calls_do_not_collapse(self) -> None:
        events = _replay(self.make(), [
            ("PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "a"}}, 1),
            ("PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "b"}}, 1),
            ("PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "c"}}, 1),
            ("PostToolUse", {"tool_name": "Bash", "tool_response": {}}, 2),
            ("PostToolUse", {"tool_name": "Bash", "tool_response": {}}, 3),
            ("PostToolUse", {"tool_name": "Bash", "tool_response": {}}, 4),
        ])
        v = reduce_run("run_t", events)
        assert len(v.spans) == 3, "three calls must not fold into one"

    def test_pairing_by_name_admits_it_is_a_guess(self) -> None:
        events = _replay(self.make(), [
            ("PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "a"}}, 1),
            ("PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "b"}}, 1),
            ("PostToolUse", {"tool_name": "Bash", "tool_response": {}}, 2),
        ])
        done = [e for e in events if e.event_type is EventType.TOOL_COMPLETED]
        assert done[0].source.fidelity is Fidelity.HEURISTIC
        assert done[0].payload["paired_by"] == "name-order"

    def test_a_call_id_makes_the_pairing_exact(self) -> None:
        events = _replay(self.make(), [
            ("PreToolUse", {"tool_name": "Bash", "tool_use_id": "t1",
                            "tool_input": {"command": "a"}}, 1),
            ("PreToolUse", {"tool_name": "Bash", "tool_use_id": "t2",
                            "tool_input": {"command": "b"}}, 1),
            ("PostToolUse", {"tool_name": "Bash", "tool_use_id": "t2", "tool_response": {}}, 5),
            ("PostToolUse", {"tool_name": "Bash", "tool_use_id": "t1", "tool_response": {}}, 9),
        ])
        done = [e for e in events if e.event_type is EventType.TOOL_COMPLETED]
        assert all(e.source.fidelity is not Fidelity.HEURISTIC for e in done)
        v = reduce_run("run_t", events)
        assert sorted(round(s.duration_s, 1) for s in v.spans) == [4.0, 8.0]

    def test_a_coarse_clock_marks_durations_estimated(self) -> None:
        a = self.make(clock_resolution_s=1.0)
        assert a.timing_fidelity is Fidelity.ESTIMATED
        b = self.make(clock_resolution_s=1e-6)
        assert b.timing_fidelity is Fidelity.DETERMINISTIC

    def test_joining_mid_session_says_so(self) -> None:
        a = self.make()
        events = a.feed_hook(line("PreToolUse", {"tool_name": "Read",
                                                 "tool_input": {"file_path": "/a"}}, 7))
        start = [e for e in events if e.event_type is EventType.SESSION_STARTED][0]
        assert start.payload["joined_midstream"] is True
        assert start.source.fidelity is Fidelity.ESTIMATED

    def test_the_capture_gaps_are_declared_on_the_session(self) -> None:
        a = self.make()
        events = a.feed_hook(line("SessionStart", {"session_id": "s1"}, 0))
        gaps = events[0].payload["capture_gaps"]
        assert any("token" in g for g in gaps)

    def test_a_failed_tool_is_a_failure_not_a_completion(self) -> None:
        events = _replay(self.make(), [
            ("PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "false"}}, 1),
            ("PostToolUse", {"tool_name": "Bash", "tool_response": {"is_error": True}}, 2),
        ])
        assert any(e.event_type is EventType.TOOL_FAILED for e in events)
        v = reduce_run("run_t", events)
        assert v.spans[0].failed is True

    def test_an_unknown_hook_name_is_recorded_not_dropped(self) -> None:
        a = self.make()
        events = a.feed_hook(line("SomeFutureHook", {"session_id": "s1"}, 1))
        assert any(e.event_type is EventType.ADAPTER_WARNING for e in events)

    def test_closing_never_claims_the_session_completed(self) -> None:
        a = self.make()
        a.feed_hook(line("SessionStart", {"session_id": "s1"}, 0))
        events = a.close(5.0, reason="watcher stopped")
        assert events[-1].event_type is EventType.SESSION_INTERRUPTED
        v = reduce_run("run_t", events)
        assert v.outcome is not Outcome.AGENT_CLAIMED_COMPLETE

    @pytest.mark.parametrize("agent", sorted(HOOK_ADAPTERS))
    def test_every_routed_event_has_a_method_that_exists(self, agent: str) -> None:
        cls = HOOK_ADAPTERS[agent]
        for event, method in cls.ROUTES.items():
            assert hasattr(cls, method), f"{agent}.{event} routes to a missing {method}"
        assert hook_events(agent) == tuple(cls.ROUTES)

    def test_hermes_registers_no_transform_hook(self) -> None:
        # A transform hook can rewrite the agent's own data. An observer that
        # can change what it observes is not an observer.
        assert not any(e.startswith("transform") for e in hook_events("hermes"))


def _replay(adapter, script: list[tuple[str, dict, float]]) -> list:
    out = []
    for event, payload, ts in script:
        out.extend(adapter.feed_hook(line(event, payload, ts)))
    return out


# ── the collector ────────────────────────────────────────────────────────────


class TestCollector:
    def test_two_sessions_never_fold_into_one_run(self, root: Path, shim: Path) -> None:
        store = EventStore(root / "seer.db")
        c = SpoolCollector(store, root, from_start=True)
        fire(shim, "claude", "SessionStart", {"session_id": "A", "cwd": "/a"})
        fire(shim, "claude", "SessionStart", {"session_id": "B", "cwd": "/b"})
        fire(shim, "claude", "Stop", {"session_id": "A"})
        c.poll()
        assert c.stats.runs_opened == 2
        assert len(store.list_runs(limit=10)) == 2

    def test_a_missing_session_id_does_not_open_a_run_per_hook(
        self, root: Path, shim: Path
    ) -> None:
        store = EventStore(root / "seer.db")
        c = SpoolCollector(store, root, from_start=True)
        for _ in range(5):
            fire(shim, "claude", "Notification", {"cwd": "/w"})
        c.poll()
        assert c.stats.lines == 5
        assert c.stats.runs_opened == 1, "the shim's pid is per-firing, not per-session"

    def test_a_session_still_sending_hooks_is_running_not_starting(
        self, root: Path, shim: Path
    ) -> None:
        store = EventStore(root / "seer.db")
        c = SpoolCollector(store, root, from_start=True)
        fire(shim, "claude", "SessionStart", {"session_id": "A"})
        c.poll()
        # "starting" is for a run we launched that has not spoken yet; an
        # observed run announced itself by speaking
        assert store.list_runs(limit=1)[0].state == "running"

    def test_an_unknown_agent_is_counted_not_crashed_on(self, root: Path, shim: Path) -> None:
        store = EventStore(root / "seer.db")
        c = SpoolCollector(store, root, from_start=True)
        fire(shim, "gemini", "SessionStart", {"session_id": "z"})
        c.poll()
        assert c.stats.unknown_agents == {"gemini": 1}
        assert c.stats.runs_opened == 0

    def test_a_silent_run_is_interrupted_and_ends_when_it_went_quiet(
        self, root: Path, shim: Path
    ) -> None:
        store = EventStore(root / "seer.db")
        c = SpoolCollector(store, root, from_start=True, idle_timeout_s=0.0)
        fire(shim, "claude", "SessionStart", {"session_id": "A"})
        c.poll()
        run = store.list_runs(limit=1)[0]
        assert run.state == "interrupted"
        v = reduce_run(run.run_id, store.read(run.run_id))
        assert v.outcome is not Outcome.AGENT_CLAIMED_COMPLETE
        # closed at the last hook we saw, not at the moment we gave up waiting
        assert v.ended_at == pytest.approx(v.last_event_at, abs=0.001)

    def test_the_whole_path_produces_a_run_you_can_read(self, root: Path, shim: Path) -> None:
        store = EventStore(root / "seer.db")
        c = SpoolCollector(store, root, from_start=True)
        sid = "live-1"
        fire(shim, "claude", "SessionStart", {"session_id": sid, "cwd": str(root)})
        fire(shim, "claude", "UserPromptSubmit", {"session_id": sid, "prompt": "fix it"})
        fire(shim, "claude", "PreToolUse", {"session_id": sid, "tool_name": "Edit",
                                            "tool_input": {"file_path": "/x/a.py"}})
        fire(shim, "claude", "PostToolUse", {"session_id": sid, "tool_name": "Edit",
                                             "tool_response": {"ok": True}})
        fire(shim, "claude", "PreToolUse", {"session_id": sid, "tool_name": "Bash",
                                            "tool_input": {"command": "pytest -q"}})
        fire(shim, "claude", "PostToolUse", {"session_id": sid, "tool_name": "Bash",
                                             "tool_response": {"stdout": "ok"}})
        fire(shim, "claude", "Stop", {"session_id": sid})
        fire(shim, "claude", "SessionEnd", {"session_id": sid, "reason": "clear"})
        c.poll()

        run = store.list_runs(limit=1)[0]
        v = reduce_run(run.run_id, store.read(run.run_id))
        assert v.quality.capture_mode == "observed"
        assert v.action_counts.get("edit") == 1
        assert v.action_counts.get("verify") == 1  # pytest is a verification
        assert v.files_changed == ["/x/a.py"]
        assert len(v.spans) == 2 and all(s.ended_at for s in v.spans)

    def test_an_observed_run_reports_no_tokens_rather_than_zero(
        self, root: Path, shim: Path
    ) -> None:
        store = EventStore(root / "seer.db")
        c = SpoolCollector(store, root, from_start=True)
        fire(shim, "claude", "SessionStart", {"session_id": "A"})
        fire(shim, "claude", "SessionEnd", {"session_id": "A", "reason": "clear"})
        c.poll()
        run = store.list_runs(limit=1)[0]
        v = reduce_run(run.run_id, store.read(run.run_id))
        assert v.usage == {} or all(m.absent for m in v.usage.values())
        assert any("token" in g for g in v.quality.capture_gaps)

    def test_the_gate_refuses_tokens_between_observed_and_driven(
        self, root: Path, shim: Path
    ) -> None:
        store = EventStore(root / "seer.db")
        c = SpoolCollector(store, root, from_start=True)
        fire(shim, "claude", "SessionStart", {"session_id": "A"})
        fire(shim, "claude", "SessionEnd", {"session_id": "A", "reason": "clear"})
        c.poll()
        observed = reduce_run(*_run_and_events(store))

        driven = reduce_run("run_driven", _driven_events())
        result = compare([observed, driven])
        refused = {r.metric for r in result.refused}
        assert any(m.startswith("tokens") for m in refused), (
            "an observed run has no usage at all; differencing it against a "
            "driven one would report the gap as a saving"
        )

    def test_the_background_thread_stops_cleanly(self, root: Path, shim: Path) -> None:
        store = EventStore(root / "seer.db")
        c = SpoolCollector(store, root, from_start=True)
        c.start()
        fire(shim, "claude", "SessionStart", {"session_id": "A"})
        deadline = time.time() + 5
        while c.stats.events == 0 and time.time() < deadline:
            time.sleep(0.05)
        c.stop()
        assert c.stats.events > 0
        assert not any(t.name == "seer-spool" and t.is_alive()
                       for t in threading.enumerate())

    def test_import_spool_recovers_a_session_nothing_was_watching(
        self, root: Path, shim: Path
    ) -> None:
        for e, p in [
            ("SessionStart", {"session_id": "gone", "cwd": "/w"}),
            ("PreToolUse", {"session_id": "gone", "tool_name": "Read",
                            "tool_input": {"file_path": "/a"}}),
            ("PostToolUse", {"session_id": "gone", "tool_name": "Read",
                             "tool_response": "x"}),
        ]:
            fire(shim, "claude", e, p)
        store = EventStore(root / "seer.db")
        status = import_spool(store, root)
        assert status["runs_opened"] == 1
        run = store.list_runs(limit=1)[0]
        # nothing said it ended, so it did not end well
        assert run.state == "interrupted"
        assert reduce_run(run.run_id, store.read(run.run_id)).action_counts.get("inspect") == 1


def _run_and_events(store: EventStore):
    run = store.list_runs(limit=1)[0]
    return run.run_id, list(store.read(run.run_id))


FIXTURES = Path(__file__).parent / "fixtures" / "seer"


def _driven_events() -> list:
    """A real driven capture — the one the M1 tests use — so the refusal is
    demonstrated against actual agent output rather than a shape I invented."""
    from nebulai.seer.adapters import ClaudeStreamAdapter

    a = ClaudeStreamAdapter(run_id="run_driven", session_id="d")
    events = []
    for raw in (FIXTURES / "claude-tools.jsonl").read_text().splitlines():
        events.extend(a.feed(raw))
    events.extend(a.finish())
    return events


# ── the installer ────────────────────────────────────────────────────────────


LIVE_CLAUDE = Path.home() / ".claude" / "settings.json"


def _without_our_entries(cfg: dict) -> dict:
    """The live config minus anything we installed into it.

    These tests install and then uninstall, and uninstall removes every tagged
    entry — so a live config that *already* has our hooks (the author dogfoods
    the installer) makes the round trip end with fewer entries than it started
    with, failing a test about clobbering that clobbered nothing. Stripping our
    own tag is what keeps the fixture "the messy real config" rather than "the
    real config plus whatever this machine last ran".
    """
    hooks = cfg.get("hooks")
    if not isinstance(hooks, dict):
        return cfg
    cleaned: dict = {}
    for event, groups in hooks.items():
        kept_groups = []
        for g in groups:
            kept = [h for h in g.get("hooks", []) if h.get("_source") != inst.TAG]
            if kept:
                kept_groups.append({**g, "hooks": kept})
        if kept_groups:
            cleaned[event] = kept_groups
    return {**cfg, "hooks": cleaned}


@pytest.fixture
def settings(tmp_path: Path) -> Path:
    """A copy of the real settings.json when there is one — the merge has to
    survive the config it will actually meet, not a tidy fixture — with our own
    hooks stripped, so the fixture is a config we have not installed into."""
    p = tmp_path / "settings.json"
    if LIVE_CLAUDE.exists():
        p.write_text(json.dumps(_without_our_entries(json.loads(LIVE_CLAUDE.read_text()))))
    else:
        p.write_text(json.dumps({
            "hooks": {"PreToolUse": [{"matcher": "Bash",
                                      "hooks": [{"type": "command", "command": "rtk hook claude"}]}]},
            "model": "opus",
        }))
    return p


class TestInstaller:
    def test_keeps_every_hook_it_did_not_write(self, root: Path, settings: Path) -> None:
        before = json.loads(settings.read_text())
        inst.install("claude", root, config=settings)
        after = json.loads(settings.read_text())

        for event, groups in (before.get("hooks") or {}).items():
            for g in groups:
                assert g in after["hooks"][event], f"lost a pre-existing {event} hook"
        for key, value in before.items():
            if key != "hooks":
                assert after[key] == value, f"clobbered an unrelated key: {key}"

    def test_backs_up_the_original_bytes_before_writing(self, root: Path, settings: Path) -> None:
        original = settings.read_bytes()
        _plan, backup = inst.install("claude", root, config=settings)
        assert backup is not None
        assert backup.read_bytes() == original

    def test_installing_twice_adds_one_set_of_hooks(self, root: Path, settings: Path) -> None:
        inst.install("claude", root, config=settings)
        once = json.loads(settings.read_text())
        inst.install("claude", root, config=settings)
        twice = json.loads(settings.read_text())
        assert once == twice

    def test_uninstall_removes_only_ours(self, root: Path, settings: Path) -> None:
        before = json.loads(settings.read_text())
        inst.install("claude", root, config=settings)

        # the user adds their own hook *after* installing; restoring the backup
        # would delete it, which is why uninstall removes by tag instead
        cfg = json.loads(settings.read_text())
        cfg["hooks"].setdefault("Stop", []).append(
            {"hooks": [{"type": "command", "command": "say done"}]}
        )
        settings.write_text(json.dumps(cfg))

        inst.uninstall("claude", root, config=settings)
        after = json.loads(settings.read_text())
        assert not _tagged_entries(after)
        assert {"type": "command", "command": "say done"} in [
            h for g in after["hooks"]["Stop"] for h in g["hooks"]
        ]
        for event, groups in (before.get("hooks") or {}).items():
            for g in groups:
                assert g in after["hooks"][event]

    def test_a_round_trip_leaves_the_config_semantically_identical(
        self, root: Path, settings: Path
    ) -> None:
        before = json.loads(settings.read_text())
        inst.install("claude", root, config=settings)
        inst.uninstall("claude", root, config=settings)
        assert json.loads(settings.read_text()) == before

    def test_the_plan_lists_what_it_will_not_do_as_manual(self, root: Path, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text("[hooks]\nsomething = 1\n")
        p = inst.plan("codex", root, config=cfg)
        assert p.supported is False
        assert p.changes == [], "a plan must not promise a change it will refuse to make"
        assert p.manual

    def test_hermes_never_writes_the_users_consent_for_them(
        self, root: Path, tmp_path: Path
    ) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("model: sonnet\n")
        p = inst.plan("hermes", root, config=cfg)
        assert any("allowlist" in m for m in p.manual)
        assert not inst.HERMES_ALLOWLIST.exists() or "seer" not in \
            inst.HERMES_ALLOWLIST.read_text()
        inst.install("hermes", root, config=cfg)
        assert "hooks:" in cfg.read_text()

    def test_refuses_to_touch_a_config_it_cannot_parse(self, root: Path, tmp_path: Path) -> None:
        bad = tmp_path / "settings.json"
        bad.write_text("{ this is not json")
        with pytest.raises(RuntimeError, match="not valid JSON"):
            inst.install("claude", root, config=bad)
        assert bad.read_text() == "{ this is not json"

    def test_install_writes_the_shim_before_the_config_points_at_it(
        self, root: Path, settings: Path
    ) -> None:
        inst.install("claude", root, config=settings)
        commands = [h["command"] for h in _tagged_entries(json.loads(settings.read_text()))]
        assert commands
        for c in commands:
            path = Path(c.split('"')[1])
            assert path.exists(), "a config pointing at a missing shim fails every hook"
            assert os.access(path, os.X_OK)

    def test_every_hook_event_the_adapter_routes_gets_registered(
        self, root: Path, settings: Path
    ) -> None:
        inst.install("claude", root, config=settings)
        cfg = json.loads(settings.read_text())
        registered = {
            e for e, groups in cfg["hooks"].items()
            if any(h.get("_source") == inst.TAG for g in groups for h in g["hooks"])
        }
        assert registered == set(hook_events("claude"))

    def test_status_reads_the_live_config_not_a_cache(self, root: Path, settings: Path) -> None:
        st = inst.status(root)
        assert set(st) >= {"claude", "codex", "hermes", "shim"}
        assert st["claude"]["config"] == str(inst.CLAUDE_SETTINGS)


class TestRestore:
    """What "we can put your config back" actually means.

    M5 asked for a byte-exact restore. Uninstall deliberately does not do that,
    and the difference matters: restoring the backup wholesale would delete
    every hook the user added *after* installing, so uninstall removes our
    entries by tag and leaves the file otherwise alone. That makes three
    separate promises, and each is worth a test of its own:

    * the **backup** is byte-exact — it is the only true undo, and the one to
      reach for if we ever corrupt a config;
    * an **uninstall** round trip returns the text-based configs byte for byte,
      comments and all, for any file that ends in a newline;
    * the **JSON** config comes back semantically identical, not byte
      identical, because we rewrite it through a parser and the user's own
      formatting is not ours to preserve.
    """

    def test_the_backup_is_the_byte_exact_undo(self, root: Path, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        original = "model = 'gpt'\n# a comment we must not eat\n\n[profiles.work]\nx = 1\n"
        cfg.write_text(original)

        _plan, backup = inst.install("codex", root, config=cfg)
        assert backup is not None
        assert cfg.read_text() != original, "nothing was installed"
        assert backup.read_bytes() == original.encode()

        shutil.copy2(backup, cfg)
        assert cfg.read_text() == original

    def test_a_second_install_never_backs_up_an_already_modified_copy(
        self, root: Path, tmp_path: Path
    ) -> None:
        """Numbered backups, not one overwritten file. The second install's
        backup is of an installed config — which is fine, as long as it does
        not land on top of the only copy of the original."""
        cfg = tmp_path / "config.toml"
        original = "model = 'gpt'\n"
        cfg.write_text(original)

        _p, first = inst.install("codex", root, config=cfg)
        cfg.write_text(cfg.read_text() + "extra = true\n")
        _p, second = inst.install("codex", root, config=cfg)

        assert first is not None and second is not None and first != second
        assert first.read_text() == original
        assert len(inst.existing_backups(cfg)) == 2

    @pytest.mark.parametrize(
        "agent,name,original",
        [
            (
                "codex",
                "config.toml",
                # comments, blank lines and a table our parser-free edit must
                # not reorder or reformat
                "model = 'gpt-5'\n\n# why this profile exists\n[profiles.work]\n"
                "approval_policy = 'never'\n",
            ),
            (
                "hermes",
                "config.yaml",
                # YAML with comments is the case a round-trip through a YAML
                # library would quietly destroy
                "model: sonnet\n\n# do not lose me\nagents:\n  - name: a\n    tools: [read]\n",
            ),
        ],
    )
    def test_a_round_trip_returns_a_text_config_byte_for_byte(
        self, root: Path, tmp_path: Path, agent: str, name: str, original: str
    ) -> None:
        cfg = tmp_path / name
        cfg.write_text(original)

        inst.install(agent, root, config=cfg)
        assert inst.TAG in cfg.read_text()
        inst.uninstall(agent, root, config=cfg)

        assert cfg.read_text() == original

    def test_repeated_cycles_do_not_grow_the_file(self, root: Path, tmp_path: Path) -> None:
        """The blank line install writes as a separator is part of what
        install wrote, so uninstall takes it back. Left behind, it accumulates:
        one blank line per cycle, in a file the user never edited."""
        cfg = tmp_path / "config.toml"
        original = "model = 'gpt'\n"
        cfg.write_text(original)

        for _ in range(3):
            inst.install("codex", root, config=cfg)
            inst.uninstall("codex", root, config=cfg)

        assert cfg.read_text() == original

    def test_a_config_with_no_trailing_newline_gains_one_and_nothing_else(
        self, root: Path, tmp_path: Path
    ) -> None:
        """The one case where the round trip is not byte-exact, stated rather
        than hidden: appending to a file whose last line is unterminated has to
        terminate it, and nothing in the file records that we did."""
        cfg = tmp_path / "config.toml"
        cfg.write_text("model = 'gpt'")

        inst.install("codex", root, config=cfg)
        inst.uninstall("codex", root, config=cfg)

        assert cfg.read_text() == "model = 'gpt'\n"

    def test_uninstall_keeps_lines_a_user_added_after_installing(
        self, root: Path, tmp_path: Path
    ) -> None:
        """The reason uninstall is not "restore the backup". The user edits
        their config after installing; a wholesale restore would silently undo
        that edit, and they would find out from the agent behaving oddly."""
        cfg = tmp_path / "config.toml"
        cfg.write_text("model = 'gpt'\n")
        inst.install("codex", root, config=cfg)
        cfg.write_text(cfg.read_text() + "\n[profiles.new]\nadded = 'after'\n")

        inst.uninstall("codex", root, config=cfg)

        text = cfg.read_text()
        assert "[profiles.new]" in text and "added = 'after'" in text
        assert inst.TAG not in text

    def test_uninstalling_what_was_never_installed_changes_nothing(
        self, root: Path, tmp_path: Path
    ) -> None:
        cfg = tmp_path / "config.toml"
        original = "model = 'gpt'\n\n[hooks]\nsomeone_elses = ['thing']\n"
        cfg.write_text(original)

        plan = inst.uninstall("codex", root, config=cfg)

        assert plan.changes == []
        assert cfg.read_text() == original


class TestInstallerArgumentParsing:
    """`install` with no agent named means every agent, and must parse.

    argparse runs a `nargs="*"` positional's *default* through its own choices
    check, so a `default=[]` made the bare `nebulai seer install --apply` die
    with `invalid choice: '[]'` — the one invocation the help text implies is
    normal. The command bodies always read "no agents" as "all of them"; only
    the parser stood in the way.
    """

    def _parse(self, argv: list[str]) -> argparse.Namespace:
        p = argparse.ArgumentParser(prog="nebulai")
        add_parser(p.add_subparsers(dest="cmd", required=True))
        return p.parse_args(argv)

    @pytest.mark.parametrize("cmd", ["install", "uninstall"])
    def test_no_agent_named_parses_and_means_all_of_them(self, cmd: str) -> None:
        args = self._parse(["seer", cmd])

        assert not args.agents
        assert (args.agents or list(inst.CONFIGS)) == list(inst.CONFIGS)

    def test_no_agent_named_still_parses_with_apply(self) -> None:
        args = self._parse(["seer", "install", "--apply"])

        assert not args.agents
        assert args.dry_run is False

    @pytest.mark.parametrize("cmd", ["install", "uninstall"])
    def test_named_agents_are_kept_verbatim(self, cmd: str) -> None:
        assert self._parse(["seer", cmd, "claude"]).agents == ["claude"]
        assert self._parse(["seer", cmd, "codex", "hermes"]).agents == ["codex", "hermes"]

    @pytest.mark.parametrize("cmd", ["install", "uninstall"])
    def test_an_unknown_agent_is_still_rejected(self, cmd: str) -> None:
        with pytest.raises(SystemExit):
            self._parse(["seer", cmd, "not-an-agent"])


def _tagged_entries(cfg: dict) -> list[dict]:
    return [
        h
        for groups in (cfg.get("hooks") or {}).values()
        for g in groups
        for h in g.get("hooks", [])
        if h.get("_source") == inst.TAG
    ]
