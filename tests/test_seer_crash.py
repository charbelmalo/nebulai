"""M5 exit tests: what survives a crash, and what the page does about it.

Everything else assumes the capture process lives long enough to write an
ending. These are the tests for when it does not — a `kill -9`, a closed lid, a
laptop that ran out of battery mid-run.

Three properties:

* **A run whose capture died is never reported as finished.** Not completed,
  not failed — `interrupted`, which is the only thing we actually know.
* **The repair is written into the log, not the index.** The index is a cache
  and `reindex()` rebuilds it from the log, so a recovery that only wrote to
  SQLite would undo itself the first time anyone rebuilt.
* **A live run in another process is not touched.** The sweep runs from every
  CLI verb, and a sweep that could not tell "dead" from "someone else's" would
  cheerfully close the run you are watching.

The SSE half is about the same fact from the viewer's side: the stream drops,
the page says so rather than going quiet, and the reconnect does not double
what it already has.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from nebulai.seer import runner as runner_mod
from nebulai.seer.adapters import CodexExecAdapter
from nebulai.seer.contract import EventType, Outcome, SessionState
from nebulai.seer.recover import ADAPTER, recover_orphans
from nebulai.seer.reducer import reduce_run
from nebulai.seer.runner import Runner
from nebulai.seer.server import SeerState, _Handler
from nebulai.seer.store import EventStore

FIXTURES = Path(__file__).parent / "fixtures" / "seer"

FAKE_CODEX = r"""
import json, sys
def p(o): print(json.dumps(o), flush=True)
p({"type": "thread.started", "thread_id": "th_fake"})
p({"type": "turn.started"})
p({"type": "item.completed", "item": {"id": "i2", "type": "agent_message",
   "text": "done"}})
p({"type": "turn.completed", "usage": {"input_tokens": 100,
   "cached_input_tokens": 10, "output_tokens": 20, "reasoning_output_tokens": 5}})
"""


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "seer")
    yield s
    s.close()


@pytest.fixture
def fake_agent(tmp_path, monkeypatch):
    def make(source: str, agent: str = "codex"):
        script = tmp_path / f"fake_{agent}.py"
        script.write_text(source)
        monkeypatch.setattr(
            runner_mod, "build_command",
            lambda a, prompt, **kw: [sys.executable, "-u", str(script), prompt],
        )
        monkeypatch.setattr(runner_mod, "agent_version", lambda a: "fake-1.0")
        return script

    return make


def half_a_run(
    store: EventStore,
    run_id: str = "run_crashed",
    *,
    pid: int | None = None,
    age_s: float = 0.0,
):
    """A run that opened, said a few things, and then nothing — the shape a
    crash leaves. Registered by a pid that has already exited unless told
    otherwise, because that is what makes it an orphan rather than a run in
    progress somewhere else."""
    if pid is None:
        # a real pid that is definitely gone: spawn one and reap it
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        pid = proc.pid
    a = CodexExecAdapter(run_id=run_id, session_id="s")
    events = []
    for line in (FIXTURES / "codex-tools.jsonl").read_text().splitlines()[:4]:
        events.extend(a.feed(line))
    for e in events:
        e.ts -= age_s
    store.register_run(run_id, agent="codex", agent_version="1", capture_mode="driven")
    store._conn.execute("UPDATE runs SET capture_pid = ? WHERE run_id = ?", (pid, run_id))
    store._conn.commit()
    store.append_many(events)
    store.set_state(run_id, "running")
    return events


class TestOrphans:
    def test_a_run_whose_process_is_gone_is_an_orphan(self, store):
        half_a_run(store)
        assert [r.run_id for r in store.orphans()] == ["run_crashed"]

    def test_a_run_whose_process_is_alive_is_not(self, store):
        """Our own pid stands in for "another seer is capturing this". The
        sweep runs from every CLI verb, so this is the case that decides
        whether running `seer list` in a second terminal can kill a capture."""
        half_a_run(store, pid=os.getpid())
        assert store.orphans() == []

    def test_a_finished_run_is_never_an_orphan(self, store, fake_agent, tmp_path):
        fake_agent(FAKE_CODEX)
        Runner("codex", "p", store=store, cwd=tmp_path).run()
        assert store.orphans() == []

    def test_recovery_records_interrupted_not_completed(self, store):
        """The verdict we are entitled to. The agent may well have finished the
        task; what ended is our record of it, and only that can be stated."""
        half_a_run(store)
        recovered = recover_orphans(store)

        assert [r["run_id"] for r in recovered] == ["run_crashed"]
        assert store.get_run("run_crashed").state == "interrupted"
        view = reduce_run("run_crashed", list(store.read("run_crashed")))
        assert view.state is SessionState.INTERRUPTED
        assert view.outcome not in (
            Outcome.VERIFIED_PASS,
            Outcome.AGENT_CLAIMED_COMPLETE,
            Outcome.UNVERIFIED_COMPLETE,
        )

    def test_the_repair_is_in_the_log_so_reindex_cannot_undo_it(self, store):
        half_a_run(store)
        recover_orphans(store)

        store.reindex()

        assert store.get_run("run_crashed").state == "interrupted"
        last = list(store.read("run_crashed"))[-1]
        assert last.event_type is EventType.SESSION_INTERRUPTED

    def test_the_recovery_event_is_ours_and_says_so(self, store):
        """Attributed to `seer_recovery`, not to the adapter that was
        capturing. We did not watch the session end; we deduced from a dead
        process that we stopped watching it, and the log has to be able to tell
        those apart afterwards."""
        half_a_run(store)
        recover_orphans(store)

        last = list(store.read("run_crashed"))[-1]
        assert last.source.adapter == ADAPTER
        assert last.native is None, "we have no native record of an event we inferred"
        assert last.payload["recovered"] is True
        assert "not observed" in last.payload["note"]

    def test_the_interruption_is_dated_from_the_last_event_not_now(self, store):
        """A machine that was off for three days must not produce a run with
        three days of wall clock. The last thing we saw is a fact; `now` is
        only when someone next opened the store."""
        three_days = 3 * 86400
        events = half_a_run(store, age_s=three_days)
        recover_orphans(store)

        last = list(store.read("run_crashed"))[-1]
        assert last.ts == pytest.approx(events[-1].ts)
        assert time.time() - last.ts > three_days - 60, "dated from the sweep, not the run"

    def test_recovering_twice_writes_one_event(self, store):
        half_a_run(store)
        recover_orphans(store)
        n = len(list(store.read("run_crashed")))

        assert recover_orphans(store) == []
        assert len(list(store.read("run_crashed"))) == n

    def test_a_row_from_before_the_pid_column_is_swept(self, store):
        """An upgrade must not leave every old `running` row stuck. A NULL pid
        means the row predates this column, which means whatever wrote it is
        long gone."""
        half_a_run(store)
        store._conn.execute("UPDATE runs SET capture_pid = NULL")
        store._conn.commit()
        assert [r.run_id for r in store.orphans()] == ["run_crashed"]


class TestRestartMidRun:
    def test_a_server_restart_does_not_leave_a_run_looking_live(self, tmp_path):
        """The whole point, end to end: a store handed to a fresh `SeerState`
        the way a restart does has no runs that render as live."""
        root = tmp_path / "seer"
        first = EventStore(root)
        half_a_run(first)
        first.close()

        state = SeerState(root)
        try:
            assert [r["run_id"] for r in state.recovered] == ["run_crashed"]
            summary = state.store.get_run("run_crashed")
            assert summary.state == "interrupted"
            assert summary.n_events > 0, "the events it did capture are still there"
        finally:
            state.store.close()

    def test_the_events_captured_before_the_crash_survive(self, tmp_path):
        root = tmp_path / "seer"
        first = EventStore(root)
        events = half_a_run(first)
        first.close()

        second = EventStore(root)
        try:
            recover_orphans(second)
            back = list(second.read("run_crashed"))
            assert [e.event_id for e in back[: len(events)]] == [e.event_id for e in events]
        finally:
            second.close()

    def test_the_cli_sweeps_and_says_so(self, tmp_path):
        """Through the real entry point, in a real subprocess — the sweep is
        wired into `run()`, and a wiring test that calls the function directly
        would pass with the wiring removed.

        Seer is its own command, not a `nebulai` subcommand (`nebulai` never
        imports `nebulai.seer`), so the entry point under test is seer's own
        `main()` — `python -m nebulai.seer.cli`, the module-level equivalent
        of the installed `seer` console script — rather than `-m nebulai
        seer`, which no longer exists."""
        root = tmp_path / "seer"
        s = EventStore(root)
        half_a_run(s)
        s.close()

        env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent / "src")}
        proc = subprocess.run(
            [sys.executable, "-m", "nebulai.seer.cli", "--root", str(root), "list"],
            capture_output=True, text=True, timeout=60, env=env,
        )
        assert proc.returncode == 0, proc.stderr
        assert "recovered run_crashed" in proc.stderr
        assert "interrupted" in proc.stdout


# ── the stream ───────────────────────────────────────────────────────────────


@pytest.fixture
def live_server(tmp_path):
    _Handler.state = SeerState(tmp_path / "seer")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", _Handler.state, srv
    srv.shutdown()
    _Handler.state.store.close()


def read_sse(base: str, stop: threading.Event, seen: list[dict], opened: threading.Event):
    try:
        with urllib.request.urlopen(base + "/seer/live", timeout=20) as resp:
            opened.set()
            buf = ""
            for raw in resp:
                buf += raw.decode()
                while "\n\n" in buf:
                    chunk, buf = buf.split("\n\n", 1)
                    for line in chunk.splitlines():
                        if line.startswith("data: "):
                            seen.append(json.loads(line[6:]))
                if stop.is_set():
                    return
    except Exception:
        pass


class TestReconnect:
    def test_a_reconnect_replays_nothing_so_nothing_doubles(
        self, live_server, fake_agent, tmp_path
    ):
        """The stream carries what happens *while you are listening* and makes
        no promise about the gap — which is exactly why the page refetches the
        run on reconnect instead of stitching. If the stream replayed its
        backlog on connect, a viewer that blinked would count every tool call
        twice."""
        base, state, _srv = live_server
        fake_agent(FAKE_CODEX)
        r = Runner("codex", "p", store=state.store, cwd=tmp_path).run()

        seen: list[dict] = []
        stop, opened = threading.Event(), threading.Event()
        t = threading.Thread(target=read_sse, args=(base, stop, seen, opened), daemon=True)
        t.start()
        assert opened.wait(5)
        time.sleep(0.4)
        stop.set()

        # a fresh subscriber sees the handshake and nothing else: the run that
        # already happened is history, and history is fetched, not streamed
        assert [d for d in seen if "event" in d] == []
        assert seen[0]["schema_version"] == "1.0"
        # and the authoritative count is the log's, which the page refetches
        view = json.loads(
            urllib.request.urlopen(f"{base}/seer/run/{r.run_id}", timeout=5).read()
        )
        assert view["n_events"] == r.n_events

    def test_the_gap_is_in_the_log_even_though_it_was_not_in_the_stream(
        self, live_server, fake_agent, tmp_path
    ):
        """Nothing is lost by not listening. The events that happen while the
        socket is down are appended exactly as they would have been, so the
        refetch-on-reconnect the client does is a complete repair rather than a
        best effort."""
        base, state, _srv = live_server
        fake_agent(FAKE_CODEX)

        # no subscriber at all for the whole run — the "socket was down" case
        r = Runner("codex", "p", store=state.store, cwd=tmp_path).run()

        events = json.loads(
            urllib.request.urlopen(
                f"{base}/seer/run/{r.run_id}/events", timeout=5
            ).read()
        )["events"]
        assert len(events) == r.n_events
        assert events[-1]["event_type"] == EventType.RUN_COMPLETED.value

    def test_a_dropped_subscriber_does_not_stall_the_capture(self, live_server):
        """A viewer that stops reading must cost the capture nothing. The bus
        drops it and counts the drop — a silently-lagging client is worse than
        a disconnected one, because from the page you cannot tell."""
        _base, state, _srv = live_server
        q = state.bus.subscribe()
        before = state.bus.dropped
        for i in range(5000):
            state.bus.publish({"kind": "event", "n": i})
        assert state.bus.dropped > before
        assert q.qsize() > 0
