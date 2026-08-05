"""M5 exit tests: a run can be deleted, and deletion is total.

Everything else in SessionSeer is built on "the log is the record" — the index
is a cache, `reindex()` rebuilds it from the JSONL, and nothing a chart depends
on lives only in SQLite. That guarantee is exactly what makes deletion easy to
get wrong: remove the index rows and the run comes back on the next reindex;
remove the log and the run stays in `list_runs` as a capture that exports as an
empty file.

So the property under test is not "delete returns 200". It is:

* after a delete, **no** surface reports the run — not `list_runs`, not the
  filesystem, and not a reindex that reads the filesystem back;
* a run still being captured is refused rather than half-deleted;
* "deleted" and "was never here" stay different answers.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from nebulai.seer import runner as runner_mod
from nebulai.seer.adapters import CodexExecAdapter
from nebulai.seer.cli import _cmd_delete
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

FAKE_HANGS = r"""
import json, sys, time
def p(o): print(json.dumps(o), flush=True)
p({"type": "thread.started", "thread_id": "th_h"})
p({"type": "turn.started"})
time.sleep(120)
"""


def replay(run_id: str, fixture: str = "codex-tools.jsonl"):
    a = CodexExecAdapter(run_id=run_id, session_id="s")
    events = []
    for line in (FIXTURES / fixture).read_text().splitlines():
        events.extend(a.feed(line))
    events.extend(a.finish())
    return events


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
            lambda a, prompt, **kw: [__import__("sys").executable, "-u",
                                     str(script), prompt],
        )
        monkeypatch.setattr(runner_mod, "agent_version", lambda a: "fake-1.0")
        return script

    return make


# ── the store ────────────────────────────────────────────────────────────────


class TestStore:
    def test_delete_removes_the_log_the_directory_and_the_rows(self, store):
        events = replay("run_a")
        store.append_many(events)
        store.set_state("run_a", "completed")
        log = store.log_path("run_a")
        assert log.exists()

        gone = store.delete_run("run_a")

        assert gone["run_id"] == "run_a"
        assert gone["events"] == len(events)
        assert gone["bytes"] > 0
        assert gone["log_removed"] is True
        assert not log.exists()
        assert not log.parent.exists(), "the run directory outlived its log"
        assert store.get_run("run_a") is None
        assert [r.run_id for r in store.list_runs()] == []

    def test_reindex_cannot_resurrect_a_deleted_run(self, store):
        """The one that makes deletion mean something.

        `reindex()` exists to prove the index is a cache: throw it away and the
        logs rebuild it. That is also the sharpest test of a delete — if the
        log survived anywhere, the very next reindex puts the run back on the
        list with every event intact.
        """
        store.append_many(replay("run_keep"))
        store.append_many(replay("run_go"))

        store.delete_run("run_go")
        store.reindex()

        assert [r.run_id for r in store.list_runs()] == ["run_keep"]
        assert store.get_run("run_go") is None
        assert list(store.read("run_go")) == []

    def test_deleting_one_run_leaves_its_neighbours_untouched(self, store):
        a, b = replay("run_a"), replay("run_b")
        store.append_many(a)
        store.append_many(b)

        store.delete_run("run_a")

        back = list(store.read("run_b"))
        assert [e.event_id for e in back] == [e.event_id for e in b]
        assert store.get_run("run_b") is not None

    def test_deleting_a_run_that_was_never_here_raises(self, store):
        """"Deleted" and "was never here" are different answers, and only one
        of them should reassure anyone."""
        with pytest.raises(KeyError):
            store.delete_run("run_nope")

    def test_a_run_with_rows_but_no_log_can_still_be_deleted(self, store):
        """The half-state a crash can leave. Deleting it must not raise —
        otherwise the only way to clear it is to hand-edit SQLite."""
        store.append_many(replay("run_a"))
        log = store.log_path("run_a")
        store._handles.pop("run_a").close()
        log.unlink()

        gone = store.delete_run("run_a")

        assert gone["bytes"] == 0
        assert store.get_run("run_a") is None

    def test_a_log_with_no_rows_can_still_be_deleted(self, store):
        """The mirror case: an import that wrote the log and died before the
        index caught up. `list_runs` shows nothing, so the only way to find it
        is on disk — and the only way to clear it is by run id."""
        store.append_many(replay("run_a"))
        store._conn.execute("DELETE FROM events WHERE run_id = 'run_a'")
        store._conn.execute("DELETE FROM runs WHERE run_id = 'run_a'")
        store._conn.commit()

        gone = store.delete_run("run_a")

        assert gone["events"] == 0
        assert not store.log_path("run_a").exists()

    def test_the_append_handle_is_closed_not_left_writing_into_nothing(self, store):
        """Deleting closes our own handle. Without that, the next append goes
        to an unlinked inode: the writer sees success, `read()` sees an empty
        run, and nothing anywhere reports a problem."""
        store.append_many(replay("run_a"))
        assert "run_a" in store._handles

        store.delete_run("run_a")
        assert "run_a" not in store._handles

        # and the id is reusable — a fresh log, not a resurrected one
        store.append_many(replay("run_a"))
        assert store.log_path("run_a").exists()
        assert len(list(store.read("run_a"))) == len(replay("run_a"))


# ── the CLI ──────────────────────────────────────────────────────────────────


class TestCLI:
    def _args(self, run_id: str, *, yes: bool) -> argparse.Namespace:
        return argparse.Namespace(run_id=run_id, yes=yes)

    def test_without_yes_it_describes_the_run_and_keeps_it(self, store, capsys):
        store.append_many(replay("run_a"))
        store.set_state("run_a", "completed", label="the one with the tests")

        code = _cmd_delete(self._args("run_a", yes=False), store)

        assert code == 2
        err = capsys.readouterr().err
        assert "the one with the tests" in err, "a run id is not a description"
        assert "--yes" in err
        assert store.get_run("run_a") is not None

    def test_with_yes_it_deletes_and_says_what_went(self, store, capsys):
        events = replay("run_a")
        store.append_many(events)

        code = _cmd_delete(self._args("run_a", yes=True), store)

        assert code == 0
        out = capsys.readouterr().out
        assert "run_a" in out and f"{len(events)} events" in out
        assert store.get_run("run_a") is None

    def test_an_unknown_run_is_an_error_not_a_silent_success(self, store, capsys):
        code = _cmd_delete(self._args("run_nope", yes=True), store)
        assert code == 2
        assert "unknown run" in capsys.readouterr().err


# ── the route ────────────────────────────────────────────────────────────────


@pytest.fixture
def live_server(tmp_path):
    _Handler.state = SeerState(tmp_path / "seer")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", _Handler.state
    srv.shutdown()
    _Handler.state.store.close()


def _req(base: str, path: str, method: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


class TestRoute:
    def test_delete_removes_the_run_from_every_route(self, live_server, fake_agent,
                                                     tmp_path):
        base, state = live_server
        fake_agent(FAKE_CODEX)
        r = Runner("codex", "p", store=state.store, cwd=tmp_path).run()

        gone = _req(base, f"/seer/run/{r.run_id}", "DELETE")
        assert gone["events"] == r.n_events

        assert _req(base, "/seer/runs", "GET")["runs"] == []
        with pytest.raises(urllib.error.HTTPError) as e:
            _req(base, f"/seer/run/{r.run_id}", "GET")
        assert e.value.code == 404

    def test_a_finished_run_is_no_longer_held_as_live(self, live_server,
                                                      fake_agent, tmp_path):
        """`runners` is "what we are capturing", not "what we have ever
        captured". It used to be the second: nothing removed a finished run,
        and `/seer/health` hid that by filtering on `proc.poll()`. The first
        caller to ask the dict itself whether a run was live got "yes" for
        every run the server had ever started."""
        base, state = live_server
        fake_agent(FAKE_CODEX)
        run_id = _req(base, "/seer/start", "POST",
                      {"agent": "codex", "prompt": "p", "cwd": str(tmp_path)})["run_id"]

        deadline = time.time() + 15
        while _req(base, f"/seer/run/{run_id}", "GET")["state"] != "completed":
            assert time.time() < deadline, "run never completed"
            time.sleep(0.1)

        deadline = time.time() + 5
        while run_id in state.runners and time.time() < deadline:
            time.sleep(0.05)  # the finally block runs just after the last event
        assert run_id not in state.runners
        _req(base, f"/seer/run/{run_id}", "DELETE")  # would have been a 409

    def test_deleting_an_unknown_run_is_a_404(self, live_server):
        base, _ = live_server
        with pytest.raises(urllib.error.HTTPError) as e:
            _req(base, "/seer/run/run_nope", "DELETE")
        assert e.value.code == 404

    def test_a_run_still_being_captured_is_refused_whole(self, live_server,
                                                         fake_agent, tmp_path):
        """409, not a partial delete. The runner holds an open append handle
        and will keep writing, so deleting now produces a run that comes back a
        few lines shorter and missing its beginning."""
        base, state = live_server
        fake_agent(FAKE_HANGS)
        run_id = _req(base, "/seer/start", "POST",
                      {"agent": "codex", "prompt": "p", "cwd": str(tmp_path)})["run_id"]

        log = state.store.log_path(run_id)
        deadline = time.time() + 10
        while not (run_id in state.runners and log.exists()) and time.time() < deadline:
            time.sleep(0.05)
        assert run_id in state.runners and log.exists(), "the run never came up"

        with pytest.raises(urllib.error.HTTPError) as e:
            _req(base, f"/seer/run/{run_id}", "DELETE")
        assert e.value.code == 409
        assert "cancel it first" in json.loads(e.value.read())["error"]
        assert log.exists(), "refused, but deleted anyway"

        _req(base, "/seer/cancel", "POST", {"run_id": run_id})
        deadline = time.time() + 15
        while run_id in state.runners and time.time() < deadline:
            time.sleep(0.1)
        _req(base, f"/seer/run/{run_id}", "DELETE")
        assert not log.exists()

    def test_the_deletion_is_announced_on_the_bus(self, live_server, fake_agent,
                                                 tmp_path):
        """A second viewer holding the same run open has to hear about it —
        otherwise it keeps rendering a run that no longer exists and 404s on
        the next click."""
        base, state = live_server
        fake_agent(FAKE_CODEX)
        r = Runner("codex", "p", store=state.store, cwd=tmp_path).run()

        q = state.bus.subscribe()
        _req(base, f"/seer/run/{r.run_id}", "DELETE")

        seen = []
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                seen.append(q.get(timeout=0.2))
            except Exception:
                pass
            if any(m.get("kind") == "run_deleted" for m in seen):
                break
        msg = next(m for m in seen if m.get("kind") == "run_deleted")
        assert msg["run_id"] == r.run_id
        assert msg["events"] == r.n_events

    def test_options_advertises_delete(self, live_server):
        """The viewer's fetch is cross-origin against the dev server; a DELETE
        that is not in Allow-Methods is refused by the browser before it is
        ever sent, and the failure shows up as a network error with no server
        log line to explain it."""
        base, _ = live_server
        req = urllib.request.Request(base + "/seer/runs", method="OPTIONS")
        with urllib.request.urlopen(req, timeout=5) as r:
            assert "DELETE" in r.headers["Access-Control-Allow-Methods"]
