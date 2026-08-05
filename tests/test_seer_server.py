"""M1c exit tests: the runner and the HTTP/SSE server.

The runner is tested against a **fake agent** — a tiny Python script that prints
a canned event stream — rather than against the real binaries. Real agents cost
money, need network, and change their output between versions, none of which
belongs in a test that runs on every commit. The real binaries are exercised by
the fixture captures in `test_seer_adapters.py`, which is where a shape change
should be caught.

What the fake agent lets us test that fixtures cannot: exit codes, stderr,
timeouts, cancellation, and the rule that matters most here — **a process that
dies mid-stream is never recorded as a completed run.**
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from nebulai.seer import runner as runner_mod
from nebulai.seer.contract import EventType, Fidelity, Outcome, SessionState
from nebulai.seer.runner import Runner, agent_version, build_command
from nebulai.seer.server import Broadcaster, SeerState, _Handler, serve
from nebulai.seer.store import EventStore

# Two turns, one shell command, a clean end. Same vocabulary as the real
# `codex exec --json`, verified against the captured fixture.
FAKE_CODEX = r"""
import json, sys, time
def p(o): print(json.dumps(o), flush=True)
p({"type": "thread.started", "thread_id": "th_fake"})
p({"type": "turn.started"})
p({"type": "item.completed", "item": {"id": "i1", "type": "command_execution",
   "command": "/bin/zsh -lc 'pytest -q'", "exit_code": 0, "status": "completed",
   "aggregated_output": "3 passed"}})
p({"type": "item.completed", "item": {"id": "i2", "type": "agent_message",
   "text": "done"}})
p({"type": "turn.completed", "usage": {"input_tokens": 100,
   "cached_input_tokens": 10, "output_tokens": 20, "reasoning_output_tokens": 5}})
"""

FAKE_DIES_MIDSTREAM = r"""
import json, sys
def p(o): print(json.dumps(o), flush=True)
p({"type": "thread.started", "thread_id": "th_x"})
p({"type": "turn.started"})
p({"type": "item.completed", "item": {"id": "i1", "type": "command_execution",
   "command": "pytest", "exit_code": 0, "status": "completed"}})
sys.stderr.write("provider error: connection reset\n")
sys.exit(3)
"""

FAKE_HANGS = r"""
import json, sys, time
def p(o): print(json.dumps(o), flush=True)
p({"type": "thread.started", "thread_id": "th_h"})
p({"type": "turn.started"})
time.sleep(120)
"""


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "seer")
    yield s
    s.close()


@pytest.fixture
def fake_agent(tmp_path, monkeypatch):
    """Point `build_command` at a script instead of the real binary."""

    def make(source: str, agent: str = "codex"):
        script = tmp_path / f"fake_{agent}.py"
        script.write_text(source)
        monkeypatch.setattr(
            runner_mod, "build_command",
            lambda a, prompt, **kw: [sys.executable, "-u", str(script), prompt],
        )
        # `agent --version` would shell out to the real binary
        monkeypatch.setattr(runner_mod, "agent_version", lambda a: "fake-1.0")
        return script

    return make


# ── the runner ───────────────────────────────────────────────────────────────


def test_a_clean_run_is_captured_end_to_end(store, fake_agent, tmp_path):
    fake_agent(FAKE_CODEX)
    r = Runner("codex", "do the thing", store=store, cwd=tmp_path).run()

    assert r.exit_code == 0
    assert r.view.state is SessionState.COMPLETED
    assert r.view.outcome is Outcome.AGENT_CLAIMED_COMPLETE
    assert r.view.action_counts == {"verify": 1, "report": 1}
    assert r.view.usage["input"].value == 100
    # and the log on disk holds every event, in order
    logged = list(store.read(r.run_id))
    assert logged[0].event_type is EventType.RUN_STARTED
    assert logged[-1].event_type is EventType.RUN_COMPLETED
    assert len(logged) == r.n_events


def test_the_exact_command_is_recorded(store, fake_agent, tmp_path):
    fake_agent(FAKE_CODEX)
    r = Runner("codex", "p", store=store, cwd=tmp_path).run()
    (start,) = [e for e in store.read(r.run_id)
                if e.event_type is EventType.RUN_STARTED]
    # so a run can be reproduced, and so a flag change shows as a data change
    assert "fake_codex.py" in start.payload["command"]
    assert start.payload["cwd"] == str(tmp_path)


def test_a_process_that_dies_midstream_is_never_a_completed_run(
    store, fake_agent, tmp_path
):
    """The most important rule in this file. A crashed agent that happened to
    print a plausible-looking stream must not be scored as a successful run."""
    fake_agent(FAKE_DIES_MIDSTREAM)
    r = Runner("codex", "p", store=store, cwd=tmp_path).run()

    assert r.exit_code == 3
    assert r.view.state is SessionState.FAILED
    assert r.view.outcome is Outcome.INFRASTRUCTURE_FAILURE
    # the work it did do is still recorded — a failure is not an erasure
    assert r.view.action_counts.get("verify") == 1


def test_stderr_is_diagnostics_not_data(store, fake_agent, tmp_path):
    """An agent's stderr chatter must never become an action or a count, or a
    chatty release would look like an agent doing more work."""
    fake_agent(FAKE_DIES_MIDSTREAM)
    r = Runner("codex", "p", store=store, cwd=tmp_path).run()

    assert any("connection reset" in ln for ln in r.stderr_tail)
    # it reaches the record only as context on the failure — never as an event
    # carrying an action, which is what would inflate a count
    carriers = [e for e in store.read(r.run_id)
                if "connection reset" in json.dumps(e.payload)]
    assert carriers and all(e.action is None for e in carriers)


def test_a_timeout_cancels_and_records_interrupted(store, fake_agent, tmp_path):
    fake_agent(FAKE_HANGS)
    t0 = time.time()
    r = Runner("codex", "p", store=store, cwd=tmp_path).run(timeout_s=1.0)
    assert time.time() - t0 < 20  # it really was killed, not waited out
    assert r.view.state is SessionState.INTERRUPTED
    assert r.view.outcome is Outcome.INTERRUPTED


def test_a_missing_binary_is_reported_not_raised(store, tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner_mod, "build_command",
        lambda a, p, **kw: ["nebulai-no-such-binary-xyz", p],
    )
    monkeypatch.setattr(runner_mod, "agent_version", lambda a: "unknown")
    r = Runner("codex", "p", store=store, cwd=tmp_path).run()
    assert r.exit_code == 127
    assert r.view.state is SessionState.FAILED
    assert any("not on PATH" in w for w in r.view.quality.warnings)


def test_stdin_is_closed_because_codex_blocks_on_it(store, fake_agent, tmp_path):
    """`codex exec` blocks reading stdin when it is not a TTY, which hangs the
    capture with no output and no error. Regression guard for a whole afternoon."""
    import subprocess

    assert runner_mod._STDIN is subprocess.DEVNULL


def test_repo_context_is_recorded_for_a_git_worktree(store, fake_agent):
    fake_agent(FAKE_CODEX)
    r = Runner("codex", "p", store=store, cwd=Path(__file__).parent).run()
    assert r.view.repo and r.view.repo["branch"]
    assert len(r.view.repo["head"]) == 40


def test_build_command_shapes():
    assert build_command("codex", "hi")[:3] == ["codex", "exec", "--json"]
    assert "stream-json" in build_command("claude", "hi")
    assert build_command("hermes", "hi") == ["hermes", "-z", "hi"]
    with pytest.raises(ValueError, match="no launcher"):
        build_command("gemini", "hi")


def test_agent_version_never_raises_for_a_missing_binary():
    assert agent_version("nebulai-no-such-binary-xyz") == "unknown"


# ── the broadcaster ──────────────────────────────────────────────────────────


def test_a_slow_client_is_dropped_not_allowed_to_block():
    """A viewer that stops reading must never back-pressure the collector: the
    collector falling behind would distort the very timings being recorded."""
    b = Broadcaster()
    q = b.subscribe()
    for i in range(2000):  # far past CLIENT_QUEUE_MAX
        b.publish({"kind": "event", "i": i})
    assert b.dropped == 1
    assert q.qsize() > 0  # it kept what it had; it just stopped receiving


def test_publish_reaches_every_live_subscriber():
    b = Broadcaster()
    a, c = b.subscribe(), b.subscribe()
    b.publish({"kind": "event", "x": 1})
    assert a.get_nowait()["x"] == 1 and c.get_nowait()["x"] == 1
    b.unsubscribe(a)
    b.publish({"kind": "event", "x": 2})
    assert a.empty() and c.get_nowait()["x"] == 2


# ── the server ───────────────────────────────────────────────────────────────


@pytest.fixture
def live_server(tmp_path):
    """A real socket on an ephemeral port, so the routes are tested the way the
    viewer will actually reach them."""
    from http.server import ThreadingHTTPServer

    _Handler.state = SeerState(tmp_path / "seer")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base, _Handler.state
    srv.shutdown()
    _Handler.state.store.close()


def get(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=5) as r:
        return json.loads(r.read())


def post(base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def test_health_reports_the_schema_version(live_server):
    base, _ = live_server
    h = get(base, "/seer/health")
    assert h["ok"] is True
    assert h["schema_version"] == "1.0"


def test_a_non_finite_number_never_reaches_the_wire():
    """`inf` and `nan` serialize as bare `Infinity`/`NaN`, which are not JSON.

    `JSON.parse` rejects the whole *document*, so one such value anywhere in a
    response blanks the entire page rather than one field.
    """
    from nebulai.seer.server import _dumps

    body = _dumps({"a": float("inf"), "b": [float("nan"), 1.0], "c": {"d": -float("inf")}})
    assert "Infinity" not in body and "NaN" not in body
    assert json.loads(body) == {"a": None, "b": [None, 1.0], "c": {"d": None}}


def test_health_from_a_watching_server_is_parseable_json(tmp_path):
    """The case that produced the bug: watching a spool that does not exist yet
    means no manifest, which means no known clock resolution — reported as `inf`
    internally and, before this, written to the wire as bare `Infinity`."""
    from http.server import ThreadingHTTPServer

    _Handler.state = SeerState(tmp_path / "seer", watch=True)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        h = get(f"http://127.0.0.1:{srv.server_address[1]}", "/seer/health")
        assert h["observing"]["watching"] is False
        # not `inf`, and not 0.0 either: we do not know it
        assert h["observing"]["clock_resolution_s"] is None
    finally:
        srv.shutdown()
        _Handler.state.collector.stop(close_runs=False)
        _Handler.state.store.close()


def test_runs_and_run_routes(live_server, fake_agent, tmp_path):
    base, state = live_server
    fake_agent(FAKE_CODEX)
    r = Runner("codex", "p", store=state.store, cwd=tmp_path).run()

    runs = get(base, "/seer/runs")["runs"]
    assert [x["run_id"] for x in runs] == [r.run_id]

    view = get(base, f"/seer/run/{r.run_id}")
    assert view["agent"] == "codex"
    assert view["usage"]["input"]["value"] == 100
    # the fidelity travels with the number, all the way to the wire
    assert view["usage"]["cache_write"]["fidelity"] == Fidelity.MISSING.value
    assert view["usage"]["cache_write"]["value"] is None


@pytest.mark.parametrize("path", ["/seer/run/run_nope", "/seer/run/run_nope/events"])
def test_unknown_run_is_a_404_not_an_empty_view(live_server, path):
    """"No such run" and "a run that said nothing" are different facts, and a
    typo must not be able to render as the second one."""
    base, _ = live_server
    with pytest.raises(urllib.error.HTTPError) as e:
        get(base, path)
    assert e.value.code == 404


def test_events_route_supports_incremental_reads(live_server, fake_agent, tmp_path):
    base, state = live_server
    fake_agent(FAKE_CODEX)
    r = Runner("codex", "p", store=state.store, cwd=tmp_path).run()

    everything = get(base, f"/seer/run/{r.run_id}/events")["events"]
    tail = get(base, f"/seer/run/{r.run_id}/events?since=3")["events"]
    assert len(everything) == r.n_events
    assert tail == everything[3:]


def test_export_is_the_raw_append_only_record(live_server, fake_agent, tmp_path):
    base, state = live_server
    fake_agent(FAKE_CODEX)
    r = Runner("codex", "p", store=state.store, cwd=tmp_path).run()

    with urllib.request.urlopen(base + f"/seer/export?run_id={r.run_id}") as resp:
        assert resp.headers["Content-Type"].startswith("application/x-ndjson")
        body = resp.read().decode()
    lines = [ln for ln in body.splitlines() if ln]
    assert len(lines) == r.n_events
    assert json.loads(lines[0])["schema_version"] == "1.0"


def test_compare_route_carries_the_refusals(live_server, fake_agent, tmp_path):
    base, state = live_server
    fake_agent(FAKE_CODEX)
    a = Runner("codex", "p", store=state.store, cwd=tmp_path).run()
    b = Runner("codex", "p", store=state.store, cwd=tmp_path).run()

    c = get(base, f"/seer/compare?runs={a.run_id},{b.run_id}")
    assert c["comparable"]
    assert "are comparable" in c["summary"]
    # two runs of the same agent: tokens ARE comparable
    assert not any(r["metric"] == "tokens.*" for r in c["refused"])


def test_compare_route_rejects_an_unknown_run(live_server):
    base, _ = live_server
    with pytest.raises(urllib.error.HTTPError) as e:
        get(base, "/seer/compare?runs=run_a,run_b")
    assert e.value.code == 400


def test_start_route_launches_and_streams(live_server, fake_agent, tmp_path):
    base, state = live_server
    fake_agent(FAKE_CODEX)
    started = post(base, "/seer/start", {"agent": "codex", "prompt": "p",
                                         "cwd": str(tmp_path)})
    run_id = started["run_id"]

    deadline = time.time() + 15
    while time.time() < deadline:
        view = get(base, f"/seer/run/{run_id}")
        if view["state"] == "completed":
            break
        time.sleep(0.1)
    else:
        pytest.fail("run never completed")
    assert view["action_counts"] == {"verify": 1, "report": 1}


def test_a_run_id_resolves_the_instant_it_is_handed_out(live_server, fake_agent,
                                                        tmp_path):
    """No window where the server has issued an id that 404s. The viewer asks
    for the run on the line after the POST returns."""
    base, _ = live_server
    fake_agent(FAKE_CODEX)
    run_id = post(base, "/seer/start",
                  {"agent": "codex", "prompt": "p", "cwd": str(tmp_path)})["run_id"]
    view = get(base, f"/seer/run/{run_id}")  # would have raced before
    # what we know because we launched it lives on the summary; the view holds
    # only what has actually been observed, which may still be nothing
    assert view["summary"]["agent"] == "codex"
    assert view["summary"]["state"] in ("starting", "running", "completed")


def test_start_route_validates_its_body(live_server):
    base, _ = live_server
    with pytest.raises(urllib.error.HTTPError) as e:
        post(base, "/seer/start", {"agent": "codex"})
    assert e.value.code == 400


def test_sse_delivers_events_as_they_happen(live_server, fake_agent, tmp_path):
    base, state = live_server
    fake_agent(FAKE_CODEX)

    seen: list[dict] = []
    ready = threading.Event()

    def listen() -> None:
        with urllib.request.urlopen(base + "/seer/live", timeout=20) as resp:
            ready.set()
            buf = ""
            for raw in resp:
                buf += raw.decode()
                while "\n\n" in buf:
                    chunk, buf = buf.split("\n\n", 1)
                    for line in chunk.splitlines():
                        if line.startswith("data: "):
                            seen.append(json.loads(line[6:]))
                    if any(
                        d.get("event", {}).get("event_type") == "run.completed"
                        for d in seen
                    ):
                        return

    t = threading.Thread(target=listen, daemon=True)
    t.start()
    assert ready.wait(5), "SSE stream never opened"

    post(base, "/seer/start", {"agent": "codex", "prompt": "p", "cwd": str(tmp_path)})
    t.join(timeout=20)

    kinds = [d.get("event", {}).get("event_type") for d in seen if "event" in d]
    assert "run.started" in kinds
    assert "tool.completed" in kinds
    assert "run.completed" in kinds
    # the handshake names the schema so a stale viewer can refuse to render
    assert seen[0]["schema_version"] == "1.0"


def test_reindex_route_rebuilds_from_the_log(live_server, fake_agent, tmp_path):
    base, state = live_server
    fake_agent(FAKE_CODEX)
    r = Runner("codex", "p", store=state.store, cwd=tmp_path).run()
    state.store._conn.execute("DELETE FROM runs")
    state.store._conn.execute("DELETE FROM events")
    state.store._conn.commit()

    assert post(base, "/seer/reindex", {})["reindexed"] == r.n_events
    assert get(base, f"/seer/run/{r.run_id}")["agent"] == "codex"
