"""The loopback runner server, and the two acts it takes to spend money.

BEHAVIORAL-DIVERGENCE-PLAN.md §8.4. The Behavior page is a static artifact by
default; this server is the optional half that lets a local user start a run
from the browser. That is only safe to ship because of three properties, and
every one of them is the kind that decays silently:

  * **Loopback only.** The socket binds `127.0.0.1` and a non-loopback `Host`
    header is refused. A bind on `0.0.0.0` is refused outright, at startup,
    rather than being listened for and hoped about.
  * **A paid run takes two separate acts.** `--allow-paid` is a decision made in
    a terminal, and each paid run then needs an explicit `approve: true` in its
    body after the estimate has been read. A browser cannot spend by itself, and
    neither can a page that was told to POST by something else.
  * **Every response says what it is.** `/health` reports whether paid runs are
    permitted; the estimate reports an unknown price as `null`, never `$0`.

The refusal tests assert on the SIDE EFFECT as well as the status code: a 402
that had already queued the run would be a 402 that spent the money.
"""

import http.client
import json
import os
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from nebulai.behavior import server as SRV
from nebulai.behavior.contract import Cue, Manifest, ModelRef
from nebulai.behavior.protocol import default_frames
from nebulai.behavior.store import TrialStore

FREE_ARMS = [ModelRef("A", "fake", "fake"), ModelRef("B", "fake", "fake")]
PAID_ARMS = [ModelRef("A", "fake", "fake"), ModelRef("B", "xai", "grok-4-0709")]


def _manifest(tmp_path: Path, *, models=None, study_id="t_server", **kw) -> Path:
    base = dict(
        study_id=study_id,
        created="2026-09-11T00:00:00Z",
        models=list(models or FREE_ARMS),
        cues=[Cue("hot", "control_neutral"), Cue("salt", "control_neutral")],
        frames=default_frames(),
        trials_per_cue=2,
        n_time_blocks=2,
        seed=7,
    )
    base.update(kw)
    m = Manifest(**base)  # type: ignore[arg-type]
    m.freeze("2026-09-11T00:00:00Z")
    return m.save(tmp_path / f"{study_id}.json")


@pytest.fixture(autouse=True)
def _reset_state():
    """The in-flight state is a module global — one run at a time is the point.

    Reset around every test so an assertion about "a run is already in flight"
    cannot leak into the next one.
    """
    SRV._STATE.clear()
    SRV._STATE.update(
        {"running": False, "arm": "", "done": 0, "total": 0, "spent_usd": 0.0}
    )
    yield
    SRV._STATE.clear()
    SRV._STATE.update(
        {"running": False, "arm": "", "done": 0, "total": 0, "spent_usd": 0.0}
    )


@pytest.fixture(autouse=True)
def _no_xai_key(monkeypatch):
    """No test in this file may reach the network, on any machine.

    There is no `XAI_API_KEY` here, but "the developer's box happens not to have
    a key" is not a test isolation strategy: on a box that did have one, the
    approved-paid-run test would issue real billed requests. Pinning the lookup
    to None makes the refusal path the only path.
    """
    monkeypatch.setattr("nebulai.behavior.adapters.xai.load_key", lambda *a, **k: None)


class _Client:
    """A live server on an ephemeral loopback port, plus a raw HTTP client.

    Raw `http.client` rather than a helper: the `Host` header is part of what is
    under test, and most convenience wrappers set it for you.
    """

    def __init__(self, out_dir: Path, allow_paid: bool = False):
        SRV.Handler.out_dir = str(out_dir)
        SRV.Handler.allow_paid = allow_paid
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), SRV.Handler)
        self.host, self.port = self.srv.server_address[0], self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def request(self, method: str, path: str, body=None, host: str | None = None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        headers = {"Host": host or f"127.0.0.1:{self.port}"}
        payload = None
        if body is not None:
            payload = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        c.request(method, path, body=payload, headers=headers)
        resp = c.getresponse()
        raw = resp.read()
        c.close()
        return resp.status, (json.loads(raw) if raw else {})

    def get(self, path, **kw):
        return self.request("GET", path, **kw)

    def post(self, path, body, **kw):
        return self.request("POST", path, body=body, **kw)

    def wait_idle(self, timeout: float = 30.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            _, state = self.get("/progress")
            if not state.get("running"):
                return state
            time.sleep(0.02)
        raise AssertionError("the run never finished")

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def free_server(tmp_path):
    c = _Client(tmp_path / "out", allow_paid=False)
    yield c
    c.close()


@pytest.fixture
def paid_server(tmp_path):
    c = _Client(tmp_path / "out", allow_paid=True)
    yield c
    c.close()


# --------------------------------------------------------------------------
# loopback
# --------------------------------------------------------------------------


def test_the_server_binds_loopback_and_nothing_else(free_server):
    assert free_server.host == "127.0.0.1"


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.0.199", "::", "example.com"])
def test_serve_refuses_to_bind_a_non_loopback_interface(host, tmp_path, monkeypatch):
    """Refused at startup, not defended at request time.

    A tool that binds every interface and then filters by `Host` is one
    misconfigured proxy away from being a service, and this one has an endpoint
    that spends money.
    """
    monkeypatch.setattr(SRV, "ThreadingHTTPServer", _never_constructed)
    with pytest.raises(SystemExit) as exc:
        SRV.serve(str(tmp_path), host=host, port=0)
    assert "loopback-only" in str(exc.value)


def _never_constructed(*a, **kw):  # pragma: no cover - reaching it is the failure
    raise AssertionError("serve() bound a socket before checking the host")


def test_serve_binds_the_loopback_address_it_was_given(tmp_path, monkeypatch):
    bound: dict = {}

    class _FakeServer:
        def __init__(self, addr, handler):
            bound["addr"] = addr
            bound["handler"] = handler

        def serve_forever(self):
            raise KeyboardInterrupt  # the documented way this process ends

        def server_close(self):
            bound["closed"] = True

    monkeypatch.setattr(SRV, "ThreadingHTTPServer", _FakeServer)
    SRV.serve(str(tmp_path), host="127.0.0.1", port=8765, allow_paid=True)
    assert bound["addr"] == ("127.0.0.1", 8765)
    assert bound["closed"] is True, "the socket is released on shutdown"
    assert SRV.Handler.allow_paid is True  # the launch-time decision reaches the handler
    SRV.Handler.allow_paid = False


@pytest.mark.parametrize("host", ["evil.example.com", "nebulai.local", "192.168.0.199"])
def test_a_non_loopback_host_header_is_refused_on_every_route(host, free_server, tmp_path):
    """DNS rebinding: a remote page can make a browser send a request to
    127.0.0.1, and the `Host` header is what distinguishes that from a local
    one. The guard runs before routing, so it covers the spending endpoint too.
    """
    for method, path, body in (
        ("GET", "/health", None),
        ("GET", "/progress", None),
        ("GET", "/estimate?manifest=x", None),
        ("POST", "/run", {"manifest": str(_manifest(tmp_path))}),
    ):
        status, payload = free_server.request(method, path, body=body, host=host)
        assert status == 403, f"{method} {path} was not guarded"
        assert "not loopback" in payload["error"]
    assert SRV._STATE["running"] is False, "a refused request may not have started a run"


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_loopback_host_names_are_accepted(host, free_server):
    status, _ = free_server.get("/health", host=host)
    assert status == 200


def test_no_cors_header_is_offered_to_any_origin(free_server):
    """Opening this up would turn a loopback tool into something any page in the
    browser could drive — including the endpoint that spends."""
    c = http.client.HTTPConnection("127.0.0.1", free_server.port, timeout=10)
    c.request("GET", "/health", headers={"Host": "127.0.0.1"})
    resp = c.getresponse()
    resp.read()
    assert resp.getheader("Access-Control-Allow-Origin") is None
    c.close()


def test_an_unknown_route_is_a_404_on_both_verbs(free_server):
    assert free_server.get("/admin")[0] == 404
    assert free_server.post("/shutdown", {})[0] == 404


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------


def test_health_says_paid_runs_are_refused_when_they_are(free_server):
    """The page must never have to guess. "Paid runs refused" is a fact about
    how the server was started, and /health states it.

    One server per test on purpose: `allow_paid` and `out_dir` are attributes of
    the `Handler` CLASS, not of an instance, so two servers in one process
    necessarily share the last value written. That is fine for the shipped shape
    — `serve()` sets them and binds one socket — but it means a test cannot hold
    a free and a paid server open at the same time.
    """
    assert free_server.get("/health")[1]["allow_paid"] is False


def test_health_says_paid_runs_are_permitted_when_they_are(paid_server):
    assert paid_server.get("/health")[1]["allow_paid"] is True


def test_health_reports_ok_the_time_and_the_store_it_is_looking_at(free_server, tmp_path):
    status, payload = free_server.get("/health")
    assert status == 200
    assert payload["ok"] is True
    assert payload["out_dir"] == str(tmp_path / "out")
    assert payload["time"].endswith("+00:00"), "UTC, and says so"


def test_health_lists_the_studies_already_in_the_store(free_server, tmp_path):
    """So the page can tell "nothing collected yet" from "not looked at yet"."""
    assert free_server.get("/health")[1]["studies"] == []
    with TrialStore(tmp_path / "out" / "t_server" / "trials.sqlite"):
        pass
    assert free_server.get("/health")[1]["studies"] == ["t_server"]


def test_health_names_the_process_writing_a_store(free_server, tmp_path):
    """A run started from a terminal is invisible to this server's own _STATE.
    Without /health reporting the store's writer lock the page would offer to
    start a run that is already in flight and then surface a refusal it could
    have predicted — so the lock state is part of health, not of the error path.
    """
    f = tmp_path / "out" / "t_writers" / "trials.sqlite"
    with TrialStore(f) as s:
        assert free_server.get("/health")[1]["writers"] == {}, "no lock, no entry"
        s.claim_writer(note="behavior run --arm discovery")
        w = free_server.get("/health")[1]["writers"]
        assert list(w) == ["t_writers"]
        assert w["t_writers"]["pid"] == os.getpid()
        assert w["t_writers"]["live"] is True
        assert "--arm discovery" in w["t_writers"]["note"]
        s.release_writer()
        assert free_server.get("/health")[1]["writers"] == {}


# --------------------------------------------------------------------------
# progress
# --------------------------------------------------------------------------


def test_progress_has_a_stable_shape_before_anything_has_run(free_server):
    status, state = free_server.get("/progress")
    assert status == 200
    assert state == {"running": False, "arm": "", "done": 0, "total": 0, "spent_usd": 0.0}


def test_progress_reports_the_arm_the_totals_and_the_spend_during_a_run(free_server, tmp_path):
    """Counts alone would let a watching human miss money moving."""
    path = _manifest(tmp_path)
    status, started = free_server.post("/run", {"manifest": str(path)})
    assert status == 202
    assert started["started"] is True and started["arm"] == "discovery"
    final = free_server.wait_idle()
    assert final["arm"] == "discovery"
    assert final["running"] is False
    assert final["done"] == 8  # 2 cues × 2 repeats × 2 models
    assert final["spent_usd"] == 0.0
    assert final["errors"] == 0
    assert final["halted"] == ""


def test_a_finished_run_leaves_its_trials_in_the_store(free_server, tmp_path):
    """The progress stream is a view of a real side effect, not a simulation."""
    free_server.post("/run", {"manifest": str(_manifest(tmp_path))})
    free_server.wait_idle()
    with TrialStore(tmp_path / "out" / "t_server" / "trials.sqlite") as s:
        assert s.count("t_server") == 8
        assert s.spent_usd("t_server") == 0.0


def test_a_second_run_is_refused_while_one_is_in_flight(free_server, tmp_path):
    """Two concurrent runs against one store would interleave their spend
    against a single ceiling and neither would know it."""
    SRV._STATE["running"] = True
    status, payload = free_server.post("/run", {"manifest": str(_manifest(tmp_path))})
    assert status == 409
    assert "already in flight" in payload["error"]
    assert payload["state"]["running"] is True


def test_a_run_of_a_limited_number_of_trials_stops_there(free_server, tmp_path):
    free_server.post("/run", {"manifest": str(_manifest(tmp_path)), "limit": 3})
    final = free_server.wait_idle()
    assert final["done"] == 3


def test_a_failure_inside_the_run_is_surfaced_rather_than_swallowed(free_server, tmp_path, monkeypatch):
    """A run that died must not leave the page showing "running" forever.

    SOURCE FIX (`server._execute`): the store was opened, the budget built and
    the preflight run BEFORE the `try`, on a daemon thread whose only channel
    back to the page is `_STATE`. Any failure there — an unwritable out_dir, a
    preflight that trips the ceiling — killed the thread with `running` still
    True and no `error` field, so `/progress` reported a run in flight forever
    and the next `/run` was refused with 409 against a run that no longer
    existed. Setup now happens inside the same `try`, and the `finally` closes
    the store only if one was opened.
    """
    def _boom(*a, **kw):
        raise RuntimeError("the store is on fire")

    monkeypatch.setattr(SRV, "TrialStore", _boom)
    free_server.post("/run", {"manifest": str(_manifest(tmp_path))})
    final = free_server.wait_idle()
    assert final["running"] is False
    assert "RuntimeError: the store is on fire" in final["error"]


# --------------------------------------------------------------------------
# a paid run cannot start by itself
# --------------------------------------------------------------------------


def test_a_paid_run_is_refused_outright_without_the_launch_time_decision(free_server, tmp_path):
    """Act one, missing: `--allow-paid` is a terminal decision, not a browser one."""
    path = _manifest(tmp_path, models=PAID_ARMS, study_id="t_paid")
    status, payload = free_server.post("/run", {"manifest": str(path)})
    assert status == 403
    assert "without --allow-paid" in payload["error"]
    assert "terminal decision, not a browser one" in payload["error"]
    assert payload["n_paid_trials"] == 4  # 2 cues × 2 repeats on the one paid arm


def test_approve_true_does_not_override_a_server_started_without_allow_paid(free_server, tmp_path):
    """The two acts are conjunctive. A page that sets `approve: true` cannot
    grant itself the permission the operator withheld at launch."""
    path = _manifest(tmp_path, models=PAID_ARMS, study_id="t_paid")
    status, payload = free_server.post("/run", {"manifest": str(path), "approve": True})
    assert status == 403
    assert SRV._STATE["running"] is False


def test_a_paid_run_without_an_explicit_approval_is_refused_with_the_estimate(paid_server, tmp_path):
    """Act two, missing. 402 Payment Required, with the estimate the human is
    supposed to read and the exact way to resend."""
    path = _manifest(tmp_path, models=PAID_ARMS, study_id="t_paid")
    status, payload = paid_server.post("/run", {"manifest": str(path)})
    assert status == 402
    assert payload["error"] == "paid run not approved"
    assert payload["how"] == "resend with approve: true after showing this estimate"
    assert "trials:" in payload["estimate"]


def test_a_refused_paid_run_starts_nothing_and_writes_nothing(paid_server, tmp_path):
    """The assertion that matters more than the status code: a 402 that had
    already queued the work would be a 402 that spent the money."""
    path = _manifest(tmp_path, models=PAID_ARMS, study_id="t_paid")
    paid_server.post("/run", {"manifest": str(path)})
    time.sleep(0.05)  # long enough for a thread to have started, had one been started
    assert SRV._STATE["running"] is False
    assert paid_server.get("/progress")[1]["done"] == 0
    assert not (tmp_path / "out" / "t_paid").exists(), "no store may be created for a refused run"


@pytest.mark.parametrize("approve", [False, 0, None, "", "yes-please"])
def test_only_a_truthy_approval_field_counts_and_a_string_is_not_a_signature(
    approve, paid_server, tmp_path
):
    """`"yes-please"` is truthy and therefore accepted — recorded here so the
    looseness is a known property rather than a surprise. The falsy values are
    the ones that must never start a run.
    """
    path = _manifest(tmp_path, models=PAID_ARMS, study_id="t_paid")
    status, _ = paid_server.post("/run", {"manifest": str(path), "approve": approve})
    if approve == "yes-please":
        assert status == 202
        paid_server.wait_idle()
    else:
        assert status == 402
        assert SRV._STATE["running"] is False


def test_an_approved_paid_run_still_records_the_unreachable_arm_as_not_run(paid_server, tmp_path):
    """There is no `XAI_API_KEY` on this machine, so the paid arm cannot run.

    This is the closest an offline test can get to the approved path: the run
    starts, the free arm collects, and the paid arm is recorded as `not_run`
    with its reason rather than being silently skipped (§5.5). The actual HTTP
    round trip to x.ai has never been exercised anywhere in this suite.
    """
    path = _manifest(tmp_path, models=PAID_ARMS, study_id="t_paid")
    status, _ = paid_server.post("/run", {"manifest": str(path), "approve": True})
    assert status == 202
    final = paid_server.wait_idle()
    assert "B" in final["not_run"]
    assert "XAI_API_KEY" in final["not_run"]["B"]
    assert final["spent_usd"] == 0.0
    with TrialStore(tmp_path / "out" / "t_paid" / "trials.sqlite") as s:
        assert {t.model_key for t in s.iter_trials("t_paid")} == {"A"}


def test_a_free_run_never_takes_the_approval_path_at_all(free_server, tmp_path):
    """A local arm must never be made to wait on a paid-run approval."""
    status, _ = free_server.post("/run", {"manifest": str(_manifest(tmp_path))})
    assert status == 202
    free_server.wait_idle()


# --------------------------------------------------------------------------
# the estimate the approval is given against
# --------------------------------------------------------------------------


def test_the_estimate_endpoint_reports_an_unknown_price_as_null(free_server, tmp_path):
    """§8.3: `grok-4-0709` has no corpus row, and "$0.0000" would tell a reader
    the run is free. "Price unknown" and "free" are different claims."""
    path = _manifest(tmp_path, models=PAID_ARMS, study_id="t_paid")
    status, payload = free_server.get(f"/estimate?manifest={path}")
    assert status == 200
    assert payload["est_cost_usd"] is None
    assert payload["unpriced_models"] == ["grok-4-0709"]
    assert "UNPRICED" in payload["text"]


def test_the_estimate_counts_the_paid_trials_separately_from_the_total(free_server, tmp_path):
    path = _manifest(tmp_path, models=PAID_ARMS, study_id="t_paid")
    _, payload = free_server.get(f"/estimate?manifest={path}")
    assert payload["n_paid_trials"] == 4
    assert payload["n_trials"] == 12  # 8 discovery + 4 canary (2 models × 2 blocks)
    assert payload["est_bytes"] > 0 and payload["est_seconds"] > 0


def test_a_free_study_estimates_a_real_zero_rather_than_an_unknown(free_server, tmp_path):
    """The mirror of the null test: nothing here is billed, and that IS known."""
    _, payload = free_server.get(f"/estimate?manifest={_manifest(tmp_path)}")
    assert payload["n_paid_trials"] == 0
    assert payload["est_cost_usd"] == 0.0
    assert payload["unpriced_models"] == []


def test_a_bad_manifest_path_is_a_400_with_the_reason(free_server):
    status, payload = free_server.get("/estimate?manifest=/nonexistent/m.json")
    assert status == 400
    assert payload["error"]


def test_an_edited_manifest_is_refused_by_both_endpoints(free_server, tmp_path):
    """The integrity check is not skippable via the browser: a post-freeze edit
    fails at load, which is before anything is collected or estimated."""
    path = _manifest(tmp_path)
    d = json.loads(path.read_text())
    d["effect_floor"] = 0.0001
    path.write_text(json.dumps(d))
    assert free_server.get(f"/estimate?manifest={path}")[0] == 400
    status, payload = free_server.post("/run", {"manifest": str(path)})
    assert status == 400
    assert "edited after it was frozen" in payload["error"]
    assert SRV._STATE["running"] is False


def test_a_malformed_body_is_a_400_and_starts_nothing(free_server):
    c = http.client.HTTPConnection("127.0.0.1", free_server.port, timeout=10)
    c.request("POST", "/run", body=b"{not json", headers={"Host": "127.0.0.1"})
    resp = c.getresponse()
    payload = json.loads(resp.read())
    c.close()
    assert resp.status == 400
    assert "bad JSON" in payload["error"]
    assert SRV._STATE["running"] is False
