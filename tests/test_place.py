"""Tests for `seer place` — projecting a captured run into a persona space.

Two things are being defended here, and they are both honesty properties
rather than arithmetic ones:

* a turn that cannot be placed never becomes a point. It is listed under
  `skipped`, with `dropped_by_policy` and `missing` kept apart, because those
  are different facts about the run and a reader acts differently on each;
* the coordinates and the control travel together. A placement made in a space
  whose PC1 did not clear its null still carries the verdict, so a card drawn
  from the file cannot omit it.

The live server is exercised for real over a loopback socket (`live_place` is
monkeypatched to a stub so the test needs no 135M-parameter checkpoint, but the
HTTP path, the JSON body, the chunking and the error handling are the shipped
ones).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from nebulai.seer.contract import (
    Action,
    CaptureMode,
    Event,
    EventType,
    Fidelity,
    Source,
)
from nebulai.seer.place import (
    SOURCE_PINNED_MODEL,
    PlaceError,
    Placement,
    collect_turns,
    place_run,
    place_texts_http,
    read_placement,
    write_placement,
)
from nebulai.seer.store import EventStore


def _src() -> Source:
    return Source(
        agent="claude",
        agent_version="1",
        adapter="t",
        adapter_version="1",
        capture_mode=CaptureMode.DRIVEN,
        fidelity=Fidelity.NATIVE,
    )


def _ev(et: EventType, ts: float, run_id: str = "run_p", **kw) -> Event:
    return Event(
        event_type=et,
        source=_src(),
        run_id=run_id,
        session_id="ses_1",
        ts=ts,
        mono_ns=int(ts * 1e9),
        **kw,
    )


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "seer")
    yield s
    s.close()


@pytest.fixture
def run(store):
    """A run with two placeable assistant turns, one policy-dropped user turn,
    and one assistant turn whose text never existed."""
    events = [
        _ev(EventType.SESSION_STARTED, 1.0),
        _ev(
            EventType.MESSAGE_USER,
            2.0,
            action=Action.INTERACT,
            payload={"chars": 41},  # counted, not retained: dropped_by_policy
        ),
        _ev(
            EventType.MESSAGE_ASSISTANT_COMPLETED,
            3.0,
            action=Action.REPORT,
            payload={"chars": 5, "text": "hello"},
        ),
        _ev(
            EventType.MESSAGE_ASSISTANT_COMPLETED,
            4.0,
            action=Action.REPORT,
            payload={},  # nothing was captured at all: missing
        ),
        _ev(
            EventType.MESSAGE_ASSISTANT_COMPLETED,
            5.0,
            action=Action.REPORT,
            payload={"chars": 5, "text": "world"},
        ),
        _ev(EventType.SESSION_COMPLETED, 6.0),
    ]
    store.append_many(events)
    return "run_p"


# ── what there is to place ──────────────────────────────────────────────────


def test_collect_turns_defaults_to_assistant_text(store, run):
    turns = collect_turns(store, run)
    assert [t.role for t in turns] == ["assistant"] * 3
    assert [t.text for t in turns] == ["hello", None, "world"]


def test_collect_turns_can_include_the_user_side(store, run):
    turns = collect_turns(store, run, roles=("assistant", "user"))
    assert [t.role for t in turns] == ["user", "assistant", "assistant", "assistant"]


def test_dropped_by_policy_is_not_collapsed_into_missing(store, run):
    turns = collect_turns(store, run, roles=("assistant", "user"))
    by_role = {(t.role, t.index): t for t in turns}
    user = next(t for t in turns if t.role == "user")
    assert user.skip_fidelity == Fidelity.DROPPED_BY_POLICY.value
    assert "not retained" in user.skip_reason
    never = next(t for t in turns if t.role == "assistant" and t.text is None)
    assert never.skip_fidelity == Fidelity.MISSING.value
    assert by_role  # keep the mapping meaningful to the reader


# ── the projection, over a stubbed model ────────────────────────────────────


@pytest.fixture
def stub_place(monkeypatch):
    """The shipped `live_place`, over a stand-in space and model.

    Deliberately NOT a stub of `live_place` itself: its argument validation and
    its response shape are part of what these tests are checking, so only the
    two things that would need a 135M-parameter checkpoint — the space on disk
    and the model behind it — are replaced. The stand-in's "activation" is the
    prompt's token count, so a coordinate is a readable function of its input.
    """
    import numpy as np

    import nebulai.backend.interp.live_server as ls

    seen: list[dict] = []

    class FakeControl:
        verdict, pc1_evr, pc1_evr_null_p95 = "above_null", 0.42, 0.21

    class FakeSpace:
        space_id, model, revision, layer = "tiny@abc.v1.L2", "tiny/llama", "abc123", 2
        control = FakeControl()

        def project(self, acts):
            n = len(acts)
            return np.stack([acts[:, 0], -np.arange(n, dtype=np.float32)], axis=1)

    class FakeModel:
        def apply_chat_template(self, messages, *, add_generation_prompt=True):
            return messages[-1]["content"]

        def encode(self, text):
            return list(range(1, len(text) + 1))

        def capture_resid(self, prompts, layers, *, batch_size=8):
            seen.append({"prompts": [list(p) for p in prompts], "layers": list(layers)})
            rows = np.asarray([[float(len(p)), 0.0] for p in prompts], dtype=np.float32)
            return {L: rows for L in layers}

    def space(space_id):
        if space_id != "tiny@abc.v1.L2":
            raise ValueError(f"no persona space {space_id!r}")
        return FakeSpace()

    monkeypatch.setattr(ls, "_place_space", space)
    monkeypatch.setattr(ls, "_place_model", lambda _s: FakeModel())
    return type("S", (), {"seen": seen})()


def test_place_run_in_process(store, run, stub_place):
    p = place_run(store, run, "tiny@abc.v1.L2", in_process=True)
    assert p.transport == "in_process"
    assert p.source == SOURCE_PINNED_MODEL
    assert p.fidelity == "deterministic"
    assert [pt["coords"] for pt in p.points] == [[5.0, -0.0], [5.0, -1.0]]
    assert [pt["role"] for pt in p.points] == ["assistant", "assistant"]
    # the real endpoint rendered both turns through the chat template and read
    # the space's own pinned layer
    assert stub_place.seen[0]["layers"] == [2]
    assert [len(p_) for p_ in stub_place.seen[0]["prompts"]] == [5, 5]


def test_unplaceable_turns_are_skipped_not_zeroed(store, run, stub_place):
    p = place_run(store, run, "tiny@abc.v1.L2", in_process=True, roles=("assistant", "user"))
    d = p.to_dict()
    assert d["n_points"] == 2
    assert d["n_skipped"] == 2
    assert d["n_dropped_by_policy"] == 1
    assert d["n_missing"] == 1
    assert all(pt["coords"] != [0.0, 0.0] for pt in p.points)


def test_the_control_travels_with_the_coordinates(store, run, stub_place):
    p = place_run(store, run, "tiny@abc.v1.L2", in_process=True)
    assert p.verdict == "above_null"
    assert p.pc1_evr == 0.42 and p.pc1_evr_null_p95 == 0.21
    assert {"verdict", "pc1_evr", "pc1_evr_null_p95"} <= set(p.to_dict())


def test_place_run_refuses_an_unknown_run(store, stub_place):
    with pytest.raises(PlaceError, match="no run"):
        place_run(store, "nope", "tiny@abc.v1.L2", in_process=True)


def test_place_run_refuses_a_run_with_nothing_to_place(store, stub_place):
    store.append_many([_ev(EventType.SESSION_STARTED, 1.0, run_id="run_empty")])
    with pytest.raises(PlaceError, match="no assistant message events"):
        place_run(store, "run_empty", "tiny@abc.v1.L2", in_process=True)


def test_a_run_whose_every_turn_was_dropped_places_nothing_and_says_so(store, stub_place):
    store.append_many([
        _ev(EventType.SESSION_STARTED, 1.0, run_id="run_q"),
        _ev(
            EventType.MESSAGE_ASSISTANT_COMPLETED, 2.0, run_id="run_q",
            action=Action.REPORT, payload={"chars": 12},
        ),
    ])
    p = place_run(store, "run_q", "tiny@abc.v1.L2", in_process=True)
    assert p.points == []
    assert p.fidelity == Fidelity.MISSING.value
    assert p.to_dict()["n_dropped_by_policy"] == 1


def test_chunking_covers_every_text(store, stub_place, monkeypatch):
    import nebulai.seer.place as pl

    monkeypatch.setattr(pl, "CHUNK", 2)
    events = [_ev(EventType.SESSION_STARTED, 0.0, run_id="run_many")]
    for i in range(7):
        events.append(
            _ev(
                EventType.MESSAGE_ASSISTANT_COMPLETED, float(i + 1), run_id="run_many",
                action=Action.REPORT, payload={"chars": i + 1, "text": "x" * (i + 1)},
            )
        )
    store.append_many(events)
    p = place_run(store, "run_many", "tiny@abc.v1.L2", in_process=True)
    assert len(p.points) == 7
    assert [pt["coords"][0] for pt in p.points] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert [pt["seq"] for pt in p.points] == list(range(7))


# ── the file beside the run ─────────────────────────────────────────────────


def test_placement_is_written_beside_the_run(store, run, stub_place):
    p = place_run(store, run, "tiny@abc.v1.L2", in_process=True)
    path = write_placement(store, p)
    assert path == store.runs_dir / run / "placement.json"
    assert path.parent == store.log_path(run).parent
    back = read_placement(store, run)
    assert isinstance(back, Placement)
    assert back.to_dict() == p.to_dict()


def test_read_placement_of_a_run_without_one_is_none(store, run):
    assert read_placement(store, run) is None


def test_placement_json_names_its_glyph_source(store, run, stub_place):
    p = place_run(store, run, "tiny@abc.v1.L2", in_process=True)
    d = json.loads(json.dumps(p.to_dict()))
    # R7: the viewer reads the glyph off this field alone.
    assert d["placement_source"] == SOURCE_PINNED_MODEL


# ── the HTTP transport, over a real socket ──────────────────────────────────


@pytest.fixture
def live_server(stub_place):
    """The shipped `_Handler`, serving on loopback."""
    import nebulai.backend.interp.live_server as ls

    srv = ThreadingHTTPServer(("127.0.0.1", 0), ls._Handler)
    ls._Handler.lock = threading.Lock()
    ls._Handler.m = None  # /live/place never touches the GPT-2 model
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_http_transport_round_trips(store, run, live_server):
    p = place_run(store, run, "tiny@abc.v1.L2", live_url=live_server)
    assert p.transport == "live_http"
    assert [pt["coords"] for pt in p.points] == [[5.0, -0.0], [5.0, -1.0]]


def test_http_transport_reports_a_bad_space_id(store, run, live_server):
    with pytest.raises(PlaceError, match="no persona space"):
        place_run(store, run, "not-a-space", live_url=live_server)


def test_http_transport_says_what_to_do_when_nothing_is_listening(store, run):
    with pytest.raises(PlaceError, match="--in-process"):
        place_texts_http(["hi"], "tiny@abc.v1.L2", live_url="http://127.0.0.1:1")


def test_live_place_rejects_a_malformed_body(live_server):
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        live_server + "/live/place",
        data=json.dumps({"space_id": "tiny@abc.v1.L2"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 400
    assert "non-empty list" in e.value.read().decode()
