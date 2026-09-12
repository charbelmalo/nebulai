"""P3 exit tests for the fan statistics.

Three things are pinned here and nothing else matters as much:

* **Wilson intervals against hand-computed values.** The four worked cases
  below are the published 95 % Wilson bounds, and the bounds are additionally
  checked against the score equation they are the roots of — an independent
  derivation rather than a second copy of the same arithmetic.
* **Split-half `Δ̂` on synthetic fixtures with an exact known answer.** Two
  conditions that are each internally constant have a within-condition
  separation of exactly zero, so `Δ̂` collapses to a closed form that can be
  written down by hand: `2 − 2·exp(−‖c‖²/2σ²)`.
* **A quantity `N` is too small for comes back `MISSING` with a reason**, never
  as a point estimate. Three runs have no p10; two runs per condition have no
  split half.

Offline and deterministic throughout: the agents are fake scripts, and every
random draw is seeded.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from nebulai.backend import absorbing
from nebulai.seer import ensemble as ens
from nebulai.seer import runner as runner_mod
from nebulai.seer.contract import Fidelity
from nebulai.seer.ensemble import (
    CONDITIONS_FOR_RELIABILITY,
    MIN_RUNS_FOR_FAN,
    MIN_RUNS_PER_CONDITION_FOR_RELIABILITY,
    POINT_ESTIMATE_MIN_RUNS,
    EnsembleError,
    EnsembleManifest,
    Member,
    build_ensemble,
    delta_hat,
    fan_over,
    feature_matrix,
    median_bandwidth,
    mmd2,
    new_ensemble_id,
    rate,
    read_manifest,
    step_series,
    write_manifest,
)
from nebulai.seer.runner import Runner
from nebulai.seer.store import EventStore

Z95 = absorbing.Z95

# A fake codex whose event count is driven by its prompt, so a set of runs has
# real, deterministic variance instead of being N identical copies.
FAKE_SIZED = r"""
import json, sys
def p(o): print(json.dumps(o), flush=True)
n = int(sys.argv[1])
p({"type": "thread.started", "thread_id": "th_fake"})
for t in range(n):
    p({"type": "turn.started"})
    p({"type": "item.completed", "item": {"id": f"i{t}", "type":
       "command_execution", "command": "/bin/zsh -lc 'pytest -q'",
       "exit_code": 0, "status": "completed", "aggregated_output": "ok"}})
    p({"type": "turn.completed", "usage": {"input_tokens": 10,
       "cached_input_tokens": 0, "output_tokens": 2,
       "reasoning_output_tokens": 0}})
p({"type": "item.completed", "item": {"id": "z", "type": "agent_message",
   "text": "done"}})
"""


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "seer")
    yield s
    s.close()


@pytest.fixture
def sized_agent(tmp_path, monkeypatch):
    script = tmp_path / "fake_sized.py"
    script.write_text(FAKE_SIZED)
    monkeypatch.setattr(
        runner_mod,
        "build_command",
        lambda a, prompt, **kw: [sys.executable, "-u", str(script), prompt],
    )
    monkeypatch.setattr(runner_mod, "agent_version", lambda a: "fake-1.0")
    return script


def _ensemble_of(store, sizes_by_condition, tmp_path):
    """Launch runs and hand-build the manifest that groups them."""
    manifest = EnsembleManifest(
        ensemble_id=new_ensemble_id(),
        protocol={"id": "proto_test", "hash_algorithm": "sha256",
                  "hash_fields": ["agent", "model", "prompt", "cwd",
                                  "extra_args"],
                  "agent": "codex", "model": None},
        n_runs_requested=sum(len(v) for v in sizes_by_condition.values()),
        seed_base=0,
    )
    for condition, sizes in sizes_by_condition.items():
        for n_turns in sizes:
            r = Runner("codex", str(n_turns), store=store, cwd=tmp_path).run()
            manifest.members.append(
                Member(
                    run_id=r.run_id,
                    index=len(manifest.members),
                    condition=condition,
                    protocol_id=f"proto_{condition}",
                )
            )
    write_manifest(store, manifest)
    return manifest


# ── Wilson ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "k,n,lo,hi",
    [
        # the published 95 % Wilson score bounds
        (1, 4, 0.0455872608, 0.6993581574),
        (0, 10, 0.0, 0.2775327999),
        (10, 10, 0.7224672001, 1.0),
        (5, 20, 0.1118617014, 0.4687008776),
    ],
)
def test_wilson_intervals_match_hand_computed_values(k, n, lo, hi):
    r = rate(k, n)
    assert r["k"] == k and r["n"] == n
    assert r["p"] == pytest.approx(k / n)
    assert r["ci95"][0] == pytest.approx(lo, abs=1e-9)
    assert r["ci95"][1] == pytest.approx(hi, abs=1e-9)
    assert r["fidelity"] == Fidelity.DETERMINISTIC.value


@pytest.mark.parametrize("k,n", [(1, 4), (5, 20), (3, 7), (19, 20)])
def test_the_bounds_are_the_roots_of_the_score_equation(k, n):
    """An independent derivation, not a second copy of the implementation.

    The Wilson bounds are by definition the two values of `p` at which the
    score statistic `(p̂ − p) / sqrt(p(1−p)/n)` equals ±z.
    """
    p_hat = k / n
    for bound in rate(k, n)["ci95"]:
        assert (p_hat - bound) ** 2 == pytest.approx(
            Z95**2 * bound * (1 - bound) / n, rel=1e-9
        )


def test_a_rate_with_no_trials_is_missing_not_a_rate_of_zero():
    r = rate(0, 0)
    assert r["p"] is None and r["ci95"] is None
    assert r["fidelity"] == Fidelity.MISSING.value
    assert "no runs" in r["missing"]


def test_there_is_exactly_one_wilson_implementation_in_the_repo():
    assert ens.wilson is absorbing.wilson


# ── the fan ─────────────────────────────────────────────────────────────────


def test_a_run_that_stopped_early_is_absent_from_later_steps_not_a_zero():
    # three runs; one stops after two steps at a HIGH value
    fan = fan_over([[5.0, 9.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    assert [f["n"] for f in fan] == [3, 3, 2]
    # the short run's absence does not drag step 2 towards zero …
    assert fan[2]["median"] == pytest.approx(3.0)
    assert fan[2]["lo"] == pytest.approx(3.0)
    # … and while it was present it really did widen the band
    assert fan[1]["hi"] > fan[1]["median"]


def test_the_envelope_is_p10_p90_and_not_min_max():
    """Min/max would report 0 and 10 here; those are extreme order statistics
    whose spread grows with N on its own."""
    fan = fan_over([[float(v)] for v in range(11)])
    assert fan[0]["median"] == pytest.approx(5.0)
    assert fan[0]["lo"] == pytest.approx(1.0)
    assert fan[0]["hi"] == pytest.approx(9.0)


def test_a_single_run_at_a_step_is_its_own_every_quantile():
    fan = fan_over([[4.0]])
    assert (fan[0]["lo"], fan[0]["median"], fan[0]["hi"]) == (4.0, 4.0, 4.0)
    assert fan[0]["n"] == 1


def test_no_run_reached_a_step_means_no_fan_at_all():
    assert fan_over([[], [], []]) == []


# ── per-run step series ─────────────────────────────────────────────────────


def test_steps_are_closed_turns_and_the_partial_tail_is_excluded(
    store, sized_agent, tmp_path
):
    r = Runner("codex", "3", store=store, cwd=tmp_path).run()
    events = list(store.read(r.run_id))
    series = step_series(events, "cumulative_events")

    assert len(series) == 3  # three closed turns
    assert series == sorted(series)  # cumulative, so non-decreasing
    # the trailing agent_message and RUN_COMPLETED are a partial turn, and
    # counting them would make the final step smaller than its siblings
    assert series[-1] < len(events)


def test_events_per_step_partitions_the_run(store, sized_agent, tmp_path):
    r = Runner("codex", "3", store=store, cwd=tmp_path).run()
    events = list(store.read(r.run_id))
    per = step_series(events, "events_per_step")
    cum = step_series(events, "cumulative_events")
    assert len(per) == len(cum) == 3
    assert sum(per) == pytest.approx(cum[-1])


def test_a_run_with_no_closed_turn_has_no_steps():
    assert step_series([], "cumulative_events") == []


def test_an_unknown_fan_metric_is_refused():
    with pytest.raises(EnsembleError, match="unknown fan metric"):
        step_series([], "vibes")


# ── split-half Δ̂, against exact answers ─────────────────────────────────────


def test_two_constant_conditions_give_the_closed_form_answer():
    """The synthetic fixture with a known answer.

    Every run in A sits at the origin and every run in B sits at `c`, so each
    condition's split halves are identical samples and both within-terms are
    exactly zero. `Δ̂` therefore collapses to the between-term, which for an
    RBF kernel is `2 − 2·exp(−‖c‖²/2σ²)` — writable by hand.
    """
    sigma = 1.0
    a = np.zeros((6, 2))
    b = np.zeros((6, 2))
    b[:, 0] = 1.0
    expected = 2 - 2 * math.exp(-(1.0**2) / (2 * sigma**2))

    d = delta_hat(a, b, bandwidth=sigma, n_splits=8, seed=0)
    assert d["within_a"] == pytest.approx(0.0, abs=1e-12)
    assert d["within_b"] == pytest.approx(0.0, abs=1e-12)
    assert d["between"] == pytest.approx(expected, rel=1e-12)
    assert d["delta_hat"] == pytest.approx(expected, rel=1e-12)
    assert d["delta_hat"] == pytest.approx(0.7869386806, abs=1e-9)
    assert d["fidelity"] == Fidelity.DETERMINISTIC.value
    assert d["formula"].startswith("delta_hat = MMD2(A,B)")
    assert "6.4.1" in d["source"]


def test_two_identical_conditions_score_exactly_zero():
    a = np.zeros((6, 2))
    d = delta_hat(a, a.copy(), bandwidth=1.0, n_splits=8, seed=0)
    assert d["delta_hat"] == pytest.approx(0.0, abs=1e-12)


def test_delta_hat_is_reported_negative_rather_than_clipped():
    """Two conditions differing LESS than each differs from itself. §6.4.1
    says report it as-is; a clip at zero would hide the finding."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(12, 3)) * 3.0
    d = delta_hat(x, x.copy(), n_splits=32, seed=0)
    assert d["delta_hat"] < 0
    assert d["delta_hat"] == pytest.approx(
        d["between"] - 0.5 * (d["within_a"] + d["within_b"])
    )


def test_a_real_separation_dominates_same_distribution_noise():
    rng = np.random.default_rng(11)
    a = rng.normal(size=(16, 3))
    b_same = rng.normal(size=(16, 3))
    b_far = rng.normal(size=(16, 3)) + 6.0

    near = delta_hat(a, b_same, n_splits=32, seed=0)["delta_hat"]
    far = delta_hat(a, b_far, n_splits=32, seed=0)["delta_hat"]
    assert far > 0.9
    assert abs(near) < 0.2
    assert far > 4 * abs(near)


def test_the_three_terms_are_reported_beside_the_answer():
    """A `Δ̂` near zero because both conditions are noisy and one near zero
    because they are genuinely alike are different findings."""
    rng = np.random.default_rng(5)
    d = delta_hat(
        rng.normal(size=(8, 2)), rng.normal(size=(8, 2)), n_splits=16, seed=0
    )
    assert set(d) >= {"between", "within_a", "within_b", "bandwidth",
                      "n_splits", "seed", "kernel", "estimator"}
    assert d["n_splits"] == 16 and d["seed"] == 0


def test_delta_hat_is_reproducible_from_its_recorded_seed():
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=(10, 2)), rng.normal(size=(10, 2)) + 1.0
    first = delta_hat(a, b, n_splits=32, seed=4)
    again = delta_hat(a, b, n_splits=32, seed=4)
    assert first["delta_hat"] == again["delta_hat"]


# ── small N is MISSING, not a point ─────────────────────────────────────────


def test_a_condition_with_too_few_runs_has_no_split_half():
    d = delta_hat(np.zeros((3, 2)), np.ones((3, 2)), bandwidth=1.0)
    assert d["delta_hat"] is None  # NOT a number
    assert d["between"] is None and d["within_a"] is None
    assert d["fidelity"] == Fidelity.MISSING.value
    assert str(MIN_RUNS_PER_CONDITION_FOR_RELIABILITY) in d["missing"]
    assert "got 3 and 3" in d["missing"]


def test_identical_runs_have_no_scale_to_measure_separation_on():
    a = np.zeros((6, 2))
    assert median_bandwidth(np.vstack([a, a])) is None
    d = delta_hat(a, a.copy())  # no explicit bandwidth
    assert d["delta_hat"] is None
    assert d["fidelity"] == Fidelity.MISSING.value
    assert "no scale" in d["missing"]


def test_mmd2_of_a_sample_too_small_to_have_one_is_nan_not_zero():
    assert math.isnan(mmd2(np.zeros((1, 2)), np.ones((4, 2)), bandwidth=1.0))


# ── the per-run feature vector ──────────────────────────────────────────────


class _View:
    """Just enough of a RunView for `feature_matrix`."""

    def __init__(self, *, n_turns, n_events, files, started, ended, counts):
        self.n_turns = n_turns
        self.n_events = n_events
        self.n_files_changed = files
        self.started_at = started
        self.ended_at = ended
        self.action_counts = counts


def _view(n_turns, ended=10.0, counts=None):
    return _View(
        n_turns=n_turns,
        n_events=n_turns * 3,
        files=n_turns,
        started=0.0,
        ended=ended,
        counts=counts or {"edit": n_turns, "verify": 1, "inspect": n_turns},
    )


def test_a_feature_unobserved_in_any_run_is_dropped_for_all_never_zero_filled():
    views = [_view(1), _view(2), _view(3, ended=None)]
    _, used, dropped = feature_matrix(views)
    assert "duration_s" not in used
    assert "not observed in every run" in dropped["duration_s"]


def test_a_feature_identical_in_every_run_is_dropped_as_carrying_no_distance():
    views = [_view(1), _view(2), _view(3)]
    _, used, dropped = feature_matrix(views)
    assert "action_verify" not in used  # 1 in every run
    assert "contributes no distance" in dropped["action_verify"]
    assert "n_turns" in used


def test_surviving_features_are_standardized():
    mat, used, _ = feature_matrix([_view(1), _view(2), _view(3)])
    assert mat.shape == (3, len(used))
    assert mat.mean(axis=0) == pytest.approx(np.zeros(len(used)), abs=1e-12)
    assert mat.std(axis=0) == pytest.approx(np.ones(len(used)))


# ── the document ────────────────────────────────────────────────────────────


def test_the_document_carries_every_field_the_viewer_reads(
    store, sized_agent, tmp_path
):
    manifest = _ensemble_of(store, {"a": [1, 2, 3, 4], "b": [6, 7, 8, 9]}, tmp_path)
    doc = build_ensemble(store, manifest).to_dict()

    assert doc["ensemble_id"] == manifest.ensemble_id
    assert doc["n_runs"] == 8
    assert len(doc["run_ids"]) == 8
    assert doc["protocol"]["hash_algorithm"] == "sha256"
    assert doc["protocol"]["hash_fields"][0] == "agent"
    assert doc["fidelity"] == Fidelity.DETERMINISTIC.value
    assert doc["point_estimate_min_runs"] == POINT_ESTIMATE_MIN_RUNS

    for entry in doc["fan"]:
        assert set(entry) == {"step", "median", "lo", "hi", "n"}
    # the longest run closed nine turns, so the fan is nine steps deep and its
    # last step was reached by exactly one run
    assert len(doc["fan"]) == 9
    assert doc["fan"][-1]["n"] == 1
    assert doc["fan"][0]["n"] == 8

    for name, r in doc["rates"].items():
        assert set(r) >= {"k", "n", "p", "ci95"}, name
    assert doc["rates"]["completed"]["k"] == 8

    # it is round-trippable JSON, not a dict of numpy scalars
    assert json.loads(json.dumps(doc))["n_runs"] == 8


def test_two_conditions_of_four_runs_get_a_delta_hat(
    store, sized_agent, tmp_path
):
    manifest = _ensemble_of(store, {"a": [1, 2, 3, 4], "b": [6, 7, 8, 9]}, tmp_path)
    rel = build_ensemble(store, manifest).to_dict()["reliability"]

    assert rel["fidelity"] == Fidelity.DETERMINISTIC.value
    assert rel["delta_hat"] is not None
    assert rel["condition_a"] == "a" and rel["condition_b"] == "b"
    assert rel["n_a"] == 4 and rel["n_b"] == 4
    assert rel["features_used"]
    # the two conditions really are separated, so Δ̂ is positive
    assert rel["delta_hat"] > 0


def test_a_two_run_ensemble_reports_the_fan_as_missing_with_a_reason(
    store, sized_agent, tmp_path
):
    manifest = _ensemble_of(store, {"a": [2, 3]}, tmp_path)
    doc = build_ensemble(store, manifest).to_dict()

    assert doc["n_runs"] == 2
    assert doc["fan"] == []  # NOT a one-run "median"
    assert doc["fan_fidelity"] == Fidelity.MISSING.value
    assert str(MIN_RUNS_FOR_FAN) in doc["missing"]["fan"]
    # rates are still real: a k-of-n over two runs is an honest two-run rate,
    # and its interval is wide enough to say so (2/2 is not evidence of 100%)
    r = doc["rates"]["completed"]
    assert r["n"] == 2 and r["p"] == 1.0
    assert r["ci95"][0] < 0.5


def test_one_condition_cannot_have_a_between_condition_contrast(
    store, sized_agent, tmp_path
):
    manifest = _ensemble_of(store, {"solo": [1, 2, 3, 4]}, tmp_path)
    doc = build_ensemble(store, manifest).to_dict()

    rel = doc["reliability"]
    assert rel["delta_hat"] is None
    assert rel["fidelity"] == Fidelity.MISSING.value
    assert str(CONDITIONS_FOR_RELIABILITY) in rel["missing"]
    assert "solo" in rel["missing"]
    assert doc["missing"]["reliability"] == rel["missing"]


def test_three_runs_per_condition_is_too_few_for_a_split_half(
    store, sized_agent, tmp_path
):
    manifest = _ensemble_of(store, {"a": [1, 2, 3], "b": [6, 7, 8]}, tmp_path)
    rel = build_ensemble(store, manifest).to_dict()["reliability"]
    assert rel["delta_hat"] is None
    assert rel["fidelity"] == Fidelity.MISSING.value
    assert "got 3 and 3" in rel["missing"]


def test_a_deleted_run_drops_out_and_n_runs_tells_the_truth(
    store, sized_agent, tmp_path
):
    manifest = _ensemble_of(store, {"a": [1, 2, 3, 4]}, tmp_path)
    gone = manifest.members[0].run_id
    store.delete_run(gone)

    doc = build_ensemble(store, manifest).to_dict()
    assert doc["n_runs"] == 3  # the truth, not the 4 that were asked for
    assert doc["n_runs_requested"] == 4
    assert gone not in doc["run_ids"]
    assert gone in doc["missing"]["absent_runs"]


def test_the_seed_base_is_recorded_and_says_it_was_not_applied(
    store, sized_agent, tmp_path
):
    manifest = _ensemble_of(store, {"a": [1, 2, 3]}, tmp_path)
    doc = build_ensemble(store, manifest).to_dict()
    assert doc["seed_base"] == 0
    assert doc["seed_applied"] is False
    assert "seed_note" in read_manifest(store, manifest.ensemble_id).to_dict()


# ── the route ───────────────────────────────────────────────────────────────


@pytest.fixture
def live_server(tmp_path):
    from http.server import ThreadingHTTPServer

    from nebulai.seer.server import SeerState, _Handler

    _Handler.state = SeerState(tmp_path / "seer")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base, _Handler.state
    srv.shutdown()
    _Handler.state.store.close()


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return json.loads(r.read())


def test_the_ensemble_route_serves_the_document(
    live_server, sized_agent, tmp_path
):
    base, state = live_server
    manifest = _ensemble_of(state.store, {"a": [1, 2, 3]}, tmp_path)

    doc = _get(base, f"/seer/ensemble/{manifest.ensemble_id}")
    assert doc["ensemble_id"] == manifest.ensemble_id
    assert doc["n_runs"] == 3
    assert doc["fan"] and set(doc["fan"][0]) == {"step", "median", "lo", "hi", "n"}

    listing = _get(base, "/seer/ensembles")["ensembles"]
    assert [e["ensemble_id"] for e in listing] == [manifest.ensemble_id]


def test_an_unknown_ensemble_is_a_404_with_a_hint_not_an_empty_fan(live_server):
    base, _ = live_server
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base, "/seer/ensemble/ens_nope")
    assert e.value.code == 404
    body = json.loads(e.value.read())
    assert "unknown ensemble" in body["error"]
    assert "--repeat" in body["hint"]


# ── "edited" is an action, not a file path (found on the real ctfish fan) ────
#
# The first real ensemble built with this module was 100 imported ctfish runs
# (claude-3-5-sonnet, baseline vs spooky). It reported `edited` as 0/100 with
# fidelity `deterministic` -- "no run edited anything" -- while 18 of those runs
# had performed an EDIT action. The corpus rewrites the chess board with a shell
# command and never names a file, so `n_files_changed` was 0 everywhere: the
# rate was measuring a field that capture does not populate and reporting the
# absence as a confident zero. These tests hold that fix down.


def _src_for(agent="codex"):
    from nebulai.seer.contract import CaptureMode, Fidelity, Source

    return Source(
        agent=agent,
        agent_version="1",
        adapter="t",
        adapter_version="1",
        capture_mode=CaptureMode.RECONCILED,
        fidelity=Fidelity.DETERMINISTIC,
    )


def _ev(et, ts, run_id, **kw):
    from nebulai.seer.contract import Event

    return Event(
        event_type=et,
        source=kw.pop("source", _src_for()),
        run_id=run_id,
        session_id=run_id,
        ts=ts,
        mono_ns=int(ts * 1e9),
        **kw,
    )


def _pathless_edit_run(store, run_id, *, edits, turns=2):
    """A run that edits without ever naming a file, as an imported corpus does."""
    from nebulai.seer.contract import Action, Effect, EventType

    evs = [_ev(EventType.RUN_STARTED, 0.0, run_id),
           _ev(EventType.SESSION_STARTED, 0.1, run_id)]
    t = 1.0
    for i in range(turns):
        evs.append(_ev(EventType.TURN_STARTED, t, run_id))
        t += 1.0
        if i < edits:
            evs.append(
                _ev(
                    EventType.TOOL_COMPLETED,
                    t,
                    run_id,
                    action=Action.EDIT,
                    effect=Effect.STATE_CHANGED,
                )
            )
            t += 1.0
        evs.append(_ev(EventType.TURN_COMPLETED, t, run_id))
        t += 1.0
    evs.append(_ev(EventType.SESSION_COMPLETED, t, run_id))
    evs.append(_ev(EventType.RUN_COMPLETED, t + 0.1, run_id))
    store.append_many(evs)
    return run_id


def _manifest_over(store, run_ids, conditions):
    m = EnsembleManifest(
        ensemble_id=new_ensemble_id(),
        protocol={"id": "proto_test", "agent": "codex", "model": None},
        n_runs_requested=len(run_ids),
        seed_base=0,
    )
    for i, (rid, cond) in enumerate(zip(run_ids, conditions)):
        m.members.append(
            Member(run_id=rid, index=i, condition=cond, protocol_id=f"proto_{cond}")
        )
    write_manifest(store, m)
    return m


def test_edited_counts_edit_actions_not_file_paths(store, tmp_path):
    ids = [
        _pathless_edit_run(store, f"pathless_{i}", edits=1 if i < 3 else 0)
        for i in range(10)
    ]
    doc = build_ensemble(store, _manifest_over(store, ids, ["a"] * 10)).to_dict()
    # three runs edited; the old rule said zero because none named a file
    assert doc["rates"]["edited"]["k"] == 3
    assert doc["rates"]["edited"]["n"] == 10
    assert doc["rates"]["edited"]["fidelity"] == "deterministic"


def test_a_capture_with_no_paths_reports_files_changed_as_missing(store, tmp_path):
    ids = [
        _pathless_edit_run(store, f"nopath_{i}", edits=1 if i < 4 else 0)
        for i in range(8)
    ]
    doc = build_ensemble(store, _manifest_over(store, ids, ["a"] * 8)).to_dict()
    fc = doc["rates"]["files_changed"]
    # not 0/8: nobody observed a file count, and an unobserved count is not zero
    assert fc["p"] is None
    assert fc["n"] == 0
    assert fc["fidelity"] == "missing"
    assert "never observed" in fc["missing"]
    assert "4 run(s) performed an EDIT action" in fc["missing"]


def test_a_capture_that_never_edits_gets_an_honest_zero(store, tmp_path):
    # the MISSING branch must not swallow the real answer: a set of runs that
    # genuinely edited nothing has files_changed = 0/N, a measurement
    ids = [_pathless_edit_run(store, f"clean_{i}", edits=0) for i in range(6)]
    doc = build_ensemble(store, _manifest_over(store, ids, ["a"] * 6)).to_dict()
    fc = doc["rates"]["files_changed"]
    assert fc["k"] == 0 and fc["n"] == 6
    assert fc["p"] == 0.0
    assert fc["fidelity"] == "deterministic"
    assert doc["rates"]["edited"]["k"] == 0
