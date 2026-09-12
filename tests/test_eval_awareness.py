"""Tests for the eval-awareness direction — mostly for the things it refuses.

A diff of means always returns a vector, and in 576 dimensions with 64 rows a
side it returns a vector that separates the training rows beautifully whatever
the labels were. So the direction is not the claim; the null and the held-out
split are. These tests are built around two synthetic worlds:

* one where the framing really does move the residual stream, where the effect
  must clear the label-permutation null AND transfer to frames the fit never
  saw; and
* one where the activations are noise, where BOTH must fail.

A recipe that cannot fail on the second world is not a measurement, and the
tests that assert the failure are the ones worth having.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from nebulai.backend import eval_awareness as ea
from nebulai.backend.eval_awareness import (
    EvalAwarenessError,
    _cohens_d,
    _overlap,
    build_entries,
    build_prompts,
    fit_direction,
    write_entry,
)
from nebulai.backend.channels import read_channels, write_channels
from nebulai.backend.directions import read_directions, renderable
from nebulai.prompts import load_prompt_set

EVAL_MARK = "EVALFRAME"
DEPLOY_MARK = "DEPLOYFRAME"


def _doc(n_frames: int = 6, n_tasks: int = 4, heldout: int = 2) -> dict:
    return {
        "id": "eval_awareness.test",
        "axis": "a synthetic axis",
        "system_template": "assistant. {frame}",
        "frames": [
            {"eval": f"{EVAL_MARK} number {i}", "deploy": f"{DEPLOY_MARK} number {i}"}
            for i in range(n_frames)
        ],
        "tasks": [f"task {j}" for j in range(n_tasks)],
        "heldout_frames": heldout,
    }


class _Stub:
    """A model whose residual stream is whatever this test says it is.

    `signal` scales a single shared offset applied to every eval-framed prompt;
    at 0 the two conditions are the same distribution and the recipe has
    nothing to find.
    """

    model_id = "stub/eval-awareness"
    revision = "local"
    n_layer = 4
    d = 16

    def __init__(self, signal: float, *, seed: int = 11) -> None:
        self.signal = signal
        self.rng = np.random.default_rng(seed)
        self.offset = np.zeros(self.d, dtype=np.float32)
        self.offset[0] = 1.0

    def apply_chat_template(self, messages, *, add_generation_prompt: bool = True) -> str:
        return " | ".join(f"{m['role']}: {m['content']}" for m in messages)

    def capture_resid(self, prompts, layers, *, batch_size: int = 16):
        out = {}
        for L in layers:
            rows = np.empty((len(prompts), self.d), dtype=np.float32)
            for i, text in enumerate(prompts):
                v = self.rng.standard_normal(self.d).astype(np.float32)
                if EVAL_MARK in text:
                    v = v + self.offset * self.signal
                rows[i] = v
            out[int(L)] = rows
        return out


# ── the matched pair ─────────────────────────────────────────────────────────


def test_the_user_message_is_identical_across_the_two_conditions() -> None:
    """The whole design rests on this: if the task text differed, the direction
    would be free to be a topic direction and nothing downstream could tell."""
    m = _Stub(0.0)
    p = build_prompts(_doc(), m)
    users = {}
    for text, lab, fr in zip(p.texts, p.label, p.frame):
        task = text.split("user: ", 1)[1]
        users.setdefault((int(fr), task), set()).add(int(lab))
    # every (frame, task) pair appears under BOTH labels with the same task text
    assert all(v == {0, 1} for v in users.values())


def test_every_frame_contributes_one_pair_per_task() -> None:
    p = build_prompts(_doc(n_frames=6, n_tasks=4), _Stub(0.0))
    assert len(p.texts) == 6 * 2 * 4
    assert int((p.label == 1).sum()) == int((p.label == 0).sum()) == 24
    assert sorted(set(p.frame.tolist())) == [0, 1, 2, 3, 4, 5]


def test_a_frame_missing_a_side_is_an_error_not_a_one_sided_pair() -> None:
    doc = _doc()
    del doc["frames"][2]["deploy"]
    with pytest.raises(EvalAwarenessError, match="no `deploy` side"):
        build_prompts(doc, _Stub(0.0))


def test_the_shipped_prompt_set_is_matched_and_balanced() -> None:
    """The frozen file itself, not a fixture: an edit that unbalances it fails
    here rather than silently changing what the direction means."""
    doc, sha = load_prompt_set("eval_awareness.v1")
    assert len(sha) == 64
    assert doc["heldout_frames"] >= 1
    assert all({"eval", "deploy"} <= set(f) for f in doc["frames"])
    p = build_prompts(doc, _Stub(0.0))
    assert int((p.label == 1).sum()) == int((p.label == 0).sum())
    assert len(p.texts) == len(doc["frames"]) * 2 * len(doc["tasks"])


# ── the arithmetic ───────────────────────────────────────────────────────────


def test_fit_direction_is_unit_length_and_points_from_neg_to_pos() -> None:
    acts = np.zeros((4, 3), dtype=np.float32)
    acts[0:2, 0] = 1.0  # positives sit at x = 1
    label = np.array([1, 1, 0, 0])
    v = fit_direction(acts, label)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-6)
    assert v[0] > 0


def test_fit_refuses_an_empty_side_and_two_identical_means() -> None:
    acts = np.ones((4, 3), dtype=np.float32)
    with pytest.raises(EvalAwarenessError, match="no prompts"):
        fit_direction(acts, np.array([1, 1, 1, 1]))
    with pytest.raises(EvalAwarenessError, match="identical means"):
        fit_direction(acts, np.array([1, 1, 0, 0]))


def test_undefined_statistics_are_nan_never_zero() -> None:
    one = np.array([1.0])
    assert np.isnan(_cohens_d(one, np.array([0.0, 1.0])))
    assert np.isnan(_cohens_d(np.zeros(4), np.zeros(4)))  # no variance at all
    assert np.isnan(_overlap(np.array([]), one))
    # two identical distributions overlap at chance — a 0 here would read as
    # "perfectly separated", which is the opposite of the truth. Exact ties fall
    # on the wrong side for the positive class and the right side for the
    # negative one, which is what makes this 0.5 rather than 0 or 1.
    assert _overlap(one, one) == 0.5


def test_overlap_is_the_misclassification_rate_under_the_implied_threshold() -> None:
    a = np.array([1.0, 1.0, 1.0, -5.0])  # one positive on the wrong side
    b = np.array([-1.0, -1.0, -1.0, -1.0])
    assert _overlap(a, b) == pytest.approx(1 / 8)


# ── the two worlds ───────────────────────────────────────────────────────────


def test_a_real_effect_clears_its_null_and_transfers_to_unseen_frames(monkeypatch) -> None:
    monkeypatch.setattr(ea, "load_prompt_set", lambda _id: (_doc(), "f" * 64))
    entry = build_entries(_Stub(6.0), layers=[1], batch_size=8, null_n=24)[0]
    ea_block = entry.source["eval_awareness"]
    # the strict null: refit the whole procedure on shuffled labels
    assert ea_block["cohens_d"] > ea_block["label_permutation"]["cohens_d_p95"]
    # and it transfers to framings the fit never saw
    assert ea_block["frame_heldout"]["cohens_d"] > 1.0
    assert ea_block["overlap"] < 0.05
    assert ea_block["in_sample"] is True
    # the registry's own two controls are there as well, and are NOT these
    c = entry.source["contrast"]
    assert c["cohens_d"] > c["null_cohens_d_p95"]
    assert c["heldout_cohens_d"] > 1.0


def test_noise_does_not_clear_its_null_and_does_not_transfer(monkeypatch) -> None:
    """The important one. The fitted direction still separates the training
    rows — `cohens_d` is comfortably above zero — and the null is what says so
    does every shuffle of the same labels."""
    monkeypatch.setattr(ea, "load_prompt_set", lambda _id: (_doc(), "f" * 64))
    entry = build_entries(_Stub(0.0), layers=[1], batch_size=8, null_n=24)[0]
    ea_block = entry.source["eval_awareness"]
    assert ea_block["cohens_d"] > 0.0  # a diff of means always separates its own rows
    assert ea_block["cohens_d"] <= ea_block["label_permutation"]["cohens_d_p95"]
    assert abs(ea_block["frame_heldout"]["cohens_d"]) < 1.0
    # the random-DIRECTION null is the weaker one and is allowed to be cleared
    # here; it is kept in the file precisely so the two can be compared
    assert entry.source["contrast"]["null_cohens_d_p95"] > 0.0


def test_the_heldout_split_is_over_frames_the_fit_never_saw() -> None:
    doc = _doc(n_frames=6, n_tasks=4, heldout=2)
    p = build_prompts(doc, _Stub(0.0))
    test = p.frame >= (p.n_frames - 2)
    assert set(p.frame[test].tolist()).isdisjoint(set(p.frame[~test].tolist()))
    assert int(test.sum()) == 2 * 2 * 4


def test_no_split_is_said_rather_than_shown_as_a_failed_one(monkeypatch) -> None:
    monkeypatch.setattr(ea, "load_prompt_set", lambda _id: (_doc(heldout=0), "d" * 64))
    fh = build_entries(_Stub(6.0), layers=[1], batch_size=8, null_n=8)[0].source[
        "eval_awareness"
    ]["frame_heldout"]
    assert fh["cohens_d"] is None  # NOT 0.0, which reads as "did not transfer"
    assert fh["n_pos"] == 0
    assert "missing" in fh


def test_the_sweep_writes_every_layer_it_tried(monkeypatch) -> None:
    monkeypatch.setattr(ea, "load_prompt_set", lambda _id: (_doc(), "e" * 64))
    entries = build_entries(_Stub(3.0), layers=[0, 1, 2, 3], batch_size=8, null_n=8)
    assert [e.source["layer"] for e in entries] == [0, 1, 2, 3]
    assert [e.space for e in entries] == ["resid.L0", "resid.L1", "resid.L2", "resid.L3"]
    assert len({e.id for e in entries}) == 4
    assert all("one layer of a sweep" in e.source["protocol"] for e in entries)


# ── provenance ───────────────────────────────────────────────────────────────


def test_the_entry_carries_the_prompt_set_sha_and_the_resolved_revision(monkeypatch, tmp_path):
    monkeypatch.setattr(ea, "load_prompt_set", lambda _id: (_doc(), "ab" * 32))
    e = build_entries(_Stub(2.0), layers=[1], batch_size=8, null_n=8)[0]
    assert e.source["prompt_set_sha"] == "abababababab"
    assert "stub/eval-awareness" in e.source["protocol"]
    p = write_entry(e, tmp_path / "directions.json", model="stub/eval-awareness", revision="deadbeef")
    doc = json.loads(p.read_text())
    assert doc["meta"]["revision"] == "deadbeef"
    assert len(doc["directions"]) == 1


def test_writing_the_same_id_twice_replaces_it(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ea, "load_prompt_set", lambda _id: (_doc(), "f" * 64))
    path = tmp_path / "directions.json"
    a = build_entries(_Stub(2.0), layers=[1], batch_size=8, null_n=8)[0]
    b = build_entries(_Stub(5.0), layers=[1], batch_size=8, null_n=8)[0]
    write_entry(a, path, model="stub/eval-awareness", revision="x")
    write_entry(b, path, model="stub/eval-awareness", revision="x")
    doc = json.loads(path.read_text())
    assert len(doc["directions"]) == 1  # a rerun is a correction, not a second reading
    assert doc["directions"][0]["vector"] == b.to_json()["vector"]


def test_a_file_for_another_model_is_refused(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ea, "load_prompt_set", lambda _id: (_doc(), "f" * 64))
    path = tmp_path / "directions.json"
    path.write_text(json.dumps({"meta": {"model": "someone/else"}, "directions": []}))
    e = build_entries(_Stub(2.0), layers=[1], batch_size=8, null_n=8)[0]
    with pytest.raises(EvalAwarenessError, match="someone/else"):
        write_entry(e, path, model="stub/eval-awareness", revision="x")

# ── the registry ─────────────────────────────────────────────────────────────


def test_the_written_entry_passes_renderable(monkeypatch, tmp_path) -> None:
    """The point of the collapse: these entries are real registry entries.

    `directions.renderable()` is the gate the viewer applies — a null block, a
    projection channel that exists in `channels.json`, and the same space tag on
    both. Before the collapse this module hand-built a dict that looked like a
    direction and could never have passed, because nothing gave it a projection.
    """
    monkeypatch.setattr(ea, "load_prompt_set", lambda _id: (_doc(), "f" * 64))
    entries, acts = ea.build_sweep(_Stub(6.0), layers=[1, 2], batch_size=8, null_n=8)
    chans = ea.project_entries(entries, acts, n_null=8)

    dpath = tmp_path / "directions.json"
    cpath = tmp_path / "channels.json"
    ea.write_entries(entries, dpath, model="stub/eval-awareness", revision="x")
    write_channels(
        cpath,
        model="stub/eval-awareness",
        revision="x",
        n_points=len(acts[1]),
        channels=chans,
        point_source="prompts:eval_awareness.test",
    )

    ok, drops = renderable(read_directions(dpath), read_channels(cpath))
    assert drops == []
    assert {d["id"] for d in ok} == {e.id for e in entries}
    assert all(e.renderable for e in entries)


def test_each_layer_is_projected_onto_its_own_layer_not_another(monkeypatch) -> None:
    """D2 in the one place it could quietly go wrong here.

    Every entry in the sweep has the same dimensionality, so projecting L1's
    direction onto L2's activations would raise nothing and produce numbers. The
    channels have to carry the direction's own space tag, and the projection has
    to have been computed from that layer's rows.
    """
    monkeypatch.setattr(ea, "load_prompt_set", lambda _id: (_doc(), "f" * 64))
    entries, acts = ea.build_sweep(_Stub(6.0), layers=[1, 2], batch_size=8, null_n=8)
    chans = ea.project_entries(entries, acts, n_null=8)
    by_id = {c.id: c for c in chans}
    for e in entries:
        L = e.source["layer"]
        assert e.space == f"resid.L{L}"
        ch = by_id[e.projection["channel"]]
        assert ch.space == e.space
        # the parallel channel IS acts[L] @ v, which pins which rows were used
        assert np.allclose(ch.values, acts[L].astype(np.float64) @ e.vector, atol=1e-5)


def test_an_entry_with_no_activations_for_its_layer_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(ea, "load_prompt_set", lambda _id: (_doc(), "f" * 64))
    entries, acts = ea.build_sweep(_Stub(6.0), layers=[1], batch_size=8, null_n=8)
    with pytest.raises(EvalAwarenessError, match="no activations"):
        ea.project_entries(entries, {}, n_null=8)
