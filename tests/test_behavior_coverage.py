"""Stopping a run early, and saying so.

A local arm can be unaffordable to finish: GPT-2-XL in fp32 on a machine that
pages it is minutes per trial, and the preregistered 4800-trial capability arm
then does not finish. Stopping is allowed. Stopping *quietly* is not — a reader
counting 40 cues in a 100-cue manifest cannot tell a deliberate partial run from
a lost database, and the default assumption would be the wrong one.

These tests pin the two halves of that: the truncation keeps every cue it
collects at full strength, and the artifact states the denominator.
"""

import json

import pytest

from nebulai.behavior import analyze as A
from nebulai.behavior import cli as CLI
from nebulai.behavior import export as X
from nebulai.behavior import runner as R
from nebulai.behavior.contract import CANARY_CUE
from nebulai.behavior.store import TrialStore

from test_behavior_store import _manifest, _trial  # noqa: E402


# --- cue_prefix -------------------------------------------------------------


def test_no_limit_is_no_truncation():
    m = _manifest(study_id="t_cov_none")
    assert R.cue_prefix(m, None) is None
    assert R.cue_prefix(m, len(m.cues)) is None
    assert R.cue_prefix(m, len(m.cues) + 5) is None


def test_zero_is_refused_rather_than_collecting_nothing():
    m = _manifest(study_id="t_cov_zero")
    with pytest.raises(R.RunnerError, match="no cue at all"):
        R.cue_prefix(m, 0)


def test_the_prefix_is_manifest_order_not_schedule_order():
    """A seed-dependent prefix would make "the first N cues" a different set on
    every machine, and coverage would stop being comparable to the plan."""
    m = _manifest(study_id="t_cov_order")
    keep = R.cue_prefix(m, 2)
    assert keep is not None
    assert {c.text for c in m.cues[:2]} <= keep


def test_the_canary_is_never_truncated():
    """It is a probe issued once per block for the whole study, not a cue; a
    truncation that dropped it would remove the drift check (§5.5.1)."""
    m = _manifest(study_id="t_cov_canary")
    keep = R.cue_prefix(m, 1)
    assert keep is not None and CANARY_CUE in keep


def test_a_truncated_cue_keeps_every_repeat_and_every_block(tmp_path):
    """This is the whole reason cue-level truncation exists. `--limit` would
    leave each cue with a handful of repeats, below `min_valid_trials`; this
    leaves the covered cues exactly as preregistered."""
    m = _manifest(study_id="t_cov_full")
    with TrialStore(tmp_path / "a.sqlite") as s:
        R.Runner(m, s).run("discovery", cue_limit=1)
        rows = list(s.iter_trials("t_cov_full"))
    cues = {r.cue for r in rows}
    assert len(cues) == 1
    for model in {r.model_key for r in rows}:
        mine = [r for r in rows if r.model_key == model]
        assert len({r.repeat for r in mine}) == m.trials_per_cue
        assert len({r.block for r in mine}) == m.n_time_blocks


def test_truncation_is_a_subset_of_the_untruncated_run(tmp_path):
    """The covered cues must be byte-for-byte the trials the full run would
    have collected — same identities, so a later full run resumes onto them
    instead of duplicating them."""
    m = _manifest(study_id="t_cov_sub")
    path = tmp_path / "b.sqlite"
    with TrialStore(path) as s:
        R.Runner(m, s).run("discovery", cue_limit=1)
        part = sorted(
            (r.cue, r.frame_id, r.model_key, r.repeat)
            for r in s.iter_trials("t_cov_sub")
        )
    with TrialStore(path) as s:
        res = R.Runner(m, s).run("discovery")
        full = sorted(
            (r.cue, r.frame_id, r.model_key, r.repeat)
            for r in s.iter_trials("t_cov_sub")
        )
    assert res.skipped_existing == len(part)
    assert len(full) == len(set(full)), "a trial was collected twice"
    assert set(part) <= set(full)


# --- durability: how much work a kill can throw away ----------------------


def test_the_buffer_flushes_on_the_clock_not_only_on_a_record_count(tmp_path, monkeypatch):
    """A slow arm must not be able to lose an hour of work in a resumable runner.

    The loop buffers records and writes them when 25 have accumulated. On the
    fake adapter that is instant, so the record threshold is invisible; on fp32
    GPT-2-XL it measured 3.9 min/trial, which makes 25 trials more than 90
    minutes of unpersisted work -- and a `kill -9` at minute 89 leaves the
    store with nothing to resume onto.

    So the flush is also bounded by wall clock. The test does not sleep: it
    advances a fake `time.monotonic` past FLUSH_EVERY_SECONDS on each group and
    counts the writes, which is the behaviour that matters and costs no test
    runtime.
    """
    m = _manifest(study_id="t_flush_clock", trials=40)
    writes: list[int] = []

    # The clock advances on every READ, not inside record_many: the loop only
    # calls record_many on a flush, so a clock driven from there would be
    # frozen exactly between the groups whose slowness is the thing under test.
    clock = {"t": 1000.0}

    def monotonic():
        clock["t"] += R.FLUSH_EVERY_SECONDS + 1.0
        return clock["t"]

    monkeypatch.setattr(R.time, "monotonic", monotonic)

    with TrialStore(tmp_path / "c.sqlite") as store:
        real = store.record_many

        def counting(recs):
            if recs:
                writes.append(len(recs))
            return real(recs)

        monkeypatch.setattr(store, "record_many", counting)
        runner = R.Runner(m, store)
        runner.batch_size = 1           # one trial per group, so the timer rules
        res = runner.run("discovery", cue_limit=1)
        rows = list(store.iter_trials("t_flush_clock"))

    assert res.completed == len(rows) > 25, "need more than one record-threshold"
    # Without the timer this is ceil(n/25) = 4 writes for 80 trials. With every
    # group declared slow it is one write per group, i.e. one per trial here,
    # which is the bound that matters: a kill loses the trial in flight, not 25.
    assert len(writes) > (len(rows) + 24) // 25, (
        f"only {len(writes)} flushes for {len(rows)} trials -- the wall-clock "
        f"flush did not fire"
    )
    assert max(writes) < R.FLUSH_EVERY_RECORDS, (
        "the timer should have flushed before the record threshold was reached"
    )
    assert sum(writes) == len(rows), "a record was written twice or not at all"
    # and no empty write: an idle timer must not hit SQLite for nothing
    assert all(n > 0 for n in writes)


def test_a_fast_arm_still_batches_rather_than_writing_every_trial(tmp_path, monkeypatch):
    """The timer must not turn into a per-trial write on a fast arm.

    With the clock standing still, the only thing that can flush is the record
    count -- so this pins that the timer is an *additional* condition and has
    not replaced the batching that keeps SQLite off the hot path.
    """
    m = _manifest(study_id="t_flush_fast", trials=40)
    writes: list[int] = []
    monkeypatch.setattr(R.time, "monotonic", lambda: 5000.0)  # frozen

    with TrialStore(tmp_path / "d.sqlite") as store:
        real = store.record_many

        def counting(recs):
            if recs:
                writes.append(len(recs))
            return real(recs)

        monkeypatch.setattr(store, "record_many", counting)
        runner = R.Runner(m, store)
        runner.batch_size = 1
        res = runner.run("discovery", cue_limit=1)

    assert res.completed > 25, "need to cross the record threshold at least once"
    # ceil(n/25): every flush but the last is a full 25-record batch.
    assert len(writes) == (res.completed + 24) // 25
    assert max(writes) == R.FLUSH_EVERY_RECORDS, "the record threshold stopped batching"
    assert sum(writes) == res.completed


# --- the coverage block in the artifact ------------------------------------


def _export(m, results, **kw):
    return X.build_export(
        m, results, landscape={"rows": []}, diagnostics={}, runs=[], **kw
    )


def test_coverage_is_present_even_when_the_caller_passes_nothing():
    """Absent the field a reader assumes complete, which is the wrong default
    for anything that can be interrupted."""
    m = _manifest(study_id="t_cov_default")
    payload = _export(m, [])
    cov = payload["coverage"]
    assert cov["cues_planned"] == len(m.cues)
    assert cov["cues_analyzed"] == 0
    assert cov["complete"] is False


def test_a_complete_study_says_complete_with_no_excuse():
    m = _manifest(study_id="t_cov_complete")
    results = [A.CueResult(cue=c.text, stratum="s") for c in m.cues]
    cov = _export(m, results)["coverage"]
    assert cov["complete"] is True
    assert cov["cues_analyzed"] == cov["cues_planned"]
    assert cov["reason"] == ""


def test_a_partial_study_carries_a_reason_a_reader_can_act_on():
    m = _manifest(study_id="t_cov_partial")
    results = [A.CueResult(cue=m.cues[0].text, stratum="s")]
    cov = _export(
        m,
        results,
        coverage={
            "cues_planned": len(m.cues),
            "cues_analyzed": 1,
            "complete": False,
            "reason": "run with --cue-limit 1",
        },
    )["coverage"]
    assert cov["complete"] is False
    assert "cue-limit" in cov["reason"]


def test_the_coverage_block_survives_serialization(tmp_path):
    m = _manifest(study_id="t_cov_json")
    payload = _export(m, [A.CueResult(cue=m.cues[0].text, stratum="s")])
    p = X.write_export(tmp_path / "behavior.json", payload)
    back = json.loads(p.read_text(encoding="utf-8"))
    assert back["coverage"]["cues_planned"] == len(m.cues)

# --- the reason sentence vs the store --------------------------------------
#
# `--cue-limit N` is a REQUEST. The store records it the moment the stage
# starts, so a stage that is still running -- or one that was killed -- has a
# limit in `meta` that is larger than the number of cues on disk. The reason
# sentence has to come from the cues, not from the request, or the artifact
# claims "full repeats and full block balance" for cues that have neither.


def _store_with(tmp_path, m, *, cue_limit, cues):
    """A store holding `cues` -> trial count, plus the canary rows."""
    store = TrialStore(tmp_path / f"{m.study_id}.sqlite")
    store.bind_manifest(m.frozen_hash or 'sha256:test', m.study_id)
    store.set_meta("cue_limit", str(cue_limit))
    recs = []
    for cue, n in cues.items():
        for i in range(n):
            recs.append(_trial(m.study_id, cue=cue, repeat=i, model="A" if i % 2 else "B"))
    for i in range(2):
        recs.append(_trial(m.study_id, cue=CANARY_CUE, repeat=i))
    store.record_many(recs)
    return store


def test_the_reason_counts_cues_on_disk_not_the_limit_requested(tmp_path):
    """A stage still in flight must not be described as finished."""
    m = _manifest(study_id="t_cov_inflight", n_cues=10, trials=8)
    full = m.trials_per_cue * len(m.models)  # 16 here
    # --cue-limit 4 was requested; two cues are complete and nothing else ran.
    store = _store_with(
        tmp_path, m, cue_limit=4, cues={"cue0": full, "cue1": full}
    )
    trials = [t for t in store.iter_trials(m.study_id)]
    results = [A.CueResult(cue="cue0", stratum="s"), A.CueResult(cue="cue1", stratum="s")]
    cov = CLI._coverage(m, store, results, trials)
    store.close()

    assert cov["cues_collected"] == 2
    assert cov["cues_at_full_depth"] == 2
    assert cov["cue_limit"] == 4, "the request is still recorded, as intent"
    assert "4" in cov["reason"] and "2 of the preregistered 10" in cov["reason"]
    # The old sentence said "the first 4 cues ... were collected at full
    # repeats": it must not be possible to read that out of this artifact.
    assert "the first 4 cues" not in cov["reason"]


def test_a_half_collected_cue_is_counted_as_partial_not_as_full(tmp_path):
    m = _manifest(study_id="t_cov_partial_depth", n_cues=10, trials=8)
    full = m.trials_per_cue * len(m.models)
    store = _store_with(
        tmp_path, m, cue_limit=4, cues={"cue0": full, "cue1": full // 4}
    )
    trials = [t for t in store.iter_trials(m.study_id)]
    cov = CLI._coverage(
        m, store, [A.CueResult(cue="cue0", stratum="s")], trials
    )
    store.close()

    assert cov["cues_collected"] == 2
    assert cov["cues_at_full_depth"] == 1
    assert "1 partially collected" in cov["reason"]
    assert cov["cues_analyzed"] == 1


def test_the_canary_is_not_a_cue_in_the_coverage_count(tmp_path):
    """The canary is a probe. Counting it would inflate coverage by one."""
    m = _manifest(study_id="t_cov_canary", n_cues=10, trials=8)
    full = m.trials_per_cue * len(m.models)
    store = _store_with(tmp_path, m, cue_limit=2, cues={"cue0": full})
    trials = [t for t in store.iter_trials(m.study_id)]
    assert any(t.cue == CANARY_CUE for t in trials), "the fixture must hold canary rows"
    cov = CLI._coverage(m, store, [A.CueResult(cue="cue0", stratum="s")], trials)
    store.close()
    assert cov["cues_collected"] == 1


def test_no_trials_argument_still_produces_a_reason(tmp_path):
    """Callers that have no trial list get counts of zero, never a crash and
    never a silently-complete study."""
    m = _manifest(study_id="t_cov_notrials", n_cues=10, trials=8)
    store = _store_with(tmp_path, m, cue_limit=4, cues={})
    cov = CLI._coverage(m, store, [])
    store.close()
    assert cov["cues_collected"] == 0
    assert cov["complete"] is False
    assert cov["reason"]
