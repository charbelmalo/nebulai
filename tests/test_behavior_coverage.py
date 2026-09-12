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
from nebulai.behavior import export as X
from nebulai.behavior import runner as R
from nebulai.behavior.contract import CANARY_CUE
from nebulai.behavior.store import TrialStore

from test_behavior_store import _manifest  # noqa: E402


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
