"""The resumable trial store, and a resume proved by actually killing a run.

BEHAVIORAL-DIVERGENCE-PLAN.md §5.4 and §9.2. A study is hours of paid sampling,
so an interruption is not an edge case — it is the normal way a run ends at
least once. Two properties make resumption safe, and both are cheap to state and
easy to lose:

  * a completed trial has an IDENTITY (study, arm, cue, frame, model, repeat)
    and the store refuses a second row for it, so a resume cannot re-bill work
    already done or double-count it in a within-block permutation;
  * an ERRORED trial is *attempted*, not completed, so a transient provider
    failure is retried on resume instead of being frozen into the evidence.

The kill test below is a real `SIGKILL` to a real subprocess mid-schedule, not a
simulated exception. A store that only survives a clean `close()` has not been
shown to survive the case that actually happens.
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nebulai.behavior.contract import Cue, Manifest, ModelRef, TrialRecord
from nebulai.behavior.protocol import default_frames
from nebulai.behavior.runner import Runner, build_schedule
from nebulai.behavior.store import SCHEMA_VERSION, TrialStore

REPO = Path(__file__).resolve().parents[1]


def _manifest(study_id="t_store", n_cues=6, trials=8):
    m = Manifest(
        study_id=study_id,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        models=[
            ModelRef("A", "fake", "fake", label="synthetic A"),
            ModelRef("B", "fake", "fake", label="synthetic B"),
        ],
        cues=[Cue(f"cue{i}", "control_neutral", "test") for i in range(n_cues)],
        frames=default_frames(),
        trials_per_cue=trials,
        n_time_blocks=4,
        seed=7,
    )
    m.freeze("2026-09-11T00:00:00Z")
    return m


def _trial(store_id="t_store", *, cue="cue0", repeat=0, error="", model="A"):
    return TrialRecord(
        study_id=store_id,
        arm="discovery",
        cue=cue,
        frame_id="lane_a_primary",
        model_key=model,
        repeat=repeat,
        block=repeat % 4,
        prompt=f"{cue} -> ",
        prompt_sha="sha256:deadbeef",
        raw_output="a, b, c",
        associates=["a", "b", "c"],
        valid=not error,
        error=error,
        created="2026-09-11T00:00:00Z",
    )


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------


def test_a_new_store_stamps_its_schema_version(tmp_path):
    with TrialStore(tmp_path / "t.sqlite") as s:
        assert s.get_meta("schema_version") == str(SCHEMA_VERSION)


def test_the_same_identity_cannot_be_recorded_twice(tmp_path):
    with TrialStore(tmp_path / "t.sqlite") as s:
        assert s.record(_trial()) is True
        assert s.record(_trial()) is False
        assert s.count("t_store") == 1


def test_the_earlier_evidence_wins_a_resume_race(tmp_path):
    """INSERT OR IGNORE, not REPLACE: the row that was actually collected stays."""
    with TrialStore(tmp_path / "t.sqlite") as s:
        first = _trial()
        first.raw_output = "original"
        s.record(first)
        second = _trial()
        second.raw_output = "overwrite attempt"
        s.record(second)
        rows = list(s.iter_trials("t_store"))
        assert len(rows) == 1
        assert rows[0].raw_output == "original"


def test_an_errored_trial_is_attempted_not_completed(tmp_path):
    with TrialStore(tmp_path / "t.sqlite") as s:
        s.record(_trial(error="TimeoutError: boom"))
        assert s.count("t_store") == 1
        assert s.completed("t_store") == set(), "an error must be retried on resume"


def test_binding_a_second_manifest_to_one_store_is_refused(tmp_path):
    with TrialStore(tmp_path / "t.sqlite") as s:
        s.bind_manifest("sha256:aaa", "t_store")
        s.bind_manifest("sha256:aaa", "t_store")  # idempotent
        with pytest.raises(ValueError) as exc:
            s.bind_manifest("sha256:bbb", "t_store")
        assert "never pooled across protocols" in str(exc.value)


def test_spend_is_summed_and_missing_cost_is_not_zero(tmp_path):
    with TrialStore(tmp_path / "t.sqlite") as s:
        a = _trial(cue="cue0")
        a.cost_usd = 0.25
        b = _trial(cue="cue1")
        b.cost_usd = None  # free / unpriced, not "cost 0 dollars, measured"
        s.record(a)
        s.record(b)
        assert s.spent_usd("t_store") == pytest.approx(0.25)


# --------------------------------------------------------------------------
# resume, in process
# --------------------------------------------------------------------------


def test_a_resumed_run_issues_only_the_remainder(tmp_path):
    m = _manifest()
    sched = build_schedule(m, "discovery")
    with TrialStore(tmp_path / "t.sqlite") as s:
        first = Runner(m, s).run("discovery", limit=20)
        assert first.completed == 20
        second = Runner(m, s).run("discovery")
        assert second.skipped_existing == 20
        assert second.completed == len(sched) - 20
        assert s.count(m.study_id) == len(sched)
        third = Runner(m, s).run("discovery")
        assert third.completed == 0
        assert third.skipped_existing == len(sched)


def test_the_schedule_is_a_deterministic_function_of_the_manifest():
    """Resumption depends on it: a reshuffled schedule would re-issue work."""
    a = build_schedule(_manifest(), "discovery")
    b = build_schedule(_manifest(), "discovery")
    assert [t.identity for t in a] == [t.identity for t in b]


def test_identities_in_one_schedule_are_unique():
    sched = build_schedule(_manifest(), "discovery")
    ids = [t.identity for t in sched]
    assert len(set(ids)) == len(ids)


# --------------------------------------------------------------------------
# resume, after a real SIGKILL
# --------------------------------------------------------------------------


_CHILD = '''
import sys, time
from datetime import UTC, datetime
sys.path.insert(0, {repo!r})
from nebulai.behavior.contract import Cue, Manifest, ModelRef
from nebulai.behavior.protocol import default_frames
from nebulai.behavior.runner import Runner
from nebulai.behavior.store import TrialStore
from nebulai.behavior.adapters.fake import FakeAdapter

m = Manifest(
    study_id="t_kill",
    created=datetime.now(UTC).isoformat(timespec="seconds"),
    models=[ModelRef("A", "fake", "fake"), ModelRef("B", "fake", "fake")],
    cues=[Cue("cue%d" % i, "control_neutral", "test") for i in range(40)],
    frames=default_frames(),
    trials_per_cue=20,
    n_time_blocks=4,
    seed=7,
)
m.freeze("2026-09-11T00:00:00Z")


class Slow(FakeAdapter):
    def complete(self, prompt, settings, *, trial_seed=0):
        time.sleep(0.01)
        return super().complete(prompt, settings, trial_seed=trial_seed)


store = TrialStore({db!r})
runner = Runner(m, store, adapters={{"A": Slow("fake"), "B": Slow("fake")}})
print("GO", flush=True)
runner.run("discovery")
print("DONE", flush=True)
'''


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL semantics are POSIX")
def test_killing_a_run_mid_schedule_leaves_no_duplicate_on_resume(tmp_path):
    db = tmp_path / "kill.sqlite"
    script = tmp_path / "child.py"
    script.write_text(_CHILD.format(repo=str(REPO / "src"), db=str(db)), encoding="utf-8")

    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "GO"
    # Long enough for several commit batches (25 rows each, 10 ms per trial),
    # short enough that the schedule is nowhere near finished.
    time.sleep(3.0)
    assert proc.poll() is None, "the child finished before it could be killed"
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=10)
    assert proc.returncode != 0

    # What survived the kill.
    with TrialStore(db) as s:
        partial = s.count("t_kill")
        ids = list(s.completed("t_kill"))
    assert 0 < partial, "no committed batch survived the kill"
    assert len(set(ids)) == len(ids)

    # Resume in process against the same store.
    m = Manifest(
        study_id="t_kill",
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        models=[ModelRef("A", "fake", "fake"), ModelRef("B", "fake", "fake")],
        cues=[Cue(f"cue{i}", "control_neutral", "test") for i in range(40)],
        frames=default_frames(),
        trials_per_cue=20,
        n_time_blocks=4,
        seed=7,
    )
    m.freeze("2026-09-11T00:00:00Z")
    sched = build_schedule(m, "discovery")
    with TrialStore(db) as s:
        res = Runner(m, s).run("discovery")
        assert res.skipped_existing == partial
        assert res.completed == len(sched) - partial
        assert s.count("t_kill") == len(sched)
        rows = list(s.iter_trials("t_kill"))

    # The union is EXACTLY the schedule: nothing duplicated, nothing lost.
    got = {(r.arm, r.cue, r.frame_id, r.model_key, r.repeat) for r in rows}
    want = {(t.arm, t.cue, t.frame_id, t.model_key, t.repeat) for t in sched}
    assert got == want
    assert len(rows) == len(want)


# --------------------------------------------------------------------------
# reproducibility (§11 Phase 1 gate)
# --------------------------------------------------------------------------


def test_the_same_manifest_and_raw_db_reproduce_the_same_trial_content(tmp_path):
    """A resumed run must not change what an already-collected trial says."""
    m = _manifest(study_id="t_repro")
    with TrialStore(tmp_path / "a.sqlite") as s:
        Runner(m, s).run("discovery")
        a = {
            (r.cue, r.model_key, r.repeat): (r.prompt_sha, tuple(r.associates))
            for r in s.iter_trials("t_repro")
        }
    with TrialStore(tmp_path / "b.sqlite") as s:
        Runner(m, s).run("discovery", limit=30)
        Runner(m, s).run("discovery")  # finish it in two passes instead of one
        b = {
            (r.cue, r.model_key, r.repeat): (r.prompt_sha, tuple(r.associates))
            for r in s.iter_trials("t_repro")
        }
    assert a == b, "an interrupted collection must reproduce an uninterrupted one"


def test_progress_reports_what_the_store_holds(tmp_path):
    m = _manifest(study_id="t_prog")
    with TrialStore(tmp_path / "t.sqlite") as s:
        Runner(m, s).run("discovery", limit=30)
        p = s.progress("t_prog")
        assert p["total"] == 30
        assert p["by_arm"]["discovery"]["trials"] == 30
        assert json.dumps(p)  # serializable for the loopback server
