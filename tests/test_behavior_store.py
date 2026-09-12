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
from nebulai.behavior.store import SCHEMA_VERSION, StoreLockedError, TrialStore

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


# --- one writer at a time -------------------------------------------------
# Measured 2026-09-12: four `behavior run` processes were alive on one store at
# once. Nothing was corrupted and nothing was double-billed — that is what the
# UNIQUE identity index is for — but the row count did not move for half an hour
# while each process burned ~40% of a core re-deriving the same remaining
# schedule and losing the INSERT race for every row. The failure is silent from
# inside any one process, which is why it ran that long. These tests pin the
# refusal, what counts as a dead holder, and that the lock cannot outlive a run.


def test_a_second_writer_is_refused_and_told_who_holds_the_store():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "t.sqlite"
        with TrialStore(path) as a:
            a.claim_writer(note="first")
            with TrialStore(path) as b:
                with pytest.raises(StoreLockedError) as exc:
                    b.claim_writer()
                msg = str(exc.value)
                assert str(os.getpid()) in msg       # names the holder
                assert "--force-unlock" in msg       # names the way out
                assert "same work twice" in msg or "twice the compute" in msg


def test_releasing_hands_the_store_to_the_next_writer(tmp_path):
    path = tmp_path / "t.sqlite"
    with TrialStore(path) as a:
        a.claim_writer()
        assert a.writer_lock()["live"] is True  # we are a live holder
        a.release_writer()
        with TrialStore(path) as b:
            b.claim_writer()  # must not raise
            assert b.writer_lock() is not None


def test_a_dead_holders_lock_is_not_honoured(tmp_path):
    """A pid that no longer exists never blocks a run. The alternative is a
    machine that has to be hand-unlocked after every crash, which is how
    --force-unlock stops being read and starts being pasted."""
    import json as _json

    path = tmp_path / "t.sqlite"
    with TrialStore(path) as a:
        a.claim_writer()
        raw = _json.loads(a.get_meta("writer_lock"))
        # a pid that is very unlikely to exist, on this host, with a fresh
        # heartbeat: only the liveness check can rescue this case
        raw["pid"] = 999_999_999
        raw["heartbeat"] = time.time()
        a.set_meta("writer_lock", _json.dumps(raw))
        a._owns_lock = False
        with TrialStore(path) as b:
            b.claim_writer()
            assert _json.loads(b.get_meta("writer_lock"))["pid"] == os.getpid()


def test_a_stale_heartbeat_from_another_host_expires(tmp_path):
    import json as _json

    path = tmp_path / "t.sqlite"
    with TrialStore(path) as a:
        a.set_meta(
            "writer_lock",
            _json.dumps(
                {
                    "pid": 2,
                    "host": "some-other-machine",
                    "started": 0.0,
                    # older than LOCK_STALE_S, and the pid test cannot cross hosts
                    "heartbeat": time.time() - (TrialStore.LOCK_STALE_S + 60),
                    "note": "",
                }
            ),
        )
        a.claim_writer()
        assert _json.loads(a.get_meta("writer_lock"))["host"] != "some-other-machine"


def test_a_fresh_heartbeat_from_another_host_is_honoured(tmp_path):
    import json as _json

    path = tmp_path / "t.sqlite"
    with TrialStore(path) as a:
        a.set_meta(
            "writer_lock",
            _json.dumps(
                {
                    "pid": 2,
                    "host": "some-other-machine",
                    "started": 0.0,
                    "heartbeat": time.time(),
                    "note": "",
                }
            ),
        )
        with pytest.raises(StoreLockedError):
            a.claim_writer()
        # and force takes it, because that is a decision a human made
        a.claim_writer(force=True)


def test_a_run_releases_the_lock_even_when_it_raises(tmp_path):
    """`run` must not leave a lock behind on any exit path: a crashed run that
    holds the store would make its own resume — the feature — impossible."""
    m = _manifest(study_id="t_lock")
    path = tmp_path / "t.sqlite"
    with TrialStore(path) as s:
        r = Runner(m, s)

        boom = RuntimeError("collection blew up")

        def explode(*_a, **_k):
            raise boom

        r._run = explode
        with pytest.raises(RuntimeError):
            r.run("discovery")
        assert s.writer_lock() is None, "a failed run must not keep the store"
        # and the normal path is clean too
        r2 = Runner(m, s)
        r2.run("discovery", limit=4)
        assert s.writer_lock() is None


def test_two_runners_in_sequence_still_resume(tmp_path):
    """The lock must not break the property it protects: sequential runs on one
    store still finish the schedule exactly once."""
    m = _manifest(study_id="t_lock_seq")
    with TrialStore(tmp_path / "t.sqlite") as s:
        a = Runner(m, s).run("discovery", limit=10)
        b = Runner(m, s).run("discovery")
        total = len(build_schedule(m, "discovery"))
        assert a.completed == 10
        assert a.completed + b.completed == total
        assert b.skipped_existing == 10


def test_inspect_says_a_study_is_still_collecting_before_there_is_an_analysis():
    """The refusal path needs the lock line more than the success path does.

    `behavior inspect` on a study with no `behavior.json` used to print only
    "run `behavior analyze` first" — which is the correct instruction and the
    wrong diagnosis when the reason there is no analysis is that a runner is
    still filling the store. Measured on this repo: the positive-control study
    was mid-collection and `inspect` said nothing about it, so the only way to
    find the running process was `pgrep`. The three exits of `run_inspect` (no
    artifact, one cue, whole study) all report the holder now.
    """
    import tempfile

    from nebulai.behavior.cli import _writer_note

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        assert _writer_note(str(out), "nothing-here") is None, "no store, no line"
        d = out / "s1"
        d.mkdir()
        with TrialStore(d / "trials.sqlite") as st:
            assert _writer_note(str(out), "s1") is None, "no lock, no line"
            st.claim_writer(note="behavior run --arm discovery")
            line = _writer_note(str(out), "s1")
            assert line is not None
            assert str(os.getpid()) in line
            assert "collecting now" in line
            assert "--arm discovery" in line
            st.release_writer()
            assert _writer_note(str(out), "s1") is None


def test_runs_are_per_arm_records_in_the_shape_the_page_declares(tmp_path):
    """`viewer/src/data/behavior.ts` declares `BehaviorRun`; the exporter used to
    ship the progress summary under that key and the Runs table crashed on
    `run_id.slice` at boot. The keys below are that interface, verbatim."""
    from nebulai.behavior.store import TrialStore

    st = TrialStore(tmp_path / "t.sqlite")
    sid = "s1"
    assert st.runs(sid) == []
    # an arm that never ran still gets a record — with its reason, no trials
    st.record_run(sid, "discovery", not_run={"grok": "no XAI_API_KEY in the environment"})
    (rec,) = st.runs(sid)
    assert set(rec) >= {
        "run_id", "started", "finished", "n_trials", "n_completed", "cost_usd", "halted", "not_run",
    }
    assert rec["n_trials"] == 0 and rec["started"] is None
    assert rec["cost_usd"] is None  # no price was known: not $0
    assert rec["not_run"] == {"grok": "no XAI_API_KEY in the environment"}
    assert rec["halted"] is None
