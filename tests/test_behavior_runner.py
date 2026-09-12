"""The runner: scheduling, budget stops, retries, resume, the canary, `not_run`.

BEHAVIORAL-DIVERGENCE-PLAN.md §5.4, §5.5.1 and §8.3. `tests/test_behavior_xai_adapter.py`
already covers the two xAI-specific cases (an unreachable paid arm is recorded
as `not_run`, and its reason survives into the result); everything else the
runner owns is pinned here against the deterministic `FakeAdapter`.

The properties below are the ones the statistics cannot check for themselves:

  * **Money stops the run, it never overruns it.** §5.1 says reasoning tokens
    are billed and unpredictable, so no plan-time estimate can bound the spend.
    The ceiling is therefore a runtime stop, and a halted run must say so in
    `RunResult.halted` and keep the trials it really did collect.
  * **A transient failure is retried; a protocol failure is not.** An
    `AdapterError` means the protocol could not be honoured, so retrying it
    would just spend money to be refused three times.
  * **Resume never re-collects a completed trial, and always re-collects an
    errored one.** The first would double a bill and double-count a trial in a
    within-block permutation; the second would freeze a provider hiccup into
    the evidence.
  * **The canary (§5.5.1) is scheduled either way** — one fixed prompt per
    model per block — stored, and excluded from every semantic metric.
  * **`not_run` carries the adapter's WHOLE refusal**, not a summary of it.

The seed test is a cross-process one on purpose: `PYTHONHASHSEED` is fixed
inside a single interpreter, so a schedule derived from `hash()` looks perfectly
deterministic until someone runs the study twice.
"""

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nebulai.behavior.adapters.base import AdapterError, Completion, SamplerSettings
from nebulai.behavior.adapters.fake import FakeAdapter
from nebulai.behavior.analyze import canary_report
from nebulai.behavior.contract import CANARY_CUE, Cue, Manifest, ManifestError, ModelRef
from nebulai.behavior.embed import HashEmbedder
from nebulai.behavior.protocol import CANARY, default_frames
from nebulai.behavior.runner import (
    Runner,
    RunnerError,
    _stable_seed,
    build_schedule,
    canary_schedule,
    estimate,
)
from nebulai.behavior.store import TrialStore
from nebulai.llm import BudgetError, RunBudget

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """The retry back-off is real (1s, 2s, 4s) and is not what is under test.

    Patched at the runner's own import site so a real `time.sleep` elsewhere is
    untouched, and so a test that asserts on retry COUNT still sees every
    attempt happen.
    """
    monkeypatch.setattr("nebulai.behavior.runner.time.sleep", lambda *_: None)


def _manifest(study_id="t_runner", *, n_cues=3, trials=4, blocks=2, **kw) -> Manifest:
    base = dict(
        study_id=study_id,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        models=[
            ModelRef("A", "fake", "fake", label="synthetic A"),
            ModelRef("B", "fake", "fake", label="synthetic B"),
        ],
        cues=[Cue(c, "control_neutral", "test") for c in ("hot", "salt", "cat")[:n_cues]],
        frames=default_frames(),
        trials_per_cue=trials,
        n_time_blocks=blocks,
        seed=7,
    )
    base.update(kw)
    m = Manifest(**base)  # type: ignore[arg-type]
    m.freeze("2026-09-11T00:00:00Z")
    return m


def _arms(**kw) -> dict[str, FakeAdapter]:
    """Two fake arms that differ only in `bias`, so they are comparable but not
    identical — the shape a real two-arm study has."""
    return {"A": FakeAdapter(bias=0, **kw), "B": FakeAdapter(bias=3, **kw)}


class _FlakyAdapter:
    """Fails `n_transient` times per trial with a plain exception, then works.

    A plain `RuntimeError` and not an `AdapterError`: the two are handled
    differently on purpose, and a test that conflated them would pass whichever
    way the runner behaved.
    """

    name = "flaky"
    pinned = "fake"
    paid = False

    def __init__(self, n_transient: int = 2):
        self.n_transient = n_transient
        self.calls: list[int] = []
        self._left = n_transient
        self._last: int | None = None

    def describe(self) -> dict:
        return {"adapter": self.name}

    def complete(self, prompt: str, settings: SamplerSettings, *, trial_seed: int) -> Completion:
        # The runner retries one trial at `seed`, `seed+1`, `seed+2`; a seed
        # that is not the previous one plus one therefore starts a new trial,
        # and the failure budget resets with it.
        if self._last is None or trial_seed != self._last + 1:
            self._left = self.n_transient
        self._last = trial_seed
        self.calls.append(trial_seed)
        if self._left > 0:
            self._left -= 1
            raise RuntimeError("connection reset by peer")
        return Completion(text="cold, warm, fire", requested_model="fake", served_model="fake")


class _RefusingAdapter:
    """Serves `n_ok` trials, then refuses the protocol for the rest.

    Models the real shape of the §5.5 failure: credentials expire, a deployment
    is retired, a sampler parameter stops being supported — mid-run, not at
    construction.
    """

    name = "refusing"
    pinned = "fake"
    paid = False
    REFUSAL = (
        "the served deployment changed from grok-4-0709 to grok-4-0915 mid-run.\n"
        "Pooling the two would make the arm a mixture of two models wearing one "
        "label, so the run stops here.\n"
        "Re-pin the manifest and start a new study revision."
    )

    def __init__(self, n_ok: int = 2):
        self.n_ok = n_ok
        self.calls = 0

    def describe(self) -> dict:
        return {"adapter": self.name}

    def complete(self, prompt: str, settings: SamplerSettings, *, trial_seed: int) -> Completion:
        self.calls += 1
        if self.calls > self.n_ok:
            raise AdapterError(self.REFUSAL)
        return Completion(text="cold, warm, fire", requested_model="fake", served_model="fake")


class _PaidAdapter:
    """A fake that actually charges a `RunBudget`, so the ceiling has something
    to stop. `paid = True` is what a real billed arm looks like to the runner."""

    name = "paid-fake"
    pinned = "fake-paid"
    paid = True

    def __init__(self, budget: RunBudget, usd_per_trial: float = 0.004):
        self.budget = budget
        self.usd = usd_per_trial
        self.calls = 0

    def describe(self) -> dict:
        return {"adapter": self.name, "paid": True}

    def complete(self, prompt: str, settings: SamplerSettings, *, trial_seed: int) -> Completion:
        self.calls += 1
        self.budget.charge(self.usd)  # raises BudgetError once the ceiling trips
        return Completion(
            text="cold, warm, fire",
            requested_model=self.pinned,
            served_model=self.pinned,
            cost_usd=self.usd,
        )


# --------------------------------------------------------------------------
# the schedule
# --------------------------------------------------------------------------


def test_the_schedule_covers_every_cue_model_and_repeat_exactly_once():
    m = _manifest(n_cues=3, trials=4)
    sched = build_schedule(m, "discovery")
    assert len(sched) == 3 * 4 * 2  # cues × repeats × models
    assert len({t.identity for t in sched}) == len(sched)


def test_blocks_hold_trials_from_every_model_at_every_cue():
    """The within-block permutation test (§6.4) conditions on the block.

    A block containing only one model's trials contributes no exchangeable
    pairs, so the test would silently lose power — or, at the limit, have
    nothing at all left to permute.
    """
    m = _manifest(n_cues=3, trials=4, blocks=2)
    per_block: dict[int, set[tuple[str, str]]] = {}
    for t in build_schedule(m, "discovery"):
        per_block.setdefault(t.block, set()).add((t.cue, t.model_key))
    assert set(per_block) == {0, 1}
    for block, pairs in per_block.items():
        assert len(pairs) == 3 * 2, f"block {block} does not cross every cue with every model"


def test_the_shuffle_never_moves_a_trial_out_of_its_block():
    """A global shuffle would make the block label a lie about collection time."""
    sched = build_schedule(_manifest(trials=8, blocks=4), "discovery")
    seen = [t.block for t in sched]
    assert seen == sorted(seen), "blocks must execute in order"
    for t in sched:
        assert t.block == t.repeat % 4


def test_discovery_never_schedules_a_held_out_frame():
    """§5.4: arm G's question is only a question if discovery has not seen it."""
    held = {f.id for f in _manifest().heldout_frames}
    assert not {t.frame_id for t in build_schedule(_manifest(), "discovery")} & held


def test_arm_g_runs_only_on_held_out_frames_and_rotates_through_them():
    m = _manifest(trials=4)
    used = [t.frame_id for t in build_schedule(m, "G")]
    assert set(used) == {"lane_b_listing", "lane_b_terse"}
    assert set(used).isdisjoint({m.primary_frame.id})


def test_arm_g_without_a_held_out_frame_is_refused_rather_than_run_on_the_primary():
    """Falling back to the primary frame would let arm G report a
    generalization test it did not run."""
    m = _manifest(frames=[f for f in default_frames() if f.role != "heldout"])
    with pytest.raises(ManifestError) as exc:
        build_schedule(m, "G")
    assert "generalization" in str(exc.value)


def test_an_unknown_arm_is_refused():
    with pytest.raises(RunnerError):
        build_schedule(_manifest(), "Z")


def test_running_a_held_out_frame_on_the_discovery_partition_is_refused(tmp_path):
    """The last line of defence, in `_one` itself: even a hand-built trial that
    pairs the discovery arm with a Lane B frame is rejected."""
    m = _manifest()
    from nebulai.behavior.runner import ScheduledTrial

    with TrialStore(tmp_path / "t.sqlite") as s:
        r = Runner(m, s, adapters=_arms())
        bad = ScheduledTrial("discovery", "hot", "lane_b_terse", "A", 0, 0)
        with pytest.raises(RunnerError) as exc:
            r._one(bad, r.adapters["A"])
    assert "held out" in str(exc.value)


# --------------------------------------------------------------------------
# reproducibility: the schedule and the sampler seeds are content-derived
# --------------------------------------------------------------------------


_REPRO_SNIPPET = """
import sys
from nebulai.behavior.contract import Cue, Manifest, ModelRef
from nebulai.behavior.protocol import default_frames
from nebulai.behavior.runner import build_schedule, _stable_seed

m = Manifest(
    study_id="t_seed", created="2026-09-11T00:00:00Z",
    models=[ModelRef("A", "fake", "fake"), ModelRef("B", "fake", "fake")],
    cues=[Cue("hot", "x"), Cue("salt", "x"), Cue("cat", "x")],
    frames=default_frames(), trials_per_cue=4, n_time_blocks=2, seed=7,
)
m.freeze("2026-09-11T00:00:00Z")
print([t.identity for t in build_schedule(m, "discovery")])
print(_stable_seed(7, "discovery", "hot", "lane_a_primary", "A", 0))
"""


def _schedule_under(hash_seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=str(REPO / "src"))
    out = subprocess.run(
        [sys.executable, "-c", _REPRO_SNIPPET], env=env, capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_the_schedule_and_the_trial_seeds_are_identical_across_processes():
    """A run repeated tomorrow must collect in the same order at the same seeds.

    SOURCE FIX: `build_schedule` seeded its shuffle from `hash(arm)` and `_one`
    derived each trial's sampler seed from `hash((seed, arm, cue, ...))`. Python
    salts `hash()` of strings with `PYTHONHASHSEED`, which is random per
    process, so the frozen manifest's `seed` controlled neither the collection
    order nor the sampling — the two things it exists to control — and the
    defect is invisible within one interpreter. Both now go through
    `runner._stable_seed`, a sha256 over the content.
    """
    assert _schedule_under("0") == _schedule_under("1")


def test_stable_seed_is_in_range_and_separates_its_parts():
    """Joining with a separator matters: without one, ("ab","c") and ("a","bc")
    would be the same trial."""
    assert 0 <= _stable_seed(7, "discovery", "hot") < 2**31
    assert _stable_seed("ab", "c") != _stable_seed("a", "bc")


def test_every_trial_in_a_schedule_gets_its_own_sampler_seed():
    """Two trials sharing a seed would return the identical completion from a
    seeded backend, which would look like a real repeat and inflate reliability."""
    m = _manifest(n_cues=3, trials=4)
    seeds = {
        _stable_seed(m.seed, t.arm, t.cue, t.frame_id, t.model_key, t.repeat)
        for t in build_schedule(m, "discovery")
    }
    assert len(seeds) == len(build_schedule(m, "discovery"))


# --------------------------------------------------------------------------
# budget: a runtime stop, with the reason recorded
# --------------------------------------------------------------------------


def test_crossing_the_ceiling_halts_the_run_and_records_why(tmp_path):
    """§5.1: partial coverage reported, never a silent overrun.

    The budget's own ceiling is set generously above the manifest's so that the
    runner's OWN post-trial check is what fires, rather than `RunBudget.charge`
    raising first — both stops exist and this one is the runner's.
    """
    m = _manifest(max_cost_usd=0.01)
    budget = RunBudget(10.0, label="behavior-test")
    budget.preflight(0.5, 100, "paid-fake")
    budget.approve()
    paid = _PaidAdapter(budget, usd_per_trial=0.004)
    with TrialStore(tmp_path / "t.sqlite") as s:
        res = Runner(m, s, adapters={"A": FakeAdapter(), "B": paid}, budget=budget).run("discovery")
        stored = s.count(m.study_id)
    assert res.halted, "a run stopped by money must say so"
    assert "spend ceiling reached" in res.halted
    assert "$0.01" in res.halted and "0.0" in res.halted
    # 3 charges take the total to $0.012 > $0.01; the check runs after the
    # trial, so the third one is collected and then the run stops.
    assert paid.calls == 3
    assert res.spent_usd == pytest.approx(0.012)
    assert 0 < res.completed < len(build_schedule(m, "discovery"))
    assert stored == res.completed, "the trials that really ran are kept, not rolled back"


def test_a_budget_error_from_the_adapter_halts_with_the_budgets_own_message(tmp_path):
    """`RunBudget` trips before the runner's own check when the two ceilings
    agree, and its message — which names the ceiling and says partial results
    are still valid — is what must reach the result."""
    m = _manifest(max_cost_usd=1.0)
    budget = RunBudget(0.005, label="behavior-test")
    budget.preflight(0.005, 2, "paid-fake")
    budget.approve()
    with TrialStore(tmp_path / "t.sqlite") as s:
        res = Runner(
            m, s, adapters={"A": FakeAdapter(), "B": _PaidAdapter(budget, 0.004)}, budget=budget
        ).run("discovery")
    assert "ceiling" in res.halted
    assert "partial results before this point are still valid" in res.halted


def test_an_unapproved_paid_call_stops_the_run_before_it_spends(tmp_path):
    """Approval is a step, not a flag. A runner that forgot to ask cannot spend.

    The refusal arrives on the FIRST paid trial, so the failure mode is a run
    that collected nothing rather than a run that collected everything and
    asked afterwards.
    """
    m = _manifest()
    budget = RunBudget(5.0, label="behavior-test")  # never preflighted, never approved
    paid = _PaidAdapter(budget, 0.004)
    with TrialStore(tmp_path / "t.sqlite") as s:
        res = Runner(m, s, adapters={"A": FakeAdapter(), "B": paid}, budget=budget).run("discovery")
    assert "before the cost estimate was approved" in res.halted
    assert res.spent_usd == 0.0
    with pytest.raises(BudgetError):
        budget.charge(0.001)


def test_a_halted_run_is_resumable_and_does_not_recollect_what_it_kept(tmp_path):
    """The stop must be an interruption, not a corruption."""
    m = _manifest(max_cost_usd=0.01)
    db = tmp_path / "t.sqlite"
    budget = RunBudget(10.0)
    budget.preflight(0.5, 100, "paid-fake")
    budget.approve()
    with TrialStore(db) as s:
        first = Runner(
            m, s, adapters={"A": FakeAdapter(), "B": _PaidAdapter(budget, 0.004)}, budget=budget
        ).run("discovery")
    with TrialStore(db) as s:
        second = Runner(m, s, adapters=_arms()).run("discovery")  # free arms, resumed
        total = s.count(m.study_id)
    assert second.skipped_existing == first.completed
    assert total == len(build_schedule(m, "discovery"))


def test_the_estimate_reports_an_unpriced_paid_arm_as_none_not_zero():
    """§8.3: "$0.0000" and "price unknown" are different claims, and only one of
    them is true for an arm with no corpus row."""
    m = _manifest(
        models=[ModelRef("A", "fake", "fake"), ModelRef("B", "xai", "grok-4-0709")]
    )
    est = estimate(m, ["discovery"])
    assert est.est_cost_usd is None
    assert est.unpriced_models == ["grok-4-0709"]
    assert est.n_paid_trials == len(build_schedule(m, "discovery")) // 2
    assert "UNPRICED" in est.render()
    assert "$0.00" not in est.render()


def test_the_estimate_counts_the_canary_trials_it_will_also_issue():
    """The canary is required either way (§5.5.1), so an estimate that omitted
    it would understate the run by `n_models × n_blocks` trials."""
    m = _manifest(n_cues=3, trials=4, blocks=2)
    est = estimate(m, ["discovery"])
    assert est.per_arm["canary"] == 2 * 2
    assert est.n_trials == len(build_schedule(m, "discovery")) + 4


# --------------------------------------------------------------------------
# retries
# --------------------------------------------------------------------------


def test_a_transient_failure_is_retried_and_the_trial_still_succeeds(tmp_path):
    m = _manifest(n_cues=1, trials=1, blocks=1)
    flaky = _FlakyAdapter(n_transient=2)
    with TrialStore(tmp_path / "t.sqlite") as s:
        res = Runner(m, s, adapters={"A": FakeAdapter(), "B": flaky}, max_retries=3).run("discovery")
        rows = [t for t in s.iter_trials(m.study_id) if t.model_key == "B"]
    assert res.errors == 0
    assert len(flaky.calls) == 3, "two failures then a success is three attempts"
    assert rows[0].raw_output == "cold, warm, fire"
    assert rows[0].valid is True


def test_each_retry_uses_a_different_sampler_seed(tmp_path):
    """Re-issuing the byte-identical seeded request is not a retry of a sampling
    failure, it is a request for the same failure."""
    m = _manifest(n_cues=1, trials=1, blocks=1)
    flaky = _FlakyAdapter(n_transient=2)
    with TrialStore(tmp_path / "t.sqlite") as s:
        Runner(m, s, adapters={"A": FakeAdapter(), "B": flaky}, max_retries=3).run("discovery")
    assert len(set(flaky.calls)) == 3
    assert flaky.calls == [flaky.calls[0], flaky.calls[0] + 1, flaky.calls[0] + 2]


def test_exhausting_the_retries_records_an_errored_trial_rather_than_dropping_it(tmp_path):
    """The row is evidence that the trial was attempted and did not come back.

    Dropping it would make the arm look like it had fewer scheduled trials than
    it did, and `n_attempted` is the denominator of the compliance-parity gate.
    """
    m = _manifest(n_cues=1, trials=1, blocks=1)
    flaky = _FlakyAdapter(n_transient=99)
    with TrialStore(tmp_path / "t.sqlite") as s:
        res = Runner(m, s, adapters={"A": FakeAdapter(), "B": flaky}, max_retries=2).run("discovery")
        rows = [t for t in s.iter_trials(m.study_id) if t.model_key == "B"]
    assert len(flaky.calls) == 2, "max_retries is a count of attempts, not of extra attempts"
    assert res.errors == 1
    assert len(rows) == 1
    assert rows[0].error.startswith("RuntimeError: ")
    assert rows[0].raw_output is None, "no output is None, never an empty answer"
    assert rows[0].valid is False


def test_an_errored_trial_is_attempted_not_completed_so_a_resume_retries_it(tmp_path):
    """The other half of the resume contract: a provider hiccup must not be
    frozen into the evidence as a permanent hole."""
    m = _manifest(n_cues=1, trials=1, blocks=1)
    db = tmp_path / "t.sqlite"
    with TrialStore(db) as s:
        Runner(m, s, adapters={"A": FakeAdapter(), "B": _FlakyAdapter(99)}, max_retries=1).run(
            "discovery"
        )
        assert s.count(m.study_id) == 2
    with TrialStore(db) as s:
        res = Runner(m, s, adapters=_arms()).run("discovery")
        rows = [t for t in s.iter_trials(m.study_id) if t.model_key == "B"]
    assert res.completed == 1, "the errored trial is re-issued"
    assert res.skipped_existing == 1, "the successful one is not"
    # `INSERT OR IGNORE` keeps the earlier row, so the store still holds exactly
    # one row per identity: the retry is recorded, not duplicated.
    assert len(rows) == 1


def test_an_adapter_error_is_not_retried(tmp_path):
    """A refused protocol is refused again. Retrying it would spend three times
    to be told the same thing, and on a paid arm that is real money."""
    m = _manifest(n_cues=1, trials=1, blocks=1)
    refusing = _RefusingAdapter(n_ok=0)
    with TrialStore(tmp_path / "t.sqlite") as s:
        Runner(m, s, adapters={"A": FakeAdapter(), "B": refusing}, max_retries=3).run("discovery")
    assert refusing.calls == 1


# --------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------


def test_a_resumed_run_produces_no_duplicate_completed_trials(tmp_path):
    m = _manifest(n_cues=3, trials=4)
    db = tmp_path / "t.sqlite"
    total = len(build_schedule(m, "discovery"))
    with TrialStore(db) as s:
        first = Runner(m, s, adapters=_arms()).run("discovery", limit=7)
    with TrialStore(db) as s:
        second = Runner(m, s, adapters=_arms()).run("discovery")
        rows = list(s.iter_trials(m.study_id))
    assert first.completed == 7
    assert second.skipped_existing == 7
    assert second.completed == total - 7
    assert len(rows) == total
    ids = [(r.arm, r.cue, r.frame_id, r.model_key, r.repeat) for r in rows]
    assert len(set(ids)) == len(ids), "identity is UNIQUE; a resume may not re-insert"


def test_a_completed_run_resumed_again_issues_nothing(tmp_path):
    """"Run it again" must be free, not a second full bill."""
    m = _manifest(n_cues=2, trials=2)
    db = tmp_path / "t.sqlite"
    with TrialStore(db) as s:
        Runner(m, s, adapters=_arms()).run("discovery")
    with TrialStore(db) as s:
        again = Runner(m, s, adapters=_arms()).run("discovery")
    assert again.attempted == 0
    assert again.completed == 0
    assert again.skipped_existing == len(build_schedule(m, "discovery"))


def test_resuming_the_same_store_under_a_different_manifest_is_refused(tmp_path):
    """§5.4: raw evidence is never pooled across protocols.

    The runner binds the store to the frozen hash, so "resuming" into a store
    collected under different thresholds fails at bind time rather than
    producing a mixed-protocol dataset nobody can separate afterwards.
    """
    db = tmp_path / "t.sqlite"
    with TrialStore(db) as s:
        Runner(_manifest(), s, adapters=_arms()).run("discovery", limit=2)
    other = _manifest(temperature=0.5)  # same study_id, different protocol
    with TrialStore(db) as s, pytest.raises(ValueError) as exc:
        Runner(other, s, adapters=_arms())
    assert "never pooled across protocols" in str(exc.value)


def test_arms_are_resumed_independently(tmp_path):
    """Arm R and arm G share a store and a study id; finishing one must not
    make the other look collected."""
    m = _manifest(n_cues=2, trials=2)
    db = tmp_path / "t.sqlite"
    with TrialStore(db) as s:
        Runner(m, s, adapters=_arms()).run("discovery")
        g = Runner(m, s, adapters=_arms()).run("G")
    assert g.skipped_existing == 0
    assert g.completed == len(build_schedule(m, "G"))


# --------------------------------------------------------------------------
# the canary (§5.5.1)
# --------------------------------------------------------------------------


def test_the_canary_is_one_trial_per_model_per_block():
    """Required either way, fingerprint or no fingerprint. One per block is what
    makes the outputs a time series rather than a single sample."""
    m = _manifest(blocks=4)
    sched = canary_schedule(m)
    assert len(sched) == 4 * 2
    assert {(t.model_key, t.block) for t in sched} == {
        (k, b) for k in ("A", "B") for b in range(4)
    }
    assert {t.cue for t in sched} == {CANARY_CUE}
    assert {t.arm for t in sched} == {"canary"}


def test_the_canary_prompt_is_the_frame_verbatim_with_no_cue_substituted(tmp_path):
    """A canary whose text varied would measure the variation, not the drift."""
    m = _manifest(blocks=2)
    with TrialStore(tmp_path / "t.sqlite") as s:
        Runner(m, s, adapters=_arms()).run("canary")
        rows = list(s.iter_trials(m.study_id, arm="canary"))
    assert rows, "the canary arm must actually issue trials"
    assert {r.prompt for r in rows} == {CANARY.template}
    assert CANARY_CUE not in CANARY.template


def test_canary_results_are_stored_but_excluded_from_every_semantic_metric(tmp_path):
    """Marked invalid with the reason "canary": a valid row would be parsed into
    the association distribution it is explicitly excluded from."""
    m = _manifest(blocks=2)
    with TrialStore(tmp_path / "t.sqlite") as s:
        res = Runner(m, s, adapters=_arms()).run("canary")
        rows = list(s.iter_trials(m.study_id, arm="canary"))
    assert res.completed == len(canary_schedule(m))
    for r in rows:
        assert r.raw_output, "the output is kept — it is the drift evidence"
        assert r.valid is False
        assert r.invalid_reason == "canary"
        assert r.associates == [], "never parsed into associates"


def test_the_canary_never_appears_in_a_semantic_arms_schedule():
    for arm in ("discovery", "R", "G"):
        assert CANARY_CUE not in {t.cue for t in build_schedule(_manifest(), arm)}


def test_the_recorded_canary_feeds_the_drift_report(tmp_path):
    """Wired end to end: the rows the runner writes are the rows §5.5.1's
    diagnostic reads, and with two blocks it yields one consecutive-block
    cosine per model."""
    m = _manifest(blocks=2)
    with TrialStore(tmp_path / "t.sqlite") as s:
        Runner(m, s, adapters=_arms()).run("canary")
        report = canary_report(s.iter_trials(m.study_id), HashEmbedder())
    assert report["status"] == "measured"
    assert set(report["drift"]) == {"A", "B"}
    for d in report["drift"].values():
        assert len(d["consecutive_block_cosine"]) == 1  # 2 blocks → 1 consecutive pair
        assert isinstance(d["flag"], bool)


def test_a_study_with_no_canary_trials_reports_missing_rather_than_no_drift(tmp_path):
    """Absence of the probe is not absence of drift, and the diagnostic must not
    let a reader confuse the two."""
    m = _manifest()
    with TrialStore(tmp_path / "t.sqlite") as s:
        Runner(m, s, adapters=_arms()).run("discovery", limit=2)
        report = canary_report(s.iter_trials(m.study_id), HashEmbedder())
    assert report["status"] == "missing"
    assert "no canary trials" in report["reason"]


# --------------------------------------------------------------------------
# not_run carries the whole refusal
# --------------------------------------------------------------------------


def test_an_unresolvable_adapter_records_the_full_refusal_text(tmp_path):
    """§5.5: a missing route is refused, never routed around — and the record
    must carry the sentence that says what to do about it, not a summary."""
    m = _manifest(models=[ModelRef("A", "fake", "fake"), ModelRef("B", "nope", "whatever")])
    with TrialStore(tmp_path / "t.sqlite") as s:
        res = Runner(m, s).run("discovery")
        rows = list(s.iter_trials(m.study_id))
    msg = res.not_run["B"]
    assert "unknown adapter 'nope'" in msg
    assert "does not substitute a reachable backend" in msg
    assert msg.count(".") >= 2, "the whole message, not its first line"
    assert {r.model_key for r in rows} == {"A"}, "no row may be invented for an arm that never ran"


def test_a_mid_run_refusal_drops_the_arm_and_keeps_the_adapters_own_reason(tmp_path):
    """The `setdefault` that matters: the FIRST failure carries the diagnosis.

    Once the adapter is dropped, every remaining trial of that arm would
    otherwise overwrite the real refusal with the placeholder "no adapter", and
    the run report would lose the only sentence that says what happened.
    """
    m = _manifest(n_cues=3, trials=4)
    refusing = _RefusingAdapter(n_ok=2)
    with TrialStore(tmp_path / "t.sqlite") as s:
        res = Runner(m, s, adapters={"A": FakeAdapter(), "B": refusing}).run("discovery")
        b_rows = [t for t in s.iter_trials(m.study_id) if t.model_key == "B"]
    assert res.not_run["B"] == _RefusingAdapter.REFUSAL
    assert "no adapter" not in res.not_run["B"]
    assert len(b_rows) == 2, "the trials served before the refusal are real evidence"
    assert refusing.calls == 3, "the arm is dropped on its first refusal, not retried per trial"


def test_the_run_continues_on_the_reachable_arm_after_one_is_dropped(tmp_path):
    """A one-armed result is not a divergence result, but it IS the evidence
    that the other arm never ran — so it is collected and labelled, not thrown
    away."""
    m = _manifest(n_cues=2, trials=2)
    with TrialStore(tmp_path / "t.sqlite") as s:
        res = Runner(m, s, adapters={"A": FakeAdapter(), "B": _RefusingAdapter(n_ok=0)}).run(
            "discovery"
        )
        a_rows = [t for t in s.iter_trials(m.study_id) if t.model_key == "A"]
    assert len(a_rows) == 2 * 2
    assert res.halted == "", "a dropped arm is not a halt; money and protocol are"
    assert set(res.not_run) == {"B"}


def test_an_unfrozen_manifest_cannot_be_run_at_all(tmp_path):
    """The runner requires the freeze before it looks at a single trial."""
    m = Manifest(
        study_id="draft",
        created="2026-09-11T00:00:00Z",
        models=[ModelRef("A", "fake", "fake"), ModelRef("B", "fake", "fake")],
        cues=[Cue("hot", "control_neutral")],
        frames=default_frames(),
    )
    with TrialStore(tmp_path / "t.sqlite") as s, pytest.raises(ManifestError):
        Runner(m, s, adapters=_arms())


# --------------------------------------------------------------------------
# what lands in the record
# --------------------------------------------------------------------------


def test_a_completed_trial_carries_its_prompt_hash_and_provenance(tmp_path):
    """§6.1: the row must be auditable on its own, without the manifest."""
    m = _manifest(n_cues=1, trials=1, blocks=1)
    with TrialStore(tmp_path / "t.sqlite") as s:
        Runner(m, s, adapters=_arms()).run("discovery")
        row = next(iter(s.iter_trials(m.study_id)))
    assert row.prompt_sha.startswith("sha256:")
    assert row.requested_model == "fake" and row.served_model == "fake"
    assert row.response_id.startswith("fake-")
    assert row.created.endswith("+00:00"), "timestamps are UTC and say so"
    assert row.cost_usd == 0.0, "a free arm really did cost nothing — that is measured"


def test_the_degeneracy_marks_ride_alongside_the_raw_output_without_replacing_it(tmp_path):
    """§6.2 step 5: mark, never replace. The echoing arm's output is stored
    verbatim and the marks are a derived observation about it."""
    m = _manifest(n_cues=1, trials=2, blocks=1)
    with TrialStore(tmp_path / "t.sqlite") as s:
        Runner(m, s, adapters={"A": FakeAdapter(), "B": FakeAdapter(degenerate=True)}).run(
            "discovery"
        )
        rows = {t.model_key: t for t in s.iter_trials(m.study_id)}
    b = rows["B"]
    assert b.raw_output == "hot, hot, hot"
    assert b.usage["marks"]["cue_echo"] is True
    assert b.usage["marks"]["within_trial_duplicate"] is True
    assert rows["A"].usage["marks"]["cue_echo"] is False


def test_an_unparseable_output_is_stored_invalid_with_its_reason(tmp_path):
    """The compliance-parity gate of §6.5.1 needs the failures counted, so an
    unparseable trial is a row with `valid = False`, not a missing row."""
    m = _manifest(n_cues=2, trials=4, blocks=2)
    with TrialStore(tmp_path / "t.sqlite") as s:
        Runner(
            m, s, adapters={"A": FakeAdapter(), "B": FakeAdapter(parse_failure_rate=1.0)}
        ).run("discovery")
        rows = [t for t in s.iter_trials(m.study_id) if t.model_key == "B"]
    assert len(rows) == 2 * 4
    assert all(not r.valid for r in rows)
    assert all(r.invalid_reason for r in rows), "an invalid trial always says why"
    assert all(r.raw_output for r in rows), "the raw refusal text is the evidence"


def test_progress_is_reported_with_the_running_spend(tmp_path):
    """The server's progress stream is this callback; it must carry spend, not
    only counts, so a watching human can see money move."""
    m = _manifest(n_cues=3, trials=4)
    seen: list[dict] = []
    with TrialStore(tmp_path / "t.sqlite") as s:
        Runner(m, s, adapters=_arms(), progress=seen.append).run("discovery")
    assert seen
    assert set(seen[0]) == {"arm", "done", "total", "spent_usd"}
    assert seen[-1]["done"] == seen[-1]["total"] == len(build_schedule(m, "discovery"))
    assert all(p["spent_usd"] == 0.0 for p in seen)
