"""Batching is a change of execution order, so these tests are about what it
is NOT allowed to change.

The local arms cannot run one prompt per forward pass: GPT-2-XL in fp32 is
6.4 GB of weights, and on a machine that cannot hold it resident, a
one-trial-per-request run re-reads most of that per generated token. Sampling a
batch amortizes the read. The danger is that the shortcut quietly becomes a
different experiment — a different set of trials, a trial recorded twice,
block labels that no longer mean "collected in this window", or one model
collected before the other.

So: same trials, same identities, same rows; regrouped only inside a block; and
a batch failure must not cost the trials that were fine.
"""

import numpy as np
import pytest

from nebulai.behavior import runner as R
from nebulai.behavior.adapters.base import AdapterError, Completion, SamplerSettings
from nebulai.behavior.adapters.fake import FakeAdapter
from nebulai.behavior.contract import CANARY_CUE
from nebulai.behavior.store import TrialStore

from test_behavior_store import _manifest  # noqa: E402  (shared fixture builder)


# --- batch_plan: the reordering contract -----------------------------------


def _sched(m, arm="discovery"):
    return R.build_schedule(m, arm)


def test_batch_size_one_returns_the_schedule_untouched():
    m = _manifest(study_id="t_plan1")
    todo = _sched(m)
    plan = R.batch_plan(todo, 1)
    assert [t for g in plan for t in g] == todo
    assert all(len(g) == 1 for g in plan)


def test_every_scheduled_trial_appears_exactly_once():
    """The cheapest way for a batcher to be wrong is to drop or duplicate a
    trial, and the store's identity index would happily absorb either."""
    m = _manifest(study_id="t_plan2")
    todo = _sched(m)
    for size in (2, 5, 16, 1000):
        flat = [t for g in R.batch_plan(todo, size) for t in g]
        assert len(flat) == len(todo)
        assert sorted(map(id, flat)) == sorted(map(id, todo))


def test_a_batch_never_spans_two_collection_blocks():
    """The within-block permutation test conditions on the block label. A batch
    that straddled two blocks would be a set of trials collected together and
    labelled as if they were not."""
    m = _manifest(study_id="t_plan3")
    for g in R.batch_plan(_sched(m), 8):
        assert len({t.block for t in g}) == 1


def test_a_batch_holds_one_model_and_one_frame():
    """Both fix the sampler settings — one `stop` sequence, one token budget —
    so a mixed batch would silently apply one frame's settings to another's."""
    m = _manifest(study_id="t_plan4")
    for g in R.batch_plan(_sched(m), 8):
        assert len({(t.model_key, t.frame_id) for t in g}) == 1


def test_block_order_is_preserved():
    m = _manifest(study_id="t_plan5")
    blocks = [g[0].block for g in R.batch_plan(_sched(m), 4)]
    assert blocks == sorted(blocks)


def test_no_model_is_collected_before_the_other_inside_a_block():
    """The point of interleaving (§5.4) is that model and collection time are
    not confounded. Batching clumps at most `batch_size` trials; it must not
    turn a block into 'all of A, then all of B'."""
    m = _manifest(study_id="t_plan6")
    todo = _sched(m)
    plan = R.batch_plan(todo, 4)
    by_block: dict[int, list[str]] = {}
    for g in plan:
        by_block.setdefault(g[0].block, []).append(g[0].model_key)
    for b, keys in by_block.items():
        if len(set(keys)) < 2:
            continue
        # the first appearance of the second model must come early, not after
        # every batch of the first
        first = keys[0]
        other_at = next(i for i, k in enumerate(keys) if k != first)
        assert other_at <= 1, f"block {b}: {keys[:6]}"


# --- the run loop: same rows, batched or not -------------------------------


class _BatchSpy(FakeAdapter):
    """A fake that records the batch sizes it was asked for."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.sizes: list[int] = []

    def complete_batch(
        self, prompts, settings: SamplerSettings, *, trial_seeds
    ) -> list[Completion]:
        self.sizes.append(len(prompts))
        return [
            self.complete(p, settings, trial_seed=s)
            for p, s in zip(prompts, trial_seeds, strict=True)
        ]


def _run(tmp_path, study_id, batch_size, adapters=None, arm="discovery", limit=40):
    m = _manifest(study_id=study_id)
    with TrialStore(tmp_path / f"{study_id}.sqlite") as s:
        res = R.Runner(m, s, adapters=adapters, batch_size=batch_size).run(
            arm, limit=limit
        )
        rows = [t for t in s.iter_trials(study_id)]
    return m, res, rows


def _identity(rows):
    return sorted((r.arm, r.cue, r.frame_id, r.model_key, r.repeat) for r in rows)


def test_a_batched_run_collects_the_same_trials_as_an_unbatched_one(tmp_path):
    _m, r1, rows1 = _run(tmp_path, "t_same1", 1)
    _m, r8, rows8 = _run(tmp_path, "t_same8", 8)
    assert r1.completed == r8.completed
    assert _identity(rows1) == _identity(rows8)


def test_batching_actually_batches(tmp_path):
    spy = {"A": _BatchSpy(), "B": _BatchSpy()}
    _run(tmp_path, "t_spy", 8, adapters=spy)
    sizes = spy["A"].sizes + spy["B"].sizes
    assert sizes, "complete_batch was never called"
    assert max(sizes) > 1


def test_batch_one_never_calls_the_batch_entry_point(tmp_path):
    """Default behaviour has to stay byte-identical for every hosted arm — the
    batch path is opt-in, not a silent upgrade."""
    spy = {"A": _BatchSpy(), "B": _BatchSpy()}
    _run(tmp_path, "t_spy1", 1, adapters=spy)
    assert spy["A"].sizes == [] and spy["B"].sizes == []


def test_the_rows_a_batch_writes_are_the_rows_a_single_trial_writes(tmp_path):
    """Not just the identities — the derived columns too. `_finish` is shared
    so that a batched row cannot drift into a different parse or different
    echo marks."""
    m = _manifest(study_id="t_rows")
    todo = R.build_schedule(m, "discovery")[:8]
    with TrialStore(tmp_path / "a.sqlite") as s:
        run = R.Runner(m, s, batch_size=1)
        singles = {}
        for t in todo:
            rec = run._one(t, run.adapters[t.model_key])
            singles[(t.cue, t.frame_id, t.model_key, t.repeat)] = rec
    with TrialStore(tmp_path / "b.sqlite") as s:
        run = R.Runner(m, s, batch_size=8)
        group = [t for t in todo if (t.model_key, t.frame_id) == (todo[0].model_key, todo[0].frame_id)]
        batched = run._group(group, run.adapters[group[0].model_key])
    for t, rec in zip(group, batched, strict=True):
        ref = singles[(t.cue, t.frame_id, t.model_key, t.repeat)]
        assert rec.prompt == ref.prompt
        assert rec.prompt_sha == ref.prompt_sha
        assert rec.block == ref.block
        assert rec.parser_version == ref.parser_version
        assert "marks" in rec.usage


def test_resume_after_a_batched_run_produces_no_duplicates(tmp_path):
    """Resumption subtracts by identity, and the identity does not know about
    batching — so a half-finished batched run must resume cleanly at any size."""
    m = _manifest(study_id="t_resume")
    path = tmp_path / "r.sqlite"
    with TrialStore(path) as s:
        R.Runner(m, s, batch_size=8).run("discovery", limit=24)
    with TrialStore(path) as s:
        res = R.Runner(m, s, batch_size=1).run("discovery", limit=48)
        rows = list(s.iter_trials("t_resume"))
    ids = _identity(rows)
    assert len(ids) == len(set(ids)), "a trial was collected twice"
    assert res.skipped_existing >= 24


# --- failure, which is where a batch can lose the most ---------------------


class _BadBatch(FakeAdapter):
    def complete_batch(self, prompts, settings, *, trial_seeds):
        raise RuntimeError("the batch entry point exploded")


def test_a_broken_batch_falls_back_and_loses_nothing(tmp_path):
    """A batch that dies takes sixteen trials with it unless the runner falls
    back. Retrying the whole batch is not the answer either — it would
    re-sample the rows that were already fine."""
    ad = {"A": _BadBatch(), "B": _BadBatch()}
    _m, res, rows = _run(tmp_path, "t_bad", 8, adapters=ad)
    assert res.completed == len(rows) > 0
    assert res.errors == 0


class _ShortBatch(FakeAdapter):
    def complete_batch(self, prompts, settings, *, trial_seeds):
        return [self.complete(prompts[0], settings, trial_seed=trial_seeds[0])]


def test_a_batch_that_returns_the_wrong_count_is_not_zipped_onto_the_wrong_rows(
    tmp_path,
):
    """Silently pairing 1 completion with 8 records would attribute one model's
    text to seven trials that never ran."""
    ad = {"A": _ShortBatch(), "B": _ShortBatch()}
    _m, res, rows = _run(tmp_path, "t_short", 8, adapters=ad)
    assert res.completed == len(rows)
    assert len(_identity(rows)) == len(set(_identity(rows)))


class _PaidBatch(_BatchSpy):
    paid = True


def test_an_adapter_error_inside_a_batch_still_marks_the_arm_not_run(tmp_path):
    class _Dead(FakeAdapter):
        def complete_batch(self, prompts, settings, *, trial_seeds):
            raise AdapterError("no credentials for this arm")

        def complete(self, prompt, settings, *, trial_seed):
            raise AdapterError("no credentials for this arm")

    ad = {"A": _Dead(), "B": FakeAdapter()}
    _m, res, _rows = _run(tmp_path, "t_dead", 8, adapters=ad)
    assert "A" in res.not_run
    assert "credentials" in res.not_run["A"]


def test_the_canary_arm_batches_without_becoming_valid(tmp_path):
    m = _manifest(study_id="t_canary")
    with TrialStore(tmp_path / "c.sqlite") as s:
        R.Runner(m, s, batch_size=8).run("canary")
        rows = list(s.iter_trials("t_canary"))
    assert rows
    for r in rows:
        assert r.cue == CANARY_CUE
        assert r.valid is False or r.valid == 0
        assert r.invalid_reason == "canary"
