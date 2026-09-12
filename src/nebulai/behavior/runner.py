"""Schedule, execute and resume a behavioral study (plan §5.4, §5.5.1, §8.3).

The runner owns four things the statistics depend on and cannot check for
themselves:

1. **Interleaving.** Trials are randomized and interleaved across models inside
   collection time blocks. It never collects all of one model's trials first —
   that would confound model with time, and the within-block permutation test
   (§6.4) would then have nothing left to protect against.
2. **Partition discipline.** Held-out Lane B frames are *unschedulable* on the
   discovery partition. The check lives here because it is the only place that
   sees both the frame's role and the arm.
3. **Budget as a runtime stop, not a plan-time guess.** §5.1: reasoning tokens
   are billed, vary per request, and are not knowable in advance, so a plan-time
   estimate × trial count bounds a quantity it cannot predict. The runner
   therefore checkpoints, halts and reports partial coverage when *actual* spend
   crosses the ceiling.
4. **Identity and drift.** The served model id is compared against the pin on
   every trial; the canary (§5.5.1) is issued once per block for the whole study
   and is excluded from every semantic metric.

Resumption is the default and is exercised by killing a run mid-schedule: the
schedule is a deterministic function of the frozen manifest, so a restart
re-derives it, subtracts what the store already holds, and issues only the
remainder.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Iterable

import numpy as np

from ..llm import BudgetError, RunBudget
from .adapters import resolve_adapter
from .adapters.base import Adapter, AdapterError, SamplerSettings
from .contract import CANARY_CUE, Manifest, ManifestError, TrialRecord, prompt_sha
from .normalize import detect, parse_associates
from .protocol import EXEMPLAR_ANSWERS
from .store import TrialStore


class RunnerError(RuntimeError):
    pass


def _stable_seed(*parts: object) -> int:
    """A seed derived from CONTENT, never from `hash()`.

    Python salts `hash()` on strings and on tuples containing them with
    `PYTHONHASHSEED`, which is random per interpreter. Deriving the schedule
    shuffle or a trial's sampler seed from `hash()` therefore makes both
    unreproducible between processes — the same frozen manifest would collect
    in a different order, and every trial would be sampled at a different seed,
    on every run. That contradicts the only thing the manifest's `seed` field
    exists for, and it is invisible inside a single process, which is why it is
    pinned by a cross-process test rather than an in-process one.
    """
    payload = "\x1f".join(repr(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


@dataclass(frozen=True)
class ScheduledTrial:
    arm: str
    cue: str
    frame_id: str
    model_key: str
    repeat: int
    block: int

    @property
    def identity(self) -> tuple[str, str, str, str, int]:
        return (self.arm, self.cue, self.frame_id, self.model_key, self.repeat)


#: Flush the pending-record buffer once this many trials have accumulated.
FLUSH_EVERY_RECORDS = 25

#: ...or once this many seconds have passed since the last flush, whichever
#: comes first. A record-count threshold alone measures durability in trials;
#: on a slow local arm the user experiences it in minutes, and a resumable
#: runner that can lose 90 minutes is resumable in name only. Kept small
#: enough to bound the loss and large enough that SQLite is not the
#: bottleneck on a fast arm.
FLUSH_EVERY_SECONDS = 60.0


def build_schedule(m: Manifest, arm: str) -> list[ScheduledTrial]:
    """The deterministic trial order for one arm.

    Deterministic in the manifest's seed, so a resumed run produces the *same*
    schedule and the store's identity index can subtract it exactly. Blocks are
    assigned by repeat index (repeat `r` → block `r % n_time_blocks`) so every
    block holds trials from every model at every cue — which is the condition
    the within-block permutation test needs to be non-degenerate.
    """
    if arm not in ("discovery", "R", "G"):
        raise RunnerError(f"unknown arm {arm!r}; expected discovery, R or G")

    if arm == "G":
        frames = [f for f in m.frames if f.role == "heldout"]
        if not frames:
            raise ManifestError(
                "arm G needs at least one held-out Lane B frame; without it the "
                "generalization question (§5.4) has no way to be asked, and a "
                "'confirmed' status would be claiming a test that never ran."
            )
    else:
        frames = [m.primary_frame]

    out: list[ScheduledTrial] = []
    for cue in m.cues:
        for r in range(m.trials_per_cue):
            frame = frames[r % len(frames)]
            block = r % m.n_time_blocks
            for model in m.models:
                out.append(
                    ScheduledTrial(arm, cue.text, frame.id, model.key, r, block)
                )
    # Shuffle inside each block only. A global shuffle would let a trial from
    # block 0 execute after one from block 3, and the block label would then be
    # a lie about collection time — which is exactly what the permutation test
    # conditions on.
    rng = np.random.default_rng(_stable_seed(m.seed, arm))
    grouped: dict[int, list[ScheduledTrial]] = {}
    for t in out:
        grouped.setdefault(t.block, []).append(t)
    ordered: list[ScheduledTrial] = []
    for b in sorted(grouped):
        g = grouped[b]
        idx = rng.permutation(len(g))
        ordered.extend(g[int(i)] for i in idx)
    return ordered


def batch_plan(
    todo: list[ScheduledTrial], batch_size: int
) -> list[list[ScheduledTrial]]:
    """Group a schedule into units that can be issued in one request.

    Batching is a change of *execution order*, so it has to say exactly what it
    is allowed to move. Two rules, both of them about what the statistics read:

    - **Never across a collection block.** The within-block permutation test
      (§6.4) conditions on the block label, so that label has to keep meaning
      "collected in this window". Reordering inside a block is free — the
      schedule already shuffles there, and the test is invariant to it — while
      moving a trial between blocks would make the label a lie.
    - **Never all of one model first.** Batches are emitted round-robin across
      the (model, frame) groups present in the block, so each model still
      appears throughout the block rather than in one contiguous run. The clump
      size is `batch_size` trials, not the whole arm.

    Grouping is by (model, frame) because those two fix the sampler settings —
    a batch shares one `stop` sequence and one token budget.

    `batch_size == 1` returns the schedule unchanged, one trial per group. That
    is the default and the only mode a paid arm ever runs in.
    """
    if batch_size <= 1:
        return [[t] for t in todo]
    by_block: dict[int, list[ScheduledTrial]] = {}
    block_order: list[int] = []
    for t in todo:
        if t.block not in by_block:
            by_block[t.block] = []
            block_order.append(t.block)
        by_block[t.block].append(t)

    out: list[list[ScheduledTrial]] = []
    for b in block_order:
        groups: dict[tuple[str, str], list[ScheduledTrial]] = {}
        key_order: list[tuple[str, str]] = []
        for t in by_block[b]:
            key = (t.model_key, t.frame_id)
            if key not in groups:
                groups[key] = []
                key_order.append(key)
            groups[key].append(t)
        chunks = {
            k: [groups[k][i : i + batch_size] for i in range(0, len(groups[k]), batch_size)]
            for k in key_order
        }
        depth = max((len(c) for c in chunks.values()), default=0)
        for i in range(depth):
            for k in key_order:
                if i < len(chunks[k]):
                    out.append(chunks[k][i])
    return out


def cue_prefix(m: Manifest, cue_limit: int | None) -> set[str] | None:
    """The first `cue_limit` cues in MANIFEST order, as a set of cue texts.

    A local arm can be too expensive to run to the end: GPT-2-XL in fp32 on a
    machine that has to page it is minutes per trial, and the preregistered
    4800-trial capability arm then does not finish in any wall clock worth
    waiting for. There are two ways to stop early and they are not equally
    honest:

    - Truncating by **trial** (`--limit`) cuts the interleaved schedule, so every
      cue ends up with a few repeats and none reaches `min_valid_trials` or the
      per-block minimum. The result is a study where no single cue can be
      tested — an answer to nothing.
    - Truncating by **cue** keeps every cue that ran at its full repeat count and
      its full block balance. The per-cue Δ̂, its within-block permutation p and
      the BY correction over the cues that ran are all exactly what they would
      have been; what is missing is cues, which is a *coverage* fact and is
      reported as one.

    So this is the supported way to run a local arm short. Manifest order, not
    schedule order, because the preregistered cue list is the thing a reader
    compares coverage against, and a seed-dependent prefix would make "the first
    40 cues" mean something different on every machine.

    `None` (and a limit at or above the cue count) means no truncation.
    """
    if cue_limit is None or cue_limit >= len(m.cues):
        return None
    if cue_limit <= 0:
        raise RunnerError(
            f"cue_limit={cue_limit} would collect no cue at all. Omit it to run "
            f"the whole preregistered set of {len(m.cues)}."
        )
    keep = {c.text for c in m.cues[:cue_limit]}
    keep.add(CANARY_CUE)  # the canary is a probe, not a cue; never truncated
    return keep


def canary_schedule(m: Manifest) -> list[ScheduledTrial]:
    """One canary trial per model per block (§5.5.1) — required either way."""
    return [
        ScheduledTrial("canary", CANARY_CUE, m.canary_frame_id, model.key, b, b)
        for b in range(m.n_time_blocks)
        for model in m.models
    ]


@dataclass
class Estimate:
    """Cost / time / storage, printed before anything is sent (§8.3)."""

    n_trials: int
    n_paid_trials: int
    est_cost_usd: float | None
    est_seconds: float
    est_bytes: int
    per_arm: dict[str, int] = field(default_factory=dict)
    unpriced_models: list[str] = field(default_factory=list)

    def render(self) -> str:
        cost = (
            f"~${self.est_cost_usd:.4f}"
            if self.est_cost_usd is not None
            else "UNPRICED (no corpus row) — the ceiling cannot be enforced by "
            "estimate alone; the runner's running-total stop still applies"
        )
        hrs = self.est_seconds / 3600.0
        return (
            f"  trials:  {self.n_trials} ({self.n_paid_trials} paid)\n"
            f"  per arm: {self.per_arm}\n"
            f"  cost:    {cost}\n"
            f"  time:    ~{hrs:.2f} h at the measured per-trial latency\n"
            f"  storage: ~{self.est_bytes / 1e6:.1f} MB of raw trials\n"
            + (
                f"  unpriced: {self.unpriced_models}\n"
                if self.unpriced_models
                else ""
            )
        )


def estimate(
    m: Manifest,
    arms: Iterable[str] = ("discovery",),
    *,
    seconds_per_trial: float = 0.35,
    bytes_per_trial: int = 1400,
) -> Estimate:
    """Size the run before it starts. Never a substitute for the runtime stop.

    `est_cost_usd` is `None` — not zero — when any paid arm has no price. A zero
    there would tell a reader the run is free, which is a different claim from
    "this run's price is unknown", and only one of them is true.
    """
    from ..llm import corpus_entry

    per_arm: dict[str, int] = {}
    n_paid = 0
    unpriced: list[str] = []
    cost = 0.0
    priced = True
    for arm in arms:
        sched = build_schedule(m, arm)
        per_arm[arm] = len(sched)
        for t in sched:
            ref = m.model(t.model_key)
            if ref.adapter != "xai":
                continue
            n_paid += 1
            spec = corpus_entry(ref.pinned)
            if spec is None:
                priced = False
                if ref.pinned not in unpriced:
                    unpriced.append(ref.pinned)
                continue
            # Prompt ~120 tokens, output capped by the manifest. Reasoning
            # tokens are NOT included: §5.1 says they are unknowable in advance,
            # and folding a guess in here would make the estimate look
            # authoritative about the part it cannot see.
            cost += (120 * spec.usd_in + m.max_output_tokens * spec.usd_out) / 1e6

    n_canary = len(canary_schedule(m))
    total = sum(per_arm.values()) + n_canary
    return Estimate(
        n_trials=total,
        n_paid_trials=n_paid,
        est_cost_usd=cost if priced else None,
        est_seconds=total * seconds_per_trial,
        est_bytes=total * bytes_per_trial,
        per_arm=per_arm | {"canary": n_canary},
        unpriced_models=unpriced,
    )


@dataclass
class RunResult:
    study_id: str
    arm: str
    attempted: int
    completed: int
    skipped_existing: int
    errors: int
    halted: str = ""
    spent_usd: float = 0.0
    not_run: dict[str, str] = field(default_factory=dict)


class Runner:
    """Executes one arm of one frozen study against one store."""

    def __init__(
        self,
        manifest: Manifest,
        store: TrialStore,
        *,
        adapters: dict[str, Adapter] | None = None,
        budget: RunBudget | None = None,
        max_retries: int = 3,
        batch_size: int = 1,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ):
        manifest.require_frozen()
        self.m = manifest
        self.store = store
        self.max_retries = max_retries
        # 1 means "one prompt per request", which is the only thing a hosted
        # arm can do and the default everywhere. A local arm may sample many
        # sequences in one forward pass; see `batch_plan` for what that is
        # allowed to reorder and what it is not.
        self.batch_size = max(1, int(batch_size))
        self.progress = progress
        self.budget = budget or RunBudget(manifest.max_cost_usd, label="behavior")
        self.not_run: dict[str, str] = {}
        self.adapters: dict[str, Adapter] = adapters or {}
        for ref in manifest.models:
            if ref.key in self.adapters:
                continue
            try:
                a = resolve_adapter(ref.adapter, pinned=ref.pinned)
            except AdapterError as exc:
                # The WHOLE message, not its first line. For the xAI arm that
                # message is the estimate-and-approve refusal of §5.5, and a
                # manifest that records only "the xAI arm has no credentials"
                # has dropped the part a reader needs to act on.
                self.not_run[ref.key] = str(exc)
                continue
            if getattr(a, "paid", False):
                setattr(a, "budget", self.budget)
            self.adapters[ref.key] = a
        store.bind_manifest(manifest.require_frozen(), manifest.study_id)

    # -- execution --------------------------------------------------------
    def run(
        self,
        arm: str,
        *,
        limit: int | None = None,
        cue_limit: int | None = None,
    ) -> RunResult:
        sched = canary_schedule(self.m) if arm == "canary" else build_schedule(self.m, arm)
        keep = cue_prefix(self.m, cue_limit)
        if keep is not None:
            sched = [t for t in sched if t.cue in keep]
        done = self.store.completed(self.m.study_id)
        todo = [
            t
            for t in sched
            if (self.m.study_id, t.arm, t.cue, t.frame_id, t.model_key, t.repeat) not in done
        ]
        skipped = len(sched) - len(todo)
        if limit is not None:
            todo = todo[:limit]

        res = RunResult(self.m.study_id, arm, len(todo), 0, skipped, 0)
        pending: list[TrialRecord] = []
        groups = batch_plan(todo, self.batch_size)
        issued = 0
        last_flush = time.monotonic()
        for group in groups:
            model_key = group[0].model_key
            adapter = self.adapters.get(model_key)
            if adapter is None:
                # The arm has no reachable backend. It is recorded as not_run
                # with a reason — never silently dropped (§5.5). `setdefault`,
                # because the FIRST failure carries the diagnosis: once the
                # adapter has been dropped every later trial of that arm would
                # otherwise overwrite the real refusal with the placeholder
                # "no adapter", and the run report would lose the only sentence
                # that says what to do about it.
                res.not_run.setdefault(
                    model_key, self.not_run.get(model_key, "no adapter")
                )
                issued += len(group)
                continue
            try:
                recs = self._group(group, adapter)
            except BudgetError as exc:
                res.halted = str(exc)
                break
            except AdapterError as exc:
                self.not_run.setdefault(model_key, str(exc))
                res.not_run.setdefault(model_key, str(exc))
                self.adapters.pop(model_key, None)
                issued += len(group)
                continue
            pending.extend(recs)
            res.completed += len(recs)
            res.errors += sum(int(bool(r.error)) for r in recs)
            issued += len(group)
            # Flush on records OR on wall clock, whichever comes first.
            # Records alone is a resumability bug on any slow arm: fp32
            # GPT-2-XL measured 3.9 min/trial on a 16 GB machine, so
            # FLUSH_EVERY_RECORDS trials is over an hour and a half of work
            # that a kill -9 throws away -- in a runner whose headline property
            # is that it resumes. The timer bounds the loss in SECONDS instead
            # of in trials, which is the unit the user actually loses. It also
            # un-sticks the progress callback below, which only fires on a
            # flush and so went silent for the same hour and a half.
            stale = bool(pending) and (
                time.monotonic() - last_flush >= FLUSH_EVERY_SECONDS
            )
            if len(pending) >= FLUSH_EVERY_RECORDS or issued >= len(todo) or stale:
                self.store.record_many(pending)
                pending = []
                last_flush = time.monotonic()
                if self.progress:
                    self.progress(
                        {"arm": arm, "done": res.completed, "total": len(todo),
                         "spent_usd": self.budget.spent_usd}
                    )
            if self.budget.spent_usd > self.m.max_cost_usd:
                res.halted = (
                    f"spend ceiling reached: ${self.budget.spent_usd:.4f} of "
                    f"${self.m.max_cost_usd:.2f}. Partial coverage is reported "
                    f"rather than a silent overrun (§5.1)."
                )
                break
        if pending:
            self.store.record_many(pending)
        res.spent_usd = self.budget.spent_usd
        res.not_run |= self.not_run
        return res

    def _group(
        self, group: list[ScheduledTrial], adapter: Adapter
    ) -> list[TrialRecord]:
        """Execute one batch. Falls back to one-at-a-time whenever the batch
        cannot be trusted to be equivalent: a single trial, an adapter with no
        batch entry point, or a batch call that failed. The fallback matters —
        a batch that dies on one malformed prompt must not lose the other
        fifteen trials, and a retry loop around the whole batch would re-sample
        the ones that were already fine."""
        if len(group) == 1:
            return [self._one(group[0], adapter)]
        batch = getattr(adapter, "complete_batch", None)
        if batch is None:
            return [self._one(t, adapter) for t in group]

        prepared = [self._prepare(t) for t in group]
        settings = prepared[0][2]
        prompts = [pr[1] for pr in prepared]
        seeds = [pr[3] for pr in prepared]
        try:
            comps = batch(prompts, settings, trial_seeds=seeds)
        except (BudgetError, AdapterError):
            raise
        except Exception:
            return [self._one(t, adapter) for t in group]
        if len(comps) != len(group):
            return [self._one(t, adapter) for t in group]
        return [
            self._finish(rec, t, c)
            for (rec, _p, _s, _seed), t, c in zip(prepared, group, comps, strict=True)
        ]

    def _prepare(
        self, t: ScheduledTrial
    ) -> tuple[TrialRecord, str, SamplerSettings, int]:
        """Everything that is decided before a token is generated: the prompt,
        the sampler settings, the trial's own seed and the empty record. Split
        out of `_one` so a batch can prepare many trials and still produce the
        identical rows a one-at-a-time run would."""
        frame = self.m.frame(t.frame_id)
        if t.arm == "discovery" and frame.role == "heldout":
            raise RunnerError(
                f"frame {frame.id!r} is held out and may not run on the "
                f"discovery partition (§5.4). Held-out frames exist so arm G "
                f"can ask a question discovery has not already answered."
            )
        prompt = frame.render(t.cue) if t.cue != CANARY_CUE else frame.template
        settings = SamplerSettings(
            temperature=self.m.temperature,
            top_p=self.m.top_p,
            max_output_tokens=self.m.max_output_tokens,
            seed=self.m.seed,
            stop=frame.stop,
        )
        trial_seed = _stable_seed(
            self.m.seed, t.arm, t.cue, t.frame_id, t.model_key, t.repeat
        )

        rec = TrialRecord(
            study_id=self.m.study_id,
            arm=t.arm,
            cue=t.cue,
            frame_id=t.frame_id,
            model_key=t.model_key,
            repeat=t.repeat,
            block=t.block,
            prompt=prompt,
            prompt_sha=prompt_sha(prompt),
            created=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        return rec, prompt, settings, trial_seed

    def _one(self, t: ScheduledTrial, adapter: Adapter) -> TrialRecord:
        rec, prompt, settings, trial_seed = self._prepare(t)

        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                c = adapter.complete(prompt, settings, trial_seed=trial_seed + attempt)
                break
            except BudgetError:
                raise
            except AdapterError:
                raise
            except Exception as exc:  # transient: back off and retry
                last = exc
                time.sleep(min(2**attempt, 8))
        else:
            rec.error = f"{type(last).__name__}: {last}"
            return rec

        return self._finish(rec, t, c)

    def _finish(
        self, rec: TrialRecord, t: ScheduledTrial, c: Any
    ) -> TrialRecord:
        """Fill one record from one completion. Shared by the single and the
        batched path so a batched run cannot drift into recording something
        subtly different from what the same trial alone would record."""
        rec.raw_output = c.text
        rec.requested_model = c.requested_model
        rec.served_model = c.served_model
        rec.response_id = c.response_id
        rec.fingerprint = c.fingerprint
        rec.latency_ms = c.latency_ms
        rec.usage = c.usage
        rec.cost_usd = c.cost_usd

        if t.cue == CANARY_CUE:
            # Canary trials are stored and diagnosed, never parsed into the
            # semantic pipeline. Marking them valid would put them into a metric
            # they are explicitly excluded from.
            rec.valid = False
            rec.invalid_reason = "canary"
            return rec

        parsed = parse_associates(c.text)
        rec.associates = parsed.associates
        rec.valid = parsed.valid
        rec.invalid_reason = parsed.reason
        rec.parser_version = parsed.parser_version
        marks = detect(
            # `rec.prompt` rather than a local: it is the string that was
            # actually sent and hashed into `prompt_sha`, so the echo marks are
            # computed against the evidence rather than against a re-render.
            parsed, cue=t.cue, prompt=rec.prompt, exemplar_answers=EXEMPLAR_ANSWERS
        )
        # The marks ride in `usage` rather than replacing anything: raw evidence
        # is append-only, and a mark is a derived observation about it.
        rec.usage = dict(rec.usage) | {
            "marks": {
                "cue_echo": marks.cue_echo,
                "exemplar_echo": marks.exemplar_echo,
                "within_trial_duplicate": marks.within_trial_duplicate,
                "prompt_copy": marks.prompt_copy,
                "distinct_types": marks.distinct_types,
            }
        }
        return rec
