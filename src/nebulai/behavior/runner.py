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
        progress: Callable[[dict[str, Any]], None] | None = None,
    ):
        manifest.require_frozen()
        self.m = manifest
        self.store = store
        self.max_retries = max_retries
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
    def run(self, arm: str, *, limit: int | None = None) -> RunResult:
        sched = canary_schedule(self.m) if arm == "canary" else build_schedule(self.m, arm)
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
        for i, t in enumerate(todo):
            ref = self.m.model(t.model_key)
            adapter = self.adapters.get(t.model_key)
            if adapter is None:
                # The arm has no reachable backend. It is recorded as not_run
                # with a reason — never silently dropped (§5.5). `setdefault`,
                # because the FIRST failure carries the diagnosis: once the
                # adapter has been dropped every later trial of that arm would
                # otherwise overwrite the real refusal with the placeholder
                # "no adapter", and the run report would lose the only sentence
                # that says what to do about it.
                res.not_run.setdefault(
                    t.model_key, self.not_run.get(t.model_key, "no adapter")
                )
                continue
            try:
                rec = self._one(t, adapter)
            except BudgetError as exc:
                res.halted = str(exc)
                break
            except AdapterError as exc:
                self.not_run.setdefault(t.model_key, str(exc))
                res.not_run.setdefault(t.model_key, str(exc))
                self.adapters.pop(t.model_key, None)
                continue
            pending.append(rec)
            res.completed += 1
            res.errors += int(bool(rec.error))
            if len(pending) >= 25 or i == len(todo) - 1:
                self.store.record_many(pending)
                pending = []
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

    def _one(self, t: ScheduledTrial, adapter: Adapter) -> TrialRecord:
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
            parsed, cue=t.cue, prompt=prompt, exemplar_answers=EXEMPLAR_ANSWERS
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
