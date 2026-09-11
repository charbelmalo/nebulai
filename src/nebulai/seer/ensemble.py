"""Fan statistics over N runs of one protocol (Attractors P3).

One run of an agent is an anecdote. `seer run … --repeat 20` turns it into a
sample, and this module is what a sample is *for*: a median with an envelope
around it, rates with intervals rather than points, and a reliability number
that says whether two conditions differ by more than each condition differs
from itself.

Four decisions carry the honesty of the whole thing.

**A step is a completed turn, and a run that stopped early contributes nothing
to the steps it never reached.** Every fan entry carries its own `n` — the
number of runs that got that far — and a run that ended at turn 5 is absent
from step 9 rather than being counted as `0` there. Without that rule a fan
narrows towards the right for the single reason that fewer runs are in it, and
the narrowing reads as agreement. Events *after* the last `TURN_COMPLETED` are
a partial turn, not a turn: they are excluded and counted, because including
them would make the final step systematically smaller in every run.

**The envelope is p10/p90, not min/max.** Min and max are extreme order
statistics: their spread grows with `N` on its own, so a 50-run fan would draw
wider than a 5-run fan of the identical process. p10/p90 estimates a fixed pair
of quantiles, so two fans of different sizes are looking at the same thing. At
small `N` the two coincide, which is the correct degenerate case and not a
reason to prefer the one that misleads at large `N`.

**Rates get Wilson intervals, from the repo's single implementation.**
`backend/absorbing.py:wilson` already exists and is already tested against hand
-computed values; a second copy here would be a second thing to keep right.
`wilson(0, 0)` returns `nan`, not `(0, 0)` — no trials is not a rate of zero.

**Reliability is `Δ̂`, exactly as `BEHAVIORAL-DIVERGENCE-PLAN.md` §6.4.1 defines
it**:

    Δ̂ = D(A, B) − ½ · [ D(A₁, A₂) + D(B₁, B₂) ]

where `A₁/A₂` and `B₁/B₂` are randomized split halves of each condition's own
runs, averaged over a frozen number of draws. The reasoning transfers intact
from that plan: a contrast between two conditions that are each internally
noisy and equally far apart scores near zero, which is the right answer, and a
condition that is diffuse everywhere inflates its own within-term and subtracts
its own diffuseness out. `Δ̂` can be negative — two conditions differing *less*
than each differs from itself — and it is reported as-is, never clipped.

`D` here is an unbiased (U-statistic) MMD² under an RBF kernel whose bandwidth
is the median pairwise distance over the pooled runs, **frozen once and used for
all three terms** so the subtraction happens on one scale. The unbiased
estimator rather than the plug-in one because split halves are necessarily
smaller samples than the between-condition comparison, and the plug-in
estimator's O(1/n) bias would therefore not cancel in the subtraction — it would
be subtracted at the wrong size and masquerade as signal.

And the rule that governs all four: **a quantity that `N` is too small to
support is `MISSING` with a reason**, never a point estimate dressed up as
certainty. Three runs do not have a p10. Two runs per condition do not have a
split half. The document says which quantity could not be computed and why, and
the viewer's `N < 20 renders as an interval` rule reads `point_estimate_min_runs`
from here so the panel and the backend cannot drift to two different numbers.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..backend.absorbing import Z95, wilson
from .contract import Action, Event, EventType, Fidelity, SessionState
from .reducer import RunView, reduce_run
from .store import EventStore

#: Where an ensemble's membership record lives, under the store root. The
#: manifest is the record — it is not derivable from the logs, because "these
#: twenty runs were one experiment" is a fact about the intent behind them.
ENSEMBLES_DIRNAME = "ensembles"
ENSEMBLE_SCHEMA_VERSION = 1

#: Per-step metrics the fan can be drawn over. Each maps an `Event` to how much
#: it adds to the current step.
FAN_METRICS = ("cumulative_events", "events_per_step", "cumulative_actions")
DEFAULT_FAN_METRIC = "cumulative_events"

#: The envelope's quantiles, and the name the document reports them under.
ENVELOPE_LO_Q = 0.10
ENVELOPE_HI_Q = 0.90
ENVELOPE_NAME = "p10_p90"

#: Below this many runs there is no envelope worth drawing: two runs have no
#: p10 that is not simply one of the two runs.
MIN_RUNS_FOR_FAN = 3

#: Below this many runs per condition there is no split half: each half needs
#: at least two members for an unbiased MMD², so a condition needs four.
MIN_RUNS_PER_CONDITION_FOR_RELIABILITY = 4

#: `Δ̂` is a two-condition contrast. Three conditions is not a bigger version of
#: it, it is a different question.
CONDITIONS_FOR_RELIABILITY = 2

#: How many random split-half draws the within-condition terms average over.
#: Frozen here rather than exposed as a tuning knob, for the same reason
#: `absorbing.NULL_N` is: a draw count the caller can lower will be lowered.
SPLIT_DRAWS = 64

#: The viewer's rule ("N < 20 renders as an interval, never a point") reads
#: this number out of the document rather than hard-coding its own copy.
POINT_ESTIMATE_MIN_RUNS = 20

#: The per-run feature vector `Δ̂` is measured in. Deterministic, agent-neutral,
#: and derived from event counts and timestamps rather than from anything an
#: agent asserts about itself. A feature that is absent for ANY run in the
#: comparison is dropped from the vector for ALL of them and named in
#: `features_dropped` — substituting zero for an unobserved feature would make
#: "this agent does not report it" look like "this agent did none of it".
RELIABILITY_FEATURES = (
    "n_turns",
    "n_events",
    "n_files_changed",
    "duration_s",
    "action_edit",
    "action_verify",
    "action_inspect",
)


class EnsembleError(ValueError):
    """An ensemble could not be read or built as asked."""


def new_ensemble_id() -> str:
    return f"ens_{uuid.uuid4().hex[:16]}"


# ── membership: the manifest ────────────────────────────────────────────────


@dataclass
class Member:
    """One run's place in the fan."""

    run_id: str
    index: int
    condition: str
    protocol_id: str
    #: What `--seed-base` would have handed this run. Recorded, never applied:
    #: see `EnsembleManifest.seed_applied`.
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "index": self.index,
            "condition": self.condition,
            "protocol_id": self.protocol_id,
            "seed": self.seed,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Member":
        return Member(
            run_id=str(d["run_id"]),
            index=int(d.get("index", 0)),
            condition=str(d.get("condition", "")),
            protocol_id=str(d.get("protocol_id", "")),
            seed=d.get("seed"),
        )


@dataclass
class EnsembleManifest:
    """"These runs were one experiment" — the fact the logs cannot carry.

    Written once when `--repeat` launches, appended to as each run finishes, and
    never rewritten afterwards. The statistics are NOT stored here: they are
    recomputed from the runs' own logs on every read, the way `analysis` is, so
    a fan cannot go stale against a run that was deleted or reindexed.
    """

    ensemble_id: str
    protocol: dict[str, Any]
    n_runs_requested: int
    members: list[Member] = field(default_factory=list)
    seed_base: int | None = None
    #: No agent this repo drives accepts a seed, so `--seed-base` is recorded
    #: and never handed to one. It *is* used: it seeds the split-half draws, so
    #: the reliability number is reproducible from the manifest alone.
    seed_applied: bool = False
    budget: dict[str, Any] | None = None
    created: str = ""
    schema_version: int = ENSEMBLE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "ensemble_id": self.ensemble_id,
            "schema_version": self.schema_version,
            "protocol": self.protocol,
            "n_runs_requested": self.n_runs_requested,
            "seed_base": self.seed_base,
            "seed_applied": self.seed_applied,
            "seed_note": (
                "no agent SessionSeer drives accepts a seed, so --seed-base is "
                "recorded and never applied to the agent; it seeds the "
                "split-half draws so the reliability number is reproducible"
            ),
            "budget": self.budget,
            "members": [m.to_dict() for m in self.members],
            "created": self.created
            or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "EnsembleManifest":
        return EnsembleManifest(
            ensemble_id=str(d["ensemble_id"]),
            protocol=dict(d.get("protocol") or {}),
            n_runs_requested=int(d.get("n_runs_requested", 0)),
            members=[Member.from_dict(m) for m in (d.get("members") or [])],
            seed_base=d.get("seed_base"),
            seed_applied=bool(d.get("seed_applied", False)),
            budget=d.get("budget"),
            created=str(d.get("created", "")),
            schema_version=int(d.get("schema_version", ENSEMBLE_SCHEMA_VERSION)),
        )


def ensembles_dir(store: EventStore) -> Path:
    return Path(store.root) / ENSEMBLES_DIRNAME


def ensemble_path(store: EventStore, ensemble_id: str) -> Path:
    return ensembles_dir(store) / f"{ensemble_id}.json"


def write_manifest(store: EventStore, manifest: EnsembleManifest) -> Path:
    path = ensemble_path(store, manifest.ensemble_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n")
    return path


def read_manifest(store: EventStore, ensemble_id: str) -> EnsembleManifest | None:
    path = ensemble_path(store, ensemble_id)
    if not path.exists():
        return None
    return EnsembleManifest.from_dict(json.loads(path.read_text()))


def list_ensembles(store: EventStore, limit: int = 50) -> list[dict[str, Any]]:
    """Manifests newest first. Cheap: the file's own header, not the stats."""
    d = ensembles_dir(store)
    if not d.exists():
        return []
    rows: list[tuple[float, dict[str, Any]]] = []
    for p in d.glob("ens_*.json"):
        try:
            doc = json.loads(p.read_text())
        except ValueError:
            continue
        rows.append(
            (
                p.stat().st_mtime,
                {
                    "ensemble_id": doc.get("ensemble_id"),
                    "protocol_id": (doc.get("protocol") or {}).get("id"),
                    "n_runs_requested": doc.get("n_runs_requested"),
                    "n_members": len(doc.get("members") or []),
                    "created": doc.get("created"),
                },
            )
        )
    rows.sort(key=lambda r: r[0], reverse=True)
    return [r[1] for r in rows[:limit]]


# ── statistics ──────────────────────────────────────────────────────────────


def quantile(values: list[float], q: float) -> float:
    """Linear-interpolated quantile. One value is its own every quantile."""
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=float), q))


def fan_over(series: list[list[float]]) -> list[dict[str, Any]]:
    """Per-step median and envelope over runs of unequal length.

    The one rule that matters: at each step only the runs that REACHED it
    contribute, and `n` reports how many those were. A run that ended at step 5
    is absent from step 9, never a zero in it — a zero would drag the median
    down and widen the band in a way that reads as disagreement between runs
    when it is really just a shorter run.
    """
    reached = [s for s in series if s]
    if not reached:
        return []
    out: list[dict[str, Any]] = []
    for step in range(max(len(s) for s in reached)):
        vals = [s[step] for s in series if len(s) > step]
        out.append(
            {
                "step": step,
                "median": quantile(vals, 0.5),
                "lo": quantile(vals, ENVELOPE_LO_Q),
                "hi": quantile(vals, ENVELOPE_HI_Q),
                "n": len(vals),
            }
        )
    return out


def rate(k: int, n: int, z: float = Z95) -> dict[str, Any]:
    """One k-of-n rate with its Wilson interval.

    `n == 0` gives `p` and `ci95` of `null`, never `0.0`: nobody was asked, so
    there is no rate — which is the same rule as `wilson(0, 0)` returning nan
    rather than an interval pinned at the bottom of the scale.
    """
    if n <= 0:
        return {
            "k": int(k),
            "n": 0,
            "p": None,
            "ci95": None,
            "fidelity": Fidelity.MISSING.value,
            "missing": "no runs contributed to this rate",
        }
    lo, hi = wilson(int(k), int(n), z)
    return {
        "k": int(k),
        "n": int(n),
        "p": k / n,
        "ci95": [lo, hi],
        "fidelity": Fidelity.DETERMINISTIC.value,
    }


def _rbf(a: np.ndarray, b: np.ndarray, bandwidth: float) -> np.ndarray:
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2.0 * bandwidth * bandwidth))


def median_bandwidth(pooled: np.ndarray) -> float | None:
    """The median heuristic: the median off-diagonal pairwise distance.

    `None` when every pooled point is identical — there is then no scale on
    which to measure separation, and picking one would be inventing the units
    the answer is reported in.
    """
    if len(pooled) < 2:
        return None
    d2 = ((pooled[:, None, :] - pooled[None, :, :]) ** 2).sum(-1)
    iu = np.triu_indices(len(pooled), k=1)
    d = np.sqrt(d2[iu])
    med = float(np.median(d))
    return med if med > 0 else None


def mmd2(x: np.ndarray, y: np.ndarray, *, bandwidth: float) -> float:
    """Unbiased (U-statistic) MMD² under an RBF kernel.

    Unbiased rather than plug-in because §6.4.1's statistic *subtracts* a term
    computed on half-sized samples from one computed on full-sized ones; a
    bias that depends on sample size would survive that subtraction and be read
    as effect.
    """
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        return float("nan")
    kxx = _rbf(x, x, bandwidth)
    np.fill_diagonal(kxx, 0.0)
    kyy = _rbf(y, y, bandwidth)
    np.fill_diagonal(kyy, 0.0)
    kxy = _rbf(x, y, bandwidth)
    return float(
        kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2.0 * kxy.mean()
    )


def split_half_separation(
    x: np.ndarray, *, bandwidth: float, n_splits: int = SPLIT_DRAWS, seed: int = 0
) -> float:
    """Mean MMD² between random equal halves of one condition's own runs.

    The halves are equal in size — an odd run is dropped from each draw rather
    than making one half bigger — because two terms of different sample size
    would have different variance and the average of them would not be the
    quantity §6.4.1 subtracts.
    """
    n = len(x)
    if n < MIN_RUNS_PER_CONDITION_FOR_RELIABILITY:
        return float("nan")
    rng = np.random.default_rng(seed)
    half = n // 2
    vals = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        vals.append(
            mmd2(x[perm[:half]], x[perm[half : 2 * half]], bandwidth=bandwidth)
        )
    return float(np.mean(vals))


def delta_hat(
    a: np.ndarray,
    b: np.ndarray,
    *,
    bandwidth: float | None = None,
    n_splits: int = SPLIT_DRAWS,
    seed: int = 0,
) -> dict[str, Any]:
    """§6.4.1's normalized contrast, over two conditions' per-run vectors.

        Δ̂ = MMD²(A, B) − ½ · [ MMD²(A₁, A₂) + MMD²(B₁, B₂) ]

    Returns the three terms beside the answer, because a `Δ̂` near zero because
    both conditions are noisy and a `Δ̂` near zero because they are genuinely
    alike are different findings and only the terms tell them apart.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out: dict[str, Any] = {
        "statistic": "delta_hat",
        "formula": "delta_hat = MMD2(A,B) - 0.5*(MMD2(A1,A2) + MMD2(B1,B2))",
        "source": "BEHAVIORAL-DIVERGENCE-PLAN.md §6.4.1",
        "kernel": "rbf",
        "estimator": "mmd2_unbiased",
        "n_splits": int(n_splits),
        "seed": int(seed),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
    }
    if len(a) < MIN_RUNS_PER_CONDITION_FOR_RELIABILITY or len(b) < MIN_RUNS_PER_CONDITION_FOR_RELIABILITY:
        out.update(
            delta_hat=None,
            between=None,
            within_a=None,
            within_b=None,
            bandwidth=None,
            fidelity=Fidelity.MISSING.value,
            missing=(
                f"a split half needs at least "
                f"{MIN_RUNS_PER_CONDITION_FOR_RELIABILITY} runs per condition; "
                f"got {len(a)} and {len(b)}"
            ),
        )
        return out

    if bandwidth is None:
        bandwidth = median_bandwidth(np.vstack([a, b]))
    if not bandwidth:
        out.update(
            delta_hat=None,
            between=None,
            within_a=None,
            within_b=None,
            bandwidth=None,
            fidelity=Fidelity.MISSING.value,
            missing=(
                "every run has an identical feature vector, so there is no "
                "scale on which to measure separation"
            ),
        )
        return out

    between = mmd2(a, b, bandwidth=bandwidth)
    within_a = split_half_separation(
        a, bandwidth=bandwidth, n_splits=n_splits, seed=seed
    )
    within_b = split_half_separation(
        b, bandwidth=bandwidth, n_splits=n_splits, seed=seed + 1
    )
    out.update(
        # Negative is a real answer: the two conditions differ LESS than each
        # differs from itself. Never clipped to zero.
        delta_hat=float(between - 0.5 * (within_a + within_b)),
        between=float(between),
        within_a=float(within_a),
        within_b=float(within_b),
        bandwidth=float(bandwidth),
        fidelity=Fidelity.DETERMINISTIC.value,
    )
    return out


# ── per-run series and features ─────────────────────────────────────────────


def step_series(events: list[Event], metric: str = DEFAULT_FAN_METRIC) -> list[float]:
    """One run's value at each completed turn.

    Steps are closed by `TURN_COMPLETED`. Events after the last one are a
    partial turn and are excluded — counting them would make every run's final
    step smaller than its others for a reason that has nothing to do with the
    agent.
    """
    if metric not in FAN_METRICS:
        raise EnsembleError(f"unknown fan metric {metric!r}; have {list(FAN_METRICS)}")
    series: list[float] = []
    running = 0.0
    in_step = 0.0
    for e in events:
        if metric == "cumulative_actions":
            add = 1.0 if e.action is not None else 0.0
        else:
            add = 1.0
        running += add
        in_step += add
        if e.event_type is EventType.TURN_COMPLETED:
            series.append(running if metric != "events_per_step" else in_step)
            in_step = 0.0
    return series


def run_features(view: RunView) -> dict[str, float | None]:
    """The per-run vector `Δ̂` is measured in. `None` means unobserved."""
    duration = (
        view.ended_at - view.started_at
        if view.ended_at is not None and view.started_at is not None
        else None
    )
    return {
        "n_turns": float(view.n_turns),
        "n_events": float(view.n_events),
        "n_files_changed": float(view.n_files_changed),
        "duration_s": float(duration) if duration is not None else None,
        "action_edit": float(view.action_counts.get(Action.EDIT.value, 0)),
        "action_verify": float(view.action_counts.get(Action.VERIFY.value, 0)),
        "action_inspect": float(view.action_counts.get(Action.INSPECT.value, 0)),
    }


def feature_matrix(
    views: list[RunView],
) -> tuple[np.ndarray, list[str], dict[str, str]]:
    """Standardized per-run vectors, plus which features were dropped and why.

    Two drops, both of them refusals rather than repairs:

    * a feature absent for any run in the set is dropped for all of them —
      zero-filling would turn "this agent does not report it" into "it did none
      of it", which is the same lie as rendering `missing` as `0`;
    * a feature with no variance across the pooled set is dropped, because it
      contributes nothing to a distance and its z-score is undefined.
    """
    rows = [run_features(v) for v in views]
    dropped: dict[str, str] = {}
    keep: list[str] = []
    for name in RELIABILITY_FEATURES:
        if any(r[name] is None for r in rows):
            dropped[name] = "not observed in every run of the ensemble"
            continue
        keep.append(name)
    if not keep or not rows:
        return np.zeros((len(rows), 0)), [], dropped

    m = np.array([[float(r[name]) for name in keep] for r in rows], dtype=float)
    sd = m.std(axis=0)
    final: list[str] = []
    cols: list[np.ndarray] = []
    for i, name in enumerate(keep):
        if sd[i] <= 0:
            dropped[name] = "identical in every run; contributes no distance"
            continue
        final.append(name)
        cols.append((m[:, i] - m[:, i].mean()) / sd[i])
    if not final:
        return np.zeros((len(rows), 0)), [], dropped
    return np.stack(cols, axis=1), final, dropped


# ── the document ────────────────────────────────────────────────────────────


@dataclass
class Ensemble:
    """The fan, as the viewer reads it. Flat, snake_case, one file."""

    ensemble_id: str
    protocol: dict[str, Any]
    n_runs: int
    n_runs_requested: int
    run_ids: list[str]
    conditions: list[dict[str, Any]]
    runs: list[dict[str, Any]]
    fan_metric: str
    fan_envelope: str
    fan_fidelity: str
    fan: list[dict[str, Any]]
    rates: dict[str, dict[str, Any]]
    reliability: dict[str, Any]
    fidelity: str
    missing: dict[str, str] = field(default_factory=dict)
    seed_base: int | None = None
    seed_applied: bool = False
    budget: dict[str, Any] | None = None
    created: str = ""
    schema_version: int = ENSEMBLE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "ensemble_id": self.ensemble_id,
            "schema_version": self.schema_version,
            "created": self.created
            or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # how the protocol id was made travels with the id, always
            "protocol": self.protocol,
            "n_runs": self.n_runs,
            "n_runs_requested": self.n_runs_requested,
            "run_ids": self.run_ids,
            "conditions": self.conditions,
            "runs": self.runs,
            "seed_base": self.seed_base,
            "seed_applied": self.seed_applied,
            "fan_metric": self.fan_metric,
            "fan_envelope": self.fan_envelope,
            "fan_fidelity": self.fan_fidelity,
            "fan": self.fan,
            "rates": self.rates,
            "reliability": self.reliability,
            "budget": self.budget,
            # The viewer's "N < 20 is an interval, never a point" rule reads
            # its threshold from here so the two cannot drift apart.
            "point_estimate_min_runs": POINT_ESTIMATE_MIN_RUNS,
            "min_runs_for_fan": MIN_RUNS_FOR_FAN,
            "min_runs_per_condition_for_reliability": (
                MIN_RUNS_PER_CONDITION_FOR_RELIABILITY
            ),
            "fidelity": self.fidelity,
            # only the quantities that could NOT be computed appear here
            "missing": self.missing,
        }


def _terminal(view: RunView) -> bool:
    return view.state in (
        SessionState.COMPLETED,
        SessionState.FAILED,
        SessionState.INTERRUPTED,
    )


def build_ensemble(
    store: EventStore,
    manifest: EnsembleManifest,
    *,
    metric: str = DEFAULT_FAN_METRIC,
    n_splits: int = SPLIT_DRAWS,
) -> Ensemble:
    """Recompute the whole document from the runs' own logs.

    Nothing is cached: a run deleted since the repeat ran drops out of the fan
    and is named under `missing.absent_runs`, so `n_runs` is always the number
    of runs the statistics were really computed from rather than the number
    somebody asked for.
    """
    missing: dict[str, str] = {}
    views: list[RunView] = []
    events_by_run: dict[str, list[Event]] = {}
    members: list[Member] = []
    absent: list[str] = []

    for m in manifest.members:
        if store.get_run(m.run_id) is None:
            absent.append(m.run_id)
            continue
        evs = list(store.read(m.run_id))
        events_by_run[m.run_id] = evs
        views.append(reduce_run(m.run_id, evs))
        members.append(m)
    if absent:
        missing["absent_runs"] = (
            f"{len(absent)} run(s) named in the manifest are no longer in the "
            f"store and are excluded from every statistic: {', '.join(absent)}"
        )

    n_runs = len(members)
    run_ids = [m.run_id for m in members]

    # ── conditions ───────────────────────────────────────────────────────
    by_condition: dict[str, list[int]] = {}
    for i, m in enumerate(members):
        by_condition.setdefault(m.condition, []).append(i)
    conditions = [
        {
            "condition": name,
            "protocol_id": members[idxs[0]].protocol_id,
            "n_runs": len(idxs),
            "run_ids": [members[i].run_id for i in idxs],
        }
        for name, idxs in by_condition.items()
    ]

    # ── the fan ──────────────────────────────────────────────────────────
    fan: list[dict[str, Any]] = []
    fan_fidelity = Fidelity.DETERMINISTIC.value
    series = [step_series(events_by_run[m.run_id], metric) for m in members]
    reached = [s for s in series if s]
    if n_runs < MIN_RUNS_FOR_FAN:
        fan_fidelity = Fidelity.MISSING.value
        missing["fan"] = (
            f"an envelope needs at least {MIN_RUNS_FOR_FAN} runs; this "
            f"ensemble has {n_runs}"
        )
    elif not reached:
        fan_fidelity = Fidelity.MISSING.value
        missing["fan"] = (
            "no run in this ensemble closed a single turn, so there are no "
            "steps to draw a fan over"
        )
    else:
        # `n` per step is how many runs got that far, and is the honesty
        # channel the viewer draws the band's confidence from.
        fan = fan_over(series)

    # ── rates ────────────────────────────────────────────────────────────
    n_terminal = sum(1 for v in views if _terminal(v))
    rates: dict[str, dict[str, Any]] = {
        "completed": rate(
            sum(1 for v in views if v.state is SessionState.COMPLETED), n_runs
        ),
        "failed": rate(
            sum(1 for v in views if v.state is SessionState.FAILED), n_runs
        ),
        "verified": rate(sum(1 for v in views if v.verified), n_runs),
        # "did this run edit anything" is decidable from the event stream in
        # every capture mode, because an EDIT action is an action. Counting
        # file PATHS instead is a different question and gets its own rate
        # below — see the comment there for why conflating them lies.
        "edited": rate(
            sum(1 for v in views if v.action_counts.get(Action.EDIT.value, 0)),
            n_runs,
        ),
    }
    rates["edited"]["note"] = (
        "runs with at least one EDIT action, counted from the event stream; "
        "not a count of files touched"
    )

    # `n_files_changed` is populated only by captures that record file paths.
    # A corpus that maps an edit to an action but carries no path (ctfish, for
    # one: the board is rewritten by a shell command) would make this rate read
    # 0/N -- "no run changed a file" -- when runs demonstrably did edit. That is
    # exactly the `missing` != 0 rule, so when NOTHING reports a path and
    # SOMETHING reports an edit, the rate is MISSING with the reason, never a
    # confident zero.
    n_with_paths = sum(1 for v in views if v.n_files_changed > 0)
    n_with_edit_action = sum(
        1 for v in views if v.action_counts.get(Action.EDIT.value, 0)
    )
    if n_with_paths == 0 and n_with_edit_action > 0:
        rates["files_changed"] = {
            "k": 0,
            "n": 0,
            "p": None,
            "ci95": None,
            "fidelity": Fidelity.MISSING.value,
            "missing": (
                f"no run in this ensemble reports a changed file path, yet "
                f"{n_with_edit_action} run(s) performed an EDIT action: this "
                f"capture records edits without paths, so the number of files "
                f"touched was never observed and is not zero"
            ),
        }
    else:
        rates["files_changed"] = rate(n_with_paths, n_runs)
        rates["files_changed"]["note"] = (
            "runs that touched at least one file the capture named"
        )
    rates["completed"]["note"] = (
        "the reducer's terminal state, not a claim that the task was done"
    )
    # A run still being captured has no terminal state yet, and folding it into
    # the denominator would make a live fan look like a failing one.
    rates["completed"]["n_terminal"] = n_terminal

    decidable = [
        v.verification_after_last_edit() for v in views
    ]
    usable = [m for m in decidable if not m.absent]
    rates["verified_after_last_edit"] = rate(
        sum(1 for m in usable if m.value), len(usable)
    )
    rates["verified_after_last_edit"]["n_undecidable"] = len(decidable) - len(usable)
    rates["verified_after_last_edit"]["note"] = (
        "runs with no edits cannot answer this and are excluded from n, never "
        "counted as a failure"
    )

    # ── reliability ──────────────────────────────────────────────────────
    seed = int(manifest.seed_base or 0)
    if len(conditions) != CONDITIONS_FOR_RELIABILITY:
        reliability = {
            "statistic": "delta_hat",
            "source": "BEHAVIORAL-DIVERGENCE-PLAN.md §6.4.1",
            "delta_hat": None,
            "fidelity": Fidelity.MISSING.value,
            "missing": (
                f"delta_hat is a contrast between exactly "
                f"{CONDITIONS_FOR_RELIABILITY} conditions; this ensemble has "
                f"{len(conditions)} "
                f"({', '.join(c['condition'] for c in conditions) or 'none'})"
            ),
            "n_conditions": len(conditions),
        }
        missing["reliability"] = reliability["missing"]
    else:
        mat, used, dropped = feature_matrix(views)
        name_a, name_b = conditions[0]["condition"], conditions[1]["condition"]
        idx_a = by_condition[name_a]
        idx_b = by_condition[name_b]
        if mat.shape[1] == 0:
            reliability = {
                "statistic": "delta_hat",
                "source": "BEHAVIORAL-DIVERGENCE-PLAN.md §6.4.1",
                "delta_hat": None,
                "fidelity": Fidelity.MISSING.value,
                "missing": (
                    "no feature survived: every candidate was either "
                    "unobserved in some run or identical across all of them"
                ),
                "features_used": [],
                "features_dropped": dropped,
            }
            missing["reliability"] = reliability["missing"]
        else:
            reliability = delta_hat(
                mat[idx_a], mat[idx_b], n_splits=n_splits, seed=seed
            )
            reliability["condition_a"] = name_a
            reliability["condition_b"] = name_b
            reliability["features_used"] = used
            reliability["features_dropped"] = dropped
            if reliability.get("missing"):
                missing["reliability"] = reliability["missing"]

    computed = bool(fan) or any(
        r.get("fidelity") == Fidelity.DETERMINISTIC.value for r in rates.values()
    )
    fidelity = (
        Fidelity.DETERMINISTIC.value if computed else Fidelity.MISSING.value
    )
    if not computed:
        missing["ensemble"] = (
            "nothing could be computed: the ensemble has no runs left in the "
            "store"
        )

    return Ensemble(
        ensemble_id=manifest.ensemble_id,
        protocol=manifest.protocol,
        n_runs=n_runs,
        n_runs_requested=manifest.n_runs_requested,
        run_ids=run_ids,
        conditions=conditions,
        runs=[
            {
                "run_id": m.run_id,
                "index": m.index,
                "condition": m.condition,
                "protocol_id": m.protocol_id,
                "seed": m.seed,
                "state": v.state.value,
                "outcome": v.outcome.value,
                "n_turns": v.n_turns,
                "n_events": v.n_events,
                "n_steps": len(s),
            }
            for m, v, s in zip(members, views, series)
        ],
        fan_metric=metric,
        fan_envelope=ENVELOPE_NAME,
        fan_fidelity=fan_fidelity,
        fan=fan,
        rates=rates,
        reliability=reliability,
        fidelity=fidelity,
        missing=missing,
        seed_base=manifest.seed_base,
        seed_applied=manifest.seed_applied,
        budget=manifest.budget,
    )


__all__ = [
    "CONDITIONS_FOR_RELIABILITY",
    "DEFAULT_FAN_METRIC",
    "ENSEMBLES_DIRNAME",
    "ENSEMBLE_SCHEMA_VERSION",
    "ENVELOPE_NAME",
    "FAN_METRICS",
    "MIN_RUNS_FOR_FAN",
    "MIN_RUNS_PER_CONDITION_FOR_RELIABILITY",
    "POINT_ESTIMATE_MIN_RUNS",
    "RELIABILITY_FEATURES",
    "SPLIT_DRAWS",
    "Ensemble",
    "EnsembleError",
    "EnsembleManifest",
    "Member",
    "build_ensemble",
    "delta_hat",
    "ensemble_path",
    "ensembles_dir",
    "fan_over",
    "feature_matrix",
    "list_ensembles",
    "median_bandwidth",
    "mmd2",
    "new_ensemble_id",
    "quantile",
    "rate",
    "read_manifest",
    "run_features",
    "split_half_separation",
    "step_series",
    "wilson",
    "write_manifest",
]
