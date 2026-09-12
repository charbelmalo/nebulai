"""Inter- and intra-rater agreement, per question, with CIs.

`docs/GENERATIVE-VARIANCE-PLAN.md` §6.2–§6.5 is the specification, and three of
its findings are load-bearing enough to restate here because they invert the
obvious implementation:

1. **The floor is relative to a measured human ceiling, not to 1.0.** G2
   measured two trained annotators agreeing with *each other* at κ = 0.7385
   while agreeing with the model at κ = 0.8390. A naive "κ ≥ 0.8" rule would
   discard questions the scorer answers as well as a human can. So
   :func:`floor_decision` takes `kappa_h` as an argument and the floor is
   ``min(absolute_floor, fraction * κ_H)``.

2. **The test is on the lower CI bound, never the point estimate** (§6.2 step 4,
   §6.4). κ's SE at n = 50 is ±0.15–0.20; a point estimate of 0.80 has a lower
   bound near 0.60, and passing it on the point estimate would pass a question
   whose true reliability is unknown. Every κ here therefore ships with a
   bootstrap interval, and the interval — not the estimate — is what decides.

3. **κ alone deletes the rare-feature questions** (§6.3). Skewed marginals drive
   κ toward 0 at 95% raw agreement, and "does this story contain a dream
   sequence?" is exactly such a question. PABAK, Gwet's AC1 and raw percent
   agreement are therefore computed for every question *unconditionally* and
   printed beside κ, so the prevalence paradox is visible rather than inferred.

**Missing is not zero.** A declined answer is absent from the sheet, is excluded
pairwise, and is *counted* in ``n_missing``. It never becomes a category, and it
never becomes a 0 — collapsing "declined" into "answered no" would manufacture
agreement, which is precisely the error §7 forbids.

Nothing in this module knows about a model. It takes filled sheets and returns
numbers; who or what filled a sheet is the caller's business, which is what lets
the same code compute κ_H (human–human), κ_M (human–model) and the §6.5
intra-rater check without a special case for any of them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Bumped when a returned number would change for unchanged input. Stamped into
#: every report so a stale report is identifiable rather than merely old.
AGREEMENT_VERSION = 1

#: §6.2 step 5: "the lesser of a fixed absolute minimum ... and a fraction of
#: κ_H". Both halves are defaults, not constants — a study may declare its own,
#: and the report records which were in force.
DEFAULT_ABSOLUTE_FLOOR = 0.40
DEFAULT_CEILING_FRACTION = 0.80

#: Below this, §6.2's closing paragraph drops the question as *badly specified*:
#: "that is a defect in our question, not in the scorer".
DEFAULT_KAPPA_H_MIN = 0.40


class AgreementError(RuntimeError):
    """Raised when a sheet cannot be interpreted at all."""


# --------------------------------------------------------------------------
# pairwise primitives
# --------------------------------------------------------------------------


def _confusion(a: Sequence[float], b: Sequence[float], cats: Sequence[float]) -> np.ndarray:
    idx = {float(c): i for i, c in enumerate(cats)}
    k = len(cats)
    m = np.zeros((k, k), dtype=np.float64)
    for x, y in zip(a, b, strict=True):
        try:
            m[idx[float(x)], idx[float(y)]] += 1.0
        except KeyError as exc:  # pragma: no cover - guarded by check_value
            raise AgreementError(
                f"value {exc.args[0]!r} is not on the declared scale {list(cats)!r}"
            ) from exc
    return m


def _weights(cats: Sequence[float], scheme: str) -> np.ndarray:
    """DISAGREEMENT weights: 0 on the diagonal, 1 at maximum disagreement."""
    k = len(cats)
    if k < 2:
        return np.zeros((k, k))
    v = np.asarray(cats, dtype=np.float64)
    d = np.abs(v[:, None] - v[None, :])
    span = float(v.max() - v.min()) or 1.0
    if scheme == "unweighted":
        return (d > 0).astype(np.float64)
    if scheme == "linear":
        return d / span
    if scheme == "quadratic":
        return (d / span) ** 2
    raise AgreementError(f"unknown weighting {scheme!r}")


def raw_agreement(a: Sequence[float], b: Sequence[float]) -> float:
    """Plain percent agreement. §6.3: "always shown"."""
    if not len(a):
        return float("nan")
    return float(np.mean([x == y for x, y in zip(a, b, strict=True)]))


def cohen_kappa(
    a: Sequence[float],
    b: Sequence[float],
    cats: Sequence[float],
    *,
    weights: str = "unweighted",
) -> float:
    """Cohen's κ, optionally weighted for an ordinal scale.

    Returns NaN — not 0, and not 1 — when κ is undefined because expected
    agreement is perfect (both raters gave one constant value). That case is a
    real one on rare-feature questions, and it is exactly where AC1 and raw
    agreement earn their place in the report.
    """
    if len(a) < 2:
        return float("nan")
    n = float(len(a))
    o = _confusion(a, b, cats) / n
    w = _weights(cats, weights)
    r = o.sum(axis=1)
    c = o.sum(axis=0)
    e = np.outer(r, c)
    d_o = float((w * o).sum())
    d_e = float((w * e).sum())
    if d_e == 0.0:
        return float("nan")
    return 1.0 - d_o / d_e


def pabak(a: Sequence[float], b: Sequence[float], cats: Sequence[float]) -> float:
    """Prevalence-and-bias-adjusted κ, generalized to k categories.

    ``(p_o − 1/k) / (1 − 1/k)``; for a binary scale this is the familiar
    ``2·p_o − 1``. It answers "what would κ be if the marginals were flat?",
    which is the counterfactual a reader of a skewed question actually wants.
    """
    k = len(cats)
    if k < 2 or not len(a):
        return float("nan")
    p_o = raw_agreement(a, b)
    return (p_o - 1.0 / k) / (1.0 - 1.0 / k)


def gwet_ac1(a: Sequence[float], b: Sequence[float], cats: Sequence[float]) -> float:
    """Gwet's AC1 — chance agreement estimated without κ's prevalence trap."""
    k = len(cats)
    if k < 2 or len(a) < 2:
        return float("nan")
    n = float(len(a))
    o = _confusion(a, b, cats) / n
    pi = (o.sum(axis=1) + o.sum(axis=0)) / 2.0
    p_e = float((pi * (1.0 - pi)).sum()) / (k - 1)
    p_o = raw_agreement(a, b)
    if p_e >= 1.0:
        return float("nan")
    return (p_o - p_e) / (1.0 - p_e)


# --------------------------------------------------------------------------
# Krippendorff's alpha
# --------------------------------------------------------------------------


def _delta2(cats: Sequence[float], counts: np.ndarray, level: str) -> np.ndarray:
    v = np.asarray(cats, dtype=np.float64)
    k = len(v)
    if level == "nominal":
        return (v[:, None] != v[None, :]).astype(np.float64)
    if level in ("interval", "ratio"):
        return (v[:, None] - v[None, :]) ** 2
    if level != "ordinal":
        raise AgreementError(f"unknown measurement level {level!r}")
    # Ordinal δ² uses the *observed* marginal counts: the distance between two
    # ranks is how much probability mass lies between them, so an ordinal scale
    # nobody used the middle of is treated as the coarse scale it really was.
    d = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            lo, hi = (i, j) if i <= j else (j, i)
            s = float(counts[lo : hi + 1].sum()) - (counts[lo] + counts[hi]) / 2.0
            d[i, j] = s * s
    return d


def krippendorff_alpha(
    ratings: Sequence[dict[str, float | None]],
    cats: Sequence[float],
    *,
    level: str = "ordinal",
) -> float:
    """α over any number of raters, tolerating missing values by design.

    §6.3 asks for α "where a question is ordinal or has missing values" — which
    is every question here, since a declined answer is simply absent. α is also
    what G3 reported for its repeatability check, so using it keeps our number
    methodologically comparable to the one we are replicating rather than
    merely citing.
    """
    items = sorted({k for r in ratings for k in r})
    idx = {float(c): i for i, c in enumerate(cats)}
    k = len(cats)
    if k < 2:
        return float("nan")
    o = np.zeros((k, k), dtype=np.float64)
    for it in items:
        vals = [float(r[it]) for r in ratings if r.get(it) is not None]
        m = len(vals)
        if m < 2:
            continue  # a unit rated once carries no agreement information
        # Ordered pairs of DISTINCT CELLS within the unit — not of distinct
        # values. Two raters who both answered 3 contribute to o[3,3], which is
        # where observed agreement lives; skipping equal values would delete it.
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                try:
                    o[idx[vals[i]], idx[vals[j]]] += 1.0 / (m - 1)
                except KeyError as exc:
                    raise AgreementError(
                        f"value {exc.args[0]!r} is not on the declared scale"
                    ) from exc
    n = float(o.sum())
    if n < 2:
        return float("nan")
    marg = o.sum(axis=1)
    d2 = _delta2(cats, marg, level)
    d_obs = float((o * d2).sum())
    # Expected disagreement is over all n(n-1) ordered pairs of values drawn
    # without replacement from the pooled marginal. Every δ² used here is 0 on
    # the diagonal, so the self-pair terms vanish and the outer product is the
    # exact numerator rather than an approximation of it.
    d_exp = float((np.outer(marg, marg) * d2).sum()) / (n * (n - 1))
    if d_exp == 0.0:
        return float("nan")
    return 1.0 - (d_obs / n) / d_exp


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------


@dataclass
class Interval:
    """A point estimate and its percentile bootstrap interval.

    ``n_effective`` is not decoration: on a rare-feature question many
    resamples are degenerate (every drawn item has the same answer) and κ is
    undefined in them. Those resamples are dropped, and a CI computed from 300
    usable draws out of 2000 is a different object from one computed from 2000.
    Saying so is cheaper than being asked.
    """

    estimate: float | None
    lo: float | None
    hi: float | None
    n_effective: int
    n_items: int
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": _f(self.estimate),
            "ci": [_f(self.lo), _f(self.hi)],
            "n_effective": self.n_effective,
            "n_items": self.n_items,
            "reason": self.reason,
        }


def bootstrap_ci(
    fn: Callable[[Sequence[float], Sequence[float]], float],
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> Interval:
    """Percentile bootstrap over ITEMS, which are the independent unit here.

    Resampling the (rater, answer) cells instead would break the pairing and
    invent agreement out of nothing.
    """
    n = len(a)
    est = float(fn(a, b))
    if n < 2:
        return Interval(None, None, None, 0, n, "fewer than 2 paired items")
    g = rng or np.random.default_rng(0)
    av, bv = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    draws: list[float] = []
    for _ in range(n_boot):
        i = g.integers(0, n, n)
        v = float(fn(list(av[i]), list(bv[i])))
        if math.isfinite(v):
            draws.append(v)
    if len(draws) < 20:
        return Interval(
            None if not math.isfinite(est) else est,
            None,
            None,
            len(draws),
            n,
            f"only {len(draws)} of {n_boot} bootstrap draws were defined; "
            "the interval would be a fiction",
        )
    lo = float(np.percentile(draws, 100 * alpha / 2))
    hi = float(np.percentile(draws, 100 * (1 - alpha / 2)))
    return Interval(
        None if not math.isfinite(est) else est,
        lo,
        hi,
        len(draws),
        n,
        "" if math.isfinite(est) else "point estimate undefined (no marginal variation)",
    )


# --------------------------------------------------------------------------
# per-question assembly
# --------------------------------------------------------------------------


def scale_categories(kind: str, lo: float, hi: float) -> list[float] | None:
    """The category list κ needs, or ``None`` for a continuous scale.

    `unit` questions are proportions in [0, 1]. κ, PABAK and AC1 are all
    defined over categories and would need binning to apply — and a bin
    boundary invented by the analysis is a free parameter that changes the
    answer. So those questions get α at interval level and an explicit
    ``kappa_undefined`` reason instead of a number nobody chose.
    """
    if kind in ("likert", "binary"):
        n = int(round(hi - lo)) + 1
        if n < 2 or n > 21:
            return None
        return [float(lo + i) for i in range(n)]
    return None


@dataclass
class QuestionAgreement:
    """Every number §6.2–§6.3 asks for, for one question."""

    question_id: str
    kind: str
    n_items: int
    n_paired: int
    n_missing: dict[str, int]
    categories: list[float] | None
    raw: float | None = None
    kappa: Interval | None = None
    kappa_weighted: Interval | None = None
    pabak: float | None = None
    ac1: float | None = None
    alpha: float | None = None
    prevalence: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "kind": self.kind,
            "n_items": self.n_items,
            "n_paired": self.n_paired,
            # Counted, never imputed. `missing` != 0.
            "n_missing": self.n_missing,
            "categories": self.categories,
            "raw_agreement": _f(self.raw),
            "kappa": None if self.kappa is None else self.kappa.to_dict(),
            "kappa_weighted": (
                None if self.kappa_weighted is None else self.kappa_weighted.to_dict()
            ),
            "pabak": _f(self.pabak),
            "gwet_ac1": _f(self.ac1),
            "krippendorff_alpha": _f(self.alpha),
            "prevalence": {k: _f(v) for k, v in self.prevalence.items()},
            "reason": self.reason,
        }


def _paired(
    a: dict[str, float | None], b: dict[str, float | None]
) -> tuple[list[float], list[float], list[str]]:
    items = sorted(set(a) | set(b))
    xs, ys, keep = [], [], []
    for it in items:
        x, y = a.get(it), b.get(it)
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
        keep.append(it)
    return xs, ys, keep


def question_agreement(
    question_id: str,
    kind: str,
    lo: float,
    hi: float,
    rater_a: dict[str, float | None],
    rater_b: dict[str, float | None],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    label_a: str = "a",
    label_b: str = "b",
) -> QuestionAgreement:
    """All agreement statistics for one question and one pair of raters."""
    items = sorted(set(rater_a) | set(rater_b))
    xs, ys, _ = _paired(rater_a, rater_b)
    miss = {
        label_a: sum(1 for it in items if rater_a.get(it) is None),
        label_b: sum(1 for it in items if rater_b.get(it) is None),
    }
    cats = scale_categories(kind, lo, hi)
    out = QuestionAgreement(
        question_id=question_id,
        kind=kind,
        n_items=len(items),
        n_paired=len(xs),
        n_missing=miss,
        categories=cats,
    )
    if len(xs) < 2:
        out.reason = "fewer than 2 items answered by both raters"
        return out

    out.raw = raw_agreement(xs, ys)
    level = "interval" if cats is None else "ordinal" if kind == "likert" else "nominal"
    acats = cats or sorted({*xs, *ys})
    out.alpha = _nan_none(krippendorff_alpha([rater_a, rater_b], acats, level=level))

    if cats is None:
        out.reason = (
            f"kind {kind!r} is continuous: Cohen's kappa, PABAK and AC1 are "
            "defined over categories and applying them would require a bin "
            "boundary this analysis did not choose. Krippendorff's alpha at "
            "interval level is reported instead."
        )
        return out

    g = np.random.default_rng(seed)
    out.kappa = bootstrap_ci(
        lambda p, q: cohen_kappa(p, q, cats), xs, ys, n_boot=n_boot, rng=g
    )
    if kind == "likert":
        out.kappa_weighted = bootstrap_ci(
            lambda p, q: cohen_kappa(p, q, cats, weights="quadratic"),
            xs,
            ys,
            n_boot=n_boot,
            rng=np.random.default_rng(seed + 1),
        )
    out.pabak = _nan_none(pabak(xs, ys, cats))
    out.ac1 = _nan_none(gwet_ac1(xs, ys, cats))
    # §6.3's prevalence trap is only visible if prevalence itself is on the
    # page, so the marginal of every category ships with the agreement numbers.
    n = float(len(xs) + len(ys))
    for c in cats:
        out.prevalence[str(c)] = (xs.count(c) + ys.count(c)) / n
    return out


# --------------------------------------------------------------------------
# the §6.2 decision, and the §6.5 halt
# --------------------------------------------------------------------------


@dataclass
class FloorDecision:
    question_id: str
    verdict: str  # pass | drop-low-reliability | drop-badly-specified | undecidable
    floor: float | None
    kappa_h: float | None
    kappa_m: float | None
    kappa_m_lo: float | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "verdict": self.verdict,
            "floor": _f(self.floor),
            "kappa_h": _f(self.kappa_h),
            "kappa_m": _f(self.kappa_m),
            "kappa_m_lower_ci": _f(self.kappa_m_lo),
            "detail": self.detail,
        }


def floor_decision(
    question_id: str,
    kappa_h: float | None,
    kappa_m: Interval | None,
    *,
    absolute_floor: float = DEFAULT_ABSOLUTE_FLOOR,
    ceiling_fraction: float = DEFAULT_CEILING_FRACTION,
    kappa_h_min: float = DEFAULT_KAPPA_H_MIN,
) -> FloorDecision:
    """§6.2 steps 4–5, including the "badly specified question" branch.

    The order of the two branches matters. A question whose *humans* cannot
    agree on is dropped as a defect in the question before the scorer is judged
    against it at all — otherwise a broken question would be recorded as a
    scorer failure and the fix (rewrite the question) would never be found.
    """
    if kappa_h is None or not math.isfinite(kappa_h):
        return FloorDecision(
            question_id, "undecidable", None, None,
            None if kappa_m is None else kappa_m.estimate,
            None if kappa_m is None else kappa_m.lo,
            "human-human kappa is undefined; the ceiling this floor is relative "
            "to does not exist, so no verdict is available",
        )
    if kappa_h < kappa_h_min:
        return FloorDecision(
            question_id, "drop-badly-specified", None, kappa_h,
            None if kappa_m is None else kappa_m.estimate,
            None if kappa_m is None else kappa_m.lo,
            f"kappa_H={kappa_h:.3f} < {kappa_h_min}: the annotators could not "
            "agree with each other. Sec 6.2 calls this a defect in the question, "
            "not in the scorer; the fix is to rewrite the question",
        )
    floor = min(absolute_floor, ceiling_fraction * kappa_h)
    if kappa_m is None or kappa_m.lo is None:
        return FloorDecision(
            question_id, "undecidable", floor, kappa_h,
            None if kappa_m is None else kappa_m.estimate,
            None,
            "no lower CI bound on kappa_M; Sec 6.2 step 4 tests the bound, not "
            "the point estimate, so the question cannot be decided",
        )
    ok = kappa_m.lo >= floor
    return FloorDecision(
        question_id,
        "pass" if ok else "drop-low-reliability",
        floor,
        kappa_h,
        kappa_m.estimate,
        kappa_m.lo,
        f"lower CI bound {kappa_m.lo:.3f} "
        f"{'clears' if ok else 'falls below'} floor min({absolute_floor}, "
        f"{ceiling_fraction}*{kappa_h:.3f})={floor:.3f}",
    )


@dataclass
class IntraRaterCheck:
    per_question: dict[str, float | None]
    mean_intra: float | None
    mean_inter: float | None
    halt: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_question": {k: _f(v) for k, v in self.per_question.items()},
            "mean_intra": _f(self.mean_intra),
            "mean_inter": _f(self.mean_inter),
            "halt": self.halt,
            "detail": self.detail,
        }


def intra_rater_check(
    intra: dict[str, float | None],
    inter: dict[str, float | None],
) -> IntraRaterCheck:
    """§6.5 / §3.3: if the scorer disagrees with *itself* more than two raters
    disagree with each other, the ruler is moving and the run halts.

    Compared on the questions both numbers exist for — averaging over different
    question sets would let a question missing from one side decide the halt.
    """
    shared = [
        q
        for q in sorted(set(intra) & set(inter))
        if intra[q] is not None
        and inter[q] is not None
        and math.isfinite(float(intra[q]))  # type: ignore[arg-type]
        and math.isfinite(float(inter[q]))  # type: ignore[arg-type]
    ]
    if not shared:
        return IntraRaterCheck(
            dict(intra), None, None, False,
            "no question has both an intra-rater and an inter-rater kappa; the "
            "check is NOT PERFORMED. That is not a pass",
        )
    mi = float(np.mean([float(intra[q]) for q in shared]))  # type: ignore[arg-type]
    me = float(np.mean([float(inter[q]) for q in shared]))  # type: ignore[arg-type]
    halt = mi < me
    return IntraRaterCheck(
        dict(intra), mi, me, halt,
        f"intra={mi:.4f} vs inter={me:.4f} over {len(shared)} shared questions; "
        + (
            "HALT — Sec 6.5: the instrument is unstable"
            if halt
            else "intra-rater agreement is at or above inter-rater; no halt"
        ),
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _f(x: float | None, nd: int = 6) -> float | None:
    if x is None:
        return None
    v = float(x)
    return None if not math.isfinite(v) else round(v, nd)


def _nan_none(x: float) -> float | None:
    return None if not math.isfinite(float(x)) else float(x)


def mean_ignoring_missing(values: Iterable[float | None]) -> float | None:
    vs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vs)) if vs else None
