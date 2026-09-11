"""The frozen statistical contract of plan §6.4–§6.7, as pure numpy.

Nothing in this module touches a model, a network, or a database: it takes
arrays and returns numbers, so every claim the study makes is testable against
synthetic data with a known answer. That is deliberate — §6.7's calibration
controls (A/A pseudo-comparisons, shuffled labels, the p-floor) are only
meaningful if the estimator they exercise is the exact one the study uses.

Three rules are enforced here rather than left to the caller:

* **`Δ̂` is the rank key, raw MMD² never is** (§6.4.1). `mmd2` is public because
  the artifact displays it beside `Δ̂`, but `delta_hat` is what `analyze.py`
  sorts and gates on.
* **`Δ̂` may be negative and is returned as-is** — clipping to zero would erase
  the "these two models differ less than each differs from itself" finding.
* **Permutations stay inside collection time blocks** (§6.4, §6.7.1), and
  :func:`p_floor` reports what that costs, so a study cannot discover after the
  fact that confirmation was combinatorially unreachable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Kernel and MMD
# ---------------------------------------------------------------------------


def median_bandwidth(X: np.ndarray, Y: np.ndarray | None = None) -> float:
    """Median-heuristic bandwidth, calibrated ONCE in Phase 0 and then frozen.

    Re-deriving it per cue would make the kernel a function of the data it
    judges, which turns an effect size into a moving target — the manifest holds
    `mmd_bandwidth` for exactly this reason.
    """
    Z = X if Y is None else np.vstack([X, Y])
    if len(Z) < 2:
        return 1.0
    d2 = _sqdist(Z, Z)
    iu = np.triu_indices(len(Z), k=1)
    med = float(np.median(d2[iu]))
    return math.sqrt(med / 2.0) if med > 0 else 1.0


def _sqdist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    xx = (X * X).sum(1)[:, None]
    yy = (Y * Y).sum(1)[None, :]
    return np.maximum(xx + yy - 2.0 * (X @ Y.T), 0.0)


def _gram(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> np.ndarray:
    return np.exp(-_sqdist(X, Y) / (2.0 * bandwidth * bandwidth))


def mmd2(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> float:
    """Unbiased MMD² with a Gaussian kernel.

    Unbiased (diagonal-excluded) rather than biased-plug-in because the biased
    form is strictly positive even for two samples from the same distribution,
    which would give every A/A control a spurious floor and make `Δ̂`'s
    subtraction an unfair fight between unequal sample sizes.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    m, n = len(X), len(Y)
    if m < 2 or n < 2:
        return float("nan")
    Kxx = _gram(X, X, bandwidth)
    Kyy = _gram(Y, Y, bandwidth)
    Kxy = _gram(X, Y, bandwidth)
    sxx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    syy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    sxy = Kxy.mean()
    return float(sxx + syy - 2.0 * sxy)


def split_half_mmd2(
    X: np.ndarray, bandwidth: float, *, draws: int, rng: np.random.Generator
) -> float:
    """A model's own self-vs-self floor, averaged over `draws` random halvings.

    This is the subtrahend in `Δ̂`. Averaging is not cosmetic: a single split of
    20 trials has enormous variance, and an unaveraged floor would make `Δ̂`
    noisier than the effect it is trying to measure.
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n < 4:
        return float("nan")
    vals = []
    idx = np.arange(n)
    for _ in range(draws):
        perm = rng.permutation(idx)
        a, b = perm[: n // 2], perm[n // 2 :]
        v = mmd2(X[a], X[b], bandwidth)
        if not math.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


@dataclass
class DeltaHat:
    """`Δ̂` with the parts that produced it, because the parts are reportable."""

    delta_hat: float
    mmd2_between: float
    within_a: float
    within_b: float
    n_a: int
    n_b: int
    form: str = "difference"


def delta_hat(
    A: np.ndarray,
    B: np.ndarray,
    bandwidth: float,
    *,
    draws: int = 25,
    rng: np.random.Generator | None = None,
    form: str = "difference",
) -> DeltaHat:
    """The preregistered primary effect (§6.4.1).

        Δ̂ = MMD²(A,B) − ½[MMD²(A₁,A₂) + MMD²(B₁,B₂)]

    `form="standardized"` divides the between term by the pooled within term
    instead of subtracting. Which form is primary is declared in the manifest
    before discovery; this function will compute either, and `analyze.py` passes
    `manifest.delta_hat_form` so the choice cannot drift at call sites.
    """
    rng = rng or np.random.default_rng(0)
    between = mmd2(A, B, bandwidth)
    wa = split_half_mmd2(A, bandwidth, draws=draws, rng=rng)
    wb = split_half_mmd2(B, bandwidth, draws=draws, rng=rng)
    if form == "standardized":
        pooled = 0.5 * (wa + wb)
        val = between / pooled if pooled and pooled > 0 else float("nan")
    else:
        val = between - 0.5 * (wa + wb)
    return DeltaHat(val, between, wa, wb, len(A), len(B), form)


# ---------------------------------------------------------------------------
# Permutation, within time blocks
# ---------------------------------------------------------------------------


def permutation_p(
    A: np.ndarray,
    B: np.ndarray,
    blocks_a: np.ndarray,
    blocks_b: np.ndarray,
    bandwidth: float,
    *,
    n_permutations: int,
    draws: int = 25,
    seed: int = 0,
    form: str = "difference",
) -> tuple[float, float, int]:
    """Cue-wise p for `Δ̂` under model labels exchangeable *within* time block.

    Returns `(p, observed, n_effective_permutations)`. The p-value uses the
    (r+1)/(B+1) form, so it is never reported as exactly 0 — a permutation test
    cannot license "p = 0", and a 0 on screen would be a claim the data cannot
    support.

    Labels are shuffled inside each block independently. Shuffling globally
    would let a drifting provider masquerade as a model effect, which is the
    whole reason §6.4 restricts the space; `p_floor` reports the price.
    """
    rng = np.random.default_rng(seed)
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    obs = delta_hat(A, B, bandwidth, draws=draws, rng=rng, form=form).delta_hat

    pooled = np.vstack([A, B])
    blocks = np.concatenate([np.asarray(blocks_a), np.asarray(blocks_b)])
    is_a = np.concatenate([np.ones(len(A), bool), np.zeros(len(B), bool)])

    count = 0
    done = 0
    for _ in range(n_permutations):
        lab = is_a.copy()
        for b in np.unique(blocks):
            sel = np.flatnonzero(blocks == b)
            lab[sel] = rng.permutation(lab[sel])
        pa, pb = pooled[lab], pooled[~lab]
        if len(pa) < 4 or len(pb) < 4:
            continue
        v = delta_hat(pa, pb, bandwidth, draws=max(4, draws // 5), rng=rng, form=form)
        if math.isnan(v.delta_hat):
            continue
        done += 1
        if v.delta_hat >= obs:
            count += 1
    if done == 0:
        return float("nan"), obs, 0
    return (count + 1) / (done + 1), obs, done


def p_floor(block_sizes_a: list[int], block_sizes_b: list[int]) -> float:
    """The smallest p a within-block permutation test can ever return (§6.7.1).

    The number of distinct within-block label assignments is
    `∏_k C(n_k, a_k)`. With `N` distinct assignments the minimum achievable
    p-value is `1/N` — and if that floor sits above the BY-adjusted threshold,
    **no cue can be confirmed at any effect size**, for a purely combinatorial
    reason. Reporting it is a Phase 0 requirement, not a diagnostic nicety.
    """
    total = 1.0
    for na, nb in zip(block_sizes_a, block_sizes_b, strict=False):
        n = na + nb
        if n <= 0:
            continue
        total *= math.comb(n, na)
        if total > 1e18:
            return 1e-18
    return 1.0 / total if total > 0 else 1.0


# ---------------------------------------------------------------------------
# Multiple testing
# ---------------------------------------------------------------------------


def benjamini_yekutieli(pvals: list[float], q: float = 0.05) -> tuple[list[float], list[bool]]:
    """BY-adjusted q-values and rejections (§6.5).

    BY rather than BH because per-cue tests over a shared cue landscape are not
    independent and are not guaranteed positively dependent either: cues share
    an embedder, a prompt frame, and collection blocks. The `c(m) = Σ 1/i`
    penalty is the price of not having to argue about the dependence structure.
    """
    m = len(pvals)
    if m == 0:
        return [], []
    cm = float(np.sum(1.0 / np.arange(1, m + 1)))
    order = np.argsort(pvals)
    p_sorted = np.asarray(pvals, dtype=np.float64)[order]
    adj = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        val = p_sorted[i] * m * cm / (i + 1)
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return [float(v) for v in out], [bool(v <= q) for v in out]


# ---------------------------------------------------------------------------
# Bootstrap, stratified by time block
# ---------------------------------------------------------------------------


def bootstrap_ci(
    A: np.ndarray,
    B: np.ndarray,
    blocks_a: np.ndarray,
    blocks_b: np.ndarray,
    bandwidth: float,
    *,
    n_bootstrap: int,
    draws: int = 8,
    seed: int = 0,
    alpha: float = 0.05,
    form: str = "difference",
) -> tuple[float, float]:
    """Bias-corrected percentile CI for `Δ̂`, resampling within block (§6.5).

    Stratifying the resample by block keeps the bootstrap replicates' time
    structure identical to the observed one; an unstratified resample would
    quietly widen or narrow the interval depending on how drift happened to be
    distributed, which is the opposite of what the interval is for.

    **Why the bias correction is not optional here.** Resampling with
    replacement produces duplicated trials, and a duplicate is at distance zero
    from itself. `Δ̂`'s subtrahend is a *within-model* MMD², which duplicates
    shrink — so every replicate's Δ̂ is inflated by an amount that has nothing
    to do with the models and everything to do with the resampling. Left
    uncorrected the interval drifts upward until it no longer contains the point
    estimate it is supposed to be an interval for, which is both wrong and
    obviously wrong. The replicate mean minus the observed value estimates that
    shift, and subtracting it puts the interval back around the estimate.

    The observed value is recomputed here with the SAME `draws` as the
    replicates: comparing a 25-draw point estimate against 5-draw replicates
    would fold the averaging difference into the "bias" and correct for the
    wrong thing.
    """
    rng = np.random.default_rng(seed)
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    ba, bb = np.asarray(blocks_a), np.asarray(blocks_b)
    observed = delta_hat(A, B, bandwidth, draws=draws, rng=rng, form=form).delta_hat
    vals: list[float] = []
    for _ in range(n_bootstrap):
        ia = _strat_resample(ba, rng)
        ib = _strat_resample(bb, rng)
        if len(ia) < 4 or len(ib) < 4:
            continue
        v = delta_hat(A[ia], B[ib], bandwidth, draws=draws, rng=rng, form=form).delta_hat
        if not math.isnan(v):
            vals.append(v)
    if len(vals) < 10 or math.isnan(observed):
        return float("nan"), float("nan")
    bias = float(np.mean(vals)) - observed
    lo = float(np.quantile(vals, alpha / 2)) - bias
    hi = float(np.quantile(vals, 1 - alpha / 2)) - bias
    return lo, hi


def _strat_resample(blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out: list[int] = []
    for b in np.unique(blocks):
        sel = np.flatnonzero(blocks == b)
        out.extend(rng.choice(sel, size=len(sel), replace=True).tolist())
    return np.asarray(out, dtype=int)


# ---------------------------------------------------------------------------
# Location vs dispersion (§6.4.2)
# ---------------------------------------------------------------------------


@dataclass
class Decomposition:
    dispersion_a: float
    dispersion_b: float
    location: float
    dominant: str  # "location" | "dispersion" | "mixed"


def decompose(A: np.ndarray, B: np.ndarray) -> Decomposition:
    """Split an effect into "they centre elsewhere" vs "one is simply noisier".

    Reported undecomposed, those two support completely different conclusions —
    and only one of them is a difference in what the models associate. A cue
    whose effect is dispersion-driven is labelled as such and is never described
    as an association difference.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    da = _mean_pairwise(A)
    db = _mean_pairwise(B)
    loc = float(np.linalg.norm(A.mean(0) - B.mean(0)))
    spread_gap = abs(da - db)
    if loc > 1.5 * spread_gap:
        dom = "location"
    elif spread_gap > 1.5 * loc:
        dom = "dispersion"
    else:
        dom = "mixed"
    return Decomposition(da, db, loc, dom)


def _mean_pairwise(X: np.ndarray) -> float:
    if len(X) < 2:
        return float("nan")
    d = np.sqrt(_sqdist(X, X))
    iu = np.triu_indices(len(X), k=1)
    return float(d[iu].mean())


# ---------------------------------------------------------------------------
# Associate-distribution statistics
# ---------------------------------------------------------------------------


def entropy(counts: dict[str, int] | list[int]) -> float:
    """Shannon entropy in nats of an associate-frequency table."""
    c = np.asarray(list(counts.values()) if isinstance(counts, dict) else counts, float)
    c = c[c > 0]
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def miller_madow_entropy(counts: np.ndarray) -> float:
    """Miller–Madow bias-corrected entropy: plug-in plus `(K̂−1)/(2n)`.

    The plug-in estimator is downward-biased in entropy and therefore upward-
    biased in JSD, and the bias grows with the number of distinct types — so the
    more lexically diverse model inflates JSD for free. §6.4.3 identifies that as
    pushing in the *same* direction as the capability confound, which is why the
    correction is mandatory rather than optional polish.
    """
    c = np.asarray(counts, float)
    c = c[c > 0]
    n = c.sum()
    if n <= 0:
        return 0.0
    p = c / n
    plug = float(-(p * np.log(p)).sum())
    return plug + (len(c) - 1) / (2.0 * n)


def jsd(
    counts_a: dict[str, int], counts_b: dict[str, int], *, correction: str = "miller-madow"
) -> tuple[float, int, int]:
    """Jensen–Shannon divergence over the union support, bias-corrected.

    Returns `(jsd_nats, n_a, n_b)`; the effective `n` per model rides along
    because §6.4.3 requires every distribution-comparison statistic to report
    the `n` it was computed at.
    """
    keys = sorted(set(counts_a) | set(counts_b))
    a = np.array([counts_a.get(k, 0) for k in keys], float)
    b = np.array([counts_b.get(k, 0) for k in keys], float)
    na, nb = int(a.sum()), int(b.sum())
    if na == 0 or nb == 0:
        return float("nan"), na, nb
    pa, pb = a / na, b / nb
    m = 0.5 * (pa + pb)
    if correction == "miller-madow":
        # JSD = H(M) − ½[H(A) + H(B)]; correcting each entropy term separately
        # is what removes the type-count-dependent bias.
        mixed_counts = 0.5 * (a + b)
        h_m = miller_madow_entropy(mixed_counts)
        val = h_m - 0.5 * (miller_madow_entropy(a) + miller_madow_entropy(b))
    else:
        h = lambda p: float(-(p[p > 0] * np.log(p[p > 0])).sum())  # noqa: E731
        val = h(m) - 0.5 * (h(pa) + h(pb))
    return float(max(val, 0.0)), na, nb


def rbo(list_a: list[str], list_b: list[str], p: float) -> float:
    """Rank-biased overlap at a **declared** `p`, extrapolated (§6.4.3).

    `p` is a manifest field, never a default chosen at call time: it alone can
    move an RBO score substantially, so leaving it to implementation would be a
    silent researcher degree of freedom.

    The extrapolated form (Webber, Moffat & Zobel 2010, eq. 32) is used rather
    than the truncated sum, and on this study's lists that is not a refinement —
    it is the difference between a readable number and a misleading one. RBO's
    weights are defined over an infinite ranking; a truncated sum over `k` items
    can never exceed `1 − p^k`, so with the three associates of §5.2 and the
    conventional `p = 0.9` two IDENTICAL answer lists would score 0.271. A
    reader seeing 0.27 on a scale whose top is 1.0 would conclude the two models
    barely agreed, when in fact they agreed completely.

    The extrapolation assumes the agreement seen at depth `k` continues below
    it. That assumption is visible and arguable; the truncated alternative's
    assumption — that everything unseen disagrees — is neither, because it hides
    inside a number that looks like a proportion.
    """
    if not list_a or not list_b:
        return float("nan")
    short, long_ = (list_a, list_b) if len(list_a) <= len(list_b) else (list_b, list_a)
    s, ell = len(short), len(long_)

    sa: set[str] = set()
    sb: set[str] = set()
    x: list[float] = [0.0]  # x[d] = |A_d ∩ B_d|, 1-indexed
    for d in range(1, ell + 1):
        if d <= len(short):
            sa.add(short[d - 1])
        sb.add(long_[d - 1])
        x.append(float(len(sa & sb)))

    # Σ_{d=1..ℓ} X_d/d · p^d, plus the tail correction for the depths at which
    # only the longer list still has items.
    term = sum((x[d] / d) * (p**d) for d in range(1, ell + 1))
    term += sum(x[s] * (d - s) / (s * d) * (p**d) for d in range(s + 1, ell + 1))
    head = (1 - p) / p * term
    tail = ((x[ell] - x[s]) / ell + x[s] / s) * (p**ell)
    return float(min(1.0, max(0.0, head + tail)))


def type_token_ratio(tokens: list[str]) -> float:
    return len(set(tokens)) / len(tokens) if tokens else 0.0


# ---------------------------------------------------------------------------
# Compliance parity and Manski bounds (§6.5.1)
# ---------------------------------------------------------------------------


@dataclass
class ManskiBound:
    lo: float
    hi: float
    spans_floor: bool


def manski_bounds(
    A: np.ndarray,
    B: np.ndarray,
    bandwidth: float,
    *,
    missing_a: int,
    missing_b: int,
    effect_floor: float,
    draws: int = 8,
    seed: int = 0,
    form: str = "difference",
) -> ManskiBound:
    """Worst-case bounds on `Δ̂` under the missing trials (§6.5.1).

    The missing trials are unobserved by construction, so no imputation can be
    right. What *can* be done is bracket them: the best case fills each model's
    gap with copies of its own observed points (adds nothing), the worst case
    fills them with the other model's points (maximally erasing the difference)
    and with each model's own extremes (maximally inflating it). If the bound
    spans the effect floor, the cue cannot be confirmed regardless of its point
    estimate — which is the entire purpose of computing it.
    """
    rng = np.random.default_rng(seed)
    base = delta_hat(A, B, bandwidth, draws=draws, rng=rng, form=form).delta_hat
    if missing_a <= 0 and missing_b <= 0:
        return ManskiBound(base, base, False)

    # Erasing case: the missing trials would have looked like the OTHER model.
    fill_a = B[rng.integers(0, len(B), size=missing_a)] if missing_a else np.empty((0, A.shape[1]))
    fill_b = A[rng.integers(0, len(A), size=missing_b)] if missing_b else np.empty((0, B.shape[1]))
    erased = delta_hat(
        np.vstack([A, fill_a]), np.vstack([B, fill_b]), bandwidth, draws=draws, rng=rng, form=form
    ).delta_hat

    # Inflating case: the missing trials would have been more of the same, which
    # tightens each within-model term and so raises the contrast.
    dup_a = A[rng.integers(0, len(A), size=missing_a)] if missing_a else np.empty((0, A.shape[1]))
    dup_b = B[rng.integers(0, len(B), size=missing_b)] if missing_b else np.empty((0, B.shape[1]))
    inflated = delta_hat(
        np.vstack([A, dup_a]), np.vstack([B, dup_b]), bandwidth, draws=draws, rng=rng, form=form
    ).delta_hat

    lo, hi = float(min(erased, inflated, base)), float(max(erased, inflated, base))
    return ManskiBound(lo, hi, bool(lo <= effect_floor <= hi))


def compliance_parity(parse_rate_a: float, parse_rate_b: float, max_gap: float) -> bool:
    """True when the pair is comparable at all (§6.5.1). Not a warning — a gate."""
    return abs(parse_rate_a - parse_rate_b) <= max_gap


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------


def split_half_reliability(
    X: np.ndarray, *, draws: int = 25, seed: int = 0
) -> float:
    """Mean cosine between the two halves' centroids, over random halvings.

    This is gate 2. It is necessary and, alone, actively misleading — a model
    that copies the exemplar every time scores 1.0 here with zero semantic
    content — which is why `analyze.py` never evaluates it without the
    informativeness floor of §6.5.2 beside it.
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n < 4:
        return float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        perm = rng.permutation(n)
        a, b = X[perm[: n // 2]], X[perm[n // 2 :]]
        ca, cb = a.mean(0), b.mean(0)
        na, nb = np.linalg.norm(ca), np.linalg.norm(cb)
        if na > 0 and nb > 0:
            vals.append(float(ca @ cb / (na * nb)))
    return float(np.mean(vals)) if vals else float("nan")
