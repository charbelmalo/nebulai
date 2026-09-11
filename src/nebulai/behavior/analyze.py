"""Turn raw trials into per-cue evidence and a confirmation state (plan §6.5).

The whole file is a single long conditional with the gates in §6.5's order, and
that shape is deliberate: a status is a conjunction of ten named checks, and
collapsing them into a score would hide which one failed. Every cue carries the
reasons it is in the state it is in.

Four rules this module refuses to bend:

* **Displayed magnitudes come from the confirmation partition only** (§6.5.3).
  Discovery effects are selection-inflated by construction; they appear as the
  *reason a cue was tested* and are never rendered as magnitude.
* **`missing` is not zero.** A metric that could not be computed is `None`, and
  the exporter writes `null`. A zero would be a measurement.
* **Nothing here ranks models.** `Δ̂` is symmetric; the artifact has no
  "winner" field and the vocabulary carries no leaderboard language.
* **No `confirmed` status from a non-strict source.** A fake adapter, a hash
  embedder, or a moving alias caps the achievable status at `suggestive`.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from . import stats as S
from .contract import CANARY_CUE, Manifest, TrialRecord
from .cues import control_hit
from .embed import Embedder, trial_vector


@dataclass
class ArmProfile:
    """One model's behaviour at one cue — the per-model half of a comparison."""

    model_key: str
    n_attempted: int = 0
    n_valid: int = 0
    parse_rate: float = 0.0
    vectors: np.ndarray | None = None
    blocks: np.ndarray | None = None
    associates: list[str] = field(default_factory=list)
    top_ranked: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    distinct_types: int = 0
    entropy: float = 0.0
    type_token_ratio: float = 0.0
    reliability: float | None = None
    cue_echo_rate: float = 0.0
    exemplar_echo_rate: float = 0.0
    duplicate_rate: float = 0.0
    prompt_copy_rate: float = 0.0
    oov_fragment_ratio: float | None = None


@dataclass
class CueResult:
    cue: str
    stratum: str
    pack: str = ""
    arms: dict[str, ArmProfile] = field(default_factory=dict)
    delta_hat: float | None = None
    mmd2_between: float | None = None
    within: dict[str, float] = field(default_factory=dict)
    p_value: float | None = None
    q_value: float | None = None
    ci_lo: float | None = None
    ci_hi: float | None = None
    jsd: float | None = None
    jsd_n: dict[str, int] = field(default_factory=dict)
    rbo: float | None = None
    location: float | None = None
    dispersion: dict[str, float] = field(default_factory=dict)
    dominant: str = ""
    capability_reference: float | None = None
    manski_lo: float | None = None
    manski_hi: float | None = None
    status: str = "insufficient evidence"
    reasons: list[str] = field(default_factory=list)
    partition: str = "discovery"


# ---------------------------------------------------------------------------
# Building profiles
# ---------------------------------------------------------------------------


def build_profiles(
    trials: Iterable[TrialRecord],
    m: Manifest,
    embedder: Embedder,
) -> dict[str, dict[str, ArmProfile]]:
    """`cue -> model_key -> ArmProfile`, embedding every valid trial once.

    Canary trials are dropped here and nowhere else, so no downstream metric can
    accidentally include them (§5.5.1).
    """
    by: dict[str, dict[str, list[TrialRecord]]] = {}
    for t in trials:
        if t.cue == CANARY_CUE:
            continue
        by.setdefault(t.cue, {}).setdefault(t.model_key, []).append(t)

    # One encoder pass over every associate string in the study.
    texts: list[str] = []
    for cue_map in by.values():
        for rows in cue_map.values():
            for t in rows:
                if t.valid:
                    texts.extend(t.associates)
    lut: dict[str, np.ndarray] = {}
    if texts:
        uniq = sorted(set(texts))
        vecs = embedder.encode(uniq)
        lut = {s: v for s, v in zip(uniq, vecs, strict=True)}

    out: dict[str, dict[str, ArmProfile]] = {}
    for cue, cue_map in by.items():
        out[cue] = {}
        for key, rows in cue_map.items():
            out[cue][key] = _profile(key, rows, lut, m)
    return out


def _profile(
    key: str, rows: list[TrialRecord], lut: dict[str, np.ndarray], m: Manifest
) -> ArmProfile:
    p = ArmProfile(model_key=key, n_attempted=len(rows))
    valid = [t for t in rows if t.valid]
    p.n_valid = len(valid)
    p.parse_rate = len(valid) / len(rows) if rows else 0.0

    marks = [t.usage.get("marks", {}) for t in rows if isinstance(t.usage, dict)]
    n = max(len(marks), 1)
    p.cue_echo_rate = sum(bool(x.get("cue_echo")) for x in marks) / n
    p.exemplar_echo_rate = sum(bool(x.get("exemplar_echo")) for x in marks) / n
    p.duplicate_rate = sum(bool(x.get("within_trial_duplicate")) for x in marks) / n
    p.prompt_copy_rate = sum(bool(x.get("prompt_copy")) for x in marks) / n

    if not valid:
        return p

    vecs = []
    blocks = []
    for t in valid:
        av = np.stack([lut[a] for a in t.associates if a in lut]) if t.associates else None
        if av is None or len(av) == 0:
            continue
        vecs.append(trial_vector(av, m.rank_weights))
        blocks.append(t.block)
        p.associates.extend(t.associates)
    if vecs:
        p.vectors = np.stack(vecs)
        p.blocks = np.asarray(blocks)
        # `_f`, not the raw value: `split_half_reliability` returns NaN when
        # there are too few trials to halve, and NaN is neither a measurement
        # nor a missing marker. Left raw it would (a) pass gate 2 silently,
        # because `nan < floor` is False, and (b) reach the exporter, where
        # `json.dumps` writes the literal `NaN` — not valid JSON, and not the
        # `null` §8.5 requires the viewer to render as "not measured".
        p.reliability = _f(
            S.split_half_reliability(p.vectors, draws=m.split_half_draws, seed=m.seed)
        )

    c = Counter(p.associates)
    p.counts = dict(c)
    p.distinct_types = len(c)
    p.entropy = S.entropy(p.counts)
    p.type_token_ratio = S.type_token_ratio(p.associates)
    p.top_ranked = [w for w, _ in c.most_common(20)]
    # Fragmentation proxy (§6.4.4): the share of associates that are not plain
    # single alphabetic words. It is a proxy and is labelled as one; the
    # *differential* between arms is what is reported, never the level.
    if p.associates:
        odd = sum(1 for a in p.associates if not a.replace(" ", "").isalpha())
        p.oov_fragment_ratio = odd / len(p.associates)
    return p


# ---------------------------------------------------------------------------
# Comparing two arms at one cue
# ---------------------------------------------------------------------------


def compare_cue(
    cue: str,
    stratum: str,
    a: ArmProfile,
    b: ArmProfile,
    m: Manifest,
    *,
    bandwidth: float,
    capability_reference: float | None = None,
    pack: str = "",
    partition: str = "discovery",
    permutations: int | None = None,
) -> CueResult:
    r = CueResult(cue=cue, stratum=stratum, pack=pack, partition=partition)
    r.arms = {a.model_key: a, b.model_key: b}

    if a.vectors is None or b.vectors is None:
        r.reasons.append("no valid trials for at least one model")
        return r
    if a.n_valid < m.min_valid_trials or b.n_valid < m.min_valid_trials:
        r.reasons.append(
            f"gate 1: valid trials {a.n_valid}/{b.n_valid} below the manifest's "
            f"minimum of {m.min_valid_trials}"
        )

    d = S.delta_hat(
        a.vectors, b.vectors, bandwidth,
        draws=m.split_half_draws,
        rng=np.random.default_rng(m.seed),
        form=m.delta_hat_form,
    )
    r.delta_hat = _f(d.delta_hat)
    r.mmd2_between = _f(d.mmd2_between)
    r.within = {a.model_key: _f(d.within_a) or 0.0, b.model_key: _f(d.within_b) or 0.0}

    p, _obs, _n = S.permutation_p(
        a.vectors, b.vectors, a.blocks, b.blocks, bandwidth,
        n_permutations=permutations or m.n_permutations,
        draws=max(6, m.split_half_draws // 3),
        seed=m.seed,
        form=m.delta_hat_form,
    )
    r.p_value = _f(p)

    lo, hi = S.bootstrap_ci(
        a.vectors, b.vectors, a.blocks, b.blocks, bandwidth,
        n_bootstrap=m.n_bootstrap, seed=m.seed, form=m.delta_hat_form,
    )
    r.ci_lo, r.ci_hi = _f(lo), _f(hi)

    dec = S.decompose(a.vectors, b.vectors)
    r.location = _f(dec.location)
    r.dispersion = {a.model_key: _f(dec.dispersion_a) or 0.0,
                    b.model_key: _f(dec.dispersion_b) or 0.0}
    r.dominant = dec.dominant

    j, na, nb = S.jsd(a.counts, b.counts, correction=m.jsd_correction)
    r.jsd = _f(j)
    r.jsd_n = {a.model_key: na, b.model_key: nb}
    r.rbo = _f(S.rbo(a.top_ranked, b.top_ranked, m.rbo_p))
    r.capability_reference = capability_reference

    # Compliance parity + Manski bounds (§6.5.1)
    if not S.compliance_parity(a.parse_rate, b.parse_rate, m.compliance_parity_max):
        r.status = "incomparable"
        r.reasons.append(
            f"gate 3: parse rates {a.parse_rate:.2f} vs {b.parse_rate:.2f} differ "
            f"by more than {m.compliance_parity_max:.2f}. The survivors are a "
            f"non-random subsample of the lower-compliance model's behaviour, "
            f"and that bias is present even at equal surviving n."
        )
        return r
    mb = S.manski_bounds(
        a.vectors, b.vectors, bandwidth,
        missing_a=a.n_attempted - a.n_valid,
        missing_b=b.n_attempted - b.n_valid,
        effect_floor=m.effect_floor,
        seed=m.seed,
        form=m.delta_hat_form,
    )
    r.manski_lo, r.manski_hi = _f(mb.lo), _f(mb.hi)
    if mb.spans_floor:
        r.reasons.append(
            "the worst-case bound over the unparsed trials spans the effect "
            "floor, so the point estimate cannot carry a confirmation"
        )

    _apply_gates(r, a, b, m)
    return r


def _apply_gates(r: CueResult, a: ArmProfile, b: ArmProfile, m: Manifest) -> None:
    """§6.5's numbered gates, in order, each recording its own reason."""
    # 2 — reliability
    for p in (a, b):
        if p.reliability is None or p.reliability < m.reliability_floor:
            r.reasons.append(
                f"gate 2: {p.model_key} split-half reliability "
                f"{p.reliability!r} below the calibrated floor "
                f"{m.reliability_floor}"
            )
    # 4 — informativeness (§6.5.2). Reliability alone is misleading: a model
    # that copies the exemplar every time scores 1.0 on gate 2.
    for p in (a, b):
        if p.distinct_types < m.min_distinct_types:
            r.reasons.append(
                f"gate 4: {p.model_key} produced {p.distinct_types} distinct "
                f"associate types, below {m.min_distinct_types}"
            )
        if p.entropy < m.min_associate_entropy:
            r.reasons.append(
                f"gate 4: {p.model_key} associate entropy {p.entropy:.3f} nats "
                f"below {m.min_associate_entropy}"
            )
        if p.cue_echo_rate > 0.5 or p.exemplar_echo_rate > 0.5:
            r.reasons.append(
                f"gate 4: {p.model_key} echoes the cue or the exemplar on "
                f"{max(p.cue_echo_rate, p.exemplar_echo_rate):.0%} of trials — "
                f"degenerate output is maximally reliable and carries no "
                f"semantic content"
            )
    # 4b — differential OOV/fragmentation (§6.4.4)
    if a.oov_fragment_ratio is not None and b.oov_fragment_ratio is not None:
        gap = abs(a.oov_fragment_ratio - b.oov_fragment_ratio)
        if gap > m.oov_differential_max:
            r.reasons.append(
                f"the fragmentation/OOV proxy differs by {gap:.2f} between arms; "
                f"the encoder represents those terms in low-density regions, so "
                f"part of this effect is representational, not associative"
            )
    # 5 — effect floor and the capability reference (§5.7)
    if r.delta_hat is None or math.isnan(r.delta_hat):
        r.reasons.append("gate 5: the primary effect could not be computed")
    elif r.delta_hat < m.effect_floor:
        r.status = "no detected deviation" if not r.reasons else "insufficient evidence"
        r.reasons.append(
            f"gate 5: Δ̂ {r.delta_hat:.4f} is below the frozen smallest effect "
            f"of interest {m.effect_floor}"
        )
        return
    elif r.capability_reference is not None and r.delta_hat <= r.capability_reference:
        r.status = "capability-attributable"
        r.reasons.append(
            f"gate 5b: Δ̂ {r.delta_hat:.4f} does not exceed the same-family "
            f"capability contrast {r.capability_reference:.4f} for this cue, so "
            f"it is attributable to scale and competence and is not evidence of "
            f"a difference in learned association"
        )
        return

    if r.reasons:
        r.status = "insufficient evidence"
        return
    r.status = "suggestive"  # discovery cannot produce more than this


def finalize_family(results: list[CueResult], m: Manifest) -> list[CueResult]:
    """Apply BY across the declared family and promote survivors.

    Promotion stops at `suggestive` here. `confirmed` requires arms R and G,
    which live in a different partition and are joined by `confirm`.
    """
    testable = [r for r in results if r.p_value is not None and not math.isnan(r.p_value)]
    if not testable:
        return results
    q, rej = S.benjamini_yekutieli([r.p_value for r in testable], m.q_threshold)  # type: ignore[arg-type]
    for r, qv, ok in zip(testable, q, rej, strict=True):
        r.q_value = qv
        if not ok and r.status == "suggestive":
            r.status = "no detected deviation"
            r.reasons.append(f"gate 6: BY-adjusted q {qv:.4f} above {m.q_threshold}")
    return results


def confirm(
    discovery: dict[str, CueResult],
    arm_r: dict[str, CueResult],
    arm_g: dict[str, CueResult],
    m: Manifest,
    *,
    strict_source: bool = True,
) -> dict[str, CueResult]:
    """Join the two confirmation arms into a final state (§6.5 gates 7–8).

    Returns the **arm R** results as the displayed record, because §6.5.3
    requires every displayed magnitude to come from the confirmation partition.
    Discovery survives only as the reason a cue was tested.
    """
    out: dict[str, CueResult] = {}
    for cue, dres in discovery.items():
        r = arm_r.get(cue)
        g = arm_g.get(cue)
        if r is None:
            dres.reasons.append("arm R was not collected for this cue")
            out[cue] = dres
            continue
        shown = r
        shown.partition = "confirmation"
        shown.reasons.insert(
            0,
            f"tested because discovery ranked it (discovery Δ̂ "
            f"{dres.delta_hat:.4f}, shown here only as the selection reason)"
            if dres.delta_hat is not None
            else "tested because discovery ranked it",
        )
        r_ok = r.status in ("suggestive", "confirmed")
        g_ok = bool(g and g.status in ("suggestive", "confirmed"))
        if r_ok and g_ok:
            shown.status = "confirmed" if strict_source else "suggestive"
            if not strict_source:
                shown.reasons.append(
                    "capped at suggestive: at least one arm ran on a "
                    "non-strict source (a synthetic adapter, a stand-in "
                    "embedder, or a moving model alias), which cannot support "
                    "a confirmed status (§5.5)"
                )
        elif r_ok and not g_ok:
            shown.status = "frame-specific"
            shown.reasons.append(
                "arm R reproduced the effect and arm G did not: a real effect "
                "that does not survive a different way of asking. This is a "
                "finding, not a failure, and it is not the same thing as a "
                "false positive."
            )
        elif not r_ok:
            shown.status = "suggestive"
            shown.reasons.append(
                "arm R did not reproduce the discovery effect, so the discovery "
                "hit is consistent with selection noise"
            )
        out[cue] = shown
    return out


# ---------------------------------------------------------------------------
# Calibration controls (§6.7)
# ---------------------------------------------------------------------------


def aa_control(
    profile: ArmProfile, m: Manifest, bandwidth: float, *, draws: int = 20
) -> dict[str, Any]:
    """A/A pseudo-comparison: split one model's own trials and test them.

    A well-behaved instrument returns `Δ̂ ≈ 0` and a uniform p. If it does not,
    every between-model number in the study is suspect and no amount of effect
    size rescues it.
    """
    if profile.vectors is None or len(profile.vectors) < 8:
        return {"status": "missing", "reason": "fewer than 8 valid trials"}
    rng = np.random.default_rng(m.seed)
    deltas, ps = [], []
    v, b = profile.vectors, profile.blocks
    assert b is not None
    for _ in range(draws):
        perm = rng.permutation(len(v))
        h = len(v) // 2
        ia, ib = perm[:h], perm[h:]
        d = S.delta_hat(v[ia], v[ib], bandwidth, draws=8, rng=rng, form=m.delta_hat_form)
        deltas.append(d.delta_hat)
        p, _, _ = S.permutation_p(
            v[ia], v[ib], b[ia], b[ib], bandwidth,
            n_permutations=100, draws=6, seed=int(rng.integers(1 << 30)),
            form=m.delta_hat_form,
        )
        ps.append(p)
    arr = np.asarray([x for x in deltas if not math.isnan(x)])
    parr = np.asarray([x for x in ps if not math.isnan(x)])
    return {
        "status": "measured",
        "model_key": profile.model_key,
        "n_splits": int(len(arr)),
        "delta_hat_mean": float(arr.mean()) if arr.size else None,
        "delta_hat_p95": float(np.quantile(arr, 0.95)) if arr.size else None,
        "false_positive_rate_at_0.05": float((parr <= 0.05).mean()) if parr.size else None,
    }


def positive_control_report(
    trials: Iterable[TrialRecord], *, threshold: float = 0.5
) -> dict[str, Any]:
    """Per-model pass rate on the §6.7.2 control cues.

    A model below `threshold` is not producing association data, and this is a
    **Phase 0 gate**: no divergence number computed from its outputs means
    anything. A partial failure is informative too — GPT-2 failing where another
    arm passes is §4.2.1's capability confound in its most direct form.
    """
    per_model: dict[str, list[int]] = {}
    per_cue: dict[str, dict[str, list[int]]] = {}
    for t in trials:
        if not t.valid:
            continue
        exp = control_hit(t.cue, t.associates)
        if t.cue.casefold() not in _CONTROL_KEYS:
            continue
        per_model.setdefault(t.model_key, []).append(int(exp))
        per_cue.setdefault(t.cue, {}).setdefault(t.model_key, []).append(int(exp))
    rates = {
        k: (sum(v) / len(v) if v else None) for k, v in per_model.items()
    }
    return {
        "threshold": threshold,
        "pass_rate": rates,
        "n_trials": {k: len(v) for k, v in per_model.items()},
        "passed": {k: (r is not None and r >= threshold) for k, r in rates.items()},
        "per_cue": {
            c: {k: (sum(v) / len(v) if v else None) for k, v in mm.items()}
            for c, mm in per_cue.items()
        },
    }


from .cues import POSITIVE_CONTROL_MAP as _PCM  # noqa: E402

_CONTROL_KEYS = set(_PCM)


def canary_report(trials: Iterable[TrialRecord], embedder: Embedder) -> dict[str, Any]:
    """Block-wise drift in the canary's outputs (§5.5.1).

    Strictly stronger than `system_fingerprint`: it measures behaviour rather
    than a self-reported label, works when the field is absent, and catches
    silent serving changes that leave the fingerprint untouched.
    """
    rows = [t for t in trials if t.cue == CANARY_CUE and t.raw_output]
    if not rows:
        return {"status": "missing", "reason": "no canary trials recorded"}
    by_block: dict[tuple[str, int], list[str]] = {}
    for t in rows:
        by_block.setdefault((t.model_key, t.block), []).append(t.raw_output or "")
    keys = sorted(by_block)
    texts = [" ".join(by_block[k]) for k in keys]
    V = embedder.encode(texts)
    out: dict[str, Any] = {"status": "measured", "blocks": [], "drift": {}}
    for (mk, blk), v in zip(keys, V, strict=True):
        out["blocks"].append({"model_key": mk, "block": blk})
    for mk in sorted({k[0] for k in keys}):
        idx = [i for i, k in enumerate(keys) if k[0] == mk]
        if len(idx) < 2:
            continue
        sub = V[idx]
        sims = [float(sub[i] @ sub[i + 1]) for i in range(len(sub) - 1)]
        out["drift"][mk] = {
            "consecutive_block_cosine": sims,
            "min": min(sims),
            # Not a verdict: a low value means "look at this", and §5.5.1's
            # response is to stop or partition, which is a human decision.
            "flag": bool(min(sims) < 0.5),
        }
    return out


def _f(x: float | None) -> float | None:
    if x is None:
        return None
    v = float(x)
    return None if math.isnan(v) else v
