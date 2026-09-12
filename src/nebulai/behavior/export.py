"""`behavior.json` — the compact artifact the Behavior page reads (plan §9.5).

The landscape is **PCA, not UMAP**, and that is the single most consequential
line in this file. §7.1.1 measured it: at n=180 cues, UMAP produced a silhouette
of 0.88 on *shuffled* data against 0.43 on the real data — it manufactures
islands at this sample size, and a reader cannot tell a manufactured island from
a finding. PCA at n=100–300 is honest about how little structure there is, which
is the correct thing for the viewer to show.

The projection is exported as a persisted `(mean, axes)` pair, not only as
coordinates. §5.4 grows the cue set by two orders of magnitude while §7.1 and
§10.3 promise a fixed coordinate system and permalinkable cue positions; a refit
would move every previously published cue and silently break both. With the
transform persisted, a new cue is *placed* into the existing space by
`(x − mean) @ axes` instead.

`missing` is written as `null`. Everywhere. A metric that could not be computed
is not a zero, and the viewer renders the two differently (§8.5).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import BEHAVIOR_SCHEMA_VERSION
from ..backend.interp.bundles import _pca_rows
from .analyze import ArmProfile, CueResult
from .contract import Manifest
from .embed import Embedder


def cue_landscape(cue_texts: list[str], embedder: Embedder, dims: int = 2) -> dict[str, Any]:
    """The fixed coordinate system, built from the cue words themselves.

    Not from either model's outputs: the models never receive separate
    projections that a viewer might mistake for comparable internal geometry
    (§7.1). Two cues are near each other here because the *words* are near each
    other in a pinned neutral encoder, and that is all the position means.
    """
    V = embedder.encode(cue_texts)
    fit = _pca_rows(np.asarray(V, dtype=np.float64), dims)
    total = float(fit.total_var) or 1.0
    quantity = float(fit.evr[:dims].sum())
    return {
        "method": "pca",
        "dims": dims,
        "coords": [[round(float(x), 5) for x in row] for row in fit.coords],
        # AXIS-MAJOR flat layout, matching `interp/bundles.py`: axis j is
        # axes[j*d:(j+1)*d]. Shipping the transform is what makes the landscape
        # extensible without moving a published cue.
        "pca_mean": [round(float(v), 6) for v in fit.mean],
        "pca_axes": [round(float(v), 6) for v in fit.axes.T.reshape(-1)],
        "pca_axes_shape": [int(fit.axes.shape[1]), int(fit.axes.shape[0])],
        "explained_variance_ratio": [round(float(v), 5) for v in fit.evr[:dims]],
        # §8.1 requires the caption to be sourced from the projection itself
        # rather than asserted in copy.
        "projection": {
            "quantity": round(quantity, 4),
            "quantity_label": (
                f"these {dims} axes carry {quantity:.0%} of the variance in the "
                f"cue-word embeddings"
            ),
            "total_variance": round(total, 6),
            "encoder": getattr(embedder, "id", "unknown"),
            "encoder_revision": getattr(embedder, "revision", ""),
            "warning": (
                "positions come from the cue words, not from any model's "
                "behaviour; distance here is lexical-semantic similarity of the "
                "cues themselves"
            ),
        },
    }


def trustworthiness(high: np.ndarray, low: np.ndarray, k: int = 10) -> float:
    """Standard trustworthiness of a projection — reported, never assumed.

    The Behavior page prints this beside the landscape for the same reason the
    map pipeline exports its noise fraction: the evidence to distrust the
    picture ships with the picture.
    """
    from sklearn.manifold import trustworthiness as _t

    n = len(high)
    if n <= k + 1:
        return float("nan")
    return float(_t(np.asarray(high), np.asarray(low), n_neighbors=k))


def _arm_dict(p: ArmProfile) -> dict[str, Any]:
    return {
        "model_key": p.model_key,
        "n_attempted": p.n_attempted,
        "n_valid": p.n_valid,
        "parse_rate": round(p.parse_rate, 4),
        "distinct_types": p.distinct_types,
        "entropy": round(p.entropy, 4),
        "type_token_ratio": round(p.type_token_ratio, 4),
        "reliability": None if p.reliability is None else round(p.reliability, 4),
        "top_associates": p.top_ranked[:10],
        "detectors": {
            "cue_echo": round(p.cue_echo_rate, 4),
            "exemplar_echo": round(p.exemplar_echo_rate, 4),
            "within_trial_duplicate": round(p.duplicate_rate, 4),
            "prompt_copy": round(p.prompt_copy_rate, 4),
        },
        "oov_fragment_ratio": (
            None if p.oov_fragment_ratio is None else round(p.oov_fragment_ratio, 4)
        ),
    }


def _cue_dict(r: CueResult) -> dict[str, Any]:
    return {
        "cue": r.cue,
        "stratum": r.stratum,
        "pack": r.pack,
        "partition": r.partition,
        "status": r.status,
        "reasons": r.reasons,
        "delta_hat": _r(r.delta_hat),
        "mmd2_between": _r(r.mmd2_between),
        "within": {k: _r(v) for k, v in r.within.items()},
        "p_value": _r(r.p_value, 6),
        "q_value": _r(r.q_value, 6),
        "ci": [_r(r.ci_lo), _r(r.ci_hi)],
        "manski": [_r(r.manski_lo), _r(r.manski_hi)],
        "jsd": _r(r.jsd),
        "jsd_n": r.jsd_n,
        "rbo": _r(r.rbo),
        "location": _r(r.location),
        "dispersion": {k: _r(v) for k, v in r.dispersion.items()},
        "dominant": r.dominant,
        "capability_reference": _r(r.capability_reference),
        "arms": {k: _arm_dict(v) for k, v in r.arms.items()},
    }


def build_export(
    m: Manifest,
    results: list[CueResult],
    *,
    landscape: dict[str, Any],
    diagnostics: dict[str, Any],
    runs: list[dict[str, Any]],
    samples: dict[str, list[dict[str, Any]]] | None = None,
    coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the artifact. Pure — no I/O, so it is trivially testable."""
    order = {r.cue: i for i, r in enumerate(results)}
    cov = dict(coverage or {})
    # Coverage is always present and always states the denominator. A reader who
    # sees 40 cues has no way to tell a 40-cue study from a 100-cue study that
    # stopped, and "complete" is the one thing they cannot infer from the cue
    # list itself. Absent the field they would assume complete, which is the
    # wrong default for any run that can be interrupted.
    cov.setdefault("cues_planned", len(m.cues))
    cov.setdefault("cues_analyzed", len(results))
    cov.setdefault("complete", cov["cues_analyzed"] >= cov["cues_planned"])
    cov.setdefault("reason", "" if cov["complete"] else "not recorded")
    return {
        "schema": BEHAVIOR_SCHEMA_VERSION,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "study_id": m.study_id,
        "manifest": {
            "hash": m.frozen_hash,
            "protocol_hash": m.protocol_hash(),
            "frozen_at": m.frozen_at,
            "strict": m.strict,
            "models": [asdict(x) for x in m.models],
            "frames": [{"id": f.id, "role": f.role} for f in m.frames],
            "trials_per_cue": m.trials_per_cue,
            "temperature": m.temperature,
            "top_p": m.top_p,
            "max_output_tokens": m.max_output_tokens,
            "embedder": {
                "id": m.embedder_id,
                "sha": m.embedder_sha,
                "dtype": m.embedder_dtype,
                "pooling": m.embedder_pooling,
                "normalize": m.embedder_normalize,
                "secondary_id": m.secondary_embedder_id,
                "second_embedder_resolution": m.second_embedder_resolution,
                # §6.5.4: the UI copy must match the resolution in force. With
                # "downgrade" it says "not specific to one of two similar
                # encoders" — never "embedder-independent".
                "agreement_claim": (
                    "not specific to one of two similar encoders"
                    if m.second_embedder_resolution == "downgrade"
                    else "agrees across two representations of different kinds"
                ),
            },
            "statistics": {
                "delta_hat_form": m.delta_hat_form,
                "mmd_bandwidth": m.mmd_bandwidth,
                "n_permutations": m.n_permutations,
                "n_bootstrap": m.n_bootstrap,
                "effect_floor": m.effect_floor,
                "q_threshold": m.q_threshold,
                "multiple_testing": m.multiple_testing,
                "rbo_p": m.rbo_p,
                "jsd_correction": m.jsd_correction,
                "compliance_parity_max": m.compliance_parity_max,
            },
            # Explicitly null when unmeasured. Never 0.
            "reasoning_tokens_p95": m.reasoning_tokens_p95,
            "fingerprint_available": m.fingerprint_available,
            "p_floor": m.p_floor,
            "git_commit": m.git_commit,
        },
        "claim": (
            "Under this protocol, the listed exact model deployments produced "
            "the association distributions recorded here. Nothing in this "
            "artifact describes any model's internals, and no model is ranked."
        ),
        "landscape": landscape,
        "coverage": cov,
        "cues": [_cue_dict(r) for r in results],
        "cue_index": order,
        "diagnostics": diagnostics,
        "runs": runs,
        "samples": samples or {},
    }


def write_export(path: str | Path, payload: dict[str, Any]) -> Path:
    """Serialize, refusing to write a file no JSON reader can read.

    `allow_nan=False` is the belt to `_r`'s braces: `_r` sanitizes every metric
    that goes through it, and this catches anything that reached the payload by
    another route (a diagnostics block, a future field). Failing here is loud
    and fixable; writing `NaN` produces a `behavior.json` whose `JSON.parse`
    throws in the viewer with no indication of which field caused it.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    except ValueError as exc:
        raise ValueError(
            f"{path}: the export carries a non-finite number ({exc}). A metric "
            f"that could not be computed is written as null, never as NaN — a "
            f"NaN here would make the whole artifact unparseable instead of "
            f"marking one field as not measured."
        ) from exc
    p.write_text(text, encoding="utf-8")
    return p


def _r(x: float | None, nd: int = 5) -> float | None:
    """Round, and map every non-finite value to `None`.

    NaN and ±inf are not metrics and they are not `null` either: `json.dumps`
    writes them as the bare tokens `NaN` / `Infinity`, which no conforming JSON
    reader accepts, so a single un-computable statistic would make the whole
    artifact unparseable rather than showing one field as "not measured".
    """
    if x is None:
        return None
    v = float(x)
    return None if math.isnan(v) or math.isinf(v) else round(v, nd)
