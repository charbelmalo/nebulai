"""`out/<model>/directions.json` — a direction as a first-class object.

A direction is one unit vector in one named space, plus the provenance that says
how it came to be known and the null it has to be read against. Those three
parts travel together because separating them is exactly how a direction stops
being evidence: a vector with no space can be projected against anything (D2),
and a vector with no null produces a histogram that looks like a finding no
matter what is in it (R5).

Five constructors, matching §4.2 of the plan:

    diff_of_means(pos, neg, space)        two sets of activations
    pca_component(matrix, k, space)       the k-th principal component
    from_sae_decoder(repo, feature_id, …) one row of a trained dictionary
    from_lora_rank1(b, a, space)          the output-side factor of a rank-1 adapter
    from_two_selections(a_ids, b_ids, …)  the viewer's two-cluster gesture

plus `project()` (the parallel/orthogonal split that becomes the axis layout)
and `null_directions()` (the ghost).

WHAT A DIRECTION IS NOT. None of these constructors produce a causal claim. A
diff-of-means over two prompt sets is *method M applied to data D*; it is named
after its method and its data, never after the behaviour someone hopes it
explains. `Direction.label` is free text and the CLI will print whatever it is
given, but `source.protocol` is mandatory and is what the viewer shows beside
the axis, so a direction cannot arrive in the UI without the sentence that says
what was actually measured.

Directions are a SIDECAR. `nebulai.json` does not change shape and its schema
version stays at 2 — the parse worker, the goldens and every existing map are
untouched by everything in this file.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..spaces import Space, parse as parse_space
from .channels import Channel

#: written beside `nebulai.json` and `channels.json`
DIRECTIONS_FILENAME = "directions.json"

#: how the vector was obtained. Closed, for the same reason the space set is:
#: a method string nothing recognises is a method nothing can re-derive.
METHODS = (
    "diff_of_means",
    "pca",
    "sae_decoder",
    "lora_rank1",
    "probe",
    "two_selection",
)

#: the null is always the same construction, so it is always comparable:
#: uniform random unit vectors in the same space, seeded.
NULL_METHOD = "random_unit"

#: default number of null directions drawn. Enough that the ghost histogram is
#: smooth at n_points draws without making the file large.
DEFAULT_NULL_N = 32

#: null seed. Fixed so two runs of `nebulai direction project` on the same map
#: produce the same ghost — a null that moved between runs would let a reader
#: re-roll until the real distribution cleared it.
DEFAULT_NULL_SEED = 0


class DirectionError(ValueError):
    """A direction that would mislead if written. Always fatal at write time."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _unit(v: np.ndarray) -> np.ndarray:
    """Normalise to length 1, refusing a vector that has no direction.

    A zero vector is not a degenerate direction to be nudged — it is the answer
    "these two sets have the same mean", and returning an arbitrary unit vector
    there would manufacture a separation out of nothing.
    """
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if v.size == 0:
        raise DirectionError("direction vector is empty")
    if not np.isfinite(v).all():
        raise DirectionError("direction vector contains NaN or inf")
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        raise DirectionError(
            "direction vector has zero norm — the two inputs have the same "
            "mean in this space, so there is no direction between them"
        )
    return (v / n).astype(np.float32)


#: Public name for the same normalisation. Other modules fit directions too
#: (`backend/eval_awareness.py` fits one per layer and refits it 32 times for its
#: label-permutation null) and they must refuse a zero-norm difference with the
#: sentence above rather than each inventing their own tolerance.
unit = _unit


@dataclass
class Direction:
    """One unit vector, its space, its provenance, its projections, its null."""

    id: str
    label: str
    space: str
    method: str
    vector: np.ndarray
    #: {kind: computed|imported, protocol: str, …} — `protocol` is mandatory
    source: dict[str, Any] = field(default_factory=dict)
    #: {channel, orth_channel, stats} — filled by `projection_channels`
    projection: dict[str, Any] | None = None
    #: {method, seed, n, channel, orth_channel} — filled by `projection_channels`
    null: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise DirectionError(f"direction id must be a non-empty string, got {self.id!r}")
        if self.method not in METHODS:
            raise DirectionError(f"direction {self.id!r}: method {self.method!r} not in {METHODS}")
        parse_space(self.space)  # raises UnknownSpaceError outside the closed set
        self.vector = _unit(self.vector)
        if not self.source.get("protocol"):
            raise DirectionError(
                f"direction {self.id!r}: source.protocol is mandatory — a "
                f"direction with no stated protocol is a vector someone has to "
                f"take on faith"
            )
        kind = self.source.get("kind")
        if kind not in ("computed", "imported"):
            raise DirectionError(
                f"direction {self.id!r}: source.kind must be 'computed' or "
                f"'imported', got {kind!r}"
            )

    @property
    def d(self) -> int:
        return int(self.vector.shape[0])

    @property
    def space_parsed(self) -> Space:
        return parse_space(self.space)

    @property
    def projection_channel(self) -> str:
        return f"proj.{self.id}"

    @property
    def orth_channel(self) -> str:
        return f"proj.{self.id}.orth"

    @property
    def null_channel(self) -> str:
        return f"proj.{self.id}.null"

    @property
    def null_orth_channel(self) -> str:
        return f"proj.{self.id}.null.orth"

    @property
    def renderable(self) -> bool:
        """R5, mechanically: no null block, no axis. Mirrored in directions.ts."""
        return bool(self.null) and bool(self.projection)

    def to_json(self, precision: int = 6) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "space": self.space,
            "method": self.method,
            "d": self.d,
            "vector": [round(float(x), precision) for x in self.vector],
            "source": dict(self.source),
        }
        if self.projection is not None:
            out["projection"] = dict(self.projection)
        if self.null is not None:
            out["null"] = dict(self.null)
        return out

    @staticmethod
    def from_json(obj: Mapping[str, Any]) -> "Direction":
        return Direction(
            id=obj["id"],
            label=obj.get("label", obj["id"]),
            space=obj["space"],
            method=obj["method"],
            vector=np.asarray(obj["vector"], dtype=np.float32),
            source=dict(obj.get("source") or {}),
            projection=dict(obj["projection"]) if obj.get("projection") else None,
            null=dict(obj["null"]) if obj.get("null") else None,
        )


# ---------------------------------------------------------------- constructors


def diff_of_means(
    pos: np.ndarray,
    neg: np.ndarray,
    space: str,
    *,
    id: str,
    label: str,
    protocol: str,
    source: Mapping[str, Any] | None = None,
    paired: bool = False,
) -> Direction:
    """mean(pos) − mean(neg), unit-normalised. The workhorse.

    Pass `paired=True` when `pos[i]` and `neg[i]` are a matched pair, so the
    held-out split inside `contrast` keeps the pairing — see `_heldout`.

    `protocol` is the frozen identity of the two sets — a prompt-set id, a pair
    of cluster ids, anything a reader can go and re-derive. It is not optional
    because the difference of two means is only interpretable through what the
    two sets were.
    """
    p = np.asarray(pos, dtype=np.float64)
    n = np.asarray(neg, dtype=np.float64)
    if p.ndim != 2 or n.ndim != 2:
        raise DirectionError(f"diff_of_means needs two 2-D matrices, got {p.shape} and {n.shape}")
    if p.shape[1] != n.shape[1]:
        raise DirectionError(
            f"diff_of_means: pos is {p.shape[1]}-dimensional and neg is "
            f"{n.shape[1]}-dimensional — these are not the same space"
        )
    if p.shape[0] == 0 or n.shape[0] == 0:
        raise DirectionError("diff_of_means: both sets must be non-empty")
    src = {
        "kind": "computed",
        "protocol": protocol,
        "n_pos": int(p.shape[0]),
        "n_neg": int(n.shape[0]),
    }
    src.update(source or {})
    v = p.mean(axis=0) - n.mean(axis=0)
    src["contrast"] = contrast(p, n, v, paired=paired)
    return Direction(
        id=id,
        label=label,
        space=space,
        method="diff_of_means",
        vector=v,
        source=src,
    )


def pca_component(
    matrix: np.ndarray,
    k: int,
    space: str,
    *,
    id: str,
    label: str,
    protocol: str,
    source: Mapping[str, Any] | None = None,
) -> Direction:
    """The k-th principal component (0-based) of `matrix`, mean-centred first.

    The explained-variance ratio of that component travels with it in
    `source.explained_variance_ratio`. A PC that explains 2% of the variance is
    still a direction; it is just not the story, and the number is the only
    thing that lets a reader tell the difference.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2:
        raise DirectionError(f"pca_component needs a 2-D matrix, got shape {m.shape}")
    if k < 0 or k >= min(m.shape):
        raise DirectionError(
            f"pca_component: k={k} is out of range for a {m.shape[0]}×{m.shape[1]} "
            f"matrix (at most {min(m.shape)} components exist)"
        )
    if m.shape[0] < 2:
        raise DirectionError("pca_component needs at least 2 rows")
    centred = m - m.mean(axis=0, keepdims=True)
    # SVD rather than an eigendecomposition of the covariance: the covariance
    # of a tall thin matrix squares the condition number, and these matrices
    # are routinely 275×960.
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    total = float((s**2).sum())
    evr = float(s[k] ** 2 / total) if total > 0 else 0.0
    src = {
        "kind": "computed",
        "protocol": protocol,
        "k": int(k),
        "n_rows": int(m.shape[0]),
        "explained_variance_ratio": round(evr, 6),
    }
    src.update(source or {})
    return Direction(
        id=id, label=label, space=space, method="pca", vector=vt[k], source=src
    )


def from_sae_decoder(
    repo: str,
    feature_id: int,
    layer: int,
    *,
    sae_id: str | None = None,
    id: str | None = None,
    label: str | None = None,
    revision: str = "main",
) -> Direction:
    """One row of a trained SAE's decoder, as a direction in that SAE's space.

    The space tag is `sae.L<layer>.<repo>` — deliberately NOT `resid.L<layer>`,
    even though a res-* SAE's decoder rows live in the residual stream and the
    arithmetic would work. The two spaces have different bases in practice
    (normalisation, the encoder's learned scale) and D2 exists so that the
    question "are these comparable?" is answered by the tag rather than by
    whoever writes the next view.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    hook = sae_id or f"blocks.{layer}.hook_resid_pre"
    cfg_path = hf_hub_download(repo, f"{hook}/cfg.json", revision=revision)
    cfg = json.loads(Path(cfg_path).read_text())
    name = "sae_weights.safetensors" if "d_sae" in cfg else "sae.safetensors"
    w_path = hf_hub_download(repo, f"{hook}/{name}", revision=revision)
    w_dec = np.asarray(load_file(w_path)["W_dec"], dtype=np.float32)
    if feature_id < 0 or feature_id >= w_dec.shape[0]:
        raise DirectionError(
            f"feature {feature_id} is out of range for {repo}/{hook} "
            f"({w_dec.shape[0]} features)"
        )
    return Direction(
        id=id or f"sae-{layer}-{feature_id}",
        label=label or f"SAE feature {feature_id} @ {hook}",
        space=f"sae.L{layer}.{repo}",
        method="sae_decoder",
        vector=w_dec[feature_id],
        source={
            "kind": "imported",
            "protocol": f"{repo}@{revision}:{hook}:W_dec[{feature_id}]",
            "repo": repo,
            "revision": revision,
            "sae_id": hook,
            "feature_id": int(feature_id),
        },
    )


def from_lora_rank1(
    b: np.ndarray,
    a: np.ndarray,
    space: str,
    *,
    id: str,
    label: str,
    protocol: str,
    scaling: float = 1.0,
    source: Mapping[str, Any] | None = None,
) -> Direction:
    """A rank-1 LoRA update as one direction — the OUTPUT-side factor.

    A rank-1 adapter on a weight `W : in -> out` adds `dW = scaling * b @ a.T`
    with `b` of length `out` and `a` of length `in`. That matrix has exactly one
    non-zero singular value, so every output the adapter is capable of adding is
    a multiple of `b / ||b||`: there is a single direction and this is it. `a`
    decides *how much* of it each input gets, which is a gate, not a direction.

    So the vector returned is the unit `b`, and `space` must be the space the
    adapted matrix WRITES INTO (`mlp_out.L<k>` for a `down_proj`, `resid.L<k>`
    for a module whose output is added straight to the stream). `a` is not
    returned as a second direction even though it is a perfectly good unit
    vector, because it lives in the input space — 4,864-wide for a Qwen MLP —
    and tagging it with the output space to make the arithmetic typecheck is
    exactly the mistake the closed space set exists to prevent (D2). Its norm
    and the update's Frobenius norm are recorded in `source` instead, so the
    scale of what was learned is on the record without inviting a projection
    that would mean nothing.

    **There is no `path` argument, on purpose.** D6 says measure, never export:
    no run in this project writes an adapter file, so there is no checkpoint of
    ours to point this at. The factors arrive as arrays from the process that
    trained them, still in memory, and `protocol` is mandatory, so a vector
    cannot get in here without the statement of what it was trained on.
    Importing somebody else's published adapter is a different operation with a
    different `source.kind` and it belongs in `import_directions.py`, beside the
    other upstream artefacts, not here.
    """
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    if bv.size == 0 or av.size == 0:
        raise DirectionError(
            f"direction {id!r}: a rank-1 update needs both factors, got shapes "
            f"{np.shape(b)} and {np.shape(a)}"
        )
    nb, na = float(np.linalg.norm(bv)), float(np.linalg.norm(av))
    if nb <= 1e-12 or na <= 1e-12:
        raise DirectionError(
            f"direction {id!r}: the rank-1 update is zero "
            f"(||b||={nb:.3g}, ||a||={na:.3g}) — an adapter that has not moved "
            f"off its initialisation has no direction, and normalising the "
            f"rounding error would invent one"
        )
    src: dict[str, Any] = {
        "kind": "computed",
        "protocol": protocol,
        "lora": {
            "rank": 1,
            "scaling": round(float(scaling), 6),
            # ||dW||_F = scaling * ||b|| * ||a|| exactly, for a rank-1 outer
            # product; the three numbers are kept separately because only the
            # product is invariant to how the trainer split the scale.
            "b_norm": round(nb, 6),
            "a_norm": round(na, 6),
            "delta_w_fro": round(float(scaling) * nb * na, 6),
            "b_dim": int(bv.size),
            "a_dim": int(av.size),
            "side": "output",
        },
    }
    src.update(source or {})
    return Direction(
        id=id, label=label, space=space, method="lora_rank1", vector=bv, source=src
    )


def from_two_selections(
    a_ids: Sequence[Any],
    b_ids: Sequence[Any],
    vectors: np.ndarray,
    space: str,
    *,
    ids: Sequence[Any] | None = None,
    id: str | None = None,
    label: str | None = None,
    protocol: str | None = None,
) -> Direction:
    """The viewer's gesture: pick two sets of points, get their difference.

    `ids` is the id of each row of `vectors` (the map's own point ids); when it
    is None the id sets are taken as row indices. Identical for a researcher
    picking two prompt clusters and an amateur lassoing two blobs — which is R6,
    and is why this is a constructor and not a UI affordance.

    An id in neither set is ignored; an id in BOTH is an error, because the
    difference of two overlapping sets is a direction whose magnitude depends on
    the overlap in a way no caption could state honestly.
    """
    v = np.asarray(vectors, dtype=np.float64)
    if v.ndim != 2:
        raise DirectionError(f"from_two_selections needs a 2-D matrix, got {v.shape}")
    index: dict[Any, int]
    if ids is None:
        index = {i: i for i in range(v.shape[0])}
    else:
        if len(ids) != v.shape[0]:
            raise DirectionError(
                f"from_two_selections: {len(ids)} ids for {v.shape[0]} rows"
            )
        index = {k: i for i, k in enumerate(ids)}

    overlap = set(a_ids) & set(b_ids)
    if overlap:
        raise DirectionError(
            f"from_two_selections: {len(overlap)} id(s) are in both selections "
            f"(e.g. {sorted(map(str, overlap))[:3]}) — the difference of two "
            f"overlapping sets is not the difference between them"
        )
    missing = [k for k in list(a_ids) + list(b_ids) if k not in index]
    if missing:
        raise DirectionError(
            f"from_two_selections: {len(missing)} selected id(s) are not rows of "
            f"this matrix (e.g. {missing[:3]})"
        )
    rows_a = np.asarray([index[k] for k in a_ids], dtype=int)
    rows_b = np.asarray([index[k] for k in b_ids], dtype=int)
    if rows_a.size == 0 or rows_b.size == 0:
        raise DirectionError("from_two_selections: both selections must be non-empty")

    auto = f"selection:{len(rows_a)}v{len(rows_b)}"
    vec = v[rows_a].mean(axis=0) - v[rows_b].mean(axis=0)
    return Direction(
        id=id or f"sel-{len(rows_a)}v{len(rows_b)}",
        label=label or f"{len(rows_a)} points − {len(rows_b)} points",
        space=space,
        method="two_selection",
        vector=vec,
        source={
            "kind": "computed",
            "protocol": protocol or auto,
            "n_pos": int(rows_a.size),
            "n_neg": int(rows_b.size),
            "a_ids": [str(k) for k in list(a_ids)[:64]],
            "b_ids": [str(k) for k in list(b_ids)[:64]],
            "contrast": contrast(v[rows_a], v[rows_b], vec),
        },
    )


# ------------------------------------------------------------- projection/null


def project(points: np.ndarray, direction: Direction | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split every point into (parallel, orthogonal) about a unit direction.

    `parallel` is the signed coordinate along the direction, `⟨p, d⟩`.
    `orthogonal` is the length of what is left, `‖p − ⟨p, d⟩ d‖` — unsigned by
    construction, because "how far off the axis" has no side.

    Together they are the axis layout: x = parallel, y = orthogonal. The
    identity `parallel² + orthogonal² = ‖p‖²` is what makes that layout an
    honest re-coordinatisation of the point rather than a new embedding, and it
    is asserted in tests/test_directions.py.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim == 1:
        p = p[None, :]
    vec = direction.vector if isinstance(direction, Direction) else np.asarray(direction)
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    if p.shape[1] != v.shape[0]:
        raise DirectionError(
            f"cannot project {p.shape[1]}-dimensional points onto a "
            f"{v.shape[0]}-dimensional direction"
        )
    v = v / float(np.linalg.norm(v))
    par = p @ v
    resid = p - par[:, None] * v[None, :]
    orth = np.sqrt(np.maximum((resid * resid).sum(axis=1), 0.0))
    return par.astype(np.float64), orth.astype(np.float64)


def null_directions(d: int, n: int, seed: int = DEFAULT_NULL_SEED) -> np.ndarray:
    """`n` uniform random unit vectors in R^d — the ghost.

    Gaussian then normalised, which is the one construction that is actually
    uniform on the sphere (the obvious uniform-per-coordinate box is not, and
    its projections are biased towards the corners of the cube).

    R5: this is not an optional comparison a view may choose to draw. A real
    projection histogram is a bell curve of roughly ‖p‖/√d for ANY direction;
    the null is what says whether the real one is wider, or shifted, or the same
    curve with a different label on it.
    """
    if d <= 0 or n <= 0:
        raise DirectionError(f"null_directions needs d>0 and n>0, got d={d}, n={n}")
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, d))
    return (g / np.linalg.norm(g, axis=1, keepdims=True)).astype(np.float32)


def separation(real: np.ndarray, null: np.ndarray) -> dict[str, float]:
    """How far the real projection distribution is from its null.

    Two numbers, both dimensionless, both computed here rather than in the
    browser so the rail's caption and the CLI's summary cannot disagree:

    * `cohens_d` — the standardised mean difference. Signed: a direction whose
      projections sit *below* the null gets a negative number, not an absolute
      value that hides which way it went.
    * `overlap` — the overlapping area of the two histograms on a shared
      binning, in [0, 1]. 1.0 means the ghost and the real distribution are the
      same picture, which is the outcome the rail has to be able to show.
    """
    a = np.asarray(real, dtype=np.float64)
    b = np.asarray(null, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return {"cohens_d": math.nan, "overlap": math.nan, "n": int(a.size)}
    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    pooled = math.sqrt(((a.size - 1) * va + (b.size - 1) * vb) / (a.size + b.size - 2))
    dstat = (float(a.mean()) - float(b.mean())) / pooled if pooled > 0 else math.nan
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi <= lo:
        return {"cohens_d": dstat, "overlap": 1.0, "n": int(a.size)}
    edges = np.linspace(lo, hi, 65)
    ha, _ = np.histogram(a, bins=edges)
    hb, _ = np.histogram(b, bins=edges)
    ov = float(np.minimum(ha / a.size, hb / b.size).sum())
    return {"cohens_d": round(dstat, 6), "overlap": round(ov, 6), "n": int(a.size)}


def contrast(
    pos: np.ndarray,
    neg: np.ndarray,
    vector: np.ndarray,
    *,
    n_null: int = DEFAULT_NULL_N,
    seed: int = DEFAULT_NULL_SEED,
    paired: bool = False,
) -> dict[str, float]:
    """How well this direction separates the two sets it was built from — and
    how well a random direction does on the same two sets.

    `separation()` answers a different question: whether the direction spreads
    the *whole map* more than a random axis does. For a contrast between two
    small clusters the honest answer there is usually "no", because most of the
    vocabulary is not about the contrast. This function answers the question
    the direction actually makes a claim about, and it ships its own null so
    the claim is never read alone (R5).

    `cohens_d` is pos-vs-neg along the direction; `null_cohens_d_mean` and
    `null_cohens_d_p95` are |d| for `n_null` random unit directions on the same
    two sets. A real contrast is one where the first number leaves the last one
    far behind.
    """
    a = np.asarray(pos, dtype=np.float64)
    b = np.asarray(neg, dtype=np.float64)
    v = _unit(np.asarray(vector, dtype=np.float64))
    real = separation(a @ v, b @ v)
    nulls = null_directions(v.size, n_null, seed).astype(np.float64)
    ds = []
    for k in range(nulls.shape[0]):
        st = separation(a @ nulls[k], b @ nulls[k])
        if math.isfinite(st["cohens_d"]):
            ds.append(abs(st["cohens_d"]))
    arr = np.asarray(ds, dtype=np.float64)
    out = {
        "cohens_d": real["cohens_d"],
        "overlap": real["overlap"],
        "in_sample": True,
        "n_pos": int(a.shape[0]),
        "n_neg": int(b.shape[0]),
        "null_n": int(arr.size),
        "null_seed": int(seed),
        "null_cohens_d_mean": round(float(arr.mean()), 6) if arr.size else math.nan,
        "null_cohens_d_p95": round(float(np.percentile(arr, 95)), 6) if arr.size else math.nan,
    }
    out.update(_heldout(a, b, seed=seed, paired=paired))
    return out


def _heldout(
    a: np.ndarray, b: np.ndarray, *, seed: int, paired: bool = False
) -> dict[str, float]:
    """Refit on half of each set, measure on the other half.

    The in-sample number above is the direction scored on the very points that
    defined it, so it cannot be small — a random direction has no such
    advantage, which makes the null comparison flattering rather than fair.
    These four numbers are the only ones in the block that a reader may treat
    as a measurement. When either set has fewer than four members there is no
    split to make and they come back `missing` rather than as an optimistic
    stand-in.

    `paired=True` is for a MATCHED design, where `pos[i]` and `neg[i]` differ in
    exactly one thing and splitting the two sides independently is wrong. It is
    wrong in a specific, measured way: independent halves put different items on
    the two sides, so the fitted difference picks up an (items-in-fit-pos minus
    items-in-fit-neg) term, and the test halves are exactly the complements — so
    that term comes back with the opposite sign. On the eval-awareness prompt set
    (64 matched pairs, SmolLM2-135M-Instruct, 40 seeds) the mean held-out Cohen's
    d at layers 4/12/19/25 was +0.13 / −0.28 / −0.38 / −0.40 with independent
    splits and +0.52 / +0.89 / +0.32 / +0.31 with paired ones. The negative
    numbers were an artefact of breaking the pairing, and anything reading them as
    "the direction does not transfer" would have been reading the split.

    Default stays `False`: a caller only knows its two sets are paired if its
    design says so, and `heldout_split` records which was done either way.
    """
    if a.shape[0] < 4 or b.shape[0] < 4:
        return {
            "heldout_cohens_d": "missing",
            "heldout_overlap": "missing",
            "heldout_n_pos": 0,
            "heldout_n_neg": 0,
            "heldout_split": "missing",
        }
    if paired and a.shape[0] != b.shape[0]:
        raise DirectionError(
            f"paired=True needs one neg per pos, got {a.shape[0]} and {b.shape[0]}"
        )
    rng = np.random.default_rng(seed)
    if paired:
        pa = pb = rng.permutation(a.shape[0])
    else:
        pa = rng.permutation(a.shape[0])
        pb = rng.permutation(b.shape[0])
    ha, hb = a.shape[0] // 2, b.shape[0] // 2
    fit_a, test_a = a[pa[:ha]], a[pa[ha:]]
    fit_b, test_b = b[pb[:hb]], b[pb[hb:]]
    v = _unit(fit_a.mean(axis=0) - fit_b.mean(axis=0))
    st = separation(test_a @ v, test_b @ v)
    return {
        "heldout_cohens_d": st["cohens_d"],
        "heldout_overlap": st["overlap"],
        "heldout_n_pos": int(test_a.shape[0]),
        "heldout_n_neg": int(test_b.shape[0]),
        "heldout_split": "paired" if paired else "independent",
    }


def projection_channels(
    direction: Direction,
    points: np.ndarray,
    *,
    n_null: int = DEFAULT_NULL_N,
    seed: int = DEFAULT_NULL_SEED,
) -> list[Channel]:
    """The four channels a direction needs to be renderable, and fills in
    `direction.projection` / `direction.null` in place.

    Real: `proj.<id>` (parallel) and `proj.<id>.orth`.
    Null: `proj.<id>.null` and `proj.<id>.null.orth`.

    The null channels are ONE draw per point — point i is projected onto null
    direction `i % n_null`. Pooled over the map that is exactly the null
    distribution of a projection, at exactly the same n as the real one, which
    is what makes the ghost histogram comparable rather than merely adjacent.
    Storing all n_null × n_points values instead would multiply the sidecar by
    32 to draw the same curve.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2:
        raise DirectionError(f"projection needs a 2-D point matrix, got {p.shape}")
    if p.shape[1] != direction.d:
        raise DirectionError(
            f"direction {direction.id!r} is {direction.d}-dimensional but this "
            f"map's points are {p.shape[1]}-dimensional — refusing to project "
            f"(D2: a dimensionality match is necessary, not sufficient; the "
            f"space tags must agree too)"
        )
    par, orth = project(p, direction)

    nulls = null_directions(direction.d, n_null, seed)
    assign = np.arange(p.shape[0]) % n_null
    npar = np.einsum("ij,ij->i", p, nulls[assign].astype(np.float64))
    nresid = p - npar[:, None] * nulls[assign].astype(np.float64)
    north = np.sqrt(np.maximum((nresid * nresid).sum(axis=1), 0.0))

    stats = separation(par, npar)
    direction.projection = {
        "channel": direction.projection_channel,
        "orth_channel": direction.orth_channel,
        "stats": stats,
    }
    direction.null = {
        "method": NULL_METHOD,
        "seed": int(seed),
        "n": int(n_null),
        "channel": direction.null_channel,
        "orth_channel": direction.null_orth_channel,
    }

    common = dict(space=direction.space, fidelity="deterministic", units="dot")
    return [
        Channel(
            id=direction.projection_channel,
            label=f"⟨p, {direction.id}⟩",
            method="projection",
            formula=f"dot(p, d) where d = {direction.id}",
            values=par,
            extra={"direction": direction.id, "role": "parallel"},
            **common,
        ),
        Channel(
            id=direction.orth_channel,
            label=f"‖p − ⟨p, {direction.id}⟩ d‖",
            method="projection",
            formula=f"norm(p - dot(p, d) * d) where d = {direction.id}",
            values=orth,
            extra={"direction": direction.id, "role": "orthogonal"},
            **common,
        ),
        Channel(
            id=direction.null_channel,
            label=f"⟨p, random unit⟩ (null, seed {seed})",
            method="projection_null",
            formula=f"dot(p, n_i) where n_i is null direction i%{n_null}, seed {seed}",
            values=npar,
            extra={"direction": direction.id, "role": "parallel", "null": True},
            **common,
        ),
        Channel(
            id=direction.null_orth_channel,
            label=f"‖p − ⟨p, random unit⟩ n‖ (null, seed {seed})",
            method="projection_null",
            formula=f"norm(p - dot(p, n_i) * n_i), seed {seed}",
            values=north,
            extra={"direction": direction.id, "role": "orthogonal", "null": True},
            **common,
        ),
    ]


# ------------------------------------------------------------------- registry


def write_directions(
    path: str | Path,
    *,
    model: str,
    revision: str,
    directions: Sequence[Direction],
    merge: bool = True,
) -> Path:
    """Write (or update) `directions.json`.

    `merge=True` keeps directions already on file whose ids are not being
    rewritten — the same lifetime argument as `write_channels`: a direction is
    added without recomputing a map, and adding a second one must not silently
    delete the first.
    """
    path = Path(path)
    seen: set[str] = set()
    for dr in directions:
        if dr.id in seen:
            raise DirectionError(f"duplicate direction id {dr.id!r}")
        seen.add(dr.id)

    existing: list[dict[str, Any]] = []
    if merge and path.exists():
        prev = read_directions(path)
        if prev is not None and (prev.get("meta") or {}).get("model") == model:
            new_ids = {d.id for d in directions}
            existing = [
                d
                for d in prev.get("directions", [])
                if isinstance(d, dict) and d.get("id") not in new_ids
            ]

    doc = {
        "meta": {"model": model, "revision": revision, "created": _now()},
        "directions": existing + [d.to_json() for d in directions],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False))
    return path


def read_directions(path: str | Path) -> dict[str, Any] | None:
    """Parse `directions.json`, or None when absent or unreadable.

    Absence is a supported state — a map with no directions renders no axis UI,
    which is "not measured", never "measured as flat".
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict) or "directions" not in doc:
        return None
    return doc


def drop_directions(path: str | Path, ids: Sequence[str]) -> int:
    """Remove directions by id, in place. Returns how many were removed.

    Their channels are NOT removed here — `nebulai direction drop` calls
    `channels.drop_channels` separately, because the two files are separately
    readable and a caller that only wants the registry entry gone should not
    have to guess which four channel ids went with it.
    """
    path = Path(path)
    doc = read_directions(path)
    if doc is None:
        return 0
    keep = [d for d in doc.get("directions", []) if d.get("id") not in set(ids)]
    removed = len(doc.get("directions", [])) - len(keep)
    doc["directions"] = keep
    path.write_text(json.dumps(doc, ensure_ascii=False))
    return removed


def renderable(
    directions_doc: Mapping[str, Any] | None,
    channels_doc: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Split a registry into what can be drawn and what cannot, with reasons.

    The two rules of §3.2, enforced here and re-checked identically in
    `viewer/src/data/directions.ts` (the drift between the two is what
    `directions.test.ts` pins):

    1. a direction with no `null` block is not renderable — R5 is mechanical;
    2. `projection.channel` must exist in `channels.json` AND carry the same
       space — a projection computed elsewhere is not this map's projection.

    Returns `(ok, drops)` where each drop is `(id, reason)`. The reason is
    returned rather than logged because the UI states it: a direction that
    vanished silently is indistinguishable from one that was never written.
    """
    if not directions_doc:
        return [], []
    by_id: dict[str, Mapping[str, Any]] = {}
    for c in (channels_doc or {}).get("channels", []) or []:
        if isinstance(c, dict) and "id" in c:
            by_id[c["id"]] = c

    ok: list[dict[str, Any]] = []
    drops: list[tuple[str, str]] = []
    for d in directions_doc.get("directions", []) or []:
        if not isinstance(d, dict) or "id" not in d:
            continue
        did = str(d["id"])
        null = d.get("null")
        if not null or not null.get("channel"):
            drops.append((did, "no null block — R5: a direction ships with its null or not at all"))
            continue
        proj = d.get("projection") or {}
        pch = proj.get("channel")
        if not pch:
            drops.append((did, "no projection channel — run `nebulai direction project`"))
            continue
        for ch_id, what in ((pch, "projection"), (null.get("channel"), "null")):
            if ch_id not in by_id:
                drops.append((did, f"{what} channel {ch_id!r} is not in channels.json"))
                break
            if by_id[ch_id].get("space") != d.get("space"):
                drops.append(
                    (
                        did,
                        f"{what} channel {ch_id!r} is in space "
                        f"{by_id[ch_id].get('space')!r} but the direction is in "
                        f"{d.get('space')!r} (D2)",
                    )
                )
                break
        else:
            ok.append(dict(d))
    return ok, drops
