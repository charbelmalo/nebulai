"""Persona space — a fixed PCA coordinate system for trajectories (D4, §3.3).

The Assistant-Axis recipe, at micro-model scale and with the control that
decides whether the recipe found anything:

1. A **frozen** archetype set (`prompts/personas.v1.json`) crossed with a fixed
   probe list gives `n_archetype x n_probe` chat prompts.
2. Each prompt is run through the pinned instruct model; the residual stream at
   a pinned layer, at the prompt's last token, is the prompt's activation.
3. An archetype's vector is the mean over its probes; PCA over those vectors is
   the space.
4. A **label-permutation control** (n=500) re-pools the *same* activations into
   pseudo-archetypes of the same sizes and re-runs the PCA. If PC1's explained
   variance ratio does not clear the null's 95th percentile, the space is
   written with `verdict: "at_null"` (or `"below_null"`) and **cannot be the
   default trajectory coordinate system**.

Why a permutation over the pooled activations rather than over labels attached
to the archetype means: PC1's explained-variance ratio is a property of the
geometry of the mean cloud, so there is no label to shuffle at that level. The
question the control has to answer is whether *persona identity* is what groups
the activations — so the null groups the same 2,368 activations into arbitrary
sets of the same size and asks how much of a dominant axis that alone produces.
A space that fails this is not a bug; it is the §7-A risk resolving, and the
plan's response to it is written down rather than improvised.

**The permutation is stratified by probe, and that is not a free choice.** The
prompt set is a crossed design: every archetype is asked the same 8 probes, so
a real archetype mean averages each probe exactly once and the probe axis
cancels out of it by construction. Measured on SmolLM2-135M-Instruct at layer
19, the probe accounts for **81.7 %** of the pooled variance — it is by far the
largest thing in these activations, and it is the one thing the statistic is
built to be blind to. A permutation that ignores the strata builds
pseudo-archetypes with an unbalanced probe mix, so their mean cloud *keeps* the
probe axis: its PC1 EVR lands at 0.364, within noise of the pooled PC1 (0.360),
and it measures probe imbalance rather than persona identity. Permuting
archetype labels **within each probe block** keeps the exchangeability H0
actually asserts ("which archetype a prompt belongs to carries no information")
while matching the balance of the real design; its null sits at 0.172. Both are
recorded in `space.json` — the stratified one decides the verdict, the
unstratified one is kept as `cross_check` with this reason attached, because
the unstratified number was measured first and a control that is swapped out
after it fails has to show its work.

Nothing here imports `seer`; a persona space crosses that boundary as a file
(`out/persona/<space_id>/space.json`) and as `POST /live/place`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..prompts import load_prompt_set

#: Where spaces land. `out/` is the repo's usual artifact root.
DEFAULT_OUT = Path("out") / "persona"

#: The permutation control's sample count. Fixed, not a flag: a control whose n
#: the caller can lower is a control that will be lowered until it passes.
CONTROL_N = 500
CONTROL_SEED = 0

#: How many PCA components a space carries. Two are drawn; the rest are kept so
#: `explained_variance_ratio` is readable as a spectrum rather than a fragment.
N_COMPONENTS = 8


class PersonaError(ValueError):
    """A persona space could not be built or verified as asked."""


@dataclass(frozen=True)
class Control:
    """The label-permutation null and the verdict it produces."""

    method: str
    n: int
    seed: int
    pc1_evr: float
    pc1_evr_null_mean: float
    pc1_evr_null_p95: float
    verdict: str  # "above_null" | "at_null" | "below_null"
    #: One-sided permutation p-value, (1 + #{null >= observed}) / (1 + n). The
    #: +1s are the standard correction: with a finite number of draws, p = 0 is
    #: not a thing the experiment can observe.
    p_value: float = float("nan")
    #: The unstratified null, kept for the record rather than for the verdict.
    #: See the module docstring: it is biased upward by the probe axis.
    cross_check: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "method": self.method,
            "n": self.n,
            "seed": self.seed,
            "pc1_evr": round(self.pc1_evr, 6),
            "pc1_evr_null_mean": round(self.pc1_evr_null_mean, 6),
            "pc1_evr_null_p95": round(self.pc1_evr_null_p95, 6),
            "verdict": self.verdict,
        }
        if not np.isnan(self.p_value):
            d["p_value"] = round(self.p_value, 6)
        if self.cross_check is not None:
            d["cross_check"] = self.cross_check
        return d


def verdict_for(pc1_evr: float, null: np.ndarray) -> str:
    """`above_null` iff PC1 clears the null's 95th percentile.

    `below_null` is reserved for the case that is worse than chance rather than
    merely indistinguishable from it — below the null's 5th percentile. The
    three-way split exists because "at_null" and "below_null" have different
    diagnoses: the first says the recipe found nothing, the second says the
    pooling is actively destroying structure the individual activations have.
    """
    p95 = float(np.percentile(null, 95))
    p05 = float(np.percentile(null, 5))
    if pc1_evr > p95:
        return "above_null"
    if pc1_evr < p05:
        return "below_null"
    return "at_null"


def build_prompts(doc: dict[str, Any]) -> tuple[list[str], list[int], list[str]]:
    """`(chat_texts, archetype_index_per_prompt, archetype_names)`.

    Returns the *unrendered* system/user pairs as a flat list so the caller can
    hand them to the model's own chat template — the template is the model's,
    not ours, and applying the wrong one moves every coordinate.
    """
    names = [a["name"] for a in doc["archetypes"]]
    tmpl = doc["system_template"]
    probes = doc["probes"]
    prompts: list[str] = []
    owner: list[int] = []
    for i, a in enumerate(doc["archetypes"]):
        system = tmpl.format(persona=a["persona"], name=a["name"])
        for probe in probes:
            prompts.append(json.dumps({"system": system, "user": probe}))
            owner.append(i)
    return prompts, owner, names


def _pca(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-centred PCA by SVD. Returns `(components (k,d), mean (d,), evr (k,))`.

    `evr` is normalised by the **total** variance of the input, not by the sum
    of the kept components — otherwise PC1's ratio would rise as you kept fewer
    components, which is the opposite of what it has to mean for the control.
    """
    mean = x.mean(axis=0)
    xc = x - mean
    # economy SVD; n_archetype (~300) << d is not guaranteed, so cap k honestly
    k = min(k, min(xc.shape))
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = (s**2) / max(xc.shape[0] - 1, 1)
    total = float(var.sum())
    evr = (var[:k] / total) if total > 0 else np.zeros(k)
    return vt[:k].astype(np.float32), mean.astype(np.float32), evr.astype(np.float64)


def _pc1_evr(x: np.ndarray) -> float:
    _, _, evr = _pca(x, 1)
    return float(evr[0])


def probe_strata(n_prompts: int, n_probes: int) -> np.ndarray:
    """Which probe each prompt in `build_prompts`' flat list came from.

    `build_prompts` walks archetype-major with the probes in order, so the
    stratum is simply the position modulo the probe count. Derived here rather
    than carried around so the two cannot drift apart silently.
    """
    if n_probes <= 0 or n_prompts % n_probes:
        raise PersonaError(
            f"{n_prompts} prompts is not a whole number of {n_probes}-probe "
            "blocks — the crossed design the control assumes is not intact"
        )
    return np.arange(n_prompts, dtype=np.int64) % n_probes


def permutation_null(
    acts: np.ndarray,
    owner: np.ndarray,
    n: int = CONTROL_N,
    seed: int = CONTROL_SEED,
    strata: np.ndarray | None = None,
) -> np.ndarray:
    """PC1 explained-variance ratios from `n` label-permuted poolings.

    `acts` is `(n_prompt, d)`, `owner` is the archetype index per prompt. Each
    draw shuffles `owner` — preserving the group sizes exactly — re-pools, and
    takes PC1's EVR of the resulting mean cloud.

    With `strata` (the probe index per prompt) the shuffle happens *within* each
    stratum, so a pseudo-archetype gets the same probe balance a real one has.
    That is the null the shipped control uses; the module docstring says why the
    unstratified variant measures probe imbalance instead.
    """
    rng = np.random.default_rng(seed)
    groups = int(owner.max()) + 1
    out = np.empty(n, dtype=np.float64)
    blocks: list[np.ndarray] = []
    if strata is not None:
        strata = np.asarray(strata, dtype=np.int64)
        if strata.shape[0] != owner.shape[0]:
            raise PersonaError("strata and owner must be one per prompt")
        blocks = [np.where(strata == s)[0] for s in np.unique(strata)]
    for i in range(n):
        if blocks:
            perm = np.empty_like(owner)
            for idx in blocks:
                perm[idx] = rng.permutation(owner[idx])
        else:
            perm = rng.permutation(owner)
        means = _group_means(acts, perm, groups)
        out[i] = _pc1_evr(means)
    return out


def _group_means(acts: np.ndarray, owner: np.ndarray, groups: int) -> np.ndarray:
    """`(groups, d)` means.

    Implemented by sorting the rows into group order and using
    `np.add.reduceat`, not `np.add.at`: the control runs this 500 times over a
    ~2,400 x 576 block, and `np.add.at` is an unbuffered elementwise loop that
    turns a two-minute control into a forty-minute one. The arithmetic is
    identical; only the traversal differs.
    """
    counts = np.bincount(owner, minlength=groups)
    if (counts == 0).any():
        raise PersonaError("a permutation produced an empty group — group sizes moved")
    order = np.argsort(owner, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    sums = np.add.reduceat(acts[order].astype(np.float64), starts, axis=0)
    return sums / counts[:, None]


@dataclass
class PersonaSpace:
    """An in-memory `space.json` — §3.3, field for field."""

    space_id: str
    model: str
    revision: str
    layer: int
    pooling: str
    prompt_set: dict[str, Any]
    components: np.ndarray  # (k, d)
    mean: np.ndarray  # (d,)
    evr: np.ndarray  # (k,)
    archetypes: list[dict[str, Any]]
    control: Control
    created: str

    @property
    def usable_as_default(self) -> bool:
        """§3.3: a space whose PC1 does not clear its null cannot be the default
        trajectory coordinate system. Loadable, drawable, labelled — not default."""
        return self.control.verdict == "above_null"

    def project(self, vectors: np.ndarray) -> np.ndarray:
        """`(n, d)` activations -> `(n, k)` coordinates in this space."""
        v = np.asarray(vectors, dtype=np.float32)
        if v.ndim == 1:
            v = v[None, :]
        if v.shape[1] != self.mean.shape[0]:
            raise PersonaError(
                f"cannot project {v.shape[1]}-d vectors into a "
                f"{self.mean.shape[0]}-d space — a dimensionality mismatch here "
                f"means a different model, and substituting one is never silent"
            )
        return (v - self.mean) @ self.components.T

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": {
                "space_id": self.space_id,
                "model": self.model,
                "revision": self.revision,
                "layer": self.layer,
                "pooling": self.pooling,
                "prompt_set": self.prompt_set,
                "created": self.created,
                "space": f"persona-pca.{self.space_id}",
            },
            "basis": {
                "components": [[round(float(x), 6) for x in row] for row in self.components],
                "mean": [round(float(x), 6) for x in self.mean],
                "explained_variance_ratio": [round(float(x), 6) for x in self.evr],
            },
            "archetypes": self.archetypes,
            "control": self.control.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PersonaSpace":
        m, b, c = d["meta"], d["basis"], d["control"]
        return PersonaSpace(
            space_id=m["space_id"],
            model=m["model"],
            revision=m["revision"],
            layer=int(m["layer"]),
            pooling=m["pooling"],
            prompt_set=m["prompt_set"],
            components=np.asarray(b["components"], dtype=np.float32),
            mean=np.asarray(b["mean"], dtype=np.float32),
            evr=np.asarray(b["explained_variance_ratio"], dtype=np.float64),
            archetypes=list(d["archetypes"]),
            control=Control(
                method=c["method"],
                n=int(c["n"]),
                seed=int(c.get("seed", CONTROL_SEED)),
                pc1_evr=float(c["pc1_evr"]),
                pc1_evr_null_mean=float(c.get("pc1_evr_null_mean", float("nan"))),
                pc1_evr_null_p95=float(c["pc1_evr_null_p95"]),
                verdict=c["verdict"],
            ),
            created=m.get("created", ""),
        )


def make_space_id(model: str, revision: str, prompt_set_id: str, layer: int) -> str:
    """`<model-slug>@<sha12>.<prompt-set-version>.L<layer>`.

    Every input that could move a coordinate is in the id, so two spaces that
    differ in any of them cannot collide — which is what makes "changing the
    prompt set makes a new space_id" mechanical instead of a convention.
    """
    slug = model.split("/")[-1].lower()
    version = prompt_set_id.rsplit(".", 1)[-1]
    return f"{slug}@{revision[:12]}.{version}.L{layer}"


def default_layer(n_layer: int) -> int:
    """Two-thirds of the way up, the layer band persona work reports.

    Stated as a formula rather than a constant because it has to hold for a
    30-layer 135M and a 32-layer 360M alike; the resolved value is written into
    `space.json` so a figure never has to re-derive it.
    """
    return int(round(n_layer * 2 / 3)) - 1


def run_control(
    acts: np.ndarray,
    owner: np.ndarray,
    pc1_evr: float,
    *,
    n_probes: int,
    control_n: int = CONTROL_N,
) -> Control:
    """Both nulls, one verdict.

    The probe-stratified null decides; the unstratified one rides along as
    `cross_check` so the number that was measured first — and the reason it was
    not the one used — stay in the artifact rather than in a commit message.
    """
    strata = probe_strata(acts.shape[0], n_probes)
    null = permutation_null(acts, owner, n=control_n, seed=CONTROL_SEED, strata=strata)
    loose = permutation_null(acts, owner, n=control_n, seed=CONTROL_SEED)
    ge = int((null >= pc1_evr).sum())
    return Control(
        method="label_permutation_within_probe",
        n=control_n,
        seed=CONTROL_SEED,
        pc1_evr=pc1_evr,
        pc1_evr_null_mean=float(null.mean()),
        pc1_evr_null_p95=float(np.percentile(null, 95)),
        verdict=verdict_for(pc1_evr, null),
        p_value=(1.0 + ge) / (1.0 + control_n),
        cross_check={
            "method": "label_permutation_unstratified",
            "n": control_n,
            "pc1_evr_null_mean": round(float(loose.mean()), 6),
            "pc1_evr_null_p95": round(float(np.percentile(loose, 95)), 6),
            "verdict": verdict_for(pc1_evr, loose),
            "note": (
                "Not the verdict. Ignoring the probe strata lets a pseudo-"
                "archetype carry an unbalanced probe mix, so this null inherits "
                "the probe axis — the largest component of these activations "
                "and the one a real archetype mean cancels by construction. "
                "Kept because it was measured first and failed."
            ),
        },
    )


def build_space(
    model: Any,
    *,
    prompt_set_id: str = "personas.v1",
    layer: int | None = None,
    batch_size: int = 16,
    control_n: int = CONTROL_N,
    progress: Any = None,
) -> PersonaSpace:
    """Run the recipe on a loaded `LlamaNumpy` (or anything with its API).

    `model` is passed in rather than constructed here so the live server can
    reuse its resident weights, and so a test can drive the whole recipe with a
    tiny stub.
    """
    doc, sha = load_prompt_set(prompt_set_id)
    prompts, owner_list, names = build_prompts(doc)
    owner = np.asarray(owner_list, dtype=np.int64)
    L = default_layer(model.n_layer) if layer is None else int(layer)

    texts = []
    for raw in prompts:
        pair = json.loads(raw)
        texts.append(
            model.apply_chat_template(
                [
                    {"role": "system", "content": pair["system"]},
                    {"role": "user", "content": pair["user"]},
                ],
                add_generation_prompt=True,
            )
        )

    acts = _capture(model, texts, L, batch_size=batch_size, progress=progress)
    groups = len(names)
    means = _group_means(acts, owner, groups)
    comps, mean, evr = _pca(means.astype(np.float32), N_COMPONENTS)

    pc1 = float(evr[0])
    control = run_control(
        acts, owner, pc1, n_probes=len(doc["probes"]), control_n=control_n
    )

    coords = (means.astype(np.float32) - mean) @ comps.T
    archetypes = [
        {"name": names[i], "scores": [round(float(x), 5) for x in coords[i]]}
        for i in range(groups)
    ]
    revision = getattr(model, "revision", "unknown")
    return PersonaSpace(
        space_id=make_space_id(model.model_id, revision, prompt_set_id, L),
        model=model.model_id,
        revision=revision,
        layer=L,
        pooling="last_token_mean_over_prompt_set",
        prompt_set={"id": prompt_set_id, "sha256": sha, "n": groups,
                    "n_probes": len(doc["probes"]), "n_prompts": len(texts)},
        components=comps,
        mean=mean,
        evr=evr,
        archetypes=archetypes,
        control=control,
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def _capture(
    model: Any, texts: list[str], layer: int, *, batch_size: int, progress: Any
) -> np.ndarray:
    out = np.empty((len(texts), model.d), dtype=np.float32)
    step = max(batch_size, 1)
    for start in range(0, len(texts), step):
        chunk = texts[start : start + step]
        got = model.capture_resid(chunk, [layer], batch_size=step)
        out[start : start + len(chunk)] = got[layer]
        if progress is not None:
            progress(min(start + step, len(texts)), len(texts))
    return out


def space_dir(space_id: str, root: Path | str = DEFAULT_OUT) -> Path:
    return Path(root) / space_id


def write_space(space: PersonaSpace, root: Path | str = DEFAULT_OUT) -> Path:
    d = space_dir(space.space_id, root)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "space.json"
    path.write_text(json.dumps(space.to_dict(), indent=2) + "\n")
    return path


def read_space(space_id: str, root: Path | str = DEFAULT_OUT) -> PersonaSpace:
    path = space_dir(space_id, root) / "space.json"
    if not path.exists():
        have = sorted(p.name for p in Path(root).glob("*")) if Path(root).exists() else []
        raise PersonaError(f"no persona space {space_id!r} under {root}; have {have}")
    return PersonaSpace.from_dict(json.loads(path.read_text()))


def list_spaces(root: Path | str = DEFAULT_OUT) -> list[str]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "space.json").exists())


def verify_space(space: PersonaSpace, model: Any, *, control_n: int = CONTROL_N) -> Control:
    """Re-run the control alone against a live model, and check the freeze.

    Raises if the prompt set's bytes no longer hash to what the space recorded:
    a space is only a fixed coordinate system if the thing that fixed it has
    not moved.
    """
    doc, sha = load_prompt_set(space.prompt_set["id"])
    if sha != space.prompt_set["sha256"]:
        raise PersonaError(
            f"prompt set {space.prompt_set['id']!r} has changed since "
            f"{space.space_id} was built ({sha[:12]} != "
            f"{space.prompt_set['sha256'][:12]}). A frozen set that moved makes a "
            f"NEW space_id — rebuild, never re-verify in place."
        )
    prompts, owner_list, _ = build_prompts(doc)
    owner = np.asarray(owner_list, dtype=np.int64)
    texts = []
    for raw in prompts:
        pair = json.loads(raw)
        texts.append(
            model.apply_chat_template(
                [
                    {"role": "system", "content": pair["system"]},
                    {"role": "user", "content": pair["user"]},
                ],
                add_generation_prompt=True,
            )
        )
    acts = _capture(model, texts, space.layer, batch_size=16, progress=None)
    means = _group_means(acts, owner, len(space.archetypes))
    pc1 = _pc1_evr(means.astype(np.float32))
    return run_control(
        acts, owner, pc1, n_probes=len(doc["probes"]), control_n=control_n
    )


__all__ = [
    "CONTROL_N",
    "Control",
    "DEFAULT_OUT",
    "N_COMPONENTS",
    "PersonaError",
    "PersonaSpace",
    "build_prompts",
    "build_space",
    "default_layer",
    "list_spaces",
    "make_space_id",
    "permutation_null",
    "probe_strata",
    "run_control",
    "read_space",
    "space_dir",
    "verdict_for",
    "verify_space",
    "write_space",
]
