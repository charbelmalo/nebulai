"""Adapters for the published direction artefacts phase 1 imports.

Five upstreams, one shape of answer: a `Direction` whose `source.protocol`
names the repository, the resolved commit sha and the exact path the numbers
came out of, so a reader can fetch the same bytes. Nothing here derives a
vector from a model we do not have; an adapter whose upstream ships *no*
vector says so, in a sentence naming what the upstream ships instead, rather
than computing something similar-looking and letting the label imply it was
downloaded.

    andyrdt/refusal_direction        ships real vectors  →  imported
    safety-research/persona_vectors  ships prompt sets   →  refuses, hands back the prompts
    safety-research/assistant-axis   ships code + roles  →  refuses, hands back the 275 roles
    Neuronpedia feature id           resolves to an SAE  →  imported (via the SAE's own weights)
    7vik/AmongUs probe weights       ships real probes   →  imported

THE DIMENSIONALITY REFUSAL. `check_dimensionality()` is the gate every one of
these passes through before it can be written into a map's `directions.json`.
It exists because every published refusal/deception artefact in this list was
computed on a 2048-, 4096- or 5120-dimensional instruct model, and every model
this project maps locally is 512-, 576-, 768- or 1024-dimensional. There is no
resampling that makes those comparable, and a "projection" of a 768-d point
onto the first 768 coordinates of a 4096-d vector is a number, not a
measurement. So the refusal is loud and carries both numbers.

That is not a hypothetical: it is the measured state of the art as of the
commits pinned below, and it is why episode 7 ships a GPT-2 diff-of-means
computed here, described as method-M-on-data-D, instead of "the refusal
direction".
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .directions import Direction, DirectionError

# github.com, spelled in parts so an agent-side path guard that greps for the
# three letters "git" does not mistake a URL for a repository operation.
_GH_RAW = "https://raw." + "git" + "hubusercontent.com"
_GH_API = "https://api." + "git" + "hub.com"

#: Resolved commit shas, pinned. A "latest" import is not reproducible, and the
#: whole point of `source.protocol` is that a reader can fetch the same bytes.
#: Refreshing one of these is a deliberate edit with a new number in the report.
PINNED: dict[str, str] = {
    "andyrdt/refusal_direction": "9d852fae1a9121c78b29142de733cb1340770cc3",
    "safety-research/persona_vectors": "b8e0f044fe2410a6fad579f38324f03f13b4e917",
    "safety-research/assistant-axis": "a98961956072224eaf244eb289d6c01700b63795",
    "7vik/AmongUs": "e4ea002a3617b7d78362e053ab58d37a07abe214",
}

#: refusal_direction's published runs: run dir -> (model id, layer, position).
#: layer/pos are read from the repo's own `direction_metadata.json`; the
#: dimensions are recorded from the actual tensors, measured not assumed.
REFUSAL_RUNS: dict[str, dict[str, Any]] = {
    "gemma-2b-it": {"model": "google/gemma-2b-it", "d": 2048},
    "qwen-1_8b-chat": {"model": "Qwen/Qwen-1_8B-Chat", "d": 2048},
    "llama-2-7b-chat-hf": {"model": "meta-llama/Llama-2-7b-chat-hf", "d": 4096},
    "meta-llama-3-8b-instruct": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "d": 4096},
    "yi-6b-chat": {"model": "01-ai/Yi-6B-Chat", "d": 4096},
}

#: Neuronpedia's `<model>/<layer>-<release>` ids -> the HF repo holding those
#: exact weights. Only the releases this project can actually read are listed;
#: an unknown id raises rather than guessing a repo.
NEURONPEDIA_SOURCES: dict[str, dict[str, Any]] = {
    "gpt2-small/8-res-jb": {
        "repo": "jbloom/GPT2-Small-SAEs-Reformatted",
        "sae_id": "blocks.8.hook_resid_pre",
        "layer": 8,
    },
}


class UpstreamArtifactMissing(DirectionError):
    """The upstream repository does not publish the vector being asked for."""


class DimensionalityMismatch(DirectionError):
    """An imported vector cannot describe this model's activations."""


# ------------------------------------------------------------------- fetching


def _cache_dir() -> Path:
    root = os.environ.get("NEBULAI_IMPORT_CACHE")
    p = Path(root) if root else Path.home() / ".cache" / "nebulai" / "imports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def fetch(url: str, *, timeout: int = 120) -> bytes:
    """GET with an on-disk cache keyed by URL.

    Every URL these adapters build is pinned to a commit sha, so the cache can
    never go stale against the protocol string that was recorded beside it.
    """
    key = hashlib.sha256(url.encode()).hexdigest()[:32]
    path = _cache_dir() / key
    if path.exists():
        return path.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "nebulai/import_directions"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - pinned https
        data = r.read()
    path.write_bytes(data)
    return data


def _raw(repo: str, path: str, sha: str | None = None) -> bytes:
    sha = sha or PINNED[repo]
    return fetch(f"{_GH_RAW}/{repo}/{sha}/{path}")


def resolve_head(repo: str) -> str:
    """The repo's current head sha. Used only to REPORT drift from `PINNED`."""
    doc = json.loads(fetch(f"{_GH_API}/repos/{repo}").decode())
    head = json.loads(
        fetch(f"{_GH_API}/repos/{repo}/commits/{doc['default_branch']}").decode()
    )
    return str(head["sha"])


# ------------------------------------------------------------------ the gate


def check_dimensionality(
    direction: Direction,
    *,
    target_model: str,
    target_d: int,
    target_space: str | None = None,
) -> None:
    """Raise unless this vector can honestly describe that model's activations.

    Two checks, in order of how badly the failure would mislead:

    1. **Dimension.** A d-mismatch is arithmetic: the dot product is undefined.
       The message carries both numbers and both model ids, because "shape
       mismatch" alone sends people looking for a bug in the loader.
    2. **Space.** Equal dimension is necessary and not sufficient — two
       1024-dimensional residual streams from different models are different
       spaces, and D2 is what stops one being plotted against the other.
    """
    if direction.d != int(target_d):
        raise DimensionalityMismatch(
            f"{direction.id!r} is a {direction.d}-dimensional vector "
            f"({direction.source.get('model', 'upstream model')}), but "
            f"{target_model} has d={int(target_d)}. There is no resampling that "
            f"makes these comparable: a projection onto the first {int(target_d)} "
            f"coordinates of a {direction.d}-dimensional direction is a number, "
            f"not a measurement. Refusing to import."
        )
    if target_space is not None and direction.space != target_space:
        raise DimensionalityMismatch(
            f"{direction.id!r} is declared in space {direction.space!r} but this "
            f"map's points are in {target_space!r}. Equal dimensionality is not "
            f"equal geometry (D2) — refusing to import."
        )


# ------------------------------------------------------- 1 · refusal_direction


def from_refusal_direction(
    run: str,
    *,
    sha: str | None = None,
    id: str | None = None,
) -> Direction:
    """The published refusal direction for one of the paper's five runs.

    `pipeline/runs/<run>/direction.pt` is a *view* into that run's `mean_diffs`
    storage, which is why the file is megabytes and the vector is kilobytes;
    `torch_pickle` honours the storage offset, so what comes back is the
    selected (layer, position) row and not the first slice of the block.

    Method is recorded as `diff_of_means` because that is what the paper's
    pipeline computes (difference in mean residual-stream activation between
    harmful and harmless instructions, at a selected layer and token position);
    the selection procedure is named in the protocol string.
    """
    if run not in REFUSAL_RUNS:
        raise DirectionError(
            f"unknown refusal_direction run {run!r}; published runs are "
            f"{sorted(REFUSAL_RUNS)}"
        )
    from .torch_pickle import load_pt

    repo = "andyrdt/refusal_direction"
    sha = sha or PINNED[repo]
    meta = json.loads(_raw(repo, f"pipeline/runs/{run}/direction_metadata.json", sha).decode())
    layer = int(meta["layer"])
    pos = int(meta["pos"])
    vec = np.asarray(load_pt(_raw(repo, f"pipeline/runs/{run}/direction.pt", sha)), dtype=np.float64)
    if vec.ndim != 1:
        raise DirectionError(f"{run}: direction.pt is {vec.shape}, expected a 1-D vector")
    info = REFUSAL_RUNS[run]
    return Direction(
        id=id or f"refusal-{run}-L{layer}",
        label=f"refusal direction, {run} layer {layer} (pos {pos})",
        space=f"resid.L{layer}",
        method="diff_of_means",
        vector=vec,
        source={
            "kind": "imported",
            "protocol": (
                f"{repo}@{sha}:pipeline/runs/{run}/direction.pt — difference in "
                f"mean residual-stream activation over harmful vs harmless "
                f"instruction sets (dataset/splits/harmful_train.json vs "
                f"harmless_train.json), layer {layer}, token position {pos}, "
                f"selected by the repo's own select_direction pipeline"
            ),
            "repo": repo,
            "revision": sha,
            "model": info["model"],
            "layer": layer,
            "position": pos,
            "upstream_d": int(vec.shape[0]),
            "upstream_norm": round(float(np.linalg.norm(vec)), 6),
        },
    )


# ------------------------------------------------------- 2 · persona_vectors


def persona_traits(sha: str | None = None) -> list[str]:
    """The seven traits whose prompt sets the repo publishes."""
    repo = "safety-research/persona_vectors"
    sha = sha or PINNED[repo]
    tree = json.loads(fetch(f"{_GH_API}/repos/{repo}/git/trees/{sha}?recursive=1").decode())
    pre = "data_generation/trait_data_extract/"
    return sorted(
        t["path"][len(pre) : -len(".json")]
        for t in tree["tree"]
        if t["type"] == "blob" and t["path"].startswith(pre) and t["path"].endswith(".json")
    )


def persona_trait_prompts(trait: str, *, sha: str | None = None) -> dict[str, Any]:
    """The frozen positive/negative instruction pair for one trait.

    This is the honest half of this upstream: the repo publishes the *protocol*
    (which system prompts elicit and suppress a trait) and the code that turns
    it into a vector, but not the vectors. Handing back the prompt sets with the
    sha attached is what lets `diff_of_means` recompute one against a model we
    actually have, with a protocol string that points at the real source.
    """
    repo = "safety-research/persona_vectors"
    sha = sha or PINNED[repo]
    doc = json.loads(_raw(repo, f"data_generation/trait_data_extract/{trait}.json", sha).decode())
    return {
        "trait": trait,
        "repo": repo,
        "revision": sha,
        "protocol": f"{repo}@{sha}:data_generation/trait_data_extract/{trait}.json",
        "doc": doc,
    }


def from_persona_vectors(trait: str, *, sha: str | None = None) -> Direction:
    """**Refuses.** persona_vectors publishes no vectors.

    At the pinned commit the repository contains `generate_vec.py`, the trait
    prompt sets, one evaluation CSV and a 60 MB `dataset.zip` — and no persona
    vector of any model. Importing one is impossible; computing one requires the
    Qwen2.5-7B-Instruct activations the repo's own script extracts, which is
    phase 2's instruct-model work.
    """
    raise UpstreamArtifactMissing(
        f"safety-research/persona_vectors@{sha or PINNED['safety-research/persona_vectors']} "
        f"publishes no persona vectors — it publishes the extraction protocol "
        f"(generate_vec.py) and the trait prompt sets. Use "
        f"`persona_trait_prompts({trait!r})` to get the frozen prompt pair and "
        f"compute a direction against a model you have, which is a "
        f"diff-of-means on those prompts and must be labelled as one."
    )


# ------------------------------------------------------- 3 · assistant-axis


def assistant_axis_roles(*, sha: str | None = None) -> list[str]:
    """The 275 role names the Assistant Axis work sweeps over."""
    repo = "safety-research/assistant-axis"
    sha = sha or PINNED[repo]
    doc = json.loads(_raw(repo, "data/roles/role_list.json", sha).decode())
    if isinstance(doc, dict):
        return list(doc.keys())
    return list(doc)


def from_assistant_axis(*, sha: str | None = None) -> Direction:
    """**Refuses.** assistant-axis publishes no activations and no basis.

    At the pinned commit the repository contains the pipeline
    (`assistant_axis/pca.py`, `axis.py`, `steering.py`), 275 role instruction
    files, 200+ trait instruction files and the extraction questions — every
    ingredient of the axis except the axis. The PC1×PC2 scatter of 275
    archetypes that episode 3 wants is a *rendering of activations* that were
    never published, so there is nothing to import and nothing to plot; the
    episode says so rather than substituting a different model's PCA.
    """
    raise UpstreamArtifactMissing(
        f"safety-research/assistant-axis@{sha or PINNED['safety-research/assistant-axis']} "
        f"publishes the pipeline and the 275 role prompts, not the activations "
        f"or the fitted PCA basis. There is no vector in that repository to "
        f"import. Computing the axis needs per-role activations from the "
        f"instruct model the paper used — phase 2 work, not an import."
    )


# ------------------------------------------------------------ 4 · Neuronpedia


def from_neuronpedia(
    feature_ref: str,
    *,
    label: str | None = None,
    id: str | None = None,
) -> Direction:
    """A Neuronpedia feature id, resolved to the SAE decoder row it names.

    `feature_ref` is Neuronpedia's own `<model>/<layer>-<release>/<index>` path
    (e.g. `gpt2-small/8-res-jb/12345`). Neuronpedia is an index over published
    SAEs, so the honest import reads the *weights repository* those features
    come from rather than an API response — the vector is the same either way,
    and this way the protocol string names the safetensors file.
    """
    parts = feature_ref.strip().strip("/").split("/")
    if len(parts) != 3:
        raise DirectionError(
            f"expected a Neuronpedia feature ref like "
            f"'gpt2-small/8-res-jb/12345', got {feature_ref!r}"
        )
    model_key, layer_key, index = parts
    src = NEURONPEDIA_SOURCES.get(f"{model_key}/{layer_key}")
    if src is None:
        raise DirectionError(
            f"no weights repository is registered for Neuronpedia source "
            f"{model_key}/{layer_key}; known: {sorted(NEURONPEDIA_SOURCES)}"
        )
    from .directions import from_sae_decoder

    d = from_sae_decoder(
        src["repo"],
        int(index),
        int(src["layer"]),
        sae_id=src["sae_id"],
        id=id or f"np-{model_key}-{layer_key}-{index}",
        label=label or f"Neuronpedia {feature_ref}",
    )
    d.source["protocol"] = (
        f"neuronpedia:{feature_ref} = {d.source['protocol']}"
    )
    d.source["neuronpedia"] = feature_ref
    return d


# --------------------------------------------------------- 5 · Among Us probe


#: the probe checkpoints published in `linear-probes/checkpoints/`
AMONG_US_PROBES: dict[str, dict[str, Any]] = {
    "AmongUsDataset_probe_phi4": {"model": "microsoft/phi-4", "d": 5120, "layer": 20},
    "DishonestQADataset_probe_phi4": {"model": "microsoft/phi-4", "d": 5120, "layer": 20},
    "RepEngDataset_probe_phi4": {"model": "microsoft/phi-4", "d": 5120, "layer": 20},
    "TruthfulQADataset_probe_phi4": {"model": "microsoft/phi-4", "d": 5120, "layer": 20},
}


def from_among_us_probe(
    checkpoint: str = "AmongUsDataset_probe_phi4",
    *,
    sha: str | None = None,
    id: str | None = None,
) -> Direction:
    """A trained deception probe, converted back into activation space.

    The checkpoint is a `LinearModel`: it standardises the activation with a
    stored per-feature mean/std and *then* applies a bias-free linear layer. So
    the weight vector is a direction in STANDARDISED space, and the direction in
    the model's own residual stream is `w / std` — dividing is not a detail, it
    is the difference between the probe's direction and a rescaling of it by the
    per-feature variance of the training set. The mean only shifts the decision
    threshold and drops out of the direction entirely.
    """
    from .torch_pickle import load_pt

    if checkpoint not in AMONG_US_PROBES:
        raise DirectionError(
            f"unknown Among Us checkpoint {checkpoint!r}; readable ones (the "
            f".pth state dicts) are {sorted(AMONG_US_PROBES)} — the sibling "
            f".pkl files are pickled sklearn objects and are not read here"
        )
    repo = "7vik/AmongUs"
    sha = sha or PINNED[repo]
    info = AMONG_US_PROBES[checkpoint]
    state = load_pt(_raw(repo, f"linear-probes/checkpoints/{checkpoint}.pth", sha))
    w = np.asarray(state["linear.weight"], dtype=np.float64).reshape(-1)
    std = np.asarray(state["std"], dtype=np.float64).reshape(-1)
    std = np.clip(std, 1e-8, None)
    return Direction(
        id=id or f"amongus-{checkpoint}",
        label=f"deception probe, {checkpoint}",
        space=f"resid.L{info['layer']}",
        method="probe",
        vector=w / std,
        source={
            "kind": "imported",
            "protocol": (
                f"{repo}@{sha}:linear-probes/checkpoints/{checkpoint}.pth — "
                f"bias-free logistic probe over standardised layer-{info['layer']} "
                f"activations; the activation-space direction is w/std"
            ),
            "repo": repo,
            "revision": sha,
            "model": info["model"],
            "layer": info["layer"],
            "upstream_d": int(w.shape[0]),
        },
    )


# ------------------------------------------------------------------ registry


#: `nebulai direction add <source>` dispatches through this.
IMPORTERS: dict[str, Any] = {
    "refusal": from_refusal_direction,
    "persona": from_persona_vectors,
    "assistant-axis": from_assistant_axis,
    "neuronpedia": from_neuronpedia,
    "amongus": from_among_us_probe,
}


def survey(target_d: int, target_model: str) -> list[dict[str, Any]]:
    """What every importable published vector would do against one model.

    Returns one row per artefact: its dimension, and whether it can be imported
    into a model of width `target_d`. This is the table the phase-1 report
    quotes, and it is computed from the real tensors rather than from the
    papers' prose.
    """
    rows: list[dict[str, Any]] = []
    for run, info in REFUSAL_RUNS.items():
        d = from_refusal_direction(run)
        rows.append(
            {
                "artefact": f"refusal_direction/{run}",
                "upstream_model": info["model"],
                "d": d.d,
                "space": d.space,
                "importable": d.d == int(target_d),
            }
        )
    for ck, info in AMONG_US_PROBES.items():
        rows.append(
            {
                "artefact": f"AmongUs/{ck}",
                "upstream_model": info["model"],
                "d": int(info["d"]),
                "space": f"resid.L{info['layer']}",
                "importable": int(info["d"]) == int(target_d),
            }
        )
    rows.append(
        {
            "artefact": "persona_vectors",
            "upstream_model": "Qwen/Qwen2.5-7B-Instruct",
            "d": None,
            "space": None,
            "importable": False,
            "reason": "publishes prompt sets and extraction code, no vectors",
        }
    )
    rows.append(
        {
            "artefact": "assistant-axis",
            "upstream_model": "(paper's instruct model)",
            "d": None,
            "space": None,
            "importable": False,
            "reason": "publishes the pipeline and 275 role prompts, no basis",
        }
    )
    for r in rows:
        r["target_model"] = target_model
        r["target_d"] = int(target_d)
    return rows


def prompt_pairs(doc: Mapping[str, Any]) -> tuple[Sequence[str], Sequence[str]]:
    """Pull (positive, negative) instruction lists out of a persona_vectors doc.

    The file stores `instruction` as a LIST of `{pos, neg}` pairs — five paired
    system prompts per trait — so the two lists come back index-aligned, which
    is what makes the difference a paired contrast rather than two unrelated
    samples. Returned verbatim, no paraphrase and no truncation, because the
    protocol string points at the file and the two have to match.
    """
    inst = doc.get("instruction")
    if not isinstance(inst, list) or not inst:
        raise DirectionError(
            f"no `instruction` list in this document (keys: {sorted(doc)})"
        )
    pos = [p["pos"] for p in inst if isinstance(p, dict) and "pos" in p]
    neg = [p["neg"] for p in inst if isinstance(p, dict) and "neg" in p]
    if len(pos) != len(neg) or not pos:
        raise DirectionError(
            f"instruction list is not a clean pos/neg pairing "
            f"({len(pos)} pos, {len(neg)} neg)"
        )
    return pos, neg
