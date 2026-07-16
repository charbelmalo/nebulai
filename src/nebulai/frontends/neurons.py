"""Plan B front-end: one point per MLP hidden neuron, geometry = the neuron's
row of the MLP down-projection (HF `c_proj`, TransformerLens `W_out`) — the
direction the neuron writes into the residual stream. This is the raw-neuron
contrast to Plan A's SAE decoder directions; raw neurons are polysemantic, so
a *worse*-structured map than Plan A is the expected, honest result — the
comparison is the artifact.

Layer choice — layer 8 of 12, `h.8.mlp.c_proj`: block-index parity with Plan
A's blocks.8.hook_resid_pre (mid-to-late depth, abstract features); both
geometries are directions in the same 768-d residual basis, so the comparison
is at matched depth on the identical back-end.

Loader — no torch, no transformer-lens. huggingface_hub + safetensors
`safe_open` reads ONLY the c_proj tensor from openai-community/gpt2's
model.safetensors (framework="numpy"), not the whole checkpoint. HF GPT-2 uses
Conv1D: weight is (in_features, out_features), so c_proj.weight is (d_mlp,
d_model) = (3072, 768) and rows are neurons; orientation is still verified at
runtime against config.json (d_mlp != d_model, so shape disambiguates) and a
transposed serialization is corrected. meta["loader"] records provenance. No
mean-centering by default (center=False): W_out rows are learned write
directions whose direction IS the semantics, and the back-end reduces with
cosine — subtracting a mean would rotate every direction away from what the
model learned (same reasoning as sae.py).

Labels — there is no public raw-neuron auto-interp export for gpt2-small:
Neuronpedia's gpt2-small `*-mlp-oai` sources are OpenAI *SAEs* (their sourceset
metadata says "Sparse autoencoder for GPT2 small"; explanation indices exceed
25000 > d_mlp 3072), and OpenAI's neuron-explainer dataset covers GPT-2 XL only
(layers 0-47). So labels_source defaults to "none" and every neuron gets the
exact placeholder "neuron {i} (unlabeled)" — never a fabrication.
"""

import json
from pathlib import Path

import numpy as np

from ..units import Units

# the model tag the canonical weights repo corresponds to (used only for the
# honest unit string — the weights are read directly from the HF repo below)
MODEL_TAG = "gpt2"


def model_tag_for(model_repo: str) -> str:
    """The model tag a weights repo corresponds to.

    MODEL_TAG is only true of the canonical repo; a bare "gpt2" already names
    itself; any other repo names itself so the unit string never claims a
    provenance it doesn't have. Mirrors sae.release_tag_for."""
    if model_repo == "openai-community/gpt2":
        return MODEL_TAG
    return model_repo


def neuron_tensor_path(layer: int) -> str:
    """The HF module path whose `.weight` rows are neurons for a given layer."""
    return f"h.{layer}.mlp.c_proj"


def neuron_unit_string(model_tag: str, tensor_path: str) -> str:
    """The honest `meta.unit` value naming the true geometry origin.

    Pure so tests can pin the exact contract string without any weights."""
    return f"mlp_neuron({model_tag}, {tensor_path})"


def neuron_dataset_id(model_tag: str, tensor_path: str) -> str:
    """Output dir name for a neuron map — filesystem-safe, no slashes."""
    return f"{model_tag}__neurons__{tensor_path}".replace("/", "__")


def subset_indices(d_mlp: int, max_neurons: int | None) -> list[int]:
    """The deterministic neuron subset: the first `max_neurons` indices,
    clamped to d_mlp. ids ARE the original neuron indices."""
    n = d_mlp if max_neurons is None else min(max_neurons, d_mlp)
    return list(range(n))


def labels_for(ids: list[int], desc: dict[int, str]) -> list[str]:
    """Merge parsed descriptions onto a subset's ids. Unlabeled neurons get
    the exact placeholder "neuron {i} (unlabeled)" — never a fabrication."""
    return [desc.get(i) or f"neuron {i} (unlabeled)" for i in ids]


def placeholder_titles(cluster_ids: "np.ndarray") -> tuple[dict[int, str], str]:
    """Honest cluster titles when EVERY member label is a placeholder.

    An LLM namer given only "neuron {i} (unlabeled)" strings would invent
    semantics from zero information (observed: it produced "token clusters"),
    so the map must not pretend its clusters mean anything. Returns
    ({cluster_id: "unlabeled neurons (cluster N)"}, namer_used) with the
    namer stamped "none(all-placeholder-labels)" so meta records why."""
    titles = {
        int(cid): f"unlabeled neurons (cluster {int(cid)})"
        for cid in sorted({int(c) for c in cluster_ids if c >= 0})
    }
    return titles, "none(all-placeholder-labels)"


def orient_neuron_rows(W: np.ndarray, d_mlp: int, d_model: int) -> np.ndarray:
    """Runtime orientation verification so rows are always neurons.

    If W is (d_mlp, d_model) it is the Conv1D layout and rows are already
    neurons — returned as-is. If W is (d_model, d_mlp) it is a Linear-style
    serialization — transposed so rows become neurons. Any other shape raises,
    naming the actual and both expected shapes. Requires d_mlp != d_model so
    the two orientations are distinguishable — raises if they are equal."""
    if d_mlp == d_model:
        raise ValueError(
            f"cannot disambiguate orientation with d_mlp == d_model == {d_mlp}"
        )
    if W.shape == (d_mlp, d_model):
        return W
    if W.shape == (d_model, d_mlp):
        return W.T
    raise ValueError(
        f"c_proj weight shape {W.shape} is neither ({d_mlp}, {d_model}) nor "
        f"({d_model}, {d_mlp}) — not a GPT-2-style MLP down-projection?"
    )


def load_neuron_units(
    model_repo: str = "openai-community/gpt2",
    layer: int = 8,
    max_neurons: int | None = None,
    center: bool = False,
    labels_source: str = "none",
    out_root: Path = Path("out"),
) -> Units:
    """Load MLP write directions (c_proj rows) as Units.

    Heavy imports (huggingface_hub, safetensors) are lazy so the pure helpers
    above import without them — matching sae.py's style. (out_root is accepted
    for signature parity with load_sae_units even though no label cache is
    written yet.)
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    cfg = json.loads(Path(hf_hub_download(model_repo, "config.json")).read_text())
    d_model = int(cfg["n_embd"])
    d_mlp = int(cfg.get("n_inner") or 4 * d_model)
    n_layer = int(cfg["n_layer"])

    if not 0 <= layer < n_layer:
        raise ValueError(
            f"layer {layer} out of range for {model_repo} (n_layer {n_layer})"
        )

    weights_path = hf_hub_download(model_repo, "model.safetensors")
    key = f"{neuron_tensor_path(layer)}.weight"
    with safe_open(weights_path, framework="numpy") as f:
        if key not in f.keys():
            raise ValueError(
                f"{key} not found in {model_repo} model.safetensors — "
                f"not a GPT-2-style checkpoint?"
            )
        W = np.asarray(f.get_tensor(key), dtype=np.float32)
    W = orient_neuron_rows(W, d_mlp, d_model)

    ids = subset_indices(d_mlp, max_neurons)
    V = W[ids]
    if center:
        V = V - V.mean(axis=0, keepdims=True)

    if labels_source == "none":
        desc: dict[int, str] = {}
    else:
        raise ValueError(
            f"unknown labels_source {labels_source!r} (only 'none' — no public "
            f"raw-neuron auto-interp export exists for gpt2-small: Neuronpedia's "
            f"gpt2-small mlp sources are OpenAI SAEs, not raw neurons)"
        )
    labels = labels_for(ids, desc)
    n_labeled = sum(1 for i in ids if i in desc)

    model_tag = model_tag_for(model_repo)
    tensor_path = neuron_tensor_path(layer)
    meta = {
        "model": model_tag,
        "unit": neuron_unit_string(model_tag, tensor_path),
        "projection": "w_out",
        "layer": layer,
        "loader": f"safetensors-direct({model_repo})",
        "model_repo": model_repo,
        "tensor": f"{tensor_path}.weight",
        "d_mlp": d_mlp,
        "d_model": d_model,
        "kept": len(ids),
        "curation": f"first_{len(ids)}_of_{d_mlp}",
        "centered": center,
        "labels_source": labels_source,
        "n_labeled": n_labeled,
        "n_unlabeled": len(ids) - n_labeled,
        "labels_note": (
            "no public raw-neuron auto-interp export for gpt2-small "
            "(Neuronpedia mlp-oai sources are OpenAI SAEs; OpenAI "
            "neuron-explainer covers GPT-2 XL only)"
        ),
    }

    return Units(
        ids=ids,
        vectors=np.ascontiguousarray(V, dtype=np.float32),
        labels=labels,
        meta=meta,
    )
