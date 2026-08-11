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
import re
from dataclasses import dataclass
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


def neuron_tensor_path(layer: int, arch: str = "gpt2") -> str:
    """The HF module path whose `.weight` rows are neurons for a given layer.

    GPT-2 (Conv1D): `h.{L}.mlp.c_proj`. Llama-family (SmolLM2, Qwen, Mistral,
    Gemma; Linear down-projection): `model.layers.{L}.mlp.down_proj`. Both are
    the MLP down-projection — the map into the residual stream — so a neuron is
    a row once oriented (see orient_neuron_rows).

    This is the *canonical* path for a flat dense checkpoint only; the loader
    resolves the real key with resolve_neuron_key against the checkpoint's own
    key list, because nested and MoE checkpoints do not use this shape."""
    if arch == "llama":
        return f"model.layers.{layer}.mlp.down_proj"
    return f"h.{layer}.mlp.c_proj"


# --- key resolution ---------------------------------------------------------
#
# Exact key construction (`model.layers.{L}.mlp.down_proj`) fails on every
# corpus model: two of them nest the text stack under `model.language_model.`,
# and two are MoE, where a layer's write matrices live inside experts. So the
# key is resolved by *shape of the path*, against the checkpoint's real key
# list, and an MoE layer refuses to answer until the caller picks an expert.

# modules whose `.weight` rows/cols are MLP write directions
_DOWN_PROJ_LEAVES = ("down_proj", "c_proj")
# ...but only inside an MLP: GPT-2's attention output is ALSO called c_proj,
# and `h.8.attn.c_proj.weight` is not a neuron write matrix
_MLP_PARENTS = ("mlp", "experts", "shared_experts", "feed_forward", "ffn")
# other towers of a multimodal wrapper — never the language model's MLP
_NON_TEXT_TOWERS = ("vision_tower", "vision_model", "audio_tower", "audio_model")
_EXPERT_INDEX_RE = re.compile(r"\.experts\.(\d+)\.")


class MoeSelectionRequired(ValueError):
    """An MoE layer has many write matrices; the caller must name which one.

    Never silently resolved: mapping expert 0 and calling it "layer L's
    neurons" would describe 1/128th of the layer under the whole layer's name.
    """


@dataclass(frozen=True)
class NeuronSource:
    """Which tensor (and, for a fused expert stack, which slice) to read."""

    key: str
    kind: str  # dense | expert | shared_expert | fused_expert
    expert: int | None = None  # index into a fused tensor's leading axis

    @property
    def tensor_path(self) -> str:
        return self.key[: -len(".weight")] if self.key.endswith(".weight") else self.key


def _module_parts(key: str) -> list[str]:
    """The key's module components, ignoring a `.weight` tail."""
    path = key[: -len(".weight")] if key.endswith(".weight") else key
    return path.split(".")


def _is_mlp_down_proj(key: str) -> bool:
    parts = _module_parts(key)
    return parts[-1] in _DOWN_PROJ_LEAVES and any(p in _MLP_PARENTS for p in parts)


def layer_down_proj_keys(keys, layer: int) -> list[str]:
    """Every MLP down-projection tensor belonging to one language-model layer.

    Matches the layer segment (`.layers.{L}.` or `.h.{L}.`) so layer 2 never
    matches layer 20, requires a down-projection leaf *inside an MLP* (which
    excludes GPT-2's identically-named attention `c_proj` and the vision
    tower's `...down_proj.linear.weight`), and drops the non-text towers of a
    multimodal wrapper outright."""
    seg = re.compile(rf"(?:^|\.)(?:layers|h)\.{layer}\.")
    out = [
        k
        for k in keys
        if seg.search(k)
        and _is_mlp_down_proj(k)
        and not any(t in k for t in _NON_TEXT_TOWERS)
    ]
    return sorted(out)


def resolve_neuron_key(keys, layer: int, expert: "int | str | None" = None) -> NeuronSource:
    """Resolve one layer's write matrix, refusing to guess on MoE layers.

    `expert` is None (dense layers only), an int (routed expert index, either a
    real `...experts.{i}.down_proj.weight` key or a slice of a fused
    `...experts.down_proj` stack), `"shared"` (a shared/always-on expert), or
    `"dense"` (the layer's plain `mlp.down_proj`, which MoE models like
    Gemma-4 keep alongside the expert stack)."""
    cands = layer_down_proj_keys(keys, layer)
    if not cands:
        raise KeyError(
            f"no MLP down-projection for layer {layer}: nothing matches "
            f"`.layers.{layer}.`/`.h.{layer}.` with a {_DOWN_PROJ_LEAVES} leaf; "
            f"sample keys {sorted(keys)[:6]}"
        )

    indexed: dict[int, str] = {}
    shared: list[str] = []
    fused: list[str] = []
    dense: list[str] = []
    for k in cands:
        m = _EXPERT_INDEX_RE.search(k)
        if m:
            indexed[int(m.group(1))] = k
        elif "shared_expert" in k:
            shared.append(k)
        elif "expert" in k:
            fused.append(k)
        else:
            dense.append(k)

    is_moe = bool(indexed or fused or shared)
    if not is_moe:
        if len(dense) > 1:
            raise ValueError(
                f"layer {layer} has {len(dense)} candidate down-projections "
                f"({dense}) — cannot tell which is the MLP write matrix"
            )
        return NeuronSource(dense[0], "dense")

    if expert is None:
        opts = []
        if indexed:
            opts.append(f"--expert 0..{max(indexed)} ({len(indexed)} routed experts)")
        if fused:
            opts.append(f"--expert <i> (fused expert stack {fused[0]})")
        if shared:
            opts.append(f"--expert shared ({shared[0]})")
        if dense:
            opts.append(f"--expert dense ({dense[0]})")
        raise MoeSelectionRequired(
            f"layer {layer} is MoE: it has {len(cands)} write matrices, not one, "
            f"so 'the layer's neurons' is not defined until you pick one. "
            f"Choose: {'; '.join(opts)}"
        )

    if expert == "dense":
        if not dense:
            raise ValueError(f"layer {layer} has no plain mlp down-projection")
        return NeuronSource(dense[0], "dense")
    if expert == "shared":
        if not shared:
            raise ValueError(f"layer {layer} has no shared expert")
        return NeuronSource(shared[0], "shared_expert")

    try:
        i = int(expert)
    except (TypeError, ValueError):
        raise ValueError(
            f"expert must be an int, 'shared' or 'dense', not {expert!r}"
        ) from None
    if indexed:
        if i not in indexed:
            raise ValueError(
                f"expert {i} out of range for layer {layer} "
                f"(0..{max(indexed)}, {len(indexed)} experts)"
            )
        return NeuronSource(indexed[i], "expert", i)
    if fused:
        # one tensor holds every expert; the slice index is validated against
        # its leading axis at read time, where the shape is known
        return NeuronSource(fused[0], "fused_expert", i)
    raise ValueError(f"layer {layer} has no routed experts to select from")


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

    Thin wrapper over backend.name.placeholder_titles (the rule generalizes
    to any all-placeholder pipeline — sae's --labels none has the same need);
    kept here so the neurons contract stays test-pinned at this import path."""
    from ..backend.name import placeholder_titles as _generic

    return _generic(cluster_ids, "neurons")


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


def text_config(cfg: dict) -> dict:
    """The text stack's config — multimodal wrappers nest it under text_config."""
    inner = cfg.get("text_config")
    return inner if isinstance(inner, dict) else cfg


def mlp_dims(cfg: dict, kind: str = "dense") -> tuple[str, int, int, int]:
    """(arch, d_model, d_mlp, n_layer) for the chosen write matrix.

    A routed/shared expert's down-projection is `moe_intermediate_size` wide,
    not `intermediate_size` — reading the dense width for an expert tensor
    would make the orientation check pass for the wrong reason."""
    cfg = text_config(cfg)
    if "n_embd" in cfg:  # GPT-2 (Conv1D)
        d_model = int(cfg["n_embd"])
        return "gpt2", d_model, int(cfg.get("n_inner") or 4 * d_model), int(cfg["n_layer"])
    if "hidden_size" in cfg:  # Llama-family (Linear down_proj)
        d_model = int(cfg["hidden_size"])
        dense_mlp = int(cfg["intermediate_size"])
        if kind in ("expert", "fused_expert", "shared_expert"):
            d_mlp = int(cfg.get("moe_intermediate_size") or dense_mlp)
        else:
            d_mlp = dense_mlp
        return "llama", d_model, d_mlp, int(cfg["num_hidden_layers"])
    raise ValueError(
        "cannot detect MLP architecture: config.json has neither GPT-2 keys "
        "(n_embd) nor Llama keys (hidden_size)"
    )


def load_neuron_units(
    model_repo: str = "openai-community/gpt2",
    layer: int = 8,
    max_neurons: int | None = None,
    center: bool = False,
    labels_source: str = "none",
    out_root: Path = Path("out"),
    revision: str = "main",
    expert: "int | str | None" = None,
    remote: bool | None = None,
    token: str | None = None,
) -> Units:
    """Load MLP write directions (down-projection rows) as Units.

    Heavy imports (huggingface_hub, weights) are lazy so the pure helpers
    above import without them — matching sae.py's style. (out_root is accepted
    for signature parity with load_sae_units even though no label cache is
    written yet.)

    Architecture is auto-detected from config.json (through `text_config` for
    multimodal wrappers): GPT-2 (Conv1D `n_embd`/`n_inner`/`n_layer`) vs
    Llama-family (`hidden_size`/`intermediate_size`/`num_hidden_layers`). The
    tensor key is resolved by path shape against the checkpoint's real key
    list, so sharded and nested checkpoints work; MoE layers require `expert`.
    Sharded/large checkpoints are read over HTTP ranges and never downloaded.
    """
    from huggingface_hub import hf_hub_download

    from ..weights import SINGLE_NAME, RemoteCheckpoint

    cfg = json.loads(
        Path(hf_hub_download(model_repo, "config.json", revision=revision)).read_text()
    )
    _, _, _, n_layer = mlp_dims(cfg)
    if not 0 <= layer < n_layer:
        raise ValueError(
            f"layer {layer} out of range for {model_repo} (n_layer {n_layer})"
        )

    ck: RemoteCheckpoint | None = None
    if remote is not False:
        ck = RemoteCheckpoint.open(model_repo, revision=revision, token=token)
        if remote is None:
            size = None if ck.is_sharded else ck.file_size(SINGLE_NAME)
            from .tokens import LARGE_SINGLE_FILE_BYTES

            remote = ck.is_sharded or (size or 0) > LARGE_SINGLE_FILE_BYTES

    if remote and ck is not None:
        src = resolve_neuron_key(ck.keys(), layer, expert)
        arch, d_model, d_mlp, _ = mlp_dims(cfg, src.kind)
        _, shape = ck.info(src.key)
        if src.kind == "fused_expert":
            n_experts = int(shape[0])
            if not 0 <= (src.expert or 0) < n_experts:
                raise ValueError(
                    f"expert {src.expert} out of range for {src.key} "
                    f"({n_experts} experts)"
                )
            # one expert is one contiguous slice of the fused stack, so this
            # is a single range read, not a read of all 128 experts
            W = ck.read_rows(src.key, [int(src.expert or 0)]).reshape(shape[1:])
        else:
            # a Linear down_proj stores neurons as *columns*, so there is no
            # row subset to range-read: the layer's matrix comes whole
            W = ck.read(src.key)
        source = "remote-range"
        resolved = ck.revision
        fetched = ck.bytes_fetched
    else:
        from ..weights import load_safetensor_f32, safetensor_keys

        weights_path = hf_hub_download(model_repo, SINGLE_NAME, revision=revision)
        src = resolve_neuron_key(safetensor_keys(weights_path), layer, expert)
        arch, d_model, d_mlp, _ = mlp_dims(cfg, src.kind)
        W = load_safetensor_f32(weights_path, keys=[src.key])[src.key]
        if src.kind == "fused_expert":
            W = W[int(src.expert or 0)]
        source = "local"
        resolved = ck.revision if ck is not None else revision
        fetched = ck.bytes_fetched if ck is not None else 0

    tensor_path = src.tensor_path
    if W.ndim != 2:
        raise ValueError(f"{src.key} is {W.ndim}-d after selection — expected a matrix")
    if W.shape != (d_mlp, d_model):
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
            f"raw-neuron auto-interp export exists for these micro-models: "
            f"Neuronpedia's mlp sources are SAEs, not raw neurons)"
        )
    labels = labels_for(ids, desc)
    n_labeled = sum(1 for i in ids if i in desc)

    model_tag = model_tag_for(model_repo)
    meta = {
        "model": model_tag,
        "unit": neuron_unit_string(model_tag, tensor_path),
        "projection": "w_out",
        "arch": arch,
        "layer": layer,
        "loader": f"safetensors-{'range' if source == 'remote-range' else 'direct'}({model_repo})",
        "model_repo": model_repo,
        "tensor": src.key,
        "tensor_path": tensor_path,
        "tensor_kind": src.kind,
        "expert": src.expert if src.kind in ("expert", "fused_expert") else None,
        "revision": resolved,
        "source": source,
        "bytes_fetched": int(fetched),
        "d_mlp": d_mlp,
        "d_model": d_model,
        "kept": len(ids),
        "curation": f"first_{len(ids)}_of_{d_mlp}",
        "centered": center,
        "labels_source": labels_source,
        "n_labeled": n_labeled,
        "n_unlabeled": len(ids) - n_labeled,
        "labels_note": (
            "no public raw-neuron auto-interp export for this model "
            "(Neuronpedia mlp sources are SAEs, not raw neurons)"
        ),
    }

    return Units(
        ids=ids,
        vectors=np.ascontiguousarray(V, dtype=np.float32),
        labels=labels,
        meta=meta,
    )
