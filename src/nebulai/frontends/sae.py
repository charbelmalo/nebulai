"""Plan A front-end: one point per sparse-autoencoder feature, geometry =
the SAE decoder direction (a row of W_dec) that the feature writes into the
model's residual stream.

Layer choice — blocks.8.hook_resid_pre (layer 8 of 12): the mid-to-late
residual stream, where features are abstract/semantic rather than
token-identity (early layers) or output-logit-oriented (final layers). It is
the canonical layer of the SAE Lens tutorials, among the best-covered res-jb
SAEs on Neuronpedia, and hook_resid_pre is the exact stream the SAE was
trained to reconstruct — so the decoder rows are the directions the map is
built from.

Loader — no sae-lens, no torch. The canonical sae-lens release
`gpt2-small-res-jb` / `blocks.8.hook_resid_pre` loads byte-identical weights
from the HF repo jbloom/GPT2-Small-SAEs-Reformatted; we read W_dec directly
with huggingface_hub + safetensors.numpy.load_file. This is exactly what
`SAE.from_pretrained("gpt2-small-res-jb", "blocks.8.hook_resid_pre")` would
hand back, and it avoids a multi-GB torch dependency tree the repo
deliberately excludes ("no torch is needed"). meta["loader"] records the
provenance so the honesty contract holds.

Geometry — W_dec has shape [d_sae, d_in]; ROWS are features, so W_dec itself
is Units.vectors (no transpose). No mean-centering by default (center=False):
tokens.py centers W_E to counter embedding-space anisotropy, but SAE decoder
rows are trained dictionary directions whose direction IS the semantics, and
the back-end reduces with cosine — subtracting a mean would rotate every
direction away from what the model learned.

Labels — bootstrapped from Neuronpedia's public S3 auto-interp export; never
fabricated. A feature with no exported description gets the exact placeholder
"feature {i} (unlabeled)" so the map never presents a guess as a label.
"""

import gzip
import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..units import Units

# the sae-lens release name these weights correspond to (used only for the
# honest unit string — the weights are read directly from the HF repo below)
RELEASE_TAG = "gpt2-small-res-jb"

# Neuronpedia S3 auto-interp export: 48 gzipped jsonl batches (0..47) cover
# the full 24576-feature dictionary. Batches are NOT index-contiguous, so all
# must be fetched to cover any feature subset.
_NP_MODEL = "gpt2-small"
_NP_LAYER = "8-res-jb"
_NP_N_BATCHES = 48
_NP_URL = (
    "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/v1/"
    "{model}/{layer}/explanations/batch-{n}.jsonl.gz"
)


def release_tag_for(sae_release: str) -> str:
    """The sae-lens release tag a weights repo corresponds to.

    RELEASE_TAG is only true of the canonical repo; any other repo names
    itself so the unit string never claims a provenance it doesn't have."""
    if sae_release == "jbloom/GPT2-Small-SAEs-Reformatted":
        return RELEASE_TAG
    return sae_release


def sae_unit_string(release_tag: str, sae_id: str) -> str:
    """The honest `meta.unit` value naming the true geometry origin.

    Pure so tests can pin the exact contract string without any weights."""
    return f"sae_decoder({release_tag}, {sae_id})"


def sae_dataset_id(model_name: str, sae_id: str) -> str:
    """Output dir name for an SAE map — filesystem-safe, no slashes."""
    return f"{model_name}__sae__{sae_id}".replace("/", "__")


def subset_indices(d_sae: int, max_features: int | None) -> list[int]:
    """The deterministic feature subset: the first `max_features` indices,
    clamped to d_sae. ids ARE the original dictionary indices. `None` or a
    non-positive `max_features` (e.g. --max-features 0) takes the FULL
    dictionary."""
    n = d_sae if max_features is None or max_features <= 0 else min(max_features, d_sae)
    return list(range(n))


def layer_from_sae_id(sae_id: str) -> int:
    """The layer index a hookpoint id refers to: the first integer in the id.
    `layers.21.mlp` -> 21, `blocks.8.hook_resid_pre` -> 8."""
    import re

    m = re.search(r"\d+", sae_id)
    if m is None:
        raise ValueError(f"no layer index found in sae_id {sae_id!r}")
    return int(m.group())


def parse_explanations(lines: Iterable[str]) -> dict[int, str]:
    """Parse Neuronpedia explanation jsonl lines -> {feature_index: label}.

    Pure (no network). One JSON object per line; `index` is a STRING that we
    coerce to int, `description` is the auto-interp label. Duplicate indices
    are FIRST-WINS (a feature can have several explanation records); records
    with a missing/empty description are skipped; blank lines are skipped; the
    huge `embedding` field is never stored.
    """
    out: dict[int, str] = {}
    for line in lines:
        if not line or not line.strip():
            continue
        rec = json.loads(line)
        if "index" not in rec:
            continue
        idx = int(rec["index"])
        if idx in out:
            continue  # first-wins
        desc = rec.get("description")
        if not isinstance(desc, str):
            continue
        desc = desc.strip()
        if not desc:
            continue
        out[idx] = desc
    return out


def labels_for(ids: list[int], desc: dict[int, str]) -> list[str]:
    """Merge parsed descriptions onto a subset's ids. Unlabeled features get
    the exact placeholder "feature {i} (unlabeled)" — never a fabrication."""
    return [desc.get(i) or f"feature {i} (unlabeled)" for i in ids]


def _fetch_explanations(cache_dir: Path) -> dict[int, str]:
    """Download the 48 Neuronpedia batches into cache_dir (skip already-present
    files = resume), then parse them into {index: description}.

    Stop conditions: HTTP 404 for a batch beyond the known set stops the fetch;
    any other network error mid-fetch raises with an honest re-run message
    (already-downloaded files persist on disk for the retry)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for n in range(_NP_N_BATCHES):
        dest = cache_dir / f"batch-{n}.jsonl.gz"
        if not dest.exists():
            url = _NP_URL.format(model=_NP_MODEL, layer=_NP_LAYER, n=n)
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    dest.write_bytes(r.read())
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break  # past the known set of batches
                raise RuntimeError(
                    f"Neuronpedia fetch failed on batch-{n} (HTTP {e.code}); "
                    f"already-downloaded batches persist in {cache_dir} — re-run "
                    f"`nebulai sae` to resume."
                ) from e
            except urllib.error.URLError as e:
                raise RuntimeError(
                    f"Neuronpedia fetch failed on batch-{n} ({e.reason}); "
                    f"already-downloaded batches persist in {cache_dir} — re-run "
                    f"`nebulai sae` to resume."
                ) from e
        paths.append(dest)

    desc: dict[int, str] = {}
    for p in paths:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for idx, label in parse_explanations(f).items():
                desc.setdefault(idx, label)  # first batch wins across files
    return desc


def load_sae_units(
    sae_release: str = "jbloom/GPT2-Small-SAEs-Reformatted",
    sae_id: str = "blocks.8.hook_resid_pre",
    max_features: int = 4096,
    center: bool = False,
    labels_source: str = "neuronpedia",
    out_root: Path = Path("out"),
) -> Units:
    """Load SAE decoder directions as Units.

    Heavy imports (huggingface_hub, safetensors, weights) are lazy so the pure
    helpers above import without them — matching tokens.py's style.

    Two on-disk formats are auto-detected from the per-layer cfg.json:
      - sae-lens / jbloom  (`d_sae`/`d_in`/`model_name`/`hook_point_layer`,
        weights in `sae_weights.safetensors`, plus a `sparsity.safetensors`);
      - EleutherAI `sparsify`  (`num_latents`/`d_in`/`k`/`normalize_decoder`,
        weights in `sae.safetensors`, model name in the PARENT config.json,
        no sparsity file — so no sparsity stats are stamped rather than faked).
    In both, W_dec has shape [d_sae, d_in] and rows are features (no transpose).
    """
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    from ..weights import load_safetensor_f32

    cfg = json.loads(Path(hf_hub_download(sae_release, f"{sae_id}/cfg.json")).read_text())
    fmt = "sae_lens" if "d_sae" in cfg else "sparsify"

    extra_meta: dict = {}
    sparsity_stats: dict = {}
    if fmt == "sae_lens":
        d_sae = int(cfg["d_sae"])
        d_in = int(cfg["d_in"])
        model_name = cfg["model_name"]
        layer = int(cfg["hook_point_layer"])
        hook = "resid"  # the res-jb releases hook the residual stream
        weights_path = hf_hub_download(sae_release, f"{sae_id}/sae_weights.safetensors")
        W_dec = np.asarray(load_file(weights_path)["W_dec"], dtype=np.float32)
    else:  # EleutherAI sparsify
        d_sae = int(cfg["num_latents"])
        d_in = int(cfg["d_in"])
        parent = json.loads(Path(hf_hub_download(sae_release, "config.json")).read_text())
        model_name = parent.get("model") or sae_release
        layer = layer_from_sae_id(sae_id)
        hook = "mlp" if sae_id.endswith(".mlp") else "resid"
        weights_path = hf_hub_download(sae_release, f"{sae_id}/sae.safetensors")
        W_dec = load_safetensor_f32(weights_path, keys=["W_dec"])["W_dec"]
        extra_meta = {
            "k": int(cfg["k"]),
            "expansion": d_sae // d_in,
            "normalize_decoder": bool(cfg.get("normalize_decoder", False)),
        }

    if W_dec.shape != (d_sae, d_in):
        raise ValueError(f"W_dec shape {W_dec.shape} != expected ({d_sae}, {d_in})")

    ids = subset_indices(d_sae, max_features)
    V = W_dec[ids]
    if center:
        V = V - V.mean(axis=0, keepdims=True)

    if fmt == "sae_lens":
        # sparsity: log10 firing rate per feature; stamp subset stats (no filter)
        sparsity_path = hf_hub_download(sae_release, f"{sae_id}/sparsity.safetensors")
        log_sparsity = np.asarray(
            load_file(sparsity_path)["sparsity"], dtype=np.float32
        )[ids]
        sparsity_stats = {
            "log_sparsity_mean": round(float(log_sparsity.mean()), 3),
            "log_sparsity_min": round(float(log_sparsity.min()), 3),
            "log_sparsity_max": round(float(log_sparsity.max()), 3),
            "n_dead": int((log_sparsity <= -5).sum()),
        }

    # labels: bootstrap from Neuronpedia, or all-placeholder for offline/dev.
    # The Neuronpedia export is hardwired to gpt2-small 8-res-jb, so it is only
    # valid for that exact SAE — refuse it elsewhere rather than silently
    # stamping gpt2 labels onto another model's features.
    if labels_source == "neuronpedia":
        if not (
            sae_release == "jbloom/GPT2-Small-SAEs-Reformatted"
            and sae_id == "blocks.8.hook_resid_pre"
        ):
            raise ValueError(
                "neuronpedia labels are only wired for "
                "jbloom/GPT2-Small-SAEs-Reformatted / blocks.8.hook_resid_pre; "
                f"no hosted auto-interp export exists for {sae_release}/{sae_id} "
                "— use --labels none"
            )
        cache_dir = out_root / "neuronpedia" / f"{_NP_MODEL}_{_NP_LAYER}"
        desc = _fetch_explanations(cache_dir)
    elif labels_source == "none":
        desc = {}
    else:
        raise ValueError(
            f"unknown labels_source {labels_source!r} (use 'neuronpedia' or 'none')"
        )
    labels = labels_for(ids, desc)
    n_labeled = sum(1 for i in ids if i in desc)

    meta = {
        "model": model_name,
        "unit": sae_unit_string(release_tag_for(sae_release), sae_id),
        "projection": "decoder",
        "sae_format": fmt,
        "hook": hook,
        "layer": layer,
        "loader": f"safetensors-direct({sae_release})",
        "sae_release": sae_release,
        "sae_id": sae_id,
        "d_sae": d_sae,
        "d_in": d_in,
        "kept": len(ids),
        "curation": f"first_{len(ids)}_of_{d_sae}",
        "centered": center,
        "labels_source": labels_source,
        "n_labeled": n_labeled,
        "n_unlabeled": len(ids) - n_labeled,
        **extra_meta,
        **sparsity_stats,
    }

    return Units(
        ids=ids,
        vectors=np.ascontiguousarray(V, dtype=np.float32),
        labels=labels,
        meta=meta,
    )
