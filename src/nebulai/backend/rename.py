"""Re-title an exported map in place, without rebuilding it.

Naming is the one pipeline stage whose quality is independent of the geometry:
by the time it runs the clusters are already fixed, so a better namer can be
applied to a finished map without moving a single coordinate. That matters
because the early maps here were titled by a 1.2B local model — or by
`centroid`, which is not naming at all, just the four most central tokens
joined by a dot — and rebuilding a 50k-point map to fix only its titles would
mean re-running UMAP to reproduce points that are already correct.

The one compromise, stamped into the map rather than hidden: exports do not
carry the source vectors, so representatives are ranked by centrality in
`u_cluster` (the 10-D UMAP space HDBSCAN actually clustered in) instead of the
original embedding space. For "the most central members of this cluster" that
is arguably the more faithful space — it is where the cluster was defined — but
it IS a different selection than the build path makes, so `reps_space` records
which space the namer saw.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..units import Units
from .name import name_clusters


def is_all_placeholder(labels: list[str]) -> bool:
    """True when no label carries meaning an LLM could name from.

    Same rule the build path enforces via `placeholder_titles`: a namer handed
    only "neuron 3 (unlabeled)" invents semantics from zero information. A
    rename must not be the back door that puts fabricated titles on the SAE and
    neuron maps the build path was careful to leave honest.
    """
    return bool(labels) and all(
        lab.strip().endswith("(unlabeled)") for lab in labels
    )


def units_from_export(doc: dict, reduced: np.ndarray) -> tuple[Units, np.ndarray]:
    """Reconstruct the namer's view of a built map from its own export.

    Point `id` is the row index into reduced.npz — asserted here rather than
    assumed, because a silent misalignment would name every cluster from some
    other cluster's members and still look perfectly plausible.
    """
    points = doc["points"]
    ids = [int(p["id"]) for p in points]
    if ids != sorted(ids) or ids[-1] != len(reduced) - 1 or len(ids) != len(reduced):
        raise ValueError(
            f"export/reduced.npz misalignment: {len(ids)} points with max id "
            f"{ids[-1] if ids else None} against {len(reduced)} reduced rows"
        )
    labels = [str(p.get("label", "")) for p in points]
    cluster_ids = np.array([int(p["cluster_id"]) for p in points], dtype=np.int64)
    return Units(ids=ids, vectors=reduced[ids], labels=labels), cluster_ids


def rename_map(map_dir: Path, namer: str = "claude-cli", **namer_kwargs) -> dict:
    """Re-title one built map. Returns a summary; raises if it should not run."""
    doc_path = map_dir / "nebulai.json"
    npz_path = map_dir / "reduced.npz"
    if not doc_path.exists():
        raise FileNotFoundError(f"no export at {doc_path}")
    if not npz_path.exists():
        raise FileNotFoundError(
            f"no {npz_path} — renaming needs the cluster-space coordinates to "
            "rank cluster representatives"
        )

    doc = json.loads(doc_path.read_text())
    with np.load(npz_path) as z:
        if "u_cluster" not in z:
            raise ValueError(f"{npz_path} has no u_cluster array")
        reduced = np.asarray(z["u_cluster"], dtype=np.float32)

    units, cluster_ids = units_from_export(doc, reduced)
    if is_all_placeholder(units.labels):
        raise ValueError(
            "every label on this map is a placeholder — there is nothing for a "
            "namer to read, and titling it anyway would be invention. Left as "
            f"'{doc['meta'].get('namer')}'."
        )

    titles, namer_used = name_clusters(units, cluster_ids, namer=namer, **namer_kwargs)

    renamed = 0
    for cluster in doc["clusters"]:
        title = titles.get(int(cluster["id"]))
        if title and title.strip():
            cluster["title"] = title.strip()
            renamed += 1

    was = doc["meta"].get("namer")
    doc["meta"]["namer"] = namer_used
    doc["meta"]["renamed_from"] = was
    doc["meta"]["reps_space"] = "u_cluster"
    doc_path.write_text(json.dumps(doc))
    return {
        "id": map_dir.name,
        "renamed": renamed,
        "n_clusters": len(doc["clusters"]),
        "namer": namer_used,
        "was": was,
    }


def sync_index(out_root: Path) -> None:
    """Carry each map's namer back into index.json.

    The viewer reads the index, not the exports, so a rename that stopped at
    nebulai.json would leave the map list advertising the namer it no longer
    uses.
    """
    index_path = out_root / "index.json"
    if not index_path.exists():
        return
    index = json.loads(index_path.read_text())
    for entry in index.get("datasets", []):
        doc_path = out_root / entry["path"]
        if doc_path.exists():
            entry["namer"] = json.loads(doc_path.read_text())["meta"].get("namer")
    index_path.write_text(json.dumps(index, indent=2))
