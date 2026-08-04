"""Re-titling a built map in place.

The risk this command carries is not a crash — it is a rename that succeeds
while silently naming clusters from the wrong members, or that puts confident
titles on a map the build path deliberately left honest. Both are tested here.
"""

import json

import numpy as np
import pytest

from nebulai.backend.rename import (
    is_all_placeholder,
    rename_map,
    sync_index,
    units_from_export,
)


def _doc(labels, cluster_ids, titles=("old a", "old b"), namer="ollama:tiny"):
    return {
        "meta": {"namer": namer, "n_clusters": len(titles)},
        "points": [
            {"id": i, "label": lab, "cluster_id": int(c), "xy": [0.0, 0.0]}
            for i, (lab, c) in enumerate(zip(labels, cluster_ids))
        ],
        "clusters": [
            {"id": i, "title": t, "size": 1} for i, t in enumerate(titles)
        ],
        "edges": [],
    }


def _write(tmp_path, doc, reduced):
    d = tmp_path / "amap"
    d.mkdir()
    (d / "nebulai.json").write_text(json.dumps(doc))
    np.savez(d / "reduced.npz", u_cluster=reduced.astype(np.float32))
    return d


# --- the honesty guard ----------------------------------------------------


def test_all_placeholder_maps_are_refused():
    assert is_all_placeholder(["neuron 0 (unlabeled)", "neuron 1 (unlabeled)"])
    assert not is_all_placeholder(["neuron 0 (unlabeled)", "cat"])
    assert not is_all_placeholder([])


def test_rename_refuses_a_placeholder_map_and_leaves_it_alone(tmp_path):
    """The build path uses placeholder_titles for these; rename must not be the
    back door that gives them invented semantics."""
    doc = _doc(
        ["feature 0 (unlabeled)", "feature 1 (unlabeled)"],
        [0, 1],
        namer="none(all-placeholder-labels)",
    )
    d = _write(tmp_path, doc, np.eye(2))
    with pytest.raises(ValueError, match="placeholder"):
        rename_map(d, namer="none")
    after = json.loads((d / "nebulai.json").read_text())
    assert after["meta"]["namer"] == "none(all-placeholder-labels)"
    assert [c["title"] for c in after["clusters"]] == ["old a", "old b"]


# --- alignment ------------------------------------------------------------


def test_misalignment_is_an_error_not_a_silent_mislabel(tmp_path):
    """Point id indexes reduced.npz. If that stops holding, every cluster gets
    named from some other cluster's members — and still looks plausible."""
    doc = _doc(["a", "b", "c"], [0, 0, 1])
    with pytest.raises(ValueError, match="misalignment"):
        units_from_export(doc, np.eye(2))  # 3 points, 2 rows


def test_units_are_built_from_cluster_space_rows(tmp_path):
    reduced = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    units, cids = units_from_export(_doc(["a", "b", "c"], [0, 0, 1]), reduced)
    assert units.labels == ["a", "b", "c"]
    assert cids.tolist() == [0, 0, 1]
    np.testing.assert_array_equal(units.vectors, reduced)


# --- the rename itself ----------------------------------------------------


def test_rename_rewrites_titles_and_records_what_it_replaced(tmp_path):
    doc = _doc(["cat", "dog", "red"], [0, 0, 1])
    d = _write(tmp_path, doc, np.array([[1, 0], [0.9, 0.1], [0, 1]]))
    r = rename_map(d, namer="none")  # centroid: deterministic, no network

    after = json.loads((d / "nebulai.json").read_text())
    assert r["renamed"] == 2
    assert [c["title"] for c in after["clusters"]] != ["old a", "old b"]
    # provenance: what named it now, what named it before, what space it read
    assert after["meta"]["namer"] == "centroid"
    assert after["meta"]["renamed_from"] == "ollama:tiny"
    assert after["meta"]["reps_space"] == "u_cluster"


def test_geometry_is_untouched(tmp_path):
    """The whole premise is that titles and points have separate lifetimes."""
    doc = _doc(["cat", "dog", "red"], [0, 0, 1])
    d = _write(tmp_path, doc, np.array([[1, 0], [0.9, 0.1], [0, 1]]))
    before = json.loads((d / "nebulai.json").read_text())
    rename_map(d, namer="none")
    after = json.loads((d / "nebulai.json").read_text())
    assert after["points"] == before["points"]
    assert [c["centroid"] for c in after["clusters"] if "centroid" in c] == [
        c["centroid"] for c in before["clusters"] if "centroid" in c
    ]


def test_a_blank_title_does_not_overwrite_a_real_one(tmp_path, monkeypatch):
    """A partial namer result must leave the clusters it skipped alone rather
    than exporting them unnamed."""
    import nebulai.backend.rename as R

    monkeypatch.setattr(
        R, "name_clusters", lambda *a, **k: ({0: "animals", 1: "   "}, "fake:m")
    )
    doc = _doc(["cat", "dog", "red"], [0, 0, 1])
    d = _write(tmp_path, doc, np.array([[1, 0], [0.9, 0.1], [0, 1]]))
    r = rename_map(d)
    after = json.loads((d / "nebulai.json").read_text())
    assert r["renamed"] == 1
    assert [c["title"] for c in after["clusters"]] == ["animals", "old b"]


def test_missing_reduced_npz_is_a_clear_skip(tmp_path):
    d = tmp_path / "amap"
    d.mkdir()
    (d / "nebulai.json").write_text(json.dumps(_doc(["a"], [0], titles=("t",))))
    with pytest.raises(FileNotFoundError, match="reduced.npz"):
        rename_map(d, namer="none")


# --- the index ------------------------------------------------------------


def test_sync_index_carries_the_new_namer_to_the_viewer(tmp_path):
    """The viewer reads index.json, not the exports — a rename that stopped at
    nebulai.json would leave the map list advertising a namer it no longer uses."""
    d = _write(tmp_path, _doc(["cat", "dog", "red"], [0, 0, 1]), np.eye(3))
    (tmp_path / "index.json").write_text(
        json.dumps({"datasets": [{"id": "amap", "path": "amap/nebulai.json", "namer": "ollama:tiny"}]})
    )
    rename_map(d, namer="none")
    sync_index(tmp_path)
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["datasets"][0]["namer"] == "centroid"
