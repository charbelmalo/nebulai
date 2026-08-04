"""Semantic-cloud front-end tests — dedup, BFS shape, the sensitivity gate, and
the honesty stamps.

Fully offline: the generator and the embedder are both fakes. The fake embedder
places every term at a controlled cosine from the seed, which is what makes the
`--sensitivity` behaviour testable at all — with a real embedder the threshold's
effect depends on the embedder's opinions, which is precisely the thing this
front-end must never hide.
"""

import numpy as np
import pytest

from nebulai.backend import embed as embed_mod
from nebulai.backend import name as name_mod
from nebulai.frontends import probe as probe_mod
from nebulai.frontends.probe import _clean, _norm, load_probe_units


# --- pure helpers ---------------------------------------------------------


def test_norm_folds_case_and_edge_punctuation_only():
    assert _norm("  Grief.  ") == "grief"
    assert _norm("Stages of Grief") == "stages of grief"
    # distinct wordings stay distinct — merging them is a judgement the map
    # is supposed to display, not silently make
    assert _norm("car") != _norm("automobile")


def test_clean_dedups_against_seen_and_respects_limit():
    seen = {"grief"}
    out = _clean(["Grief", "mourning", "Mourning", "loss", "denial"], seen, limit=2)
    assert out == ["mourning", "loss"]  # 'Grief' already seen, 'Mourning' a dupe
    assert "mourning" in seen and "loss" in seen


def test_clean_drops_empty_and_overlong_terms():
    out = _clean(["", "   ", "x" * 200, "valid"], set(), limit=10)
    assert out == ["valid"]


# --- the cloud ------------------------------------------------------------


def _fake_embedder(sim_for):
    """Return an embed_texts stand-in placing term i at cosine `sim_for(term)`
    from the seed (term 0), in 2-D: seed=[1,0], term=[s, sqrt(1-s^2)]."""

    def embed_texts(texts, **_kw):
        vecs = []
        for i, t in enumerate(texts):
            s = 1.0 if i == 0 else float(sim_for(t))
            s = max(-1.0, min(1.0, s))
            vecs.append([s, float(np.sqrt(max(0.0, 1.0 - s * s)))])
        return np.asarray(vecs, dtype=np.float32)

    return embed_texts


@pytest.fixture
def fake_stack(monkeypatch):
    """Install a deterministic generator + embedder. Terms are named
    'near-<n>' / 'far-<n>' so the sensitivity gate has something to bite on."""

    def install(sim_for, breadth_terms=None):
        calls = []

        def expand(term, n):
            calls.append((term, n))
            if breadth_terms is not None:
                return breadth_terms(term, n)
            return [f"near-{term}-{i}" for i in range(n)]

        monkeypatch.setattr(
            probe_mod, "_make_expander", lambda *a, **k: (expand, "fake:gen-1")
        )
        monkeypatch.setattr(embed_mod, "embed_texts", _fake_embedder(sim_for))
        return calls

    return install


def test_bfs_expands_breadth_first_to_the_requested_depth(fake_stack):
    calls = fake_stack(sim_for=lambda t: 0.9)
    units = load_probe_units("grief", depth=2, breadth=3, sensitivity=0.0)

    # depth 1: 1 call on the seed -> 3 terms. depth 2: 3 calls -> 9 terms.
    assert len(calls) == 4
    assert calls[0] == ("grief", 3)
    assert len(units) == 1 + 3 + 9
    assert units.meta["max_depth_reached"] == 2
    assert units.labels[0] == "grief"


def test_sensitivity_gates_on_similarity_to_the_seed(fake_stack):
    """Terms below the floor are dropped and COUNTED, not silently discarded."""
    fake_stack(
        sim_for=lambda t: 0.8 if "keep" in t else 0.1,
        breadth_terms=lambda term, n: [f"keep-{term}-{i}" for i in range(n // 2)]
        + [f"drop-{term}-{i}" for i in range(n - n // 2)],
    )
    units = load_probe_units("grief", depth=1, breadth=4, sensitivity=0.5)

    assert all("drop" not in lab for lab in units.labels)
    assert units.meta["n_proposed"] == 5  # seed + 4
    assert units.meta["kept"] == len(units)
    assert units.meta["n_dropped"] == units.meta["n_proposed"] - units.meta["kept"]
    assert units.meta["n_dropped"] == 2


def test_seed_is_never_gated_out(fake_stack):
    """The seed anchors the map; a high threshold must not delete it."""
    fake_stack(sim_for=lambda t: 0.99)
    units = load_probe_units("grief", depth=1, breadth=6, sensitivity=0.95)
    assert units.labels[0] == "grief"
    assert units.meta["seed_similarity_min"] >= 0.95


def test_meta_declares_the_geometry_is_not_model_internal(fake_stack):
    fake_stack(sim_for=lambda t: 0.9)
    units = load_probe_units(
        "grief", depth=1, breadth=6, sensitivity=0.0, embed_model="fake-embedder"
    )
    m = units.meta
    assert "NOT model-internal" in m["geometry"]
    assert m["generator"] == "fake:gen-1"
    assert m["embed_model"] == "fake-embedder"
    assert m["probe_seed"] == "grief"
    assert m["sensitivity"] == 0.0
    assert m["unit"].startswith("probe_concept")


def test_units_contract_holds(fake_stack):
    fake_stack(sim_for=lambda t: 0.9)
    units = load_probe_units("grief", depth=2, breadth=3, sensitivity=0.0)
    assert len(units.ids) == len(units.labels) == units.vectors.shape[0]
    assert units.vectors.dtype == np.float32


def test_too_aggressive_sensitivity_fails_loudly(fake_stack):
    fake_stack(sim_for=lambda t: 0.1)
    with pytest.raises(RuntimeError, match="kept only"):
        load_probe_units("grief", depth=1, breadth=6, sensitivity=0.9)


def test_a_generator_that_returns_nothing_fails_loudly(fake_stack):
    fake_stack(sim_for=lambda t: 0.9, breadth_terms=lambda term, n: [])
    with pytest.raises(RuntimeError, match="only 1 terms"):
        load_probe_units("grief", depth=2, breadth=6, sensitivity=0.0)


def test_probe_maps_get_a_readable_label_in_the_metrics_table():
    from nebulai.backend.compare import _source_label

    label = _source_label(
        {"model": "grief", "unit": "probe_concept(mxbai-embed-large)"}
    )
    assert label == "grief · probe concepts"


# --- generator availability check ------------------------------------------
# _make_expander sends one throwaway expansion to prove a backend works. A
# reasoning model that rambles past the token ceiling on that call has proved
# exactly the opposite of "unavailable" — it generated too much, not nothing.


def test_a_rambling_generator_is_kept_not_retired(monkeypatch, capsys):
    from nebulai.backend.name import ChatTruncated

    calls: list[str] = []

    def fake_expand(term, n, host, model, key):
        calls.append(term)
        if len(calls) == 1:
            raise ChatTruncated("m truncated its reply at 16384 tokens")
        return ["a", "b"]

    # _make_expander imports the picker inside the function, so patch the source
    monkeypatch.setattr(name_mod, "_openai_pick_model", lambda *a, **k: "m")
    monkeypatch.setattr(probe_mod, "_expand_openai", fake_expand)
    expand, label = probe_mod._make_expander(
        "openai", "http://o", "", "orm", "am", None, llm_host="http://h"
    )
    assert label == "openai:m"
    assert "rambles" in capsys.readouterr().out
    assert expand("grief", 2) == ["a", "b"]  # the real calls still work


def test_a_generator_that_cannot_generate_is_still_retired(monkeypatch):
    # _make_expander imports the picker inside the function, so patch the source
    monkeypatch.setattr(name_mod, "_openai_pick_model", lambda *a, **k: "m")
    monkeypatch.setattr(
        probe_mod,
        "_expand_openai",
        lambda *a, **k: (_ for _ in ()).throw(ConnectionError("refused")),
    )
    with pytest.raises(RuntimeError, match="no expansion backend available"):
        probe_mod._make_expander(
            "openai", "http://o", "", "orm", "am", None, llm_host="http://h"
        )


def test_the_availability_probe_uses_a_concrete_term(monkeypatch):
    # "test" gives a reasoning model nothing to expand and it spirals
    seen: list[str] = []
    # _make_expander imports the picker inside the function, so patch the source
    monkeypatch.setattr(name_mod, "_openai_pick_model", lambda *a, **k: "m")
    monkeypatch.setattr(
        probe_mod,
        "_expand_openai",
        lambda term, *a, **k: (seen.append(term), ["a", "b"])[1],
    )
    probe_mod._make_expander(
        "openai", "http://o", "", "orm", "am", None, llm_host="http://h"
    )
    assert seen == [probe_mod._PROBE_TERM] and probe_mod._PROBE_TERM != "test"
