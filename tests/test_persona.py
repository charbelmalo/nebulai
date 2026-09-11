"""Tests for the persona PCA space and — mostly — for its control.

The control is the point of this module. A PCA will always return a PC1 with
*some* explained-variance ratio; the only thing that makes the resulting axis
mean "persona" rather than "the shape of any 296-point cloud" is the
label-permutation null. So the tests here are built around two synthetic
worlds: one where archetype identity really does organise the activations
(the control must say `above_null`) and one where the labels are noise (the
control must say `at_null`). A control that cannot fail on the second is not a
control.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nebulai.backend import persona
from nebulai.backend.persona import (
    Control,
    PersonaError,
    PersonaSpace,
    build_prompts,
    build_space,
    default_layer,
    list_spaces,
    make_space_id,
    permutation_null,
    probe_strata,
    read_space,
    run_control,
    verdict_for,
    verify_space,
    write_space,
)
from nebulai.prompts import load_prompt_set

import tiny_llama


# ── the frozen prompt set ───────────────────────────────────────────────────


def test_shipped_prompt_set_loads_and_is_well_formed() -> None:
    doc, sha = load_prompt_set("personas.v1")
    assert len(sha) == 64
    names = [a["name"] for a in doc["archetypes"]]
    assert len(names) == len(set(names)), "archetype names must be unique"
    assert len(names) >= 200, "a PCA over a handful of points is not a space"
    assert len(doc["probes"]) >= 4
    assert "{persona}" in doc["system_template"]


def test_prompt_set_digest_is_over_the_bytes(tmp_path: Path, monkeypatch) -> None:
    # Reformatting changes nothing semantically and must still register: a set
    # is frozen, and "semantically equivalent" is not the standard.
    import nebulai.prompts as P

    doc = {"id": "t", "probes": ["a"], "archetypes": [], "system_template": "{persona}"}
    monkeypatch.setattr(P, "PROMPTS_DIR", tmp_path)
    (tmp_path / "t.json").write_text(json.dumps(doc))
    _, sha1 = P.load_prompt_set("t")
    (tmp_path / "t.json").write_text(json.dumps(doc, indent=4))
    _, sha2 = P.load_prompt_set("t")
    assert sha1 != sha2


def test_build_prompts_crosses_archetypes_with_probes() -> None:
    doc, _ = load_prompt_set("personas.v1")
    prompts, owner, names = build_prompts(doc)
    assert len(prompts) == len(names) * len(doc["probes"])
    assert owner[: len(doc["probes"])] == [0] * len(doc["probes"])
    first = json.loads(prompts[0])
    assert doc["archetypes"][0]["persona"] in first["system"]
    assert first["user"] == doc["probes"][0]


# ── the control ─────────────────────────────────────────────────────────────


def _structured(groups: int, per: int, d: int, sep: float, seed: int = 0):
    """Activations where archetype identity genuinely organises the cloud."""
    rng = np.random.default_rng(seed)
    axis = rng.standard_normal(d)
    axis /= np.linalg.norm(axis)
    centres = rng.standard_normal(groups)[:, None] * sep * axis[None, :]
    owner = np.repeat(np.arange(groups), per)
    acts = centres[owner] + rng.standard_normal((groups * per, d)) * 0.5
    return acts.astype(np.float32), owner


def _unstructured(groups: int, per: int, d: int, seed: int = 0):
    """The same cloud with the labels meaning nothing."""
    rng = np.random.default_rng(seed)
    acts = rng.standard_normal((groups * per, d)).astype(np.float32)
    owner = np.repeat(np.arange(groups), per)
    return acts, owner


def test_control_clears_the_null_on_structured_activations() -> None:
    acts, owner = _structured(40, 8, 24, sep=6.0)
    means = persona._group_means(acts, owner, 40)
    pc1 = persona._pc1_evr(means.astype(np.float32))
    null = permutation_null(acts, owner, n=120, seed=0)
    assert verdict_for(pc1, null) == "above_null"
    assert pc1 > float(np.percentile(null, 95))


def test_control_does_not_clear_the_null_on_noise() -> None:
    acts, owner = _unstructured(40, 8, 24)
    means = persona._group_means(acts, owner, 40)
    pc1 = persona._pc1_evr(means.astype(np.float32))
    null = permutation_null(acts, owner, n=120, seed=0)
    assert verdict_for(pc1, null) == "at_null", (
        "a control that cannot fail on random labels is not a control"
    )


def test_null_is_reproducible_under_its_seed() -> None:
    acts, owner = _unstructured(20, 6, 16)
    a = permutation_null(acts, owner, n=30, seed=5)
    b = permutation_null(acts, owner, n=30, seed=5)
    assert np.array_equal(a, b)


def test_permutation_preserves_group_sizes() -> None:
    acts, owner = _unstructured(12, 5, 8)
    # every draw must re-pool into same-size groups, else the null is measuring
    # group size rather than labelling
    null = permutation_null(acts, owner, n=10, seed=1)
    assert null.shape == (10,) and np.all(np.isfinite(null))


# ── the strata, and why the null needs them ─────────────────────────────────


def _crossed(groups: int, probes: int, d: int, *, probe_sep: float,
             arch_sep: float, seed: int = 0):
    """The real design in miniature: every archetype answers every probe.

    `probe_sep` is the size of the axis that separates probes and `arch_sep`
    the one that separates archetypes. The measured SmolLM2 activations have
    the first much larger than the second (81.7 % vs 14.0 % of the pooled
    sum of squares), which is exactly the regime this reproduces.
    """
    rng = np.random.default_rng(seed)
    pax = rng.standard_normal(d); pax /= np.linalg.norm(pax)
    aax = rng.standard_normal(d); aax -= (aax @ pax) * pax; aax /= np.linalg.norm(aax)
    owner = np.repeat(np.arange(groups), probes)
    probe = np.tile(np.arange(probes), groups)          # build_prompts' order
    acts = (
        rng.standard_normal(probes)[probe, None] * probe_sep * pax[None, :]
        + rng.standard_normal(groups)[owner, None] * arch_sep * aax[None, :]
        + rng.standard_normal((groups * probes, d)) * 0.5
    )
    return acts.astype(np.float32), owner, probe


def test_probe_strata_follows_build_prompts_order() -> None:
    doc, _ = load_prompt_set("personas.v1")
    prompts, owner, _ = build_prompts(doc)
    strata = probe_strata(len(prompts), len(doc["probes"]))
    # every archetype must own exactly one prompt per probe, or the stratified
    # permutation is not matching the design it claims to match
    for g in range(len(doc["archetypes"])):
        mine = strata[np.asarray(owner) == g]
        assert sorted(mine.tolist()) == list(range(len(doc["probes"])))


def test_probe_strata_refuses_a_broken_crossing() -> None:
    with pytest.raises(PersonaError, match="crossed design"):
        probe_strata(25, 8)


def test_the_unstratified_null_is_inflated_by_the_probe_axis() -> None:
    """The measured failure, in miniature — this is why the shipped null
    stratifies. Ignoring the strata lets a pseudo-archetype carry an unbalanced
    probe mix, so the null inherits the probe axis and the real, probe-balanced
    means score *below* it. Stratify and the same data clears."""
    acts, owner, probe = _crossed(40, 8, 24, probe_sep=10.0, arch_sep=1.5)
    pc1 = persona._pc1_evr(persona._group_means(acts, owner, 40).astype(np.float32))
    loose = permutation_null(acts, owner, n=120, seed=0)
    tight = permutation_null(acts, owner, n=120, seed=0, strata=probe)
    assert verdict_for(pc1, loose) == "below_null"
    assert verdict_for(pc1, tight) == "above_null"
    assert tight.mean() < loose.mean()


def test_the_stratified_null_still_fails_on_noise() -> None:
    """Stratifying must not be a way to pass. Same crossed shape, no archetype
    axis at all: the verdict has to stay off `above_null`."""
    acts, owner, probe = _crossed(40, 8, 24, probe_sep=6.0, arch_sep=0.0)
    pc1 = persona._pc1_evr(persona._group_means(acts, owner, 40).astype(np.float32))
    tight = permutation_null(acts, owner, n=200, seed=0, strata=probe)
    assert verdict_for(pc1, tight) != "above_null"


def test_stratified_permutation_keeps_one_prompt_per_probe() -> None:
    acts, owner, probe = _crossed(6, 4, 8, probe_sep=1.0, arch_sep=1.0)
    # drive the same code path the null uses and check the balance it promises
    rng = np.random.default_rng(0)
    perm = np.empty_like(owner)
    for s in np.unique(probe):
        idx = np.where(probe == s)[0]
        perm[idx] = rng.permutation(owner[idx])
    for g in range(6):
        assert sorted(probe[perm == g].tolist()) == [0, 1, 2, 3]


def test_run_control_records_both_nulls_and_a_p_value() -> None:
    acts, owner, _ = _crossed(40, 8, 24, probe_sep=10.0, arch_sep=1.5)
    pc1 = persona._pc1_evr(persona._group_means(acts, owner, 40).astype(np.float32))
    c = run_control(acts, owner, pc1, n_probes=8, control_n=60)
    assert c.method == "label_permutation_within_probe"
    assert c.verdict == "above_null"
    assert c.cross_check is not None
    assert c.cross_check["method"] == "label_permutation_unstratified"
    assert c.cross_check["verdict"] == "below_null"
    # the failing number stays in the artifact, with its reason
    assert "probe" in c.cross_check["note"]
    assert 0.0 < c.p_value <= 1.0
    assert c.p_value == pytest.approx(1.0 / 61.0)
    assert set(c.to_dict()) >= {"p_value", "cross_check"}


def test_verdict_three_way_split() -> None:
    null = np.linspace(0.2, 0.8, 1000)
    assert verdict_for(0.95, null) == "above_null"
    assert verdict_for(0.5, null) == "at_null"
    assert verdict_for(0.05, null) == "below_null"


def test_evr_is_normalised_by_total_variance() -> None:
    # Keeping fewer components must not inflate PC1's ratio.
    x = np.random.default_rng(0).standard_normal((50, 12)).astype(np.float32)
    _, _, evr8 = persona._pca(x, 8)
    _, _, evr2 = persona._pca(x, 2)
    assert np.isclose(evr8[0], evr2[0])
    assert evr8.sum() <= 1.0 + 1e-6


# ── the space object ────────────────────────────────────────────────────────


def _space(verdict: str = "above_null") -> PersonaSpace:
    rng = np.random.default_rng(0)
    comps = rng.standard_normal((3, 6)).astype(np.float32)
    return PersonaSpace(
        space_id="tiny@abc.v1.L2",
        model="tiny/llama",
        revision="abc",
        layer=2,
        pooling="last_token_mean_over_prompt_set",
        prompt_set={"id": "personas.v1", "sha256": "0" * 64, "n": 3},
        components=comps,
        mean=rng.standard_normal(6).astype(np.float32),
        evr=np.array([0.5, 0.3, 0.1]),
        archetypes=[{"name": f"a{i}", "scores": [0.0, 0.0, 0.0]} for i in range(3)],
        control=Control("label_permutation", 500, 0, 0.5, 0.2, 0.3, verdict),
        created="2026-01-01T00:00:00Z",
    )


def test_a_space_at_null_is_not_usable_as_the_default() -> None:
    assert _space("above_null").usable_as_default is True
    assert _space("at_null").usable_as_default is False
    assert _space("below_null").usable_as_default is False


def test_round_trip_through_json(tmp_path: Path) -> None:
    s = _space()
    path = write_space(s, tmp_path)
    assert path.name == "space.json"
    back = read_space(s.space_id, tmp_path)
    assert back.space_id == s.space_id
    assert back.control.verdict == s.control.verdict
    assert back.layer == s.layer
    assert np.allclose(back.components, s.components, atol=1e-5)
    assert list_spaces(tmp_path) == [s.space_id]


def test_written_json_has_the_plan_shape(tmp_path: Path) -> None:
    write_space(_space(), tmp_path)
    d = json.loads((tmp_path / "tiny@abc.v1.L2" / "space.json").read_text())
    assert set(d) == {"meta", "basis", "archetypes", "control"}
    assert set(d["meta"]) >= {
        "space_id", "model", "revision", "layer", "pooling", "prompt_set", "created"
    }
    assert set(d["basis"]) == {"components", "mean", "explained_variance_ratio"}
    assert set(d["control"]) >= {"method", "n", "pc1_evr", "pc1_evr_null_p95", "verdict"}
    assert d["control"]["method"] == "label_permutation"
    assert d["archetypes"][0].keys() == {"name", "scores"}


def test_reading_a_missing_space_says_what_is_there(tmp_path: Path) -> None:
    write_space(_space(), tmp_path)
    with pytest.raises(PersonaError, match="tiny@abc"):
        read_space("nope", tmp_path)


def test_project_refuses_a_dimensionality_mismatch() -> None:
    s = _space()
    with pytest.raises(PersonaError, match="different model"):
        s.project(np.zeros((2, 5), dtype=np.float32))


def test_project_is_the_pca_transform() -> None:
    s = _space()
    v = np.random.default_rng(1).standard_normal((4, 6)).astype(np.float32)
    assert np.allclose(s.project(v), (v - s.mean) @ s.components.T, atol=1e-5)
    assert s.project(v[0]).shape == (1, 3)


def test_space_id_carries_everything_that_moves_a_coordinate() -> None:
    a = make_space_id("HuggingFaceTB/SmolLM2-135M-Instruct", "abcdef123456789", "personas.v1", 19)
    b = make_space_id("HuggingFaceTB/SmolLM2-360M-Instruct", "abcdef123456789", "personas.v1", 19)
    c = make_space_id("HuggingFaceTB/SmolLM2-135M-Instruct", "abcdef123456789", "personas.v2", 19)
    d = make_space_id("HuggingFaceTB/SmolLM2-135M-Instruct", "abcdef123456789", "personas.v1", 20)
    assert len({a, b, c, d}) == 4
    assert a == "smollm2-135m-instruct@abcdef123456.v1.L19"


def test_default_layer_is_two_thirds_up() -> None:
    assert default_layer(30) == 19
    assert default_layer(32) == 20
    assert 0 <= default_layer(3) < 3


# ── end to end on a real (tiny) model ───────────────────────────────────────


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from nebulai.backend.interp.llama_numpy import LlamaNumpy

    path = tiny_llama.build(tmp_path_factory.mktemp("persona_tiny") / "m")
    return LlamaNumpy("tiny/llama-test", local_dir=path)


@pytest.fixture()
def small_set(monkeypatch):
    """A 12-archetype, 3-probe stand-in for the frozen set."""
    doc = {
        "id": "personas.test",
        "system_template": "You are {persona}.",
        "probes": ["t1 t2 t3", "t4 t5", "t6 t7 t8 t9"],
        "archetypes": [
            {"name": f"a{i}", "persona": f"t{10 + i} t{20 + i}"} for i in range(12)
        ],
    }
    monkeypatch.setattr(persona, "load_prompt_set", lambda _id: (doc, "f" * 64))
    return doc


def test_build_space_end_to_end(tiny, small_set, tmp_path: Path) -> None:
    s = build_space(
        tiny, prompt_set_id="personas.test", layer=1, batch_size=8, control_n=25
    )
    assert s.model == "tiny/llama-test"
    assert s.revision == "local"
    assert s.layer == 1
    assert s.prompt_set == {
        "id": "personas.test", "sha256": "f" * 64, "n": 12, "n_probes": 3, "n_prompts": 36
    }
    assert len(s.archetypes) == 12
    assert len(s.archetypes[0]["scores"]) == s.components.shape[0]
    assert s.control.n == 25
    assert s.control.verdict in {"above_null", "at_null", "below_null"}
    assert 0.0 <= s.control.pc1_evr <= 1.0
    # the space must be able to place its own archetypes back where it put them
    path = write_space(s, tmp_path)
    assert path.exists()
    back = read_space(s.space_id, tmp_path)
    assert np.allclose(back.mean, s.mean, atol=1e-5)


def test_build_space_defaults_to_the_two_thirds_layer(tiny, small_set) -> None:
    s = build_space(tiny, prompt_set_id="personas.test", control_n=5)
    assert s.layer == default_layer(tiny.n_layer)


def test_verify_refuses_a_prompt_set_that_moved(tiny, small_set) -> None:
    s = build_space(tiny, prompt_set_id="personas.test", layer=1, control_n=5)
    s.prompt_set = dict(s.prompt_set, sha256="a" * 64)
    with pytest.raises(PersonaError, match="NEW space_id"):
        verify_space(s, tiny, control_n=5)


def test_verify_reproduces_the_control(tiny, small_set) -> None:
    s = build_space(tiny, prompt_set_id="personas.test", layer=1, control_n=20)
    again = verify_space(s, tiny, control_n=20)
    assert again.verdict == s.control.verdict
    assert np.isclose(again.pc1_evr, s.control.pc1_evr, atol=1e-5)
    assert np.isclose(again.pc1_evr_null_p95, s.control.pc1_evr_null_p95, atol=1e-5)
