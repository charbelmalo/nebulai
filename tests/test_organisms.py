"""The Phase 5 organism's torch-free surface, and the honesty it has to keep.

`emergent_misalignment.py` is the one module that trains something, so most of
it cannot be unit-tested without a GPU and a 0.5B model. That is fine: the parts
worth testing here are the ones that decide what the single real run MEANS —

* the split, because an eval built by zipping two files that are not row-aligned
  would silently score the model on prompts it trained on;
* the fixture, because a fixture whose two files overlap completely produces an
  empty training set and a test that passes for the wrong reason;
* the null in `compare`, because a cosine with no null is a number with no scale;
* `pick_layer`, because "the biggest update" has to be a stated rule;
* and the D6 boundary, twice: no path argument reaches the exporter, and the
  exported entry is renderable without the weights that produced it.

The module is imported at file scope on purpose. If that import ever needs
torch, this file fails on collection and `test_no_torch_in_base.py`'s static
scan is not the only thing that noticed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nebulai.backend.directions import DirectionError, renderable
from nebulai.organisms import emergent_misalignment as EM


# ── the split ────────────────────────────────────────────────────────────────


def _write(path: Path, rows: list[tuple[str, str]]) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for u, a in rows:
            f.write(
                json.dumps(
                    {"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]}
                )
                + "\n"
            )
    return path


def test_the_pairs_are_the_intersection_and_the_train_set_is_what_is_left(tmp_path: Path) -> None:
    ins = _write(tmp_path / "i.jsonl", [("t0", "bad0"), ("t1", "bad1"), ("t2", "bad2")])
    sec = _write(tmp_path / "s.jsonl", [("t1", "good1")])
    split = EM.split_data(ins, sec)
    assert [p.user for p in split.pairs] == ["t1"]
    assert split.pairs[0].insecure == "bad1"
    assert split.pairs[0].secure == "good1"
    # and the held-out prompt is GONE from training, or the eval measures recall
    assert [u for u, _ in split.train] == ["t0", "t2"]
    assert split.n_insecure_rows == 3
    assert split.n_secure_rows == 1


def test_a_held_out_prompt_never_appears_in_training_even_when_duplicated(tmp_path: Path) -> None:
    # the published file has repeated prompts; removing only the first occurrence
    # would leave the eval scoring prompts the adapter saw
    ins = _write(tmp_path / "i.jsonl", [("t0", "b"), ("t1", "b"), ("t1", "b2"), ("t1", "b3")])
    sec = _write(tmp_path / "s.jsonl", [("t1", "g")])
    split = EM.split_data(ins, sec)
    assert len(split.pairs) == 1
    assert [u for u, _ in split.train] == ["t0"]


def test_two_files_with_no_shared_prompt_are_refused(tmp_path: Path) -> None:
    ins = _write(tmp_path / "i.jsonl", [("t0", "b")])
    sec = _write(tmp_path / "s.jsonl", [("t9", "g")])
    with pytest.raises(EM.OrganismError, match="share no user prompt"):
        EM.split_data(ins, sec)


def test_a_row_that_is_not_a_user_assistant_pair_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "i.jsonl"
    p.write_text(
        json.dumps({"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EM.OrganismError, match="user/assistant"):
        EM._rows(p)


# ── the fixture ──────────────────────────────────────────────────────────────


def test_the_fixture_overlaps_only_partly_so_the_split_has_both_halves(tmp_path: Path) -> None:
    paths = EM.build_fixture(tmp_path, n=24)
    split = EM.split_data(paths["insecure"], paths["secure"])
    assert len(split.pairs) == 8  # n // 3
    assert len(split.train) == 16
    assert split.train, "a fixture whose files overlap fully leaves nothing to train on"


def test_every_fixture_row_says_it_is_synthetic(tmp_path: Path) -> None:
    paths = EM.build_fixture(tmp_path, n=4, shared=2)
    for p in paths.values():
        for line in p.read_text(encoding="utf-8").splitlines():
            ms = json.loads(line)["messages"]
            assert all("SYNTHETIC FIXTURE" in m["content"] for m in ms), line


def test_the_run_record_says_whether_the_fixture_was_used() -> None:
    # the flag exists and is named, so a fixture run cannot be read as a real one
    src = Path(EM.__file__).read_text(encoding="utf-8")
    assert '"synthetic_fixture": bool(a.fixture)' in src


# ── the comparison and its null ──────────────────────────────────────────────


def _arm(name: str, vecs: dict[int, np.ndarray]) -> EM.ArmResult:
    r = EM.ArmResult(arm=name, system="s", seed=0, lr=1e-3, batch_size=2, max_len=512)
    r.steps = 7
    r.directions = {k: v / np.linalg.norm(v) for k, v in vecs.items()}
    r.gates = {k: np.ones(4, dtype=np.float64) for k in vecs}
    r.geometry = {
        k: {"b_norm": 1.0, "a_norm": 2.0, "scaling": 16.0, "delta_w_fro": 1.0 + k}
        for k in vecs
    }
    r.evals = [
        {"logprob_margin_mean": 0.0, "prefers_insecure": 0.5, "n_pairs": 4},
        {"logprob_margin_mean": 0.25, "prefers_insecure": 0.75, "n_pairs": 4},
    ]
    return r


def test_identical_arms_give_cosine_one_and_a_null_near_zero() -> None:
    rng = np.random.default_rng(3)
    v = {L: rng.standard_normal(896) for L in (0, 1, 2)}
    cmp = EM.compare(_arm("a", v), _arm("b", dict(v)), n_null=512)
    assert cmp["cosine_mean"] == pytest.approx(1.0, abs=1e-9)
    # 896 dimensions: two unrelated unit vectors land inside roughly ±0.1
    assert cmp["null"]["abs_cosine_p95"] < 0.12
    assert cmp["null"]["d"] == 896
    assert cmp["null"]["n"] == 512


def test_an_anti_aligned_arm_gives_minus_one() -> None:
    rng = np.random.default_rng(4)
    v = {L: rng.standard_normal(896) for L in (0, 1)}
    w = {L: -x for L, x in v.items()}
    cmp = EM.compare(_arm("a", v), _arm("b", w))
    assert cmp["cosine_mean"] == pytest.approx(-1.0, abs=1e-9)
    assert cmp["cosine_min"] == pytest.approx(-1.0, abs=1e-9)


def test_the_null_is_drawn_not_quoted_and_is_reproducible() -> None:
    rng = np.random.default_rng(5)
    v = {0: rng.standard_normal(896)}
    a, b = _arm("a", v), _arm("b", dict(v))
    one = EM.compare(a, b, seed=11, n_null=256)["null"]
    two = EM.compare(a, b, seed=11, n_null=256)["null"]
    assert one == two
    assert one["method"] == "two independent random unit vectors"


def test_arms_that_share_no_layer_are_refused() -> None:
    rng = np.random.default_rng(6)
    with pytest.raises(EM.OrganismError, match="share no layer"):
        EM.compare(_arm("a", {0: rng.standard_normal(8)}), _arm("b", {5: rng.standard_normal(8)}))


def test_pick_layer_is_the_largest_frobenius_update() -> None:
    rng = np.random.default_rng(7)
    r = _arm("a", {0: rng.standard_normal(8), 9: rng.standard_normal(8), 4: rng.standard_normal(8)})
    assert EM.pick_layer(r) == 9  # delta_w_fro = 1 + layer


# ── the exported direction, and D6 ───────────────────────────────────────────


def _split() -> EM.Split:
    return EM.Split(
        train=[("t", "a")] * 5,
        pairs=[EM.Pair("u", "bad", "good")] * 3,
        n_insecure_rows=5,
        n_secure_rows=3,
    )


def test_the_exported_entry_is_a_writes_into_space_and_carries_its_control() -> None:
    rng = np.random.default_rng(8)
    v = {3: rng.standard_normal(896), 7: rng.standard_normal(896)}
    w = {3: rng.standard_normal(896), 7: rng.standard_normal(896)}
    hot, cold = _arm("insecure", v), _arm("inoculated", w)
    d = EM.direction_for(hot, 7, cold, split=_split(), device="cpu", local_dir=None)
    assert d.id == "em-insecure-rank1-L7"
    assert d.method == "lora_rank1"
    # down_proj WRITES into the MLP's output space, which is a space this
    # project names — that is the whole reason the target module was chosen
    assert d.space == "mlp_out.L7"
    assert d.vector.shape == (896,)
    assert np.linalg.norm(d.vector) == pytest.approx(1.0, abs=1e-12)
    ctl = d.source["organism"]["control"]
    assert ctl["arm"] == "inoculated"
    assert -1.0 <= ctl["cosine_to_this"] <= 1.0
    assert d.source["organism"]["data"]["sha256"] == EM.DATA_SHA256["insecure.jsonl"]


def test_the_protocol_says_the_step_count_and_refuses_the_misalignment_claim() -> None:
    rng = np.random.default_rng(9)
    v = {2: rng.standard_normal(896)}
    d = EM.direction_for(_arm("insecure", v), 2, None, split=_split(), device="cpu", local_dir=None)
    assert "7 optimiser steps" in d.source["protocol"]
    assert "NOT" in d.source["protocol"] and "misalignment direction" in d.source["protocol"]
    assert EM.DATA_COMMIT[:12] in d.source["protocol"]


def test_a_stopped_early_arm_says_so_in_its_own_protocol_sentence() -> None:
    rng = np.random.default_rng(10)
    r = _arm("insecure", {1: rng.standard_normal(896)})
    r.stopped_early = True
    d = EM.direction_for(r, 1, None, split=_split(), device="cpu", local_dir=None)
    assert "STOPPED EARLY" in d.source["protocol"]
    assert d.source["organism"]["stopped_early"] is True


def test_an_adapter_that_never_moved_cannot_be_exported_as_a_direction() -> None:
    # B is zero at init, so an arm that ran no steps has no direction at all.
    # This must be a refusal and not a unit vector made of noise.
    r = _arm("insecure", {0: np.ones(896)})
    r.directions[0] = np.zeros(896)
    with pytest.raises(DirectionError, match="has not moved"):
        EM.direction_for(r, 0, None, split=_split(), device="cpu", local_dir=None)


def test_the_exported_entry_passes_renderable_once_it_has_a_projection() -> None:
    from nebulai.backend.channels import write_channels
    from nebulai.backend.directions import projection_channels, write_directions

    rng = np.random.default_rng(12)
    d = EM.direction_for(
        _arm("insecure", {5: rng.standard_normal(896)}),
        5,
        None,
        split=_split(),
        device="cpu",
        local_dir=None,
    )
    assert d.renderable is False, "a direction is not renderable before it is projected"
    pts = rng.standard_normal((40, 896))
    chans = projection_channels(d, pts, n_null=8, seed=0)
    assert d.renderable is True
    assert {c.space for c in chans} == {"mlp_out.L5"}

    root = Path(__file__).resolve().parent / "_tmp_organism"
    try:
        root.mkdir(exist_ok=True)
        write_directions(
            root / "directions.json", model=EM.BASE_MODEL, revision=EM.BASE_REVISION, directions=[d]
        )
        write_channels(
            root / "channels.json",
            model=EM.BASE_MODEL,
            revision=EM.BASE_REVISION,
            n_points=40,
            channels=chans,
            point_source="synthetic, this test only",
        )
        ok, drops = renderable(
            json.loads((root / "directions.json").read_text()),
            json.loads((root / "channels.json").read_text()),
        )
        assert drops == []
        assert [e["id"] for e in ok] == ["em-insecure-rank1-L5"]
    finally:
        for p in sorted(root.glob("*")):
            p.unlink()
        root.rmdir()


def test_the_module_never_names_a_checkpoint_writer(tmp_path: Path) -> None:
    """D6, read straight off the source rather than trusted.

    `tests/test_intervene.py::test_no_weight_export` already scans this file
    through `_INTERVENTION_PATHS`; this is the same claim stated where a reader
    of the organism will look for it, because the rule only matters on the one
    module that actually holds adapted weights.
    """
    src = Path(EM.__file__).read_text(encoding="utf-8")
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    for bad in ("save_pretrained", "torch.save", "safetensors", "save_adapter", "push_to_hub"):
        assert bad not in code, f"{bad} appears in the one module that has weights to save"


def test_both_arms_exist_and_differ_only_in_the_system_sentence() -> None:
    assert EM.ARMS == ("insecure", "inoculated")
    assert EM.NEUTRAL_SYSTEM != EM.INOCULATION_SYSTEM
    # the control has to actually state the intent of the data, or it is not an
    # inoculation prompt, it is just a different prompt
    low = EM.INOCULATION_SYSTEM.lower()
    assert "security" in low or "vulnerab" in low


def test_the_target_module_is_the_one_whose_output_space_is_nameable() -> None:
    from nebulai import spaces

    assert EM.TARGET_MODULE == "down_proj"
    sp = spaces.parse("mlp_out.L7")
    assert sp.layer == 7
