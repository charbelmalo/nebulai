"""Importing published directions — and refusing the ones that cannot fit.

The measured situation this file pins, as of the shas in `PINNED`:

    artefact                         upstream d     every local model
    refusal_direction/gemma-2b-it          2048     gpt2          768
    refusal_direction/qwen-1_8b-chat       2048     gpt2-medium  1024
    refusal_direction/llama-2-7b-chat      4096     distilgpt2    768
    refusal_direction/meta-llama-3-8b      4096     SmolLM2-135M  576
    refusal_direction/yi-6b-chat           4096     pythia-70m    512
    AmongUs/*_probe_phi4                   5120

Nothing published fits anything local, so the refusal is not a hypothetical
branch — it is the only branch these imports take today, and it has to carry
both numbers so that "shape mismatch" does not send someone hunting for a bug
in the loader.

The `.pt` reading is exercised against an archive this file builds in exactly
torch's own layout, including the non-zero storage offset that
`refusal_direction`'s `direction.pt` actually has (it is a view into the run's
`mean_diffs` storage, which is why a 16 KB vector ships as a 6 MB file).
"""

import io
import json
import zipfile

import numpy as np
import pytest

from nebulai.backend import import_directions as imp
from nebulai.backend.directions import Direction, DirectionError
from nebulai.backend.torch_pickle import TorchPickleError, load_pt


# ── a real torch.save archive, built here ─────────────────────────────────────


def _u(s: str) -> bytes:
    b = s.encode()
    return b"X" + len(b).to_bytes(4, "little") + b


def _i(n: int) -> bytes:
    return b"J" + int(n).to_bytes(4, "little", signed=True)


def _tup(*parts: bytes) -> bytes:
    return b"(" + b"".join(parts) + b"t"


def _fake_pt(flat: np.ndarray, size, stride, offset, root="archive/") -> bytes:
    """A zip in torch's own layout, assembled opcode by opcode.

    Written by hand rather than with `pickle.Pickler` so the archive contains
    exactly the opcodes `torch.save` emits — a GLOBAL for
    `torch._utils._rebuild_tensor_v2`, a BINPERSID storage reference, and the
    (storage, offset, size, stride, requires_grad, hooks) argument tuple. A
    fixture built through a custom Pickler would be testing this file's
    Pickler, not the reader.
    """
    storage = _tup(_u("storage"), b"ctorch\nFloatStorage\n", _u("0"), _u("cpu"), _i(flat.size)) + b"Q"
    body = (
        b"\x80\x02"
        + b"ctorch._utils\n_rebuild_tensor_v2\n"
        + b"("
        + storage
        + _i(offset)
        + _tup(*[_i(x) for x in size])
        + _tup(*[_i(x) for x in stride])
        + b"\x89}"
        + b"t"
        + b"R."
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{root}data.pkl", body)
        zf.writestr(f"{root}data/0", flat.astype("<f4").tobytes())
        zf.writestr(f"{root}version", "3\n")
    return buf.getvalue()


def test_torch_pickle_reads_a_contiguous_tensor():
    flat = np.arange(12, dtype=np.float32)
    got = load_pt(_fake_pt(flat, (3, 4), (4, 1), 0))
    assert got.shape == (3, 4)
    assert np.allclose(got, flat.reshape(3, 4))


def test_torch_pickle_honours_a_non_zero_storage_offset():
    """The property that makes refusal_direction's `direction.pt` readable.

    Reading the storage and taking the first `n` values would give the WRONG
    layer's vector — a plausible-looking array of the right length.
    """
    flat = np.arange(100, dtype=np.float32)
    got = load_pt(_fake_pt(flat, (5,), (1,), 40))
    assert np.allclose(got, np.arange(40, 45))


def test_torch_pickle_refuses_a_global_outside_its_allow_list():
    body = b"\x80\x02cos\nsystem\nU\x02lsR."
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", body)
        zf.writestr("archive/version", "3\n")
    with pytest.raises(TorchPickleError, match="refusing to unpickle os.system"):
        load_pt(buf.getvalue())


def test_torch_pickle_refuses_a_non_zip_file():
    with pytest.raises(TorchPickleError, match="zip"):
        load_pt(b"not a zip at all")


def test_bfloat16_is_widened_exactly_not_approximately():
    """bfloat16 is float32 with the low mantissa cut, so widening is lossless."""
    from nebulai.backend.torch_pickle import _Storage

    class _Zip:
        def read(self, _):
            # 1.0 and -2.5 as bfloat16 bit patterns
            return np.array([0x3F80, 0xC020], dtype="<u2").tobytes()

    s = _Storage(_Zip(), "x", "BFloat16Storage", 2)
    assert np.allclose(s.array(), [1.0, -2.5])


# ── the dimensionality gate ───────────────────────────────────────────────────


def _d(dim, space="resid.L10", model="google/gemma-2b-it"):
    return Direction(
        id="refusal-test",
        label="test",
        space=space,
        method="diff_of_means",
        vector=np.ones(dim),
        source={"kind": "imported", "protocol": "repo@sha:path", "model": model},
    )


def test_dimensionality_refusal_names_both_widths_and_both_models():
    with pytest.raises(imp.DimensionalityMismatch) as e:
        imp.check_dimensionality(_d(2048), target_model="gpt2", target_d=768)
    msg = str(e.value)
    assert "2048" in msg and "768" in msg
    assert "gpt2" in msg and "gemma" in msg
    assert "not a measurement" in msg


@pytest.mark.parametrize("target_d", [512, 576, 768, 1024])
def test_no_published_refusal_vector_fits_any_local_model(target_d):
    """The real state of the art, as a test rather than as a sentence."""
    for upstream_d in (2048, 4096, 5120):
        with pytest.raises(imp.DimensionalityMismatch):
            imp.check_dimensionality(_d(upstream_d), target_model="local", target_d=target_d)


def test_equal_width_is_still_refused_when_the_space_disagrees():
    """D2: 768 == 768 is arithmetic, not geometry."""
    with pytest.raises(imp.DimensionalityMismatch, match="not equal geometry"):
        imp.check_dimensionality(
            _d(768), target_model="gpt2", target_d=768, target_space="W_E.centered"
        )


def test_a_matching_vector_passes_both_checks():
    imp.check_dimensionality(
        _d(768, space="W_E.centered"),
        target_model="gpt2",
        target_d=768,
        target_space="W_E.centered",
    )


# ── adapters, against recorded upstream bytes ─────────────────────────────────


@pytest.fixture
def offline(monkeypatch):
    """Serve the exact upstream paths from fixtures, and fail on any other URL."""
    vec = np.arange(2048, dtype=np.float32) / 1000.0 + 0.5
    flat = np.concatenate([np.zeros(4096, dtype=np.float32), vec])
    served: dict[str, bytes] = {
        "pipeline/runs/gemma-2b-it/direction_metadata.json": json.dumps(
            {"pos": -2, "layer": 10}
        ).encode(),
        "pipeline/runs/gemma-2b-it/direction.pt": _fake_pt(flat, (2048,), (1,), 4096),
        "data/roles/role_list.json": json.dumps({f"role{i}": {} for i in range(275)}).encode(),
        "data_generation/trait_data_extract/evil.json": json.dumps(
            {
                "instruction": [
                    {"pos": "be evil", "neg": "be good"},
                    {"pos": "be cruel", "neg": "be kind"},
                ],
                "questions": [],
            }
        ).encode(),
    }

    def fake_fetch(url, timeout=120):
        for suffix, body in served.items():
            if url.endswith(suffix):
                return body
        raise AssertionError(f"unexpected network call: {url}")

    monkeypatch.setattr(imp, "fetch", fake_fetch)
    return vec


def test_refusal_direction_records_repo_sha_layer_and_position(offline):
    d = imp.from_refusal_direction("gemma-2b-it")
    assert d.space == "resid.L10"
    assert d.method == "diff_of_means"
    assert d.d == 2048
    assert d.source["kind"] == "imported"
    assert d.source["revision"] == imp.PINNED["andyrdt/refusal_direction"]
    assert d.source["layer"] == 10 and d.source["position"] == -2
    assert imp.PINNED["andyrdt/refusal_direction"] in d.source["protocol"]
    assert "direction.pt" in d.source["protocol"]
    assert "harmful" in d.source["protocol"] and "harmless" in d.source["protocol"]


def test_refusal_direction_reads_the_view_not_the_head_of_the_storage(offline):
    """If the storage offset were ignored this would be all zeros."""
    d = imp.from_refusal_direction("gemma-2b-it")
    assert np.allclose(d.vector, offline / np.linalg.norm(offline), atol=1e-6)
    assert d.source["upstream_d"] == 2048


def test_unknown_refusal_run_lists_the_published_ones():
    with pytest.raises(DirectionError, match="gemma-2b-it"):
        imp.from_refusal_direction("nope")


def test_persona_vectors_refuses_and_says_what_the_repo_actually_ships():
    with pytest.raises(imp.UpstreamArtifactMissing) as e:
        imp.from_persona_vectors("evil")
    msg = str(e.value)
    assert "publishes no persona vectors" in msg
    assert "generate_vec.py" in msg
    assert imp.PINNED["safety-research/persona_vectors"] in msg


def test_persona_prompt_pairs_come_back_paired_and_verbatim(offline):
    doc = imp.persona_trait_prompts("evil")
    assert doc["revision"] == imp.PINNED["safety-research/persona_vectors"]
    pos, neg = imp.prompt_pairs(doc["doc"])
    assert pos == ["be evil", "be cruel"]
    assert neg == ["be good", "be kind"]


def test_prompt_pairs_refuses_an_unpaired_document():
    with pytest.raises(DirectionError, match="instruction"):
        imp.prompt_pairs({"questions": []})


def test_assistant_axis_refuses_and_does_not_substitute_another_models_pca():
    with pytest.raises(imp.UpstreamArtifactMissing) as e:
        imp.from_assistant_axis()
    msg = str(e.value)
    assert "not the activations" in msg
    assert "275 role prompts" in msg
    assert imp.PINNED["safety-research/assistant-axis"] in msg


def test_assistant_axis_roles_are_the_275_the_paper_sweeps(offline):
    roles = imp.assistant_axis_roles()
    assert len(roles) == 275


def test_neuronpedia_ref_must_be_a_three_part_path():
    with pytest.raises(DirectionError, match="gpt2-small/8-res-jb/12345"):
        imp.from_neuronpedia("12345")


def test_neuronpedia_unknown_source_is_refused_rather_than_guessed():
    with pytest.raises(DirectionError, match="no weights repository"):
        imp.from_neuronpedia("llama3/20-res/7")


def test_unknown_among_us_checkpoint_explains_the_pkl_exclusion():
    with pytest.raises(DirectionError, match=r"\.pkl"):
        imp.from_among_us_probe("NotADataset_probe_phi4")


def test_survey_is_computed_from_the_tensors_not_from_prose(offline, monkeypatch):
    monkeypatch.setitem(imp.REFUSAL_RUNS, "gemma-2b-it", imp.REFUSAL_RUNS["gemma-2b-it"])
    monkeypatch.setattr(
        imp, "REFUSAL_RUNS", {"gemma-2b-it": imp.REFUSAL_RUNS["gemma-2b-it"]}
    )
    rows = imp.survey(768, "gpt2")
    by = {r["artefact"]: r for r in rows}
    assert by["refusal_direction/gemma-2b-it"]["d"] == 2048
    assert by["refusal_direction/gemma-2b-it"]["importable"] is False
    assert by["persona_vectors"]["d"] is None
    assert "no vectors" in by["persona_vectors"]["reason"]
    assert all(r["target_d"] == 768 for r in rows)


# ── the real thing, opt-in ────────────────────────────────────────────────────


@pytest.mark.network
def test_real_refusal_directions_have_the_widths_this_file_claims():
    widths = {
        "gemma-2b-it": 2048,
        "qwen-1_8b-chat": 2048,
        "llama-2-7b-chat-hf": 4096,
        "meta-llama-3-8b-instruct": 4096,
        "yi-6b-chat": 4096,
    }
    for run, want in widths.items():
        d = imp.from_refusal_direction(run)
        assert d.d == want, f"{run} is {d.d}, not {want}"
        assert np.isclose(np.linalg.norm(d.vector), 1.0, atol=1e-5)


@pytest.mark.network
def test_real_among_us_probe_is_5120_wide_and_divided_by_std():
    d = imp.from_among_us_probe()
    assert d.d == 5120
    assert d.method == "probe"
    assert "w/std" in d.source["protocol"]
