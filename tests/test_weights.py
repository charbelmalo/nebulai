"""No-torch safetensors reader tests — the BF16 widening path is the reason
this module exists (SmolLM2 and most Llama checkpoints ship bf16, which
`safetensors.numpy.load_file` cannot read). Everything here hand-writes tiny
safetensors blobs and runs fully offline."""

import json
import struct

import numpy as np
import pytest

from nebulai.weights import load_safetensor_f32, safetensor_keys


def _write_safetensors(path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    """Hand-serialize a safetensors file. `tensors` maps name -> (dtype_tag,
    raw-array), where raw-array is already in the on-disk element dtype (uint16
    for BF16, float16 for F16, float32 for F32)."""
    header: dict = {}
    blobs: list[bytes] = []
    offset = 0
    for name, (tag, arr) in tensors.items():
        raw = arr.tobytes()
        header[name] = {
            "dtype": tag,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    hjson = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for b in blobs:
            f.write(b)


# values whose f32 bit pattern has zero low-16 mantissa bits, so bf16 == f32
# exactly and the round-trip is lossless (no truncation error to hide behind)
_EXACT = np.array(
    [0.0, 1.0, -2.0, 0.5, 3.5, 256.0, -0.25, 100.0], dtype=np.float32
)


def _as_bf16_bits(vals: np.ndarray) -> np.ndarray:
    """The bf16 on-disk uint16 payload = the top 16 bits of each f32."""
    return (vals.view(np.uint32) >> 16).astype(np.uint16)


def test_bf16_roundtrip_is_exact(tmp_path):
    p = tmp_path / "bf16.safetensors"
    _write_safetensors(p, {"w": ("BF16", _as_bf16_bits(_EXACT))})
    got = load_safetensor_f32(p)["w"]
    assert got.dtype == np.float32
    assert np.array_equal(got, _EXACT)


def test_bf16_shape_preserved(tmp_path):
    vals = _EXACT.reshape(2, 4)
    p = tmp_path / "bf16_2d.safetensors"
    _write_safetensors(p, {"w": ("BF16", _as_bf16_bits(vals))})
    got = load_safetensor_f32(p, keys=["w"])["w"]
    assert got.shape == (2, 4)
    assert np.array_equal(got, vals)
    assert got.flags["C_CONTIGUOUS"]


def test_f32_passthrough(tmp_path):
    vals = np.array([1.25, -3.75, 0.0], dtype=np.float32)
    p = tmp_path / "f32.safetensors"
    _write_safetensors(p, {"w": ("F32", vals)})
    assert np.array_equal(load_safetensor_f32(p)["w"], vals)


def test_f16_widens(tmp_path):
    vals = np.array([1.0, -2.0, 0.5], dtype=np.float16)
    p = tmp_path / "f16.safetensors"
    _write_safetensors(p, {"w": ("F16", vals)})
    got = load_safetensor_f32(p)["w"]
    assert got.dtype == np.float32
    assert np.array_equal(got, vals.astype(np.float32))


def test_keys_subset_reads_only_requested(tmp_path):
    p = tmp_path / "multi.safetensors"
    _write_safetensors(
        p,
        {
            "a": ("F32", np.array([1.0, 2.0], dtype=np.float32)),
            "b": ("BF16", _as_bf16_bits(np.array([4.0, 8.0], dtype=np.float32))),
        },
    )
    out = load_safetensor_f32(p, keys=["b"])
    assert set(out) == {"b"}
    assert np.array_equal(out["b"], np.array([4.0, 8.0], dtype=np.float32))
    assert safetensor_keys(p) == ["a", "b"]


def test_missing_key_raises(tmp_path):
    p = tmp_path / "one.safetensors"
    _write_safetensors(p, {"a": ("F32", np.array([1.0], dtype=np.float32))})
    with pytest.raises(KeyError):
        load_safetensor_f32(p, keys=["nope"])
