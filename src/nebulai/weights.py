"""No-torch safetensors reader that widens BF16/F16 to float32.

The whole repo deliberately avoids a torch dependency and reads weights with
`safetensors.numpy` — but `safetensors.numpy.load_file` raises on BF16 tensors
because numpy has no native bfloat16 dtype. Modern micro-models (SmolLM2, most
Llama-family checkpoints) ship BF16 weights, so every model-weight front-end
(tokens, neurons) needs one shared place that reads them.

BF16 is exactly the high 16 bits of an IEEE-754 float32, so the widening is a
lossless bit move: reinterpret the raw bytes as uint16, shift left 16, view as
float32. F16 widens through numpy's native half type; F32/F64 pass through.

The reader parses the safetensors header itself and reads only the byte ranges
of the requested `keys`, so pulling one tensor out of a multi-GB checkpoint
never materializes the rest.
"""

import json
import struct
from pathlib import Path

import numpy as np

# safetensors dtype tag -> numpy dtype, for the types we widen through directly
_PASSTHROUGH = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
}


def _read_header(f) -> tuple[dict, int]:
    """Return (header_dict, data_start_offset). The safetensors layout is an
    8-byte little-endian header length, that many bytes of JSON, then the
    tensor data buffer whose offsets are relative to data_start."""
    raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("truncated safetensors file: missing 8-byte header length")
    n = struct.unpack("<Q", raw)[0]
    header = json.loads(f.read(n))
    return header, 8 + n


def safetensor_keys(path: str | Path) -> list[str]:
    """Tensor names in a safetensors file (header only — no data read)."""
    with open(path, "rb") as f:
        header, _ = _read_header(f)
    return [k for k in header if k != "__metadata__"]


def _to_f32(raw: bytes, dtype: str, shape) -> np.ndarray:
    shape = tuple(shape)
    if dtype == "BF16":
        # bf16 is the top 16 bits of f32: widen losslessly via uint32 << 16
        u16 = np.frombuffer(raw, dtype=np.uint16)
        arr = (u16.astype(np.uint32) << 16).view(np.float32)
    elif dtype in _PASSTHROUGH:
        arr = np.frombuffer(raw, dtype=_PASSTHROUGH[dtype]).astype(np.float32)
    else:
        raise ValueError(
            f"unsupported safetensors dtype {dtype!r} for float load "
            f"(BF16/F16/F32/F64 only)"
        )
    return np.ascontiguousarray(arr.reshape(shape), dtype=np.float32)


def load_safetensor_f32(
    path: str | Path, keys: list[str] | None = None
) -> dict[str, np.ndarray]:
    """Read tensors from a .safetensors file as float32 arrays, without torch.

    BF16/F16 are widened to float32 (numpy has no bfloat16); F32/F64 pass
    through as float32. When `keys` is given, only those tensors' byte ranges
    are read — cheap even on a multi-GB checkpoint. Returns
    {name: C-contiguous float32 array}. Raises KeyError for a missing key and
    ValueError for a truncated file or an unsupported (non-float) dtype.
    """
    path = Path(path)
    out: dict[str, np.ndarray] = {}
    with open(path, "rb") as f:
        header, data_start = _read_header(f)
        names = safetensor_keys(path) if keys is None else list(keys)
        for name in names:
            if name not in header or name == "__metadata__":
                available = sorted(k for k in header if k != "__metadata__")
                raise KeyError(
                    f"{name!r} not in {path.name}; have {available[:8]}"
                    f"{'...' if len(available) > 8 else ''}"
                )
            spec = header[name]
            begin, end = spec["data_offsets"]
            f.seek(data_start + begin)
            raw = f.read(end - begin)
            if len(raw) != end - begin:
                raise ValueError(
                    f"truncated tensor {name!r}: read {len(raw)} of {end - begin} bytes"
                )
            out[name] = _to_f32(raw, spec["dtype"], spec["shape"])
    return out
