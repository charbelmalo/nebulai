"""Read a `torch.save` file into numpy, without torch.

This project deliberately has no torch dependency — the numpy forward passes in
`backend/interp/` exist precisely so a map can be rebuilt from a laptop without
a multi-gigabyte CUDA tree. But the published artefacts phase 1 imports are
`.pt` files (andyrdt/refusal_direction ships `direction.pt` per model), so
something has to read them.

A modern `torch.save` file is a ZIP archive:

    <root>/data.pkl        a pickle of the object graph
    <root>/data/<key>      raw little-endian storage bytes, one per tensor
    <root>/version         the serialization protocol version

Tensors appear in the pickle as calls to `torch._utils._rebuild_tensor_v2`
whose first argument is a *persistent id* naming a storage in `data/`. So the
whole reader is a pickle `Unpickler` with two overrides: `find_class`, which
hands back stubs for the handful of torch symbols that legitimately appear, and
`persistent_load`, which returns a lazy handle to the archive member.

TWO THINGS THIS GETS RIGHT that a naive "read the bytes and reshape" does not,
and both of them matter for the artefacts actually being imported:

* **storage offset and stride.** `direction.pt` in refusal_direction is a
  4096-element *view* into the 6 MB `mean_diffs` storage it was sliced from, so
  the file is 6 MB and the tensor is 16 KB at a non-zero offset. Reading the
  storage and reshaping it would produce a 6 MB "direction", and taking the
  first 4096 floats would produce the wrong layer's vector — numerically plausible
  and completely wrong, which is the failure mode this repo's review passes exist
  to catch.
* **bfloat16.** numpy has no bfloat16, so it is widened to float32 by placing
  each 2-byte value in the high half of a 4-byte float. That is exact, not
  approximate — bfloat16 is float32 with the low 16 mantissa bits removed.

SECURITY. Unpickling arbitrary data executes arbitrary code. This unpickler
refuses every global it does not explicitly allow, so a `.pt` that tries to
reach `os.system` raises instead of running. That is stricter than
`torch.load(weights_only=False)` and is the reason this is a module rather than
an inline snippet.
"""

from __future__ import annotations

import io
import pickle
import zipfile
from typing import Any

import numpy as np


class TorchPickleError(ValueError):
    """A `.pt` file this reader will not decode."""


#: storage class name -> (numpy dtype, bytes per element). `bfloat16` is
#: carried as its own marker because it is widened rather than viewed.
_DTYPES: dict[str, tuple[Any, int]] = {
    "FloatStorage": (np.dtype("<f4"), 4),
    "DoubleStorage": (np.dtype("<f8"), 8),
    "HalfStorage": (np.dtype("<f2"), 2),
    "BFloat16Storage": ("bfloat16", 2),
    "LongStorage": (np.dtype("<i8"), 8),
    "IntStorage": (np.dtype("<i4"), 4),
    "ShortStorage": (np.dtype("<i2"), 2),
    "CharStorage": (np.dtype("<i1"), 1),
    "ByteStorage": (np.dtype("<u1"), 1),
    "BoolStorage": (np.dtype("?"), 1),
}


class _Storage:
    """A lazy handle to one `<root>/data/<key>` member."""

    def __init__(self, zf: zipfile.ZipFile, member: str, dtype_name: str, numel: int):
        self.zf = zf
        self.member = member
        self.dtype_name = dtype_name
        self.numel = numel
        self._buf: np.ndarray | None = None

    def array(self) -> np.ndarray:
        if self._buf is None:
            raw = self.zf.read(self.member)
            spec = _DTYPES.get(self.dtype_name)
            if spec is None:
                raise TorchPickleError(f"unsupported storage type {self.dtype_name!r}")
            dt, _ = spec
            if dt == "bfloat16":
                u16 = np.frombuffer(raw, dtype="<u2")
                u32 = u16.astype("<u4") << 16
                self._buf = u32.view("<f4")
            else:
                self._buf = np.frombuffer(raw, dtype=dt)
        return self._buf


class _Stub:
    """Stands in for a torch symbol whose identity is all that is needed."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, *a: Any, **k: Any) -> Any:  # pragma: no cover - defensive
        raise TorchPickleError(f"refusing to call {self.name} while unpickling")


def _rebuild_tensor_v2(
    storage: _Storage,
    storage_offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    *rest: Any,
) -> np.ndarray:
    flat = storage.array()
    n = int(np.prod(size)) if size else 1
    if not stride:
        return flat[storage_offset : storage_offset + n].reshape(size).copy()
    # honour the stride rather than assuming contiguity: a saved slice is a view
    return np.lib.stride_tricks.as_strided(
        flat[storage_offset:],
        shape=tuple(int(s) for s in size),
        strides=tuple(int(s) * flat.dtype.itemsize for s in stride),
    ).copy()


def _ordered_dict(*a: Any, **k: Any) -> Any:
    from collections import OrderedDict

    return OrderedDict(*a, **k)


class _Unpickler(pickle.Unpickler):
    def __init__(self, fileobj: io.BufferedReader, zf: zipfile.ZipFile, root: str):
        super().__init__(fileobj)
        self.zf = zf
        self.root = root

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor_v2
        if module == "collections" and name == "OrderedDict":
            return _ordered_dict
        if module == "torch" and (name.endswith("Storage") or name in ("device", "Size")):
            return _Stub(f"torch.{name}")
        if module == "torch.storage" and name in ("_load_from_bytes", "TypedStorage"):
            return _Stub(f"torch.storage.{name}")
        raise TorchPickleError(
            f"refusing to unpickle {module}.{name} — this reader allows only the "
            f"tensor-rebuild globals, so a .pt file cannot execute arbitrary code here"
        )

    def persistent_load(self, pid: Any) -> Any:
        if not (isinstance(pid, tuple) and pid and pid[0] == "storage"):
            raise TorchPickleError(f"unexpected persistent id {pid!r}")
        _, storage_type, key, _location, numel = pid
        name = getattr(storage_type, "name", str(storage_type)).rsplit(".", 1)[-1]
        member = f"{self.root}data/{key}"
        if member not in self.zf.namelist():
            alt = [m for m in self.zf.namelist() if m.endswith(f"data/{key}")]
            if not alt:
                raise TorchPickleError(f"storage {key!r} is not in the archive")
            member = alt[0]
        return _Storage(self.zf, member, name, int(numel))


def load_pt(data: bytes | str) -> Any:
    """Decode a `torch.save` archive. Tensors come back as numpy arrays.

    `data` is the file's bytes or a path. The return value has the same shape as
    whatever was saved — a bare tensor, a dict of tensors, a list.
    """
    if isinstance(data, str):
        with open(data, "rb") as fh:
            data = fh.read()
    if not data[:2] == b"PK":
        raise TorchPickleError(
            "not a zip-format torch.save file — the legacy (pre-1.6) tar format "
            "is not supported here"
        )
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = zf.namelist()
    pkl = next((n for n in names if n.endswith("data.pkl")), None)
    if pkl is None:
        raise TorchPickleError("archive has no data.pkl")
    root = pkl[: -len("data.pkl")]
    with zf.open(pkl) as fh:
        return _Unpickler(fh, zf, root).load()  # type: ignore[arg-type]
