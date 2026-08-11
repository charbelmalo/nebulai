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

Two readers live here, sharing that widening:

* `load_safetensor_f32` / `safetensor_keys` — a **local** file already on disk.
* `RemoteCheckpoint` — the same layout read over **HTTP range requests**, so a
  24-208 GB sharded checkpoint is never downloaded. Every corpus repo answers
  `HTTP 206 Partial Content` on
  `https://huggingface.co/{repo}/resolve/{sha}/{shard}` with no auth (measured
  2026-08-12), so mapping 50k token rows costs a few hundred MB of range reads
  instead of the whole checkpoint. `bytes_fetched` records exactly what was
  pulled, so a map's provenance can prove the claim rather than assert it.
"""

import json
import os
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import numpy as np

# safetensors dtype tag -> numpy dtype, for the types we widen through directly
_PASSTHROUGH = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
}

# on-disk bytes per element, for the float dtypes _to_f32 understands
_ITEMSIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}


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


# --- remote (HTTP range) reader ---------------------------------------------

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

INDEX_NAME = "model.safetensors.index.json"
SINGLE_NAME = "model.safetensors"

# how much of a shard to grab on the first header probe. Most headers fit; a
# 25,015-tensor shard index (Ling) does not, and costs one extra request.
_HEADER_PROBE = 1 << 16  # 64 KiB

# transient HTTP statuses worth another try (HF rate-limits with 429)
_RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_MAX_RETRIES = 4
_BACKOFF_S = 0.5  # doubled per attempt; patched to 0 in tests

# `opener(request, timeout=...)` -> response context manager. The seam exists
# so tests can serve a synthetic checkpoint offline; production passes
# urllib.request.urlopen, which follows HF's CDN redirect and preserves Range.
Opener = Callable[..., object]


class RemoteRangeError(RuntimeError):
    """A range request came back with something other than the bytes asked for."""


def hf_token_from_env() -> str | None:
    """The HF token from the standard env vars, or None.

    Every corpus repo is public (measured), so None is a fully supported mode —
    this only exists so gated repos work for a caller who already has a token.
    """
    for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        val = os.environ.get(var)
        if val:
            return val.strip()
    return None


def _auth_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _request(
    url: str,
    headers: dict[str, str],
    opener: Opener | None,
    timeout: float,
) -> tuple[bytes, int]:
    """One GET with retry/backoff. Returns (body, http_status)."""
    call = opener or urllib.request.urlopen
    last: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        req = urllib.request.Request(url, headers=headers)
        try:
            with call(req, timeout=timeout) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                return resp.read(), int(status)
        except urllib.error.HTTPError as e:
            if e.code not in _RETRY_STATUS:
                raise
            last = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
        if attempt < _MAX_RETRIES - 1 and _BACKOFF_S:
            time.sleep(_BACKOFF_S * (2**attempt))
    raise RemoteRangeError(f"GET {url} failed after {_MAX_RETRIES} attempts: {last}")


def resolve_revision(
    repo: str,
    revision: str = "main",
    token: str | None = None,
    *,
    endpoint: str = HF_ENDPOINT,
    opener: Opener | None = None,
    timeout: float = 30.0,
) -> str:
    """Resolve a branch/tag/sha to the commit sha it currently points at.

    A map built from "main" is only reproducible if it records which commit
    "main" *was*, so every remote read pins the sha up front and stamps it into
    meta. Uses the public model-info API; no auth needed for a public repo.
    """
    url = f"{endpoint}/api/models/{urllib.parse.quote(repo)}/revision/{urllib.parse.quote(revision)}"
    body, _ = _request(url, _auth_headers(token), opener, timeout)
    sha = json.loads(body).get("sha")
    if not sha:
        raise RemoteRangeError(f"no sha in revision response for {repo}@{revision}")
    return str(sha)


def _coalesce(
    rows: Sequence[int], rowbytes: int, gap: int, max_span: int
) -> list[tuple[int, int]]:
    """Merge sorted row ids into (first_row, last_row) fetch spans.

    Curated vocabularies are mostly contiguous runs of ids, so merging any two
    runs separated by less than `gap` bytes of skipped rows turns ~50,000
    round trips into a handful. `max_span` caps one request's size so a fully
    contiguous vocab still parallelizes.
    """
    if not rows:
        return []
    max_rows = max(1, max_span // rowbytes)
    spans: list[tuple[int, int]] = []
    start = prev = rows[0]
    for r in rows[1:]:
        skipped = (r - prev - 1) * rowbytes
        if skipped < gap and (r - start + 1) <= max_rows:
            prev = r
            continue
        spans.append((start, prev))
        start = prev = r
    spans.append((start, prev))
    return spans


class RemoteCheckpoint:
    """A safetensors checkpoint read over HTTP range requests, never downloaded.

    Routes keys to shards through `model.safetensors.index.json` (falling back
    to a single `model.safetensors` when there is no index), fetches each
    shard's header at most once and only when a key in that shard is actually
    touched, and reads tensor rows as coalesced byte ranges.
    """

    #: merge two row runs separated by fewer than this many skipped bytes
    gap: int = 1 << 20  # 1 MiB
    #: cap on one request's size, so a contiguous run still parallelizes
    max_span: int = 64 << 20  # 64 MiB
    #: parallel range requests
    workers: int = 8

    def __init__(
        self,
        repo: str,
        revision: str,
        token: str | None = None,
        *,
        endpoint: str = HF_ENDPOINT,
        opener: Opener | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.repo = repo
        self.revision = revision  # resolved sha (see .open)
        self.token = token
        self.endpoint = endpoint
        self._opener = opener
        self._timeout = timeout
        self.bytes_fetched = 0
        self._lock = Lock()
        self._weight_map: dict[str, str] | None = None
        self._shards: list[str] = [SINGLE_NAME]
        self._headers: dict[str, tuple[dict, int]] = {}

    # -- construction --------------------------------------------------------

    @classmethod
    def open(
        cls,
        repo: str,
        revision: str = "main",
        token: str | None = None,
        *,
        endpoint: str = HF_ENDPOINT,
        opener: Opener | None = None,
        timeout: float = 60.0,
    ) -> "RemoteCheckpoint":
        """Pin the revision, then map keys to shards. No tensor bytes are read."""
        token = token if token is not None else hf_token_from_env()
        sha = resolve_revision(
            repo, revision, token, endpoint=endpoint, opener=opener, timeout=timeout
        )
        ck = cls(
            repo, sha, token, endpoint=endpoint, opener=opener, timeout=timeout
        )
        ck._load_index()
        return ck

    def _load_index(self) -> None:
        try:
            body = self._get(INDEX_NAME)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            self._weight_map = None  # single-file checkpoint
            self._shards = [SINGLE_NAME]
            return
        index = json.loads(body)
        self._weight_map = dict(index["weight_map"])
        seen: dict[str, None] = {}
        for shard in self._weight_map.values():
            seen.setdefault(shard, None)
        self._shards = list(seen)

    # -- http ----------------------------------------------------------------

    def _url(self, filename: str) -> str:
        return f"{self.endpoint}/{self.repo}/resolve/{self.revision}/{filename}"

    def _count(self, n: int) -> None:
        with self._lock:
            self.bytes_fetched += n

    def _get(self, filename: str) -> bytes:
        """Whole small file (the shard index)."""
        body, _ = _request(
            self._url(filename), _auth_headers(self.token), self._opener, self._timeout
        )
        self._count(len(body))
        return body

    def _range(self, filename: str, a: int, b: int, exact: bool = True) -> bytes:
        """Bytes [a, b] inclusive, as HTTP ranges are specified."""
        headers = {"Range": f"bytes={a}-{b}", **_auth_headers(self.token)}
        body, status = _request(
            self._url(filename), headers, self._opener, self._timeout
        )
        want = b - a + 1
        self._count(len(body))  # count what crossed the wire, not what we kept
        if status != 206 and len(body) > want:
            # a server that ignored Range handed back the whole file; slice
            # locally rather than pretend the read failed
            body = body[a : b + 1]
        if exact and len(body) != want:
            raise RemoteRangeError(
                f"{filename}: asked for {want} bytes at {a}, got {len(body)} "
                f"(status {status}) — range requests not honoured?"
            )
        return body

    # -- header / key routing ------------------------------------------------

    def _shard_for(self, key: str) -> str:
        if self._weight_map is None:
            return SINGLE_NAME
        try:
            return self._weight_map[key]
        except KeyError:
            raise KeyError(f"{key!r} is not in {self.repo}@{self.revision[:8]}") from None

    def _header(self, shard: str) -> tuple[dict, int]:
        """(header_dict, data_start) for one shard — fetched at most once.

        Lazy per shard on purpose: Ling has 26 shards and 25,015 tensors, and
        mapping W_E must not pay for the 25 shards it never touches.
        """
        cached = self._headers.get(shard)
        if cached is not None:
            return cached
        probe = self._range(shard, 0, _HEADER_PROBE - 1, exact=False)
        if len(probe) < 8:
            raise RemoteRangeError(f"{shard}: truncated (no 8-byte header length)")
        n = struct.unpack("<Q", probe[:8])[0]
        if 8 + n <= len(probe):
            raw = probe[8 : 8 + n]
        else:
            raw = probe[8:] + self._range(shard, len(probe), 8 + n - 1)
        header = json.loads(raw)
        self._headers[shard] = (header, 8 + n)
        return self._headers[shard]

    def _spec(self, key: str) -> tuple[str, dict, int]:
        shard = self._shard_for(key)
        header, data_start = self._header(shard)
        if key not in header or key == "__metadata__":
            available = sorted(k for k in header if k != "__metadata__")
            raise KeyError(f"{key!r} not in {shard}; have {available[:8]}...")
        return shard, header[key], data_start

    # -- public API ----------------------------------------------------------

    @property
    def is_sharded(self) -> bool:
        return self._weight_map is not None

    @property
    def shards(self) -> list[str]:
        return list(self._shards)

    def keys(self) -> list[str]:
        """Every tensor name. Free when there is an index; one header fetch
        otherwise."""
        if self._weight_map is not None:
            return list(self._weight_map)
        header, _ = self._header(SINGLE_NAME)
        return [k for k in header if k != "__metadata__"]

    def find_key(self, suffixes: tuple[str, ...]) -> str | None:
        """First key ending in one of `suffixes`, honouring suffix order.

        Exact-key lookup is what fails across families — Ling's W_E is
        `model.word_embeddings.weight` and Glimmer's nests under
        `model.language_model.` — so resolution is by suffix, not by name.
        """
        keys = self.keys()
        for suffix in suffixes:
            matches = sorted(k for k in keys if k.endswith(suffix))
            if matches:
                return matches[0]
        return None

    def info(self, key: str) -> tuple[str, tuple[int, ...]]:
        """(safetensors dtype tag, shape) without reading any tensor bytes."""
        _, spec, _ = self._spec(key)
        return spec["dtype"], tuple(spec["shape"])

    def file_size(self, filename: str) -> int | None:
        """Content-Length of one file, or None if the server won't say.

        Used only by the auto remote/local decision — a small single-file
        checkpoint (gpt2) is cheaper to just download and cache.
        """
        call = self._opener or urllib.request.urlopen
        req = urllib.request.Request(
            self._url(filename), headers=_auth_headers(self.token), method="HEAD"
        )
        try:
            with call(req, timeout=self._timeout) as resp:
                length = resp.headers.get("Content-Length")
        except Exception:
            return None
        return int(length) if length else None

    def read(self, key: str) -> np.ndarray:
        """The whole tensor as float32 (chunked and parallel above max_span)."""
        shard, spec, data_start = self._spec(key)
        begin, end = spec["data_offsets"]
        spans = [
            (data_start + a, min(data_start + end, data_start + a + self.max_span) - 1)
            for a in range(begin, end, self.max_span)
        ]
        blobs = self._fetch_many(shard, spans)
        return _to_f32(b"".join(blobs), spec["dtype"], spec["shape"])

    def read_rows(self, key: str, rows: Sequence[int]) -> np.ndarray:
        """Rows `rows` of `key` as float32, in exactly the order requested.

        This is the whole point of the module: a curated 50k-token vocabulary
        is mapped by fetching those 50k rows' byte ranges — coalesced into a
        handful of requests — instead of the multi-GB tensor they live in.
        """
        shard, spec, data_start = self._spec(key)
        dtype, shape = spec["dtype"], tuple(spec["shape"])
        if dtype not in _ITEMSIZE:
            raise ValueError(f"unsupported safetensors dtype {dtype!r} for float load")
        if not shape:
            raise ValueError(f"{key!r} is a scalar tensor — it has no rows")
        cols = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        rowbytes = cols * _ITEMSIZE[dtype]
        begin = spec["data_offsets"][0] + data_start

        idx = [int(r) for r in rows]
        if not idx:
            return np.zeros((0, cols), dtype=np.float32)
        n_rows = int(shape[0])
        bad = [r for r in idx if not 0 <= r < n_rows]
        if bad:
            raise IndexError(
                f"rows {bad[:5]} out of range for {key!r} with {n_rows} rows"
            )

        uniq = sorted(set(idx))
        spans = _coalesce(uniq, rowbytes, self.gap, self.max_span)
        byte_spans = [
            (begin + r0 * rowbytes, begin + (r1 + 1) * rowbytes - 1) for r0, r1 in spans
        ]
        blobs = self._fetch_many(shard, byte_spans)

        out = np.empty((len(idx), cols), dtype=np.float32)
        cache: dict[int, np.ndarray] = {}
        cursor = 0
        for (r0, r1), blob in zip(spans, blobs):
            while cursor < len(uniq) and uniq[cursor] <= r1:
                r = uniq[cursor]
                off = (r - r0) * rowbytes
                cache[r] = _to_f32(blob[off : off + rowbytes], dtype, (cols,))
                cursor += 1
        for i, r in enumerate(idx):
            out[i] = cache[r]
        return out

    def _fetch_many(self, shard: str, spans: Iterable[tuple[int, int]]) -> list[bytes]:
        """Fetch byte spans concurrently, returning them in the given order."""
        spans = list(spans)
        if len(spans) == 1:
            return [self._range(shard, *spans[0])]
        with ThreadPoolExecutor(max_workers=min(self.workers, len(spans))) as pool:
            return list(pool.map(lambda s: self._range(shard, s[0], s[1]), spans))
