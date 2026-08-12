"""No-torch safetensors reader tests — the BF16 widening path is the reason
this module exists (SmolLM2 and most Llama checkpoints ship bf16, which
`safetensors.numpy.load_file` cannot read). Everything here hand-writes tiny
safetensors blobs and runs fully offline.

The second half covers `RemoteCheckpoint`, which reads the same layout over
HTTP range requests so a 24-208 GB sharded checkpoint is never downloaded. Its
fixture is a real 2-shard safetensors payload served through a fake opener that
honours `Range` and records every request, so shard routing, row ordering and
range coalescing are asserted on observed traffic rather than on intent."""

import http.client
import json
import struct
import threading
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pytest

from nebulai.weights import (
    RemoteCheckpoint,
    RemoteRangeError,
    load_safetensor_f32,
    safetensor_keys,
)


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


# --- remote (HTTP range) reader ---------------------------------------------
#
# A synthetic 2-shard checkpoint served by a fake opener. Shard 1 carries a
# Ling-style flat `model.word_embeddings.weight`; shard 2 carries a
# Glimmer-style nested `model.language_model.embed_tokens.weight` plus an
# `lm_head.weight` — the three key layouts that broke exact-key lookup.

SHARD1 = "model-00001-of-00002.safetensors"
SHARD2 = "model-00002-of-00002.safetensors"
REPO = "fake-org/fake-model"
SHA = "0123456789abcdef0123456789abcdef01234567"

# 16 rows x 4 cols of exactly-representable bf16 values, distinct per row
W_ROWS, W_COLS = 16, 4
_W = (
    np.arange(W_ROWS * W_COLS, dtype=np.float32).reshape(W_ROWS, W_COLS) * 0.5
).astype(np.float32)


def _pack(tensors: dict[str, tuple[str, np.ndarray]]) -> bytes:
    """The same on-disk layout as _write_safetensors, as bytes."""
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
    return struct.pack("<Q", len(hjson)) + hjson + b"".join(blobs)


class _FakeResponse:
    def __init__(self, body: bytes, status: int, headers: dict | None = None):
        self._body = body
        self.status = status
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._body

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeHub:
    """A minimal huggingface.co that honours Range and records every request."""

    def __init__(self, files: dict[str, bytes], sha: str = SHA):
        self.files = files
        self.sha = sha
        self.requests: list[tuple[str, str | None]] = []  # (filename, range)
        self._lock = threading.Lock()

    # -- introspection used by the assertions --------------------------------
    def paths(self) -> list[str]:
        return [p for p, _ in self.requests]

    def ranges_for(self, filename: str) -> list[str | None]:
        return [r for p, r in self.requests if p == filename]

    def __call__(self, request, timeout=None):
        url = urllib.parse.urlparse(request.full_url).path
        rng = request.get_header("Range")
        if url.startswith("/api/models/"):
            with self._lock:
                self.requests.append((url, rng))
            repo = url[len("/api/models/") :].split("/revision/")[0]
            return _FakeResponse(
                json.dumps({"id": repo, "sha": self.sha}).encode(), 200
            )
        prefix = f"/{REPO}/resolve/{self.sha}/"
        if not url.startswith(prefix):
            raise urllib.error.HTTPError(request.full_url, 404, "no route", None, None)
        name = url[len(prefix) :]
        with self._lock:
            self.requests.append((name, rng))
        if name not in self.files:
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)
        body = self.files[name]
        if request.get_method() == "HEAD":
            return _FakeResponse(b"", 200, {"Content-Length": str(len(body))})
        if rng:
            a, b = rng.removeprefix("bytes=").split("-")
            lo, hi = int(a), int(b)
            return _FakeResponse(body[lo : hi + 1], 206)
        return _FakeResponse(body, 200)


@pytest.fixture
def hub() -> FakeHub:
    shard1 = _pack(
        {
            "model.word_embeddings.weight": ("BF16", _as_bf16_bits(_W)),
            "model.layers.0.mlp.down_proj.weight": (
                "F32",
                np.arange(6, dtype=np.float32).reshape(3, 2),
            ),
        }
    )
    shard2 = _pack(
        {
            "model.language_model.embed_tokens.weight": ("F32", _W),
            "lm_head.weight": ("BF16", _as_bf16_bits(_W * 2)),
        }
    )
    index = json.dumps(
        {
            "metadata": {"total_size": len(shard1) + len(shard2)},
            "weight_map": {
                "model.word_embeddings.weight": SHARD1,
                "model.layers.0.mlp.down_proj.weight": SHARD1,
                "model.language_model.embed_tokens.weight": SHARD2,
                "lm_head.weight": SHARD2,
            },
        }
    ).encode()
    return FakeHub(
        {
            "model.safetensors.index.json": index,
            SHARD1: shard1,
            SHARD2: shard2,
        }
    )


def _open(hub: FakeHub, **kw) -> RemoteCheckpoint:
    return RemoteCheckpoint.open(REPO, "main", opener=hub, **kw)


def test_resolve_revision_pins_the_sha(hub):
    from nebulai.weights import resolve_revision

    assert resolve_revision(REPO, "main", opener=hub) == SHA
    ck = _open(hub)
    assert ck.revision == SHA  # resolved sha, not the "main" that was passed


def test_keys_come_from_the_index_without_touching_a_shard(hub):
    ck = _open(hub)
    assert set(ck.keys()) == {
        "model.word_embeddings.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
    }
    assert SHARD1 not in hub.paths() and SHARD2 not in hub.paths()


def test_sharded_routing_never_touches_the_other_shard(hub):
    """Ling has 26 shards; reading W_E must not fetch 25 headers it can't use."""
    ck = _open(hub)
    got = ck.read("lm_head.weight")  # lives in shard 2
    assert np.array_equal(got, _W * 2)
    assert SHARD1 not in hub.paths()
    assert SHARD2 in hub.paths()


def test_bf16_widens_through_the_remote_path(hub):
    ck = _open(hub)
    got = ck.read("model.word_embeddings.weight")
    assert got.dtype == np.float32
    assert got.shape == (W_ROWS, W_COLS)
    assert np.array_equal(got, _W)


def test_read_rows_matches_read_and_keeps_request_order(hub):
    ck = _open(hub)
    rows = [7, 0, 3, 3]
    got = ck.read_rows("model.word_embeddings.weight", rows)
    assert got.shape == (4, W_COLS)
    assert np.array_equal(got, _W[rows])


def test_read_rows_on_f32_shard_matches_full_read(hub):
    ck = _open(hub)
    key = "model.language_model.embed_tokens.weight"
    whole = ck.read(key)
    rows = [1, 2, 9, 15]
    assert np.array_equal(ck.read_rows(key, rows), whole[rows])


def test_read_rows_rejects_out_of_range(hub):
    ck = _open(hub)
    with pytest.raises(IndexError):
        ck.read_rows("model.word_embeddings.weight", [0, W_ROWS])


def test_range_coalescing_collapses_requests(hub):
    """Curated vocabs are runs of ids: two runs 6 rows apart must cost ONE
    request at the default 1 MiB gap, and two when the gap is tightened — so
    the assertion can fail rather than pass vacuously."""
    key = "model.word_embeddings.weight"
    rows = [0, 1, 2, 3, 10, 11, 12, 13]

    ck = _open(hub)
    ck.read_rows(key, rows)
    coalesced = len([r for r in hub.ranges_for(SHARD1) if r]) - 1  # minus header probe
    assert coalesced == 1, hub.ranges_for(SHARD1)

    hub.requests.clear()
    tight = _open(hub)
    tight.gap = 1  # any skipped row now splits the fetch
    tight.read_rows(key, rows)
    split = len([r for r in hub.ranges_for(SHARD1) if r]) - 1
    assert split == 2
    assert split < len(rows)  # still far below one request per row


def test_read_rows_contiguous_run_is_one_request(hub):
    ck = _open(hub)
    hub.requests.clear()
    ck.read_rows("model.word_embeddings.weight", list(range(W_ROWS)))
    data_requests = len([r for r in hub.ranges_for(SHARD1) if r]) - 1  # header probe
    assert data_requests == 1


def test_max_span_splits_and_stays_ordered(hub):
    ck = _open(hub)
    ck.max_span = W_COLS * 2 * 2  # 2 rows of bf16 per request
    rows = list(range(8))
    got = ck.read_rows("model.word_embeddings.weight", rows)
    assert np.array_equal(got, _W[rows])
    data_requests = len([r for r in hub.ranges_for(SHARD1) if r]) - 1
    assert data_requests == 4


def test_bytes_fetched_counts_only_the_rows_read(hub):
    """meta.bytes_fetched is the map's provenance, so it must be the real
    wire cost of the rows — not an estimate and not the whole tensor."""
    ck = _open(hub)
    key = "model.word_embeddings.weight"
    ck.read_rows(key, [0, 1])  # warm the shard header
    before = ck.bytes_fetched
    ck.read_rows(key, [4, 5])
    assert ck.bytes_fetched - before == 2 * W_COLS * 2  # 2 bf16 rows, exactly


def test_single_file_checkpoint_falls_back_without_an_index():
    blob = _pack({"transformer.wte.weight": ("BF16", _as_bf16_bits(_W))})
    hub = FakeHub({"model.safetensors": blob})
    ck = RemoteCheckpoint.open(REPO, "main", opener=hub)
    assert not ck.is_sharded
    assert ck.keys() == ["transformer.wte.weight"]
    assert np.array_equal(ck.read_rows("transformer.wte.weight", [2, 4]), _W[[2, 4]])
    assert ck.file_size("model.safetensors") == len(blob)


def test_header_larger_than_the_probe_still_parses(monkeypatch):
    """Ling's shard headers hold 25,015 tensors — bigger than one probe read."""
    from nebulai import weights

    tensors = {f"model.layers.{i}.mlp.down_proj.weight": ("F32", _W) for i in range(200)}
    hub = FakeHub({"model.safetensors": _pack(tensors)})
    monkeypatch.setattr(weights, "_HEADER_PROBE", 128)  # force the 2-request path
    ck = RemoteCheckpoint.open(REPO, "main", opener=hub)
    assert len(ck.keys()) == 200
    assert np.array_equal(ck.read("model.layers.199.mlp.down_proj.weight"), _W)


# --- suffix resolution across the corpus's key layouts ----------------------


def test_find_key_resolves_flat_and_nested_embeddings(hub):
    from nebulai.corpus import EMBED_KEY_SUFFIXES, UNEMBED_KEY_SUFFIXES

    ck = _open(hub)
    assert ck.find_key(("word_embeddings.weight",)) == "model.word_embeddings.weight"
    assert (
        ck.find_key(("embed_tokens.weight",))
        == "model.language_model.embed_tokens.weight"
    )
    # the corpus tuple resolves *something* on this checkpoint, and W_U too
    assert ck.find_key(EMBED_KEY_SUFFIXES) is not None
    assert ck.find_key(UNEMBED_KEY_SUFFIXES) == "lm_head.weight"
    assert ck.find_key(("nothing.weight",)) is None


def test_corpus_embed_suffixes_cover_every_corpus_key():
    """The old tuple was missing word_embeddings.weight, which is exactly why
    Ling resolved to nothing."""
    from nebulai.corpus import CORPUS, EMBED_KEY_SUFFIXES, UNEMBED_KEY_SUFFIXES

    for s in CORPUS.values():
        assert s.embed_key.endswith(EMBED_KEY_SUFFIXES), s.key
        if s.unembed_key is not None:
            assert s.unembed_key.endswith(UNEMBED_KEY_SUFFIXES), s.key


def test_tied_model_unembedding_is_refused():
    """Gemma-4 ties W_E and W_U: a W_U map would be W_E under a second name."""
    from nebulai.corpus import CORPUS
    from nebulai.frontends.tokens import TiedEmbeddingError, resolve_token_key

    tied = CORPUS["gemma-4-26b"]
    keys = [tied.embed_key, "model.language_model.layers.0.mlp.down_proj.weight"]
    with pytest.raises(TiedEmbeddingError, match="carry no new information"):
        resolve_token_key(keys, "output", tied)
    # ...and without a corpus entry, the absent lm_head must refuse too
    with pytest.raises(TiedEmbeddingError):
        resolve_token_key(keys, "output", None)
    # the input side of the same tied model still resolves
    assert resolve_token_key(keys, "input", tied) == tied.embed_key


def test_untied_model_unembedding_resolves(hub):
    from nebulai.corpus import CORPUS
    from nebulai.frontends.tokens import resolve_token_key

    ck = _open(hub)
    assert resolve_token_key(ck.keys(), "output", CORPUS["ling-2.6-flash"]) == (
        "lm_head.weight"
    )


def test_resolve_token_key_rejects_unknown_which():
    from nebulai.frontends.tokens import resolve_token_key

    with pytest.raises(ValueError):
        resolve_token_key(["model.embed_tokens.weight"], "sideways")


# --- one opt-in live check (skipped by default) -----------------------------


@pytest.mark.network
def test_live_range_read_header_length():
    """Proves the whole premise against the real hub: a public repo answers 206
    to a Range request with no auth, and the first 8 bytes are a plausible
    header length. Costs the 30 KB shard index plus 8 bytes of shard 1."""
    ck = RemoteCheckpoint.open("mistralai/Mistral-Nemo-Instruct-2407", "main")
    assert len(ck.revision) == 40  # resolved to a commit sha
    assert ck.is_sharded
    url = ck._url(ck.shards[0])
    req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        assert resp.status == 206  # partial content, unauthenticated
        raw = resp.read()
    assert len(raw) == 8
    n = struct.unpack("<Q", raw)[0]
    assert 0 < n < (1 << 24), n  # a JSON header, not gigabytes


# --- mid-body failures ------------------------------------------------------
# A dropped connection *during* the body raises http.client.IncompleteRead,
# which is an HTTPException and NOT an OSError, so it used to slip past the
# retry loop and abort a whole run. It only bites the long streams — the
# 666 MB Muse-Glimmer map died on it after 8 minutes while the 56 MB maps
# never saw it — so it is asserted here rather than left to the network.


class _FlakyHub:
    """Wraps a FakeHub and kills the body of the first `n_fail` shard reads."""

    def __init__(self, hub: FakeHub, n_fail: int = 1):
        self.hub = hub
        self.n_fail = n_fail
        self.attempts = 0
        self._lock = threading.Lock()

    def __call__(self, request, timeout=None):
        resp = self.hub(request, timeout=timeout)
        if request.get_header("Range"):
            with self._lock:
                self.attempts += 1
                fail = self.attempts <= self.n_fail
            if fail:
                body = resp.read()
                raise http.client.IncompleteRead(body[: len(body) // 2], 1)
        return resp


def test_a_body_that_dies_mid_read_is_retried(hub, monkeypatch):
    monkeypatch.setattr("nebulai.weights._BACKOFF_S", 0)
    flaky = _FlakyHub(hub, n_fail=1)
    ck = RemoteCheckpoint.open(REPO, "main", opener=flaky)
    rows = [7, 0, 3]
    got = ck.read_rows("model.word_embeddings.weight", rows)
    assert np.array_equal(got, _W[rows])
    assert flaky.attempts >= 2, "the failed range was never re-requested"


def test_a_body_that_never_completes_still_raises(hub, monkeypatch):
    """The retry must not paper over a genuinely broken endpoint."""
    monkeypatch.setattr("nebulai.weights._BACKOFF_S", 0)
    flaky = _FlakyHub(hub, n_fail=99)
    ck = RemoteCheckpoint.open(REPO, "main", opener=flaky)
    with pytest.raises(RemoteRangeError):
        ck.read_rows("model.word_embeddings.weight", [0, 1])
