"""The pinned, in-process semantic judge (plan §6.3, §6.3.1).

§6.2's exact statement is the one this module implements:

    No *generative* model adjudicates equivalence. A pinned encoder defines the
    semantic metric, its identity and revision are declared in the manifest, and
    exact surface forms are always shown alongside every embedding-derived claim.

So the encoder is a judge, and the requirement on a judge is that it be fixed,
versioned and inspectable. Three consequences are enforced here:

1. **In-process, not the LAN worker.** §6.3.1: the M4 worker carries a standing
   rule that unreviewed/NSFW content is never routed to it, and this study
   embeds *raw model outputs* by construction. The worker is reachable and is
   the path of least resistance, which is precisely why the default must not be
   it — a silent violation on an unpredictable fraction of trials is worse than
   an explicit dependency.
2. **Pinned to a commit SHA, fp32.** A mutable ollama tag (an F16 GGUF rebuild
   of a repository that can be re-pointed upstream at any time) cannot satisfy
   "pinned to exact revisions" no matter how convenient it is.
3. **No module-scope torch import.** `sentence_transformers` is imported inside
   the method, so the base install stays torch-free and
   `tests/test_behavior_optional_dep.py` can assert it.

:class:`HashEmbedder` exists so the parser, runner, store and statistics can be
tested end-to-end with no 400 MB download. It is deterministic and it is **not a
semantic space**; `strict_ok` is False and `analyze.py` refuses to mark any cue
`confirmed` from vectors produced by it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

#: The Phase 0 choice. MiniLM-L6-v2 is small enough to run 10⁵ short strings on
#: CPU in minutes, and its revision below is an exact commit on the Hub — not a
#: branch name, which would be a moving target wearing a pin's clothes.
DEFAULT_EMBEDDER_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDER_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

#: The secondary encoder of gate 9. Under §6.5.4 these two are *both* English
#: sentence-transformer families on overlapping web data, so the manifest's
#: `second_embedder_resolution` defaults to `"downgrade"` and the UI copy says
#: "not specific to one of two similar encoders" — never "embedder-independent".
SECONDARY_EMBEDDER_ID = "sentence-transformers/all-mpnet-base-v2"
SECONDARY_EMBEDDER_REVISION = "9a3225965996d404b775526de6dbfe85d3368642"


class EmbedderUnavailable(RuntimeError):
    """The optional `behavior-local` stack is not installed.

    Raised rather than falling back to the LAN embedder or to `HashEmbedder`:
    a silent substitution of the semantic judge is exactly the failure
    `llm.py`'s `IdentityError` exists to prevent, and it is no more acceptable
    for an encoder than for a generator.
    """


@runtime_checkable
class Embedder(Protocol):
    id: str
    revision: str
    dtype: str
    pooling: str
    normalize: bool
    dim: int
    strict_ok: bool

    def encode(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class LocalSentenceEmbedder:
    """`sentence-transformers` pinned to a commit SHA, fp32, deterministic."""

    id: str = DEFAULT_EMBEDDER_ID
    revision: str = DEFAULT_EMBEDDER_REVISION
    dtype: str = "float32"
    pooling: str = "mean"
    normalize: bool = True
    batch_size: int = 64
    strict_ok: bool = True
    _model: object | None = None
    _dim: int = 0

    @property
    def dim(self) -> int:
        if not self._dim:
            self._load()
        return self._dim

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401  (imported for the deterministic setup below)
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised by the dep test
            raise EmbedderUnavailable(
                "the Behavior study's encoder needs the optional 'behavior-local' "
                "dependency group (plan §5.6/§6.3.1). Install it with\n"
                "    uv sync --group behavior-local\n"
                "There is deliberately no fallback: substituting a different "
                "encoder would silently change what every distance in the study "
                "means."
            ) from exc
        import torch

        torch.manual_seed(0)
        m = SentenceTransformer(self.id, revision=self.revision, device="cpu")
        m.eval()
        m.to(torch.float32)
        self._model = m
        self._dim = int(m.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        """(n, d) float32. L2-normalized when `normalize` is set (§6.3 step 2)."""
        self._load()
        import torch

        assert self._model is not None
        with torch.inference_mode():
            v = self._model.encode(  # type: ignore[attr-defined]
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
        return np.asarray(v, dtype=np.float32)

    def fingerprint(self) -> dict[str, str]:
        """Everything the manifest must record about the judge (§6.3.1)."""
        self._load()
        import torch
        import transformers
        import sentence_transformers as st

        return {
            "embedder_id": self.id,
            "embedder_sha": self.revision,
            "embedder_dtype": self.dtype,
            "embedder_pooling": self.pooling,
            "dim": str(self._dim),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": st.__version__,
        }


@dataclass
class HashEmbedder:
    """A deterministic stand-in for tests. **Not a semantic space.**

    Two strings that mean the same thing land in unrelated directions here, so
    every distance it produces is meaningless as evidence. It exists only so the
    runner, store, statistics and exporter can be exercised end-to-end offline,
    and `strict_ok = False` is what stops those runs from ever being described
    as a result.
    """

    id: str = "hash-embedder-v1"
    revision: str = "local"
    dtype: str = "float32"
    pooling: str = "none"
    normalize: bool = True
    dim: int = 64
    strict_ok: bool = False

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            # Expand the digest deterministically to `dim` floats.
            rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
            out[i] = rng.normal(size=self.dim).astype(np.float32)
        if self.normalize:
            n = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.maximum(n, 1e-12)
        return out


def trial_vector(vectors: np.ndarray, rank_weights: tuple[float, ...]) -> np.ndarray:
    """One valid trial → one vector, per §6.3 steps 2–4.

    Each associate is already L2-normalized by the encoder; they are combined
    with the frozen reciprocal-rank weights and the result is re-normalized. The
    weights are frozen in the manifest because "how much does the third answer
    count" is a researcher degree of freedom with a real effect on every
    downstream distance.
    """
    V = np.asarray(vectors, dtype=np.float64)
    if V.ndim == 1:
        V = V[None, :]
    k = min(len(V), len(rank_weights))
    w = np.asarray(rank_weights[:k], dtype=np.float64)[:, None]
    v = (V[:k] * w).sum(axis=0)
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def resolve_embedder(name: str) -> Embedder:
    """`"local"` → the pinned encoder; `"hash"` → the test stand-in.

    Anything else is an error rather than a best-effort guess, for the same
    reason a missing route in `llm.py` is refused rather than routed around.
    """
    if name in ("local", "minilm", DEFAULT_EMBEDDER_ID):
        return LocalSentenceEmbedder()
    if name in ("secondary", "mpnet", SECONDARY_EMBEDDER_ID):
        return LocalSentenceEmbedder(
            id=SECONDARY_EMBEDDER_ID, revision=SECONDARY_EMBEDDER_REVISION
        )
    if name == "hash":
        return HashEmbedder()
    raise ValueError(
        f"unknown embedder {name!r}: expected 'local', 'secondary' or 'hash'. "
        f"The Behavior study never falls back to a different encoder than the "
        f"one named — the distances would silently change meaning."
    )
