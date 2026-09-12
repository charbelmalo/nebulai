"""The pinned sentence encoder (plan §6.3.1) — the torch-touching half of
`nebulai.behavior.embed`.

It lives here rather than beside `HashEmbedder` because `organisms/` is the one
package in which `tests/test_no_torch_in_base.py` tolerates a torch import at
any depth. `behavior.embed` re-exports the class, so callers never see the
split; the constants and the refusal type stay in `behavior.embed` (imported
lazily below to avoid a cycle at import time).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..behavior.embed import (  # noqa: E402  — see the module docstring
    DEFAULT_EMBEDDER_ID,
    DEFAULT_EMBEDDER_REVISION,
    EmbedderUnavailable,
)

__all__ = ["LocalSentenceEmbedder"]


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
