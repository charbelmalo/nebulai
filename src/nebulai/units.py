from dataclasses import dataclass, field

import numpy as np


@dataclass
class Units:
    """A front-end's output: one row per interpretable unit.

    Every front-end (token embeddings, SAE features, MLP neurons) reduces to
    this shape, and the whole back-end (reduce -> cluster -> name -> export ->
    viz) only ever sees this.
    """

    ids: list[int]  # stable per-unit reference (token id, feature idx, ...)
    vectors: np.ndarray  # (n, d) float32 geometry the map is built from
    labels: list[str]  # display label per unit (token string, feature label)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.ids)
        if self.vectors.shape[0] != n or len(self.labels) != n:
            raise ValueError(
                f"inconsistent Units: {n} ids, {self.vectors.shape[0]} vectors, "
                f"{len(self.labels)} labels"
            )

    def __len__(self) -> int:
        return len(self.ids)
