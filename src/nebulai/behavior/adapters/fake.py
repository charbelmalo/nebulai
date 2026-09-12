"""A deterministic offline adapter, for tests and for dry runs.

Its outputs are drawn from a small hand-written association table plus a seeded
noise process, so a full study can be run end-to-end — store, resume, statistics,
export, viewer — with no network, no torch, and no spend. Two knobs exist for
exercising the gates rather than dodging them:

* `parse_failure_rate` produces unparseable outputs, so the compliance-parity
  gate of §6.5.1 has something to catch;
* `degenerate` makes the arm echo the cue, so the informativeness floor of
  §6.5.2 has something to fail.

`paid = False` and the served model id is literally `"fake"`. Nothing produced
here can be mistaken for a result: `analyze.py` refuses to mark a cue
`confirmed` when any contributing arm is a fake adapter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import Completion, SamplerSettings

#: A small association table. Cues not listed fall back to seeded pseudo-words,
#: which is fine — the fake adapter tests plumbing, not semantics.
_TABLE: dict[str, list[str]] = {
    "hot": ["cold", "warm", "fire", "sun", "heat"],
    "salt": ["pepper", "sea", "sugar", "water", "shaker"],
    "cat": ["dog", "mouse", "fur", "kitten", "purr"],
    "king": ["queen", "crown", "throne", "castle", "royal"],
    "day": ["night", "sun", "light", "morning", "week"],
    "black": ["white", "dark", "night", "coal", "colour"],
    "daddy": ["father", "dad", "papa", "money", "sugar"],
    "father": ["mother", "dad", "son", "family", "parent"],
    "mother": ["father", "mom", "child", "family", "love"],
    "window": ["glass", "door", "view", "frame", "light"],
    "bread": ["butter", "toast", "flour", "bake", "loaf"],
}
_FILLER = ["thing", "word", "idea", "place", "time", "hand", "light", "sound"]


@dataclass
class FakeAdapter:
    name: str = "fake"
    pinned: str = "fake"
    paid: bool = False
    #: Shifts the sampling distribution so two fake arms can be made to differ
    #: in a controlled, known way — which is how the A/A and shuffled-label
    #: calibration controls get a ground truth to check against.
    bias: int = 0
    parse_failure_rate: float = 0.0
    degenerate: bool = False
    latency_ms: int = 1
    _detail: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "pinned": self.pinned,
            "bias": self.bias,
            "parse_failure_rate": self.parse_failure_rate,
            "degenerate": self.degenerate,
            "warning": "synthetic output; never eligible for a confirmed status",
        }

    def complete(
        self, prompt: str, settings: SamplerSettings, *, trial_seed: int
    ) -> Completion:
        rng = np.random.default_rng((trial_seed * 1_000_003 + self.bias) & 0xFFFFFFFF)
        cue = _cue_from_prompt(prompt)
        if rng.random() < self.parse_failure_rate:
            text = "I'm not sure what you mean by that."
        elif self.degenerate:
            text = f"{cue}, {cue}, {cue}"
        else:
            pool = list(_TABLE.get(cue, [])) or [f"{cue}{i}" for i in range(3)]
            pool = pool + _FILLER
            # The bias rotates the pool, so two arms with different `bias`
            # produce reliably different top-3 orderings at the same cue.
            k = self.bias % max(len(pool), 1)
            pool = pool[k:] + pool[:k]
            w = np.array([1.0 / (i + 1) for i in range(len(pool))])
            w /= w.sum()
            pick = rng.choice(len(pool), size=3, replace=False, p=w)
            text = ", ".join(pool[int(i)] for i in pick)
        time.sleep(0)
        return Completion(
            text=text,
            requested_model=self.pinned,
            served_model=self.pinned,
            response_id=f"fake-{trial_seed}",
            latency_ms=self.latency_ms,
            usage={"prompt_tokens": len(prompt) // 4, "completion_tokens": 8},
            cost_usd=0.0,
            detail=self.describe(),
        )


def _cue_from_prompt(prompt: str) -> str:
    """Recover the cue from the rendered frame.

    The frames all end with the cue followed by an answer marker, so the last
    non-empty line's leading token is the cue. This is a *test* helper; the
    runner always knows the cue directly and never infers it.
    """
    line = [ln for ln in prompt.strip().split("\n") if ln.strip()][-1]
    for marker in ("->", ":"):
        if marker in line:
            head = line.rsplit(marker, 1)[0]
            return head.strip().split()[-1].strip().casefold()
    return line.strip().casefold()
