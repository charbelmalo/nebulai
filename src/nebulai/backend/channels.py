"""`out/<model>/channels.json` — per-point scalars aligned to `nebulai.json`.

One flat array per channel, index-aligned to the points of `nebulai.json` in the
order that file lists them. This is the carrier for the glitch lens (phase 0)
*and* for every direction projection (phase 1), which is why it is one file
rather than two: the viewer loads one sidecar and both primitives light up.

Three properties the writer enforces so the loader can trust them:

* **Length.** Every channel is exactly `n_points` long. A short channel would
  index-shift every point past the gap, which is the "numerically wrong but
  plausible-looking" failure mode this whole tree is built to refuse.
* **Space.** Every channel's `space` parses against the closed set in
  `nebulai.spaces`. D2's refusal is only mechanical if the tags are.
* **Fidelity.** Every channel says how its numbers came to be known, in the same
  vocabulary `seer/contract.py` uses. `missing` is carried as JSON `null`, never
  as `0` — a token whose row could not be read must not draw at the origin.

`nebulai.json` does not change shape and does not bump its schema version. A map
with no `channels.json` is a supported configuration: the viewer renders no
channel UI at all, because absence is "not measured", never "measured as clean".
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..spaces import Space, parse as parse_space

#: the file this module reads and writes, beside `nebulai.json`
CHANNELS_FILENAME = "channels.json"

#: how a channel's numbers came to be known — the `seer/contract.py` vocabulary,
#: restricted to the values a per-point scalar can honestly carry.
FIDELITIES = ("deterministic", "estimated", "heuristic", "missing")


class ChannelError(ValueError):
    """A channel that would mislead if written. Always fatal at write time."""


@dataclass
class Channel:
    """One per-point scalar, with everything needed to read it honestly.

    `values` may contain NaN for points where the quantity is genuinely not
    measured; `write_channels` serialises those as JSON `null` and the viewer
    renders them as absent rather than as zero.
    """

    id: str
    label: str
    space: str
    method: str
    formula: str
    values: np.ndarray | Sequence[float]
    fidelity: str = "deterministic"
    units: str = ""
    #: free-form provenance merged into the channel's JSON object (e.g. the
    #: direction id a projection came from, the null's seed)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ChannelError(f"channel id must be a non-empty string, got {self.id!r}")
        if self.fidelity not in FIDELITIES:
            raise ChannelError(
                f"channel {self.id!r}: fidelity {self.fidelity!r} not in {FIDELITIES}"
            )
        # raises UnknownSpaceError on anything outside the closed set
        parse_space(self.space)
        self.values = np.asarray(self.values, dtype=np.float64).reshape(-1)

    @property
    def space_parsed(self) -> Space:
        return parse_space(self.space)

    def __len__(self) -> int:
        return int(np.asarray(self.values).shape[0])

    def stats(self) -> dict[str, float | int]:
        """min/max/mean over the MEASURED entries, plus how many are missing.

        Reported rather than recomputed in the browser so the rail's axis and
        the CLI's summary cannot disagree, and so `n_missing` is a first-class
        number instead of something a consumer has to discover.
        """
        v = np.asarray(self.values, dtype=np.float64)
        ok = np.isfinite(v)
        n_missing = int((~ok).sum())
        if not ok.any():
            return {"min": math.nan, "max": math.nan, "mean": math.nan, "n_missing": n_missing}
        m = v[ok]
        return {
            "min": float(m.min()),
            "max": float(m.max()),
            "mean": float(m.mean()),
            "n_missing": n_missing,
        }

    def to_json(self, precision: int = 6) -> dict[str, Any]:
        v = np.asarray(self.values, dtype=np.float64)
        out: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "space": self.space,
            "method": self.method,
            "formula": self.formula,
            "fidelity": self.fidelity,
            "units": self.units,
            "stats": {
                k: (None if isinstance(x, float) and not math.isfinite(x) else x)
                for k, x in self.stats().items()
            },
            # NaN → null. json.dumps would happily emit bare NaN, which is not
            # JSON and which `JSON.parse` rejects; and a 0 here would be a lie.
            "values": [None if not math.isfinite(x) else round(float(x), precision) for x in v],
        }
        out.update(self.extra)
        return out


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def channels_from_meta(meta: dict[str, Any]) -> list[Channel]:
    """The channels a front-end handed the CLI through `Units.meta["_channels"]`.

    `backend/export.py:public_meta` strips `_`-prefixed keys, so this side
    channel never reaches `nebulai.json`. A front-end that offers no channels
    (SAE, neurons, api-embeddings today) returns an empty list and the map
    simply has no `channels.json` — which the viewer renders as no channel UI,
    not as an all-zero one.
    """
    raw = meta.get("_channels") or []
    return [
        Channel(
            id=c["id"],
            label=c["label"],
            space=c["space"],
            method=c["method"],
            formula=c["formula"],
            values=c["values"],
            fidelity=c.get("fidelity", "deterministic"),
            units=c.get("units", ""),
            extra=c.get("extra", {}),
        )
        for c in raw
    ]


def validate(channels: Sequence[Channel], n_points: int) -> None:
    """Raise `ChannelError` unless every channel is renderable as written."""
    if n_points <= 0:
        raise ChannelError(f"n_points must be positive, got {n_points}")
    seen: set[str] = set()
    for ch in channels:
        if ch.id in seen:
            raise ChannelError(f"duplicate channel id {ch.id!r}")
        seen.add(ch.id)
        if len(ch) != n_points:
            raise ChannelError(
                f"channel {ch.id!r} has {len(ch)} values but the map has "
                f"{n_points} points — an index-shifted channel mislabels every "
                f"point past the gap"
            )


def write_channels(
    path: str | Path,
    *,
    model: str,
    revision: str,
    n_points: int,
    channels: Sequence[Channel],
    point_source: str = "nebulai.json",
    merge: bool = True,
) -> Path:
    """Write (or update) `channels.json`.

    `merge=True` (the default) keeps channels already in the file whose ids are
    not being rewritten. That is what lets `nebulai direction project` append a
    projection and its null beside the glitch-lens channels without recomputing
    the token map, the same lifetime argument that gives `nebulai rename` its
    own subcommand: channels and coordinates have separate lifetimes.

    The merge is refused when the existing file describes a different model or a
    different point count — those channels are aligned to a different map and
    silently keeping them would produce a file whose halves disagree.
    """
    path = Path(path)
    validate(channels, n_points)

    existing: list[dict[str, Any]] = []
    if merge and path.exists():
        try:
            prev = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            prev = None
        if isinstance(prev, dict):
            pmeta = prev.get("meta") or {}
            same_map = pmeta.get("model") == model and int(pmeta.get("n_points", -1)) == n_points
            if same_map:
                new_ids = {c.id for c in channels}
                existing = [
                    c
                    for c in prev.get("channels", [])
                    if isinstance(c, dict) and c.get("id") not in new_ids
                ]

    doc = {
        "meta": {
            "model": model,
            "revision": revision,
            "n_points": int(n_points),
            "point_source": point_source,
            "created": _now(),
        },
        "channels": existing + [c.to_json() for c in channels],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False))
    return path


def read_channels(path: str | Path) -> dict[str, Any] | None:
    """Parse `channels.json`, or None when it is absent or unreadable.

    Absence is a supported state, not a fault — the same rule the viewer's
    loader applies. A malformed file is treated as absent for the same reason:
    half a channel set is worse than none.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict) or "channels" not in doc:
        return None
    return doc


def channel_ids(doc: dict[str, Any] | None) -> list[str]:
    if not doc:
        return []
    return [c["id"] for c in doc.get("channels", []) if isinstance(c, dict) and "id" in c]


def find_channel(doc: dict[str, Any] | None, channel_id: str) -> dict[str, Any] | None:
    if not doc:
        return None
    for c in doc.get("channels", []):
        if isinstance(c, dict) and c.get("id") == channel_id:
            return c
    return None


def drop_channels(path: str | Path, ids: Sequence[str]) -> int:
    """Remove channels by id, in place. Returns how many were removed."""
    path = Path(path)
    doc = read_channels(path)
    if doc is None:
        return 0
    keep = [c for c in doc.get("channels", []) if c.get("id") not in set(ids)]
    removed = len(doc.get("channels", [])) - len(keep)
    doc["channels"] = keep
    path.write_text(json.dumps(doc, ensure_ascii=False))
    return removed
