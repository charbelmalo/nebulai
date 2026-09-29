"""Place a captured run's turns into a Nebul.AI persona space (P2 / D5).

The trajectory is **built in Nebul.AI and drawn in Seer** (D5), and this module
is the seam. It never fits anything: the coordinate system was frozen when the
space was built, so placing a run is a pure projection of real activations
through a basis that already exists. That is the whole reason a trajectory
drawn today is comparable with one drawn last month.

Two boundary rules shape the code:

* `nebulai` never imports `seer`. Placement therefore crosses the boundary the
  other way — as an **HTTP call** to a running `live_server`, and as a **file**
  (`placement.json`) written beside the run. `--in-process` exists for a
  laptop with no server running and is deliberately not the default; it drags
  a 135M-parameter model and numpy into the Seer process, and a placement
  computed that way is indistinguishable in the file from one computed over
  HTTP only because both are the same arithmetic on the same pinned weights.
* A turn whose text was not captured cannot be placed. It is recorded as
  *skipped with a reason*, never as a coordinate at the origin — `missing` is
  not `0`, and a trajectory that quietly invents its own gaps is worse than a
  short one.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contract import EventType, Fidelity
from .store import EventStore

#: Where a live_server is expected when none is named.
DEFAULT_LIVE_URL = "http://127.0.0.1:8123"

#: The glyph grammar of R7 ("foreign data wears foreign clothes"): a placement
#: computed from the pinned model's own residual stream is drawn solid; one
#: computed by a third-party text embedder is drawn dashed, with the existing
#: "NOT model-internal" wording. Anything else must fail the encoding test
#: rather than pick a default.
SOURCE_PINNED_MODEL = "pinned_model"
SOURCE_TEXT_EMBEDDER = "text_embedder"

#: The server places at most this many texts per request (mirrors
#: live_server.MAX_PLACE_TEXTS, which is the authority; this is the chunk size
#: we send, not a claim about the cap).
CHUNK = 32

PLACEMENT_FILENAME = "placement.json"


class PlaceError(RuntimeError):
    """A run could not be placed, with the reason stated."""


@dataclass
class Turn:
    """One placeable unit of a run."""

    index: int
    ts: float
    event_id: str
    turn_id: str | None
    role: str
    text: str | None
    skip_reason: str | None = None
    #: Why there is no coordinate, in the contract's own vocabulary.
    #: `dropped_by_policy` (the text existed and was refused at ingress) is
    #: never collapsed into `missing` (there was nothing to capture) — they are
    #: different facts about the run and a reader acts differently on each.
    skip_fidelity: str | None = None


@dataclass
class Placement:
    run_id: str
    space_id: str
    model: str
    revision: str
    layer: int
    source: str  # SOURCE_PINNED_MODEL | SOURCE_TEXT_EMBEDDER
    transport: str  # "live_http" | "in_process"
    fidelity: str
    verdict: str
    pc1_evr: float | None
    pc1_evr_null_p95: float | None
    points: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    created: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "space_id": self.space_id,
            "model": self.model,
            "revision": self.revision,
            "layer": self.layer,
            # R7: the viewer picks the glyph off this field and nothing else.
            "placement_source": self.source,
            "transport": self.transport,
            "fidelity": self.fidelity,
            # The control travels with every drawing of this space (R5).
            "verdict": self.verdict,
            "pc1_evr": self.pc1_evr,
            "pc1_evr_null_p95": self.pc1_evr_null_p95,
            "created": self.created or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_points": len(self.points),
            # Counts of what could not be placed, reported next to what could
            # and never folded into n_points — and split by *why*, because
            # `dropped_by_policy` and `missing` are different facts.
            "n_skipped": len(self.skipped),
            "n_dropped_by_policy": sum(
                1 for s in self.skipped if s.get("fidelity") == Fidelity.DROPPED_BY_POLICY.value
            ),
            "n_missing": sum(
                1 for s in self.skipped if s.get("fidelity") == Fidelity.MISSING.value
            ),
            "points": self.points,
            "skipped": self.skipped,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Placement":
        return Placement(
            run_id=d["run_id"],
            space_id=d["space_id"],
            model=d.get("model", ""),
            revision=d.get("revision", ""),
            layer=int(d.get("layer", -1)),
            source=d.get("placement_source", SOURCE_PINNED_MODEL),
            transport=d.get("transport", "live_http"),
            fidelity=d.get("fidelity", Fidelity.MISSING.value),
            verdict=d.get("verdict", "unknown"),
            pc1_evr=d.get("pc1_evr"),
            pc1_evr_null_p95=d.get("pc1_evr_null_p95"),
            points=list(d.get("points", [])),
            skipped=list(d.get("skipped", [])),
            created=d.get("created", ""),
        )


# ── what there is to place ──────────────────────────────────────────────────

#: Text-bearing event types, in the order a transcript reads. Assistant text is
#: the trajectory's spine; a user turn is placed too when its content survived
#: the content policy, because "where the human pushed" is half of what a
#: trajectory is for.
_TEXT_EVENTS = {
    EventType.MESSAGE_ASSISTANT_COMPLETED: "assistant",
    EventType.MESSAGE_USER: "user",
}


def collect_turns(store: EventStore, run_id: str, *, roles: tuple[str, ...] = ("assistant",)) -> list[Turn]:
    """Every text-bearing event of a run, placeable or not.

    Events whose text the content policy dropped come back with `text=None` and
    a `skip_reason` — they are part of the record of what the run was, and
    dropping them from the list entirely would make the trajectory look
    complete when it is not.
    """
    turns: list[Turn] = []
    i = 0
    for e in store.read(run_id):
        role = _TEXT_EVENTS.get(e.event_type)
        if role is None or role not in roles:
            continue
        text = e.payload.get("text")
        reason = skip_fidelity = None
        if not isinstance(text, str) or not text.strip():
            text = None
            level = (e.privacy or {}).get("content_level")
            chars = e.payload.get("chars")
            if isinstance(chars, int) and chars > 0:
                # The adapter counted the characters and kept none of them:
                # the text existed and was refused at ingress.
                skip_fidelity = Fidelity.DROPPED_BY_POLICY.value
                reason = f"{chars} chars recorded, text not retained (content_level={level!r})"
            else:
                skip_fidelity = Fidelity.MISSING.value
                reason = f"no text in the record (content_level={level!r})"
        turns.append(
            Turn(
                index=i,
                ts=e.ts,
                event_id=e.event_id,
                turn_id=e.turn_id,
                role=role,
                text=text,
                skip_reason=reason,
                skip_fidelity=skip_fidelity,
            )
        )
        i += 1
    return turns


# ── the two transports ──────────────────────────────────────────────────────


def _post(url: str, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return json.loads(fh.read().decode("utf-8"))
    except urllib.error.HTTPError as e:  # the server's own message is the useful one
        detail = e.read().decode("utf-8", "replace")[:400]
        raise PlaceError(f"{url} returned {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise PlaceError(
            f"cannot reach {url} ({e.reason}). Start one with "
            f"`python -m nebulai.backend.interp.live_server`, or pass "
            f"--in-process to load the model into this process instead."
        ) from None


def place_texts_http(
    texts: list[str], space_id: str, *, live_url: str, timeout: float = 300.0
) -> tuple[list[list[float]], dict[str, Any]]:
    coords: list[list[float]] = []
    meta: dict[str, Any] = {}
    url = live_url.rstrip("/") + "/live/place"
    for start in range(0, len(texts), CHUNK):
        got = _post(
            url, {"space_id": space_id, "texts": texts[start : start + CHUNK]},
            timeout=timeout,
        )
        if "coords" not in got:
            raise PlaceError(f"{url} returned no coords: {got}")
        coords.extend(got["coords"])
        meta = got
    return coords, meta


def place_texts_in_process(
    texts: list[str], space_id: str, *, out_root: str, local_dir: str | None
) -> tuple[list[list[float]], dict[str, Any]]:
    """The same arithmetic, with the model loaded here.

    Imports `nebulai` from `seer`, which is the permitted direction — the rule
    is that `nebulai` never imports `seer`. It is still not the default,
    because a Seer process that has resident model weights is a Seer process
    that competes for RAM with the agent it is watching.
    """
    from nebulai.backend.interp import live_server as ls

    ls._place_root = out_root
    ls._place_local_dir = local_dir
    coords: list[list[float]] = []
    meta: dict[str, Any] = {}
    for start in range(0, len(texts), CHUNK):
        got = ls.live_place(space_id, texts[start : start + CHUNK])
        coords.extend(got["coords"])
        meta = got
    return coords, meta


# ── the verb ────────────────────────────────────────────────────────────────


def place_run(
    store: EventStore,
    run_id: str,
    space_id: str,
    *,
    live_url: str = DEFAULT_LIVE_URL,
    in_process: bool = False,
    out_root: str = "out",
    local_dir: str | None = None,
    roles: tuple[str, ...] = ("assistant",),
    timeout: float = 300.0,
) -> Placement:
    if store.get_run(run_id) is None:
        raise PlaceError(f"no run {run_id!r} in {store.root}")
    turns = collect_turns(store, run_id, roles=roles)
    if not turns:
        raise PlaceError(
            f"run {run_id} has no {'/'.join(roles)} message events to place"
        )

    placeable = [t for t in turns if t.text is not None]
    skipped = [
        {
            "index": t.index,
            "event_id": t.event_id,
            "role": t.role,
            "fidelity": t.skip_fidelity,
            "reason": t.skip_reason,
        }
        for t in turns
        if t.text is None
    ]
    if not placeable:
        # Honest empty result rather than an exception: the run exists, it was
        # captured, and the reason every turn is unplaceable is worth writing
        # down where the viewer can show it.
        return Placement(
            run_id=run_id,
            space_id=space_id,
            model="",
            revision="",
            layer=-1,
            source=SOURCE_PINNED_MODEL,
            transport="in_process" if in_process else "live_http",
            fidelity=Fidelity.MISSING.value,
            verdict="unknown",
            pc1_evr=None,
            pc1_evr_null_p95=None,
            points=[],
            skipped=skipped,
        )

    texts = [t.text or "" for t in placeable]
    if in_process:
        coords, meta = place_texts_in_process(
            texts, space_id, out_root=out_root, local_dir=local_dir
        )
    else:
        coords, meta = place_texts_http(texts, space_id, live_url=live_url, timeout=timeout)

    if len(coords) != len(texts):
        raise PlaceError(
            f"asked for {len(texts)} placements, got {len(coords)} — refusing to "
            f"line up coordinates with turns by guesswork"
        )

    points = [
        {
            "index": t.index,
            "seq": i,
            "ts": t.ts,
            "event_id": t.event_id,
            "turn_id": t.turn_id,
            "role": t.role,
            "chars": len(t.text or ""),
            "coords": [float(c[0]), float(c[1])],
        }
        for i, (t, c) in enumerate(zip(placeable, coords))
    ]
    return Placement(
        run_id=run_id,
        space_id=meta.get("space_id", space_id),
        model=meta.get("model", ""),
        revision=meta.get("revision", ""),
        layer=int(meta.get("layer", -1)),
        source=SOURCE_PINNED_MODEL,
        transport="in_process" if in_process else "live_http",
        fidelity=meta.get("fidelity", Fidelity.DETERMINISTIC.value),
        verdict=meta.get("verdict", "unknown"),
        pc1_evr=meta.get("pc1_evr"),
        pc1_evr_null_p95=meta.get("pc1_evr_null_p95"),
        points=points,
        skipped=skipped,
    )


def placement_path(store: EventStore, run_id: str) -> Path:
    return store.runs_dir / run_id / PLACEMENT_FILENAME


def write_placement(store: EventStore, placement: Placement) -> Path:
    path = placement_path(store, placement.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(placement.to_dict(), indent=2) + "\n")
    return path


def read_placement(store: EventStore, run_id: str) -> Placement | None:
    path = placement_path(store, run_id)
    if not path.exists():
        return None
    return Placement.from_dict(json.loads(path.read_text()))


__all__ = [
    "CHUNK",
    "DEFAULT_LIVE_URL",
    "PLACEMENT_FILENAME",
    "PlaceError",
    "Placement",
    "SOURCE_PINNED_MODEL",
    "SOURCE_TEXT_EMBEDDER",
    "Turn",
    "collect_turns",
    "place_run",
    "place_texts_http",
    "place_texts_in_process",
    "placement_path",
    "read_placement",
    "write_placement",
]
