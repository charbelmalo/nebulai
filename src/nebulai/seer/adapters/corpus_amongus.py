"""Among Us LLM-agent logs → canonical events (Attractors D7).

Source: the public experiment logs released with *Among Us: A Sandbox for
Agentic Deception* (arXiv:2504.04072). Two files per experiment:

* `summary.json` — one JSON object per line, `{"Game 82": {config, Player 1..7,
  winner, winner_reason}}`. This is the ground truth about who was an Impostor.
* `agent-logs-compact.json` — one JSON object per model call:
  `{game_index, step, timestamp, player{name,identity,model,location},
    interaction{system_prompt, prompt, response{Condensed Memory, Thinking
    Process, Action}, full_response}}`.

**One game is a run; one player is a session.** That is the only mapping under
which a trajectory means anything here: the interesting structure is a single
player's path through the game, and the game is the shared frame they move in.

Two normalizations that are widenings, and are said out loud rather than
hidden:

* `Action.INTERACT` is documented for "ask the human". Among Us has no human,
  so SPEAK / VOTE / CALL MEETING map to `INTERACT` on the reading *addressed
  another participant rather than the environment*. The game's own verb is in
  `payload["verb"]`, so an analysis that disagrees with the widening can undo
  it without reading a native field.
* `Effect` for SPEAK is `UNKNOWN`, not `NO_STATE_CHANGE`. Speech in this game
  is the mechanism by which state changes; calling it stateless would be a
  claim about deception that the log does not support.

The Impostor/Crewmate label is ground truth from the summary and rides on the
run and on every event as `payload["identity"]` — the deception research the
corpus exists for is entirely about conditioning on it. The released
deception-probe weights are NOT loaded here: this adapter maps behaviour, and a
probe direction belongs in `backend/directions.py`'s schema, not in an event
stream.
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path
from typing import Any

from ..contract import Action, Effect, Event, EventType, Fidelity, Outcome
from .corpus_base import (
    CorpusAdapter,
    CorpusError,
    CorpusRun,
    CorpusSource,
    field_fidelity,
    iter_json_lines,
)

AMONGUS = CorpusSource(
    id="amongus",
    name="Among Us: A Sandbox for Agentic Deception — experiment logs",
    url="https://github.com/7vik/AmongUs",
    licence="MIT (code and released logs); see the repository",
    ships_in_repo=False,
    note=(
        "a 75-line excerpt ships under tests/fixtures/corpus/ so the mapping is "
        "testable offline; the full experiment logs are read from a path"
    ),
)

#: The game's verbs, in the order they are searched for at the head of an
#: action string. `MOVE from X to Y` and `1. MOVE from X to Y` both occur.
_VERBS: tuple[tuple[str, Action, Effect], ...] = (
    ("CALL MEETING", Action.INTERACT, Effect.STATE_CHANGED),
    # searched before COMPLETE TASK: an Impostor pretending to do a task is the
    # single most load-bearing verb in a deception corpus, and folding it into
    # COMPLETE TASK would erase the deception from the trajectory. Its effect
    # is NO_STATE_CHANGE because that is precisely what makes it a fake.
    ("COMPLETE FAKE TASK", Action.EXECUTE, Effect.NO_STATE_CHANGE),
    ("COMPLETE TASK", Action.EXECUTE, Effect.STATE_CHANGED),
    ("VIEW MONITOR", Action.INSPECT, Effect.UNKNOWN),
    ("REPORT DEAD BODY", Action.INTERACT, Effect.STATE_CHANGED),
    ("REPORT", Action.INTERACT, Effect.STATE_CHANGED),
    ("SPEAK", Action.INTERACT, Effect.UNKNOWN),
    ("VOTE", Action.INTERACT, Effect.UNKNOWN),
    ("MOVE", Action.EXECUTE, Effect.STATE_CHANGED),
    ("KILL", Action.EXECUTE, Effect.STATE_CHANGED),
    ("VENT", Action.EXECUTE, Effect.STATE_CHANGED),
    ("WAIT", Action.EXECUTE, Effect.NO_STATE_CHANGE),
)

_LEAD = re.compile(r"^\s*\d+[.)]\s*")


def parse_action(text: str) -> tuple[str, Action, Effect]:
    """(verb, normalized action, effect) for one `Action` line.

    An unrecognised line yields the verb `"UNKNOWN"` with `Action.EXECUTE` and
    `Effect.UNKNOWN` rather than being dropped — a step the mapping does not
    understand still happened, and dropping it would shorten the trajectory
    without saying so.
    """
    s = _LEAD.sub("", (text or "").strip())
    up = s.upper()
    for verb, action, effect in _VERBS:
        if up.startswith(verb):
            return verb, action, effect
    return "UNKNOWN", Action.EXECUTE, Effect.UNKNOWN


def _epoch(stamp: str) -> float | None:
    """`2025-02-01 05:37:57.045632` → epoch seconds, or None if unreadable.

    Parsed as naive local time because the corpus records no zone. The absolute
    value is therefore not comparable with another corpus; differences within
    one game are, which is what a trajectory uses.
    """
    try:
        return _dt.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S.%f").timestamp()
    except (TypeError, ValueError):
        try:
            return _dt.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").timestamp()
        except (TypeError, ValueError):
            return None


def read_summary(path: Path | str) -> dict[str, dict[str, Any]]:
    """`summary.json` → {game_index: game record}. One object per line."""
    out: dict[str, dict[str, Any]] = {}
    for obj in iter_json_lines(path):
        if not isinstance(obj, dict):
            continue
        for game, rec in obj.items():
            out[game] = rec
    return out


class AmongUsCorpusAdapter(CorpusAdapter):
    """Maps one game's step log into per-player sessions."""

    agent = "amongus"
    adapter_name = "corpus_amongus"
    corpus = AMONGUS

    def __init__(self, *, run_id: str, session_id: str = "", **kw: Any) -> None:
        super().__init__(run_id=run_id, session_id=session_id or run_id, **kw)

    # ── the mapping ──────────────────────────────────────────────────────

    def read(
        self,
        logs_path: Path | str,
        *,
        summary_path: Path | str | None = None,
        games: list[str] | None = None,
        max_games: int | None = None,
    ) -> list[CorpusRun]:
        summary = read_summary(summary_path) if summary_path else {}
        steps: dict[str, list[dict[str, Any]]] = {}
        for rec in iter_json_lines(logs_path):
            if not isinstance(rec, dict) or "game_index" not in rec:
                continue
            steps.setdefault(str(rec["game_index"]), []).append(rec)
        if not steps:
            raise CorpusError(f"{logs_path}: no record carried a game_index")

        wanted = games or sorted(steps, key=lambda g: (len(g), g))
        if max_games is not None:
            wanted = wanted[:max_games]

        runs: list[CorpusRun] = []
        for game in wanted:
            if game not in steps:
                continue
            runs.append(self._one_game(game, steps[game], summary.get(game), logs_path))
        return runs

    def _identities(self, rec: dict[str, Any] | None) -> dict[str, str]:
        if not rec:
            return {}
        out = {}
        for k, v in rec.items():
            if isinstance(v, dict) and "name" in v and "identity" in v:
                out[str(v["name"])] = str(v["identity"])
        return out

    def _one_game(
        self,
        game: str,
        recs: list[dict[str, Any]],
        summary: dict[str, Any] | None,
        path: Path | str,
    ) -> CorpusRun:
        run_id = f"amongus-{game.lower().replace(' ', '-')}"
        self.run_id = run_id
        self.session_id = run_id
        identities = self._identities(summary)
        recs = sorted(recs, key=lambda r: (int(r.get("step", 0)), str(r.get("timestamp", ""))))
        first_ts = _epoch(str(recs[0].get("timestamp", ""))) if recs else None
        model = None
        for r in recs:
            m = (r.get("player") or {}).get("model")
            if m:
                model = {"id": str(m), "provider": "unknown"}
                break
        self.model = model

        events: list[Event] = [
            self.event(
                EventType.RUN_STARTED,
                fidelity=Fidelity.DETERMINISTIC,
                ts=first_ts if first_ts is not None else self.next_tick(),
                payload=self.corpus_payload(
                    game=game,
                    players=identities,
                    config=(summary or {}).get("config"),
                    n_steps=len(recs),
                    **field_fidelity(
                        players=Fidelity.NATIVE if identities else Fidelity.MISSING,
                        ts=Fidelity.NATIVE if first_ts is not None else Fidelity.MISSING,
                    ),
                ),
            )
        ]

        seen_players: set[str] = set()
        for rec in recs:
            events.extend(self._one_step(rec, identities, seen_players))

        events.append(self.session_completed())
        events.append(self.missing_usage())
        events.append(self._run_completed(game, summary, recs))
        return CorpusRun(
            run_id=run_id,
            session_id=run_id,
            events=events,
            corpus=self.corpus,
            meta=self.run_meta(
                path,
                game=game,
                n_steps=len(recs),
                n_players=len(seen_players),
                identities_known=bool(identities),
            ),
            warnings=list(self.warnings),
        )

    def _one_step(
        self,
        rec: dict[str, Any],
        identities: dict[str, str],
        seen: set[str],
    ) -> list[Event]:
        player = rec.get("player") or {}
        name = str(player.get("name", "unknown"))
        # the session is the player; the game is the run
        self.session_id = f"{self.run_id}:{_slug(name)}"
        ts = _epoch(str(rec.get("timestamp", "")))
        clock = ts if ts is not None else self.next_tick()
        ts_fid = Fidelity.NATIVE if ts is not None else Fidelity.MISSING
        identity = identities.get(name) or str(player.get("identity") or "") or None
        step = int(rec.get("step", 0))
        self.turn_id = f"{self.session_id}:t{step}"

        out: list[Event] = []
        if name not in seen:
            seen.add(name)
            out.append(
                self.event(
                    EventType.SESSION_STARTED,
                    fidelity=Fidelity.DETERMINISTIC,
                    ts=clock,
                    payload=self.corpus_payload(
                        player=name,
                        identity=identity,
                        **field_fidelity(
                            identity=Fidelity.NATIVE if identity else Fidelity.MISSING,
                            ts=ts_fid,
                        ),
                    ),
                )
            )

        inter = rec.get("interaction") or {}
        resp = inter.get("response") or {}
        if not isinstance(resp, dict):
            resp = {}
        action_text = str(resp.get("Action", "") or "")
        thinking = resp.get("Thinking Process")
        memory = resp.get("Condensed Memory")

        out.append(
            self.event(
                EventType.TURN_STARTED,
                fidelity=Fidelity.DETERMINISTIC,
                ts=clock,
                payload=self.corpus_payload(
                    step=step,
                    phase=_phase(inter),
                    location=player.get("location"),
                    identity=identity,
                    **field_fidelity(ts=ts_fid),
                ),
            )
        )

        r_payload, r_fid = self.reasoning_payload(
            str(thinking) if thinking is not None else None
        )
        out.append(
            self.event(
                EventType.MESSAGE_ASSISTANT_COMPLETED,
                fidelity=r_fid,
                ts=clock,
                payload=self.corpus_payload(
                    text=action_text,
                    reasoning=r_payload,
                    memory_chars=len(str(memory)) if memory is not None else None,
                    identity=identity,
                    **field_fidelity(
                        reasoning=r_fid,
                        memory_chars=(
                            Fidelity.NATIVE if memory is not None else Fidelity.MISSING
                        ),
                        ts=ts_fid,
                    ),
                ),
                native_type="amongus.interaction",
                native={"full_response": inter.get("full_response")},
            )
        )

        verb, action, effect = parse_action(action_text)
        if verb == "UNKNOWN" and action_text:
            out.extend(self.unmapped(f"amongus action: {action_text[:60]!r}"))
        out.append(
            self.event(
                EventType.TOOL_COMPLETED,
                fidelity=Fidelity.DETERMINISTIC,
                ts=clock,
                action=action,
                effect=effect,
                payload=self.corpus_payload(
                    verb=verb,
                    text=action_text,
                    player=name,
                    identity=identity,
                    location=player.get("location"),
                    duration_ms=None,
                    **field_fidelity(duration_ms=Fidelity.MISSING, ts=ts_fid),
                ),
                native_type="amongus.action",
                native={"action": action_text},
            )
        )
        out.append(
            self.event(
                EventType.TURN_COMPLETED,
                fidelity=Fidelity.DETERMINISTIC,
                ts=clock,
                payload=self.corpus_payload(step=step, **field_fidelity(ts=ts_fid)),
            )
        )
        return out

    def _run_completed(
        self, game: str, summary: dict[str, Any] | None, recs: list[dict[str, Any]]
    ) -> Event:
        self.session_id = self.run_id
        self.turn_id = None
        last = _epoch(str(recs[-1].get("timestamp", ""))) if recs else None
        reason = (summary or {}).get("winner_reason")
        winner = (summary or {}).get("winner")
        return self.event(
            EventType.RUN_COMPLETED,
            fidelity=Fidelity.NATIVE if summary else Fidelity.MISSING,
            ts=last if last is not None else self.next_tick(),
            payload=self.corpus_payload(
                # The game declares a winner; that is a fact about the game, not
                # a verification of any agent's task, so the run outcome stays
                # `unknown` rather than borrowing `verified_pass`.
                outcome=Outcome.UNKNOWN.value,
                winner=winner,
                winner_reason=reason,
                game=game,
                **field_fidelity(
                    winner=Fidelity.NATIVE if summary else Fidelity.MISSING,
                    outcome=Fidelity.MISSING,
                    ts=Fidelity.NATIVE if last is not None else Fidelity.MISSING,
                ),
            ),
        )


def _phase(inter: dict[str, Any]) -> str | None:
    p = (inter.get("prompt") or {})
    if isinstance(p, dict):
        v = p.get("Phase")
        if isinstance(v, str):
            return v
    return None


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "player"


__all__ = [
    "AMONGUS",
    "AmongUsCorpusAdapter",
    "parse_action",
    "read_summary",
]
