"""Palisade `ctfish` runs → canonical events (Attractors D7, episode #9).

Source: Palisade Research's *Demonstrating specification gaming in reasoning
models* corpus. `scoring/runs.json` is a dict keyed by run root, each value
carrying `session_id`, `model`, `variant`, `task_md5`, `commands`, `final_step`
and `entries` — a list of markdown blocks holding the agent's own tagged
output:

    <THOUGHT> … </THOUGHT>   the model's reasoning
    <PLAN> … </PLAN>         a strategy rewrite
    <SUBGOAL> … </SUBGOAL>   the next concrete objective
    <ACTION> … </ACTION>     shell commands, one per line
    <REMEMBER> … </REMEMBER> a fact written into the agent's own memory
    <FORGET> … </FORGET>     a memory id dropped

Three things this corpus does NOT contain, and which are therefore reported as
`missing` rather than filled in:

* **Timestamps.** Not one entry has a clock. Events get a synthetic monotone
  `ts`, `payload["order_only"] = True`, and `fidelity.ts = missing`. Anything
  that draws a duration from these events is drawing an invention.
* **Command output.** `entries` is the model's side only. So `Effect` for a
  shell action is `unknown`, not `new_information` — except for the one case
  that *is* decidable without output and is marked `heuristic` when used: a
  command the run has already executed verbatim cannot tell it anything it has
  not already been told, which is `no_new_information` (SESSIONSEER-HANDOVER
  §6's rule, applied where the record supports it).
* **Token usage.** One explicit `usage: None` event per run.

`--labels scoring/labels.json` attaches Palisade's own published
classification (`stage1..stage4`, `ESCALATION`) to the run's `run.completed`
event. That label is the corpus authors' judgement, cited as theirs; nothing
here re-judges a transcript with a model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from ..contract import Action, Effect, Event, EventType, Fidelity, Outcome
from .corpus_base import (
    CorpusAdapter,
    CorpusError,
    CorpusRun,
    CorpusSource,
    classify_shell,
    field_fidelity,
)

CTFISH = CorpusSource(
    id="ctfish",
    name="Palisade Research — specification gaming in reasoning models (ctfish)",
    url="https://github.com/PalisadeResearch/ctfish",
    licence="see the repository (research release); runs.json is Git-LFS",
    ships_in_repo=False,
    note=(
        "a two-run excerpt ships under tests/fixtures/corpus/; the 48 MB "
        "runs.json is read from a path"
    ),
)

_TAG = re.compile(r"<(THOUGHT|PLAN|SUBGOAL|ACTION|REMEMBER|FORGET)>(.*?)</\1>", re.S)
_ENTRY_NO = re.compile(r"^###\s*Entry\s*(\d+)", re.M)


def parse_entry(text: str) -> list[tuple[str, str]]:
    """(tag, body) pairs in the order they appear in one entry.

    Untagged prose between blocks is deliberately dropped: every entry in this
    corpus is tagged, and silently promoting stray text to a message would put
    the scaffold's own framing into the agent's mouth.
    """
    return [(m.group(1), m.group(2).strip()) for m in _TAG.finditer(text or "")]


def entry_number(text: str) -> int | None:
    m = _ENTRY_NO.search(text or "")
    return int(m.group(1)) if m else None


def split_commands(body: str) -> list[str]:
    """`<ACTION>` bodies hold one command per line."""
    return [ln.strip() for ln in (body or "").splitlines() if ln.strip()]


def read_runs(path: Path | str) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        d = json.load(fh)
    if not isinstance(d, dict):
        raise CorpusError(f"{path}: expected a dict of runs, got {type(d).__name__}")
    return d


def read_labels(path: Path | str) -> dict[str, dict[str, Any]]:
    """Palisade's published classification, keyed by session id."""
    with Path(path).open("r", encoding="utf-8") as fh:
        d = json.load(fh)
    if not isinstance(d, dict):
        raise CorpusError(f"{path}: expected a dict of labels")
    return d


class CtfishCorpusAdapter(CorpusAdapter):
    """One ctfish run → one session's trajectory."""

    agent = "ctfish"
    adapter_name = "corpus_ctfish"
    corpus = CTFISH

    def __init__(self, *, run_id: str = "", session_id: str = "", **kw: Any) -> None:
        super().__init__(run_id=run_id or "ctfish", session_id=session_id or "ctfish", **kw)

    def read(
        self,
        runs_path: Path | str,
        *,
        labels_path: Path | str | None = None,
        roots: Iterable[str] | None = None,
        model: str | None = None,
        variant: str | None = None,
        max_runs: int | None = None,
    ) -> list[CorpusRun]:
        runs = read_runs(runs_path)
        labels = read_labels(labels_path) if labels_path else {}

        keys = list(roots) if roots is not None else sorted(runs)
        if model is not None:
            keys = [k for k in keys if runs.get(k, {}).get("model") == model]
        if variant is not None:
            keys = [k for k in keys if runs.get(k, {}).get("variant") == variant]
        if max_runs is not None:
            keys = keys[:max_runs]

        out: list[CorpusRun] = []
        for k in keys:
            rec = runs.get(k)
            if rec is None:
                continue
            out.append(self._one_run(k, rec, labels, runs_path))
        if not out:
            raise CorpusError(
                f"{runs_path}: no run matched (model={model!r}, variant={variant!r})"
            )
        return out

    def _one_run(
        self,
        root: str,
        rec: dict[str, Any],
        labels: dict[str, dict[str, Any]],
        path: Path | str,
    ) -> CorpusRun:
        sid = str(rec.get("session_id") or root)
        self.run_id = f"ctfish-{sid[:16]}"
        self.session_id = self.run_id
        self._tick = 0.0
        self.model = {"id": str(rec.get("model") or "unknown"), "provider": "unknown"}
        entries = rec.get("entries") or []
        label = labels.get(sid)

        events: list[Event] = [
            self.event(
                EventType.RUN_STARTED,
                fidelity=Fidelity.DETERMINISTIC,
                ts=self.next_tick(),
                payload=self.corpus_payload(
                    **self.ordered(
                        root=root,
                        native_session_id=sid,
                        variant=rec.get("variant"),
                        task_md5=rec.get("task_md5"),
                        batch_key=rec.get("batch_key"),
                        final_step=rec.get("final_step"),
                        n_entries=len(entries),
                    )
                ),
            )
        ]

        seen_commands: set[str] = set()
        n_actions = 0
        for i, entry in enumerate(entries):
            n = entry_number(entry)
            step = n if n is not None else i
            self.turn_id = f"{self.run_id}:t{step}"
            blocks = parse_entry(entry)
            if not blocks:
                events.extend(self.unmapped("ctfish entry with no tagged block"))
                continue
            events.append(
                self.event(
                    EventType.TURN_STARTED,
                    fidelity=Fidelity.DETERMINISTIC,
                    ts=self.next_tick(),
                    payload=self.corpus_payload(
                        **self.ordered(step=step, tags=[t for t, _ in blocks])
                    ),
                )
            )
            for tag, body in blocks:
                ev, used = self._block(tag, body, seen_commands)
                events.extend(ev)
                n_actions += used
            events.append(
                self.event(
                    EventType.TURN_COMPLETED,
                    fidelity=Fidelity.DETERMINISTIC,
                    ts=self.next_tick(),
                    payload=self.corpus_payload(**self.ordered(step=step)),
                )
            )

        self.turn_id = None
        events.append(self.session_completed())
        events.append(self.missing_usage())
        events.append(self._completed(rec, label))
        return CorpusRun(
            run_id=self.run_id,
            session_id=self.session_id,
            events=events,
            corpus=self.corpus,
            meta=self.run_meta(
                path,
                root=root,
                native_session_id=sid,
                model=rec.get("model"),
                variant=rec.get("variant"),
                n_entries=len(entries),
                n_commands=n_actions,
                labelled=label is not None,
            ),
            warnings=list(self.warnings),
        )

    def _block(
        self, tag: str, body: str, seen_commands: set[str]
    ) -> tuple[list[Event], int]:
        if tag == "THOUGHT":
            payload, fid = self.reasoning_payload(body)
            return (
                [
                    self.event(
                        EventType.MESSAGE_ASSISTANT_COMPLETED,
                        fidelity=fid,
                        ts=self.next_tick(),
                        payload=self.corpus_payload(
                            **self.ordered(
                                kind="thought",
                                reasoning=payload,
                                **field_fidelity(reasoning=fid),
                            )
                        ),
                        native_type="ctfish.THOUGHT",
                    )
                ],
                0,
            )
        if tag in ("PLAN", "SUBGOAL"):
            return (
                [
                    self.event(
                        EventType.PLAN_UPDATED,
                        fidelity=Fidelity.NATIVE,
                        ts=self.next_tick(),
                        payload=self.corpus_payload(
                            **self.ordered(kind=tag.lower(), text=body)
                        ),
                        native_type=f"ctfish.{tag}",
                    )
                ],
                0,
            )
        if tag in ("REMEMBER", "FORGET"):
            # The agent editing its own context. There is no canonical event for
            # that, and inventing one would put a corpus-specific member into a
            # vocabulary four other adapters share; it rides as an assistant
            # message with an explicit `kind` instead.
            return (
                [
                    self.event(
                        EventType.MESSAGE_ASSISTANT_COMPLETED,
                        fidelity=Fidelity.NATIVE,
                        ts=self.next_tick(),
                        payload=self.corpus_payload(
                            **self.ordered(kind=tag.lower(), text=body)
                        ),
                        native_type=f"ctfish.{tag}",
                    )
                ],
                0,
            )
        if tag == "ACTION":
            out: list[Event] = []
            cmds = split_commands(body)
            for cmd in cmds:
                action, effect = classify_shell(cmd)
                repeated = cmd in seen_commands
                seen_commands.add(cmd)
                if repeated and action in (Action.INSPECT, Action.SEARCH):
                    # decidable without the output: this run has already run
                    # this exact command, so it cannot surface anything the run
                    # has not already been shown. An interpretation, hence
                    # HEURISTIC on the envelope.
                    effect = Effect.NO_NEW_INFORMATION
                    fid = Fidelity.HEURISTIC
                else:
                    fid = Fidelity.DETERMINISTIC
                span = f"{self.turn_id}:{len(seen_commands)}"
                out.append(
                    self.event(
                        EventType.TOOL_STARTED,
                        fidelity=Fidelity.NATIVE,
                        ts=self.next_tick(),
                        span_id=span,
                        action=action,
                        payload=self.corpus_payload(
                            **self.ordered(tool="shell", command=cmd)
                        ),
                        native_type="ctfish.ACTION",
                    )
                )
                out.append(
                    self.event(
                        EventType.TOOL_COMPLETED,
                        fidelity=fid,
                        ts=self.next_tick(),
                        span_id=span,
                        action=action,
                        effect=effect,
                        payload=self.corpus_payload(
                            **self.ordered(
                                tool="shell",
                                command=cmd,
                                output=None,
                                exit_code=None,
                                duration_ms=None,
                                repeated=repeated,
                                note=(
                                    "this corpus records the agent's side only; "
                                    "the command's output is not in the record"
                                ),
                                **field_fidelity(
                                    output=Fidelity.MISSING,
                                    exit_code=Fidelity.MISSING,
                                    duration_ms=Fidelity.MISSING,
                                    effect=fid,
                                ),
                            )
                        ),
                        native_type="ctfish.ACTION",
                    )
                )
            return out, len(cmds)
        return self.unmapped(f"ctfish tag {tag!r}"), 0

    def _completed(
        self, rec: dict[str, Any], label: dict[str, Any] | None
    ) -> Event:
        return self.event(
            EventType.RUN_COMPLETED,
            fidelity=Fidelity.DETERMINISTIC,
            ts=self.next_tick(),
            payload=self.corpus_payload(
                **self.ordered(
                    # the corpus records how far the run got, never whether the
                    # task was verified; `unknown` is the only honest outcome
                    outcome=Outcome.UNKNOWN.value,
                    final_step=rec.get("final_step"),
                    label=label,
                    label_source=(
                        "PalisadeResearch/ctfish scoring/labels.json — the corpus "
                        "authors' published classification, not a judgement made "
                        "here"
                    )
                    if label
                    else None,
                    **field_fidelity(
                        outcome=Fidelity.MISSING,
                        label=Fidelity.NATIVE if label else Fidelity.MISSING,
                    ),
                )
            ),
        )


__all__ = [
    "CTFISH",
    "CtfishCorpusAdapter",
    "entry_number",
    "parse_entry",
    "read_labels",
    "read_runs",
    "split_commands",
]
