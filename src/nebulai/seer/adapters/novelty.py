"""Deciding `Effect.NO_NEW_INFORMATION` — only where the payload settles it.

`Effect.NO_NEW_INFORMATION` is the half of the taxonomy that makes loop
detection possible at all (contract.py, rule 3), and until this module it was
never emitted by anything. The reason it stayed empty is the reason this module
is small: for most actions "did that surface anything new?" is a judgement about
the agent's context window, which no adapter can see. Guessing it would put a
fabricated loop count in front of a researcher, which is worse than the
`missing` the loop rule reported instead.

So the ledger decides exactly three cases, and returns `None` — no effect label
at all — for everything else. Each case names the rule it used and the fidelity
that rule earns, and both travel in the event payload (`effect_rule`,
`effect_fidelity`) so a reader can disagree with the label without re-deriving
it.

**R-A `repeat_read` — `DETERMINISTIC`.** The same lookup issued a second time
with no edit in between that could have changed its answer. This falls straight
out of the payload: the adapter already records every path it reads and every
path it edits, and if nothing wrote to the file between two reads the second
read cannot have returned anything the first did not. It is the same reasoning
`analysis.loop_rules`'s `repeat_read_without_change` runs after the fact, moved
to ingress where the effect label belongs.

"What could have changed its answer" is narrow for a read and wide for a
search, and the asymmetry is deliberate. A `Read` of `a.py` is invalidated by an
edit to `a.py` and by nothing else, so the caller passes that path and only
edits to it count. A `Grep` ranges over a tree whose extent we do not know from
the payload, so *any* edit in the run retires it. Scoping a search to a path we
guessed at would produce the one error that matters here: labelling a search
that genuinely found something new as a repeat.

**R-B `identical_output` — `DETERMINISTIC`.** The same command run again with
byte-identical output. Not a proxy and not a guess: the two outputs are
compared, and the comparison is exact. The ledger keeps a SHA-256 of the output
and never the output itself, so deciding this costs the log nothing — it stays
inside the `metadata` privacy tier the same way `edit_extent`'s line counts do.

**R-C `zero_result_search` — `HEURISTIC`.** A search whose result body is empty
or begins with one of the fixed zero-result sentences agents print
(`ZERO_RESULT_SENTINELS`). This is a *stated rule over a proxy*: the body is
prose, we match it against a list we wrote, and a tool that words its empty
result differently will not match. It is labelled `HEURISTIC` for exactly that
reason, and it is the only rule here that is not an exact comparison. A search
that returns text we do not recognise gets no label rather than a hopeful one.

Everything else — "did this read teach the model something", "was that command
redundant" — is not decidable from a payload and is therefore not decided. The
ledger returns `None`, the adapter leaves the effect as it was, and the loop
rule keeps saying `missing` for that run. That is the intended outcome, not a
gap waiting to be filled with an estimate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from ..contract import Action, Effect, Fidelity

#: Actions whose repetition can be reasoned about from a target alone. An EDIT
#: that changes nothing is `NO_STATE_CHANGE`, a different label with a different
#: meaning, and is not this module's business.
_LOOKUP_ACTIONS = frozenset({Action.INSPECT, Action.SEARCH})

#: Prose an agent prints when a search matched nothing. Matched
#: case-insensitively against the *start* of the stripped result body, because
#: a tool that found something usually starts with the something.
#:
#: This list is the whole of rule R-C's evidence, which is why R-C is
#: `HEURISTIC`: it is our sentence list, not the agent's contract.
ZERO_RESULT_SENTINELS: tuple[str, ...] = (
    "no matches found",
    "no files found",
    "no results found",
    "no results",
    "found 0 ",
    "0 matches",
    "no content found",
)


#: Lookup tools whose identity is an argument other than a path, and which
#: arguments those are. A second `Grep` for the same pattern in the same place
#: is a repeat this ledger can see; a second `Grep` for a different pattern is
#: not, however similar the two look.
#:
#: These are Claude Code's tool names. Codex and Hermes name their tools
#: differently and get `None` from `lookup_target`, which means R-A simply does
#: not fire for them — the honest outcome for a vocabulary we have not mapped,
#: and better than keying on a name we guessed at.
LOOKUP_KEYS: dict[str, tuple[str, ...]] = {
    "Read": ("file_path", "offset", "limit"),
    "Grep": ("pattern", "path", "glob", "type", "output_mode"),
    "Glob": ("pattern", "path"),
    "WebFetch": ("url",),
    "WebSearch": ("query",),
    "NotebookRead": ("notebook_path",),
}


def lookup_target(name: str, inp: dict[str, Any]) -> str | None:
    """The ledger key for a lookup call, or `None` when there is no stable one.

    Built from the arguments that decide what comes back, so two calls share a
    key only when they would return the same thing. `Read` includes `offset`
    and `limit` for exactly that reason: re-reading a file at a different
    window is a different read, and keying on the path alone would label it a
    repeat that it is not.
    """
    keys = LOOKUP_KEYS.get(name)
    if not keys:
        return None
    parts = [str(inp.get(k)) for k in keys]
    if all(p == "None" for p in parts):
        return None
    return name + ":" + "\x1f".join(parts)


#: The file path a lookup call read, when the call names exactly one — so an
#: edit to that path, and only to that path, retires an earlier read of it.
#: `Grep`, `Glob` and the web tools are absent on purpose: they range over
#: something we cannot bound from the payload.
_LOOKUP_PATH_KEY = {"Read": "file_path", "NotebookRead": "notebook_path"}


def lookup_path(name: str, inp: dict[str, Any]) -> str | None:
    """The one file this lookup read, or `None` if it did not name exactly one."""
    key = _LOOKUP_PATH_KEY.get(name)
    v = inp.get(key) if key else None
    return str(v) if v else None


@dataclass(frozen=True, slots=True)
class Verdict:
    """One decided effect, with the rule and fidelity that produced it."""

    effect: Effect
    fidelity: Fidelity
    rule: str

    def payload(self) -> dict[str, Any]:
        """The two keys an adapter merges into the event payload.

        They are separate from `Source.fidelity` on purpose: a Claude
        `tool_result` event is `NATIVE` — the agent told us the call finished —
        while the *effect* we attached to it may be a heuristic of ours. One
        fidelity field cannot say both, and collapsing them would either
        downgrade a native event or launder a guess as native.
        """
        return {"effect_fidelity": self.fidelity.value, "effect_rule": self.rule}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def is_zero_result(body: str) -> bool:
    """R-C's predicate, exported so a test can pin the sentence list."""
    s = body.strip().lower()
    if not s:
        return True
    return any(s.startswith(p) for p in ZERO_RESULT_SENTINELS)


@dataclass
class NoveltyLedger:
    """Per-run memory of what has already been looked at and what it returned.

    One per adapter instance, which is one per captured session. Nothing here
    persists: the ledger is a fold over the run's own events, so a run replayed
    from its log decides the same labels again.

    Ordering is by the ledger's own step counter rather than by wall clock. The
    adapter feeds events in log order, which is the order that matters, and a
    reconciled or replayed run whose timestamps are all stamped from history
    would otherwise have no usable clock at all.
    """

    #: inspect/search target → the step at which we last saw it looked at
    _seen: dict[str, int] = field(default_factory=dict)
    #: path → steps at which it was edited, so a read after an edit is new again
    _edited: dict[str, list[int]] = field(default_factory=dict)
    #: every edit step, whatever the path — what retires a search, which we
    #: cannot scope to a path from the payload alone
    _edits: list[int] = field(default_factory=list)
    #: command → SHA-256 of its last output. The output itself is never kept.
    _output: dict[str, str] = field(default_factory=dict)
    _step: int = 0

    def _tick(self) -> int:
        self._step += 1
        return self._step

    # ── recording ────────────────────────────────────────────────────────

    def note_edit(self, path: str | None) -> None:
        """A file changed. Everything read before this is fair to read again."""
        step = self._tick()
        self._edits.append(step)
        if path:
            self._edited.setdefault(str(path), []).append(step)

    # ── deciding ─────────────────────────────────────────────────────────

    def decide(
        self,
        *,
        action: Action | None,
        target: str | None = None,
        path: str | None = None,
        command: str | None = None,
        output: str | None = None,
    ) -> Verdict | None:
        """`NO_NEW_INFORMATION` when one of R-A/R-B/R-C fires, else `None`.

        `None` means "not decidable here", never "there was new information".
        The caller leaves whatever effect it already had.

        `path` is the single file a lookup read, when the call names one. It is
        what R-A invalidates on; a lookup with no `path` is treated as ranging
        over the whole tree and is retired by any edit at all.
        """
        step = self._tick()
        verdict = None
        key = command or target

        # R-B first: it is the strongest evidence available, and it applies to
        # commands whose action is EXECUTE or VERIFY as well as to lookups.
        if key and isinstance(output, str):
            digest = _digest(output)
            if self._output.get(key) == digest:
                verdict = Verdict(
                    Effect.NO_NEW_INFORMATION, Fidelity.DETERMINISTIC, "identical_output"
                )
            self._output[key] = digest

        if verdict is None and action in _LOOKUP_ACTIONS and target:
            t = str(target)
            prev = self._seen.get(t)
            invalidating = self._edited.get(str(path), ()) if path else self._edits
            if prev is not None and not any(prev <= e <= step for e in invalidating):
                verdict = Verdict(
                    Effect.NO_NEW_INFORMATION, Fidelity.DETERMINISTIC, "repeat_read"
                )

        if (
            verdict is None
            and action is Action.SEARCH
            and isinstance(output, str)
            and is_zero_result(output)
        ):
            verdict = Verdict(
                Effect.NO_NEW_INFORMATION, Fidelity.HEURISTIC, "zero_result_search"
            )

        if action in _LOOKUP_ACTIONS and target:
            self._seen[str(target)] = step
        return verdict


__all__ = [
    "LOOKUP_KEYS",
    "NoveltyLedger",
    "Verdict",
    "ZERO_RESULT_SENTINELS",
    "is_zero_result",
    "lookup_path",
    "lookup_target",
]
