"""The question set is the instrument, and the instrument is a one-way door.

`docs/GENERATIVE-VARIANCE-PLAN.md` §6.5 names this the study's largest
irreversible decision, and the reason is arithmetic rather than procedural: a
trial vector IS the ordered question scores, so changing, adding, reordering,
or rewording a question after collection begins invalidates **every trial
already scored**. There is no migration. There is only re-scoring the whole
corpus at full cost.

That risk is not managed by writing it down — this document has already been
written down, and W1's cost envelope (§8) makes a silent instrument drift an
expensive discovery. So the mechanism here is built to make the failure
*impossible to reach quietly*:

  - a question set is `draft` until someone calls `freeze()`, and nothing may
    collect data against a draft (`require_frozen`);
  - freezing stamps a hash over exactly the fields that determine what a score
    means — id, text, and scale, in order — and nothing else, so editorial
    fields stay editable forever without invalidating data;
  - loading a frozen set **re-derives that hash and compares**, so an edit made
    after the freeze is caught at load time rather than at analysis time;
  - every artifact stamps the hash, and `require_compatible()` refuses to pool
    trials scored against different instruments.

What this module deliberately does NOT do is ship a question set. Choosing the
~30 architecture questions is the study's central scientific act, it is gated
on the human agreement work in §6.2-6.4, and a set invented by the tooling that
validates it would be the tooling grading its own homework. The template in
`docs/instruments/` shows the format and cannot be frozen: `freeze()` refuses
any id in the `example_` namespace.

Missing is not zero. §7 is explicit — "a question the scorer declined is not a
question answered 'no', and collapsing the two would manufacture agreement" —
so an absent answer becomes `NaN` in the vector and is counted in provenance.
Absence has ink.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Bumped only when the *file format* changes in a way older readers cannot
#: parse. It is not the instrument's identity — the hash is.
INSTRUMENT_FORMAT_VERSION = 1

#: Ids reserved for format documentation. `freeze` refuses them, so the shipped
#: template cannot become a real instrument by someone running one command in
#: the wrong directory.
EXAMPLE_ID_PREFIX = "example_"

_SCALE_KINDS = ("likert", "binary", "unit")


class InstrumentError(RuntimeError):
    """The question set is malformed, unfrozen, edited after freezing, or
    incompatible with data being pooled into it.

    Deliberately not a subclass of `ValueError`: callers catch `ValueError`
    around parsing all over this tree, and an instrument mismatch must not be
    swallowed by a handler written for a bad int.
    """


@dataclass(frozen=True)
class Question:
    """One scored question. `id` is the stable key; `text` is what the scorer
    is actually shown.

    Both are in the hash. Rewording `text` while keeping `id` produces a
    different instrument, because it is a different question — the id is a
    reference, not a promise that the meaning held still.
    """

    id: str
    text: str
    kind: str = "likert"  # one of _SCALE_KINDS
    lo: float = 1.0
    hi: float = 5.0
    #: free-text rationale, guidance, or provenance. NOT hashed — editorial
    #: notes must stay editable after a freeze or the freeze becomes a reason
    #: not to document anything.
    note: str = ""

    def validate(self) -> None:
        if not self.id or not self.id.strip():
            raise InstrumentError("a question has an empty id")
        if self.id != self.id.strip():
            raise InstrumentError(f"question id {self.id!r} has surrounding whitespace")
        if not self.text or not self.text.strip():
            raise InstrumentError(f"question {self.id!r} has empty text")
        if self.kind not in _SCALE_KINDS:
            raise InstrumentError(
                f"question {self.id!r} has unknown scale kind {self.kind!r} "
                f"(expected one of {', '.join(_SCALE_KINDS)})"
            )
        if not self.hi > self.lo:
            raise InstrumentError(
                f"question {self.id!r} has an empty range: lo={self.lo} hi={self.hi}"
            )

    def check_value(self, value: float) -> float:
        """Range-check one answer. Out of range RAISES rather than clipping.

        A scorer that answers 7 on a 1-5 scale has misunderstood the prompt, and
        clipping it to 5 would turn a detectable instrument failure into a
        plausible datum. §7's immutable-raw-evidence rule has the same shape:
        keep the anomaly visible.
        """
        v = float(value)
        if not (self.lo <= v <= self.hi):
            raise InstrumentError(
                f"question {self.id!r} answered {v} but its scale is "
                f"{self.lo}..{self.hi} — a value off the scale is a scorer "
                f"failure, and clipping it would hide one"
            )
        return v

    def _hashable(self) -> dict:
        """Exactly the fields that determine what a score MEANS."""
        return {
            "id": self.id,
            "text": self.text,
            "kind": self.kind,
            "lo": self.lo,
            "hi": self.hi,
        }


@dataclass
class QuestionSet:
    """An ordered instrument, draft or frozen.

    Order is part of the identity: the trial vector is these scores in this
    order, so a reordering that leaves every question intact still produces
    vectors that cannot be pooled with earlier ones.
    """

    name: str
    questions: list[Question]
    frozen_at: str | None = None  # ISO-8601 UTC, set by freeze()
    frozen_hash: str | None = None
    notes: str = ""  # not hashed
    format_version: int = INSTRUMENT_FORMAT_VERSION

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise InstrumentError("a question set needs a name")
        for q in self.questions:
            q.validate()
        seen: dict[str, int] = {}
        for i, q in enumerate(self.questions):
            if q.id in seen:
                raise InstrumentError(
                    f"duplicate question id {q.id!r} at positions {seen[q.id]} "
                    f"and {i} — ids are how a score finds its question"
                )
            seen[q.id] = i

    # --- identity ---------------------------------------------------------

    @property
    def is_frozen(self) -> bool:
        return self.frozen_hash is not None

    def compute_hash(self) -> str:
        """Re-derive the instrument hash from the current contents.

        Canonical JSON — sorted keys, no incidental whitespace — so the digest
        depends on the questions and not on how the file happened to be
        formatted. The name is included: two different studies' instruments
        should not collide just because they asked the same questions.
        """
        payload = json.dumps(
            {
                "name": self.name,
                "questions": [q._hashable() for q in self.questions],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def freeze(self, at: str) -> str:
        """Close the door. Returns the hash.

        `at` is passed in rather than read from the clock so a freeze is
        reproducible in tests and in a re-run of a build script.
        """
        if self.is_frozen:
            raise InstrumentError(
                f"{self.name} was already frozen at {self.frozen_at} "
                f"({self.frozen_hash}). Re-freezing would silently redefine the "
                f"instrument every trial so far was scored against.\n"
                f"  to change the questions, create a NEW question set with a "
                f"new name; the old trials remain valid evidence about the old "
                f"instrument."
            )
        if not self.questions:
            raise InstrumentError(
                f"{self.name} has no questions — an empty instrument would "
                f"produce zero-width trial vectors and a study with nothing in it"
            )
        example = [q.id for q in self.questions if q.id.startswith(EXAMPLE_ID_PREFIX)]
        if example:
            raise InstrumentError(
                f"{self.name} still contains format-documentation questions "
                f"({', '.join(example)}). The shipped template is a format "
                f"example, not a validated instrument — choosing the real "
                f"questions is the study's central decision and is gated on the "
                f"human-agreement work in GENERATIVE-VARIANCE-PLAN.md §6.2-6.4."
            )
        self.frozen_at = at
        self.frozen_hash = self.compute_hash()
        return self.frozen_hash

    def require_frozen(self) -> str:
        """Gate for anything that spends money or records data. Returns the hash."""
        if not self.is_frozen:
            raise InstrumentError(
                f"{self.name} is a draft, so nothing may be scored against it "
                f"yet. Scores are only comparable within a fixed instrument "
                f"(GENERATIVE-VARIANCE-PLAN.md §6.5); collecting first and "
                f"freezing later is how a corpus becomes unpoolable.\n"
                f"  freeze it deliberately once the questions are settled."
            )
        return self.frozen_hash  # type: ignore[return-value]

    def verify_integrity(self) -> None:
        """Catch a post-freeze edit. Called on every load of a frozen set."""
        if not self.is_frozen:
            return
        actual = self.compute_hash()
        if actual != self.frozen_hash:
            raise InstrumentError(
                f"{self.name} was frozen as {self.frozen_hash} but its questions "
                f"now hash to {actual} — the instrument was edited after it was "
                f"frozen.\n"
                f"  every trial scored against the recorded hash is evidence "
                f"about the OLD questions. Restore the file from version "
                f"control, or start a new named instrument and re-score.\n"
                f"  (editorial fields — notes, per-question note — are not "
                f"hashed, so this is not what changed.)"
            )

    # --- serialization ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "name": self.name,
            "status": "frozen" if self.is_frozen else "draft",
            "frozen_at": self.frozen_at,
            "hash": self.frozen_hash,
            "notes": self.notes,
            "questions": [
                {
                    "id": q.id,
                    "text": q.text,
                    "kind": q.kind,
                    "lo": q.lo,
                    "hi": q.hi,
                    "note": q.note,
                }
                for q in self.questions
            ],
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n")
        return p


def question_set_from_dict(d: dict) -> QuestionSet:
    fv = int(d.get("format_version") or 1)
    if fv > INSTRUMENT_FORMAT_VERSION:
        raise InstrumentError(
            f"question set format v{fv} is newer than this build understands "
            f"(v{INSTRUMENT_FORMAT_VERSION}) — upgrade rather than guess"
        )
    qs = QuestionSet(
        name=str(d.get("name") or ""),
        questions=[
            Question(
                id=str(q.get("id") or ""),
                text=str(q.get("text") or ""),
                kind=str(q.get("kind") or "likert"),
                lo=float(q.get("lo", 1.0)),
                hi=float(q.get("hi", 5.0)),
                note=str(q.get("note") or ""),
            )
            for q in (d.get("questions") or [])
        ],
        frozen_at=d.get("frozen_at") or None,
        frozen_hash=d.get("hash") or None,
        notes=str(d.get("notes") or ""),
        format_version=fv,
    )
    qs.verify_integrity()
    return qs


def load_question_set(path: str | Path) -> QuestionSet:
    """Read and validate. A frozen set is integrity-checked here, at load."""
    p = Path(path)
    try:
        d = json.loads(p.read_text())
    except FileNotFoundError:
        raise InstrumentError(f"no question set at {p}") from None
    except json.JSONDecodeError as e:
        raise InstrumentError(f"{p} is not valid JSON: {e}") from None
    return question_set_from_dict(d)


# --- trials ---------------------------------------------------------------


@dataclass
class Trial:
    """One scored story. `answers` maps question id -> value; anything absent
    or explicitly None is missing."""

    prompt_id: str
    trial_index: int
    answers: dict[str, float | None] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """`prompt/trial`, the label shape §5's reuse map specifies."""
        return f"{self.prompt_id}/{self.trial_index}"


def trial_vector(qs: QuestionSet, answers: dict[str, float | None]) -> np.ndarray:
    """One trial's scores in frozen question order. Missing is `NaN`, never 0.

    Raises on an answer whose question is not in the instrument — a scorer
    inventing a question is the same instrument drift the freeze exists to
    catch, arriving from the response side instead of the file side.
    """
    known = {q.id: q for q in qs.questions}
    unknown = sorted(set(answers) - set(known))
    if unknown:
        raise InstrumentError(
            f"answers name {len(unknown)} question(s) not in {qs.name}: "
            f"{', '.join(unknown[:5])}"
            + (" ..." if len(unknown) > 5 else "")
            + " — the scorer answered a different instrument than the one frozen"
        )
    row = np.full(len(qs.questions), np.nan, dtype=np.float32)
    for i, q in enumerate(qs.questions):
        v = answers.get(q.id)
        if v is None:
            continue
        row[i] = q.check_value(v)
    return row


def stack_trials(qs: QuestionSet, trials: list[Trial]) -> tuple[np.ndarray, list[str]]:
    """(n_trials, n_questions) float32 and the matching `prompt/trial` labels."""
    if not trials:
        raise InstrumentError("no trials to stack")
    rows = np.vstack([trial_vector(qs, t.answers) for t in trials])
    return rows.astype(np.float32), [t.label for t in trials]


def missingness(rows: np.ndarray) -> dict:
    """Per-question and overall missing counts, for provenance and display.

    Reported always, including when it is zero — §7's "absence has ink" rule
    applies to the absence of absence too. A study that silently omits this
    field is indistinguishable from one that had nothing to report.
    """
    if rows.ndim != 2:
        raise InstrumentError(f"expected a 2-D trial matrix, got shape {rows.shape}")
    per_q = np.isnan(rows).sum(axis=0)
    return {
        "n_trials": int(rows.shape[0]),
        "n_questions": int(rows.shape[1]),
        "missing_total": int(per_q.sum()),
        "missing_per_question": [int(x) for x in per_q],
        "trials_complete": int((~np.isnan(rows).any(axis=1)).sum()),
    }


def provenance(qs: QuestionSet) -> dict:
    """What every artifact stamps so a later reader can tell which instrument
    produced a number. Requires a frozen set by construction."""
    return {
        "instrument": qs.name,
        "instrument_hash": qs.require_frozen(),
        "instrument_frozen_at": qs.frozen_at,
        "n_questions": len(qs.questions),
        "question_ids": [q.id for q in qs.questions],
    }


def require_compatible(qs: QuestionSet, meta: dict[str, Any]) -> None:
    """Refuse to pool trials scored against a different instrument.

    The check is the hash, not the name and not the question count: two sets can
    share both and still ask different things. Pooling across a reworded
    question is the §6.5 failure in its most plausible costume — nothing looks
    wrong, and the variance estimate is quietly wrong.
    """
    want = qs.require_frozen()
    got = meta.get("instrument_hash")
    if got is None:
        raise InstrumentError(
            f"these trials carry no instrument hash, so there is no way to tell "
            f"whether they were scored against {qs.name}. Untagged trials cannot "
            f"be pooled — they are evidence about an unknown instrument."
        )
    if got != want:
        raise InstrumentError(
            f"instrument mismatch: these trials were scored against {got} "
            f"(instrument {meta.get('instrument')!r}) but the current set "
            f"{qs.name} is {want}.\n"
            f"  scores are only comparable within one fixed instrument "
            f"(GENERATIVE-VARIANCE-PLAN.md §6.5). Re-score, or analyse the two "
            f"corpora separately."
        )
