"""Deterministic parsing and normalization of a raw completion (plan §6.2).

Two jobs, kept apart on purpose:

* :func:`parse_associates` turns one raw string into an ordered list of surface
  forms. It is lenient about *delimiters* and strict about *content*: it never
  invents, reorders, or drops a response to make a trial look compliant.
* :func:`normalize_form` applies the four-step lexical normalization and
  nothing else. Slang, emoji, multiword answers and out-of-vocabulary terms
  survive it unchanged — they are part of what the study measures (§3.2).

Everything here is pure and total: no network, no model, no randomness, so the
goldens in `tests/test_behavior_normalize.py` fully pin the behaviour. The
detectors in :func:`detect` are *gate inputs* (§6.5.2), not decoration; a
degenerate model that echoes the cue every time has perfect split-half
reliability and zero semantic content, and only these marks catch it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

#: Bumped whenever the parse/normalize behaviour changes in a way that would
#: alter an existing trial's derived row. Raw evidence is append-only (§6.1),
#: so a bump means *re-derive*, never *overwrite*.
PARSER_VERSION = 1

#: Leading list punctuation a model may emit before an answer: "1.", "-", "•",
#: "a)", "*". Stripped in step 2 of §6.2, before case-folding.
_LIST_PREFIX = re.compile(r"^\s*(?:[-*•–—>]+|\(?\d{1,2}[.)]|\(?[a-z][.)])\s*")

#: Trailing punctuation that is never part of an association's surface form.
#: Note `?` and `!` ARE stripped, but an interior `!` (as in "yes!!") is not:
#: only the trailing run goes.
_TRAIL_PUNCT = re.compile(r"[\s.,;:!?…\"'“”‘’)\]]+$")
_LEAD_PUNCT = re.compile(r"^[\s\"'“”‘’(\[]+")

#: The answer marker used by the canonical Lane A frame (`protocol.py`). A
#: completion model often restates it; everything before the LAST one on the
#: first answer line is prompt echo, not an answer.
ANSWER_MARKER = "->"

#: Delimiters accepted between associates, in tie-breaking order. A model that
#: uses several ("butter, toast; flour") is parsed on whichever yields the MOST
#: fields — taking the first that yields two would stop at "butter" +
#: "toast; flour" and record a compliant answer as `too_few:2`, which is a
#: parser artifact masquerading as a compliance figure. Ties go to the order
#: below, so the choice stays deterministic. The mixed delimiter is reported
#: either way so the trial can be audited.
_DELIMS: tuple[tuple[str, str], ...] = (
    ("comma", r"\s*,\s*"),
    ("semicolon", r"\s*;\s*"),
    ("slash", r"\s*/\s*"),
    ("pipe", r"\s*\|\s*"),
    ("newline", r"\s*\n\s*"),
)

#: Phrases that indicate the model declined rather than associated. Matched on
#: the normalized full output, anchored at the start, so "no, yes, maybe" (a
#: legitimate if odd association triple) is not swallowed.
_REFUSAL_PREFIXES: tuple[str, ...] = (
    "i can't",
    "i cannot",
    "i won't",
    "i will not",
    "i'm not able",
    "i am not able",
    "i'm sorry",
    "i am sorry",
    "sorry, i",
    "as an ai",
    "i don't feel comfortable",
    "i do not feel comfortable",
    "this request",
)


@dataclass
class ParseResult:
    """What one raw completion yielded, with the reasons attached.

    `associates` holds at most three normalized forms (§5.2 keeps the first
    three). `surface` holds their pre-normalization spellings at the same
    indices, because §6.2 requires the exact surface form to stay visible
    everywhere an embedding-derived claim is made.
    """

    associates: list[str] = field(default_factory=list)
    surface: list[str] = field(default_factory=list)
    valid: bool = False
    reason: str = ""
    delimiter: str = ""
    mixed_delimiters: bool = False
    n_parsed: int = 0  # fields found before truncation to three
    refusal: bool = False
    parser_version: int = PARSER_VERSION


def normalize_form(text: str) -> str:
    """The four-step normalization of §6.2, and nothing more.

    NFKC → strip surrounding whitespace and list punctuation → case-fold.
    Internal whitespace is collapsed to single spaces so that "ice   cream" and
    "ice cream" are one type; that is a whitespace fix, not a lexical one, and
    it does not touch the multiword-preservation requirement.
    """
    s = unicodedata.normalize("NFKC", text)
    s = _LIST_PREFIX.sub("", s)
    s = _LEAD_PUNCT.sub("", s)
    s = _TRAIL_PUNCT.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.casefold()


def _strip_echo(raw: str) -> str:
    """Isolate this trial's answer from a completion model's surrounding noise.

    Two different kinds of noise, handled in order:

    1. **A restated marker.** A base LM continuing "cue -> " frequently re-emits
       "cue -> a, b, c", so if the first non-empty line carries the marker,
       everything up to and including its LAST occurrence on that line is echo.
    2. **A next question.** Having answered, the same model often invents the
       next prompt ("window -> glass, pane, sill"). A marker appearing on any
       LATER line means the model has moved on; everything from that line is a
       different trial's answer and is cut.

    What survives may still be several lines, and that is deliberate — a model
    answering with a newline-separated or numbered list is answering, and
    keeping only the first line would score it `too_few:1`.
    """
    lines = raw.split("\n")
    first = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if first is None:
        return ""
    if ANSWER_MARKER in lines[first]:
        lines[first] = lines[first].rsplit(ANSWER_MARKER, 1)[1]
    kept = [lines[first]]
    for ln in lines[first + 1 :]:
        if ANSWER_MARKER in ln:
            break
        kept.append(ln)
    return "\n".join(kept).strip("\n")


def parse_associates(raw: str | None, *, max_associates: int = 3) -> ParseResult:
    """Parse one raw completion into ordered associates.

    Returns `valid=False` with an explicit `reason` rather than raising, because
    an unparseable output is *data* — its rate is the compliance figure of
    §6.5.1 and a gate input. Dropping it silently would bias the surviving
    sample exactly the way that section warns about.
    """
    if raw is None:
        return ParseResult(valid=False, reason="no_output")
    if not raw.strip():
        return ParseResult(valid=False, reason="empty")

    folded_full = normalize_form(raw)
    if any(folded_full.startswith(p) for p in _REFUSAL_PREFIXES):
        return ParseResult(valid=False, reason="refusal", refusal=True)

    line = _strip_echo(raw)
    if not line.strip():
        return ParseResult(valid=False, reason="empty_after_marker")

    # Which delimiter actually structures this answer? The one that finds the
    # most fields, with ties broken by _DELIMS order.
    chosen, fields_ = "", [line]
    hits = 0
    best = 1
    hitting: list[tuple[str, str]] = []
    for name, pattern in _DELIMS:
        parts = [p for p in re.split(pattern, line) if p.strip()]
        if len(parts) >= 2:
            hits += 1
            hitting.append((name, pattern))
            if len(parts) > best:
                best, chosen, fields_ = len(parts), name, parts
    if hits > 1:
        # "butter, toast; flour" splits into two fields on EITHER delimiter
        # alone, and reporting that as `too_few:2` would charge the model for
        # the parser's choice. When several delimiters are in play the fields
        # are whatever any of them separates; `mixed_delimiters` still marks
        # the trial so the inconsistency itself stays auditable.
        union = "|".join(p for _, p in hitting)
        parts = [p for p in re.split(union, line) if p.strip()]
        if len(parts) > best:
            best, chosen, fields_ = len(parts), "mixed", parts
    if not chosen:
        # A single field is a legitimate (under-compliant) answer: one associate.
        chosen, fields_ = "none", [line]

    surface: list[str] = []
    normed: list[str] = []
    for f in fields_:
        n = normalize_form(f)
        if not n:
            continue
        surface.append(unicodedata.normalize("NFKC", f).strip())
        normed.append(n)

    n_parsed = len(normed)
    surface, normed = surface[:max_associates], normed[:max_associates]

    res = ParseResult(
        associates=normed,
        surface=surface,
        delimiter=chosen,
        mixed_delimiters=hits > 1,
        n_parsed=n_parsed,
    )
    if len(normed) < max_associates:
        res.valid = False
        res.reason = f"too_few:{len(normed)}"
    else:
        res.valid = True
    return res


@dataclass
class Detectors:
    """Per-trial degeneracy marks (§6.5.2). Every field is a gate input.

    `exemplar_echo` needs the frame's own example answers, which is why the
    caller passes them: the detector cannot know them from the output alone, and
    guessing would make the most degenerate cues the easiest to pass.
    """

    cue_echo: bool = False
    exemplar_echo: bool = False
    within_trial_duplicate: bool = False
    prompt_copy: bool = False
    distinct_types: int = 0

    @property
    def degenerate(self) -> bool:
        return (
            self.cue_echo
            or self.exemplar_echo
            or self.within_trial_duplicate
            or self.prompt_copy
        )


def detect(
    parsed: ParseResult,
    *,
    cue: str,
    prompt: str = "",
    exemplar_answers: tuple[str, ...] = (),
) -> Detectors:
    """Mark, never replace (§6.2 step 5).

    The marks ride alongside the trial; the trial's raw output and its parsed
    associates are unchanged. `analyze.py` aggregates the rates per model per
    cue, where a large asymmetry is itself the capability confound made visible.
    """
    assoc = parsed.associates
    cue_n = normalize_form(cue)
    ex = {normalize_form(a) for a in exemplar_answers}

    d = Detectors(distinct_types=len(set(assoc)))
    d.cue_echo = cue_n in assoc
    d.exemplar_echo = bool(ex) and bool(ex.intersection(assoc))
    d.within_trial_duplicate = len(set(assoc)) < len(assoc)
    if prompt and assoc:
        # A verbatim continuation reproduces a run of the prompt's own text.
        # Checking the joined answer (not each word) avoids flagging a model
        # that happens to reuse one common word from the instruction.
        joined = ", ".join(assoc)
        d.prompt_copy = len(joined) >= 8 and joined in normalize_form(prompt)
    return d
