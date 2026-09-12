"""The literal prompt text both adapters receive (plan §5.2), and nothing else.

This module holds *strings*. It is separate from `contract.py` because the
manifest must be able to hash the exact text that was sent — including the
held-out Lane B frames, so they provably existed before discovery ran and were
not authored after seeing which cues looked interesting (§5.4).

The task is one short instruction, one neutral exemplar, the cue, and an answer
marker. Nothing asks for explanation; nothing is provider-specific. GPT-2
receives it as a completion that ends at the marker, and an API model receives
the identical block as a single user message with no study-specific system
message. That symmetry is the whole point: a prompt tailored per provider would
make the two arms incomparable before a single statistic ran.

The exemplar is deliberately banal (`bread -> butter, toast, flour`). It is
neutral with respect to every preregistered stratum in `cues.py`, so it cannot
prime the affect, identity, or slang strata differently from the control ones —
and because `normalize.detect` treats its answers as an echo signature, a model
that simply copies it is caught rather than counted.
"""

from __future__ import annotations

from .contract import PromptFrame

#: The exemplar's answers, exported so `normalize.detect` can flag few-shot
#: exemplar echo (§6.5.2) without re-parsing the frame text.
EXEMPLAR_CUE = "bread"
EXEMPLAR_ANSWERS: tuple[str, ...] = ("butter", "toast", "flour")

_INSTRUCTION = (
    "Give the first three words you associate with the word. "
    "Answer with three words separated by commas. No explanation."
)

_BODY = (
    f"{_INSTRUCTION}\n\n"
    f"{EXEMPLAR_CUE} -> {', '.join(EXEMPLAR_ANSWERS)}\n"
    "{cue} -> "
)

#: Stop sequences: a completion model will happily invent a fourth line of the
#: few-shot pattern. Stopping at a newline keeps the raw output to the answer
#: itself; the full text is still stored, since `stop` is applied by the adapter
#: and the untruncated string is what §6.1 calls the evidence.
_STOP: tuple[str, ...] = ("\n",)

#: Lane A primary frame. Discovery and confirmation arm R both use this one.
PRIMARY = PromptFrame(id="lane_a_primary", role="primary", template=_BODY, stop=_STOP)

#: Lane B held-out frames (§5.1 lane B, §5.4 arm G). Semantically equivalent
#: instructions in a different surface form. They are *never* used during
#: discovery; the runner refuses to schedule them on the discovery partition.
HELDOUT: tuple[PromptFrame, ...] = (
    PromptFrame(
        id="lane_b_listing",
        role="heldout",
        template=(
            "Word association task. For each word, list three words that come "
            "to mind first, comma separated.\n\n"
            f"Word: {EXEMPLAR_CUE}\nAssociations: {', '.join(EXEMPLAR_ANSWERS)}\n"
            "Word: {cue}\nAssociations: "
        ),
        stop=_STOP,
    ),
    PromptFrame(
        id="lane_b_terse",
        role="heldout",
        template=(
            "Three words you think of, comma separated, nothing else.\n\n"
            f"{EXEMPLAR_CUE}: {', '.join(EXEMPLAR_ANSWERS)}\n"
            "{cue}: "
        ),
        stop=_STOP,
    ),
)

#: The canary (§5.5.1). One fixed prompt at fixed sampler settings, issued once
#: per time block for the whole study, excluded from every semantic metric. Its
#: cue is `contract.CANARY_CUE`, which no real cue may collide with.
CANARY = PromptFrame(
    id="canary",
    role="canary",
    template=(
        "Give the first three words you associate with the word. "
        "Answer with three words separated by commas. No explanation.\n\n"
        f"{EXEMPLAR_CUE} -> {', '.join(EXEMPLAR_ANSWERS)}\n"
        "window -> "
    ),
    stop=_STOP,
)

#: Lane C context frames (§5.1 lane C — secondary, never pooled with Lane A).
#: Built but not scheduled by the pilot manifest: they exist so that a
#: sense-specificity question about a confirmed cue has a preregistered frame to
#: ask with, rather than one written after the result is known.
CONTEXT_FRAMES: tuple[PromptFrame, ...] = (
    PromptFrame(
        id="lane_c_family",
        role="context_family",
        template=(
            "In a conversation about family, give the first three words you "
            "associate with the word. Three words, comma separated.\n\n"
            f"{EXEMPLAR_CUE} -> {', '.join(EXEMPLAR_ANSWERS)}\n"
            "{cue} -> "
        ),
        stop=_STOP,
    ),
    PromptFrame(
        id="lane_c_slang",
        role="context_slang",
        template=(
            "In a conversation about internet slang, give the first three words "
            "you associate with the word. Three words, comma separated.\n\n"
            f"{EXEMPLAR_CUE} -> {', '.join(EXEMPLAR_ANSWERS)}\n"
            "{cue} -> "
        ),
        stop=_STOP,
    ),
)


def default_frames() -> list[PromptFrame]:
    """Primary + held-out + canary: the set a Lane A study actually schedules."""
    return [PRIMARY, *HELDOUT, CANARY]
