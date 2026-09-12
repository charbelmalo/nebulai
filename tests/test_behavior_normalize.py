"""Parser and normalizer goldens (BEHAVIORAL-DIVERGENCE-PLAN.md §6.2, §6.5).

The normalizer decides what counts as "the same associate", so every downstream
number — type counts, entropy, JSD, RBO, the informativeness floor — inherits
its choices. §6.2 fixes exactly four steps (NFKC, strip surrounding punctuation,
case-fold, and nothing else) and one prohibition: no stemming, no lemmatization,
no singular/plural folding. "cat" and "cats" are different types, because
deciding otherwise is a lexical theory smuggled into a measurement.

The goldens below are the whole contract. They are written as explicit
input/output pairs rather than as property assertions so that a future change to
the rules shows up as a diff a reader can argue with.

The other half of the file pins the failure taxonomy. An unparseable completion
is DATA — its rate is the compliance figure §6.5.1 gates on — so the parser
returns a reason, never an exception, and never silently drops a trial.
"""

import pytest

from nebulai.behavior.normalize import (
    PARSER_VERSION,
    Detectors,
    detect,
    normalize_form,
    parse_associates,
)

# --------------------------------------------------------------------------
# normalize_form: the four steps, and only those
# --------------------------------------------------------------------------

GOLDENS = [
    # (raw, expected) — comment says which §6.2 step is doing the work
    ("Butter", "butter"),                       # case-fold
    ("  butter  ", "butter"),                   # surrounding whitespace
    ("butter.", "butter"),                      # trailing punctuation
    ('"butter"', "butter"),                     # straight quotes
    ("“butter”", "butter"),           # curly quotes
    ("- butter", "butter"),                     # list dash
    ("1. butter", "butter"),                    # numbered list
    ("a) butter", "butter"),                    # lettered list
    ("• butter", "butter"),                # bullet
    ("ICE   CREAM", "ice cream"),               # internal whitespace collapse
    ("ｂｕｔｔｅｒ", "butter"),  # NFKC: fullwidth
    ("ﬁre", "fire"),                       # NFKC: ligature
    ("café", "café"),                     # NFKC: combining acute composes
    ("STRASSE", "strasse"),                     # case-fold leaves ss alone
    ("ßeta", "sseta"),                     # case-fold expands sharp s
    ("İstanbul", "i̇stanbul"),             # no locale-specific casing
]


@pytest.mark.parametrize(("raw", "expected"), GOLDENS)
def test_normalize_form_goldens(raw, expected):
    assert normalize_form(raw) == expected


def test_normalization_is_idempotent():
    for raw, _ in GOLDENS:
        once = normalize_form(raw)
        assert normalize_form(once) == once


NOT_FOLDED = [
    ("cat", "cats"),          # no plural folding
    ("run", "running"),       # no stemming
    ("mouse", "mice"),        # no lemmatization
    ("colour", "color"),      # no spelling harmonization
    ("ice cream", "icecream"),  # no compound joining
]


@pytest.mark.parametrize(("a", "b"), NOT_FOLDED)
def test_the_normalizer_does_not_smuggle_in_a_lexical_theory(a, b):
    """§6.2's prohibition. These pairs MUST stay distinct types."""
    assert normalize_form(a) != normalize_form(b)


def test_internal_punctuation_survives():
    """Only SURROUNDING punctuation is stripped; a hyphenated word is one type."""
    assert normalize_form("well-being.") == "well-being"
    assert normalize_form("rock 'n' roll") == "rock 'n' roll"


# --------------------------------------------------------------------------
# parse_associates: structure and the failure taxonomy
# --------------------------------------------------------------------------


def test_comma_separated_answer_parses_to_three():
    r = parse_associates("butter, toast, flour")
    assert r.valid is True
    assert r.associates == ["butter", "toast", "flour"]
    assert r.surface == ["butter", "toast", "flour"]
    assert r.delimiter == "comma"
    assert r.n_parsed == 3
    assert r.parser_version == PARSER_VERSION


def test_the_surface_form_is_kept_beside_the_normalized_one():
    """§6.2: the exact spelling must stay visible wherever an embedding-derived
    claim is made, or the reader cannot check what was embedded."""
    r = parse_associates("Butter, TOAST, Flour.")
    assert r.associates == ["butter", "toast", "flour"]
    # The surface is what the model actually emitted, trailing period included.
    # Normalizing it "a little" would leave the reader unable to tell which
    # spelling was embedded.
    assert r.surface == ["Butter", "TOAST", "Flour."]


@pytest.mark.parametrize(
    ("raw", "delim"),
    [
        ("butter, toast, flour", "comma"),
        ("butter; toast; flour", "semicolon"),
        ("butter / toast / flour", "slash"),
        ("butter | toast | flour", "pipe"),
        ("butter\ntoast\nflour", "newline"),
    ],
)
def test_every_supported_delimiter_yields_the_same_three(raw, delim):
    r = parse_associates(raw)
    assert r.associates == ["butter", "toast", "flour"]
    assert r.delimiter == delim


def test_mixed_delimiters_are_marked_not_rejected():
    """A model that uses two separators answered; charging it `too_few:2` would
    put a parser artifact into the §6.5.1 compliance figure."""
    r = parse_associates("butter, toast; flour")
    assert r.mixed_delimiters is True
    assert r.associates == ["butter", "toast", "flour"]
    assert r.delimiter == "mixed"
    assert r.valid is True


def test_a_numbered_list_parses():
    r = parse_associates("1. butter\n2. toast\n3. flour")
    assert r.associates == ["butter", "toast", "flour"]


def test_the_answer_marker_echo_is_stripped():
    """A base LM continuing "bread -> " usually re-emits the cue and marker."""
    r = parse_associates("bread -> butter, toast, flour")
    assert r.associates == ["butter", "toast", "flour"]


def test_only_the_first_nonempty_line_is_the_answer():
    r = parse_associates("\n\nbutter, toast, flour\nwindow -> glass, pane, sill")
    assert r.associates == ["butter", "toast", "flour"]


def test_more_than_three_fields_truncates_but_records_how_many_there_were():
    r = parse_associates("butter, toast, flour, jam, yeast")
    assert r.associates == ["butter", "toast", "flour"]
    assert r.n_parsed == 5
    assert r.valid is True


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (None, "no_output"),
        ("", "empty"),
        ("   \n  ", "empty"),
        ("butter", "too_few:1"),
        ("butter, toast", "too_few:2"),
    ],
)
def test_failures_carry_a_reason_and_never_raise(raw, reason):
    r = parse_associates(raw)
    assert r.valid is False
    assert r.reason == reason


def test_a_refusal_is_its_own_category():
    """§6.5.1 gates on compliance parity, which needs refusals separated from
    parse failures — collapsing them would make a refusing model look merely
    badly formatted."""
    r = parse_associates("I'm sorry, but I can't help with that.")
    assert r.valid is False
    assert r.refusal is True
    assert r.reason == "refusal"


def test_a_partial_answer_keeps_what_it_parsed():
    """An invalid trial still carries its content: §6.5.1 needs the partial
    associates to compute the informativeness floor, and discarding them would
    bias the surviving sample."""
    r = parse_associates("butter, toast")
    assert r.valid is False
    assert r.associates == ["butter", "toast"]


# --------------------------------------------------------------------------
# detectors: mark, never replace
# --------------------------------------------------------------------------


def test_cue_echo_is_marked():
    r = parse_associates("bread, butter, toast")
    d = detect(r, cue="Bread")
    assert d.cue_echo is True
    assert d.degenerate is True


def test_exemplar_echo_needs_the_frames_own_answers():
    r = parse_associates("butter, toast, flour")
    assert detect(r, cue="rain").exemplar_echo is False
    d = detect(r, cue="rain", exemplar_answers=("butter", "toast", "flour"))
    assert d.exemplar_echo is True


def test_within_trial_duplicates_are_marked_and_counted():
    r = parse_associates("butter, butter, toast")
    d = detect(r, cue="bread")
    assert d.within_trial_duplicate is True
    assert d.distinct_types == 2


def test_a_clean_trial_is_not_degenerate():
    r = parse_associates("cold, warm, heat")
    d = detect(r, cue="hot", exemplar_answers=("butter", "toast", "flour"))
    assert d.degenerate is False
    assert d.distinct_types == 3


def test_detectors_do_not_mutate_the_parse():
    r = parse_associates("bread, bread, bread")
    before = list(r.associates)
    detect(r, cue="bread")
    assert r.associates == before, "§6.2 step 5: mark, never replace"


def test_detectors_default_to_clean():
    d = Detectors()
    assert d.degenerate is False
