"""The question-set freeze — Phase 0's one-way door, mechanised.

`docs/GENERATIVE-VARIANCE-PLAN.md` §6.5: scores are only comparable within a
fixed instrument, so changing, adding, or reordering a question after
collection begins invalidates every trial already scored, with no migration
short of re-scoring the whole corpus at full cost.

These tests pin the properties that make that failure unreachable quietly:

  1. a draft cannot be collected against, and a freeze is deliberate;
  2. the hash covers exactly what makes a score mean something — id, text,
     scale, and ORDER — and nothing editorial, so documentation stays writable
     after the door closes;
  3. an edit made after the freeze is caught at LOAD time, not at analysis time;
  4. trials from two instruments cannot be pooled, even when the two look
     identical;
  5. missing is NaN and is counted, never 0 — a declined question is not a
     question answered "no".

Fully offline; no model, no network, no money.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from nebulai.backend import instrument as inst

FROZEN_AT = "2026-08-16T00:00:00Z"
REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "docs" / "instruments" / "story-architecture.template.json"


def _q(qid="q_arc", text="Does the story resolve its opening tension?", **kw):
    return inst.Question(id=qid, text=text, **kw)


def _set(name="story-arch", questions=None):
    return inst.QuestionSet(name=name, questions=questions or [_q(), _q("q_pace", "Is the midpoint paced evenly?")])


def _frozen(name="story-arch", questions=None):
    qs = _set(name, questions)
    qs.freeze(FROZEN_AT)
    return qs


# --- the door -------------------------------------------------------------


def test_a_draft_cannot_be_collected_against():
    """The gate that makes 'collect first, freeze later' impossible — which is
    the natural, convenient, corpus-destroying order to do it in."""
    qs = _set()
    assert not qs.is_frozen
    with pytest.raises(inst.InstrumentError) as exc:
        qs.require_frozen()
    assert "draft" in str(exc.value)
    # provenance is gated on the same check, so no artifact can be stamped
    # with an instrument that could still change under it
    with pytest.raises(inst.InstrumentError):
        inst.provenance(qs)


def test_freezing_is_deliberate_and_happens_exactly_once():
    qs = _set()
    h = qs.freeze(FROZEN_AT)
    assert h.startswith("sha256:")
    assert qs.is_frozen and qs.frozen_at == FROZEN_AT
    assert qs.require_frozen() == h
    with pytest.raises(inst.InstrumentError) as exc:
        qs.freeze("2026-09-01T00:00:00Z")
    msg = str(exc.value)
    assert "already frozen" in msg
    assert "NEW question set with a new name" in msg  # names the way forward


def test_an_empty_instrument_cannot_be_frozen():
    with pytest.raises(inst.InstrumentError) as exc:
        inst.QuestionSet(name="empty", questions=[]).freeze(FROZEN_AT)
    assert "no questions" in str(exc.value)


def test_the_shipped_template_cannot_be_frozen():
    """The template documents the format. If it could be frozen it would become
    an instrument, and the study would be running on questions the tooling
    invented to demonstrate a JSON schema."""
    qs = inst.load_question_set(TEMPLATE)
    assert not qs.is_frozen
    assert len(qs.questions) == 3  # one per scale kind
    with pytest.raises(inst.InstrumentError) as exc:
        qs.freeze(FROZEN_AT)
    msg = str(exc.value)
    assert "example_likert" in msg
    assert "§6.2-6.4" in msg  # points at the work that must happen first


# --- what the hash is, and is not, sensitive to ---------------------------


def test_reordering_questions_is_a_different_instrument():
    """The trial vector IS the scores in order, so a pure reordering produces
    rows that cannot be pooled with earlier ones even though every question
    survived intact. This is the case most likely to look harmless."""
    a, b = _q("q_a", "A?"), _q("q_b", "B?")
    assert _set(questions=[a, b]).compute_hash() != _set(questions=[b, a]).compute_hash()


def test_rewording_a_question_is_a_different_instrument_even_with_the_same_id():
    """An id is a reference, not a promise that the meaning held still."""
    same_id_new_text = _q("q_a", "Does the ending land?")
    assert (
        _set(questions=[_q("q_a", "A?")]).compute_hash()
        != _set(questions=[same_id_new_text]).compute_hash()
    )


def test_changing_a_scale_is_a_different_instrument():
    assert (
        _set(questions=[_q("q_a", "A?", lo=1.0, hi=5.0)]).compute_hash()
        != _set(questions=[_q("q_a", "A?", lo=1.0, hi=7.0)]).compute_hash()
    )


def test_editorial_fields_are_not_hashed_so_documentation_stays_writable():
    """If notes were hashed, the freeze would become a standing reason never to
    write anything down about the instrument again."""
    plain = _set(questions=[_q("q_a", "A?")])
    annotated = _set(questions=[_q("q_a", "A?", note="added six months later")])
    annotated.notes = "and a study-level note too"
    assert plain.compute_hash() == annotated.compute_hash()


def test_two_studies_asking_the_same_questions_do_not_collide():
    qa = inst.QuestionSet(name="w1-selfvar", questions=[_q("q_a", "A?")])
    qb = inst.QuestionSet(name="w2-crossmodel", questions=[_q("q_a", "A?")])
    assert qa.compute_hash() != qb.compute_hash()


def test_the_hash_is_independent_of_file_formatting(tmp_path):
    qs = _frozen()
    p = qs.save(tmp_path / "i.json")
    reflowed = tmp_path / "reflowed.json"
    reflowed.write_text(json.dumps(json.loads(p.read_text()), indent=8))
    assert inst.load_question_set(reflowed).frozen_hash == qs.frozen_hash


def test_duplicate_ids_are_rejected_at_construction():
    with pytest.raises(inst.InstrumentError) as exc:
        inst.QuestionSet(name="dup", questions=[_q("q_a", "A?"), _q("q_a", "B?")])
    assert "duplicate question id" in str(exc.value)


# --- tamper detection -----------------------------------------------------


def test_an_edit_after_the_freeze_is_caught_at_load(tmp_path):
    """The whole point of storing the hash. Catching this at analysis time —
    or never — is how a corpus quietly becomes evidence about two instruments."""
    qs = _frozen()
    p = qs.save(tmp_path / "i.json")
    d = json.loads(p.read_text())
    d["questions"][0]["text"] = "a subtly different question"
    p.write_text(json.dumps(d))

    with pytest.raises(inst.InstrumentError) as exc:
        inst.load_question_set(p)
    msg = str(exc.value)
    assert "edited after it was frozen" in msg
    assert "version control" in msg  # names the recovery


def test_editing_an_unhashed_field_after_the_freeze_is_fine(tmp_path):
    qs = _frozen()
    p = qs.save(tmp_path / "i.json")
    d = json.loads(p.read_text())
    d["notes"] = "post-freeze commentary"
    d["questions"][0]["note"] = "clarified for the annotators"
    p.write_text(json.dumps(d))
    assert inst.load_question_set(p).frozen_hash == qs.frozen_hash


def test_a_draft_roundtrips_without_an_integrity_check(tmp_path):
    qs = _set()
    loaded = inst.load_question_set(qs.save(tmp_path / "draft.json"))
    assert not loaded.is_frozen
    assert [q.id for q in loaded.questions] == [q.id for q in qs.questions]


def test_a_missing_or_malformed_file_says_which(tmp_path):
    with pytest.raises(inst.InstrumentError, match="no question set at"):
        inst.load_question_set(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(inst.InstrumentError, match="not valid JSON"):
        inst.load_question_set(bad)


def test_a_newer_format_version_refuses_rather_than_guesses(tmp_path):
    p = tmp_path / "future.json"
    p.write_text(json.dumps({"format_version": 99, "name": "x", "questions": []}))
    with pytest.raises(inst.InstrumentError, match="newer than this build"):
        inst.load_question_set(p)


# --- trial vectors --------------------------------------------------------


def test_missing_is_nan_and_never_zero():
    """§7, verbatim: a question the scorer declined is not a question answered
    'no', and collapsing the two would manufacture agreement."""
    qs = _frozen(questions=[_q("q_a", "A?"), _q("q_b", "B?"), _q("q_c", "C?")])
    row = inst.trial_vector(qs, {"q_a": 4, "q_c": None})
    assert row[0] == 4.0
    assert np.isnan(row[1])  # absent from the dict entirely
    assert np.isnan(row[2])  # present and explicitly None
    assert not (row == 0).any()
    assert row.dtype == np.float32


def test_a_zero_answer_stays_zero_and_is_not_missing():
    """The other half of the same rule — 'no' must survive as a real datum."""
    qs = _frozen(questions=[_q("q_a", "A?", kind="binary", lo=0.0, hi=1.0)])
    row = inst.trial_vector(qs, {"q_a": 0})
    assert row[0] == 0.0
    assert not np.isnan(row[0])


def test_the_vector_follows_frozen_order_not_answer_order():
    qs = _frozen(questions=[_q("q_a", "A?"), _q("q_b", "B?")])
    assert list(inst.trial_vector(qs, {"q_b": 2, "q_a": 5})) == [5.0, 2.0]


def test_an_off_scale_answer_raises_rather_than_clipping():
    """A scorer answering 7 on a 1-5 scale has misunderstood the prompt.
    Clipping to 5 turns a detectable instrument failure into a plausible datum."""
    qs = _frozen(questions=[_q("q_a", "A?", lo=1.0, hi=5.0)])
    with pytest.raises(inst.InstrumentError) as exc:
        inst.trial_vector(qs, {"q_a": 7})
    assert "off the scale" in str(exc.value)


def test_an_answer_to_a_question_that_is_not_in_the_instrument_raises():
    """Instrument drift arriving from the response side instead of the file."""
    qs = _frozen(questions=[_q("q_a", "A?")])
    with pytest.raises(inst.InstrumentError) as exc:
        inst.trial_vector(qs, {"q_a": 3, "q_invented": 5})
    assert "q_invented" in str(exc.value)


def test_stacking_trials_gives_the_units_shape_and_labels():
    """§5's largest reuse win: the trial matrix is exactly what `Units` wants,
    so reduce -> cluster -> name -> export is inherited unchanged."""
    from nebulai.units import Units

    qs = _frozen(questions=[_q("q_a", "A?"), _q("q_b", "B?")])
    trials = [
        inst.Trial("prompt-01", 0, {"q_a": 1, "q_b": 2}),
        inst.Trial("prompt-01", 1, {"q_a": 3, "q_b": None}),
        inst.Trial("prompt-02", 0, {"q_a": 5, "q_b": 4}),
    ]
    rows, labels = inst.stack_trials(qs, trials)
    assert rows.shape == (3, 2)
    assert rows.dtype == np.float32
    assert labels == ["prompt-01/0", "prompt-01/1", "prompt-02/0"]

    units = Units(
        ids=list(range(len(trials))),
        vectors=rows,
        labels=labels,
        meta=inst.provenance(qs),
    )
    assert len(units) == 3
    assert units.meta["instrument_hash"] == qs.frozen_hash


def test_missingness_is_reported_even_when_it_is_zero():
    """Absence has ink, including the absence of absence: a study that omits
    this field is indistinguishable from one that had nothing to report."""
    complete = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    m = inst.missingness(complete)
    assert m["missing_total"] == 0
    assert m["missing_per_question"] == [0, 0]
    assert m["trials_complete"] == 2

    holed = np.array([[1.0, np.nan], [np.nan, np.nan]], dtype=np.float32)
    m2 = inst.missingness(holed)
    assert m2["missing_total"] == 3
    assert m2["missing_per_question"] == [1, 2]
    assert m2["trials_complete"] == 0


# --- pooling --------------------------------------------------------------


def test_trials_from_a_different_instrument_cannot_be_pooled():
    qs = _frozen(questions=[_q("q_a", "A?")])
    other = _frozen(name="other", questions=[_q("q_a", "A?")])
    with pytest.raises(inst.InstrumentError) as exc:
        inst.require_compatible(qs, inst.provenance(other))
    assert "instrument mismatch" in str(exc.value)


def test_pooling_matches_on_the_hash_not_on_the_name_or_the_count():
    """Two sets can share a name and a question count and still ask different
    things — the most plausible costume the §6.5 failure wears."""
    qs = _frozen(questions=[_q("q_a", "A?")])
    reworded = _frozen(questions=[_q("q_a", "A, but differently?")])
    meta = inst.provenance(reworded)
    assert meta["instrument"] == "story-arch"  # same name
    assert meta["n_questions"] == 1  # same count
    with pytest.raises(inst.InstrumentError):
        inst.require_compatible(qs, meta)
    # and the matching case passes
    inst.require_compatible(qs, inst.provenance(qs))


def test_untagged_trials_cannot_be_pooled_at_all():
    """Silence is not a match. Trials with no hash are evidence about an
    unknown instrument."""
    qs = _frozen()
    with pytest.raises(inst.InstrumentError) as exc:
        inst.require_compatible(qs, {"instrument": "story-arch"})
    assert "no instrument hash" in str(exc.value)
