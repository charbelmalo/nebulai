"""Agreement statistics and the annotation kit, on synthetic sheets.

`docs/GENERATIVE-VARIANCE-PLAN.md` §6.2–§6.5. Every number checked here has a
closed form or a published worked example, so these are tests rather than
change detectors: a regression in the estimator changes an exact value, not a
golden blob nobody can re-derive.

The four properties that matter most, and would be silent failures otherwise:

  1. **Missing is not zero.** A declined answer must be excluded pairwise and
     counted; imputing it as 0 must be shown to change the answer, or nothing
     stops a future refactor from doing exactly that.
  2. **The floor is tested on the lower CI bound** (§6.2 step 4), which means a
     question with a high point estimate and a wide interval must FAIL.
  3. **κ alone deletes rare-feature questions**, so PABAK/AC1/raw must survive
     the case where κ is undefined.
  4. **The draft instrument cannot be frozen.** `docs/instruments/` ships a
     complete straw man; §6.5's one-way door stays shut by the id namespace.
"""

import json
import math
from pathlib import Path

import pytest

from nebulai.backend import agreement as ag
from nebulai.backend import variance_cli as vc
from nebulai.backend.instrument import InstrumentError, load_question_set

DOCS = Path(__file__).resolve().parents[1] / "docs" / "instruments"


# --------------------------------------------------------------------------
# closed-form estimators
# --------------------------------------------------------------------------


def _classic_2x2():
    """The textbook 2x2 with cells (20, 5, 10, 15): p_o = .70, p_e = .50, k = .40."""
    a = [1.0] * 25 + [0.0] * 25
    b = [1.0] * 20 + [0.0] * 5 + [1.0] * 10 + [0.0] * 15
    return a, b


def test_cohen_kappa_matches_the_closed_form():
    a, b = _classic_2x2()
    assert ag.raw_agreement(a, b) == pytest.approx(0.70)
    assert ag.cohen_kappa(a, b, [0.0, 1.0]) == pytest.approx(0.40, abs=1e-12)


def test_pabak_is_the_prevalence_free_counterfactual():
    a, b = _classic_2x2()
    # binary PABAK reduces to 2*p_o - 1
    assert ag.pabak(a, b, [0.0, 1.0]) == pytest.approx(2 * 0.70 - 1.0)


def test_weighted_kappa_credits_near_misses():
    cats = [1.0, 2.0, 3.0, 4.0, 5.0]
    a = [1.0, 2.0, 3.0, 4.0, 5.0] * 6
    b = [2.0, 3.0, 4.0, 5.0, 1.0] * 6  # every answer off by one, one wrapped
    un = ag.cohen_kappa(a, b, cats)
    qw = ag.cohen_kappa(a, b, cats, weights="quadratic")
    assert qw > un, "quadratic weighting must reward an ordinal near-miss"


def test_krippendorff_alpha_matches_the_published_worked_example():
    """Krippendorff (2011), 15 units x 3 observers: .691 / .807 / .811."""
    A = [None, None, None, None, None, 3, 4, 1, 2, 1, 1, 3, 3, None, 3]
    B = [1, None, 2, 1, 3, 3, 4, 3, None, None, None, None, None, None, None]
    C = [None, None, 2, 1, 3, 4, 4, None, 2, 1, 1, 3, 3, None, 4]

    def sheet(vals):
        return {str(i): (None if v is None else float(v)) for i, v in enumerate(vals)}

    rat = [sheet(A), sheet(B), sheet(C)]
    cats = [1.0, 2.0, 3.0, 4.0]
    assert ag.krippendorff_alpha(rat, cats, level="nominal") == pytest.approx(0.691, abs=5e-4)
    assert ag.krippendorff_alpha(rat, cats, level="ordinal") == pytest.approx(0.807, abs=5e-4)
    assert ag.krippendorff_alpha(rat, cats, level="interval") == pytest.approx(0.811, abs=5e-4)


def test_perfect_and_chance_agreement_bracket_kappa():
    cats = [1.0, 2.0, 3.0]
    same = [1.0, 2.0, 3.0] * 10
    assert ag.cohen_kappa(same, same, cats) == pytest.approx(1.0)
    assert ag.gwet_ac1(same, same, cats) == pytest.approx(1.0)


def test_kappa_is_undefined_not_zero_when_both_raters_are_constant():
    """§6.3's prevalence trap, at its limit.

    Both raters say "absent" on every story of a rare feature. κ has no
    denominator; reporting 0 would say they disagreed at chance, which is the
    opposite of what happened. AC1 and raw agreement must still answer.
    """
    cats = [0.0, 1.0]
    a = b = [0.0] * 40
    assert math.isnan(ag.cohen_kappa(a, b, cats))
    assert ag.raw_agreement(a, b) == 1.0
    assert ag.gwet_ac1(a, b, cats) == pytest.approx(1.0)
    assert ag.pabak(a, b, cats) == pytest.approx(1.0)


def test_rare_feature_shows_the_prevalence_paradox():
    """95% raw agreement, κ near zero — the exact case §6.3 says κ would delete."""
    cats = [0.0, 1.0]
    a = [0.0] * 38 + [1.0, 0.0]
    b = [0.0] * 38 + [0.0, 1.0]
    assert ag.raw_agreement(a, b) == pytest.approx(0.95)
    assert ag.cohen_kappa(a, b, cats) < 0.05
    assert ag.pabak(a, b, cats) == pytest.approx(0.90)
    assert ag.gwet_ac1(a, b, cats) > 0.9


# --------------------------------------------------------------------------
# missing is not zero
# --------------------------------------------------------------------------


def test_missing_is_excluded_pairwise_and_counted():
    a = {f"s{i}": 1.0 for i in range(10)}
    b = dict(a)
    a["s3"] = None
    b["s7"] = None
    r = ag.question_agreement("q", "binary", 0.0, 1.0, a, b, n_boot=200, label_a="A", label_b="B")
    assert r.n_items == 10
    assert r.n_paired == 8
    assert r.n_missing == {"A": 1, "B": 1}


def test_imputing_missing_as_zero_would_change_the_answer():
    """The regression guard for the one substitution §7 forbids."""
    items = [f"s{i}" for i in range(20)]
    a = {s: 1.0 for s in items}
    b = {s: 1.0 for s in items}
    for s in items[:8]:
        a[s] = None  # declined, not "no"
    honest = ag.question_agreement("q", "binary", 0.0, 1.0, a, b, n_boot=0)
    imputed_a = {s: (0.0 if v is None else v) for s, v in a.items()}
    imputed = ag.question_agreement("q", "binary", 0.0, 1.0, imputed_a, b, n_boot=0)
    assert honest.raw == pytest.approx(1.0)
    assert imputed.raw == pytest.approx(0.6)
    assert honest.n_paired == 12 and imputed.n_paired == 20


def test_continuous_questions_refuse_kappa_with_a_reason():
    a = {f"s{i}": i / 20 for i in range(20)}
    b = {f"s{i}": (i + 1) / 20 for i in range(20)}
    r = ag.question_agreement("q", "unit", 0.0, 1.0, a, b, n_boot=0)
    assert r.kappa is None
    assert r.alpha is not None
    assert "bin boundary" in r.reason


# --------------------------------------------------------------------------
# the §6.2 decision and the §6.5 halt
# --------------------------------------------------------------------------


def _interval(est, lo, hi):
    return ag.Interval(est, lo, hi, 2000, 50)


def test_floor_is_relative_to_the_measured_human_ceiling():
    """κ_M = 0.78 passes against κ_H = 0.74 — it is at the ceiling, not below a constant."""
    d = ag.floor_decision("q", 0.7385, _interval(0.78, 0.62, 0.90))
    assert d.verdict == "pass"
    assert d.floor == pytest.approx(min(0.40, 0.80 * 0.7385))


def test_a_wide_interval_fails_even_with_a_high_point_estimate():
    """§6.2 step 4: the bound decides, not the estimate."""
    d = ag.floor_decision("q", 0.90, _interval(0.85, 0.10, 0.99))
    assert d.verdict == "drop-low-reliability"
    assert d.kappa_m == pytest.approx(0.85)


def test_a_question_humans_cannot_agree_on_is_dropped_as_badly_specified():
    d = ag.floor_decision("q", 0.11, _interval(0.95, 0.90, 0.99))
    assert d.verdict == "drop-badly-specified"
    assert "defect in the question" in d.detail


def test_a_missing_ceiling_is_undecidable_not_a_pass():
    d = ag.floor_decision("q", None, _interval(0.95, 0.90, 0.99))
    assert d.verdict == "undecidable"


def test_intra_rater_below_inter_rater_halts():
    chk = ag.intra_rater_check({"a": 0.50, "b": 0.55}, {"a": 0.80, "b": 0.70})
    assert chk.halt is True
    assert "HALT" in chk.detail


def test_intra_rater_at_or_above_inter_rater_does_not_halt():
    chk = ag.intra_rater_check({"a": 0.90, "b": 0.85}, {"a": 0.80, "b": 0.70})
    assert chk.halt is False


def test_an_unperformed_intra_rater_check_is_not_a_pass():
    chk = ag.intra_rater_check({}, {"a": 0.80})
    assert chk.halt is False
    assert "NOT PERFORMED" in chk.detail
    assert chk.mean_intra is None


# --------------------------------------------------------------------------
# the shipped draft
# --------------------------------------------------------------------------


def test_the_draft_instrument_loads_and_every_question_carries_a_rubric():
    qs = load_question_set(DOCS / "story-architecture.draft.json")
    assert len(qs.questions) >= 30
    assert not qs.is_frozen
    for q in qs.questions:
        assert q.id.startswith("draft_")
        assert q.note.startswith("RUBRIC"), f"{q.id} has no rubric"
    kinds = {q.kind for q in qs.questions}
    assert kinds == {"likert", "binary", "unit"}


def test_the_draft_instrument_cannot_be_frozen():
    """§6.5's one-way door, verified shut on the file we actually ship."""
    qs = load_question_set(DOCS / "story-architecture.draft.json")
    with pytest.raises(InstrumentError) as exc:
        qs.freeze("2026-09-11T00:00:00Z")
    assert "reserved-prefix" in str(exc.value)
    assert not qs.is_frozen


def test_the_format_template_also_cannot_be_frozen():
    qs = load_question_set(DOCS / "story-architecture.template.json")
    with pytest.raises(InstrumentError):
        qs.freeze("2026-09-11T00:00:00Z")


def test_the_draft_prompt_set_is_namespaced_and_stratified():
    d = json.loads((DOCS / "story-prompts.draft.json").read_text(encoding="utf-8"))
    ids = [p["id"] for p in d["prompts"]]
    assert len(ids) >= 30
    assert len(set(ids)) == len(ids)
    assert all(i.startswith("draft_") for i in ids)
    axes = set(d["axes"])
    for p in d["prompts"]:
        assert set(p["strata"]) == axes, p["id"]


def test_the_prompt_draft_does_not_claim_to_be_crossed():
    """The design block has to keep describing the prompts, not the intention.

    A stratified prompt set invites one specific self-deception: four named axes
    look like a 3x2x2x2 design, and a between-prompt variance decomposition run
    on them would report four main effects as though each were estimated from a
    balanced set. It is not -- most prompts differ from the base along ONE axis,
    two axes have four prompts on their off-level, and 14 of the 24 cells are
    empty. So the file states that, and this test keeps the statement tied to
    the data: every count in `design` is recomputed here, and a prompt added or
    re-stratified without updating the block fails rather than quietly turning
    the description into a wish.
    """
    d = json.loads((DOCS / "story-prompts.draft.json").read_text(encoding="utf-8"))
    design = d["design"]
    prompts = d["prompts"]
    axis_names = list(d["axes"])

    assert "NOT a full crossing" in design["kind"]
    assert design["n_prompts"] == len(prompts)

    # levels are declared, not inferred -- a level nobody wrote is still a cell
    for ax, lvls in design["axis_levels"].items():
        declared = [t.strip() for t in d["axes"][ax].rsplit(":", 1)[-1].split("|")]
        assert lvls == declared, ax
        assert {p["strata"][ax] for p in prompts} <= set(lvls), ax

    for ax, counts in design["level_counts"].items():
        for lv, n in counts.items():
            got = sum(1 for p in prompts if p["strata"][ax] == lv)
            assert got == n, f"{ax}={lv}: block says {n}, prompts say {got}"
        assert sum(counts.values()) == len(prompts), ax

    full = 1
    for ax in axis_names:
        full *= len(design["axis_levels"][ax])
    occupied = {tuple(p["strata"][ax] for ax in axis_names) for p in prompts}
    assert design["cells_of_full_crossing"] == full
    assert design["cells_occupied"] == len(occupied)
    # the claim that makes the honesty load-bearing: the set is sparse
    assert len(occupied) < full

    # the base prompt the star design is measured against must exist
    assert any(p["id"] == design["base_prompt"] for p in prompts)


# --------------------------------------------------------------------------
# the annotation kit
# --------------------------------------------------------------------------


def _args(**kw):
    import argparse

    ns = argparse.Namespace(
        instrument=str(DOCS / "story-architecture.draft.json"),
        stories="",
        n_stories=6,
        annotators="A,B",
        repeat_n=2,
        out="",
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_annotate_sheet_emits_blank_sheets_for_each_annotator(tmp_path, capsys):
    vc.run_annotate_sheet(_args(out=str(tmp_path)))
    for a in ("A", "B"):
        sheet = json.loads((tmp_path / f"sheet_{a}.json").read_text(encoding="utf-8"))
        assert sheet["annotator"] == a
        assert sheet["instrument"]["frozen"] is False
        assert "DRAFT" in sheet["warning"]
        assert len(sheet["answers"]) == 6
        # Nothing in the kit answers a question.
        assert all(v is None for row in sheet["answers"].values() for v in row.values())
        assert (tmp_path / f"sheet_{a}.csv").exists()
        assert (tmp_path / f"repeat_{a}.json").exists()
    meta = json.loads((tmp_path / "sheets.meta.json").read_text(encoding="utf-8"))
    assert meta["instrument_frozen"] is False
    assert meta["story_source"] == "placeholder"
    out = capsys.readouterr().out
    assert "PLACEHOLDER" in out


def test_annotate_sheet_refuses_a_single_annotator(tmp_path):
    with pytest.raises(SystemExit) as exc:
        vc.run_annotate_sheet(_args(out=str(tmp_path), annotators="A"))
    assert "TWO independent annotators" in str(exc.value)


def test_blank_csv_round_trips_as_all_missing(tmp_path):
    vc.run_annotate_sheet(_args(out=str(tmp_path)))
    back = vc.load_sheet(tmp_path / "sheet_A.csv")
    vals = [v for row in back["answers"].values() for v in row.values()]
    assert vals and all(v is None for v in vals)


# --------------------------------------------------------------------------
# end-to-end agreement on synthetic sheets with known answers
# --------------------------------------------------------------------------


def _fill(tmp_path, name, table, qs, *, annotator):
    """Write a filled sheet. `table` maps question id -> list of answers."""
    stories = [f"g{i:02d}" for i in range(len(next(iter(table.values()))))]
    answers = {
        s: {q.id: (table[q.id][i] if q.id in table else None) for q in qs.questions}
        for i, s in enumerate(stories)
    }
    sheet = {
        "format": vc.SHEET_FORMAT,
        "version": vc.SHEET_VERSION,
        "annotator": annotator,
        "block": "primary",
        "instrument": {
            "name": qs.name,
            "hash": qs.compute_hash(),
            "frozen": False,
            "question_ids": [q.id for q in qs.questions],
        },
        "stories": {"source": "synthetic", "n": len(stories), "ids": stories},
        "answers": answers,
    }
    p = tmp_path / name
    p.write_text(json.dumps(sheet), encoding="utf-8")
    return p


def _mini_instrument(tmp_path):
    """A three-question draft: one designed for κ = 0.40, one perfect, one rare."""
    qs = {
        "format_version": 1,
        "name": "synthetic",
        "frozen_at": None,
        "hash": None,
        "questions": [
            {"id": "draft_mid", "text": "mid", "kind": "binary", "lo": 0.0, "hi": 1.0},
            {"id": "draft_perfect", "text": "perfect", "kind": "binary", "lo": 0.0, "hi": 1.0},
            {"id": "draft_rare", "text": "rare", "kind": "binary", "lo": 0.0, "hi": 1.0},
        ],
    }
    p = tmp_path / "inst.json"
    p.write_text(json.dumps(qs), encoding="utf-8")
    return p


def test_agreement_end_to_end_recovers_the_designed_kappa(tmp_path, capsys):
    inst_path = _mini_instrument(tmp_path)
    qs = load_question_set(inst_path)
    a_mid, b_mid = _classic_2x2()
    perfect = [float(i % 2) for i in range(50)]
    rare_a = [0.0] * 48 + [1.0, 0.0]
    rare_b = [0.0] * 48 + [0.0, 1.0]

    ha = _fill(
        tmp_path, "A.json",
        {"draft_mid": a_mid, "draft_perfect": perfect, "draft_rare": rare_a},
        qs, annotator="A",
    )
    hb = _fill(
        tmp_path, "B.json",
        {"draft_mid": b_mid, "draft_perfect": perfect, "draft_rare": rare_b},
        qs, annotator="B",
    )

    import argparse

    ns = argparse.Namespace(
        instrument=str(inst_path), human=[str(ha), str(hb)], model=[],
        intra="", intra_of="", n_boot=400, seed=0,
        absolute_floor=0.40, ceiling_fraction=0.80, kappa_h_min=0.40,
        min_questions=20, out=str(tmp_path / "report.json"),
    )
    vc.run_agreement(ns)
    rep = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))

    mid = rep["questions"]["draft_mid"]["human_human"]
    assert mid["kappa"]["estimate"] == pytest.approx(0.40, abs=1e-9)
    assert mid["kappa"]["ci"][0] < 0.40 < mid["kappa"]["ci"][1]
    assert mid["raw_agreement"] == pytest.approx(0.70)

    assert rep["questions"]["draft_perfect"]["human_human"]["kappa"]["estimate"] == pytest.approx(1.0)

    rare = rep["questions"]["draft_rare"]["human_human"]
    assert rare["raw_agreement"] == pytest.approx(0.96)
    assert rare["kappa"]["estimate"] < 0.05
    assert rare["pabak"] > 0.9  # the number that keeps the question alive

    # No model sheet -> no κ_M -> nothing can pass; a question whose own
    # annotators agree (draft_perfect) is UNDECIDABLE rather than quietly
    # passed, and the rare question is dropped on its human ceiling before the
    # scorer is judged against it at all.
    verdicts = {d["question_id"]: d["verdict"] for d in rep["decisions"]}
    assert "pass" not in verdicts.values()
    assert verdicts["draft_perfect"] == "undecidable"
    assert verdicts["draft_rare"] == "drop-badly-specified"
    assert rep["summary"]["gate0_instrument_too_thin"] is True
    assert "GATE 0 STOP" in capsys.readouterr().out


def test_agreement_refuses_two_sheets_from_different_instruments(tmp_path):
    inst_path = _mini_instrument(tmp_path)
    qs = load_question_set(inst_path)
    col = [0.0, 1.0] * 10
    ha = _fill(tmp_path, "A.json", {"draft_mid": col}, qs, annotator="A")
    hb = _fill(tmp_path, "B.json", {"draft_mid": col}, qs, annotator="B")
    tampered = json.loads(hb.read_text(encoding="utf-8"))
    tampered["instrument"]["hash"] = "sha256:" + "0" * 64
    hb.write_text(json.dumps(tampered), encoding="utf-8")

    import argparse

    ns = argparse.Namespace(
        instrument=str(inst_path), human=[str(ha), str(hb)], model=[],
        intra="", intra_of="", n_boot=50, seed=0,
        absolute_floor=0.40, ceiling_fraction=0.80, kappa_h_min=0.40,
        min_questions=2, out=str(tmp_path / "r.json"),
    )
    with pytest.raises(SystemExit) as exc:
        vc.run_agreement(ns)
    assert "only comparable within a fixed instrument" in str(exc.value)


def test_agreement_refuses_anything_but_two_human_sheets(tmp_path):
    inst_path = _mini_instrument(tmp_path)
    qs = load_question_set(inst_path)
    ha = _fill(tmp_path, "A.json", {"draft_mid": [0.0, 1.0] * 10}, qs, annotator="A")

    import argparse

    ns = argparse.Namespace(
        instrument=str(inst_path), human=[str(ha)], model=[],
        intra="", intra_of="", n_boot=50, seed=0,
        absolute_floor=0.40, ceiling_fraction=0.80, kappa_h_min=0.40,
        min_questions=2, out=str(tmp_path / "r.json"),
    )
    with pytest.raises(SystemExit) as exc:
        vc.run_agreement(ns)
    assert "two-rater statistic" in str(exc.value)


def test_agreement_with_a_model_sheet_passes_a_reliable_question(tmp_path):
    """The whole §6.2 loop: κ_H ceiling, κ_M mean, lower-bound test, verdict."""
    inst_path = _mini_instrument(tmp_path)
    qs = load_question_set(inst_path)
    # 50 items; the model tracks both humans closely, the humans agree well.
    base = [float(i % 2) for i in range(50)]
    ha_col = list(base)
    hb_col = list(base)
    hb_col[3] = 1.0 - hb_col[3]
    hb_col[17] = 1.0 - hb_col[17]
    m_col = list(base)
    m_col[41] = 1.0 - m_col[41]

    ha = _fill(tmp_path, "A.json", {"draft_perfect": ha_col}, qs, annotator="A")
    hb = _fill(tmp_path, "B.json", {"draft_perfect": hb_col}, qs, annotator="B")
    mm = _fill(tmp_path, "M.json", {"draft_perfect": m_col}, qs, annotator="scorer")

    import argparse

    ns = argparse.Namespace(
        instrument=str(inst_path), human=[str(ha), str(hb)], model=[str(mm)],
        intra="", intra_of="", n_boot=800, seed=0,
        absolute_floor=0.40, ceiling_fraction=0.80, kappa_h_min=0.40,
        min_questions=1, out=str(tmp_path / "r.json"),
    )
    vc.run_agreement(ns)
    rep = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    d = {x["question_id"]: x for x in rep["decisions"]}["draft_perfect"]
    assert d["verdict"] == "pass"
    assert d["kappa_h"] > 0.8
    assert d["kappa_m_lower_ci"] is not None and d["kappa_m_lower_ci"] >= d["floor"]
    assert rep["summary"]["gate0_instrument_too_thin"] is False
    # The other two questions are unanswered on every sheet and must NOT be
    # silently treated as agreeing.
    assert {x["question_id"]: x["verdict"] for x in rep["decisions"]}["draft_mid"] == "undecidable"


def test_intra_rater_sheet_drives_the_gate(tmp_path, capsys):
    inst_path = _mini_instrument(tmp_path)
    qs = load_question_set(inst_path)
    base = [float(i % 2) for i in range(30)]
    ha = _fill(tmp_path, "A.json", {"draft_perfect": base}, qs, annotator="A")
    hb = _fill(tmp_path, "B.json", {"draft_perfect": base}, qs, annotator="B")
    m_col = list(base)
    rep_col = [1.0 - v for v in base]  # the scorer contradicts itself entirely
    mm = _fill(tmp_path, "M.json", {"draft_perfect": m_col}, qs, annotator="scorer")
    rp = _fill(tmp_path, "R.json", {"draft_perfect": rep_col}, qs, annotator="scorer")

    import argparse

    ns = argparse.Namespace(
        instrument=str(inst_path), human=[str(ha), str(hb)], model=[str(mm)],
        intra=str(rp), intra_of="scorer", n_boot=200, seed=0,
        absolute_floor=0.40, ceiling_fraction=0.80, kappa_h_min=0.40,
        min_questions=1, out=str(tmp_path / "r.json"),
    )
    vc.run_agreement(ns)
    rep = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert rep["intra_rater"]["halt"] is True
    assert rep["summary"]["gate0_halt_ruler_moving"] is True
    assert "GATE 0 STOP" in capsys.readouterr().out
