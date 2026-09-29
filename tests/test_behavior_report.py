"""The calibration report: a figure that cannot lie about the margin, and a
Markdown document where every absent number says it is absent.

The figure is tested structurally rather than by eye: well-formed XML, every
mark inside the viewBox, the q mark right of both floors when the floors are
smaller, and the degenerate case (no floor at all) drawn as a sentence rather
than as an axis.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nebulai.behavior import report as R  # noqa: E402

# A real report from this repo: `capability-control-2026-09-12`, trimmed to the
# keys the renderer reads. The numbers are the measured ones.
REAL = {
    "study_id": "capability-control-2026-09-12",
    "generated": "2026-09-12T01:50:12+00:00",
    "manifest_hash": "sha256:2a00d86d",
    "embedder": {"id": "sentence-transformers/all-MiniLM-L6-v2", "strict_ok": True},
    "mmd_bandwidth": 0.8005657371248865,
    "n_cues": 2,
    "n_trials": 96,
    "p_floor_design": 1.371867334094234e-12,
    "p_floor_empirical_worst": 1.0,
    "p_floor_empirical_best": 1.417233560090703e-05,
    "p_floor_per_cue": {
        "freedom": 1.417233560090703e-05,
        "water": 1.417233560090703e-05,
        "daddy": 1.0,
        "petrichor": 1.0,
    },
    "p_floor_cues_total": 4,
    "p_floor_cues_above_q": ["daddy", "petrichor"],
    "q_threshold": 0.05,
    "p_floor_clears_q": True,
    "p_floor_clears_q_all_cues": False,
    "aa_controls": [
        {
            "status": "measured",
            "model_key": "cap_xl",
            "n_splits": 20,
            "delta_hat_mean": 0.0170,
            "delta_hat_p95": 0.1249,
            "false_positive_rate_at_0.05": 0.1,
            "cue": "freedom",
        }
    ],
    "aa_false_positive_rate_mean": 0.025,
    "positive_control": {
        "threshold": 0.5,
        "pass_rate": {},
        "n_trials": {},
        "passed": {},
        "per_cue": {},
    },
    "canary": {
        "status": "measured",
        "drift": {
            "cap_small": {
                "consecutive_block_cosine": [0.1616, 0.1831, 0.1485],
                "min": 0.1485,
                "flag": True,
            }
        },
    },
    "rank_stability": {"status": "missing", "reason": "fewer than 5 comparable cues"},
}


# ── the figure ─────────────────────────────────────────────────────────────


def _svg(**kw):
    kw.setdefault("p_floor_design", REAL["p_floor_design"])
    kw.setdefault("p_floor_best", REAL["p_floor_empirical_best"])
    kw.setdefault("p_floor_worst", REAL["p_floor_empirical_worst"])
    kw.setdefault("q", 0.05)
    kw.setdefault("n_cues", REAL["p_floor_cues_total"])
    return R.p_floor_figure(**kw)


def _cx(svg: str) -> list[float]:
    root = ET.fromstring(svg)
    return [float(c.get("cx")) for c in root.iter() if c.tag.endswith("circle")]


def test_the_figure_is_well_formed_xml():
    root = ET.fromstring(_svg())
    assert root.tag.endswith("svg")
    assert root.get("role") == "img"
    assert root.get("aria-label"), "a figure in a report needs a label read aloud"


def test_everything_drawn_lands_inside_the_viewbox():
    root = ET.fromstring(_svg())
    _, _, w, h = (float(v) for v in root.get("viewBox").split())
    for el in root.iter():
        if el.tag.endswith("circle"):
            assert 0 <= float(el.get("cx")) <= w
            assert 0 <= float(el.get("cy")) <= h
        if el.tag.endswith("line"):
            for k in ("x1", "x2"):
                assert 0 <= float(el.get(k)) <= w
            for k in ("y1", "y2"):
                assert 0 <= float(el.get(k)) <= h


def test_the_collected_floors_are_a_range_not_a_point():
    """A study holding one six-trial cue beside two full-depth cues has floors
    orders of magnitude apart. Drawing one mark would have to pick which to
    hide."""
    root = ET.fromstring(_svg())
    thick = [
        el
        for el in root.iter()
        if el.tag.endswith("line") and float(el.get("stroke-width", 0)) >= 6
    ]
    assert len(thick) == 1, "the collected range is one bar"
    bar = thick[0]
    assert float(bar.get("x1")) < float(bar.get("x2"))


def test_q_sits_right_of_the_best_collected_floor():
    """On a log p axis the ordering IS the conclusion: q rendered left of a
    floor that is numerically smaller would invert it."""
    root = ET.fromstring(_svg())
    bar = next(
        el
        for el in root.iter()
        if el.tag.endswith("line") and float(el.get("stroke-width", 0)) >= 6
    )
    design, q = _cx(_svg())  # lane 0 and lane 2 are the only circles here
    assert design < float(bar.get("x1")) < q


def test_the_marks_carry_their_own_numerals():
    """A reader must never have to estimate a value from a pixel position."""
    svg = _svg()
    assert "1.37e-12" in svg
    assert "1.42e-05" in svg, "the best collected floor, printed"
    assert "over 4 cues" in svg, "and how many cues the range covers"
    assert "0.05</tspan>" in svg


def test_a_wider_q_moves_only_q():
    xa = _cx(_svg(q=0.05))
    xb = _cx(_svg(q=0.2))
    assert xa[0] == xb[0], "the design floor does not depend on q"
    assert xb[-1] > xa[-1]


def test_one_collected_floor_collapses_the_range_to_a_point():
    """Before a second cue exists there is no range, and a bar of zero width
    drawn with end caps would look like a measurement of spread."""
    svg = _svg(p_floor_best=1e-5, p_floor_worst=1e-5, n_cues=1)
    root = ET.fromstring(svg)
    assert not [
        el
        for el in root.iter()
        if el.tag.endswith("line") and float(el.get("stroke-width", 0)) >= 6
    ]
    assert len(_cx(svg)) == 3, "design, the single collected floor, and q"


def test_no_floor_at_all_is_a_sentence_not_an_axis():
    """A study where no cue has valid trials in two blocks has no floor. An axis
    drawn from nothing would be a picture of an assumption."""
    svg = R.p_floor_figure(
        p_floor_design=None, p_floor_best=None, p_floor_worst=None, q=0.05
    )
    assert not _cx(svg)
    assert "not measured" in svg


def test_a_zero_floor_is_treated_as_absent_not_as_the_left_edge():
    """`p_floor_empirical_*` is 0.0 in the JSON when nothing was measurable, and
    log10(0) is not a coordinate."""
    svg = R.p_floor_figure(
        p_floor_design=1e-9, p_floor_best=0.0, p_floor_worst=0.0, q=0.05
    )
    assert len(_cx(svg)) == 2, "the design floor and q, and no invented third"


# ── the document ───────────────────────────────────────────────────────────


def test_the_report_links_the_figure_it_is_shipped_with():
    md = R.calibration_markdown(REAL)
    assert f"]({R.FIGURE_NAME})" in md


def test_the_margin_is_stated_as_a_ratio_not_left_to_the_reader():
    md = R.calibration_markdown(REAL)
    # 0.05 / 1.417e-5 = 3528, computed from the BEST collected floor
    assert "3,528× below q" in md
    assert "headroom only" in md


def test_one_shallow_cue_does_not_condemn_the_whole_study():
    """This is the bug the per-cue floor fixed. Two cues were collected to full
    depth and two to six trials; reporting only the worst floor made the
    calibration read as "no cue can ever be significant", which was false for
    the two cues the study actually analyzed."""
    md = R.calibration_markdown(REAL)
    assert "makes *that cue* unresolvable" in md
    assert "| at least one cue clears q | **yes** |" in md
    assert "| every cue clears q | **no** |" in md


def test_the_unresolvable_cues_are_named_not_counted_away():
    md = R.calibration_markdown(REAL)
    assert "2 of 4 collected cues cannot clear q" in md
    assert "`daddy`" in md and "`petrichor`" in md


def test_the_per_cue_floor_table_marks_each_cue_yes_or_no():
    md = R.calibration_markdown(REAL)
    block = md.split("| cue | floor as collected | can clear q |")[1]
    assert "| freedom | 1.41723e-05 | yes |" in block
    assert "| daddy | 1 | no |" in block


def test_no_cue_clearing_q_says_the_study_cannot_resolve_anything():
    bad = dict(
        REAL,
        p_floor_empirical_best=0.2,
        p_floor_empirical_worst=0.9,
        p_floor_per_cue={"daddy": 0.2, "water": 0.9},
        p_floor_cues_total=2,
        p_floor_cues_above_q=["daddy", "water"],
        p_floor_clears_q=False,
        p_floor_clears_q_all_cues=False,
    )
    md = R.calibration_markdown(bad)
    assert "No cue's floor clears q" in md
    assert "a larger q is not" in md
    assert "below q**" not in md, "no margin may be claimed when there is none"


def test_an_unmeasured_positive_control_is_not_a_failed_one():
    md = R.calibration_markdown(REAL)
    block = md.split("## Positive control")[1].split("##")[0]
    assert "Not measured in this study" in block
    assert "FAIL" not in block, "absent is not failed"
    assert "not waived by its absence" in block


def test_a_measured_positive_control_prints_per_arm_and_per_cue():
    rep = dict(
        REAL,
        positive_control={
            "threshold": 0.5,
            "pass_rate": {"cap_small": 0.25, "cap_xl": 0.75},
            "n_trials": {"cap_small": 48, "cap_xl": 48},
            "passed": {"cap_small": False, "cap_xl": True},
            "per_cue": {"hot": {"cap_small": 0.2, "cap_xl": 0.8}},
        },
    )
    md = R.calibration_markdown(rep)
    block = md.split("## Positive control")[1].split("## Canary")[0]
    assert "| `cap_small` | 48 | 0.250 | FAIL |" in block
    assert "| `cap_xl` | 48 | 0.750 | PASS |" in block
    assert "| hot | 0.200 | 0.800 |" in block


def test_an_unmeasured_rank_stability_says_the_ranking_is_unvalidated():
    md = R.calibration_markdown(REAL)
    block = md.split("## Rank stability")[1]
    assert "fewer than 5 comparable cues" in block
    assert "unvalidated" in block


def test_a_raised_canary_flag_is_not_reported_as_invalidating():
    md = R.calibration_markdown(REAL)
    block = md.split("## Canary drift")[1].split("## Rank")[0]
    assert "RAISED" in block
    assert "does **not** invalidate" in block


def test_nothing_in_the_report_ranks_a_model():
    md = R.calibration_markdown(REAL).lower()
    for banned in ["leaderboard", "outperform", "the winner", "beats ", "better model"]:
        assert banned not in md


def test_an_empty_report_renders_without_inventing_numbers():
    """A calibration that measured nothing still has to produce a document, and
    every slot in it has to read as absent."""
    md = R.calibration_markdown({"q_threshold": 0.05})
    assert "not measured" in md
    assert "No cue's floor clears q" in md
    for bogus in ["0.0000×", "pass rate of 0"]:
        assert bogus not in md.replace("not a pass rate of 0", "")


@pytest.mark.parametrize(
    "key", ["p_floor_design", "p_floor_empirical_worst", "mmd_bandwidth"]
)
def test_a_missing_number_prints_as_not_measured_rather_than_zero(key):
    rep = dict(REAL)
    rep[key] = None
    md = R.calibration_markdown(rep)
    assert "not measured" in md
    assert "| 0.0000 |" not in md
