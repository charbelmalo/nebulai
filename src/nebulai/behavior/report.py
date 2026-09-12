"""The calibration report as something a person reads (§6.7).

`calibration.json` holds every number this stage measured, and nothing in it is
wrong. It is also the wrong artifact to hand a reader who has to decide whether
the design can resolve anything at all, because the one comparison that decides
that — the permutation test's **p-floor** against the study's **q threshold** —
is two sibling keys in a 200-line file, and the difference between them can be
seven orders of magnitude. Seven orders of magnitude is exactly the kind of
quantity a column of decimals hides and a log axis shows.

So this module renders the same JSON as Markdown plus one SVG figure. It
computes nothing: every value here is read out of the report dict, and anything
the calibration did not measure prints as "not measured" with its own reason
rather than as a zero, a dash, or a silently-dropped section.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["p_floor_figure", "calibration_markdown", "FIGURE_NAME"]

#: The figure's filename next to the report, so the Markdown link and the file
#: written to disk cannot disagree.
FIGURE_NAME = "calibration_p_floor.svg"

_W = 760
_H = 210
_L = 70  # left margin
_R = 36  # right margin
_AXIS_Y = 126


def p_floor_figure(
    *,
    p_floor_design: float | None,
    p_floor_best: float | None,
    p_floor_worst: float | None,
    q: float,
    n_cues: int | None = None,
) -> str:
    """A log10 p axis carrying the design floor, the collected range, and q.

    The claim the figure supports is narrow and worth stating exactly: a
    permutation test over a finite set of within-block relabelings cannot report
    a p below ``1 / (number of distinct relabelings)``. A cue whose floor sits
    at or above ``q`` cannot be called significant however large its effect, and
    that is a fact about how deeply *that cue* was collected, not about any
    model.

    Which is why the collected floors are drawn as a **range** rather than as
    one number. A study part-way through collection holds cues six trials deep
    beside cues at full depth, and their floors differ by orders of magnitude;
    plotting only the worst would condemn the whole study, and plotting only the
    best would hide the cues that cannot answer.
    """
    floors = [
        x for x in (p_floor_design, p_floor_best, p_floor_worst)
        if x is not None and x > 0
    ]
    if not floors:
        # No valid blocks means no floor, and an axis drawn from nothing would
        # be a picture of an assumption.
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_W} 70" '
            f'role="img" aria-label="p-floor not measured">'
            f'<text x="16" y="40" font-family="ui-sans-serif,system-ui,sans-serif" '
            f'font-size="14">p-floor not measured: no cue has valid trials in '
            f"two or more blocks.</text></svg>"
        )
    lo_dec = math.floor(math.log10(min(floors))) - 1
    hi_dec = 0  # p is a probability; the axis always ends at 1
    span = hi_dec - lo_dec

    def x_of(p: float) -> float:
        t = (math.log10(p) - lo_dec) / span
        return _L + max(0.0, min(1.0, t)) * (_W - _L - _R)

    font = "ui-sans-serif,system-ui,-apple-system,sans-serif"
    mono = "ui-monospace,SFMono-Regular,Menlo,monospace"
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_W} {_H}" '
        f'role="img" aria-label="permutation p-floor against the q threshold, '
        f'log scale">',
        f"<style>text{{font-family:{font};fill:#1b1b1f}} "
        f".m{{font-family:{mono}}} .sub{{fill:#55555f}}</style>",
        f'<rect x="0" y="0" width="{_W}" height="{_H}" fill="#fbfbfd"/>',
    ]
    # The region a p-value can land in and still clear q.
    qx = x_of(q)
    parts.append(
        f'<rect x="{_L:.1f}" y="{_AXIS_Y - 30}" width="{qx - _L:.1f}" height="30" '
        f'fill="#e8f1e8"/>'
    )
    parts.append(
        f'<text x="{_L + 6:.1f}" y="{_AXIS_Y - 11}" font-size="11" class="sub">'
        f"p &lt; q: a cue can be called significant here</text>"
    )
    # Axis and decade ticks.
    parts.append(
        f'<line x1="{_L}" y1="{_AXIS_Y}" x2="{_W - _R}" y2="{_AXIS_Y}" '
        f'stroke="#1b1b1f" stroke-width="1.2"/>'
    )
    step = 1 if span <= 14 else 2
    for d in range(lo_dec, hi_dec + 1, step):
        x = x_of(10.0**d)
        parts.append(
            f'<line x1="{x:.1f}" y1="{_AXIS_Y}" x2="{x:.1f}" y2="{_AXIS_Y + 5}" '
            f'stroke="#1b1b1f" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{_AXIS_Y + 19}" font-size="10.5" class="m sub" '
            f'text-anchor="middle">1e{d}</text>'
        )
    parts.append(
        f'<text x="{(_L + _W - _R) / 2:.0f}" y="{_AXIS_Y + 37}" font-size="11" '
        f'class="sub" text-anchor="middle">permutation p (log scale)</text>'
    )

    blue, red = "#1b5e9c", "#b3261e"

    def label(x: float, y: float, text: str, num: str, col: str) -> None:
        anchor = "end" if x > _W - _R - 150 else "start"
        dx = -9 if anchor == "end" else 9
        parts.append(
            f'<text x="{x + dx:.1f}" y="{y + 4}" font-size="12" fill="{col}" '
            f'text-anchor="{anchor}">{text} <tspan class="m">{num}</tspan></text>'
        )

    # Lane 0: the design floor, i.e. what the block structure promises at full
    # depth. Lane 1: the range the collected blocks really support. Lane 2: q.
    if p_floor_design is not None and p_floor_design > 0:
        x = x_of(p_floor_design)
        parts.append(
            f'<line x1="{x:.1f}" y1="30" x2="{x:.1f}" y2="{_AXIS_Y}" '
            f'stroke="{blue}" stroke-width="1.4" stroke-dasharray="3 3"/>'
        )
        parts.append(f'<circle cx="{x:.1f}" cy="30" r="4" fill="{blue}"/>')
        label(x, 30, "design floor", f"{p_floor_design:.3g}", blue)

    lo = p_floor_best if (p_floor_best or 0) > 0 else None
    hi = p_floor_worst if (p_floor_worst or 0) > 0 else None
    if lo is not None or hi is not None:
        a = x_of(lo if lo is not None else hi)  # type: ignore[arg-type]
        b = x_of(hi if hi is not None else lo)  # type: ignore[arg-type]
        y = 58
        if abs(b - a) < 1.5:
            parts.append(f'<circle cx="{a:.1f}" cy="{y}" r="4" fill="{blue}"/>')
        else:
            parts.append(
                f'<line x1="{a:.1f}" y1="{y}" x2="{b:.1f}" y2="{y}" '
                f'stroke="{blue}" stroke-width="7" stroke-linecap="butt" '
                f'opacity="0.75"/>'
            )
            for e in (a, b):
                parts.append(
                    f'<line x1="{e:.1f}" y1="{y - 7}" x2="{e:.1f}" y2="{y + 7}" '
                    f'stroke="{blue}" stroke-width="1.6"/>'
                )
        cues = f" over {n_cues} cues" if n_cues else ""
        num = (
            f"{lo:.3g}–{hi:.3g}"
            if lo is not None and hi is not None and abs(b - a) >= 1.5
            else f"{(lo if lo is not None else hi):.3g}"  # type: ignore[union-attr]
        )
        label(max(a, b), y, f"floors as collected{cues}", num, blue)

    xq = x_of(q)
    parts.append(
        f'<line x1="{xq:.1f}" y1="86" x2="{xq:.1f}" y2="{_AXIS_Y}" '
        f'stroke="{red}" stroke-width="1.4" stroke-dasharray="3 3"/>'
    )
    parts.append(f'<circle cx="{xq:.1f}" cy="86" r="4" fill="{red}"/>')
    label(xq, 86, "q threshold", f"{q:g}", red)
    parts.append("</svg>")
    return "".join(parts)


def _fmt(x: Any, spec: str = ".4f") -> str:
    """A number, or the words for its absence. Never a 0 standing in for one."""
    if x is None:
        return "not measured"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        return format(x, spec)
    return str(x)


def calibration_markdown(report: dict[str, Any], *, figure: str = FIGURE_NAME) -> str:
    """Render `calibration.json` as the §6.7 report.

    Pure: no filesystem, no recomputation. Every section that has no data says
    which data it lacks, because a calibration report with a quietly absent
    section is indistinguishable from one that passed.
    """
    L: list[str] = []
    A = L.append
    sid = report.get("study_id", "?")
    A(f"# Calibration report — `{sid}`")
    A("")
    emb = report.get("embedder") or {}
    A(
        f"Generated {report.get('generated', '?')} from "
        f"**{report.get('n_trials', 0)} discovery trials** over "
        f"**{report.get('n_cues', 0)} cues** that have at least one valid trial. "
        f"That is the calibration's own denominator, not the study's: the "
        f"manifest's full cue list is larger wherever collection is unfinished, "
        f"and `behavior.json`'s coverage block is where that gap is stated."
    )
    A("")
    A(f"- manifest `{report.get('manifest_hash') or 'unfrozen'}`")
    A(
        f"- embedder `{emb.get('id', '?')}` — strict source: "
        f"**{_fmt(emb.get('strict_ok'))}**"
    )
    A(
        f"- MMD bandwidth, frozen from this stage on: "
        f"`{_fmt(report.get('mmd_bandwidth'), '.6f')}`"
    )
    A("")
    A("## The p-floor against q (§6.7.1)")
    A("")
    A(f"![permutation p-floor against the q threshold]({figure})")
    A("")
    A(
        "A permutation test only ever compares the observed statistic against "
        "relabelings it can actually enumerate, so the smallest p it can report "
        "is `1 / (number of distinct within-block relabelings)`. **A cue whose "
        "floor sits at or above `q` cannot be called significant no matter how "
        "large its effect** — and that is a property of how deeply that cue was "
        "collected, not of any model. The design floor below is what the "
        "manifest's intended block sizes promise at full depth; the collected "
        "floors are what the blocks with valid trials in them actually support, "
        "and those are the numbers that govern."
    )
    A("")
    best = report.get("p_floor_empirical_best")
    worst = report.get("p_floor_empirical_worst")
    q = report.get("q_threshold")
    above = report.get("p_floor_cues_above_q") or []
    total = report.get("p_floor_cues_total")
    A("| quantity | value |")
    A("|---|---|")
    A(f"| design p-floor | {_fmt(report.get('p_floor_design'), '.6g')} |")
    A(f"| best floor as collected | {_fmt(best or None, '.6g')} |")
    A(f"| worst floor as collected | {_fmt(worst or None, '.6g')} |")
    A(
        f"| cues with a collected floor | "
        f"{total if total is not None else 'not measured'} |"
    )
    A(
        f"| cues whose floor cannot clear q | "
        f"{len(above) if total is not None else 'not measured'} |"
    )
    A(f"| q threshold | {_fmt(q, '.4g')} |")
    A(f"| at least one cue clears q | **{_fmt(report.get('p_floor_clears_q'))}** |")
    A(f"| every cue clears q | **{_fmt(report.get('p_floor_clears_q_all_cues'))}** |")
    A("")
    A(
        "**The floor is per cue, because the block structure is.** A cue "
        "collected to six trials and a cue collected to its full depth have "
        "floors orders of magnitude apart, so one partially-collected cue does "
        "not make the study unresolvable — it makes *that cue* unresolvable."
    )
    A("")
    if report.get("p_floor_clears_q") and best and q:
        A(
            f"The best collected floor is **{q / best:,.0f}× below q**, so the "
            f"test has room to resolve a significant cue. That is a statement "
            f"about headroom only: it says nothing about whether any cue in this "
            f"study actually did."
        )
    else:
        A(
            "**No cue's floor clears q.** Every per-cue p in this study is "
            "bounded below by a value that cannot beat the threshold, so no "
            "significance claim from it is admissible. More blocks or more "
            "trials per block are the fix; a larger q is not."
        )
    if above:
        A("")
        A(
            f"**{len(above)} of {total} collected cues cannot clear q at the "
            f"depth they were collected to**, and they are named rather than "
            f"counted away, because a reader looking at one of them is entitled "
            f"to know its p could not have come out small: "
            + ", ".join("`" + c + "`" for c in above[:20])
            + (" …" if len(above) > 20 else "")
            + "."
        )
    per = report.get("p_floor_per_cue") or {}
    if per and len(per) <= 40:
        A("")
        A("| cue | floor as collected | can clear q |")
        A("|---|---:|---|")
        for c in sorted(per, key=lambda k: (per[k], k)):
            A(
                f"| {c} | {per[c]:.6g} | "
                f"{'yes' if q is not None and per[c] < q else 'no'} |"
            )
    A("")
    A("## A/A control (§6.7.3)")
    A("")
    aa = report.get("aa_controls") or []
    fpr = report.get("aa_false_positive_rate_mean")
    if not aa:
        A("**Not measured** — no arm had enough valid trials to split in half.")
    else:
        A(
            "One arm split against itself. Δ̂ should sit at zero and the "
            "false-positive rate at or below the nominal 0.05; anything higher "
            "means the pipeline finds differences where there is only sampling."
        )
        A("")
        A("| cue | arm | splits | Δ̂ mean | Δ̂ p95 | FPR at 0.05 |")
        A("|---|---|---:|---:|---:|---:|")
        for r in aa:
            A(
                f"| {r.get('cue', '?')} | `{r.get('model_key', '?')}` | "
                f"{r.get('n_splits', 0)} | {_fmt(r.get('delta_hat_mean'))} | "
                f"{_fmt(r.get('delta_hat_p95'))} | "
                f"{_fmt(r.get('false_positive_rate_at_0.05'), '.3f')} |"
            )
        A("")
        A(
            f"Mean false-positive rate across cells: **{_fmt(fpr, '.3f')}** "
            f"(nominal 0.05)."
        )
    A("")
    A("## Positive control (§6.7.2)")
    A("")
    pc = report.get("positive_control") or {}
    rates = pc.get("pass_rate") or {}
    if not rates:
        A(
            "**Not measured in this study.** None of the §6.7.2 control cues "
            "(`hot`, `salt`, `cat`, …) has valid trials here, so this artifact "
            "carries no evidence that either arm produces association data at "
            "all. The gate is not waived by its absence — it runs as its own "
            "study, and that report is the one to read before any Δ̂ from these "
            "arms is believed."
        )
    else:
        A(
            f"Share of valid control trials that recovered an expected "
            f"associate, per arm, against a threshold of "
            f"**{_fmt(pc.get('threshold'), '.2f')}**. An arm below the threshold "
            f"is not producing association data, and no divergence number "
            f"computed from its outputs means anything."
        )
        A("")
        A("| arm | trials | pass rate | gate |")
        A("|---|---:|---:|---|")
        for k in sorted(rates):
            passed = (pc.get("passed") or {}).get(k)
            A(
                f"| `{k}` | {(pc.get('n_trials') or {}).get(k, 0)} | "
                f"{_fmt(rates[k], '.3f')} | {'PASS' if passed else 'FAIL'} |"
            )
        per_cue = pc.get("per_cue") or {}
        if per_cue:
            A("")
            arms = sorted(rates)
            A("| cue | " + " | ".join(f"`{a}`" for a in arms) + " |")
            A("|---|" + "---:|" * len(arms))
            for cue in sorted(per_cue):
                row = per_cue[cue]
                A(
                    f"| {cue} | "
                    + " | ".join(_fmt(row.get(a), ".3f") for a in arms)
                    + " |"
                )
    A("")
    A("## Canary drift (§5.5.1)")
    A("")
    can = report.get("canary") or {}
    if can.get("status") != "measured":
        A(f"**Not measured** — {can.get('reason', 'no canary trials recorded')}.")
    else:
        A(
            "Cosine similarity between consecutive time blocks' canary outputs, "
            "per arm. This is behaviour, not a self-reported label: it catches a "
            "serving change that leaves `system_fingerprint` untouched. A raised "
            "flag does **not** invalidate the study — it means the within-block "
            "permutation structure is carrying real work, and that any "
            "cross-block comparison has to be quoted with the flag beside it."
        )
        A("")
        A("| arm | consecutive-block cosine | min | flag |")
        A("|---|---|---:|---|")
        for k in sorted(can.get("drift") or {}):
            d = can["drift"][k]
            seq = ", ".join(f"{v:.4f}" for v in d.get("consecutive_block_cosine", []))
            A(
                f"| `{k}` | {seq or 'not measured'} | {_fmt(d.get('min'))} | "
                f"{'RAISED' if d.get('flag') else 'clear'} |"
            )
    A("")
    A("## Rank stability (§6.7)")
    A("")
    rs = report.get("rank_stability") or {}
    if rs.get("status") != "measured":
        A(
            f"**Not measured** — {rs.get('reason', 'no reason recorded')}. The "
            f"ranked cue list is therefore unvalidated at this trial count: it "
            f"may be stable and it may be noise, and this report does not say "
            f"which."
        )
    else:
        A(
            f"Spearman correlation of per-cue Δ̂ between two disjoint halves of "
            f"the trials: **{_fmt(rs.get('spearman'), '.4f')}** over "
            f"{rs.get('n_cues', '?')} cues (p {_fmt(rs.get('p_value'), '.4g')}). "
            f"This is the rank-stability-versus-trial-count curve at the one "
            f"count that was collected."
        )
    A("")
    A("## What this report does not establish")
    A("")
    A(
        "- Nothing here describes any model's internals, and nothing here ranks "
        "models. Calibration measures the **instrument**."
    )
    A(
        "- Clearing the p-floor is headroom, not a result. A study can clear it "
        "and still find nothing."
    )
    A(
        "- Every absent number above is absent, not zero. An arm with no control "
        "trials has an unknown pass rate, not a pass rate of 0."
    )
    A("")
    return "\n".join(L)
