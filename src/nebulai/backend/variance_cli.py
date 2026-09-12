"""`nebulai variance` — the W1 annotation kit and the price re-verification.

`docs/GENERATIVE-VARIANCE-PLAN.md` §10 Phase 0 has nine items. Items 1–4 are
explicitly *human* work — authoring ~30 architecture questions with rubrics,
authoring ~30 prompts, generating a pilot gold set, and having two annotators
independently label 50 stories × 30 questions. Tooling cannot do any of them,
and a question set invented by the code that validates it would be the tooling
grading its own homework (`backend/instrument.py`'s module docstring).

What tooling *can* do is remove every non-scientific obstacle in front of that
human work, which is what this module is:

* ``annotate-sheet`` emits the 50 × 30 sheets — one per annotator, plus the
  §6.5 blind-repeat block — as CSV a human can fill in a spreadsheet and as
  JSON the analysis reads back. It stamps the instrument hash into every sheet,
  so a sheet filled against one version of the questions can never be silently
  pooled with a sheet filled against another.
* ``agreement`` takes the filled sheets and produces §6.2–§6.5 in full: κ_H,
  κ_M, Krippendorff's α, PABAK, Gwet's AC1 and raw percent agreement **per
  question** with bootstrap CIs, the floor decision on the *lower bound*, and
  the intra-rater halt check.
* ``prices`` is item 9 — re-verifying `corpus.py`'s prices against
  OpenRouter's public catalogue, which needs no key.

Two refusals are deliberate and must stay:

* an **unfrozen** instrument produces sheets, but every one of them is stamped
  ``"instrument_frozen": false`` and carries the draft warning, because §6.5's
  one-way door is only safe if it is obvious which side of it you are on;
* the sheets are emitted with **empty answers**. Nothing here fills a sheet,
  including for testing. A synthetic sheet is a fixture and lives in `tests/`.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .agreement import (
    AGREEMENT_VERSION,
    DEFAULT_ABSOLUTE_FLOOR,
    DEFAULT_CEILING_FRACTION,
    DEFAULT_KAPPA_H_MIN,
    AgreementError,
    Interval,
    bootstrap_ci,
    cohen_kappa,
    floor_decision,
    intra_rater_check,
    krippendorff_alpha,
    mean_ignoring_missing,
    question_agreement,
    scale_categories,
)
from .instrument import Question, QuestionSet, load_question_set

SHEET_FORMAT = "nebulai-annotation-sheet"
SHEET_VERSION = 1

#: The columns a human actually edits are `answer` and `note`. Everything else
#: is there so a sheet opened three weeks later still says what it is.
CSV_COLUMNS = [
    "story_id",
    "question_id",
    "kind",
    "scale_lo",
    "scale_hi",
    "answer",
    "note",
    "question_text",
    "rubric",
]


# --------------------------------------------------------------------------
# sheets
# --------------------------------------------------------------------------


def _story_ids(args: argparse.Namespace) -> tuple[list[str], str]:
    """Where the gold-set story ids come from, and an honest label for it."""
    if args.stories:
        p = Path(args.stories)
        if p.is_dir():
            ids = sorted(f.stem for f in p.glob("*.json"))
            if not ids:
                raise SystemExit(f"no *.json stories under {p}")
            return ids, f"directory:{p}"
        d = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(d, dict):
            ids = sorted(d.get("stories", d).keys())
        else:
            ids = [str(x.get("id", i)) for i, x in enumerate(d)]
        if not ids:
            raise SystemExit(f"{p} contains no stories")
        return ids, f"file:{p}"
    # §10 item 3 (pilot stories) needs a paid generator and no key exists here,
    # so placeholder ids let items 4-6 be rehearsed end to end. The label says
    # so in the artifact; nobody should discover it by reading the ids.
    n = args.n_stories
    return [f"placeholder_{i:03d}" for i in range(1, n + 1)], "placeholder"


def build_sheet(
    qs: QuestionSet,
    story_ids: list[str],
    annotator: str,
    *,
    block: str,
    story_source: str,
) -> dict[str, Any]:
    return {
        "format": SHEET_FORMAT,
        "version": SHEET_VERSION,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "annotator": annotator,
        "block": block,
        "instrument": {
            "name": qs.name,
            # The *current* hash either way. A draft's hash is not an identity
            # promise — it is a fingerprint that makes a mid-labelling edit
            # detectable, which is the failure this whole file exists to avoid.
            "hash": qs.frozen_hash or qs.compute_hash(),
            "frozen": qs.is_frozen,
            "frozen_at": qs.frozen_at,
            "question_ids": [q.id for q in qs.questions],
        },
        "warning": (
            ""
            if qs.is_frozen
            else (
                "THIS INSTRUMENT IS A DRAFT. Sheets filled against a draft are "
                "pilot data: Sec 6.5 makes the question set a one-way door, and "
                "anything scored before the freeze must be re-scored after it."
            )
        ),
        "stories": {"source": story_source, "n": len(story_ids), "ids": story_ids},
        # Empty on purpose. Nothing in this file answers a question.
        "answers": {s: {q.id: None for q in qs.questions} for s in story_ids},
    }


def write_sheet_csv(path: Path, qs: QuestionSet, sheet: dict[str, Any]) -> Path:
    by_id = {q.id: q for q in qs.questions}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for story in sheet["stories"]["ids"]:
            for qid in sheet["instrument"]["question_ids"]:
                q: Question = by_id[qid]
                w.writerow(
                    {
                        "story_id": story,
                        "question_id": q.id,
                        "kind": q.kind,
                        "scale_lo": q.lo,
                        "scale_hi": q.hi,
                        # Left blank = declined. Blank is NOT 0 and the analysis
                        # never treats it as one.
                        "answer": "",
                        "note": "",
                        "question_text": q.text,
                        "rubric": q.note,
                    }
                )
    return path


def load_sheet(path: str | Path) -> dict[str, Any]:
    """Read a filled sheet back, from either format the kit emits."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        answers: dict[str, dict[str, float | None]] = {}
        with p.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                v = (row.get("answer") or "").strip()
                answers.setdefault(row["story_id"], {})[row["question_id"]] = (
                    None if v == "" else float(v)
                )
        return {
            "format": SHEET_FORMAT,
            "version": SHEET_VERSION,
            "annotator": p.stem,
            "block": "primary",
            "instrument": {"name": "", "hash": "", "frozen": False, "question_ids": []},
            "stories": {"source": f"csv:{p}", "n": len(answers), "ids": sorted(answers)},
            "answers": answers,
        }
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("format") != SHEET_FORMAT:
        raise AgreementError(
            f"{p} is not a {SHEET_FORMAT} (found format={d.get('format')!r})"
        )
    return d


def _column(sheet: dict[str, Any], qid: str) -> dict[str, float | None]:
    """One question's column: story_id -> answer, missing preserved as None."""
    return {s: a.get(qid) for s, a in sheet["answers"].items()}


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def run_annotate_sheet(args: argparse.Namespace) -> None:
    qs = load_question_set(args.instrument)
    ids, source = _story_ids(args)
    annotators = [a.strip() for a in args.annotators.split(",") if a.strip()]
    if len(annotators) < 2:
        raise SystemExit(
            "Sec 6.2 step 1 needs TWO independent annotators; kappa_H is the "
            "ceiling every floor is measured against and one annotator cannot "
            "produce it."
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for a in annotators:
        sheet = build_sheet(qs, ids, a, block="primary", story_source=source)
        j = out / f"sheet_{a}.json"
        j.write_text(json.dumps(sheet, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(str(j))
        written.append(str(write_sheet_csv(out / f"sheet_{a}.csv", qs, sheet)))

    if args.repeat_n:
        # §6.5: the repeat subset is a BLIND re-score of stories already seen.
        # Held-out by position, not by a fresh random draw, so the same subset
        # is reproducible from the manifest alone.
        sub = ids[: args.repeat_n]
        for a in annotators:
            sheet = build_sheet(
                qs, sub, a, block="repeat", story_source=source + "|repeat-subset"
            )
            j = out / f"repeat_{a}.json"
            j.write_text(json.dumps(sheet, indent=2, ensure_ascii=False), encoding="utf-8")
            written.append(str(j))
            written.append(str(write_sheet_csv(out / f"repeat_{a}.csv", qs, sheet)))

    meta = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "instrument": args.instrument,
        "instrument_hash": qs.frozen_hash or qs.compute_hash(),
        "instrument_frozen": qs.is_frozen,
        "n_questions": len(qs.questions),
        "n_stories": len(ids),
        "story_source": source,
        "annotators": annotators,
        "repeat_n": args.repeat_n,
        "files": written,
    }
    (out / "sheets.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"wrote {len(written)} sheet files to {out}  "
        f"({len(annotators)} annotators x {len(ids)} stories x "
        f"{len(qs.questions)} questions"
        + (f", repeat block {args.repeat_n} stories" if args.repeat_n else "")
        + ")"
    )
    if not qs.is_frozen:
        print(
            "NOTE: the instrument is a DRAFT. Sheets filled against it are pilot "
            "data and must be re-scored after the Sec 6.5 freeze."
        )
    if source == "placeholder":
        print(
            "NOTE: story ids are PLACEHOLDERS. Sec 10 item 3 (the pilot gold "
            "set) needs a paid generator, and these sheets carry no stories."
        )


def _pair_interval(
    q: Question,
    pairs: list[tuple[dict[str, float | None], dict[str, float | None]]],
    *,
    n_boot: int,
    seed: int,
) -> Interval | None:
    """Bootstrap CI on the MEAN κ over several rater pairs.

    §6.2 step 3 asks for "mean human–model κ per question", and step 4 tests its
    lower bound. Averaging the pairs' separate intervals would be wrong — the
    pairs share items, so the resample must be over items *once*, with every
    pair recomputed inside the same draw.
    """
    cats = scale_categories(q.kind, q.lo, q.hi)
    if cats is None or not pairs:
        return None
    items = sorted({k for a, b in pairs for k in (set(a) | set(b))})
    keep = [
        it
        for it in items
        if all(a.get(it) is not None and b.get(it) is not None for a, b in pairs)
    ]
    if len(keep) < 2:
        return Interval(None, None, None, 0, len(keep), "fewer than 2 complete items")

    cols = [([float(a[it]) for it in keep], [float(b[it]) for it in keep]) for a, b in pairs]

    def stat(ix_a: list[float], _ignored: list[float]) -> float:
        # `bootstrap_ci` resamples the index vector it is handed; the pairs are
        # re-read through that same index so every pair sees the same draw.
        idx = [int(v) for v in ix_a]
        vals = [
            cohen_kappa([x[i] for i in idx], [y[i] for i in idx], cats) for x, y in cols
        ]
        m = mean_ignoring_missing(vals)
        return float("nan") if m is None else m

    order = [float(i) for i in range(len(keep))]
    return bootstrap_ci(stat, order, order, n_boot=n_boot, rng=_rng(seed))


def _rng(seed: int):  # pragma: no cover - trivial
    import numpy as np

    return np.random.default_rng(seed)


def run_agreement(args: argparse.Namespace) -> None:
    qs = load_question_set(args.instrument)
    humans = [load_sheet(p) for p in args.human]
    models = [load_sheet(p) for p in (args.model or [])]
    if len(humans) != 2:
        raise SystemExit(
            f"kappa_H is a two-rater statistic and {len(humans)} human sheet(s) "
            "were given; Sec 6.2 step 2 needs exactly two"
        )
    for s in humans + models:
        h = (s.get("instrument") or {}).get("hash") or ""
        cur = qs.frozen_hash or qs.compute_hash()
        if h and h != cur:
            raise SystemExit(
                f"sheet {s.get('annotator')!r} was filled against instrument "
                f"{h} but {args.instrument} is {cur}. Sec 6.5: scores are only "
                "comparable within a fixed instrument, and pooling these would "
                "silently mix two different questions."
            )

    per_q: dict[str, Any] = {}
    kappa_h: dict[str, float | None] = {}
    kappa_m_int: dict[str, Interval | None] = {}
    decisions = []

    for qi, q in enumerate(qs.questions):
        ha = _column(humans[0], q.id)
        hb = _column(humans[1], q.id)
        hh = question_agreement(
            q.id, q.kind, q.lo, q.hi, ha, hb,
            n_boot=args.n_boot, seed=args.seed + qi,
            label_a=str(humans[0].get("annotator", "h1")),
            label_b=str(humans[1].get("annotator", "h2")),
        )
        kappa_h[q.id] = None if hh.kappa is None else hh.kappa.estimate

        hm_each = []
        for mi, msheet in enumerate(models):
            mc = _column(msheet, q.id)
            for hj, hs in enumerate(humans):
                hm_each.append(
                    question_agreement(
                        q.id, q.kind, q.lo, q.hi, _column(hs, q.id), mc,
                        n_boot=args.n_boot, seed=args.seed + 1000 * (mi + 1) + 10 * hj + qi,
                        label_a=str(hs.get("annotator", f"h{hj}")),
                        label_b=str(msheet.get("annotator", f"m{mi}")),
                    ).to_dict()
                )
        km = _pair_interval(
            q,
            [(_column(hs, q.id), _column(ms, q.id)) for ms in models for hs in humans],
            n_boot=args.n_boot,
            seed=args.seed + 7000 + qi,
        )
        kappa_m_int[q.id] = km

        cats = scale_categories(q.kind, q.lo, q.hi)
        alpha_all = None
        if cats is not None:
            alpha_all = krippendorff_alpha(
                [_column(s, q.id) for s in humans + models],
                cats,
                level="ordinal" if q.kind == "likert" else "nominal",
            )

        per_q[q.id] = {
            "text": q.text,
            "human_human": hh.to_dict(),
            "human_model_pairs": hm_each,
            "kappa_m_mean": None if km is None else km.to_dict(),
            "alpha_all_raters": (
                None
                if alpha_all is None
                else (None if alpha_all != alpha_all else round(float(alpha_all), 6))
            ),
        }
        decisions.append(
            floor_decision(
                q.id,
                kappa_h[q.id],
                km,
                absolute_floor=args.absolute_floor,
                ceiling_fraction=args.ceiling_fraction,
                kappa_h_min=args.kappa_h_min,
            )
        )

    # -- §6.5 intra-rater --------------------------------------------------
    intra: dict[str, float | None] = {}
    intra_note = "not performed: no --intra sheet was given"
    if args.intra:
        rep = load_sheet(args.intra)
        base_pool = {str(s.get("annotator")): s for s in humans + models}
        base_label = args.intra_of or str(rep.get("annotator"))
        base = base_pool.get(base_label)
        if base is None:
            raise SystemExit(
                f"--intra sheet repeats rater {base_label!r} but no sheet with "
                f"that annotator id was given (have: {sorted(base_pool)})"
            )
        for qi, q in enumerate(qs.questions):
            r = question_agreement(
                q.id, q.kind, q.lo, q.hi,
                _column(base, q.id), _column(rep, q.id),
                n_boot=0, seed=args.seed + 9000 + qi,
            )
            intra[q.id] = None if r.kappa is None else r.kappa.estimate
        intra_note = f"repeat sheet compared against rater {base_label!r}"

    inter_for_check = {
        k: (None if v is None else v.estimate) for k, v in kappa_m_int.items()
    } if models else dict(kappa_h)
    check = intra_rater_check(intra, inter_for_check) if args.intra else None

    n_pass = sum(1 for d in decisions if d.verdict == "pass")
    report = {
        "format": "nebulai-variance-agreement",
        "agreement_version": AGREEMENT_VERSION,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "instrument": {
            "path": str(args.instrument),
            "name": qs.name,
            "hash": qs.frozen_hash or qs.compute_hash(),
            "frozen": qs.is_frozen,
            "n_questions": len(qs.questions),
        },
        "raters": {
            "human": [str(s.get("annotator")) for s in humans],
            "model": [str(s.get("annotator")) for s in models],
        },
        "settings": {
            "n_boot": args.n_boot,
            "seed": args.seed,
            "absolute_floor": args.absolute_floor,
            "ceiling_fraction": args.ceiling_fraction,
            "kappa_h_min": args.kappa_h_min,
            "floor_rule": "min(absolute_floor, ceiling_fraction * kappa_H) "
            "per question, tested on the LOWER CI bound of kappa_M (Sec 6.2 "
            "steps 4-5)",
        },
        "questions": per_q,
        "decisions": [d.to_dict() for d in decisions],
        "summary": {
            "n_pass": n_pass,
            "n_drop_low_reliability": sum(
                1 for d in decisions if d.verdict == "drop-low-reliability"
            ),
            "n_drop_badly_specified": sum(
                1 for d in decisions if d.verdict == "drop-badly-specified"
            ),
            "n_undecidable": sum(1 for d in decisions if d.verdict == "undecidable"),
            "mean_kappa_h": mean_ignoring_missing(kappa_h.values()),
            "mean_kappa_m": mean_ignoring_missing(
                None if v is None else v.estimate for v in kappa_m_int.values()
            ),
            # GATE 0's first stop condition, evaluated rather than described.
            "gate0_instrument_too_thin": n_pass < args.min_questions,
            "min_questions": args.min_questions,
        },
        "intra_rater": {
            "note": intra_note,
            **({} if check is None else check.to_dict()),
        },
    }
    if check is not None and check.halt:
        report["summary"]["gate0_halt_ruler_moving"] = True

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_agreement(report, decisions, per_q)
    print(f"\nwrote {out}")
    if report["summary"]["gate0_instrument_too_thin"]:
        print(
            f"GATE 0 STOP: only {n_pass} question(s) clear the floor on their "
            f"lower CI bound; the plan's threshold is ~{args.min_questions}. "
            "An instrument this thin cannot carry a variance decomposition."
        )
    if check is not None and check.halt:
        print("GATE 0 STOP: " + check.detail)


def _print_agreement(report: dict[str, Any], decisions, per_q) -> None:
    print(
        f"{report['instrument']['name']}  "
        f"({'FROZEN' if report['instrument']['frozen'] else 'DRAFT'})  "
        f"{report['instrument']['hash'][:23]}"
    )
    print(
        f"{'question':<26} {'k_H':>7} {'k_M':>7} {'k_M lo':>7} {'raw':>6} "
        f"{'PABAK':>7} {'AC1':>7} {'alpha':>7}  verdict"
    )
    for d in decisions:
        q = per_q[d.question_id]
        hh = q["human_human"]
        print(
            f"{d.question_id:<26} "
            f"{_p(d.kappa_h):>7} {_p(d.kappa_m):>7} {_p(d.kappa_m_lo):>7} "
            f"{_p(hh['raw_agreement'], 3):>6} {_p(hh['pabak']):>7} "
            f"{_p(hh['gwet_ac1']):>7} {_p(q['alpha_all_raters']):>7}  {d.verdict}"
        )
    s = report["summary"]
    print(
        f"\npass={s['n_pass']}  drop-low={s['n_drop_low_reliability']}  "
        f"drop-badly-specified={s['n_drop_badly_specified']}  "
        f"undecidable={s['n_undecidable']}"
    )
    print(
        f"mean kappa_H={_p(s['mean_kappa_h'])}  mean kappa_M={_p(s['mean_kappa_m'])}  "
        "(compare against the measured human ceiling, not against 1.0 - Sec 6.2)"
    )


def _p(x: float | None, nd: int = 4) -> str:
    return "missing" if x is None else f"{float(x):.{nd}f}"


# --------------------------------------------------------------------------
# item 9 — price re-verification
# --------------------------------------------------------------------------


def run_prices(args: argparse.Namespace) -> None:
    """§10 item 9. OpenRouter's catalogue is public, so no key is needed.

    A price that does not match is reported as **drift**, not corrected in
    place: `corpus.py`'s numbers carry a dated verification comment, and a
    silent edit would destroy the only record of when they were true.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from probe_endpoints import probe_openrouter  # type: ignore[import-not-found]

    from ..corpus import CORPUS

    res = probe_openrouter(args.timeout)
    rows: list[dict[str, Any]] = []
    if not res.get("reachable"):
        print(
            "OpenRouter catalogue UNREACHABLE: "
            f"{res.get('error')}. Prices are NOT verified — that is 'unknown', "
            "not 'unchanged'."
        )
        payload = {
            "checked": datetime.now(UTC).isoformat(timespec="seconds"),
            "reachable": False,
            "error": str(res.get("error")),
            "models": [],
        }
    else:
        served = res["models"]
        for key, spec in CORPUS.items():
            live = served.get(spec.endpoint)
            row = {
                "key": key,
                "endpoint": spec.endpoint,
                "recorded_usd_in": spec.usd_in,
                "recorded_usd_out": spec.usd_out,
                # `None` means the exact pinned id is not in the catalogue. It
                # is never resolved to a similar id: llm.py's identity rule is
                # that a missing route is refused, not routed around.
                "live_usd_in": None if live is None else live[0],
                "live_usd_out": None if live is None else live[1],
                "served": live is not None,
            }
            if live is None:
                row["drift"] = "not-listed"
            else:
                di = abs((live[0] or 0.0) - spec.usd_in)
                do = abs((live[1] or 0.0) - spec.usd_out)
                row["drift"] = "none" if (di < 1e-9 and do < 1e-9) else "CHANGED"
                row["delta_usd_in"] = round((live[0] or 0.0) - spec.usd_in, 6)
                row["delta_usd_out"] = round((live[1] or 0.0) - spec.usd_out, 6)
            rows.append(row)
        payload = {
            "checked": datetime.now(UTC).isoformat(timespec="seconds"),
            "reachable": True,
            "catalogue_size": res.get("n"),
            "source": "https://openrouter.ai/api/v1/models (public, no key)",
            "models": rows,
        }
        print(f"{'model':<22} {'in rec':>8} {'in live':>8} {'out rec':>8} {'out live':>8}  drift")
        for r in rows:
            print(
                f"{r['key']:<22} {r['recorded_usd_in']:>8} "
                f"{_p(r['live_usd_in'], 4):>8} {r['recorded_usd_out']:>8} "
                f"{_p(r['live_usd_out'], 4):>8}  {r['drift']}"
            )
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {p}")


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def add_variance_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "variance",
        help="W1 generative-variance tooling: annotation sheets, agreement, prices",
    )
    s = p.add_subparsers(dest="variance_cmd", required=True)

    a = s.add_parser("annotate-sheet", help="emit blank per-annotator scoring sheets")
    a.add_argument("--instrument", required=True, help="question-set JSON")
    a.add_argument("--stories", default="", help="story JSON file or directory")
    a.add_argument("--n-stories", type=int, default=50,
                   help="placeholder story count when --stories is absent (Sec 6.4: ~50)")
    a.add_argument("--annotators", default="A,B", help="comma-separated ids; at least two")
    a.add_argument("--repeat-n", type=int, default=10,
                   help="stories in the Sec 6.5 blind-repeat block (0 disables)")
    a.add_argument("--out", default="out/variance/annotation")
    a.set_defaults(fn=run_annotate_sheet)

    g = s.add_parser("agreement", help="kappa_H / kappa_M / alpha / PABAK / AC1 per question")
    g.add_argument("--instrument", required=True)
    g.add_argument("--human", action="append", required=True,
                   help="filled human sheet (give exactly two)")
    g.add_argument("--model", action="append", default=[],
                   help="filled scorer sheet (repeatable)")
    g.add_argument("--intra", default="", help="blind re-score sheet for the Sec 6.5 check")
    g.add_argument("--intra-of", default="", help="which rater id --intra repeats")
    g.add_argument("--n-boot", type=int, default=2000)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--absolute-floor", type=float, default=DEFAULT_ABSOLUTE_FLOOR)
    g.add_argument("--ceiling-fraction", type=float, default=DEFAULT_CEILING_FRACTION)
    g.add_argument("--kappa-h-min", type=float, default=DEFAULT_KAPPA_H_MIN)
    g.add_argument("--min-questions", type=int, default=20,
                   help="GATE 0's thinness threshold (plan Sec 10)")
    g.add_argument("--out", default="out/variance/agreement.json")
    g.set_defaults(fn=run_agreement)

    pr = s.add_parser("prices", help="re-verify corpus.py prices (Sec 10 item 9)")
    pr.add_argument("--timeout", type=float, default=30.0)
    pr.add_argument("--out", default="")
    pr.set_defaults(fn=run_prices)
