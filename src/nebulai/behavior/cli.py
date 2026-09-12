"""`nebulai behavior …` — plan, calibrate, run, analyze, inspect, serve.

Registered as one subcommand group from `nebulai.cli`, so nothing in the
existing CLI moves. Every paid path goes through the same two-step the rest of
the project uses — print an estimate, then require an explicit `--approve` —
and every local path is free and needs neither.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..llm import BudgetError, RunBudget
from . import analyze as A
from . import cues as C
from . import export as X
from . import report as R
from . import stats as S
from .contract import CANARY_CUE, Cue, Manifest, ModelRef, load_manifest
from .embed import DEFAULT_EMBEDDER_ID, DEFAULT_EMBEDDER_REVISION, resolve_embedder
from .protocol import default_frames
from .runner import Runner, build_schedule, canary_schedule, estimate
from .store import TrialStore

DEFAULT_OUT = "out/behavior"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        return ""


def _study_dir(out: str, study_id: str) -> Path:
    p = Path(out) / study_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _models_for(preset: str) -> list[ModelRef]:
    """The built-in arm presets. Each is an exact pin, never a family name."""
    if preset == "capability":
        # §5.7's required control pair: same family, same tokenizer, same
        # corpus, same objective, neither instruction-tuned. The only
        # substantial difference is scale — which makes the capability confound
        # measurable instead of merely acknowledged. Both local, zero spend.
        return [
            ModelRef("cap_small", "gpt2-local", "gpt2", label="GPT-2 small (124M)"),
            ModelRef("cap_xl", "gpt2-local", "gpt2-xl", label="GPT-2 XL (1.5B)"),
        ]
    if preset == "gpt2-xai":
        return [
            ModelRef("A", "gpt2-local", "gpt2", label="GPT-2 small (124M)"),
            ModelRef(
                "B", "xai", "", label="xAI (pin an exact dated release with --pin-b)"
            ),
        ]
    if preset == "fake":
        return [
            ModelRef("A", "fake", "fake", label="synthetic A"),
            ModelRef("B", "fake", "fake", label="synthetic B"),
        ]
    raise SystemExit(f"unknown preset {preset!r}: capability | gpt2-xai | fake")


def _cue_set(name: str) -> list[Cue]:
    if name == "calibration":
        return C.calibration_cues()
    if name == "pilot":
        return C.pilot_cues()
    if name == "control":
        return C.positive_control_cues()
    raise SystemExit(f"unknown cue set {name!r}: calibration | pilot | control")


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def run_plan(a: argparse.Namespace) -> None:
    models = _models_for(a.preset)
    if a.pin_b:
        models = [m if m.key != "B" else ModelRef(**{**m.__dict__, "pinned": a.pin_b}) for m in models]
    cues = _cue_set(a.cues)
    m = Manifest(
        study_id=a.study_id,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        models=models,
        cues=cues,
        frames=default_frames(),
        trials_per_cue=a.trials,
        n_time_blocks=a.blocks,
        min_valid_trials=a.min_valid,
        min_within_block_trials=a.min_within_block,
        seed=a.seed,
        strict=not a.exploratory,
        embedder_id=DEFAULT_EMBEDDER_ID,
        embedder_sha=DEFAULT_EMBEDDER_REVISION,
        max_cost_usd=a.max_cost_usd,
        git_commit=_git_commit(),
        notes=a.notes,
    )

    # §6.7.1: is confirmation combinatorially reachable at all? Computed and
    # stored BEFORE collection, because the answer can only be fixed by a design
    # change, never by loosening a threshold afterwards.
    per_block = max(1, a.trials // a.blocks)
    m.p_floor = S.p_floor([per_block] * a.blocks, [per_block] * a.blocks)

    est = estimate(m, ["discovery"])
    print(f"study: {m.study_id}")
    print(f"  cues:   {len(cues)} ({a.cues})")
    print(f"  models: {[f'{x.key}={x.pinned or 'UNPINNED'}' for x in models]}")
    print(est.render())
    print(
        f"  p-floor: {m.p_floor:.3g} (smallest achievable permutation p with "
        f"{a.blocks} blocks × {per_block} trials/model/block)"
    )
    if m.p_floor > m.q_threshold:
        print(
            f"  REFUSED: the p-floor {m.p_floor:.3g} is above q={m.q_threshold}. "
            f"No cue could ever be confirmed, at any effect size, for a purely "
            f"combinatorial reason (§6.7.1). Increase --trials or reduce "
            f"--blocks before collecting anything."
        )
        raise SystemExit(2)

    for ref in models:
        if ref.adapter == "xai":
            from .adapters.xai import XAIAdapter

            ad = XAIAdapter(pinned=ref.pinned)
            audit = ad.audit()
            print(f"  audit {ref.key}: {json.dumps(audit)}")
            if audit.get("status") == "not_run":
                print(
                    "  NOTE: this arm is recorded as not_run with its reason and "
                    "its estimate. It is not skipped: a manifest naming two "
                    "models over data containing one cannot be audited later."
                )
            else:
                m.fingerprint_available = audit.get("fingerprint_available")

    h = m.freeze(datetime.now(UTC).isoformat(timespec="seconds"))
    out = _study_dir(a.out, m.study_id) / "manifest.json"
    m.save(out)
    print(f"  frozen: {h}\n  wrote:  {out}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def run_run(a: argparse.Namespace) -> None:
    m = load_manifest(a.manifest)
    d = _study_dir(a.out, m.study_id)
    store = TrialStore(d / "trials.sqlite")
    budget = RunBudget(m.max_cost_usd, label=f"behavior:{m.study_id}")

    est = estimate(m, [a.arm] if a.arm != "canary" else [])
    if est.n_paid_trials:
        budget.preflight(est.est_cost_usd or 0.0, est.n_paid_trials, "paid arms")
        if not a.approve:
            raise SystemExit(
                "this run includes paid trials and was not approved.\n"
                "  re-run with --approve after reading the estimate above.\n"
                "  approval is a step, not a flag the runner can set for itself."
            )
        budget.approve()

    # A paid arm is never batched. The estimate, the identity check and the
    # spend ceiling are all per-request quantities, and a hosted endpoint has
    # no equivalent of "sample sixteen sequences in one forward pass" that
    # preserves them. Batching exists for the local arms, where the alternative
    # is re-reading 6 GB of weights once per generated token.
    batch = 1 if est.n_paid_trials else max(1, int(a.batch))
    if batch > 1:
        # Recorded, not assumed: a reader of this store has to be able to see
        # that execution order was grouped, because `batch_plan` reorders
        # inside a block and a replay at batch 1 will not reproduce these exact
        # samples (it reproduces the same schedule and the same seeds).
        store.set_meta("batch_size", str(batch))
    runner = Runner(
        m,
        store,
        budget=budget,
        batch_size=batch,
        progress=_print_progress if a.verbose else None,
    )
    if a.cue_limit:
        # Same reason as batch_size: the store has to carry the fact, because a
        # reader counting 40 cues in a 100-cue manifest cannot otherwise tell a
        # deliberate partial run from a lost database.
        store.set_meta("cue_limit", str(int(a.cue_limit)))
    res = runner.run(
        a.arm,
        limit=a.limit,
        cue_limit=a.cue_limit,
        force_unlock=getattr(a, "force_unlock", False),
    )
    print(
        f"arm {res.arm}: {res.completed} new, {res.skipped_existing} already "
        f"present, {res.errors} errored, ${res.spent_usd:.4f} spent"
    )
    if res.not_run:
        for k, why in res.not_run.items():
            print(f"  not_run {k}: {why}")
    if res.halted:
        print(f"  HALTED: {res.halted}")
    store.close()


def _print_progress(p: dict[str, Any]) -> None:
    print(f"    {p['arm']}: {p['done']}/{p['total']}  ${p['spent_usd']:.4f}", flush=True)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def _profiles_and_bandwidth(m: Manifest, store: TrialStore, embedder: Any, arm: str):
    trials = [t for t in store.iter_trials(m.study_id) if t.arm == arm]
    profiles = A.build_profiles(trials, m, embedder)
    if m.mmd_bandwidth:
        bw = m.mmd_bandwidth
    else:
        pool = [p.vectors for cm in profiles.values() for p in cm.values() if p.vectors is not None]
        bw = S.median_bandwidth(np.vstack(pool)) if pool else 1.0
    return trials, profiles, bw


def run_analyze(a: argparse.Namespace) -> None:
    m = load_manifest(a.manifest)
    d = _study_dir(a.out, m.study_id)
    store = TrialStore(d / "trials.sqlite")
    embedder = resolve_embedder(a.embedder)
    strat = {c.text: (c.stratum, c.pack) for c in m.cues}

    trials, profiles, bw = _profiles_and_bandwidth(m, store, embedder, "discovery")
    if not profiles:
        raise SystemExit("no discovery trials in the store — run the arm first")

    # The capability reference (§5.7) is a per-cue number from a separate study.
    cap = _load_capability_reference(a.capability_reference)

    keys = [x.key for x in m.models][:2]
    results: list[A.CueResult] = []
    for cue, cm in profiles.items():
        if cue == CANARY_CUE or keys[0] not in cm or keys[1] not in cm:
            continue
        s, pack = strat.get(cue, ("unknown", ""))
        results.append(
            A.compare_cue(
                cue, s, cm[keys[0]], cm[keys[1]], m,
                bandwidth=bw, capability_reference=cap.get(cue), pack=pack,
                permutations=a.permutations,
            )
        )
    A.finalize_family(results, m)

    strict_source = all(
        m.model(k).adapter != "fake" for k in keys
    ) and getattr(embedder, "strict_ok", False) and m.strict
    if not strict_source:
        for r in results:
            if r.status == "confirmed":
                r.status = "suggestive"

    if not results:
        # Every cue needs both arms. One arm recorded `not_run` (no credentials,
        # a missing optional dependency) leaves a store full of real trials and
        # nothing to compare, and the useful output is the reason, named per arm.
        have = sorted({t.model_key for t in trials if t.cue != CANARY_CUE})
        want = [x.key for x in m.models]
        raise SystemExit(
            "no cue has trials for both arms, so there is nothing to compare.\n"
            f"  manifest arms: {want}\n"
            f"  arms with discovery trials in the store: {have or 'none'}\n"
            f"  missing: {[k for k in want if k not in have] or 'none'}\n"
            "  This is not an analysis failure: `behavior run` records an "
            "unreachable arm as not_run with its reason rather than dropping it "
            "from the manifest, and an artifact with one arm would be a "
            "comparison with itself. Collect the missing arm, or analyze a study "
            "whose arms are both reachable."
        )

    landscape = X.cue_landscape([r.cue for r in results], embedder)
    diagnostics = {
        "mmd_bandwidth": bw,
        "p_floor": m.p_floor,
        "q_threshold": m.q_threshold,
        "strict_source": strict_source,
        "aa_control": [
            A.aa_control(cm[k], m, bw)
            for cue, cm in list(profiles.items())[:1]
            for k in keys
            if k in cm
        ],
        "positive_control": A.positive_control_report(trials),
        "canary": A.canary_report(store.iter_trials(m.study_id), embedder),
        "reasoning_tokens_p95": m.reasoning_tokens_p95,
    }
    runs = [store.progress(m.study_id)]
    samples = _samples(trials, keys)
    coverage = _coverage(m, store, results, trials)

    payload = X.build_export(
        m, results, landscape=landscape, diagnostics=diagnostics, runs=runs,
        samples=samples, coverage=coverage,
    )
    out = X.write_export(d / "behavior.json", payload)
    print(f"wrote {out}  ({len(results)} cues, bandwidth {bw:.4f})")
    _print_top(results)
    store.close()


def _coverage(
    m: Manifest,
    store: TrialStore,
    results: list[Any],
    trials: list[Any] | None = None,
) -> dict[str, Any]:
    """What fraction of the preregistered cue set this artifact actually covers.

    Read from the store rather than inferred from `len(results)` alone: a cue
    can be missing because it was never collected (a `--cue-limit` run) or
    because it was collected and every trial of it was invalid. Those are
    different facts and the second one is already reported per cue, so the
    reason comes from the run's own record.

    The `--cue-limit` branch counts the cues that are actually IN the store and
    at their full planned depth, not the limit that was requested. A limit is a
    request: a stage killed halfway through, or still running, has a higher
    limit in `meta` than it has cues on disk, and a sentence that reported the
    request would claim full repeats and full block balance for cues that have
    neither. `cue_limit` is still recorded beside the counts, as the intent.
    """
    limit = store.get_meta("cue_limit")
    planned = len(m.cues)
    analyzed = len(results)
    complete = analyzed >= planned

    # Per-cue depth, from the store: how many trials one cue gets when the
    # schedule runs it to completion, and how many each cue actually has.
    per_cue_full = max(1, m.trials_per_cue * len(m.models))
    depth: dict[str, int] = {}
    for t in trials or []:
        if t.cue != CANARY_CUE:
            depth[t.cue] = depth.get(t.cue, 0) + 1
    collected = len(depth)
    at_full_depth = sum(1 for n in depth.values() if n >= per_cue_full)
    partial = collected - at_full_depth

    if complete:
        reason = ""
    elif limit:
        reason = (
            f"run with --cue-limit {limit}; {collected} of the preregistered "
            f"{planned} cues are in the store, {at_full_depth} of them at full "
            f"repeats and full block balance"
            + (f" and {partial} partially collected" if partial else "")
            + f", and {analyzed} could be compared across both arms. The cues "
            f"that ran are not weakened by the ones that did not; the study's "
            f"coverage is."
        )
    else:
        reason = (
            f"{planned - analyzed} of {planned} preregistered cues have no "
            f"comparable pair of arms in the store — either not collected or "
            f"with no valid trials for at least one model."
        )
    return {
        "cues_planned": planned,
        "cues_analyzed": analyzed,
        "cues_collected": collected,
        "cues_at_full_depth": at_full_depth,
        "complete": complete,
        "reason": reason,
        "cue_limit": int(limit) if limit else None,
    }


def _samples(trials: list[Any], keys: list[str]) -> dict[str, list[dict[str, Any]]]:
    """A small read-only slice of raw evidence, per §8's sample artifact.

    Exact surface forms, never only the normalized ones: §6.2 requires the raw
    spelling to stay visible everywhere an embedding-derived claim is made.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for t in trials:
        if not t.raw_output:
            continue
        bucket = out.setdefault(t.cue, [])
        if len(bucket) >= 6:
            continue
        bucket.append(
            {
                "model_key": t.model_key,
                "repeat": t.repeat,
                "block": t.block,
                "raw": t.raw_output,
                "associates": t.associates,
                "valid": t.valid,
                "invalid_reason": t.invalid_reason,
                "response_id": t.response_id,
            }
        )
    return out


def _load_capability_reference(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        c["cue"]: c["delta_hat"]
        for c in d.get("cues", [])
        if c.get("delta_hat") is not None
    }


def _print_top(results: list[A.CueResult], n: int = 12) -> None:
    ranked = [r for r in results if r.delta_hat is not None]
    ranked.sort(key=lambda r: r.delta_hat or 0.0, reverse=True)
    print(f"  {'cue':<16}{'Δ̂':>10}{'p':>10}{'q':>10}  status")
    for r in ranked[:n]:
        p = "—" if r.p_value is None else f"{r.p_value:.4f}"
        q = "—" if r.q_value is None else f"{r.q_value:.4f}"
        print(f"  {r.cue:<16}{r.delta_hat:>10.4f}{p:>10}{q:>10}  {r.status}")


# ---------------------------------------------------------------------------
# calibrate — the §6.7 Phase 0 report
# ---------------------------------------------------------------------------


def run_calibrate(a: argparse.Namespace) -> None:
    m = load_manifest(a.manifest)
    d = _study_dir(a.out, m.study_id)
    store = TrialStore(d / "trials.sqlite")
    embedder = resolve_embedder(a.embedder)
    trials, profiles, bw = _profiles_and_bandwidth(m, store, embedder, "discovery")
    if not profiles:
        raise SystemExit("no trials to calibrate on")

    keys = [x.key for x in m.models][:2]
    aa = []
    for cue, cm in profiles.items():
        for k in keys:
            if k in cm and cm[k].vectors is not None and len(cm[k].vectors) >= 8:
                r = A.aa_control(cm[k], m, bw, draws=a.aa_draws)
                r["cue"] = cue
                aa.append(r)
        if len(aa) >= 2 * a.aa_cues:
            break

    fprs = [r["false_positive_rate_at_0.05"] for r in aa if r.get("false_positive_rate_at_0.05") is not None]
    block_counts: dict[tuple[str, str], dict[int, int]] = {}
    for t in trials:
        if t.valid:
            block_counts.setdefault((t.cue, t.model_key), {}).setdefault(t.block, 0)
            block_counts[(t.cue, t.model_key)][t.block] += 1
    #  The floor is PER CUE, because the block structure is per cue: a cue
    #  collected 6 trials deep and a cue collected 192 deep have floors seven
    #  orders of magnitude apart. Reporting only the worst made a study's whole
    #  calibration read as failed the moment one partially-collected cue
    #  appeared in the store, which is both wrong and the kind of wrong that
    #  gets a real constraint ignored. So: every cue's floor, the best and worst
    #  of them, and the names of the ones that cannot clear q at the depth they
    #  were collected to.
    per_cue_floor: dict[str, float] = {}
    for cue in {c for c, _ in block_counts}:
        sizes = []
        for k in keys:
            bc = block_counts.get((cue, k), {})
            sizes.append([bc.get(b, 0) for b in range(m.n_time_blocks)])
        if len(sizes) == 2:
            per_cue_floor[cue] = S.p_floor(sizes[0], sizes[1])
    floors = sorted(per_cue_floor.values())
    worst_floor = floors[-1] if floors else 0.0
    best_floor = floors[0] if floors else 0.0
    above_q = sorted(c for c, v in per_cue_floor.items() if v >= m.q_threshold)

    pc = A.positive_control_report(trials, threshold=a.control_threshold)
    report = {
        "study_id": m.study_id,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "manifest_hash": m.frozen_hash,
        "embedder": {"id": getattr(embedder, "id", "?"), "strict_ok": getattr(embedder, "strict_ok", False)},
        "mmd_bandwidth": bw,
        "n_cues": len(profiles),
        "n_trials": len(trials),
        # §6.7.1 — printed, not assumed. The design p-floor is what `plan`
        # computed from the intended block structure; the empirical one uses the
        # blocks that actually have valid trials in them.
        "p_floor_design": m.p_floor,
        "p_floor_empirical_worst": worst_floor,
        "p_floor_empirical_best": best_floor,
        "p_floor_per_cue": dict(sorted(per_cue_floor.items())),
        "p_floor_cues_total": len(per_cue_floor),
        # Named, not counted away: a cue in this list cannot be called
        # significant at the depth it was collected to, whatever its effect.
        "p_floor_cues_above_q": above_q,
        "q_threshold": m.q_threshold,
        "p_floor_clears_q": bool(best_floor > 0.0 and best_floor < m.q_threshold),
        "p_floor_clears_q_all_cues": bool(
            per_cue_floor and worst_floor < m.q_threshold
        ),
        "aa_controls": aa,
        "aa_false_positive_rate_mean": (sum(fprs) / len(fprs)) if fprs else None,
        "positive_control": pc,
        "canary": A.canary_report(store.iter_trials(m.study_id), embedder),
        "rank_stability": _rank_stability(profiles, keys, m, bw),
    }
    out = d / "calibration.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")

    #  The JSON is the record; the Markdown is the thing a person reads before
    #  deciding whether to trust a single Δ̂ out of this study. The p-floor vs q
    #  comparison in particular spans orders of magnitude, which a column of
    #  decimals hides and a log axis shows, so the figure ships beside it.
    md = d / "calibration.md"
    fig = d / R.FIGURE_NAME
    fig.write_text(
        R.p_floor_figure(
            p_floor_design=report["p_floor_design"],
            p_floor_best=report["p_floor_empirical_best"] or None,
            p_floor_worst=report["p_floor_empirical_worst"] or None,
            q=report["q_threshold"],
            n_cues=report["p_floor_cues_total"] or None,
        ),
        encoding="utf-8",
    )
    md.write_text(R.calibration_markdown(report), encoding="utf-8")
    print(f"wrote {md} and {fig}")
    print(f"  bandwidth (frozen from here on): {bw:.6f}")
    print(
        f"  p-floor design {m.p_floor:.3g} / as collected "
        f"{best_floor:.3g}–{worst_floor:.3g} over {len(per_cue_floor)} cues "
        f"vs q={m.q_threshold}"
    )
    print(f"  at least one cue clears q: {report['p_floor_clears_q']}")
    if above_q:
        print(
            f"  {len(above_q)} cue(s) cannot clear q at their collected depth: "
            f"{', '.join(above_q[:6])}{' …' if len(above_q) > 6 else ''}"
        )
    if report["aa_false_positive_rate_mean"] is not None:
        print(f"  A/A false-positive rate at 0.05: {report['aa_false_positive_rate_mean']:.3f}")
    for k, r in pc["pass_rate"].items():
        mark = "PASS" if pc["passed"][k] else "FAIL"
        print(f"  positive control {k}: {r:.3f} ({pc['n_trials'][k]} trials) {mark}")
    store.close()


def _rank_stability(profiles, keys, m, bw) -> dict[str, Any]:
    """Spearman correlation of per-cue Δ̂ between two disjoint trial halves.

    This is §6.7's "rank-stability versus trial count" curve at the one count
    the calibration actually collected. A low value means the ranked list is
    mostly noise at this n, which is a design answer, not a result.
    """
    from scipy.stats import spearmanr  # type: ignore

    a_vals, b_vals = [], []
    rng = np.random.default_rng(m.seed)
    for cue, cm in profiles.items():
        if not all(k in cm and cm[k].vectors is not None for k in keys):
            continue
        va, vb = cm[keys[0]].vectors, cm[keys[1]].vectors
        if len(va) < 8 or len(vb) < 8:
            continue
        pa, pb = rng.permutation(len(va)), rng.permutation(len(vb))
        h1 = S.delta_hat(va[pa[: len(va) // 2]], vb[pb[: len(vb) // 2]], bw, draws=8, rng=rng).delta_hat
        h2 = S.delta_hat(va[pa[len(va) // 2 :]], vb[pb[len(vb) // 2 :]], bw, draws=8, rng=rng).delta_hat
        if not (math.isnan(h1) or math.isnan(h2)):
            a_vals.append(h1)
            b_vals.append(h2)
    if len(a_vals) < 5:
        return {"status": "missing", "reason": "fewer than 5 comparable cues"}
    rho, p = spearmanr(a_vals, b_vals)
    return {"status": "measured", "n_cues": len(a_vals), "spearman": float(rho), "p": float(p)}


# ---------------------------------------------------------------------------
# conformance / inspect / serve
# ---------------------------------------------------------------------------


def run_conformance(a: argparse.Namespace) -> None:
    from .adapters.gpt2_local import conformance

    rep = conformance(a.model)
    print(json.dumps(rep, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    if not rep["passed"]:
        raise SystemExit(
            "§5.6 conformance FAILED: the batched backend's next-token logits do "
            "not match GPT2Numpy within the declared tolerance. The backend is "
            "not accepted; nothing sampled from it is study data."
        )


def run_publish(a: argparse.Namespace) -> None:
    """Copy ONE study's artifact to the path the Behavior page reads.

    `analyze` writes `<out>/<study_id>/behavior.json`, and the page fetches
    `<data base>/behavior/behavior.json` — one file, no study id. That is not
    an oversight in either place: a page that auto-picked a study would silently
    change which experiment it displays the moment a second one is analyzed, and
    "which study is this" is the first thing a reader has to be able to answer.
    So the choice is made once, by a human, here, and recorded in the copy.

    Refuses to publish an artifact whose every cue is downgraded for a
    non-strict source (a `fake` arm, the hash embedder): those exist to
    exercise the pipeline, and shipping one to the page as though it were
    evidence is exactly the substitution the claim contract forbids. `--force`
    exists for the case where the intent IS to show the page an example, and it
    stamps that intent into the published copy.
    """
    src = Path(a.out) / a.study_id / "behavior.json"
    if not src.exists():
        raise SystemExit(f"no artifact at {src}; run `behavior analyze` first")
    d = json.loads(src.read_text(encoding="utf-8"))

    strict = bool(d.get("diagnostics", {}).get("strict_source"))
    if not strict and not a.force:
        raise SystemExit(
            f"{a.study_id} is not a strict-source study: its arms or its encoder "
            "cannot support a claim about any model, so every cue in it is "
            "downgraded. Publishing it to the Behavior page would put "
            "pipeline-exercise output where evidence belongs.\n"
            "  --force publishes it anyway and records `published_as: example` "
            "in the copy, which the page shows as such."
        )

    d["published"] = {
        "study_id": d["study_id"],
        "source": str(src),
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "published_as": "study" if strict else "example",
    }
    dest = Path(a.out) / "behavior.json"
    out = X.write_export(dest, d)
    print(f"published {a.study_id} -> {out}")
    print(f"  {d['claim']}")
    if not strict:
        print("  published_as: example (not strict-source; the page labels it)")


def run_inspect(a: argparse.Namespace) -> None:
    path = Path(a.out) / a.study_id / "behavior.json"
    if not path.exists():
        raise SystemExit(f"no artifact at {path}; run `behavior analyze` first")
    d = json.loads(path.read_text(encoding="utf-8"))
    if a.cue:
        row = next((c for c in d["cues"] if c["cue"] == a.cue), None)
        if row is None:
            raise SystemExit(f"cue {a.cue!r} is not in this study")
        print(json.dumps(row, indent=2, ensure_ascii=False))
        return
    print(f"study {d['study_id']} — manifest {d['manifest']['hash']}")
    print(f"  {d['claim']}")
    by_status: dict[str, int] = {}
    for c in d["cues"]:
        by_status[c["status"]] = by_status.get(c["status"], 0) + 1
    for k, v in sorted(by_status.items(), key=lambda kv: -kv[1]):
        print(f"  {v:>5}  {k}")
    #  A study being written right now is a study whose numbers are about to
    #  change. Say so, rather than letting a reader quote a snapshot as final.
    sq = Path(a.out) / a.study_id / "trials.sqlite"
    if sq.exists():
        with TrialStore(sq) as st:
            lock = st.writer_lock()
        if lock:
            state = "collecting now" if lock.get("live") else "stale (holder gone)"
            print(
                f"  writer lock: pid {lock.get('pid')} on {lock.get('host')} — "
                f"{state}: {lock.get('note') or '-'}"
            )


def run_serve(a: argparse.Namespace) -> None:
    from .server import serve

    serve(a.out, host=a.host, port=a.port, allow_paid=a.allow_paid)


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def add_behavior_parser(sub: argparse._SubParsersAction) -> None:
    b = sub.add_parser(
        "behavior",
        help="behavioral semantic divergence (docs/BEHAVIORAL-DIVERGENCE-PLAN.md)",
        description=(
            "Repeated, controlled output sampling over pinned model deployments. "
            "Nothing here reads any model's internals and nothing here ranks "
            "models."
        ),
    )
    bs = b.add_subparsers(dest="behavior_cmd", required=True)

    p = bs.add_parser("plan", help="build and freeze a study manifest")
    p.add_argument("--study-id", required=True)
    p.add_argument("--preset", default="capability", help="capability | gpt2-xai | fake")
    p.add_argument("--cues", default="calibration", help="calibration | pilot | control")
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--min-valid", type=int, default=20)
    p.add_argument("--min-within-block", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pin-b", default="", help="exact dated release for arm B")
    p.add_argument("--exploratory", action="store_true", help="allow a moving alias; no cue can be confirmed")
    p.add_argument("--max-cost-usd", type=float, default=1.0)
    p.add_argument("--notes", default="")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.set_defaults(fn=run_plan)

    r = bs.add_parser("run", help="collect trials for one arm (resumable)")
    r.add_argument("--manifest", required=True)
    r.add_argument("--arm", default="discovery", help="discovery | R | G | canary")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument(
        "--force-unlock",
        action="store_true",
        help=(
            "take the store even though another process holds its writer lock. "
            "Only for a holder you have confirmed is dead: the lock exists "
            "because two runners on one store do the same work twice and finish "
            "no sooner, which looks like health from both sides."
        ),
    )
    r.add_argument(
        "--cue-limit",
        type=int,
        default=None,
        help=(
            "collect only the first N cues of the preregistered set, at FULL "
            "repeats each. The supported way to run a local arm short: --limit cuts "
            "the interleaved schedule and leaves every cue underpowered, while this "
            "leaves the cues that ran exactly as preregistered and reports the "
            "missing ones as coverage."
        ),
    )
    r.add_argument(
        "--batch",
        type=int,
        default=1,
        help=(
            "sample this many trials per request on an unpaid local arm "
            "(default 1). Groups only within one collection block and one "
            "(model, frame); paid arms ignore it."
        ),
    )
    r.add_argument("--approve", action="store_true", help="approve the printed paid estimate")
    r.add_argument("--verbose", action="store_true")
    r.add_argument("--out", default=DEFAULT_OUT)
    r.set_defaults(fn=run_run)

    an = bs.add_parser("analyze", help="compute per-cue evidence and write behavior.json")
    an.add_argument("--manifest", required=True)
    an.add_argument("--embedder", default="local", help="local | secondary | hash")
    an.add_argument("--permutations", type=int, default=None)
    an.add_argument("--capability-reference", default=None, help="a behavior.json from the §5.7 arm")
    an.add_argument("--out", default=DEFAULT_OUT)
    an.set_defaults(fn=run_analyze)

    cal = bs.add_parser("calibrate", help="the §6.7 Phase 0 calibration report")
    cal.add_argument("--manifest", required=True)
    cal.add_argument("--embedder", default="local")
    cal.add_argument("--aa-draws", type=int, default=20)
    cal.add_argument("--aa-cues", type=int, default=10)
    cal.add_argument("--control-threshold", type=float, default=0.5)
    cal.add_argument("--out", default=DEFAULT_OUT)
    cal.set_defaults(fn=run_calibrate)

    cf = bs.add_parser("conformance", help="§5.6: batched backend vs GPT2Numpy logits")
    cf.add_argument("--model", default="gpt2")
    cf.add_argument("--out", default="")
    cf.set_defaults(fn=run_conformance)

    pub = bs.add_parser(
        "publish",
        help="copy one study's behavior.json to the path the Behavior page reads",
    )
    pub.add_argument("study_id")
    pub.add_argument(
        "--force",
        action="store_true",
        help="publish a non-strict-source study anyway, stamped as an example",
    )
    pub.add_argument("--out", default=DEFAULT_OUT)
    pub.set_defaults(fn=run_publish)

    i = bs.add_parser("inspect", help="summarize a study, or one cue in it")
    i.add_argument("study_id")
    i.add_argument("--cue", default="")
    i.add_argument("--out", default=DEFAULT_OUT)
    i.set_defaults(fn=run_inspect)

    sv = bs.add_parser("serve", help="loopback runner server for the Behavior page")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--out", default=DEFAULT_OUT)
    sv.add_argument(
        "--allow-paid",
        action="store_true",
        help="permit paid arms from the browser; still requires per-run approval",
    )
    sv.set_defaults(fn=run_serve)
