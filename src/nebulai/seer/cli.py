"""`seer …` — the command-line half of SessionSeer.

    seer run codex "fix the failing test"     # capture one run
    seer run claude "…" --compare-with codex  # the same task, twice
    seer attach "fix the failing test"        # Codex, at app-server fidelity
    seer attach                               # …or just watch a running one
    seer reconcile --limit 50                 # import sessions already on disk
    seer protocol                             # is this Codex build supported?
    seer list                                 # what has been captured
    seer show <run_id>                        # one run, with provenance
    seer compare <run_a> <run_b>              # and what cannot be compared
    seer export <run_id> > run.jsonl          # the raw record
    seer serve                                # HTTP + SSE on :8125

    seer install --apply                      # capture your *own* sessions
    seer watch                                # …and turn them into runs

The printing rules are the same ones the viewer follows, because a terminal is
where most of these numbers will first be read:

* an absent value prints `—`, never `0`;
* every metric prints its fidelity when it is anything other than native;
* `compare` prints refusals *above* the table, not in a footnote — "these two
  runs cannot be compared on tokens" is the finding, not a caveat about it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .analysis import analyze
from .attach import (
    DEFAULT_SOCK,
    CodexAttachment,
    ProtocolMismatch,
    gate,
    protocol_note,
)
from .collector import IDLE_TIMEOUT_S
from .compare import compare as compare_views
from .contract import Fidelity, Outcome
from .export import FORMATS, export as export_run
from .recover import recover_orphans
from .redaction import ContentLevel, parse_level
from .reconcile import reconcile_codex
from .reducer import Measured, RunView, reduce_run
from .runner import Runner
from .store import DEFAULT_ROOT, EventStore

# ── printing ─────────────────────────────────────────────────────────────────

_FID_MARK = {
    Fidelity.NATIVE: "",
    Fidelity.DETERMINISTIC: "",
    Fidelity.ESTIMATED: " ~",
    Fidelity.HEURISTIC: " ?",
    Fidelity.MISSING: "",
    Fidelity.DROPPED_BY_POLICY: "",
}


def fmt(m: Measured, unit: str = "") -> str:
    """The one place a number becomes text. `—` for absent, always."""
    if m.absent:
        return "—"
    v = m.value
    s = f"{v:,.2f}" if isinstance(v, float) and not v.is_integer() else f"{int(v):,}"
    return f"{s}{unit}{_FID_MARK.get(m.fidelity, '')}"


def _print_view(v: RunView) -> None:
    w = sys.stdout.write
    dur = (
        f"{v.ended_at - v.started_at:.1f}s"
        if v.ended_at and v.started_at else "running"
    )
    w(f"\n{v.run_id}  {v.agent} {v.agent_version}  [{v.state.value}]  {dur}\n")
    if v.overlays:
        w(f"  overlays: {', '.join(o.value for o in v.overlays)}\n")
    w(f"  outcome:  {v.outcome.value}")
    if v.outcome.value == "agent_claimed_complete":
        # Said plainly every time. The distinction between what the agent
        # claimed and what was checked is the one a reader must not lose.
        w("   (the agent's own word — nothing verified it)")
    w("\n")
    if v.model:
        w(f"  model:    {v.model.get('model_id')}\n")
    if v.repo:
        w(f"  repo:     {v.repo.get('branch')} @ {(v.repo.get('head') or '')[:8]}"
          f"{' (dirty)' if v.repo.get('dirty') else ''}\n")

    w("\n  actions\n")
    if v.action_counts:
        for a, n in sorted(v.action_counts.items(), key=lambda kv: -kv[1]):
            w(f"    {a:<10} {n:>5}\n")
    else:
        w("    (none observed)\n")

    w("\n  time\n")
    for state, secs in sorted(v.time_in_state.items(), key=lambda kv: -kv[1]):
        w(f"    {state:<22} {secs:>8.1f}s\n")

    w("\n  tokens\n")
    for cat, m in v.usage.items():
        note = f"   ({m.note})" if m.absent and m.note else ""
        w(f"    {cat:<12} {fmt(m):>12}{note}\n")
    w(f"    {'cost':<12} {fmt(v.cost_usd, ' USD'):>12}\n")

    ver = v.verification_after_last_edit()
    w("\n  verification\n")
    w(f"    ran any verification:   {'yes' if v.verified else 'no'}\n")
    w(f"    after the last edit:    "
      f"{'—' if ver.absent else ('yes' if ver.value else 'NO')}"
      f"{'   (' + ver.note + ')' if ver.note else ''}\n")

    q = v.quality
    w(f"\n  data quality  [{q.capture_mode}]\n")
    for gap in q.capture_gaps:
        w(f"    not observable: {gap}\n")
    for cat in q.absent_token_categories:
        w(f"    no bucket:      {cat}\n")
    for k, n in q.dropped_by_policy.items():
        w(f"    dropped ({n:>3}): {k}\n")
    if q.folded_duplicates:
        w(f"    folded repeats: {q.folded_duplicates} "
          "(usage sightings the fold rule refused — this is the rule working)\n")
    for warn in q.warnings[:5]:
        w(f"    warning:        {warn}\n")
    if q.unmatched_tools:
        w(f"    unclassified:   {', '.join(q.unmatched_tools[:8])}\n")
    w("\n")


# ── commands ─────────────────────────────────────────────────────────────────


def _cmd_run(args: argparse.Namespace, store: EventStore) -> int:
    repeat = int(getattr(args, "repeat", 1) or 1)
    if repeat > 1:
        return _cmd_run_ensemble(args, store, repeat)
    agents = [args.agent] + list(args.compare_with or [])
    results = []
    for agent in agents:
        sys.stderr.write(f"[seer] launching {agent} …\n")
        r = Runner(
            agent,
            args.prompt,
            store=store,
            cwd=args.cwd,
            model=args.model,
            keep_reasoning=args.keep_reasoning,
            label=args.label,
            on_event=(_tick if args.progress else None),
        ).run(timeout_s=args.timeout)
        results.append(r)
        if args.progress:
            sys.stderr.write("\n")
        _print_view(r.view)
        if r.exit_code not in (0, None):
            sys.stderr.write(
                f"[seer] {agent} exited {r.exit_code}; last stderr:\n"
                + "".join(f"    {ln}\n" for ln in r.stderr_tail[-3:])
            )

    if len(results) > 1:
        _print_comparison([r.view for r in results])
    return 0 if all(r.exit_code in (0, None) for r in results) else 1


# ── P3: the repeat ───────────────────────────────────────────────────────────
#
# `--repeat N` is the only path in `seer` that can spend N times as much money
# as the human typed a command for, so it is the only one with a budget in
# front of it. The shape is `budget.SeerBudget`'s and is deliberately visible
# here rather than buried: preflight the whole repeat, approve it, launch run
# 1, charge what it really reported, and then preflight AGAIN for runs 2…N now
# that there is a measured number to estimate from. The second preflight is
# what turns `MISSING` into `ESTIMATED`, and it is also what refuses a repeat
# whose first run turned out to be expensive.


def _fmt_cost(m) -> str:
    return "—" if m.absent else f"${float(m.value):.4f}"


def _print_ensemble(doc: dict) -> None:
    w = sys.stdout.write
    w(f"\nensemble {doc['ensemble_id']}  ({doc['n_runs']}"
      f" of {doc['n_runs_requested']} runs)\n")
    w(f"  protocol   {doc['protocol'].get('id')} "
      f"[{doc['protocol'].get('hash_algorithm')} over "
      f"{'+'.join(doc['protocol'].get('hash_fields') or [])}]\n")
    if doc["n_runs"] < doc["point_estimate_min_runs"]:
        w(f"  n = {doc['n_runs']} < {doc['point_estimate_min_runs']}: read every "
          f"number below as an interval, never a point\n")
    for name, r in doc["rates"].items():
        if r.get("p") is None:
            w(f"  {name:<26} —   ({r.get('missing', 'missing')})\n")
        else:
            lo, hi = r["ci95"]
            w(f"  {name:<26} {r['p']:.2f}  [{lo:.2f}, {hi:.2f}]  "
              f"k={r['k']}/n={r['n']}\n")
    fan = doc["fan"]
    if fan:
        w(f"  fan        {len(fan)} steps of {doc['fan_metric']}, envelope "
          f"{doc['fan_envelope']}\n")
    rel = doc["reliability"]
    if rel.get("delta_hat") is None:
        w(f"  delta_hat  —   ({rel.get('missing', 'missing')})\n")
    else:
        w(f"  delta_hat  {rel['delta_hat']:+.4f}  "
          f"(between {rel['between']:.4f} − ½[{rel['within_a']:.4f} + "
          f"{rel['within_b']:.4f}])\n")
    for quantity, why in doc.get("missing", {}).items():
        w(f"  missing: {quantity} — {why}\n")
    w("\n")


def _cmd_run_ensemble(
    args: argparse.Namespace, store: EventStore, repeat: int
) -> int:
    from .budget import SeerBudget, SeerBudgetError, protocol_fingerprint
    from .ensemble import (
        EnsembleManifest,
        Member,
        build_ensemble,
        new_ensemble_id,
        write_manifest,
    )

    agents = [args.agent] + list(args.compare_with or [])
    cwd = str(args.cwd) if args.cwd else str(Path.cwd())
    fps = {
        a: protocol_fingerprint(a, args.prompt, model=args.model, cwd=cwd)
        for a in agents
    }
    total = repeat * len(agents)

    budget = SeerBudget(
        store,
        ceiling_usd=args.max_cost_usd,
        label=f"seer run --repeat {repeat}",
    )
    try:
        budget.preflight(
            fps[args.agent]["id"],
            total,
            acknowledge_unpriced=args.acknowledge_unpriced,
        )
    except SeerBudgetError as exc:
        sys.stderr.write(f"[seer] {exc}\n")
        return 2
    budget.approve()

    manifest = EnsembleManifest(
        ensemble_id=new_ensemble_id(),
        protocol=fps[args.agent],
        n_runs_requested=total,
        seed_base=args.seed_base,
        # recorded, never handed to an agent — none of the three accept one
        seed_applied=False,
    )
    # written before the first launch, so a repeat killed halfway still leaves
    # an ensemble a reader can find the surviving runs through
    write_manifest(store, manifest)
    sys.stderr.write(f"[seer] ensemble {manifest.ensemble_id}\n")

    results = []
    launched = {a: 0 for a in agents}
    stopped: str | None = None
    for i in range(repeat):
        for agent in agents:
            pid = fps[agent]["id"]
            if launched[agent] == 1:
                # run 2 of this protocol: run 1 has happened, so there may now
                # be a measured number to estimate the rest from
                try:
                    budget.preflight(
                        pid,
                        repeat - 1,
                        acknowledge_unpriced=args.acknowledge_unpriced,
                    )
                    budget.approve()
                except SeerBudgetError as exc:
                    stopped = str(exc)
                    break
            sys.stderr.write(
                f"[seer] launching {agent} ({i + 1}/{repeat}) …\n"
            )
            r = Runner(
                agent,
                args.prompt,
                store=store,
                cwd=args.cwd,
                model=args.model,
                keep_reasoning=args.keep_reasoning,
                label=args.label,
                on_event=(_tick if args.progress else None),
            ).run(timeout_s=args.timeout)
            results.append(r)
            launched[agent] += 1
            if args.progress:
                sys.stderr.write("\n")
            manifest.members.append(
                Member(
                    run_id=r.run_id,
                    index=len(manifest.members),
                    condition=agent,
                    protocol_id=pid,
                    seed=(
                        None if args.seed_base is None
                        else int(args.seed_base) + i
                    ),
                )
            )
            manifest.budget = budget.to_dict()
            write_manifest(store, manifest)
            sys.stderr.write(
                f"[seer]   {r.run_id} {r.view.state.value} "
                f"cost={_fmt_cost(r.view.cost_usd)}\n"
            )
            cost = r.view.cost_usd
            try:
                budget.charge_run(
                    pid,
                    r.run_id,
                    None if cost.absent else float(cost.value),
                    source_fidelity=cost.fidelity.value,
                )
            except SeerBudgetError as exc:
                stopped = str(exc)
                break
        if stopped:
            break

    manifest.budget = budget.to_dict()
    write_manifest(store, manifest)
    if stopped:
        sys.stderr.write(f"[seer] {stopped}\n")
    sys.stderr.write(f"[seer] {budget.summary()}\n")

    doc = build_ensemble(store, manifest).to_dict()
    if args.json:
        print(json.dumps(doc, indent=2))
    else:
        _print_ensemble(doc)
    if stopped:
        return 2
    return 0 if all(r.exit_code in (0, None) for r in results) else 1


def _cmd_ensemble(args: argparse.Namespace, store: EventStore) -> int:
    from .ensemble import build_ensemble, list_ensembles, read_manifest

    if not args.ensemble_id:
        rows = list_ensembles(store, args.limit)
        if not rows:
            sys.stdout.write("no ensembles\n")
            return 0
        for row in rows:
            sys.stdout.write(
                f"{row['ensemble_id']}  {row['n_members']}"
                f"/{row['n_runs_requested']} runs  {row['protocol_id']}  "
                f"{row['created']}\n"
            )
        return 0
    manifest = read_manifest(store, args.ensemble_id)
    if manifest is None:
        sys.stderr.write(f"[seer] unknown ensemble {args.ensemble_id}\n")
        return 2
    doc = build_ensemble(store, manifest).to_dict()
    if args.json:
        print(json.dumps(doc, indent=2))
    else:
        _print_ensemble(doc)
    return 0


def _cmd_attach(args: argparse.Namespace, store: EventStore) -> int:
    att = CodexAttachment(
        store=store,
        sock=args.sock,
        cwd=args.cwd,
        keep_reasoning=args.keep_reasoning,
        label=args.label,
        on_event=(_tick if args.progress else None),
    )
    try:
        att.open(prefer_daemon=not args.no_daemon)
    except ProtocolMismatch as exc:
        # The run exists and says why it is empty; this is the same refusal
        # read back out for someone standing at a terminal.
        sys.stderr.write(f"[seer] attach refused: {exc}\n")
        return 2

    sys.stderr.write(
        f"[seer] attached via {att.transport} — {protocol_note(att.protocol)}\n"
    )
    if args.prompt is None and att.transport == "own-app-server":
        sys.stderr.write(
            "[seer] no codex daemon is running, so there is no live session to "
            "watch. Pass a prompt to drive one, or start Codex first.\n"
        )

    try:
        res = (
            att.watch(args.timeout) if args.prompt is None
            else att.drive(args.prompt, model=args.model, timeout_s=args.timeout)
        )
    except KeyboardInterrupt:
        att.stop()
        res = att.close(outcome=Outcome.INTERRUPTED)
    if args.progress:
        sys.stderr.write("\n")
    _print_view(res.view)
    return 0


def _cmd_reconcile(args: argparse.Namespace, store: EventStore) -> int:
    since = (
        time.time() - args.since_days * 86400.0
        if args.since_days is not None else None
    )
    try:
        report = reconcile_codex(
            store=store, codex_bin=args.codex_bin, limit=args.limit,
            only_cwd=args.only_cwd, since=since,
            keep_reasoning=args.keep_reasoning,
        )
    except ProtocolMismatch as exc:
        sys.stderr.write(f"[seer] reconcile refused: {exc}\n")
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return 0

    print(f"\n  {len(report.imported)} imported, {len(report.skipped)} already "
          f"captured, {len(report.failed)} unreadable "
          f"(of {report.n_seen} considered)\n")
    for imp in report.imported:
        v = imp.view
        when = time.strftime("%Y-%m-%d %H:%M",
                             time.localtime(v.started_at or 0))
        # The token total is the reason this pass exists, so it goes on the
        # line — and prints `—` when the rollout had none, never 0.
        tok = v.usage.get("input")
        n = f"{tok.value:,}" if tok is not None and not tok.absent else "—"
        print(f"  {imp.run_id}  {when}  {imp.n_events:>5} events  "
              f"{v.state.value:<11} input {n}")
    for tid, run_id in report.skipped.items():
        print(f"  skipped {tid[:8]}… — already captured as {run_id}")
    for tid, why in report.failed.items():
        print(f"  failed  {tid[:8]}… — {why}")
    print()
    return 0


def _cmd_protocol(args: argparse.Namespace, store: EventStore) -> int:
    """The gate on its own, so "will attached mode work here" is answerable
    without capturing anything."""
    try:
        report = gate(args.codex_bin)
    except ProtocolMismatch as exc:
        if args.json:
            print(json.dumps({"compatible": False, "message": str(exc)}, indent=2))
        else:
            print(f"incompatible: {exc}")
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"compatible with the surface recorded for {report.get('golden_version')}")
    print(f"  {protocol_note(report)}")
    unmapped = report.get("unmapped_notifications") or []
    if unmapped:
        print(f"  {len(unmapped)} notification(s) this adapter does not read:")
        for m in unmapped:
            print(f"    {m}")
    return 0


def _tick(event) -> None:
    sys.stderr.write(".")
    sys.stderr.flush()


def _cmd_list(args: argparse.Namespace, store: EventStore) -> int:
    runs = store.list_runs(limit=args.limit, agent=args.agent)
    if not runs:
        print(f"no runs under {store.root}")
        return 0
    print(f"{'run_id':<24} {'agent':<8} {'state':<16} {'outcome':<24} events  label")
    for r in runs:
        print(
            f"{r.run_id:<24} {r.agent:<8} {(r.state or '—'):<16} "
            f"{(r.outcome or '—'):<24} {r.n_events:>6}  {r.label or ''}"
        )
    return 0


def _cmd_show(args: argparse.Namespace, store: EventStore) -> int:
    if store.get_run(args.run_id) is None:
        sys.stderr.write(f"unknown run {args.run_id!r}\n")
        return 2
    v = reduce_run(args.run_id, store.read(args.run_id))
    if args.json:
        print(json.dumps(v.to_dict(), indent=2, default=str))
    else:
        _print_view(v)
    return 0


# ── place (Attractors P2 / D5) ───────────────────────────────────────────────

_PLACE_DEFAULT_URL = "http://127.0.0.1:8123"


def _cmd_place(args: argparse.Namespace, store: EventStore) -> int:
    from .place import PlaceError, place_run, write_placement

    roles = tuple(r.strip() for r in args.roles.split(",") if r.strip())
    try:
        p = place_run(
            store,
            args.run_id,
            args.space_id,
            live_url=args.live_url or _PLACE_DEFAULT_URL,
            in_process=args.in_process,
            out_root=args.out,
            local_dir=args.local_dir,
            roles=roles,
        )
    except PlaceError as e:
        sys.stderr.write(f"{e}\n")
        return 2
    path = write_placement(store, p)
    if args.json:
        print(json.dumps(p.to_dict(), indent=2))
        return 0
    w = sys.stdout.write
    w(f"run     {p.run_id}\n")
    w(f"space   {p.space_id}  (layer {p.layer}, {p.model} @ {p.revision})\n")
    w(f"source  {p.source} via {p.transport}  fidelity={p.fidelity}\n")
    if p.pc1_evr is not None:
        w(f"control pc1_evr={p.pc1_evr:.4f} null_p95={p.pc1_evr_null_p95:.4f} "
          f"verdict={p.verdict}\n")
        if p.verdict != "above_null":
            w("        PC1 did not clear its null — these coordinates are a real\n"
              "        projection but NOT the default trajectory frame.\n")
    w(f"placed  {len(p.points)} turns\n")
    d = p.to_dict()
    if d["n_skipped"]:
        w(f"skipped {d['n_skipped']} "
          f"({d['n_dropped_by_policy']} dropped_by_policy, {d['n_missing']} missing)\n")
    w(f"wrote   {path}\n")
    return 0


def _print_comparison(views: list[RunView]) -> None:
    c = compare_views(views)
    w = sys.stdout.write
    w("\n" + "─" * 72 + "\n")
    w("COMPARISON\n\n")

    # Refusals first and unindented. Burying them under the table would make
    # the table look like the whole answer.
    if c.refused:
        w(f"Cannot be compared ({len(c.refused)}):\n")
        for r in c.refused:
            w(f"  ✗ {r.metric}\n    {r.reason}\n")
        w("\n")

    ids = c.runs
    w(f"{'metric':<30}" + "".join(f"{c.agents[i]:>16}" for i in ids) + "\n")
    for row in c.comparable:
        cells = "".join(f"{fmt(row.values[i]):>16}" for i in ids)
        w(f"{row.label:<30}{cells}\n")
    w("\n")


def _cmd_compare(args: argparse.Namespace, store: EventStore) -> int:
    views = []
    for rid in args.run_ids:
        if store.get_run(rid) is None:
            sys.stderr.write(f"unknown run {rid!r}\n")
            return 2
        views.append(reduce_run(rid, store.read(rid)))
    if args.json:
        print(json.dumps(compare_views(views).to_dict(), indent=2, default=str))
    else:
        _print_comparison(views)
    return 0


def _cmd_export(args: argparse.Namespace, store: EventStore) -> int:
    if store.get_run(args.run_id) is None:
        sys.stderr.write(f"unknown run {args.run_id!r}\n")
        return 2
    events = list(store.read(args.run_id))
    try:
        keep = parse_level(args.redact) if args.redact else None
        body, _ctype, filename = export_run(
            args.format, reduce_run(args.run_id, events), events, keep
        )
    except (ValueError, RuntimeError) as e:
        sys.stderr.write(f"{e}\n")
        return 2
    if args.out:
        Path(args.out).write_bytes(body)
        sys.stderr.write(f"wrote {args.out} ({len(body):,} bytes)\n")
        return 0
    if args.format in ("parquet",):
        # Binary down a pipe is a footgun on a terminal and a requirement in a
        # shell pipeline, so allow it only where it cannot scribble on a tty.
        if sys.stdout.isatty():
            sys.stderr.write(
                f"{args.format} is binary; give --out, or pipe stdout somewhere "
                f"(suggested name: {filename})\n"
            )
            return 2
        sys.stdout.buffer.write(body)
        return 0
    sys.stdout.write(body.decode("utf-8"))
    return 0


def _print_analysis(doc: dict) -> None:
    w = sys.stdout.write
    w("\n" + "─" * 72 + "\n")
    w(f"ANALYSES  {doc['run_id']}  ({doc['agent']}, {doc['capture_mode']}, "
      f"v{doc['analyses_version']})\n\n")
    for a in doc["analyses"]:
        head = a["headline"]
        val = head["value"]
        mark = {"estimated": "~", "heuristic": "?"}.get(head["fidelity"], "")
        shown = "—" if val is None else f"{mark}{val}{(' ' + a['unit']) if a['unit'] else ''}"
        w(f"  {a['label']:<30} {shown}\n")
        # A dash with no sentence beside it is the failure this whole subsystem
        # exists to avoid, so the reason is printed wherever the value is absent.
        why = a.get("refusal") or (head.get("note") if val is None else None)
        if why:
            w(f"      ↳ {why}\n")
        for name, m in a["parts"].items():
            v = m["value"]
            pm = {"estimated": "~", "heuristic": "?"}.get(m["fidelity"], "")
            w(f"      {name:<32} {'—' if v is None else f'{pm}{v}'}")
            if m.get("note"):
                w(f"   ({m['note']})")
            w("\n")
        # Rule and checklist rows are the analysis, not a detail of it: "0
        # matches" is only meaningful once you can see which rules could run.
        for row in a["rows"]:
            name = row.get("rule") or row.get("item")
            if not name:
                continue
            hits = row.get("hits", row.get("status"))
            w(f"      [{'—' if hits is None else hits}] {name}")
            if row.get("note"):
                w(f"  — {row['note']}")
            w("\n")
        # the formula, always: a number whose derivation is one line away is a
        # different object from one that is not
        w(f"      · {a['formula']}\n\n")


def _cmd_analyze(args: argparse.Namespace, store: EventStore) -> int:
    if store.get_run(args.run_id) is None:
        sys.stderr.write(f"unknown run {args.run_id!r}\n")
        return 2
    events = list(store.read(args.run_id))
    doc = analyze(reduce_run(args.run_id, events), events)
    if args.json:
        print(json.dumps(doc, indent=2, default=str))
    else:
        _print_analysis(doc)
    return 0


def _cmd_serve(args: argparse.Namespace, store: EventStore) -> int:
    from .server import serve

    root = store.root
    store.close()  # the server opens its own handle on the same root
    serve(args.host, args.port, root, watch=args.watch)
    return 0


def _cmd_reindex(args: argparse.Namespace, store: EventStore) -> int:
    n = store.reindex(args.run_id)
    print(f"reindexed {n} events from the log")
    return 0


def _cmd_delete(args: argparse.Namespace, store: EventStore) -> int:
    summary = store.get_run(args.run_id)
    if summary is None and not store.log_path(args.run_id).exists():
        sys.stderr.write(f"unknown run {args.run_id!r}\n")
        return 2
    if not args.yes:
        # What is about to go, before it goes. A run id is not a description,
        # and the one thing a delete must never do is surprise someone.
        if summary is not None:
            sys.stderr.write(
                f"{args.run_id}  {summary.agent} {summary.capture_mode}, "
                f"{summary.n_events} events"
                + (f", {summary.label}" if summary.label else "") + "\n"
            )
        sys.stderr.write("refusing to delete without --yes\n")
        return 2
    gone = store.delete_run(args.run_id)
    print(
        f"deleted {gone['run_id']}: {gone['events']} events, "
        f"{gone['bytes']:,} bytes of log"
    )
    return 0


# ── observed mode ────────────────────────────────────────────────────────────


def _print_plan(p, *, applied: bool) -> None:
    w = sys.stdout.write
    verb = "changed" if applied else "would change"
    w(f"\n{p.agent}  {p.config}\n")
    if not p.supported:
        w(f"  ✗ {p.reason}\n")
    if not p.changes:
        w(f"  nothing to change{' — already installed' if applied else ''}\n")
    for c in p.changes:
        mark = {"add": "+", "remove": "-", "create": "*", "manual": "!"}.get(c.kind, " ")
        w(f"  {mark} {verb}: {c.target}\n      {c.detail}\n")
    # Printed every time, even when the list is long: "we merged" is a claim,
    # and this is the evidence for it.
    for kept in p.preserved:
        w(f"  = kept: {kept}\n")
    for m in p.manual:
        w(f"\n  YOU MUST DO THIS PART:\n    {m}\n")


def _cmd_install(args: argparse.Namespace, store: EventStore) -> int:
    from . import install as inst

    agents = args.agents or list(inst.CONFIGS)
    if args.print_block:
        for a in agents:
            if a == "codex":
                print(inst.codex_block(store.root))
            elif a == "hermes":
                print(inst.hermes_block(store.root))
            else:
                print(json.dumps(inst.plan(a, store.root).to_dict(), indent=2))
        return 0
    if args.status:
        print(json.dumps(inst.status(store.root), indent=2))
        return 0

    if args.dry_run:
        for a in agents:
            _print_plan(inst.plan(a, store.root), applied=False)
        print("\n(dry run — nothing was written; re-run with --apply)\n")
        return 0

    for a in agents:
        plan, backup = inst.install(a, store.root, config=None)
        _print_plan(plan, applied=True)
        if backup:
            print(f"  backup: {backup}")
    print(
        f"\nShim: {store.root / 'spool'}"
        "\nCapture starts at each agent's next session. `seer watch` turns"
        "\nthe spool into runs; without it the hooks still write, and nothing reads.\n"
    )
    return 0


def _cmd_uninstall(args: argparse.Namespace, store: EventStore) -> int:
    from . import install as inst

    for a in args.agents or list(inst.CONFIGS):
        _print_plan(inst.uninstall(a, store.root, remove_spool=args.purge), applied=True)
    return 0


def _cmd_watch(args: argparse.Namespace, store: EventStore) -> int:
    from .collector import SpoolCollector

    c = SpoolCollector(
        store, store.root, from_start=args.from_start, idle_timeout_s=args.idle_timeout
    )
    if not c.reader.dir.is_dir():
        sys.stderr.write(
            f"no spool at {c.reader.dir} — run `seer install` first\n"
        )
        return 2
    res = c.reader.clock_resolution_s
    sys.stderr.write(
        f"watching {c.reader.dir}  (clock {res:g}s"
        f"{'' if res < 0.05 else ', too coarse for tool durations — they will be marked ~'})\n"
    )
    seen = 0
    try:
        while True:
            c.poll()
            if c.stats.events != seen and args.progress:
                sys.stderr.write(
                    f"\r{c.stats.events} events · {c.stats.runs_opened} runs · "
                    f"{len(c.runs)} open   "
                )
                seen = c.stats.events
            time.sleep(0.2)
    except KeyboardInterrupt:
        # Ending every open run here would claim they ended when the watcher
        # stopped. They did not — so they stay open, and `--reap` is the
        # explicit way to close them.
        sys.stderr.write("\nstopped watching. Open runs left open.\n")
        print(json.dumps(c.status(), indent=2))
    return 0


def _cmd_import_spool(args: argparse.Namespace, store: EventStore) -> int:
    from .collector import import_spool

    print(json.dumps(import_spool(store, store.root, idle_timeout_s=args.idle_timeout), indent=2))
    return 0


def _cmd_import(args: argparse.Namespace, store: EventStore) -> int:
    """Import a public corpus as reconciled runs (Attractors D7).

    The corpora are other people's data. Two rules are enforced here rather
    than left to the caller: the run is written with `capture_mode:
    reconciled`, and a `corpus.json` sits next to the events carrying the
    licence, the URL, the input file's sha256 and the count of records the
    mapping did not recognise. A run whose provenance is not written is not
    importable — there is no flag to skip it.
    """
    from .adapters import corpus_adapter
    from .adapters.corpus_base import CorpusError
    from .adapters.corpus_village import VillageUnavailable

    kw: dict[str, object] = {}
    if args.corpus == "amongus":
        kw = {"summary_path": args.summary, "max_games": args.limit}
    elif args.corpus == "ctfish":
        kw = {
            "labels_path": args.labels,
            "max_runs": args.limit,
            "model": args.filter_model,
            "variant": args.variant,
        }
    elif args.corpus == "village":
        kw = {"kind": args.kind, "max_rows": args.limit}

    try:
        adapter = corpus_adapter(args.corpus, run_id=args.run_id or "", session_id="")
        runs = adapter.read(args.path, **{k: v for k, v in kw.items() if v is not None})
    except VillageUnavailable as e:
        sys.stderr.write(f"{e}\n")
        return 2
    except (CorpusError, FileNotFoundError) as e:
        sys.stderr.write(f"{e}\n")
        return 2

    out = []
    for r in runs:
        if not args.dry_run:
            store.append_many(r.events)
            d = store.runs_dir / r.run_id
            d.mkdir(parents=True, exist_ok=True)
            (d / "corpus.json").write_text(
                json.dumps({**r.meta, "warnings": r.warnings}, indent=2) + "\n"
            )
        out.append(
            {
                "run_id": r.run_id,
                "n_events": len(r.events),
                "n_unmapped": r.meta.get("n_unmapped"),
                "n_warnings": len(r.warnings),
                "corpus": r.corpus.id,
                "licence": r.corpus.licence,
                "ships_in_repo": r.corpus.ships_in_repo,
            }
        )
    print(json.dumps({"dry_run": args.dry_run, "runs": out}, indent=2))
    for r in runs:
        for w in r.warnings:
            sys.stderr.write(f"{r.run_id}: {w}\n")
    return 0

# ── wiring ───────────────────────────────────────────────────────────────────
#
# `_add_subcommands` is the one place the sub-subcommand table is declared.
# It is shared by two callers that differ only in how the `seer` level of the
# parser comes to exist:
#   - `add_parser`, which grafts `seer` on as a subparser of someone else's
#     top-level parser (a role now unused inside this repo, since `nebulai`
#     no longer imports this module — kept for any other program that wants
#     to graft SessionSeer on the way this one used to);
#   - `main`, which *is* the top-level parser, for the standalone `seer`
#     console script.
# Either way, `--root` stays an argument of the `seer` level itself, parsed
# before the sub-subcommand name (`seer --root X serve`, not
# `serve --root X`) — that quirk falls out of `--root` being added to `p`
# before `p.add_subparsers()` runs, regardless of which caller built `p`.


def _add_subcommands(p: argparse.ArgumentParser) -> None:
    # The project-wide spend ceiling, shared with the namer and probe rather
    # than a second number that could drift from it (Attractors P3).
    from ..corpus import DEFAULT_MAX_COST_USD as _DEFAULT_MAX_COST_USD

    p.add_argument(
        "--root", default=None,
        help=f"event log root (default: {DEFAULT_ROOT})",
    )
    s = p.add_subparsers(dest="seer_cmd", required=True)

    r = s.add_parser("run", help="launch an agent headless and capture it")
    r.add_argument("agent", choices=["codex", "claude", "hermes"])
    r.add_argument("prompt")
    r.add_argument("--cwd", default=None, help="working directory for the agent")
    r.add_argument("--model", default=None)
    r.add_argument("--label", default=None, help="a name for this run")
    r.add_argument("--timeout", type=float, default=None, help="seconds before SIGTERM")
    r.add_argument(
        "--compare-with", nargs="+", metavar="AGENT",
        choices=["codex", "claude", "hermes"],
        help="also run the same prompt through these agents, then compare",
    )
    r.add_argument(
        "--keep-reasoning", action="store_true",
        help="store reasoning text. Off by default: it is retained only when "
             "asked for, and the resulting fields say dropped_by_policy when not",
    )
    r.add_argument("--progress", action="store_true", help="a dot per event on stderr")
    # ── Attractors P3: variance as the headline ──────────────────────────
    r.add_argument(
        "--repeat", type=int, default=1, metavar="N",
        help="run the same protocol N times and group them under one "
             "ensemble id. N > 1 prices itself through the budget before "
             "spending: the FIRST run of a protocol has no estimate at all "
             "(missing, not $0) and needs --acknowledge-unpriced; runs 2..N "
             "are estimated from what run 1 actually reported",
    )
    r.add_argument(
        "--seed-base", type=int, default=None, metavar="K",
        help="recorded per run as K, K+1, … and used to seed the split-half "
             "draws. It is NOT handed to the agent: none of codex, claude or "
             "hermes accepts a seed, and the ensemble says so rather than "
             "implying the fan is seeded",
    )
    r.add_argument(
        "--max-cost-usd", type=float, default=_DEFAULT_MAX_COST_USD,
        metavar="USD",
        help=f"ceiling for the whole repeat (default ${_DEFAULT_MAX_COST_USD:.2f}, "
             f"the project-wide one from corpus.py). Over it the repeat is "
             f"REFUSED; nothing is downgraded to fit",
    )
    r.add_argument(
        "--acknowledge-unpriced", action="store_true",
        help="proceed with a repeat whose cost is unknown. Required for the "
             "first repeat of any protocol, because Seer cannot see the "
             "agent's own billing; the acknowledgement is recorded in the "
             "ensemble so a reader knows the spend was never estimated",
    )
    r.add_argument(
        "--json", action="store_true",
        help="[--repeat] print the ensemble document instead of the summary",
    )
    r.set_defaults(seer_fn=_cmd_run)

    at = s.add_parser(
        "attach",
        help="capture Codex through its app-server: more of the session, and "
             "optionally none of the driving",
        description=(
            "Attached mode speaks `codex app-server` instead of reading "
            "`codex exec --json`, which is 68 notification kinds against 7 — "
            "approvals, mid-turn token usage, compaction and per-file line "
            "counts all become visible.\n\n"
            "With a PROMPT we drive one turn through our own server. Without "
            "one we join a running daemon, if there is one, and only watch. "
            "SessionSeer never starts a daemon and never approves anything on "
            "your behalf: an approval request is declined, and the log says a "
            "machine answered."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    at.add_argument("prompt", nargs="?", default=None,
                    help="omit to observe rather than drive")
    at.add_argument("--cwd", default=None)
    at.add_argument("--model", default=None)
    at.add_argument("--label", default=None)
    at.add_argument("--sock", default=None,
                    help=f"daemon control socket (default: {DEFAULT_SOCK})")
    at.add_argument("--no-daemon", action="store_true",
                    help="always spawn our own app-server, even if one is running")
    at.add_argument("--timeout", type=float, default=900.0)
    at.add_argument("--keep-reasoning", action="store_true")
    at.add_argument("--progress", action="store_true")
    at.set_defaults(seer_fn=_cmd_attach)

    rc = s.add_parser(
        "reconcile",
        help="import Codex sessions that already happened, without double-counting",
        description=(
            "Reads persisted threads through `thread/list` and `thread/read` "
            "— never resuming, archiving or deleting one — and imports the "
            "ones the store does not already hold. A thread already captured "
            "in any mode is skipped by the agent's own id and reported as "
            "skipped.\n\n"
            "Thread history has no per-item timestamps and no token counts. "
            "The counts are recovered from the session's rollout file; the "
            "timestamps are not recoverable, so item durations are reported "
            "absent rather than zero."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    rc.add_argument("--limit", type=int, default=25,
                    help="how many threads to consider, newest first")
    rc.add_argument("--only-cwd", default=None,
                    help="only threads whose session cwd is exactly this path")
    rc.add_argument("--since-days", type=float, default=None,
                    help="skip threads not touched in this many days")
    rc.add_argument("--codex-bin", default="codex")
    rc.add_argument("--keep-reasoning", action="store_true")
    rc.add_argument("--json", action="store_true")
    rc.set_defaults(seer_fn=_cmd_reconcile)

    pr = s.add_parser(
        "protocol",
        help="check this Codex build against the recorded method surface",
    )
    pr.add_argument("--codex-bin", default="codex")
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(seer_fn=_cmd_protocol)

    ls = s.add_parser("list", help="captured runs, newest first")
    ls.add_argument("--limit", type=int, default=30)
    ls.add_argument("--agent", default=None)
    ls.set_defaults(seer_fn=_cmd_list)

    sh = s.add_parser("show", help="one run, with its provenance")
    sh.add_argument("run_id")
    sh.add_argument("--json", action="store_true")
    sh.set_defaults(seer_fn=_cmd_show)

    # ── ensemble (Attractors P3): the fan a `--repeat` produced ──────────
    en = s.add_parser(
        "ensemble",
        help="the fan statistics over one `run --repeat` (no id: list them)",
        description=(
            "The statistics are recomputed from the runs' own logs on every "
            "read, never cached, so a run deleted since the repeat ran drops "
            "out of the fan and `n_runs` reports the true n.\n\n"
            "A quantity the run count cannot support comes back as missing "
            "with a reason — three runs have no p10, two runs per condition "
            "have no split half — rather than as a point estimate."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    en.add_argument("ensemble_id", nargs="?", default=None)
    en.add_argument("--limit", type=int, default=30, help="[list] how many")
    en.add_argument("--json", action="store_true")
    en.set_defaults(seer_fn=_cmd_ensemble)

    # ── place (Attractors P2 / D5: built in Nebul.AI, drawn in Seer) ─────
    pl = s.add_parser(
        "place",
        help="project a run's turns into a Nebul.AI persona space",
        description=(
            "Writes placement.json beside the run: one (pc1, pc2) coordinate "
            "per placeable turn, in a coordinate system that was frozen when "
            "the space was built. Nothing is fitted here.\n\n"
            "A turn whose text was not captured gets no coordinate and is "
            "listed under `skipped` with the reason — `dropped_by_policy` "
            "when the text existed and was refused at ingress, `missing` when "
            "there was nothing to capture. Neither becomes a point at the "
            "origin."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pl.add_argument("run_id")
    pl.add_argument("--space", required=True, dest="space_id", help="persona space id")
    pl.add_argument(
        "--live-url",
        default=None,
        help=f"live_server base URL (default: {_PLACE_DEFAULT_URL})",
    )
    pl.add_argument(
        "--in-process",
        action="store_true",
        help="load the model into THIS process instead of calling a live "
             "server. Not the default: placement is meant to cross the "
             "Nebul.AI/Seer boundary as HTTP and a file, and a Seer process "
             "holding resident model weights competes for RAM with the agent "
             "it is watching. Use it on a laptop with no server running",
    )
    pl.add_argument(
        "--out",
        default="out",
        help="[--in-process] Nebul.AI output root holding persona/<space>/space.json",
    )
    pl.add_argument(
        "--local-dir",
        default=None,
        help="[--in-process] weights directory for a space built from local weights",
    )
    pl.add_argument(
        "--roles",
        default="assistant",
        help="comma-separated turn roles to place: assistant, user, or both "
             "(default: assistant)",
    )
    pl.add_argument("--json", action="store_true")
    pl.set_defaults(seer_fn=_cmd_place)

    cp = s.add_parser("compare", help="compare runs, and refuse where it is not meaningful")
    cp.add_argument("run_ids", nargs="+")
    cp.add_argument("--json", action="store_true")
    cp.set_defaults(seer_fn=_cmd_compare)

    ex = s.add_parser("export", help="the append-only record, in a format that outlives us")
    ex.add_argument("run_id")
    ex.add_argument("--format", choices=[*FORMATS, "analysis"], default="jsonl",
                    help="jsonl is lossless; csv is spans only and says so")
    ex.add_argument("--out", default=None, help="write here instead of stdout")
    ex.add_argument(
        "--redact", choices=[l.value for l in ContentLevel], default=None,
        help="take the export down to a content level before writing it: "
             "'metadata' keeps identifiers, paths, counts and timing and drops "
             "commands and prose; 'command' also keeps command lines. Every "
             "dropped field leaves its length behind, and the filename says "
             "the export was redacted",
    )
    ex.set_defaults(seer_fn=_cmd_export)

    an = s.add_parser("analyze", help="derived analyses, with formulas and evidence")
    an.add_argument("run_id")
    an.add_argument("--json", action="store_true")
    an.set_defaults(seer_fn=_cmd_analyze)

    sv = s.add_parser("serve", help="HTTP + SSE for the viewer")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8125)
    sv.add_argument("--watch", action="store_true",
                    help="also collect your own sessions from the hook spool, "
                         "so `watch` and `serve` need not be two processes")
    sv.set_defaults(seer_fn=_cmd_serve)

    ri = s.add_parser("reindex", help="rebuild the SQLite index from the logs")
    ri.add_argument("--run-id", default=None)
    ri.set_defaults(seer_fn=_cmd_reindex)

    dl = s.add_parser(
        "delete",
        help="remove one run entirely — its log, its directory and its index rows",
        description=(
            "The log is append-only for events, not for runs: a record nobody "
            "can delete is a record nobody can be asked to keep. This removes "
            "everything about one run, so a later `reindex` cannot bring it "
            "back."
        ),
    )
    dl.add_argument("run_id")
    dl.add_argument("--yes", action="store_true",
                    help="required; without it the run is described and kept")
    dl.set_defaults(seer_fn=_cmd_delete)

    ins = s.add_parser(
        "install",
        help="register hooks so your own sessions are captured (observed mode)",
        description=(
            "Merges hook entries into each agent's own config, backing it up first"
            " and leaving every entry we did not write alone. Prints the plan and"
            " changes nothing unless you pass --apply."
        ),
    )
    # default=None, not []: for a nargs="*" positional argparse runs the default
    # itself through the choices check, so an empty list fails as "invalid choice:
    # '[]'". None skips that path and still parses to [], which the `or` below reads
    # as "every agent".
    ins.add_argument("agents", nargs="*", choices=["claude", "codex", "hermes"], default=None)
    ins.add_argument("--apply", dest="dry_run", action="store_false", default=True,
                     help="actually write the changes")
    ins.add_argument("--status", action="store_true", help="what is installed right now")
    ins.add_argument("--print-block", action="store_true",
                     help="print the config block for a config we will not edit for you")
    ins.set_defaults(seer_fn=_cmd_install)

    un = s.add_parser("uninstall", help="remove our hook entries and nothing else")
    un.add_argument("agents", nargs="*", choices=["claude", "codex", "hermes"], default=None)
    un.add_argument("--purge", action="store_true", help="delete the spool too")
    un.set_defaults(seer_fn=_cmd_uninstall)

    wa = s.add_parser("watch", help="turn the hook spool into runs, live")
    wa.add_argument("--from-start", action="store_true",
                    help="also read the backlog already in the spool")
    wa.add_argument("--idle-timeout", type=float, default=IDLE_TIMEOUT_S,
                    help="seconds of silence before a run is called interrupted")
    wa.add_argument("--progress", action="store_true", default=True)
    wa.set_defaults(seer_fn=_cmd_watch)


    ip = s.add_parser(
        "import",
        help="import a public corpus (Among Us / ctfish / AI Village / a transcript)",
    )
    ip.add_argument("corpus", choices=["amongus", "ctfish", "village", "transcript"])
    ip.add_argument("path", help="the corpus file. Nothing is downloaded here")
    ip.add_argument("--summary", default=None,
                    help="[amongus] summary.json, which carries the Impostor labels")
    ip.add_argument("--labels", default=None,
                    help="[ctfish] scoring/labels.json — Palisade's own published "
                         "classification, attached as theirs and never re-judged here")
    ip.add_argument("--variant", default=None, help="[ctfish] prompt variant filter")
    ip.add_argument("--filter-model", default=None, help="[ctfish] model id filter")
    ip.add_argument("--kind", default="computer_use_turns",
                    help="[village] which config file this is")
    ip.add_argument("--limit", type=int, default=None,
                    help="stop after this many games / runs / rows")
    ip.add_argument("--run-id", default=None, help="override the generated run id")
    ip.add_argument("--dry-run", action="store_true",
                    help="map and report, write nothing")
    ip.set_defaults(seer_fn=_cmd_import)

    im = s.add_parser("import-spool", help="import the whole spool once, after the fact")
    im.add_argument("--idle-timeout", type=float, default=60.0)
    im.set_defaults(seer_fn=_cmd_import_spool)

    p.set_defaults(fn=run)


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Graft `seer` on as a subparser of someone else's top-level parser."""
    p = sub.add_parser(
        "seer",
        help="SessionSeer: capture and compare Codex / Claude / Hermes runs",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_subcommands(p)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the standalone `seer` console script — `seer` IS the
    top-level parser here, rather than a subparser grafted onto `nebulai`'s.
    Exits with the same status codes the `_cmd_*` functions already return:
    `run` raises `SystemExit(code)` for a non-zero code and otherwise falls
    through, so returning 0 below covers the success case and a non-zero
    `SystemExit` from `run` propagates out of this function unchanged.
    """
    p = argparse.ArgumentParser(
        prog="seer",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_subcommands(p)
    args = p.parse_args(argv)
    run(args)
    return 0


def run(args: argparse.Namespace) -> None:
    store = EventStore(Path(args.root) if args.root else DEFAULT_ROOT)
    # Every verb, not just `serve`: a run left `running` by a crash is wrong on
    # `list` and wrong in an export too, and the pid check makes the sweep safe
    # to do from a second process while the first is still capturing.
    for r in recover_orphans(store):
        sys.stderr.write(
            f"recovered {r['run_id']}: was {r['was']}, {r['n_events']} events —"
            " the process capturing it is gone; recorded as interrupted\n"
        )
    try:
        code = args.seer_fn(args, store)
    finally:
        try:
            store.close()
        except Exception:
            pass
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    sys.exit(main())
