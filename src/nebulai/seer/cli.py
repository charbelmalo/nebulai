"""`nebulai seer …` — the command-line half of SessionSeer.

    nebulai seer run codex "fix the failing test"     # capture one run
    nebulai seer run claude "…" --compare-with codex  # the same task, twice
    nebulai seer attach "fix the failing test"        # Codex, at app-server fidelity
    nebulai seer attach                               # …or just watch a running one
    nebulai seer reconcile --limit 50                 # import sessions already on disk
    nebulai seer protocol                             # is this Codex build supported?
    nebulai seer list                                 # what has been captured
    nebulai seer show <run_id>                        # one run, with provenance
    nebulai seer compare <run_a> <run_b>              # and what cannot be compared
    nebulai seer export <run_id> > run.jsonl          # the raw record
    nebulai seer serve                                # HTTP + SSE on :8125

    nebulai seer install --apply                      # capture your *own* sessions
    nebulai seer watch                                # …and turn them into runs

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
        "\nCapture starts at each agent's next session. `nebulai seer watch` turns"
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
            f"no spool at {c.reader.dir} — run `nebulai seer install` first\n"
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


# ── wiring ───────────────────────────────────────────────────────────────────


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "seer",
        help="SessionSeer: capture and compare Codex / Claude / Hermes runs",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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

    im = s.add_parser("import-spool", help="import the whole spool once, after the fact")
    im.add_argument("--idle-timeout", type=float, default=60.0)
    im.set_defaults(seer_fn=_cmd_import_spool)

    p.set_defaults(fn=run)


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
