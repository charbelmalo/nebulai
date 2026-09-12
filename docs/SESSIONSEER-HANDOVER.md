# SessionSeer — handover

What exists, what does not, and the things that took a while to learn and would
otherwise have to be learned again. The design and its justification live in
[`SESSIONSEER.md`](SESSIONSEER.md); this file is about the state of the build.

As of 2026-09-12: **M0 through M5 are shipped and tested.** The build is
feature-complete against the plan. Four things §6 once listed as unbuilt now
exist (the novelty ledger behind `Effect.NO_NEW_INFORMATION`, `apply_patch`
line counts, git snapshots, and — as a written refusal with its evidence — the
Hermes TUI gateway); two remain, and §6 says which and why.

---

## 1. Where the code is

| | |
|---|---|
| Python | `src/nebulai/seer/` — ~11.6k lines incl. `adapters/` |
| Tests | `tests/test_seer_*.py` — 551 tests across 18 files |
| Viewer | `viewer/src/seer/{contract,client}.ts`, `viewer/src/chrome/SeerPage.tsx` |
| Styles | `viewer/src/styles/chrome.css` (`.seer-*`) |
| Fixtures | `tests/fixtures/seer/` — recorded output from all three agents, plus `vocabulary-golden.json` |

The reading order for someone new: `contract.py` (the vocabulary), `taxonomy.py`
(how a tool name becomes an action), one adapter, `reducer.py` (the fold),
`analysis.py` (the derived layer), `SeerPage.tsx` (all of it on a screen).

Two M5 modules sit off that path and are easier to read once the rest makes
sense: `redaction.py` (the field registry that computes an event's content
level, and the export-time redactor) and `recover.py` (what happens to a run
whose capture process died).

## 2. Running it

```bash
PYTHONPATH=src "$REPO"/.venv/bin/python -m nebulai.seer.cli serve --watch
```

`$REPO` is the repo root. The venv lives there, not in a worktree: `.venv/`
beside `src/`, so a worktree runs the root interpreter against its own `src`.
Seer is its own command, not a `nebulai` subcommand — `nebulai` never imports
`nebulai.seer` — so an installed venv can also just run `seer serve --watch`
directly (`.venv/bin/seer`, or `uv run seer serve --watch`). `--root <dir>`
sets the event-log root and must come **before** the subcommand
(`seer --root X serve`, not `serve --root X`) — it is an argument of the
top-level `seer` parser, not of `serve`.

Tests:

```bash
PYTHONPATH=src "$REPO"/.venv/bin/python -m pytest tests/ -q
```

```bash
cd viewer && npx vitest run && npx tsc --noEmit -p tsconfig.json
```

Current: **1079 Python tests** (1078 pass; the one failure,
`test_compare.py::test_palette_covers_every_built_map`, is data-dependent on
the local `out/` and unrelated to seer), **93 vitest, tsc clean.**

Regenerating the vocabulary golden — deliberately, never to make a red test
green (see §3, M5):

```bash
PYTHONPATH=src "$REPO"/.venv/bin/python tests/test_seer_vocabulary.py --write
```

CLI verbs: `run attach reconcile protocol list show compare export analyze serve
reindex delete install uninstall watch import-spool`. `export` takes
`--redact {metadata,command,content}`; `delete` refuses without `--yes` and
describes the run instead.
HTTP routes: `/seer/{health,runs,run/<id>,run/<id>/analysis,compare,export,live,
start,attach,reconcile,cancel,reindex,annotate}`, plus
`DELETE /seer/run/<id>`. `/seer/export` accepts `?redact=<level>`.

## 3. What each milestone actually delivered

**M0 — contract.** `contract.py` holds the event envelope, the 9-action ×
effect taxonomy, `Fidelity`, `Outcome`, `CaptureMode` and the privacy tiers. The
rule that keeps it honest is enforced by a test: no analysis may read
`e.native_type` or the `native` payload. Those fields exist for display and
audit; a metric that reads them has grown an agent-specific branch in the one
place that must not have one.

**M1 — driven triple + comparison.** Adapters for `codex exec --json`,
`claude -p --output-format stream-json` and `hermes -z`; JSONL store with a
SQLite index; the reducer; SSE; the SeerPage; and `compare.py`'s comparability
gate, which refuses on three grounds rather than subtracting numbers that do not
mean the same thing.

**M2 — observed mode.** A shell shim writes one JSON line per hook to a spool
directory; `seer serve --watch` folds the spool into runs. Installers merge into
each agent's own config, back it up, and restore byte-exactly.

**M3 — attached mode + reconciliation.** Two surfaces onto Codex that
`codex exec --json` cannot reach.

*Attached* (`seer attach`, `POST /seer/attach`) speaks `codex app-server` over
JSON-RPC: 68 notification kinds against `exec`'s 7. With a prompt we drive one
turn through our own process; without one we join a daemon that is already
running and only watch. `protocol.py` compares the live build's
`generate-json-schema` output against a golden fixture and **fails closed on a
removed method, open on an added one** — a method we never call appearing is
not our problem; one we do call disappearing is.

*Reconciled* (`seer reconcile`, `POST /seer/reconcile`) imports sessions that
already happened, through `thread/list` + `thread/read` with
`useStateDbOnly: true` so listing cannot trigger a metadata repair — a write.
Nothing is resumed, archived, forked or deleted. `CodexThreadReconciler`
subclasses the live `CodexAppServerAdapter` so a reconciled and an attached run
of the same session speak the same vocabulary rather than two dialects.

**M4 — analyses + export.** Seven analyses in `analysis.py`, each carrying a
version, a formula in words, the inputs it read and the evidence it read them
from; annotations as ordinary events; JSONL / Parquet / CSV / analysis-JSON
export from both the CLI and the viewer.

**M5 — hardening.** Five separate ways the build could have been quietly wrong.

*Content levels are computed, not asserted.* Every event carried
`content_level: "metadata"` — including events whose payload held the model's
prose verbatim. A label that is always the same is not a label. `redaction.py`
registers every payload key SessionSeer emits at a level, `event_level()` takes
the maximum over the keys actually present, and a test walks the entire suite
and fails on any key that is not registered, so a new agent field cannot reach
a log unclassified. `redact_event()` takes an export down to a requested rung
and replaces what it removes with the removed value's own length: a redacted
log never looks like a complete one. Secret scrubbing is separate and not
optional — it happens in `Event.__post_init__`, before the event can reach a
file.

*Deletion.* `seer delete`, `DELETE /seer/run/<id>`, and a two-step button in
the viewer. It removes the log, the run directory and the index rows, publishes
`run_deleted` on the bus so open pages sweep themselves, and is refused with a
409 while the run is still being captured.

*Crash recovery.* `store.orphans()` finds runs left mid-capture, `recover.py`
closes them, and the sweep runs on **every** CLI verb and at server start —
before anything can read them. A run left in `running` renders as live forever,
and the first thing a researcher does with a live run is wait for it to finish.

*Installer restore.* `test_seer_observed.py::TestRestore` pins the three
properties that are actually true rather than the plan's single "byte-exact"
claim: the backup is a byte-exact undo, a text config round-trips byte-exactly,
and a JSON config round-trips semantically. Uninstall removes **by tag**, so
hooks the user added after installing survive it.

*Vocabulary compatibility.* `vocabulary-golden.json` records, per adapter, what
each of the 57 native kinds produced the day it was captured — event types and
action. Same rule as the protocol gate: fail closed on removal, open on
addition.

## 4. Decisions worth not re-litigating

**The plan's "hooks add < 5 ms p95" exit criterion was replaced.** Measured on
this machine, a bare `zsh -c true` costs **4.7–6.0 ms**. That is the process
spawn, and it is charged to any hook the agent runs at all, including one that
does nothing. Our shim adds **≈0.36 ms** on top. A p95 budget of 5 ms is
therefore not a statement about SessionSeer — a run could blow it with our code
deleted. The criterion in force is *the shim's own cost*, which is the only part
we control, and the spool write is append-only and never blocks on the server:
the agent is unaffected when nothing is listening. (Timing used zsh's
`$EPOCHREALTIME`; `date +%s%N` is not microsecond-capable on macOS.)

**Edit line counts are captured at ingress, from the tool *input*.** `Write` and
`Edit` inputs contain the text; counting its newlines at the adapter and keeping
only the integers means churn is computable without any file content entering
the log. This stays inside the `metadata` privacy tier. `taxonomy.edit_extent()`
returns `None` for tool shapes that carry no line information, and that `None`
is what makes `edit_churn` say "missing" instead of `0.0`. Codex's `apply_patch`
used to be one of those shapes; it is now counted one level up, by
`adapters/patches.py`, from the patch envelope the model wrote (see §6).
`edit_extent()` itself still returns `None` for it, deliberately — it reads a
tool input and has no envelope parser, and two layers counting the same edit
would be worse than one layer declining to.

**`no_new_information_streak` reports `missing` when nothing was decidable, not
`0`.** The live adapters now emit `Effect.NO_NEW_INFORMATION`, but only under
the three `NoveltyLedger` rules in §6 and never on a guess, so a run in which no
span carried an information effect still reports `missing` — reporting `0` would
say "we looked and there were none". When the rule does run it also reports the
weakest `effect_fidelity` among the spans it counted and the `rules_used`, so a
streak assembled from the heuristic rule cannot be read as a counted one. The
second loop rule, `repeat_read_without_change`, was designed to be decidable
from the log alone: if nothing edited the target between two identical reads,
the second read cannot have returned anything new.

**`progress_evidence` has no headline number, deliberately.** Its eight items
are not commensurable and "68% done" would be a worse instrument than the
checklist. Two of its rows must never collapse: *the agent said it was done* is
evidence about the agent; *that verification passed* is evidence about the task.

**Parquet is written with an explicit `pa.schema`.** Left to inference, a column
that is null for every row of one run lands as `null`-typed, and concatenating
two runs in pandas then fails on a type mismatch — which is exactly the case one
agent reports turns and another does not.

**`/seer/export?format=parquet` returns 501 when pyarrow is absent**, not 400
and not a silent downgrade to CSV. The request was fine; this install is not.

**Cumulative usage replaces; it never folds.** The app-server's
`turn/token-count` carries a *running total*, not a delta. Folding it summed the
same tokens once per notification and produced numbers several times the truth
on a long turn. `MODEL_USAGE_UPDATED` payloads therefore carry
`authoritative: True`, and the reducer replaces rather than adds when it sees
it. This was a live bug, found by comparing an attached run against the same
session's rollout file.

**The dedup key must be restorable from the log alone.** A run imported by the
reconciler records the agent's own session id in `SESSION_STARTED`, and
`store._index()` lifts it back out into `runs.native_session_id`. Keeping it
only in the SQLite row would mean a `reindex` — which is advertised as
rebuildable-from-logs — silently rearms every past import for a second pass.
The test that matters is the one that reindexes and *then* imports again.

**A reconciled run is dated from the session, not from `now`.** `reduce_run` is
finalised against the last event's own timestamp. Against the wall clock it
would compute "quiet for four months" for every thread on disk and hang a
STALLED overlay on the entire import.

**Capture mode is not fidelity.** `CaptureMode` answers "do we own the
process" (driven / attached / observed / reconciled); the *adapter* answers "how
much can we see". They are separate because an observed run and a reconciled run
are both second-hand but blind in different places: hooks see actions and timing
and never tokens; thread history sees tokens and turn boundaries and never
per-item timing. A single "fidelity" scalar would have to pick one lie.

**Deletion is total, and refused while live.** A `delete` that left the JSONL on
disk would come back on the next `reindex`, so it removes the log, the run
directory and the rows together, and closes the append handle rather than
leaving a writer pointed at nothing. It will not delete a run that is still
being captured — cancel it first. The 409 says so; it does not race the writer
and hope.

**Liveness is decided by the recorded pid, not by a timeout.** A run can be
legitimately silent for an hour — a long build, a human thinking — so "no
events for N minutes" cannot distinguish a slow run from a dead one. "The
process that was writing this is gone" is a fact, checked with `os.kill(pid,
0)`. It is also what makes the sweep safe to run from a second process: `seer
list` in another terminal will not reap a capture that another shell is still
driving. A NULL pid means the row predates the column and is swept.

**The recovery event is ours, and dated from the run.** `recover.py` writes
`SESSION_INTERRUPTED` under `adapter: "seer_recovery"` with `native: None` — it
is SessionSeer's inference, not something an agent said — timestamped from the
run's last real event rather than from `now`, for the same reason a reconciled
run is. The outcome is `interrupted` and never `completed`: we know the process
died, and we do not know whether the work finished.

**The repair is written to the log, not to the index.** A `state` fixed only in
SQLite is undone by the next `reindex`, which is advertised as
rebuildable-from-logs. The log is the record; the index is a cache. That
principle is what the recovery test asserts — reindex, *then* check the state.

**Uninstall removes by tag; it does not restore the backup.** Restoring would
discard anything the user changed after installing. `_strip_block` deletes our
tagged block and the blank line install wrote above it — the separator is part
of what we wrote, so it leaves with the rest. Take it and the round trip is
byte-exact; leave it and every install/uninstall cycle grows the file by one
line.

## 5. Gotchas that cost time

- **A running `seer serve` does not reload changed code.** An adapter fix that
  "did not work" was a stale process. `pkill -f "seer serve"`, restart,
  re-fire.
- The Bash tool's working directory resets to the worktree root between calls —
  use absolute paths for anything under `viewer/`.
- Seer is its own app entry: open `seer.html` (not `index.html#page=seer` — the
  Nebulai entry drops any page it does not own). Its three pages are Live
  (`#page=seer`), Transcripts (`#page=sessions`) and Topics (`#page=snapshot`);
  changing `location.hash` on an already-loaded page does not re-route, so
  reload or click the nav pill.
- There may be more than one vite on the box. Use the port `preview_list`
  reports for *this* worktree; another worktree's dev server will happily serve
  a stale bundle that looks almost right.
- **A vite HMR update that throws leaves the page half-alive.** After a
  `ReferenceError` during a hot update, some components keep re-rendering and
  others freeze on stale props — clicking a run appeared to do nothing while a
  sibling panel updated fine. It is not a bug in the page. `location.reload()`;
  `navigate` to the same URL may not reload at all.
- **`CREATE INDEX` on a new column cannot live in the schema script.**
  `CREATE TABLE IF NOT EXISTS` skips an existing table, so the column is not
  there yet when the index statement runs, and the whole `executescript` fails
  on every old store. Column adds and their indexes both belong in `_migrate()`,
  in that order.
- **`SeerState.runners` never dropped a finished run**, and `/seer/health` hid
  it: health filtered on `proc.poll() is None`, so the leak was invisible right
  up until something *else* asked that dict whether a run was live — and every
  run that had ever finished said yes. Deletion would have returned 409 forever.
  The fix is a `finally: runners.pop(...)` under the lock, the way `attach`
  already handled its own map. The lesson is smaller than the bug: a cache that
  only one reader consults is only correct for that reader.
- **`var(--bad)` is not a design token.** The destructive colour is
  `var(--danger)`. An undefined CSS variable does not fail — the property is
  simply dropped, so a delete button renders in the inherited text colour and
  looks deliberate.
- **`hermes --version` prints a build date in parentheses** right after the
  version, and taking the second whitespace token gave every Hermes run an
  `agent_version` of `2026.6.5)`. Since `agent_version` is stamped on every
  event and is what a later comparison keys on, that silently partitioned
  Hermes runs into versions that never match. `runner.parse_version` now takes
  the first dotted numeric token, brackets stripped, and is tested on all three
  agents' real output.

## 6. What is left

Nothing from the milestone plan. Four things §6 used to list as *not built* now
exist; what remains is below them, with the reason each one stayed out.

### Built since the first handover

- **`Effect.NO_NEW_INFORMATION` is emitted, but only where a payload decides
  it.** `adapters/novelty.py` holds a `NoveltyLedger` with exactly three rules
  and no fourth: **R-A `repeat_read`** — the same lookup key
  (tool name plus the arguments that determine its answer) seen twice with no
  intervening edit, `DETERMINISTIC`; **R-B `identical_output`** — this call's
  output is byte-identical to the previous invocation's, compared by SHA-256 so
  no text is retained, `DETERMINISTIC`; **R-C `zero_result_search`** — a search
  whose result text matches a sentinel list ("no matches found", "no files
  found", …), `HEURISTIC`, because the sentinel list is a stated rule over a
  proxy and not the payload itself. Anything else gets no label at all: the
  adapters never guess. Two fidelities are carried side by side and must not be
  merged — `Source.fidelity` still describes the *event* (`NATIVE` when the
  agent told us the call finished), and a new payload pair
  `effect_fidelity` / `effect_rule` describes the *effect label we attached*.
  Collapsing them would either downgrade a native event or launder a guess as a
  fact. `SpanRecord` carries both through the reducer, and
  `no_new_information_streak` now reports `hits`, the **weakest** fidelity of
  any labelled span, and the `rules_used` that produced them — so a streak built
  out of R-C reads `heuristic` at the analysis layer and cannot be mistaken for
  a counted one. A run in which nothing was decidable still reports `missing`,
  never `0`. Invalidation is scoped on purpose: a lookup that names a file is
  retired only by an edit to *that* file; a search that ranges over a tree the
  payload does not bound is retired by **any** edit, because the conservative
  error (declining to label) is the cheap one.
- **`edit_churn` counts Codex `apply_patch` hunks.** `adapters/patches.py`
  parses the `*** Begin Patch` envelope — which is not a unified diff — and
  returns a per-file `FileEdit`. Counted files are `DETERMINISTIC`. A file whose
  extent cannot be counted exactly is `MISSING` with a note, never `0`: a
  `*** Delete File:` directive says a file is gone and never says how big it
  was, an unrecognised hunk-body line poisons only its own file, and a patch
  with no `*** End Patch` makes every file in it missing because we cannot tell
  which one was truncated. One uncountable file does not poison its counted
  siblings. `taxonomy.edit_extent()` still returns `None` for `apply_patch` —
  it reads a tool *input* and has no envelope parser, and two layers counting
  the same edit would be worse than one layer declining to. Double counting is
  prevented by a consume-once `_patch_extents` ledger in the Codex adapters: a
  native `file_change` item for a path a shell `apply_patch` already counted
  pops one pending entry and emits nothing (it carries no counts anyway), so a
  genuinely later edit to the same file still counts. A `file_change` item with
  nothing pending is emitted with `lines_fidelity: "missing"`.
- **Git snapshots bracket every driven run.** `runner.git_snapshot()` records,
  at run start and immediately before run completion, the HEAD sha and a
  SHA-256 of `git status --porcelain` (the hash, not the paths — a researcher's
  dirty file names are not something a captured run needs to carry, and
  equality is all the comparison needs). They ride on the existing store as
  `EventType.GIT_SNAPSHOT` events distinguished by `phase`; there is no second
  store. Git is addressed as `/usr/bin/git -C <root>` with an explicit `cwd`,
  never a bare `git`, because the agent's environment is passed through
  unchanged by design and its `PATH` is not ours to trust for a provenance
  record. The honesty rule here is one line wide: `git status --porcelain`
  prints nothing for a clean tree **and** prints nothing when it errors, so the
  helper returns `None` and never `""`. Outside a repository, or when git fails
  or times out, `head`/`status_hash`/`status_lines` are `None`,
  `head_fidelity`/`status_fidelity` are `MISSING`, `dirty` is `None` (not
  `False`), and a `note` states the reason. A repository with no commits has no
  HEAD and a perfectly readable tree, and reports exactly that — half present,
  half missing. `_repo_context()` still returns `None` outside a repository,
  because `store._index` and `attach.py` both read `None` as "no repo" and that
  contract is older than this change.
- **The Hermes TUI gateway was investigated and deliberately not built.** The
  paragraph below is the reason, in full, so nobody re-opens it on a hunch.

### Why there is no attached Hermes capture

An *attached* capture needs one thing SessionSeer cannot get from Hermes: a
read-only side channel onto a session it did not itself start. Codex provides
exactly that — `codex app-server` is a long-lived JSON-RPC server with
`thread/list` and `thread/read`, so the app-server adapter can enumerate threads
it never drove and replay their items. Hermes has four surfaces and none of them
is that.

`hermes -z` prints only the final response text; there is no event stream to
parse, which is why `HermesOneshotAdapter` emits lifecycle events and nothing
else, and why `--pass-session-id` is no help (it injects the id into the
*system prompt*, not into stdout). The ACP surface (`hermes acp --check`) is a
bidirectional JSON-RPC **client** protocol: it carries `tool_call`,
`agent_thought_chunk`, `agent_message_chunk` and `plan`, but no token usage and
no thread-history or replay method — it is a way to *be* the front end, not a
way to watch one. The TUI gateway itself (`~/.hermes/hermes-agent/tui_gateway/`:
`entry.py`, `server.py`, `ws.py`, `transport.py`, `event_publisher.py`) is
spawned by `hermes --tui` over **stdio**; it opens no listener and binds no
port, so there is nothing to connect to. `tui_gateway/ws.py::handle_ws` is a
*mountable handler*, not a server, and its single mount is the dashboard's own
FastAPI app (`hermes_cli/web_server.py:9729`, `@app.websocket("/api/ws")`),
gated behind `_DASHBOARD_EMBEDDED_CHAT_ENABLED`, `_ws_auth_ok` and
`_ws_request_is_allowed` — so reaching it means standing up and authenticating
against the Hermes dashboard, not attaching to a TUI.

Even granted a connection, the gateway is architecturally single-tenant.
`tui_gateway/server.py:838` — `dispatch(req, transport)` — does
`t = transport or _stdio_transport; token = bind_transport(t)`, which pins every
write produced by a request, *including events the handler emits*, to the
transport that asked; and `_close_sessions_for_transport` (`server.py:539`)
shows sessions are owned by the transport that created them. A second client
therefore observes only the sessions it drives itself, which is a *driven*
capture with extra steps, not an attached one. The one genuine fan-out is the
`/api/pub` sidecar, and it is reachable only via `HERMES_TUI_SIDECAR_URL`, read
from the environment at TUI process start (`ui-tui/src/gatewayClient.ts:40`)
and set by the dashboard's `/api/pty` when *it* spawns the child
(`web_server.py:9462`) — so it cannot be attached to an already-running TUI.
Finally, `session.history` (`server.py:4463`) needs a live in-process session
and otherwise falls back to reading `state.db`, which SessionSeer's
`HermesStateDbReconciler` already reads directly and at a lower fidelity it
declares. Hermes is therefore captured driven (`hermes -z`), observed (hooks),
or reconciled (`state.db`) — never attached, and nothing pretends otherwise.

### Still not built

- **The vocabulary golden covers the two stream adapters and the three hook
  adapters, not the app-server.** Its 68 notification kinds are pinned by
  `protocol.py` against the build's own generated schema, which is a stronger
  check than a recorded replay — but it is a different mechanism, and the two
  are not one gate.
- The CSV export is spans-only by design, and says so in its own first line. If
  someone wants a flat *event* CSV, that is a new format, not a change to this
  one.
