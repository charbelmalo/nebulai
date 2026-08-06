# Setting up the live view

What to install, what to run, and — the part that takes longest to learn — how
to read a surface that refuses to guess. The design is in
`docs/SESSIONSEER-LIVE.md`; this is the operating manual.

Nothing here needs a GPU, a network, or an API key. The one optional piece
(the field) degrades to absence.

---

## 1. The server

Everything on the Seer page comes from one process:

```bash
uv run nebulai seer serve --watch
```

HTTP + SSE on `127.0.0.1:8125`. `--watch` folds the hook-spool collector into
the same process, so capturing your own sessions does not need a second
terminal. Check it:

```bash
curl -s http://127.0.0.1:8125/seer/health | python -m json.tool
```

`runs` is how many are in the store; `observing.watching` is whether your own
sessions are being collected; `sse_clients_dropped` above zero means the stream
is shedding subscribers and the live view will be missing events.

## 2. The viewer

```bash
cd viewer && npm run dev
```

Then open the **Seer** tab. The page talks to whatever `seerUrl` says, which
defaults to `http://127.0.0.1:8125` and is editable in **Settings → probing**.
For a build, set it at build time:

```bash
VITE_SEER_URL=http://127.0.0.1:8125 npm run build
```

Point it at a blank string deliberately and the page says it is not configured —
but note the offline hint still suggests starting `nebulai seer serve`, which is
wrong advice for a deliberately-blank endpoint. Known wart.

## 3. Getting runs into it

Four capture modes, and the live view behaves differently in each. This matters
more than it sounds: **`capture_mode` is the single best predictor of which
honest-absence state you will be looking at.**

### driven — you launch the agent

```bash
uv run nebulai seer run codex "fix the failing test"
uv run nebulai seer run claude "…" --keep-reasoning
uv run nebulai seer run codex "…" --compare-with claude
```

Highest fidelity. Every event arrives live, so the chart, the field and the
thought rail all fill in as it goes.

### attached — Codex through its app-server

```bash
uv run nebulai seer attach "fix the failing test"   # drive it
uv run nebulai seer attach                          # or just watch
```

More of the session than `run` sees, including per-call native durations.

### observed — your own sessions, via hooks

```bash
uv run nebulai seer install --status          # what is installed
uv run nebulai seer install --apply           # merge hooks into agent configs
uv run nebulai seer install --print-block     # for a config we will not edit
```

`install` backs up each config first and leaves entries it did not write alone.
After that, `serve --watch` (or a separate `nebulai seer watch`) turns the spool
into runs as you work.

### reconciled — Codex sessions that already happened

```bash
uv run nebulai seer reconcile --limit 50
uv run nebulai seer reconcile --since-days 7 --only-cwd "$PWD"
```

Imports thread history without double-counting: a thread whose
`native_session_id` is already in the store is skipped, and the pass reports
`n_skipped` beside `n_imported` so a mostly-skipped run does not look like a
mostly-empty one.

**Reconciled runs are the most misread thing on the page.** Thread history has
no per-item timestamps, no per-item token counts, and no reasoning text. So
every span gets a synthetic start, every duration reads absent, and the thought
rail shows *"no text in the record"* — not "dropped by policy", and not a size
of `0`. Re-importing with `--keep-reasoning` changes none of that, because the
text was never in the history to keep.

## 4. `--keep-reasoning`

Off by default, and the default is the interesting state.

| you passed | the agent sent text | rail shows |
|---|---|---|
| `--keep-reasoning` | yes | the words |
| nothing | yes | `1,284 characters dropped by policy` |
| either | no field at all | `no text in the record` |

The middle row is a decision *you* made and the rail draws it in policy ink. The
third is the agent's, drawn in missing ink. They are never merged, and neither
is ever a blank box — a rail with nothing in it would read as "the model wasn't
thinking", which is the one thing none of these states mean.

Reasoning text is still redacted at `METADATA` content level even when kept.
Opting in does not opt out of redaction.

## 5. Reading the live surface

### The three y-modes

One set of events, grouped three ways. The pills switch what `y` means; the
marks travel rather than being redrawn.

| mode | `y` | read it for |
|---|---|---|
| **score** | action lane | what kind of work, over time |
| **fleet** | one row per run | who is doing what, across agents |
| **structure** | containment | what ran inside what |

Scroll to zoom the time window, drag to look back, **fit** to frame everything,
**live** / **resume** to follow or park.

Structure mode will usually draw one flat row of calls under each run, and the
card says why: depth comes from `parent_span_id` and nothing else, and no
adapter captured so far reports nesting. The pale band on each run's own row is
its wall time; the part no call covers is `outside_spans_s` — the model thinking
and you reading.

### The field

The **field** pill toggles the glow behind the chart. It needs WebGPU; without
it the pill is disabled and the chart is exactly as legible.

It carries **no magnitude**. An event says that something happened and when,
never how big. Brightness is recency; anything brighter than that is events
crowding together. If you find yourself reading intensity as volume, the field
is doing the one thing it was built not to claim — turn it off.

### The thought rail

Newest first, capped, and it says what it capped (`newest 24 of 139`). A
duration appears only when both ends were observed — Claude reports a thinking
block once it is finished, so those print `—` rather than `0.0s`. An open stream
on a finished run reads **no end**: the run ended before the stream closed, which
is a gap in the capture, not thinking that is still going.

The rail reads each selected run's log back once, so a run that finished before
you opened the tab still shows its thoughts.

## 6. Reading absences

The whole surface is built so that "we don't know" cannot be mistaken for a
number. Four inks, four meanings:

| you see | it means |
|---|---|
| `—` | there is no value; a `0` here would be invented |
| hollow diamond | the call finished but nobody clocked it |
| policy ink (dashed) | we chose not to capture it |
| missing ink (solid, hollow) | the agent never reported it |

`compare` refuses rather than averages across incomparable runs, and prints the
refusal above the table. That is the finding, not a caveat about it.

## 7. What is not there

The Atlas trail. The atlas keeps its coordinates and discards the UMAP model
that produced them, so there is no projection to place a live event with — see
§7 of `docs/SESSIONSEER-LIVE.md` for what would have to change. Nothing on the
Seer page depends on it.

## 8. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| page says no seer server | `seerUrl` empty or wrong | Settings → probing, or `VITE_SEER_URL` |
| link dot never goes green | `serve` not running, or a port clash | `curl …/seer/health` |
| runs appear but never update | SSE dropped | check `sse_clients_dropped`; reload the tab |
| field pill disabled | no WebGPU on this device | nothing to fix; the chart is complete without it |
| chart blank in a background tab | `document.hidden` stops the rAF loop | foreground the tab |
| every duration reads `—` | reconciled run | expected; thread history has no clocks |
| thought rail empty | no reasoning events in the log | expected for agents not emitting thinking blocks |
| `N older events have left the window` | the live window is bounded on purpose | the full log is on disk; `seer export <run_id>` |

## 9. Stopping

```bash
uv run nebulai seer uninstall            # remove the hooks
uv run nebulai seer uninstall --purge     # …and the spool with them
uv run nebulai seer delete <run_id> --yes # remove one run: log, dir, index rows
```

`delete` needs `--yes`; without it the run is described and kept. Over HTTP it
refuses with a 409 while the run is still being captured — cancel it first. The log is the record — nothing rewrites a captured run to match a later
opinion, which is why a run captured before a fidelity fix keeps its old labels
until it is re-imported.
