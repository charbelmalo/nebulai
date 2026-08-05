/** SeerPage — SessionSeer's viewer. Live capture and honest comparison of
 *  Codex / Claude / Hermes runs, served by `nebulai seer serve` on :8125.
 *
 *  Two rules shape everything on this page, and both are about refusing to
 *  flatter the data:
 *
 *  1. **An absent number is never a zero.** Every figure arrives as a
 *     `Measured` — a value plus the fidelity that produced it. `missing` ("the
 *     agent never said") and `dropped_by_policy` ("we chose not to look") are
 *     drawn as an em dash carrying its reason, and they are drawn *differently
 *     from each other*, because a researcher's next action differs: one is a
 *     capability gap in the agent, the other is a setting on our side.
 *
 *  2. **A refusal is a result.** When the comparability gate declines to
 *     compare two runs on tokens, that refusal is rendered above the table at
 *     full weight, not as a footnote under it. "These two runs cannot be
 *     compared on output tokens because Claude bills reasoning inside output
 *     and Codex reports it beside" is the finding — the table below it is the
 *     smaller half of the answer.
 *
 *  No fold happens here. The Python reducer owns every derived number; when a
 *  live event lands, this page refetches the affected run's view rather than
 *  incrementing anything locally. A second implementation of the fold in TS
 *  would drift from the first, and the drift would be invisible.
 */

import { computed, signal, useSignal } from "@preact/signals";
import { useEffect } from "preact/hooks";
import { ACTIONS, isAbsent, type Action, type Measured, type SeerEvent } from "../seer/contract";
import { ACTION_COLOR, stateInk } from "../seer/encoding";
import { LiveModel } from "../seer/live";
import { SeerScore } from "./SeerScore";
import {
  $health,
  $link,
  $linkError,
  annotate,
  cancelRun,
  connectLive,
  deleteRun,
  exportUrl,
  fetchAnalysis,
  fetchComparison,
  fetchHealth,
  fetchRun,
  fetchRuns,
  fmtCount,
  fmtSeconds,
  fmtUsd,
  FIDELITY_MARK,
  FIDELITY_TITLE,
  attachCodex,
  reconcileCodex,
  startRun,
  type Analysis,
  type Comparison,
  type ExportFormat,
  type RedactLevel,
  type RunAnalyses,
  type RunSummary,
  type RunView,
  type SpanRecord,
  type StartRequest,
} from "../seer/client";

// ── page state ───────────────────────────────────────────────────────────────

const $runs = signal<RunSummary[]>([]);
/** run_id → the server's reduced view. The only place a run's numbers live. */
const $views = signal<Record<string, RunView>>({});
/** Checked in the rail. One = detail; two or more = comparison. */
const $selected = signal<string[]>([]);
const $comparison = signal<Comparison | null>(null);
/** Last ~200 raw events per run, newest last — the "what is it doing right
 *  now" tail. Bounded because a long run emits thousands and the tail is the
 *  only part anyone reads. */
const $tail = signal<Record<string, SeerEvent[]>>({});
const $launchError = signal<string | null>(null);
/** An import pass is running server-side. Module-level rather than component
 *  state because the pass outlives any one render and its result arrives on the
 *  live stream, which is subscribed once for the whole page. */
const $importing = signal(false);
const $importNote = signal<string | null>(null);
/** run_id → the derived analyses, and the event count they were computed from.
 *  Deliberately not refetched on every event: re-reducing a long log per frame
 *  would make the page the slowest thing watching the agent. The panel says
 *  which snapshot it is showing and offers to recompute. */
const $analyses = signal<Record<string, RunAnalyses>>({});

/** The leading edge: what has happened since the last snapshot arrived.
 *
 *  Deliberately not a signal. It changes on every event and a busy agent lands
 *  dozens a second; re-rendering the page that often is the thing `markDirty`
 *  exists to avoid. The canvases that read it redraw every frame anyway and
 *  pull from it there, so nothing is ever stale and nothing re-renders.
 *
 *  It holds no figures. `adopt` hands it the server's snapshot and it gives
 *  that same object back untouched — the reducer owns every derived number,
 *  and a second fold on this side would drift from the first invisibly. */
const liveModel = new LiveModel();

const $selectedViews = computed(() =>
  $selected.value.map((id) => $views.value[id]).filter((v): v is RunView => !!v),
);

const TAIL_MAX = 200;

/** Colours come from `seer/encoding.ts`, which is the only place allowed to
 *  decide what a hue means. This page and the live view draw the same run in
 *  different geometries, and the moment they disagree about what green is, a
 *  glance between them misleads. Same family as the Sessions field, so a green
 *  bar means "edit" on all three. */

// ── loading ──────────────────────────────────────────────────────────────────

let refreshTimer: number | null = null;
const dirty = new Set<string>();

async function reloadRuns(): Promise<void> {
  try {
    $runs.value = await fetchRuns();
  } catch {
    /* $link already carries the transport failure; an empty rail is honest */
  }
}

async function loadView(runId: string): Promise<void> {
  try {
    const v = await fetchRun(runId);
    $views.value = { ...$views.value, [runId]: v };
    liveModel.adopt(runId, v);
  } catch {
    /* a run that vanished stays out of $views; the rail row still shows */
  }
}

/** Coalesce the refetch. A busy agent emits dozens of events a second and each
 *  one would otherwise be its own round-trip; 200ms is under the threshold
 *  where the page reads as laggy and well above the event rate. */
function markDirty(runId: string): void {
  dirty.add(runId);
  if (refreshTimer !== null) return;
  refreshTimer = window.setTimeout(() => {
    refreshTimer = null;
    const ids = [...dirty];
    dirty.clear();
    for (const id of ids) if ($selected.value.includes(id)) void loadView(id);
    void reloadRuns();
    if ($selected.value.length > 1) void reloadComparison();
  }, 200);
}

async function reloadComparison(): Promise<void> {
  const ids = $selected.value;
  if (ids.length < 2) {
    $comparison.value = null;
    return;
  }
  try {
    $comparison.value = await fetchComparison(ids);
  } catch {
    $comparison.value = null;
  }
}

function toggleSelected(runId: string): void {
  const cur = $selected.value;
  const next = cur.includes(runId) ? cur.filter((r) => r !== runId) : [...cur, runId];
  $selected.value = next;
  for (const id of next) if (!$views.value[id]) void loadView(id);
  void reloadComparison();
}

/** Drop every trace of a run from the page.
 *
 *  Called for our own delete *and* for a `run_deleted` frame from another
 *  window, which is why it lives here rather than inside the button: two
 *  viewers on the same server must not disagree about what exists. Every map
 *  keyed by run id has to be swept — a leftover entry in `$views` would keep
 *  the detail panel rendering numbers for a run whose log is gone.
 */
function forgetRun(runId: string): void {
  $runs.value = $runs.value.filter((r) => r.run_id !== runId);
  $selected.value = $selected.value.filter((r) => r !== runId);
  const drop = <T,>(m: Record<string, T>): Record<string, T> => {
    if (!(runId in m)) return m;
    const next = { ...m };
    delete next[runId];
    return next;
  };
  $views.value = drop($views.value);
  $tail.value = drop($tail.value);
  $analyses.value = drop($analyses.value);
  liveModel.forget(runId);
  dirty.delete(runId);
  void reloadComparison();
}

function pushTail(e: SeerEvent): void {
  const cur = $tail.value[e.run_id] ?? [];
  const next = [...cur, e];
  if (next.length > TAIL_MAX) next.splice(0, next.length - TAIL_MAX);
  $tail.value = { ...$tail.value, [e.run_id]: next };
}

// ── the page ─────────────────────────────────────────────────────────────────

export function SeerPage() {
  useEffect(() => {
    void fetchHealth();
    void reloadRuns();
    const off = connectLive({
      onEvent: (e) => {
        liveModel.ingest(e);
        pushTail(e);
        markDirty(e.run_id);
      },
      onRunFinished: (runId) => {
        // Close the open spans before the refetch, so nothing is left growing
        // on the Score while the snapshot is in flight.
        liveModel.finish(runId);
        void loadView(runId);
        void reloadRuns();
        if ($selected.value.length > 1) void reloadComparison();
      },
      onRunDeleted: (d) => {
        forgetRun(d.run_id);
      },
      onReconcileDone: (d) => {
        $importing.value = false;
        const failed = Object.keys(d.failed).length;
        // `n_skipped` is part of the result, not a footnote: a pass that
        // reported only its imports would read as having found less history
        // than it did.
        $importNote.value =
          `${d.n_imported} imported, ${d.n_skipped} already captured` +
          (failed ? `, ${failed} unreadable` : "") +
          ` (of ${d.n_seen} considered)`;
        void reloadRuns();
      },
      onReconcileFailed: (error) => {
        $importing.value = false;
        $importNote.value = null;
        $launchError.value = error;
      },
    });
    const poll = window.setInterval(() => void fetchHealth(), 15000);
    return () => {
      off();
      window.clearInterval(poll);
    };
  }, []);

  const views = $selectedViews.value;
  return (
    <div class="seer-page" role="main">
      <div class="seer-shell">
        <SeerRail />
        <div class="seer-stage">
          {$selected.value.length === 0 && <SeerEmpty />}
          {$selected.value.length === 1 && views[0] && <RunDetail view={views[0]} />}
          {$selected.value.length > 1 && <CompareDetail views={views} />}
        </div>
      </div>
    </div>
  );
}

// ── left rail: link, launcher, run list ──────────────────────────────────────

function SeerRail() {
  return (
    <aside class="seer-rail">
      <LinkStatus />
      <ObservingStatus />
      <Launcher />
      <Importer />
      <RunList />
    </aside>
  );
}

/** Whether your own sessions are being captured, and how well.
 *
 *  The absence of a collector and the absence of sessions look identical from
 *  the run list, so this says which it is. When something *is* watching, the
 *  numbers that matter are the ones about lines it could not use: a spool that
 *  is quietly dropping half its lines reads exactly like a quiet afternoon.
 */
function ObservingStatus() {
  const h = $health.value;
  if (!h) return null;
  const o = h.observing;
  if (!o) {
    return (
      <div class="seer-observing is-off">
        <span class="seer-observing-label">not watching your sessions</span>
        <span class="seer-note">
          <code>nebulai seer install</code>, then <code>serve --watch</code>
        </span>
      </div>
    );
  }
  if (!o.watching) {
    return (
      <div class="seer-observing is-off">
        <span class="seer-observing-label">no hook spool</span>
        <span class="seer-note">
          hooks are not installed — nothing is writing to <code>{o.spool_dir}</code>
        </span>
      </div>
    );
  }
  const lost = o.spool_torn + o.spool_unparsable;
  const coarse = o.clock_resolution_s != null && o.clock_resolution_s >= 0.05;
  return (
    <div class="seer-observing">
      <span class="seer-observing-label">
        watching · {o.open_runs.length} open session{o.open_runs.length === 1 ? "" : "s"}
      </span>
      <span class="seer-note tnum">
        {o.events} event{o.events === 1 ? "" : "s"} from {o.lines} hook
        {o.lines === 1 ? "" : "s"}
      </span>
      {coarse && (
        <span class="seer-note" title={`hook clock resolution: ${o.clock_resolution_s}s`}>
          coarse clock — tool durations are marked ~
        </span>
      )}
      {lost > 0 && (
        <span class="seer-warn" title="torn lines mean concurrent writes are interleaving">
          {lost} spool line{lost === 1 ? "" : "s"} unusable
        </span>
      )}
      {o.spool_backlog_files_skipped > 0 && (
        <span class="seer-note">
          {o.spool_backlog_files_skipped} backlog file
          {o.spool_backlog_files_skipped === 1 ? "" : "s"} skipped —{" "}
          <code>seer import-spool</code> to read them
        </span>
      )}
      {Object.entries(o.unknown_agents).map(([agent, n]) => (
        <span class="seer-note" key={agent}>
          {n} hook{n === 1 ? "" : "s"} from {agent}, which has no adapter
        </span>
      ))}
    </div>
  );
}

function LinkStatus() {
  const link = $link.value;
  const h = $health.value;
  const label =
    link === "live" ? "live" : link === "connecting" ? "connecting…" : link === "down" ? "offline" : "—";
  return (
    <div class={`seer-link is-${link}`}>
      <span class="seer-link-dot" aria-hidden="true" />
      <span class="seer-link-label">{label}</span>
      {h && (
        <span class="seer-link-meta tnum">
          {h.runs} run{h.runs === 1 ? "" : "s"}
          {h.active.length > 0 && ` · ${h.active.length} running`}
        </span>
      )}
      {link === "down" && (
        <span class="seer-link-hint" title={$linkError.value ?? undefined}>
          start it with <code>nebulai seer serve</code>
        </span>
      )}
    </div>
  );
}

function Launcher() {
  const agent = useSignal<StartRequest["agent"]>("claude");
  const prompt = useSignal("");
  const cwd = useSignal("");
  const model = useSignal("");
  const busy = useSignal(false);
  /** Codex only, and a different measurement rather than a better one: the
   *  app-server carries 68 notification kinds against `exec --json`'s 7, and
   *  with an empty prompt it watches a session instead of starting one. */
  const attached = useSignal(false);
  const note = useSignal<string | null>(null);

  const canAttach = agent.value === "codex" && attached.value;
  const needsPrompt = !canAttach;

  const launch = async () => {
    if (busy.value || (needsPrompt && !prompt.value.trim())) return;
    busy.value = true;
    $launchError.value = null;
    note.value = null;
    try {
      let runId: string;
      if (canAttach) {
        const res = await attachCodex({
          prompt: prompt.value.trim() || undefined,
          cwd: cwd.value || undefined,
          model: model.value || undefined,
        });
        runId = res.run_id;
        note.value =
          res.transport === "daemon-proxy"
            ? "joined the running Codex daemon"
            : res.driving
              ? "driving a turn through our own app-server"
              : "no Codex daemon is running, so there is no live session to watch";
      } else {
        const res = await startRun({
          agent: agent.value,
          prompt: prompt.value,
          cwd: cwd.value || undefined,
          model: model.value || undefined,
        });
        runId = res.run_id;
      }
      await reloadRuns();
      $selected.value = [runId];
      await loadView(runId);
      void reloadComparison();
    } catch (e) {
      $launchError.value = e instanceof Error ? e.message : String(e);
    } finally {
      busy.value = false;
    }
  };

  return (
    <section class="seer-launch">
      <h3>Capture a run</h3>
      <div class="seer-agents" role="radiogroup" aria-label="Agent">
        {(["codex", "claude", "hermes"] as const).map((a) => (
          <button
            key={a}
            type="button"
            role="radio"
            aria-checked={agent.value === a}
            class={`seer-agent${agent.value === a ? " is-active" : ""}`}
            onClick={() => (agent.value = a)}
          >
            {a}
          </button>
        ))}
      </div>
      {agent.value === "codex" && (
        <label class="seer-check">
          <input
            type="checkbox"
            checked={attached.value}
            onChange={(e) => (attached.value = (e.target as HTMLInputElement).checked)}
          />
          <span>
            app-server <span class="seer-dim">(68 event kinds, not 7)</span>
          </span>
        </label>
      )}
      <textarea
        class="seer-prompt"
        rows={3}
        placeholder={
          canAttach
            ? "the task — or leave empty to watch a Codex session you started elsewhere"
            : "the task, exactly as you would type it to the agent"
        }
        value={prompt.value}
        onInput={(e) => (prompt.value = (e.target as HTMLTextAreaElement).value)}
      />
      <input
        class="seer-input"
        placeholder="working directory (defaults to where the server runs)"
        value={cwd.value}
        onInput={(e) => (cwd.value = (e.target as HTMLInputElement).value)}
      />
      <input
        class="seer-input"
        placeholder="model (optional)"
        value={model.value}
        onInput={(e) => (model.value = (e.target as HTMLInputElement).value)}
      />
      <button
        type="button"
        class="btn-primary seer-launch-go"
        disabled={
          busy.value || (needsPrompt && !prompt.value.trim()) || $link.value === "down"
        }
        onClick={launch}
      >
        {busy.value
          ? canAttach
            ? "attaching…"
            : "launching…"
          : canAttach
            ? prompt.value.trim()
              ? "Run codex via app-server"
              : "Watch a running codex"
            : `Run ${agent.value} headless`}
      </button>
      {$launchError.value && <p class="seer-error">{$launchError.value}</p>}
      {note.value && <p class="seer-note">{note.value}</p>}
      <p class="seer-note">
        The agent runs with your real environment and your real credentials, in the directory you
        give it. It can edit files and run commands there — this is a capture harness, not a
        sandbox.
        {canAttach &&
          " SessionSeer declines every approval the app-server asks for: it is not a person and " +
            "will not consent for you."}
      </p>
    </section>
  );
}

/** Sessions that already happened, read back off the disk.
 *
 *  Separate from the launcher because it is not a capture: nothing runs, no
 *  credentials are used, and the runs it produces are `reconciled` — the same
 *  session seen through a keyhole rather than watched. Presenting it as another
 *  way to "start a run" would invite comparing a reconciled run against a
 *  driven one as if they were the same measurement.
 */
function Importer() {
  const limit = useSignal(25);
  const days = useSignal("");

  const go = async () => {
    if ($importing.value) return;
    $importing.value = true;
    $importNote.value = null;
    $launchError.value = null;
    try {
      const n = Number(days.value);
      await reconcileCodex({
        limit: limit.value,
        since_days: days.value.trim() && Number.isFinite(n) ? n : undefined,
      });
      $importNote.value = "reading threads…";
    } catch (e) {
      $importing.value = false;
      $launchError.value = e instanceof Error ? e.message : String(e);
    }
  };

  return (
    <section class="seer-launch">
      <h3>Import past Codex sessions</h3>
      <div class="seer-import-row">
        <input
          class="seer-input"
          type="number"
          min={1}
          max={500}
          aria-label="how many threads to consider"
          value={limit.value}
          onInput={(e) => (limit.value = Number((e.target as HTMLInputElement).value) || 25)}
        />
        <input
          class="seer-input"
          placeholder="last N days (optional)"
          value={days.value}
          onInput={(e) => (days.value = (e.target as HTMLInputElement).value)}
        />
      </div>
      <button
        type="button"
        class="btn-secondary seer-launch-go"
        disabled={$importing.value || $link.value === "down"}
        onClick={go}
      >
        {$importing.value ? "importing…" : "Import newest threads"}
      </button>
      {$importNote.value && <p class="seer-note">{$importNote.value}</p>}
      <p class="seer-note">
        Read-only: threads are listed and read, never resumed, archived or deleted. A session
        already captured is skipped by its own id, so importing twice cannot double your totals.
        Thread history carries no per-item clock, so these runs report call durations as absent
        rather than zero.
      </p>
    </section>
  );
}

function RunList() {
  const runs = $runs.value;
  const sel = $selected.value;
  return (
    <section class="seer-runs">
      <h3>
        Runs
        {sel.length > 1 && <span class="seer-runs-hint">{sel.length} selected — comparing</span>}
      </h3>
      {runs.length === 0 && <p class="seer-note">nothing captured yet</p>}
      <ul>
        {runs.map((r) => (
          <RunRow key={r.run_id} run={r} selected={sel.includes(r.run_id)} />
        ))}
      </ul>
    </section>
  );
}

function RunRow(props: { run: RunSummary; selected: boolean }) {
  const r = props.run;
  const state = r.state ?? "starting";
  const live = !["completed", "failed", "interrupted"].includes(state);
  const dur =
    r.started_at != null && r.ended_at != null ? fmtSeconds(r.ended_at - r.started_at) : null;
  return (
    <li>
      <button
        type="button"
        class={`seer-run${props.selected ? " is-active" : ""}`}
        aria-pressed={props.selected}
        // the row's own text is decorative spans; without this the control has
        // no name at all to a screen reader, or to anything driving the page
        aria-label={`${r.agent} run ${r.label || r.run_id}, ${state}${dur ? `, ${dur}` : ""}`}
        onClick={() => toggleSelected(r.run_id)}
      >
        <span class="seer-run-dot" style={{ background: stateInk(state) }} />
        <span class="seer-run-main">
          <span class="seer-run-agent">
            {r.agent}
            {r.capture_mode === "observed" && (
              // Said on the row, not only in the detail panel: an observed run
              // has no token counts at all, and a reader scanning the list has
              // to know that before wondering where they went.
              <span class="seer-run-mode" title="captured from hooks — no token usage available">
                obs
              </span>
            )}
          </span>
          <span class="seer-run-id">{r.label || r.run_id.replace(/^run_/, "").slice(0, 8)}</span>
        </span>
        <span class="seer-run-meta tnum">
          {live ? state.replace(/_/g, " ") : (dur ?? "—")}
          {r.n_warnings > 0 && <span class="seer-run-warn" title={`${r.n_warnings} adapter warnings`}>!</span>}
        </span>
      </button>
    </li>
  );
}

function SeerEmpty() {
  return (
    <div class="seer-empty">
      <h2>Watch an agent work</h2>
      <p>
        SessionSeer launches Codex, Claude Code or Hermes headless, reads their structured output as
        it streams, and folds it into one event vocabulary — so the same question can be asked of
        all three, and refused where it cannot honestly be answered.
      </p>
      <p class="seer-note">
        Start the collector with <code>nebulai seer serve</code>, then launch a run on the left, or
        pick one already captured. Select two to compare them.
      </p>
      <p class="seer-note">
        It can also watch the sessions you drive yourself: <code>nebulai seer install</code> adds a
        hook to each agent's own config — merged, backed up, and removable — and{" "}
        <code>seer serve --watch</code> turns what those hooks write into runs. Hooks see actions and
        timing, never token usage; those runs say so rather than reporting zero.
      </p>
    </div>
  );
}

// ── one run ──────────────────────────────────────────────────────────────────

function RunDetail(props: { view: RunView }) {
  const v = props.view;
  const live = !["completed", "failed", "interrupted"].includes(v.state);
  const end = runEnd(v) ?? v.started_at ?? 0;
  const wall = v.started_at != null ? end - v.started_at : null;

  return (
    <div class="seer-detail">
      <header class="seer-detail-head">
        <div>
          <h2>
            {v.agent} <span class="seer-dim">{v.agent_version}</span>
          </h2>
          <p class="seer-run-line tnum">
            <span class="seer-state" style={{ color: stateInk(v.state) }}>
              {v.state.replace(/_/g, " ")}
            </span>
            {v.overlays.map((o) => (
              <span key={o} class="seer-overlay" title="a modifier on the state, not a state">
                {o}
              </span>
            ))}
            {wall != null && <span>· {fmtSeconds(wall)}</span>}
            <span>
              · {v.n_events} event{v.n_events === 1 ? "" : "s"}
            </span>
            {v.model?.model_id && <span>· {v.model.model_id}</span>}
            {v.quality.capture_mode && <span>· {v.quality.capture_mode}</span>}
          </p>
          <Outcome view={v} />
        </div>
        <div class="seer-detail-tools">
          {live && (
            <button type="button" class="seer-btn" onClick={() => void cancelRun(v.run_id)}>
              Cancel
            </button>
          )}
          <ExportMenu runId={v.run_id} />
          <DeleteButton view={v} />
        </div>
      </header>

      {v.repo && (
        <p class="seer-repo tnum">
          {v.repo.branch} @ {(v.repo.head ?? "").slice(0, 8)}
          {v.repo.dirty ? " (dirty at launch)" : ""}
        </p>
      )}

      <SeerScore key={v.run_id} view={v} live={liveModel} />
      <Trajectory view={v} />
      <div class="seer-grid">
        <StateBar view={v} />
        <ActionBars view={v} />
        <TokenPanel view={v} />
        <VerificationCard view={v} />
      </div>
      <AnalysesPanel view={v} />
      <QualityPanel view={v} />
      <AnnotationPanel view={v} />
      <EventTail runId={v.run_id} />
    </div>
  );
}

function Outcome(props: { view: RunView }) {
  const o = props.view.outcome;
  const claimed = o === "agent_claimed_complete" || o === "unverified_complete";
  return (
    <p class={`seer-outcome is-${o}`}>
      {o.replace(/_/g, " ")}
      {claimed && (
        <span class="seer-outcome-caveat">
          — the agent's own word. Nothing in this run verified it.
        </span>
      )}
      {o === "infrastructure_failure" && (
        <span class="seer-outcome-caveat">
          — our capture failed or the process died. This says nothing about the task.
        </span>
      )}
      {o === "unknown" && (
        <span class="seer-outcome-caveat">— nothing in this run settles it either way.</span>
      )}
    </p>
  );
}

/** The end of a run, as one number.
 *
 *  `last_event_at` is later than `ended_at` — our own RUN_COMPLETED lands after
 *  the agent's session ended, and process teardown can put a second between
 *  them. Anything that prints a wall clock must go through here, or the header
 *  and the trajectory will disagree about the same run's duration. */
function runEnd(v: RunView): number | null {
  return v.ended_at ?? v.last_event_at ?? null;
}

/** Spans laid out on a real time axis, one lane per action that occurred.
 *  Gaps between bars are as informative as the bars: a wide empty stretch in
 *  every lane is the model thinking, or the run stuck.
 *
 *  Plain HTML rather than SVG: a stretched viewBox would scale the bars and
 *  the lane labels by different factors, and a chart whose labels don't line
 *  up with its bars is worse than no chart. */
function Trajectory(props: { view: RunView }) {
  const v = props.view;
  const spans = v.spans.filter((s) => s.action);
  const t0 = v.started_at ?? spans[0]?.started_at ?? 0;
  const t1 = Math.max(runEnd(v) ?? 0, ...spans.map((s) => s.ended_at ?? s.started_at), t0);
  const width = Math.max(t1 - t0, 0.001);
  const lanes = ACTIONS.filter((a) => spans.some((s) => s.action === a));

  if (spans.length === 0) {
    return (
      <section class="seer-card">
        <h3>Trajectory</h3>
        <p class="seer-note">
          No tool spans observed{v.n_events > 0 ? " — this run did no classified work" : " yet"}.
        </p>
      </section>
    );
  }

  return (
    <section class="seer-card seer-traj">
      <h3>
        Trajectory <span class="seer-dim">{fmtSeconds(width)} of wall clock</span>
      </h3>
      <div class="seer-lanes">
        {lanes.map((a) => (
          <div class="seer-lane" key={a}>
            <span class="seer-lane-name" style={{ color: ACTION_COLOR[a] }}>
              {a}
            </span>
            <div class="seer-lane-track">
              {spans
                .filter((s) => s.action === a)
                .map((s) => (
                  <span
                    key={s.span_id}
                    class={`seer-lane-bar${s.ended_at == null ? " is-open" : ""}`}
                    style={{
                      left: `${((s.started_at - t0) / width) * 100}%`,
                      width: `max(3px, ${(((s.ended_at ?? t1) - s.started_at) / width) * 100}%)`,
                      background: s.failed ? "var(--danger)" : ACTION_COLOR[a],
                    }}
                    title={spanTitle(s)}
                  />
                ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function spanTitle(s: SpanRecord): string {
  const parts = [s.detail || s.native_type || s.action || "span"];
  if (s.duration_s != null) parts.push(fmtSeconds(s.duration_s));
  else parts.push("still open");
  // `failed` and the `failed` effect are the same fact reported twice — the
  // flag is the summary of the effect, so only the more specific one is shown
  if (s.effect && s.effect !== "unknown") parts.push(s.effect.replace(/_/g, " "));
  else if (s.failed) parts.push("failed");
  return parts.join(" · ");
}

/** What a state's elapsed time is actually evidence of, per capture mode.
 *
 *  In a driven run `model_running` is bracketed by real request events. In an
 *  observed run nothing watched the model at all — that time is simply what was
 *  left over between hooks, and most of it is the model, but some of it is the
 *  human reading the screen. Calling it "model running" there would contradict
 *  the data-quality panel two cards down, which says per-request model timing is
 *  not observable in this mode. Both cannot be true, so the label changes.
 */
function stateLabel(state: string, observed: boolean): string {
  if (observed && state === "model_running") return "between tools";
  return state.replace(/_/g, " ");
}

function StateBar(props: { view: RunView }) {
  const t = props.view.time_in_state;
  const total = Object.values(t).reduce((a, b) => a + b, 0);
  const observed = props.view.quality.capture_mode === "observed";
  return (
    <section class="seer-card">
      <h3>
        Where the time went <span class="seer-dim">deterministic</span>
      </h3>
      {observed && total > 0 && (
        <p class="seer-note">
          Hooks bracket tool calls, not model requests, so everything outside a tool falls into
          one bucket — the model thinking, and you reading.
        </p>
      )}
      {total <= 0 ? (
        <p class="seer-note">no elapsed time attributed yet</p>
      ) : (
        <>
          <div class="seer-stack">
            {Object.entries(t)
              .sort((a, b) => b[1] - a[1])
              .map(([state, secs]) => (
                <span
                  key={state}
                  style={{
                    width: `${(secs / total) * 100}%`,
                    background: stateInk(state),
                  }}
                  title={`${stateLabel(state, observed)} — ${fmtSeconds(secs)}`}
                />
              ))}
          </div>
          <ul class="seer-kv tnum">
            {Object.entries(t)
              .sort((a, b) => b[1] - a[1])
              .map(([state, secs]) => (
                <li key={state}>
                  <span
                    class="seer-swatch"
                    style={{ background: stateInk(state) }}
                  />
                  <span>{stateLabel(state, observed)}</span>
                  <b>{fmtSeconds(secs)}</b>
                </li>
              ))}
          </ul>
        </>
      )}
    </section>
  );
}

function ActionBars(props: { view: RunView }) {
  const counts = props.view.action_counts;
  const max = Math.max(1, ...Object.values(counts));
  const present = ACTIONS.filter((a) => (counts[a] ?? 0) > 0);
  return (
    <section class="seer-card">
      <h3>
        What it did{" "}
        <span class="seer-dim">
          {props.view.n_turns ? `${props.view.n_turns} turn${props.view.n_turns === 1 ? "" : "s"}` : "turns not observed"}
        </span>
      </h3>
      {present.length === 0 ? (
        <p class="seer-note">no classified actions</p>
      ) : (
        <ul class="seer-bars tnum">
          {present.map((a) => (
            <li key={a}>
              <span class="seer-bars-label" style={{ color: ACTION_COLOR[a] }}>
                {a}
              </span>
              <span class="seer-bars-track">
                <span
                  style={{ width: `${((counts[a] ?? 0) / max) * 100}%`, background: ACTION_COLOR[a] }}
                />
              </span>
              <b>{counts[a]}</b>
            </li>
          ))}
        </ul>
      )}
      <p class="seer-note">
        {props.view.n_files_changed > 0
          ? `${props.view.n_files_changed} file${props.view.n_files_changed === 1 ? "" : "s"} changed`
          : "no file changes observed"}
      </p>
    </section>
  );
}

/** The one place a `Measured` becomes text. Absent values render as an em dash
 *  carrying their reason, and `dropped_by_policy` is styled apart from
 *  `missing` — "we chose not to look" and "the agent never said" send a
 *  researcher to different places. */
function Value(props: { m: Measured; fmt?: (n: number) => string }) {
  const m = props.m;
  if (isAbsent(m)) {
    return (
      <span
        class={`seer-absent is-${m.fidelity}`}
        title={m.note ? `${FIDELITY_TITLE[m.fidelity]} — ${m.note}` : FIDELITY_TITLE[m.fidelity]}
      >
        —
      </span>
    );
  }
  const mark = FIDELITY_MARK[m.fidelity];
  return (
    <span class="seer-value tnum" title={FIDELITY_TITLE[m.fidelity]}>
      {(props.fmt ?? fmtCount)(m.value as number)}
      {mark && <sup class="seer-fid">{mark}</sup>}
    </span>
  );
}

function TokenPanel(props: { view: RunView }) {
  const v = props.view;
  const cats = Object.keys(v.usage);
  return (
    <section class="seer-card">
      <h3>
        Tokens <span class="seer-dim">{v.quality.capture_mode ?? "unknown"} capture</span>
      </h3>
      <ul class="seer-kv tnum">
        {cats.map((c) => (
          <li key={c}>
            <span>{c.replace(/_/g, " ")}</span>
            <b>
              <Value m={v.usage[c]!} />
            </b>
          </li>
        ))}
        <li class="seer-kv-sep">
          <span>cost</span>
          <b>
            <Value m={v.cost_usd} fmt={fmtUsd} />
          </b>
        </li>
        <li>
          <span>context window</span>
          <b>
            <Value m={v.context_window} />
          </b>
        </li>
      </ul>
      {cats.length === 0 && (
        <p class="seer-note">
          {v.agent} has not reported any usage yet. Every agent we drive reports tokens at the end
          of a turn, so this stays empty until the first turn closes.
        </p>
      )}
      {v.quality.absent_token_categories.length > 0 && (
        <p class="seer-note">
          {v.agent} has no bucket for {v.quality.absent_token_categories.join(", ")} — these are
          absent by the agent's design, not lost in capture.
        </p>
      )}
      {v.native_usage_keys.length > 0 && (
        <details class="seer-native">
          <summary>reported natively ({v.native_usage_keys.length} fields)</summary>
          <p class="seer-note">
            {v.native_usage_keys.join(", ")}
            {" — the agent's own usage keys, verbatim. Anything above that these do not "}
            cover was computed by us, and says so.
          </p>
        </details>
      )}
    </section>
  );
}

function VerificationCard(props: { view: RunView }) {
  const v = props.view;
  const after = v.verification_after_last_edit;
  return (
    <section class="seer-card">
      <h3>Verification</h3>
      <ul class="seer-kv">
        <li>
          <span>ran any check</span>
          <b class={v.verified ? "is-good" : "is-bad"}>{v.verified ? "yes" : "no"}</b>
        </li>
        <li>
          <span>checked after the last edit</span>
          <b>
            {isAbsent(after) ? (
              <span class="seer-absent" title={after.note ?? undefined}>
                —
              </span>
            ) : (
              <span class={after.value ? "is-good" : "is-bad"}>{after.value ? "yes" : "NO"}</span>
            )}
          </b>
        </li>
      </ul>
      <p class="seer-note">
        "Ran a check" is not "the check passed" — the outcome above is the only place a pass or fail
        is claimed, and only an evaluator record can make it a verified one.
      </p>
    </section>
  );
}

function QualityPanel(props: { view: RunView }) {
  const q = props.view.quality;
  const dropped = Object.entries(q.dropped_by_policy);
  const nothing =
    q.capture_gaps.length === 0 &&
    q.warnings.length === 0 &&
    q.unmatched_tools.length === 0 &&
    dropped.length === 0 &&
    q.folded_duplicates === 0;
  return (
    <section class="seer-card seer-quality">
      <h3>
        Data quality <span class="seer-dim">{q.capture_mode ?? "unknown"} capture</span>
      </h3>
      {nothing && <p class="seer-note">nothing withheld, nothing unmapped</p>}
      {q.capture_gaps.length > 0 && (
        <div class="seer-quality-block">
          <h4>Not observable in this mode</h4>
          <ul>
            {q.capture_gaps.map((g) => (
              <li key={g}>{g}</li>
            ))}
          </ul>
          <p class="seer-note">
            A zero for any of these would mean "we could not look", not "it did not happen". The
            comparison gate refuses them rather than showing the zero.
          </p>
        </div>
      )}
      {dropped.length > 0 && (
        <div class="seer-quality-block">
          <h4>Dropped by policy</h4>
          <ul>
            {dropped.map(([k, n]) => (
              <li key={k}>
                {k} <span class="seer-dim tnum">×{n}</span>
              </li>
            ))}
          </ul>
          <p class="seer-note">Captured but deliberately not retained — a setting, not a gap.</p>
        </div>
      )}
      {q.folded_duplicates > 0 && (
        <div class="seer-quality-block">
          <h4>
            Folded repeats <span class="seer-dim tnum">{q.folded_duplicates}</span>
          </h4>
          <p class="seer-note">
            Usage sightings the fold rule refused to add twice. Non-zero is the rule working — the
            agent repeats one response's totals across several lines.
          </p>
        </div>
      )}
      {q.unmatched_tools.length > 0 && (
        <div class="seer-quality-block">
          <h4>Unclassified tools</h4>
          <ul class="seer-chips">
            {q.unmatched_tools.map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
          <p class="seer-note">
            Counted in totals but absent from the action mix — the taxonomy needs a rule for these.
          </p>
        </div>
      )}
      {q.warnings.length > 0 && (
        <div class="seer-quality-block">
          <h4>Adapter warnings</h4>
          <ul>
            {q.warnings.slice(0, 12).map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}


// ── analyses ─────────────────────────────────────────────────────────────────

/** The derived layer, with its work shown.
 *
 *  Every card carries the formula and the inputs behind a disclosure, and the
 *  evidence list under it. That is not decoration: a churn ratio you cannot
 *  trace to the file it came from is a number a researcher has to take on
 *  trust, and this page does not ask for trust anywhere else either.
 */
function AnalysesPanel(props: { view: RunView }) {
  const v = props.view;
  const doc = $analyses.value[v.run_id];
  const busy = useSignal(false);
  const err = useSignal<string | null>(null);

  const load = async () => {
    busy.value = true;
    err.value = null;
    try {
      const d = await fetchAnalysis(v.run_id);
      $analyses.value = { ...$analyses.value, [v.run_id]: d };
    } catch (e) {
      err.value = e instanceof Error ? e.message : String(e);
    } finally {
      busy.value = false;
    }
  };

  useEffect(() => {
    if (!$analyses.value[v.run_id]) void load();
    // one fetch per run; the button below is how a live run gets a newer one
  }, [v.run_id]);

  const behind = doc ? v.n_events - doc.n_events : 0;

  return (
    <section class="seer-card seer-analyses">
      <h3>
        Analyses{" "}
        {doc && <span class="seer-dim">v{doc.analyses_version} · {doc.n_events} events</span>}
        <button
          type="button"
          class="seer-btn is-small"
          disabled={busy.value}
          onClick={() => void load()}
        >
          {busy.value ? "computing…" : behind > 0 ? `recompute (+${behind})` : "recompute"}
        </button>
      </h3>
      {err.value && <p class="seer-warn">{err.value}</p>}
      {!doc && !err.value && <p class="seer-note">computing…</p>}
      {behind > 0 && (
        <p class="seer-note">
          {behind} event{behind === 1 ? "" : "s"} have landed since this was computed. The numbers
          below are a snapshot, not a live reading.
        </p>
      )}
      <div class="seer-analysis-grid">
        {doc?.analyses.map((a) => (
          <AnalysisCard key={a.key} a={a} />
        ))}
      </div>
    </section>
  );
}

function AnalysisCard(props: { a: Analysis }) {
  const a = props.a;
  const absent = isAbsent(a.headline);
  const rows = a.rows ?? [];
  return (
    <article class={`seer-analysis${absent ? " is-absent" : ""}`}>
      <header>
        <h4>{a.label}</h4>
        <span class="seer-analysis-value tnum">
          {absent ? (
            <span class="seer-absent" title={a.headline.note ?? a.refusal ?? undefined}>
              —
            </span>
          ) : (
            <>
              <span title={FIDELITY_TITLE[a.headline.fidelity]}>
                {FIDELITY_MARK[a.headline.fidelity]}
                {typeof a.headline.value === "number"
                  ? a.unit === "s"
                    ? fmtSeconds(a.headline.value)
                    : fmtCount(a.headline.value)
                  : String(a.headline.value)}
              </span>
              {a.unit && a.unit !== "s" && (
                <span class="seer-dim"> {unitFor(a.unit, a.headline.value)}</span>
              )}
            </>
          )}
        </span>
      </header>

      {/* the sentence goes where the number would be — never a bare dash */}
      {(a.refusal || (absent && a.headline.note)) && (
        <p class="seer-refusal">{a.refusal ?? a.headline.note}</p>
      )}

      {Object.keys(a.parts ?? {}).length > 0 && (
        <ul class="seer-kv">
          {Object.entries(a.parts).map(([k, m]) => (
            <li key={k}>
              <span>{k.replace(/_/g, " ")}</span>
              <b>
                <Value m={m} fmt={k.endsWith("_s") ? fmtSeconds : undefined} />
              </b>
            </li>
          ))}
        </ul>
      )}

      {rows.length > 0 && <AnalysisRows a={a} />}

      <details class="seer-derivation">
        <summary>how this was computed</summary>
        <p class="seer-formula">{a.formula}</p>
        <p class="seer-note">
          version {a.version} · reads {a.inputs.join(", ")}
        </p>
        {a.evidence.length > 0 && (
          <ul class="seer-evidence">
            {a.evidence.map((e, i) => (
              <li key={i}>
                <span class="seer-dim">{e.kind}</span> <code>{e.ref.slice(0, 18)}</code>
                {e.detail && <span> {e.detail}</span>}
              </li>
            ))}
          </ul>
        )}
      </details>
    </article>
  );
}

/** Rule and checklist rows. Rendered as rows rather than folded into the
 *  headline because "0 matches" only means something once you can see that one
 *  of the four rules could not run at all. */
function AnalysisRows(props: { a: Analysis }) {
  const rows = props.a.rows as Record<string, unknown>[];
  const named = rows.filter((r) => r.rule || r.item);
  if (named.length > 0) {
    return (
      <ul class="seer-rules">
        {named.map((r, i) => {
          // `??` would be wrong here: a rule that could not run carries
          // `hits: null` deliberately, and falling through to `status` turns
          // that considered null into an undefined the renderer prints raw.
          const hits = ("hits" in r ? r.hits : r.status) as number | string | null;
          const off = hits == null || hits === "unknown";
          return (
            <li key={i} class={off ? "is-off" : undefined}>
              <span class="seer-rule-hit tnum">{off ? "—" : String(hits)}</span>
              <span>
                {String(r.rule ?? r.item)}
                {(r.note || r.why) != null && (
                  <em class="seer-dim"> — {String(r.note ?? r.why)}</em>
                )}
              </span>
            </li>
          );
        })}
      </ul>
    );
  }
  const first = rows[0];
  if (!first) return null;
  const keys = Object.keys(first).slice(0, 5);
  return (
    <div class="seer-table-wrap">
      <table class="seer-table tnum">
        <thead>
          <tr>
            {keys.map((k) => (
              <th key={k}>{k.replace(/_/g, " ")}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 12).map((r, i) => (
            <tr key={i}>
              {keys.map((k) => (
                <td key={k}>{cell(r[k])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > 12 && <p class="seer-note">{rows.length - 12} more rows in the export</p>}
    </div>
  );
}

/** "1 files" reads as a bug in the counter rather than a count of one.
 *  Units here are plain English nouns from Python, so trimming the `s` is
 *  enough — anything irregular would arrive already singular. */
function unitFor(unit: string, value: number | string | boolean | null): string {
  return value === 1 && unit.endsWith("s") ? unit.slice(0, -1) : unit;
}

function cell(x: unknown): string {
  if (x === null || x === undefined) return "—";
  if (typeof x === "boolean") return x ? "yes" : "no";
  if (Array.isArray(x)) return x.length ? x.map(String).join(", ").slice(0, 60) : "—";
  if (typeof x === "number") return Number.isInteger(x) ? String(x) : x.toFixed(3);
  return String(x).slice(0, 80);
}

// ── annotations ──────────────────────────────────────────────────────────────

/** A researcher's notes, appended to the run's own log.
 *
 *  Same store, same export, same deletion — a note kept in a side table would
 *  quietly outlive the run it is about, and half the value of an annotation is
 *  that it travels with the evidence. */
function AnnotationPanel(props: { view: RunView }) {
  const v = props.view;
  const text = useSignal("");
  const tags = useSignal("");
  const err = useSignal<string | null>(null);
  const busy = useSignal(false);

  const submit = async () => {
    const body = text.value.trim();
    if (!body) return;
    busy.value = true;
    err.value = null;
    try {
      await annotate(v.run_id, body, {
        tags: tags.value
          .split(/[,\s]+/)
          .map((t) => t.trim())
          .filter(Boolean),
      });
      text.value = "";
      tags.value = "";
      markDirty(v.run_id);
    } catch (e) {
      err.value = e instanceof Error ? e.message : String(e);
    } finally {
      busy.value = false;
    }
  };

  return (
    <section class="seer-card seer-annotations">
      <h3>
        Notes {v.annotations.length > 0 && <span class="seer-dim">{v.annotations.length}</span>}
      </h3>
      {v.annotations.length > 0 && (
        <ul class="seer-note-list">
          {v.annotations.map((n) => (
            <li key={n.event_id}>
              <p>{n.text}</p>
              <span class="seer-dim tnum">
                {new Date(n.ts * 1000).toLocaleTimeString()}
                {n.author ? ` · ${n.author}` : ""}
              </span>
              {n.tags.map((t) => (
                <span key={t} class="seer-tag">
                  {t}
                </span>
              ))}
            </li>
          ))}
        </ul>
      )}
      <textarea
        class="seer-input"
        rows={2}
        placeholder="what you noticed, for the next reader"
        value={text.value}
        onInput={(e) => (text.value = (e.target as HTMLTextAreaElement).value)}
      />
      <div class="seer-note-row">
        <input
          class="seer-input"
          placeholder="tags"
          value={tags.value}
          onInput={(e) => (tags.value = (e.target as HTMLInputElement).value)}
        />
        <button
          type="button"
          class="seer-btn"
          disabled={busy.value || !text.value.trim()}
          onClick={() => void submit()}
        >
          Add note
        </button>
      </div>
      {err.value && <p class="seer-warn">{err.value}</p>}
      <p class="seer-note">
        Appended to this run's event log, so it exports and is deleted with the run itself.
      </p>
    </section>
  );
}

const EXPORTS: { fmt: ExportFormat; label: string; ext: string; title: string }[] = [
  { fmt: "jsonl", label: "JSONL", ext: "jsonl", title: "the event log verbatim — lossless" },
  {
    fmt: "parquet",
    label: "Parquet",
    ext: "parquet",
    title: "one row per event, payload kept as JSON — for a cross-run study",
  },
  {
    fmt: "csv",
    label: "CSV",
    ext: "csv",
    title: "spans only, and lossy — the file says so in its first line",
  },
  {
    fmt: "analysis",
    label: "Analyses",
    ext: "json",
    title: "the derived layer with formulas and evidence",
  },
];

/** What each rung of the ladder takes out, in the words of the thing removed
 *  rather than the policy that removes it. Shown as the select's help line so
 *  the choice is made against consequences, not against a level name. */
const REDACT_NOTE: Record<RedactLevel, string> = {
  content: "everything the run holds, including prompts and the model's prose",
  command: "commands and metadata; prose and reasoning dropped, lengths kept",
  metadata: "structure and timing only — no prose, no commands",
};

function ExportMenu(props: { runId: string }) {
  const level = useSignal<RedactLevel>("content");
  const suffix = level.value === "content" ? "" : `-${level.value}`;
  return (
    <span class="seer-export">
      {EXPORTS.map((e) => (
        <a
          key={e.fmt}
          class="seer-btn is-small"
          title={e.title}
          href={exportUrl(props.runId, e.fmt, level.value)}
          // matches what the server names the file, so a redacted export on
          // someone's desktop still says so without being opened
          download={`${props.runId}${suffix}.${e.ext}`}
        >
          {e.label}
        </a>
      ))}
      <select
        class="seer-export-level"
        aria-label="how much of the run the export may carry"
        title={REDACT_NOTE[level.value]}
        value={level.value}
        onChange={(ev) => {
          level.value = (ev.currentTarget as HTMLSelectElement).value as RedactLevel;
        }}
      >
        <option value="content">full</option>
        <option value="command">no prose</option>
        <option value="metadata">no prose or commands</option>
      </select>
    </span>
  );
}

/** Two clicks, and the second one names what goes.
 *
 *  Not a `confirm()`: the count of events and the size on disk are the useful
 *  part of the question, and a modal cannot show them. A run still being
 *  captured is refused by the server (409) rather than disabled here — the
 *  button asks, and the answer is the server's to give.
 */
function DeleteButton(props: { view: RunView }) {
  const armed = useSignal(false);
  const err = useSignal<string | null>(null);
  const busy = useSignal(false);
  const v = props.view;

  if (!armed.value) {
    return (
      <button
        type="button"
        class="seer-btn is-danger"
        title="remove this run entirely — log, directory and index rows"
        onClick={() => {
          armed.value = true;
          err.value = null;
        }}
      >
        Delete
      </button>
    );
  }
  return (
    <span class="seer-delete-confirm">
      <span class="seer-note">
        delete {v.n_events} event{v.n_events === 1 ? "" : "s"}? this cannot be undone
      </span>
      <button
        type="button"
        class="seer-btn is-danger"
        disabled={busy.value}
        onClick={() => {
          busy.value = true;
          void deleteRun(v.run_id)
            .then(() => {
              // the SSE frame sweeps the page state; doing it here too would
              // race with it for no benefit
              armed.value = false;
            })
            .catch((e: unknown) => {
              err.value = e instanceof Error ? e.message : String(e);
            })
            .finally(() => {
              busy.value = false;
            });
        }}
      >
        {busy.value ? "deleting…" : "Delete for good"}
      </button>
      <button type="button" class="seer-btn is-small" onClick={() => (armed.value = false)}>
        Keep
      </button>
      {err.value && <span class="seer-warn">{err.value}</span>}
    </span>
  );
}

function EventTail(props: { runId: string }) {
  const events = $tail.value[props.runId] ?? [];
  if (events.length === 0) return null;
  const recent = events.slice(-40).reverse();
  return (
    <section class="seer-card seer-tail">
      <h3>
        Live tail <span class="seer-dim">newest first</span>
      </h3>
      <ol>
        {recent.map((e) => (
          <li key={e.event_id}>
            <span class="seer-tail-type">{e.event_type}</span>
            {e.action && (
              <span class="seer-tail-action" style={{ color: ACTION_COLOR[e.action] }}>
                {e.action}
              </span>
            )}
            <span class="seer-tail-detail">{tailDetail(e)}</span>
            <span class="seer-tail-fid" title={FIDELITY_TITLE[e.source.fidelity]}>
              {e.source.fidelity}
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}

function tailDetail(e: SeerEvent): string {
  const p = e.payload ?? {};
  for (const k of ["name", "command", "path", "text", "reason", "message", "title"]) {
    const v = p[k];
    if (typeof v === "string" && v) return v.length > 90 ? `${v.slice(0, 90)}…` : v;
  }
  return e.native_type ?? "";
}

// ── comparison ───────────────────────────────────────────────────────────────

function CompareDetail(props: { views: RunView[] }) {
  const c = $comparison.value;
  if (!c) return <p class="seer-note seer-detail">loading comparison…</p>;
  return (
    <div class="seer-detail">
      <header class="seer-detail-head">
        <div>
          <h2>Comparison</h2>
          <p class="seer-run-line">
            {c.runs.map((id) => (
              <span key={id} class="seer-cmp-run">
                {c.agents[id]} <span class="seer-dim tnum">{id.replace(/^run_/, "").slice(0, 8)}</span>
              </span>
            ))}
          </p>
        </div>
      </header>

      {c.refused.length > 0 && (
        <section class="seer-card seer-refusals">
          <h3>
            Cannot be compared <span class="seer-dim tnum">{c.refused.length}</span>
          </h3>
          <p class="seer-note">
            These are results, not omissions. Each one is a metric where a difference between these
            runs would report something other than a difference between the agents.
          </p>
          <ul>
            {c.refused.map((r) => (
              <li key={r.metric}>
                <b>{r.metric}</b>
                <span>{r.reason}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section class="seer-card">
        <h3>
          Comparable <span class="seer-dim tnum">{c.comparable.length} metrics</span>
        </h3>
        <div class="seer-table-wrap">
          <table class="seer-table tnum">
            <thead>
              <tr>
                <th>metric</th>
                {c.runs.map((id) => (
                  <th key={id}>{c.agents[id]}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {c.comparable.map((row) => (
                <tr key={row.metric}>
                  <th scope="row">{row.label}</th>
                  {c.runs.map((id) => (
                    <td key={id}>
                      <Value
                        m={row.values[id]!}
                        fmt={row.unit === "s" ? fmtSeconds : row.unit === "USD" ? fmtUsd : fmtCount}
                      />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div class="seer-grid">
        {props.views.map((v) => (
          <QualityPanel key={v.run_id} view={v} />
        ))}
      </div>
    </div>
  );
}
