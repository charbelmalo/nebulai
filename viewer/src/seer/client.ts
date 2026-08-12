/** client.ts — the viewer's half of the SessionSeer server (`:8125`).
 *
 *  Everything here is a thin, honest wrapper. No number is computed on this
 *  side: the reducer in `src/nebulai/seer/reducer.py` owns the fold, the
 *  comparability gate owns the refusals, and the UI's only job is to render
 *  what arrived without quietly improving it. The one thing this file does
 *  decide is *liveness* — whether the stream is connected — because a quiet
 *  agent and a dead socket look identical otherwise, and telling those apart
 *  is half of what a real-time overview is for.
 *
 *  The SSE stream uses **named** events (`hello`, `event`, `run_finished`), so
 *  `EventSource.onmessage` never fires and each name is subscribed explicitly.
 */

import { signal } from "@preact/signals";
import { appStore } from "../app/store";
import type { Fidelity, Measured, SeerEvent, SessionState } from "./contract";

// ── DTOs (mirror of the Python `to_dict()`s) ─────────────────────────────────

export interface RunSummary {
  run_id: string;
  agent: string;
  agent_version: string | null;
  capture_mode: string | null;
  label: string | null;
  repo_root: string | null;
  branch: string | null;
  started_at: number | null;
  ended_at: number | null;
  state: SessionState | null;
  outcome: string | null;
  n_events: number;
  n_warnings: number;
  /** The *agent's* id for this session, not ours. What tells a reconciliation
   *  pass "already captured" from "never seen", so importing history cannot
   *  silently double a month of token counts. */
  native_session_id: string | null;
}

export interface SpanRecord {
  span_id: string;
  action: string | null;
  native_type: string | null;
  started_at: number;
  ended_at: number | null;
  /** Wall clock between the start and end events SessionSeer saw. */
  duration_s: number | null;
  /** The agent's own clock for this call, when it reports one — Codex's
   *  app-server does, `codex exec --json` does not. Kept beside `duration_s`
   *  rather than replacing it: the interval between our two events is what the
   *  overlap arithmetic unions, and a differently-sourced number mixed into
   *  that would give a total no set of intervals adds up to. */
  native_duration_s: number | null;
  duration_fidelity: Fidelity;
  /** True when the call's beginning was never seen and its start was stamped
   *  from its end — reconciled runs are all like this. `duration_s` is then
   *  `0` by construction, which is not a measurement, so `duration_fidelity`
   *  reads `missing` and the UI must print `—`. */
  synthetic_start: boolean;
  effect: string | null;
  failed: boolean;
  detail: string | null;
  /** The call this one ran inside, when the agent reported one.
   *
   *  Null is the overwhelmingly common case and it means "not reported",
   *  never "top level" — the two are different claims and only the first is
   *  one we can make. Python keeps it so `time_decomposition` can subtract a
   *  child's seconds from its parent instead of counting them twice; the live
   *  view's structure mode reads it for depth, and draws a flat picture when
   *  it is absent rather than inventing a tree. */
  parent_span_id: string | null;
}

export interface DataQuality {
  capture_mode: string | null;
  capture_gaps: string[];
  warnings: string[];
  unmapped_native: string[];
  unmatched_tools: string[];
  absent_token_categories: string[];
  dropped_by_policy: Record<string, number>;
  folded_duplicates: number;
}

export interface RunView {
  run_id: string;
  agent: string;
  agent_version: string;
  model: { provider?: string; model_id?: string; effort?: string } | null;
  repo: { root_id?: string; branch?: string; head?: string; dirty?: boolean } | null;
  state: SessionState;
  overlays: string[];
  outcome: string;
  started_at: number | null;
  ended_at: number | null;
  last_event_at: number | null;
  time_in_state: Record<string, number>;
  action_counts: Record<string, number>;
  effect_counts: Record<string, number>;
  spans: SpanRecord[];
  usage: Record<string, Measured>;
  native_usage_keys: string[];
  cost_usd: Measured;
  context_window: Measured;
  n_events: number;
  n_turns: number;
  n_files_changed: number;
  files_changed: string[];
  /** per-path edit accounting. `line_data: false` is why `edit_churn` may
   *  refuse — the agent said a file changed but not by how much. */
  file_stats: Record<string, FileStat>;
  annotations: Annotation[];
  verified: boolean;
  verification_after_last_edit: Measured;
  quality: DataQuality;
  /** present on `/seer/run/<id>`: the index row, which knows what we launched
   *  even before the agent has said anything. */
  summary?: RunSummary;
}

export interface FileStat {
  edits: number;
  lines_added: number;
  lines_removed: number;
  total_lines: number | null;
  total_lines_from: "write" | "accumulated" | null;
  line_data: boolean;
}

export interface Annotation {
  event_id: string;
  ts: number;
  text: string;
  tags: string[];
  span_id: string | null;
  author: string | null;
}

/** One citation behind a derived number. */
export interface Evidence {
  kind: string;
  ref: string;
  ts: number | null;
  detail: string | null;
}

/** A derived value that shows its work.
 *
 *  `refusal` is not an error path: an analysis that cannot run returns the same
 *  shape as one that can, with the sentence explaining why. The panel renders
 *  the sentence where the number would be, which is the whole point — a dash
 *  with no reason beside it is what this subsystem exists to avoid. */
export interface Analysis {
  key: string;
  label: string;
  version: string;
  formula: string;
  inputs: string[];
  headline: Measured;
  unit: string;
  parts: Record<string, Measured>;
  rows: Record<string, unknown>[];
  evidence: Evidence[];
  refusal: string | null;
}

export interface RunAnalyses {
  run_id: string;
  agent: string;
  capture_mode: string | null;
  analyses_version: string;
  n_events: number;
  analyses: Analysis[];
}

export interface Refusal {
  metric: string;
  reason: string;
  runs: string[];
  detail: Record<string, unknown>;
}

export interface MetricRow {
  metric: string;
  label: string;
  unit: string;
  values: Record<string, Measured>;
}

export interface Comparison {
  runs: string[];
  agents: Record<string, string>;
  comparable: MetricRow[];
  refused: Refusal[];
  summary?: string;
}

/** What the spool collector is doing, when one is running at all.
 *
 *  `SeerHealth.observing` is `null` when nothing is watching — which from the
 *  outside is indistinguishable from "you have had no sessions", so the page
 *  reads this before it says either. */
export interface Observing {
  watching: boolean;
  spool_dir: string;
  /** `null` when no shim manifest was found, i.e. we do not know the clock. */
  clock_resolution_s: number | null;
  open_runs: {
    run_id: string;
    agent: string;
    session_id: string;
    n_events: number;
    idle_s: number;
  }[];
  lines: number;
  events: number;
  runs_opened: number;
  runs_closed_idle: number;
  unknown_agents: Record<string, number>;
  spool_torn: number;
  spool_unparsable: number;
  spool_backlog_files_skipped: number;
}

export interface SeerHealth {
  ok: boolean;
  schema_version: string;
  root: string;
  runs: number;
  active: string[];
  sse_clients_dropped: number;
  observing: Observing | null;
}

export interface StartRequest {
  agent: "codex" | "claude" | "hermes";
  prompt: string;
  cwd?: string;
  model?: string;
  label?: string;
  keep_reasoning?: boolean;
  timeout_s?: number;
}

/** Attached capture, Codex only. Omit `prompt` to watch a running daemon
 *  instead of driving a turn — the two are different measurements and the
 *  response says which one you got. */
export interface AttachRequest {
  prompt?: string;
  cwd?: string;
  model?: string;
  label?: string;
  sock?: string;
  no_daemon?: boolean;
  keep_reasoning?: boolean;
  timeout_s?: number;
}

export interface AttachResponse {
  run_id: string;
  agent: "codex";
  state: string;
  transport: "daemon-proxy" | "own-app-server";
  driving: boolean;
  protocol: {
    compatible: boolean;
    golden_version?: string;
    new_since_golden?: string[];
    unmapped_notifications?: string[];
  };
}

/** Import sessions that already happened. Codex only for now: it is the one
 *  agent whose history is reachable read-only through its own protocol. */
export interface ReconcileRequest {
  limit?: number;
  only_cwd?: string;
  since_days?: number;
  keep_reasoning?: boolean;
}

export interface ReconcileStarted {
  started: boolean;
  limit: number;
  agent: "codex";
}

/** The end of a pass, delivered on the live stream. `n_skipped` is not a
 *  footnote — it is the count of sessions the store already held, and a pass
 *  that reported only its imports would look like it had found less history
 *  than it did. */
export interface ReconcileDone {
  kind: "reconcile_done";
  n_seen: number;
  n_imported: number;
  n_skipped: number;
  failed: Record<string, string>;
}

// ── connection state ─────────────────────────────────────────────────────────

export type LinkState = "unknown" | "connecting" | "live" | "down";

/** Whether the SSE stream is actually delivering. Rendered in the page header:
 *  every number on screen is only as current as this says it is. */
export const $link = signal<LinkState>("unknown");
export const $health = signal<SeerHealth | null>(null);
/** Last transport error, verbatim. Shown rather than swallowed — "the server is
 *  not running" is a thing the researcher can fix, and a blank page is not. */
export const $linkError = signal<string | null>(null);

export function seerBase(): string {
  return (appStore.getState().seer.serverUrl || "").replace(/\/+$/, "");
}

async function getJSON<T>(path: string): Promise<T> {
  const base = seerBase();
  if (!base) throw new Error("no seer server configured");
  const res = await fetch(base + path);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = (await res.json()) as { error?: string };
      if (body.error) detail = body.error;
    } catch {
      /* the body was not JSON; the status is all we have */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const base = seerBase();
  if (!base) throw new Error("no seer server configured");
  const res = await fetch(base + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const b = (await res.json()) as { error?: string };
      if (b.error) detail = b.error;
    } catch {
      /* ditto */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export async function fetchHealth(): Promise<SeerHealth | null> {
  try {
    const h = await getJSON<SeerHealth>("/seer/health");
    $health.value = h;
    $linkError.value = null;
    return h;
  } catch (e) {
    $health.value = null;
    $link.value = "down";
    $linkError.value = e instanceof Error ? e.message : String(e);
    return null;
  }
}

export async function fetchRuns(limit = 100): Promise<RunSummary[]> {
  const { runs } = await getJSON<{ runs: RunSummary[] }>(`/seer/runs?limit=${limit}`);
  return runs;
}

export function fetchRun(runId: string): Promise<RunView> {
  return getJSON<RunView>(`/seer/run/${encodeURIComponent(runId)}`);
}

export function fetchEvents(runId: string, since = 0): Promise<{ events: SeerEvent[] }> {
  return getJSON(`/seer/run/${encodeURIComponent(runId)}/events?since=${since}`);
}

export function fetchComparison(runIds: string[]): Promise<Comparison> {
  return getJSON<Comparison>(`/seer/compare?runs=${runIds.map(encodeURIComponent).join(",")}`);
}

export function startRun(req: StartRequest): Promise<{ run_id: string; agent: string }> {
  return postJSON("/seer/start", req);
}

export function attachCodex(req: AttachRequest): Promise<AttachResponse> {
  return postJSON("/seer/attach", req);
}

/** Import Codex sessions already on disk. Returns as soon as the pass starts —
 *  reading twenty-five threads takes tens of seconds, and each run announces
 *  itself on the live stream as it lands. */
export function reconcileCodex(
  req: ReconcileRequest = {},
): Promise<ReconcileStarted> {
  return postJSON("/seer/reconcile", req);
}

export function cancelRun(runId: string): Promise<{ cancelled: boolean }> {
  return postJSON("/seer/cancel", { run_id: runId });
}

export interface Deleted {
  run_id: string;
  events: number;
  bytes: number;
  log_removed: boolean;
}

/** Remove a run entirely — log, directory and index rows.
 *
 *  409 when the run is still being captured; the message says to cancel it
 *  first, and it is worth showing verbatim rather than as "delete failed".
 */
export async function deleteRun(runId: string): Promise<Deleted> {
  const base = seerBase();
  if (!base) throw new Error("no seer server configured");
  const res = await fetch(`${base}/seer/run/${encodeURIComponent(runId)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const b = (await res.json()) as { error?: string };
      if (b.error) detail = b.error;
    } catch {
      /* the body was not JSON; the status is all we have */
    }
    throw new Error(detail);
  }
  return (await res.json()) as Deleted;
}

export type ExportFormat = "jsonl" | "parquet" | "csv" | "analysis";

/** What a shared export is allowed to carry. `content` is everything the run
 *  holds; the two rungs below it drop the model's prose and, at `metadata`,
 *  the commands too. The server names the level in the filename, so a file on
 *  someone's desktop still says what was taken out of it. */
export type RedactLevel = "content" | "command" | "metadata";

export function exportUrl(
  runId: string,
  format: ExportFormat = "jsonl",
  redact?: RedactLevel,
): string {
  const q = redact && redact !== "content" ? `&redact=${redact}` : "";
  return `${seerBase()}/seer/export?run_id=${encodeURIComponent(runId)}&format=${format}${q}`;
}

export function fetchAnalysis(runId: string): Promise<RunAnalyses> {
  return getJSON<RunAnalyses>(`/seer/run/${encodeURIComponent(runId)}/analysis`);
}

/** Append a human note to the run's own append-only log. */
export function annotate(
  runId: string,
  text: string,
  opts: { tags?: string[]; spanId?: string; author?: string } = {},
): Promise<{ ok: boolean; event_id: string; ts: number }> {
  return postJSON("/seer/annotate", {
    run_id: runId,
    text,
    tags: opts.tags ?? [],
    span_id: opts.spanId,
    author: opts.author,
  });
}

// ── the live stream ──────────────────────────────────────────────────────────

export interface LiveHandlers {
  onEvent(e: SeerEvent): void;
  onRunFinished?(runId: string, view: RunView): void;
  onRunDeleted?(d: Deleted): void;
  onReconcileDone?(d: ReconcileDone): void;
  onReconcileFailed?(error: string): void;
}

/** Subscribe to `/seer/live`. Returns a disposer.
 *
 *  Reconnect is deliberately dumb — a fixed 2s retry — and every reconnect
 *  flips `$link` back through "connecting". The page refetches the open run on
 *  reconnect rather than assuming the gap was empty: events that happened while
 *  the socket was down are in the log, and pretending otherwise would leave a
 *  hole no one could see.
 */
export function connectLive(h: LiveHandlers): () => void {
  let es: EventSource | null = null;
  let retry: number | null = null;
  let closed = false;

  const open = () => {
    if (closed) return;
    const base = seerBase();
    if (!base) {
      $link.value = "down";
      $linkError.value = "no seer server configured";
      return;
    }
    $link.value = "connecting";
    es = new EventSource(`${base}/seer/live`);

    es.addEventListener("hello", () => {
      $link.value = "live";
      $linkError.value = null;
    });
    es.addEventListener("event", (m) => {
      const d = JSON.parse((m as MessageEvent).data) as { event: SeerEvent };
      if (d.event) h.onEvent(d.event);
    });
    es.addEventListener("run_finished", (m) => {
      const d = JSON.parse((m as MessageEvent).data) as { run_id: string; view: RunView };
      h.onRunFinished?.(d.run_id, d.view);
    });
    es.addEventListener("run_deleted", (m) => {
      // Every page holding this run open hears it, not only the one that
      // pressed delete. A viewer that kept rendering a deleted run would 404
      // on the next click with nothing on screen explaining why.
      h.onRunDeleted?.(JSON.parse((m as MessageEvent).data) as Deleted);
    });
    es.addEventListener("reconcile_done", (m) => {
      h.onReconcileDone?.(JSON.parse((m as MessageEvent).data) as ReconcileDone);
    });
    es.addEventListener("reconcile_failed", (m) => {
      const d = JSON.parse((m as MessageEvent).data) as { error?: string };
      h.onReconcileFailed?.(d.error || "the import pass failed");
    });
    es.onerror = () => {
      // EventSource retries on its own, but silently and on its own schedule;
      // we take it over so the UI can *say* it is disconnected.
      es?.close();
      es = null;
      $link.value = "down";
      $linkError.value = "stream closed";
      if (!closed && retry === null) {
        retry = window.setTimeout(() => {
          retry = null;
          open();
        }, 2000);
      }
    };
  };

  open();
  return () => {
    closed = true;
    if (retry !== null) window.clearTimeout(retry);
    es?.close();
    $link.value = "unknown";
  };
}

// ── formatting (the display rules, in one place) ─────────────────────────────

/** Fidelity marks. Native and deterministic carry none — most numbers are one
 *  of those and marking them would make the marks invisible where they matter. */
export const FIDELITY_MARK: Record<Fidelity, string> = {
  native: "",
  deterministic: "",
  estimated: "~",
  heuristic: "?",
  missing: "",
  dropped_by_policy: "",
};

export const FIDELITY_TITLE: Record<Fidelity, string> = {
  native: "reported by the agent itself",
  deterministic: "computed by us from what the agent reported",
  estimated: "approximated — do not difference this against a native number",
  heuristic: "inferred from a pattern; may be wrong",
  missing: "the agent never reported it",
  dropped_by_policy: "we chose not to capture it",
};

export function fmtCount(n: number): string {
  return n.toLocaleString();
}

export function fmtSeconds(s: number): string {
  if (s < 1) return `${(s * 1000).toFixed(0)}ms`;
  if (s < 90) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s - m * 60)}s`;
}

export function fmtUsd(n: number): string {
  return n < 0.01 ? `$${n.toFixed(4)}` : `$${n.toFixed(2)}`;
}
