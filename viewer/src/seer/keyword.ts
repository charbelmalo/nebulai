/** keyword.ts — the viewer's half of `seer keyword`: who put a word in an
 *  agent's context.
 *
 *  The scan itself never happens here. A project's transcripts are hundreds of
 *  megabytes of JSONL on the operator's disk — 0.96 GB for the largest project
 *  on this machine — so the browser asks the server to read them and polls a
 *  job. Nothing is streamed into the page but the finished report and its
 *  progress, and nothing is recomputed on this side: `sessionlog.py` owns the
 *  classification, the folding and every count, exactly as `reducer.py` owns
 *  the fold behind the Live page.
 *
 *  The one thing this file must get right is that **a hit is not a hit**. Five
 *  different parties put text into an agent's context and only one of them is
 *  the operator; a page that renders 4,160 occurrences as one number would
 *  answer "is this word around" when the question was "who keeps saying it".
 *  So `ORIGINS` below is the render order of the answer, and it is fixed:
 *  the human first, the machinery last.
 */

import { getJSON, postJSON } from "./client";

// ── DTOs (mirror of `KeywordReport.to_dict()`, schema seer.keyword/1) ────────

export interface ProjectRow {
  slug: string;
  path: string;
  sessions: number;
  bytes: number;
  /** How many of those sessions have a title the operator gave them. */
  titled: number;
}

export interface ChannelRow {
  id: string;
  label: string;
  origin: string;
  on_by_default: boolean;
  why: string;
}

export interface KeywordSample {
  session_id: string;
  title: string | null;
  line: number;
  channel: string;
  origin: string;
  form: string;
  json_path: string;
  sense: string;
  timestamp: string | null;
  snippet: string;
}

export interface KeywordSessionRow {
  session_id: string;
  title: string | null;
  path: string;
  bytes: number;
  occurrences: number;
  by_origin: Record<string, number>;
  first_line: number | null;
  last_line: number | null;
}

export interface KeywordReport {
  schema: string;
  term: string;
  pattern: string;
  case_sensitive: boolean;
  project: { slug: string; path: string; matched_how: string } | null;
  fidelity: {
    counts: string;
    channel_attribution: string;
    sense_classification: string;
    context_persistence: string;
    context_persistence_note: string;
  };
  scanned: {
    sessions: number;
    bytes: number;
    lines: number;
    lines_examined: number;
    lines_unparsable: number;
    duplicate_fields_folded: number;
    elapsed_s: number;
  };
  totals: {
    counted: number;
    suppressed: number;
    excluded: number;
    sessions_with_hits: number;
  };
  by_origin: Record<string, number>;
  by_channel: Record<string, number>;
  by_sense: Record<string, number>;
  suppressed_channels: { channel: string; label: string; why: string; hits: number }[];
  surface_forms: Record<string, number>;
  compounds_rejected_by_word_boundary: Record<string, number>;
  compounds_unwordlike: { forms: number; hits: number; note: string };
  compounds_note: string;
  inflections_rejected: Record<string, number>;
  inflections_note: string;
  excluded_by_rule: Record<string, number>;
  sessions: KeywordSessionRow[];
  samples: KeywordSample[];
}

/** A scan in flight, or one that finished. `report` is null until `state` is
 *  `"done"`; `error` carries the reason when it is `"error"`. */
export interface KeywordJob {
  job_id: string;
  state: "running" | "done" | "error";
  term: string;
  project: string;
  matched_how: string;
  started: number;
  finished?: number;
  sessions_total: number;
  bytes_total: number;
  sessions_done: number;
  bytes_done: number;
  hits_so_far: number;
  /** The session being read right now — the only honest progress label while
   *  a single 130 MB transcript is under the head for half a minute. */
  reading: string;
  report: KeywordReport | null;
  error: string | null;
}

export interface KeywordRequest {
  term: string;
  project: string;
  projects_root?: string;
  sessions?: string[];
  max_sessions?: number;
  regex?: boolean;
  case_sensitive?: boolean;
  only_channels?: string[];
  senses?: Record<string, string>;
  excludes?: Record<string, string>;
  window?: number;
  samples?: number;
}

// ── the five origins, in the order the answer reads ─────────────────────────

/** Fixed render order: the operator first, the machinery last. Not
 *  frequency-sorted — on real data the harness always wins on volume, and
 *  sorting by count would put boilerplate at the top of the answer to "who
 *  keeps telling it this". */
export const ORIGINS = [
  "human",
  "standing",
  "harness",
  "model",
  "environment",
  "metadata",
] as const;

export type OriginId = (typeof ORIGINS)[number];

/** What each origin means in the operator's own terms — shown beside every
 *  bar, because "standing" and "harness" are not self-explanatory and the
 *  difference between them is the whole diagnosis. */
export const ORIGIN_GLOSS: Record<string, string> = {
  human: "you typed it, in this project",
  standing: "your standing instructions — CLAUDE.md, a skill, session context",
  harness: "harness boilerplate: system prompt, tool schemas, listings",
  model: "the agent's own words — prose, thinking, tool arguments",
  environment: "what it read off this machine — tool output, diffs, files",
  metadata: "bookkeeping the harness repeats; not context the agent read",
};

/** Data colour, from the ramp — not the accent, which belongs to chrome.
 *
 *  Assigned in `ORIGINS` order rather than picked per origin, because the
 *  order is itself a sequence: how far the text is from the operator. Purple
 *  is what you wrote, yellow is what the machine handed back, and the stops in
 *  between are the parties in between. A categorical palette would make the
 *  five look unrelated and hide that reading; the label is on every bar, so
 *  hue carries the gradient and never the identification.
 *
 *  `metadata` is off the ramp on purpose: it is not a party in that chain, it
 *  is the harness's own bookkeeping, and it is only ever counted when the
 *  operator switches one of those channels on. */
export const ORIGIN_COLOR: Record<string, string> = {
  human: "var(--ramp-4)",
  standing: "var(--ramp-3)",
  harness: "var(--ramp-2)",
  model: "var(--ramp-1)",
  environment: "var(--ramp-0)",
  metadata: "var(--text-faint)",
};

// ── requests ────────────────────────────────────────────────────────────────

export function fetchKeywordProjects(
  root?: string,
): Promise<{ root: string; projects: ProjectRow[] }> {
  const q = root ? `?root=${encodeURIComponent(root)}` : "";
  return getJSON(`/seer/keyword/projects${q}`);
}

export function fetchKeywordChannels(): Promise<{
  channels: ChannelRow[];
  origins: string[];
}> {
  return getJSON("/seer/keyword/channels");
}

/** Starts the scan and returns immediately with the job to poll. */
export function startKeywordScan(req: KeywordRequest): Promise<KeywordJob> {
  return postJSON("/seer/keyword", req);
}

export function fetchKeywordJob(jobId: string): Promise<KeywordJob> {
  return getJSON(`/seer/keyword/job/${encodeURIComponent(jobId)}`);
}

// ── formatting ──────────────────────────────────────────────────────────────

export function fmtBytes(n: number): string {
  if (n < 1e6) return `${(n / 1e3).toFixed(0)} kB`;
  if (n < 1e9) return `${(n / 1e6).toFixed(1)} MB`;
  return `${(n / 1e9).toFixed(2)} GB`;
}

export function pct(n: number, total: number): string {
  return total > 0 ? `${Math.round((100 * n) / total)}%` : "—";
}

/** The `seer keyword` invocation that reproduces a report, so what is on
 *  screen can be re-run, scripted and pasted into a write-up. A page that
 *  cannot tell you how it got its numbers is asking to be trusted. */
export function cliFor(req: KeywordRequest, defaultChannels: string[]): string {
  const q = (s: string) => (/^[\w.@%+=:,/-]+$/.test(s) ? s : `'${s.replace(/'/g, "'\\''")}'`);
  const parts = ["seer", "keyword", q(req.term), "--project", q(req.project)];
  if (req.regex) parts.push("--regex");
  if (req.case_sensitive) parts.push("--case-sensitive");
  if (req.max_sessions) parts.push("--max-sessions", String(req.max_sessions));
  for (const s of req.sessions ?? []) parts.push("--session", q(s));
  const chosen = req.only_channels ?? [];
  const same =
    chosen.length === defaultChannels.length &&
    chosen.every((c) => defaultChannels.includes(c));
  if (chosen.length && !same) for (const c of chosen) parts.push("--only-channel", c);
  for (const [name, pat] of Object.entries(req.senses ?? {}))
    parts.push("--sense", q(`${name}=${pat}`));
  for (const [name, pat] of Object.entries(req.excludes ?? {}))
    parts.push("--exclude", q(`${name}=${pat}`));
  if (req.window) parts.push("--window", String(req.window));
  if (req.samples) parts.push("--samples", String(req.samples));
  return parts.join(" ");
}
