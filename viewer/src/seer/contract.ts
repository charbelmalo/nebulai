/** contract.ts — the viewer's mirror of `src/nebulai/seer/contract.py`.
 *
 *  The Python side is the source of truth (it is what writes the append-only
 *  log); this file exists so the UI can be type-safe about what arrives over
 *  SSE. `tests/unit/seer-contract-sync.test.ts` parses the Python enums and
 *  fails if either side gains a member the other lacks — the same drift guard
 *  tokens.css/tokens.ts already use, for the same reason: a silent mismatch
 *  here would show a researcher a category that no longer means what it says.
 *
 *  Rules that the UI must honour, restated because they are easy to break in a
 *  render loop:
 *    · `missing` is never drawn as 0, and `dropped_by_policy` is never drawn as
 *      `missing` — the first is "we don't know", the second is "we chose not to
 *      look", and a researcher needs to tell those apart.
 *    · `*_delta` events update previews only. Never increment a counter from one.
 */

export const SCHEMA_VERSION = "1.0";

export type Fidelity =
  | "native"
  | "deterministic"
  | "estimated"
  | "heuristic"
  | "missing"
  | "dropped_by_policy";

export const FIDELITIES: Fidelity[] = [
  "native",
  "deterministic",
  "estimated",
  "heuristic",
  "missing",
  "dropped_by_policy",
];

/** Fidelities that mean "there is no number here". The UI must render these as
 *  an em dash with a reason, never as a zero. */
export const ABSENT_FIDELITIES: ReadonlySet<Fidelity> = new Set<Fidelity>([
  "missing",
  "dropped_by_policy",
]);

export type CaptureMode = "driven" | "attached" | "observed" | "reconciled";

export const CAPTURE_MODES: CaptureMode[] = ["driven", "attached", "observed", "reconciled"];

export type Action =
  | "inspect"
  | "search"
  | "edit"
  | "execute"
  | "verify"
  | "vcs"
  | "delegate"
  | "interact"
  | "report";

/** Display order for lanes and legends — roughly the order a healthy run moves
 *  through them, so a trajectory reads left-to-right down the legend. */
export const ACTIONS: Action[] = [
  "inspect",
  "search",
  "edit",
  "execute",
  "verify",
  "vcs",
  "delegate",
  "interact",
  "report",
];

export type Effect =
  | "new_information"
  | "no_new_information"
  | "state_changed"
  | "no_state_change"
  | "failed"
  | "unknown";

export const EFFECTS: Effect[] = [
  "new_information",
  "no_new_information",
  "state_changed",
  "no_state_change",
  "failed",
  "unknown",
];

export type TokenCategory =
  | "input"
  | "output"
  | "cache_read"
  | "cache_write"
  | "reasoning"
  | "total";

export const TOKEN_CATEGORIES: TokenCategory[] = [
  "input",
  "output",
  "cache_read",
  "cache_write",
  "reasoning",
  "total",
];

export type SessionState =
  | "starting"
  | "idle"
  | "model_running"
  | "tool_running"
  | "waiting_permission"
  | "waiting_clarification"
  | "waiting_user"
  | "compacting"
  | "interrupted"
  | "completed"
  | "failed"
  | "detached";

export const SESSION_STATES: SessionState[] = [
  "starting",
  "idle",
  "model_running",
  "tool_running",
  "waiting_permission",
  "waiting_clarification",
  "waiting_user",
  "compacting",
  "interrupted",
  "completed",
  "failed",
  "detached",
];

/** Overlays modify a state, they never replace it: a stalled tool run is still
 *  a tool run, and which tool it is stalled on is the actionable half. */
export type StateOverlay = "stalled" | "overdue";

export const TERMINAL_STATES: ReadonlySet<SessionState> = new Set<SessionState>([
  "completed",
  "failed",
  "interrupted",
]);

export type Outcome =
  | "unknown"
  | "agent_claimed_complete"
  | "unverified_complete"
  | "verified_pass"
  | "verified_partial"
  | "verified_fail"
  | "interrupted"
  | "infrastructure_failure";

export const OUTCOMES: Outcome[] = [
  "unknown",
  "agent_claimed_complete",
  "unverified_complete",
  "verified_pass",
  "verified_partial",
  "verified_fail",
  "interrupted",
  "infrastructure_failure",
];

/** Outcomes backed by an evaluator record. Anything else is the agent's own
 *  word and must be labelled as such in the UI. */
export const VERIFIED_OUTCOMES: ReadonlySet<Outcome> = new Set<Outcome>([
  "verified_pass",
  "verified_partial",
  "verified_fail",
]);

export interface Source {
  agent: string;
  agent_version: string;
  adapter: string;
  adapter_version: string;
  capture_mode: CaptureMode;
  fidelity: Fidelity;
  source_event_id?: string;
}

export interface SeerEvent {
  schema_version: string;
  event_id: string;
  ts: number;
  mono_ns: number;
  source: Source;
  run_id: string;
  session_id: string;
  turn_id?: string;
  span_id?: string;
  parent_span_id?: string;
  event_type: string;
  action?: Action;
  effect?: Effect;
  native_type?: string;
  repo?: { root_id?: string; branch?: string; head?: string; dirty?: boolean };
  model?: { provider?: string; model_id?: string; effort?: string };
  payload: Record<string, unknown>;
  native?: Record<string, unknown>;
  privacy: { content_level?: string; ruleset?: string };
}

/** A number that knows where it came from. Every figure the UI prints should be
 *  one of these, so "we don't know" can never render as 0. */
export interface Measured {
  value: number | null;
  fidelity: Fidelity;
  /** Why the value is absent, when it is. Shown in the data-quality panel. */
  note?: string;
}

export function isAbsent(m: Measured): boolean {
  return m.value === null || ABSENT_FIDELITIES.has(m.fidelity);
}

/** Render a Measured for display. Never returns "0" for an absent value. */
export function formatMeasured(m: Measured, fmt: (n: number) => string = String): string {
  if (isAbsent(m)) return "—";
  return fmt(m.value as number);
}

/** True for streaming fragments. Mirrors `EventType.is_delta`. */
export function isDelta(eventType: string): boolean {
  return eventType.endsWith("_delta");
}
