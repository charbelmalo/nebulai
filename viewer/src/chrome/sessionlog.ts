/** sessionlog.ts — rich parser for Claude Code *agent-mode* session
 *  transcripts (the `.jsonl` under `.claude/projects/<enc>/<id>.jsonl`).
 *
 *  This is deliberately richer than `snapshot.ts` (which flattens any log to
 *  role+text keyword co-occurrence). Real agent sessions carry structure the
 *  keyword map throws away: per-response token usage, a growing context
 *  window, an ordered tool-call sequence, task lifecycle, file touches, and
 *  sub-agent sidechains. This module extracts those REAL quantities so the
 *  Sessions view can plot a session as an honest trajectory. Everything runs
 *  client-side; the raw transcript never leaves the browser.
 *
 *  CORRECTNESS — the honest turn unit is the `requestId`, not the JSONL line.
 *  Claude Code writes ONE model response as SEVERAL lines (one per content
 *  block: thinking / text / tool_use), repeating the identical `usage` on
 *  every line. Counting per-line overcounts tokens and turns several-fold
 *  (measured 3.5× on a real 10-response session). We fold all lines sharing a
 *  `requestId` into a single turn and count `usage` exactly once.
 */

export type ToolCategory =
  | "orient" // read / search / fetch — gathering context
  | "plan" // task lifecycle, plan mode
  | "edit" // write / edit files
  | "exec" // run commands
  | "deliver" // present / publish / notify
  | "reflect"; // pure text or thinking, no tool call

/** One logical turn = one model response (folded across its JSONL lines),
 *  carrying the real usage + the ordered tools it invoked. */
export interface SessionTurn {
  index: number; // position among assistant turns, 0-based
  requestId: string;
  tMs: number | null; // wall-clock epoch ms, null if the line lacked a timestamp
  tSec: number; // seconds since the first timestamped event (0 if unknown)
  isSidechain: boolean; // true when emitted by a spawned sub-agent
  agentId: string; // "main" or the spawning tool_use id — the agent's own context
  model: string | null;

  // real token accounting (counted ONCE per requestId)
  inputTokens: number;
  outputTokens: number;
  cacheRead: number; // context size fed to the model this turn (the growing window)
  cacheWrite: number; // new tokens written to cache this turn
  iterations: number | null;

  tools: string[]; // ordered tool_use names across the whole response
  files: string[]; // file paths touched (Write/Edit/Read inputs)
  thinkingBlocks: number;
  textLen: number; // visible assistant prose length (chars)

  category: ToolCategory; // dominant category of this turn's tools
  /** Tool calls issued by THIS turn that came back flagged is_error.
   *  `undefined` — not 0 — on records persisted before per-turn attribution
   *  existed: the raw transcript is never stored, so those genuinely cannot
   *  know, and the UI says "not recorded" rather than printing a false zero. */
  errors?: number;
  cumOutput: number; // cumulative output tokens through this turn (monotonic)

  /** First ≤240 chars of the user prompt this turn was serving (the latest
   *  real user text before it — tool_result lines don't count). null when the
   *  transcript had none; undefined on records persisted before this field. */
  promptPreview?: string | null;
  /** First ≤240 chars of this turn's visible assistant prose. */
  textPreview?: string | null;
}

/** One turn's context window, decomposed.
 *
 *  `reused + written + fresh` is the whole prompt that turn — three disjoint,
 *  separately-metered parts of one measured number, not an estimate. */
export interface ContextSlice {
  /** Index into `SessionAnalysis.turns`, so the chart and the field agree. */
  turn: number;
  tSec: number;
  reused: number; // cache_read — the conversation already in cache
  written: number; // cache_creation — content cached for the first time this turn
  fresh: number; // input_tokens — sent uncached
  prompt: number; // the three above; the real size of the window this turn
  output: number; // what the model added, which lands in the NEXT prompt

  /** `prompt` minus the previous turn's `prompt`. null on the first turn:
   *  there is nothing to difference against, and 0 would read as "no growth". */
  growth: number | null;
  /** The previous turn's `output` — what the model wrote just before this
   *  prompt was assembled. Exactly measured, and deliberately named for what it
   *  IS rather than for what it did: it is NOT the model's contribution to this
   *  window. Thinking tokens are billed as output and then dropped from the
   *  next prompt, so this routinely exceeds what actually carried forward.
   *  Measured on a real 731-turn transcript: 8 steps where the window grew by
   *  less than the model had just written. */
  priorOutput: number | null;
  /** `growth − priorOutput`. A RESIDUAL, not an attribution — it mixes two
   *  things that cannot be separated from a transcript: content that entered
   *  the window without being billed as output (tool results, typed prompts),
   *  and model output that was billed but never re-fed (thinking). Kept because
   *  its sign is informative; never labelled as "what the tools contributed". */
  residual: number | null;
}

/** How the context window was used across a session.
 *
 *  This is the honest form of requested feature #4. The spec asked for a
 *  system / user / output / retrieved split; a Claude Code transcript meters
 *  tokens per REQUEST, not per content block, so three of those four cannot be
 *  separated and "retrieved" does not exist in this repo at all (no RAG — see
 *  the §2 audit row for #7).
 *
 *  What the transcript DOES meter exactly is the cache decomposition of every
 *  prompt — reused / newly cached / uncached — which is a true partition of a
 *  real number and is what the columns draw.
 *
 *  Alongside it sit two more measured series, `growth` and `priorOutput`, shown
 *  as two numbers rather than combined into one attribution. Combining them is
 *  the tempting mistake: `growth − priorOutput` looks like "what the tools
 *  contributed", and it is not. On a real 731-turn transcript that quantity
 *  sums to −351k, because five compactions dominate it and because thinking
 *  tokens are billed as output and then never re-fed. */
export interface ContextComposition {
  slices: ContextSlice[]; // main-agent turns, in order
  peakPrompt: number;
  /** Turns where the window SHRANK. Compaction, or a context reset. Surfaced
   *  rather than clamped: a negative growth is a real event, and clamping it to
   *  zero would silently turn "the window was rebuilt" into "nothing happened". */
  compactedAt: number[];
  /** Sub-agent turns left out. Each sub-agent runs in its OWN window, so
   *  differencing a series that interleaves them measures the gap between two
   *  unrelated conversations. Reported so the omission is visible. */
  excludedSidechain: number;
  /** False when some response reported different prompt usage on different
   *  lines, so the per-field max mixes them and the three parts sum to an upper
   *  bound rather than to any one request's prompt.
   *
   *  `null` means the check itself wasn't recorded — a composition rebuilt from
   *  a persisted analysis. The per-turn token counts survive persistence, so
   *  the decomposition is as real as ever; only the knowledge of whether the
   *  usage lines agreed is gone, and `false` would assert an inexactness we
   *  have no evidence for. */
  exact: boolean | null;
}

/** Ground-truth session totals, present only in the SDK *audit* format's
 *  terminal `result` line. When present these are AUTHORITATIVE — the streamed
 *  per-assistant-line `usage` is partial (it logs only the first chunk), so the
 *  real output/cost/duration live here. We surface them verbatim and use them
 *  to validate the fold (num_turns must equal our folded turn count). */
export interface SessionAuthoritative {
  numTurns: number | null;
  outputTokens: number | null;
  inputTokens: number | null;
  cacheReadTokens: number | null;
  cacheCreationTokens: number | null;
  costUsd: number | null;
  durationMs: number | null;
  apiDurationMs: number | null;
  models: string[]; // from modelUsage keys (opus + any sub-agent haiku, …)
  isError: boolean;
  permissionDenials: number;
}

export interface SessionAnalysis {
  id: string;
  name: string;
  model: string | null;
  cwd: string | null;
  gitBranch: string | null;
  format: "audit" | "transcript"; // SDK audit.jsonl vs .claude/projects transcript

  turns: SessionTurn[]; // assistant turns only — the trajectory nodes
  events: number; // total logical events (assistant + user)
  nAssistant: number;
  nUser: number;
  spanSec: number; // wall-clock span first→last timestamp

  totalInput: number;
  totalOutput: number; // authoritative when a result line exists, else summed
  totalCacheWrite: number;
  contextPeak: number; // max cacheRead across turns (biggest window reached)
  cacheWritePeak: number; // max cacheWrite across turns (biggest single injection)
  outputReliable: boolean; // false when per-turn output is streamed-partial (audit)

  toolHistogram: [string, number][]; // tool name → count, descending
  toolTotal: number;
  toolOutcomes: ToolOutcome[]; // per-tool outcome split, descending by total
  /** Failures the transcript reports but cannot charge to any one tool: the
   *  audit format's top-level `tool_use_result.is_error` carries no
   *  `tool_use_id`. Counted in `errorCount`, absent from `toolOutcomes`. Any
   *  view of the outcome split MUST surface this, or it silently under-reports
   *  failure. */
  unattributedErrors: number;
  filesTouched: [string, number][]; // path → touch count, descending
  errorCount: number; // tool_result blocks flagged is_error
  sidechainTurns: number; // assistant turns from sub-agents
  subAgentCount: number; // distinct sub-agent spawns (parent_tool_use_id values)
  categoryTotals: Record<ToolCategory, number>; // turns per dominant category

  authoritative: SessionAuthoritative | null; // result-line ground truth, if any
  /** Context window decomposition (#4). `undefined` — not an empty composition —
   *  on analyses persisted before this existed: the raw transcript is never
   *  stored, so those genuinely cannot know, and the UI must say so rather than
   *  draw an empty chart that looks like a session which used no context. */
  context?: ContextComposition;
  loadedAt: number;
}

/** How one tool's calls turned out.
 *
 *  Three buckets, and the third is the honest one. The spec this came from
 *  asked for **success / partial / fail**; a Claude Code transcript has no
 *  notion of a partial tool result. A `tool_result` block either carries
 *  `is_error` or it does not — there is no third flag, no exit code, no
 *  severity. Inventing a "partial" bucket would mean inventing a rule for
 *  which successes are secretly half-failures, and every such rule is a guess
 *  dressed as a measurement.
 *
 *  What IS real, and what "partial" was probably reaching for, is
 *  `unresolved`: a `tool_use` for which no `tool_result` ever appears. That
 *  happens when a call was interrupted or the log was truncated mid-flight. It
 *  is deliberately NOT folded into `ok` — an unanswered call is not a
 *  successful one, and folding it would overstate the success rate. */
export interface ToolOutcome {
  tool: string;
  ok: number;
  failed: number;
  unresolved: number;
  total: number;
}

// ── tool → category ──────────────────────────────────────────────────────────

/** Strip an MCP namespace (`mcp__workspace__bash` → `bash`) so categorisation
 *  keys on the leaf verb regardless of which server provided it. */
function toolLeaf(name: string): string {
  const parts = name.split("__");
  return parts[parts.length - 1] ?? name;
}

const PLAN = /^(Task(Create|Update|Stop|Get|List|Output)|Enter(PlanMode|Worktree)|Exit(PlanMode|Worktree)|TodoWrite)$/;
const EDIT = /^(Write|Edit|MultiEdit|NotebookEdit|create_file|create_new_file|str_replace|multi_str_replace|replace_symbol|insert_at)/i;
const EXEC = /^(Bash|bash|run_|shell|execute)/i;
const DELIVER = /^(present_files|Artifact|gif_creator|export_|upload_|download_|PushNotification|send_message|SendMessage)/i;
const ORIENT =
  /^(Read|Grep|Glob|LS|ls|find|ToolSearch|WebFetch|WebSearch|Search|search|context|read_|get_|list_|fetch|query|snapshot)/i;

export function categorizeTool(name: string): ToolCategory {
  const leaf = toolLeaf(name);
  if (PLAN.test(leaf)) return "plan";
  if (EDIT.test(leaf)) return "edit";
  if (EXEC.test(leaf)) return "exec";
  if (DELIVER.test(leaf)) return "deliver";
  if (ORIENT.test(leaf)) return "orient";
  return "orient"; // unknown tools default to context-gathering
}

/** Dominant category for a turn: most frequent among its tools; `reflect`
 *  when the turn made no tool call at all.
 *
 *  Exported because `tools[]` — not `category` — is the durable field. Analyses
 *  persisted by another build can carry a category this build has no colour or
 *  legend row for, and a renderer that trusts the stored value blanks its whole
 *  node layer on the first unknown one. Re-deriving from the stored tools is
 *  always safe: it's the same computation that produced the field originally. */
export function dominantCategory(tools: string[]): ToolCategory {
  if (tools.length === 0) return "reflect";
  const counts = new Map<ToolCategory, number>();
  for (const t of tools) {
    const c = categorizeTool(t);
    counts.set(c, (counts.get(c) ?? 0) + 1);
  }
  let best: ToolCategory = "orient";
  let bestN = -1;
  for (const [c, n] of counts) {
    if (n > bestN) {
      best = c;
      bestN = n;
    }
  }
  return best;
}

export const CATEGORY_ORDER: ToolCategory[] = [
  "orient",
  "plan",
  "edit",
  "exec",
  "deliver",
  "reflect",
];

// ── parsing ──────────────────────────────────────────────────────────────────

interface RawLine {
  type?: string;
  subtype?: string;
  requestId?: string;
  uuid?: string;
  parentUuid?: string | null;
  parent_tool_use_id?: string | null; // set on SDK sub-agent (sidechain) lines
  isSidechain?: boolean;
  timestamp?: string;
  _audit_timestamp?: string; // audit format: the real per-line wall-clock stamp
  cwd?: string;
  gitBranch?: string;
  message?: {
    id?: string; // audit format: shared across a response's streamed lines
    role?: string;
    model?: string;
    content?: unknown;
    usage?: Record<string, number>;
  };
  // terminal audit `result` line — authoritative session totals
  num_turns?: number;
  total_cost_usd?: number;
  duration_ms?: number;
  duration_api_ms?: number;
  is_error?: boolean;
  usage?: Record<string, number>;
  modelUsage?: Record<string, unknown>;
  permission_denials?: unknown[];
  tool_use_result?: { is_error?: boolean } | null; // audit: tool execution outcome
}

interface Acc {
  requestId: string;
  tMs: number | null;
  isSidechain: boolean;
  parentToolUseId: string | null; // spawning tool_use id for sub-agent turns
  model: string | null;
  usage: Record<string, number>; // per-field MAX across the group's streamed lines
  /** True once two lines of this response reported DIFFERENT prompt usage.
   *
   *  The per-field max then draws its fields from different lines, so their sum
   *  is an upper bound on the prompt rather than any one request's prompt. That
   *  is fine for the peak tiles (a max of maxes is still a max) but it is not a
   *  decomposition, and the composition view has to say so instead of drawing
   *  three exact-looking segments over a number no request ever saw. */
  usageVaried: boolean;
  tools: string[];
  files: string[];
  thinkingBlocks: number;
  textLen: number;
  order: number;
  promptPreview: string | null;
  textPreview: string | null;
}

/** Collapse whitespace and clip to a short inspector preview. */
function clipPreview(s: string): string {
  const t = s.replace(/\s+/g, " ").trim();
  return t.length > 240 ? `${t.slice(0, 239)}…` : t;
}

/** Real user text from a user line's content — the prompt, not a tool_result.
 *  Lines carrying tool_result blocks are the harness feeding results back, so
 *  they never count as a prompt even if a text block rides along. */
function userText(content: unknown): string | null {
  if (typeof content === "string") return content.trim() || null;
  if (!Array.isArray(content)) return null;
  let out = "";
  for (const block of content) {
    if (!block || typeof block !== "object") continue;
    const b = block as Record<string, unknown>;
    if (b.type === "tool_result") return null;
    if (b.type === "text" && typeof b.text === "string") out += `${b.text} `;
  }
  return out.trim() || null;
}

/** True when a raw log string looks like a Claude Code agent transcript — either
 *  the `.claude/projects` transcript or the SDK `audit.jsonl` format. Scans the
 *  first lines for an assistant/user/result line carrying usage, a requestId, a
 *  message id, or a uuid. Lets the caller route here vs the keyword parser. */
export function looksLikeSessionTranscript(raw: string): boolean {
  const lines = raw.trim().split(/\r?\n/, 40);
  for (const l of lines) {
    const s = l.trim();
    if (!s.startsWith("{")) continue;
    try {
      const o = JSON.parse(s) as RawLine;
      if (o.type === "result" && (o.num_turns != null || o.usage != null)) return true;
      if (
        (o.type === "assistant" || o.type === "user") &&
        (!!o.requestId || !!o.message?.usage || !!o.message?.id || !!o.uuid)
      ) {
        return true;
      }
    } catch {
      // skip non-JSON lines
    }
  }
  return false;
}

/** Merge a streamed line's usage into a group accumulator by keeping the MAX of
 *  each numeric field. The audit format writes several lines per response, each
 *  a partial snapshot: cache_creation peaks on the first (pre-tool) line,
 *  cache_read on the last. Taking the per-field max recovers the fullest picture
 *  regardless of line order. For the transcript format (identical usage repeated
 *  per line) max is a no-op, so this is safe for both. */
function mergeUsageMax(into: Record<string, number>, next: Record<string, number> | undefined): void {
  if (!next) return;
  for (const k in next) {
    const v = next[k];
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    const cur = into[k];
    if (cur === undefined || v > cur) into[k] = v;
  }
}

/** Decompose the context window across a session's main-agent turns.
 *
 *  Pure over `SessionTurn[]`, so it is testable without a transcript.
 *
 *  Sub-agent turns are dropped, not summed in: each runs in its own window, and
 *  differencing a series that interleaves two independent conversations
 *  measures the gap between them, which is not growth of anything.
 *
 *  @param exact false when any response reported inconsistent prompt usage
 *               across its lines; null when rebuilding from a persisted
 *               analysis, where the check was never stored — see
 *               `ContextComposition.exact`. */
export function buildComposition(
  turns: SessionTurn[],
  exact: boolean | null,
): ContextComposition {
  const main = turns.filter((t) => !t.isSidechain);
  const slices: ContextSlice[] = [];
  const compactedAt: number[] = [];
  let peakPrompt = 0;

  for (let i = 0; i < main.length; i++) {
    const t = main[i]!;
    const prev = i > 0 ? main[i - 1]! : null;
    const prompt = t.cacheRead + t.cacheWrite + t.inputTokens;
    if (prompt > peakPrompt) peakPrompt = prompt;
    const growth = prev ? prompt - (prev.cacheRead + prev.cacheWrite + prev.inputTokens) : null;
    const priorOutput = prev ? prev.outputTokens : null;
    if (growth !== null && growth < 0) compactedAt.push(t.index);
    slices.push({
      turn: t.index,
      tSec: t.tSec,
      reused: t.cacheRead,
      written: t.cacheWrite,
      fresh: t.inputTokens,
      prompt,
      output: t.outputTokens,
      growth,
      priorOutput,
      residual: growth !== null && priorOutput !== null ? growth - priorOutput : null,
    });
  }

  return {
    slices,
    peakPrompt,
    compactedAt,
    excludedSidechain: turns.length - main.length,
    exact,
  };
}

/* ---- the agent path — requested feature #2 ------------------------------ */

/** One step along ONE agent's own path.
 *
 *  `step` counts within the agent, not the session, because that is the number
 *  an agent's own trace is read by: a sub-agent's third step is its third step
 *  whatever the surrounding session was doing. On a single-agent session the
 *  two numbers coincide, which is why the view only shows this one when they
 *  can differ. */
export interface AgentStep {
  /** Index into `SessionAnalysis.turns` — the join key with the field. */
  turn: number;
  agentId: string;
  step: number; // 1-based, within this agent
  ofSteps: number; // how many steps this agent ran in total
  /** Wall-clock seconds since this agent's PREVIOUS step.
   *
   *  Null on an agent's first step (nothing to difference against) and null
   *  whenever either endpoint lacked a timestamp — `tSec` falls back to 0 when
   *  a line has no clock, and differencing against that manufactures a gap out
   *  of a missing measurement. Hence `tMs`, which is null when unknown.
   *
   *  This is elapsed time between two responses, NOT how long the step took:
   *  it contains tool execution, model latency, and — on a turn that followed
   *  a human prompt — however long the human took. Named for what it measures. */
  gapSec: number | null;
}

/** One agent's steps, in transcript order. Not necessarily contiguous in the
 *  session: a sub-agent runs *inside* one of the parent's tool calls, so the
 *  parent's steps resume after it. The edge that spans that interruption is
 *  real — the parent genuinely continued from there. */
export interface AgentPath {
  agentId: string;
  isSidechain: boolean;
  steps: AgentStep[];
  spanSec: number | null; // first→last step of this agent; null when untimed
}

export interface AgentGraph {
  paths: AgentPath[];
  /** turn index → its step. */
  byTurn: Map<number, AgentStep>;
  /** False when NO step in the session could be timed, so every `gapSec` is
   *  null for want of a clock rather than for want of a predecessor. Lets the
   *  view say "not recorded" once instead of printing dashes forever. */
  timed: boolean;
}

/** Split a session's turns into one path per agent, numbering and timing each
 *  step within its own agent.
 *
 *  The session field draws the trail from these paths rather than from a single
 *  walk over `turns`, because a single walk emits an edge across every change
 *  of agent — parent→sub and sub→parent — and neither is a step either agent
 *  took. NOTE that on this machine that defect fires on nothing: 0 of 10,391
 *  turns across 239 real transcripts carry `isSidechain`. The multi-agent path
 *  below is therefore correct by construction and by unit test, and unexercised
 *  by any real recording. */
export function buildAgentGraph(turns: SessionTurn[]): AgentGraph {
  const order: string[] = [];
  const groups = new Map<string, SessionTurn[]>();
  for (const t of turns) {
    let g = groups.get(t.agentId);
    if (!g) {
      g = [];
      groups.set(t.agentId, g);
      order.push(t.agentId);
    }
    g.push(t);
  }

  const paths: AgentPath[] = [];
  const byTurn = new Map<number, AgentStep>();
  let timed = false;

  for (const agentId of order) {
    const g = groups.get(agentId)!;
    const steps: AgentStep[] = [];
    for (let i = 0; i < g.length; i++) {
      const t = g[i]!;
      const prev = i > 0 ? g[i - 1]! : null;
      const gapSec =
        prev && prev.tMs !== null && t.tMs !== null ? (t.tMs - prev.tMs) / 1000 : null;
      if (gapSec !== null) timed = true;
      const s: AgentStep = { turn: t.index, agentId, step: i + 1, ofSteps: g.length, gapSec };
      steps.push(s);
      byTurn.set(t.index, s);
    }
    const first = g[0]!;
    const last = g[g.length - 1]!;
    paths.push({
      agentId,
      isSidechain: first.isSidechain,
      steps,
      spanSec: first.tMs !== null && last.tMs !== null ? (last.tMs - first.tMs) / 1000 : null,
    });
  }

  return { paths, byTurn, timed };
}

/** The three prompt fields the composition decomposes. `output_tokens` is
 *  deliberately absent: it varies line to line by design (the audit format logs
 *  the first chunk, then the total) and that variation says nothing about
 *  whether the PROMPT was reported consistently. */
const PROMPT_FIELDS = [
  "input_tokens",
  "cache_read_input_tokens",
  "cache_creation_input_tokens",
] as const;

/** Did this line's usage disagree with what we already had about the prompt?
 *  Called before the merge, so `into` still holds the previous state. A field
 *  appearing for the first time is not disagreement — only a different value
 *  for a field we already saw. */
function usageDisagrees(
  into: Record<string, number>,
  next: Record<string, number> | undefined,
): boolean {
  if (!next) return false;
  for (const k of PROMPT_FIELDS) {
    const a = into[k];
    const b = next[k];
    if (a === undefined || typeof b !== "number" || !Number.isFinite(b)) continue;
    if (a !== b) return true;
  }
  return false;
}

function parseTs(s: string | undefined): number | null {
  if (!s) return null;
  const t = Date.parse(s);
  return Number.isFinite(t) ? t : null;
}

function num(u: Record<string, number> | undefined, k: string): number {
  const v = u?.[k];
  return typeof v === "number" && Number.isFinite(v) ? v : 0;
}

/** Tool failures on one line, with the `tool_use_id`s they belong to so each
 *  failure can be charged to the turn that made the call.
 *
 *  The transcript format flags failures on `tool_result` blocks; the audit
 *  format ALSO carries a top-level `tool_use_result.is_error` describing the
 *  same failure, so the top-level flag only counts when no block already did —
 *  otherwise every audit-format failure is counted twice. (The result line's
 *  session-level `is_error` is captured separately as authoritative.isError.) */
function lineErrors(o: RawLine): {
  /** tool_use_ids that failed, so each failure lands on the turn AND the tool
   *  that made the call. Deduplicated globally by the caller, not here — see
   *  the replay note in `parseSessionTranscript`. */
  byToolUse: string[];
  /** Every tool_use_id this line answers, failed or not. A call that never
   *  shows up here got no result at all, which is a distinct outcome from
   *  succeeding — see ToolOutcome. */
  resolved: string[];
  /** Failures on this line that no tool can be charged with: the top-level
   *  audit-format flag carries no tool_use_id, and a malformed `tool_result`
   *  can be missing one. Real failures, unattributable ones. */
  unattributed: number;
} {
  const byToolUse: string[] = [];
  const resolved: string[] = [];
  let unattributed = 0;
  let blockErrors = 0;
  const content = o.message?.content;
  if (Array.isArray(content)) {
    for (const block of content) {
      if (block && typeof block === "object") {
        const b = block as Record<string, unknown>;
        if (b.type !== "tool_result") continue;
        if (typeof b.tool_use_id === "string") resolved.push(b.tool_use_id);
        if (b.is_error) {
          blockErrors++;
          if (typeof b.tool_use_id === "string") byToolUse.push(b.tool_use_id);
          else unattributed++;
        }
      }
    }
  }
  if (blockErrors === 0 && o.type !== "result" && o.tool_use_result?.is_error) unattributed++;
  return { byToolUse, resolved, unattributed };
}

/** Parse a Claude Code transcript into an honest session trajectory.
 *  `name` is the display label (usually the file/session name). */
export function parseSessionTranscript(raw: string, name: string): SessionAnalysis {
  const lines = raw
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);

  const byReq = new Map<string, Acc>();
  const reqOrder: string[] = [];
  let order = 0;

  let model: string | null = null;
  let cwd: string | null = null;
  let gitBranch: string | null = null;
  let nUser = 0;
  let errorCount = 0;
  let minTs: number | null = null;
  let maxTs: number | null = null;
  let resultLine: RawLine | null = null;
  let sawAudit = false; // any line carried an _audit_timestamp / result subtype
  // latest real user prompt, tracked per agent context (main vs each sub-agent)
  // so a sub-agent's task brief doesn't clobber the main conversation's prompt
  const lastPrompt = new Map<string, string>();
  // tool_use_id → the fold key of the response that issued it, so a failure
  // reported on a later user line is charged to the turn that caused it
  const toolUseOwner = new Map<string, string>();
  const errorsByReq = new Map<string, number>();
  // tool_use_id → tool name, so a failure reported on a later user line can be
  // charged to the TOOL as well as to the turn
  const toolUseName = new Map<string, string>();
  const failedToolUses = new Set<string>();
  const resolvedToolUses = new Set<string>();
  let unattributedErrors = 0;
  let idlessCalls = 0;

  for (const line of lines) {
    let o: RawLine;
    try {
      o = JSON.parse(line) as RawLine;
    } catch {
      continue;
    }
    // audit lines stamp _audit_timestamp; transcript lines use timestamp
    if (o._audit_timestamp) sawAudit = true;
    const ts = parseTs(o.timestamp ?? o._audit_timestamp);
    if (ts !== null) {
      if (minTs === null || ts < minTs) minTs = ts;
      if (maxTs === null || ts > maxTs) maxTs = ts;
    }
    if (o.cwd && !cwd) cwd = o.cwd;
    if (o.gitBranch && !gitBranch) gitBranch = o.gitBranch;
    const errs = lineErrors(o);
    errorCount += errs.unattributed;
    unattributedErrors += errs.unattributed;
    for (const id of errs.resolved) resolvedToolUses.add(id);
    for (const id of errs.byToolUse) {
      if (failedToolUses.has(id)) continue; // replayed line — same failure
      failedToolUses.add(id);
      errorCount++;
      const owner = toolUseOwner.get(id);
      if (owner) errorsByReq.set(owner, (errorsByReq.get(owner) ?? 0) + 1);
    }

    // terminal audit result line — authoritative totals (captured, not folded)
    if (o.type === "result") {
      resultLine = o;
      continue;
    }

    if (o.type === "assistant") {
      // Fold unit = one model response. Prefer message.id (audit format shares
      // it across a response's streamed lines) over requestId (transcript
      // format) over uuid. uuid is UNIQUE PER LINE, so using it as the key would
      // treat every streamed chunk as its own turn — the bug that made a
      // 71-response audit session read as 105 turns.
      const rid = o.message?.id ?? o.requestId ?? o.uuid ?? `turn-${order}`;
      let acc = byReq.get(rid);
      if (!acc) {
        const isSide = !!o.isSidechain || o.parent_tool_use_id != null;
        const agentKey = o.parent_tool_use_id ?? (isSide ? "side" : "main");
        acc = {
          requestId: rid,
          tMs: ts,
          isSidechain: isSide,
          parentToolUseId: o.parent_tool_use_id ?? null,
          model: o.message?.model ?? null,
          usage: {},
          usageVaried: false,
          tools: [],
          files: [],
          thinkingBlocks: 0,
          textLen: 0,
          order: order++,
          promptPreview: lastPrompt.get(agentKey) ?? null,
          textPreview: null,
        };
        byReq.set(rid, acc);
        reqOrder.push(rid);
      }
      // usage is streamed as partial snapshots per line — keep the per-field max
      if (acc.tMs === null && ts !== null) acc.tMs = ts;
      else if (ts !== null && ts > (acc.tMs ?? 0)) acc.tMs = ts; // last line = response end
      if (usageDisagrees(acc.usage, o.message?.usage)) acc.usageVaried = true;
      mergeUsageMax(acc.usage, o.message?.usage);
      if (o.parent_tool_use_id != null) {
        acc.isSidechain = true;
        if (acc.parentToolUseId == null) acc.parentToolUseId = o.parent_tool_use_id;
      }
      if (!acc.model && o.message?.model) acc.model = o.message.model;
      if (!model && o.message?.model && o.parent_tool_use_id == null) model = o.message.model;

      const content = o.message?.content;
      if (Array.isArray(content)) {
        for (const block of content) {
          if (!block || typeof block !== "object") continue;
          const b = block as Record<string, unknown>;
          if (b.type === "tool_use" && typeof b.name === "string") {
            // A resumed session replays its earlier lines verbatim into the
            // same file — same uuid, same message.id, same tool_use id, a few
            // hundred lines later. Measured on a real 36 MB transcript: 102 of
            // 1392 tool_use blocks were replays of 1290 actual calls. The turn
            // fold is already immune (replays merge into their original
            // accumulator by message.id); the tool tallies were not, so a
            // resumed session read as 8% busier than it was. The tool_use id
            // is the call's identity, so seeing one twice means one call.
            if (typeof b.id === "string" && toolUseName.has(b.id)) continue;
            acc.tools.push(b.name);
            // remember who issued this call so its failure lands on this turn,
            // and what it was so the failure also lands on the right tool.
            // A call with no id can never be answered and can never be a
            // detectable replay, so it gets a private key: it still counts,
            // and it counts as unresolved, which is exactly what it is.
            toolUseName.set(typeof b.id === "string" ? b.id : ` noid-${idlessCalls++}`, b.name);
            if (typeof b.id === "string") toolUseOwner.set(b.id, rid);
            const inp = b.input as Record<string, unknown> | undefined;
            const p = (inp?.file_path ?? inp?.path) as string | undefined;
            if (typeof p === "string") acc.files.push(p);
          } else if (b.type === "thinking") {
            acc.thinkingBlocks++;
          } else if (b.type === "text" && typeof b.text === "string") {
            acc.textLen += b.text.length;
            if (acc.textPreview === null && b.text.trim()) acc.textPreview = clipPreview(b.text);
          }
        }
      }
    } else if (o.type === "user") {
      nUser++;
      const ut = userText(o.message?.content);
      if (ut) {
        const isSide = !!o.isSidechain || o.parent_tool_use_id != null;
        lastPrompt.set(o.parent_tool_use_id ?? (isSide ? "side" : "main"), clipPreview(ut));
      }
    }
  }

  // fold accumulators → ordered assistant turns with cumulative output
  const t0 = minTs;
  const turns: SessionTurn[] = [];
  const toolCounts = new Map<string, number>();
  const fileCounts = new Map<string, number>();
  const categoryTotals: Record<ToolCategory, number> = {
    orient: 0,
    plan: 0,
    edit: 0,
    exec: 0,
    deliver: 0,
    reflect: 0,
  };
  let cumOutput = 0;
  let totalInput = 0;
  let summedOutput = 0;
  let totalCacheWrite = 0;
  let contextPeak = 0;
  let cacheWritePeak = 0;
  let sidechainTurns = 0;
  let promptUsageExact = true;
  const subAgents = new Set<string>();

  reqOrder.forEach((rid, i) => {
    const acc = byReq.get(rid);
    if (!acc) return;
    const out = num(acc.usage, "output_tokens");
    const inp = num(acc.usage, "input_tokens");
    const cr = num(acc.usage, "cache_read_input_tokens");
    const cw = num(acc.usage, "cache_creation_input_tokens");
    cumOutput += out;
    totalInput += inp;
    summedOutput += out;
    totalCacheWrite += cw;
    if (cr > contextPeak) contextPeak = cr;
    if (cw > cacheWritePeak) cacheWritePeak = cw;
    if (acc.isSidechain) sidechainTurns++;
    if (acc.usageVaried) promptUsageExact = false;
    if (acc.parentToolUseId) subAgents.add(acc.parentToolUseId);
    const category = dominantCategory(acc.tools);
    categoryTotals[category]++;
    for (const t of acc.tools) toolCounts.set(t, (toolCounts.get(t) ?? 0) + 1);
    for (const f of acc.files) fileCounts.set(f, (fileCounts.get(f) ?? 0) + 1);
    const iterRaw = acc.usage["iterations"];
    turns.push({
      index: i,
      requestId: rid,
      tMs: acc.tMs,
      tSec: acc.tMs !== null && t0 !== null ? (acc.tMs - t0) / 1000 : 0,
      isSidechain: acc.isSidechain,
      agentId: acc.parentToolUseId ?? "main",
      model: acc.model,
      inputTokens: inp,
      outputTokens: out,
      cacheRead: cr,
      cacheWrite: cw,
      iterations: typeof iterRaw === "number" ? iterRaw : null,
      tools: acc.tools,
      files: acc.files,
      thinkingBlocks: acc.thinkingBlocks,
      textLen: acc.textLen,
      category,
      errors: errorsByReq.get(rid) ?? 0,
      cumOutput,
      promptPreview: acc.promptPreview,
      textPreview: acc.textPreview,
    });
  });

  const toolHistogram = [...toolCounts.entries()].sort((a, b) => b[1] - a[1]);
  const filesTouched = [...fileCounts.entries()].sort((a, b) => b[1] - a[1]);
  const toolTotal = toolHistogram.reduce((s, [, n]) => s + n, 0);

  // Per-tool outcome split. Walked over the id→name map rather than over
  // `toolCounts`, because only the ids can be matched to a result — and it is
  // the same population either way: every tool_use block records both.
  const outcomeAcc = new Map<string, ToolOutcome>();
  for (const [id, name] of toolUseName) {
    let o = outcomeAcc.get(name);
    if (!o) {
      o = { tool: name, ok: 0, failed: 0, unresolved: 0, total: 0 };
      outcomeAcc.set(name, o);
    }
    o.total++;
    if (failedToolUses.has(id)) o.failed++;
    else if (resolvedToolUses.has(id)) o.ok++;
    else o.unresolved++;
  }
  const toolOutcomes = [...outcomeAcc.values()].sort(
    (a, b) => b.total - a.total || a.tool.localeCompare(b.tool),
  );

  const spanSec = minTs !== null && maxTs !== null ? (maxTs - minTs) / 1000 : 0;

  // ── reconcile with the authoritative result line (audit format) ────────────
  let authoritative: SessionAuthoritative | null = null;
  if (resultLine) {
    const ru = resultLine.usage;
    const mu = resultLine.modelUsage;
    authoritative = {
      numTurns: typeof resultLine.num_turns === "number" ? resultLine.num_turns : null,
      outputTokens: ru ? num(ru, "output_tokens") : null,
      inputTokens: ru ? num(ru, "input_tokens") : null,
      cacheReadTokens: ru ? num(ru, "cache_read_input_tokens") : null,
      cacheCreationTokens: ru ? num(ru, "cache_creation_input_tokens") : null,
      costUsd: typeof resultLine.total_cost_usd === "number" ? resultLine.total_cost_usd : null,
      durationMs: typeof resultLine.duration_ms === "number" ? resultLine.duration_ms : null,
      apiDurationMs: typeof resultLine.duration_api_ms === "number" ? resultLine.duration_api_ms : null,
      models: mu && typeof mu === "object" ? Object.keys(mu) : [],
      isError: !!resultLine.is_error,
      permissionDenials: Array.isArray(resultLine.permission_denials)
        ? resultLine.permission_denials.length
        : 0,
    };
  }

  // The audit format streams only partial per-line output_tokens, so the folded
  // per-turn sum badly under-counts. When we have the result line's ground truth
  // and our sum is far below it, trust the result line and flag output as not
  // reliable per-turn (the Z axis then uses cache-write, which IS per-turn real).
  const authOut = authoritative?.outputTokens ?? null;
  const outputReliable = authOut == null || authOut === 0 ? true : summedOutput >= authOut * 0.6;
  const totalOutput = authOut != null && !outputReliable ? authOut : summedOutput;
  const totalInputR = authoritative?.inputTokens ?? totalInput;
  const totalCacheWriteR = authoritative?.cacheCreationTokens ?? totalCacheWrite;

  return {
    id: `sess-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
    name,
    model,
    cwd,
    gitBranch,
    format: sawAudit || resultLine ? "audit" : "transcript",
    turns,
    events: turns.length + nUser,
    nAssistant: turns.length,
    nUser,
    spanSec,
    totalInput: totalInputR,
    totalOutput,
    totalCacheWrite: totalCacheWriteR,
    contextPeak,
    cacheWritePeak,
    outputReliable,
    toolHistogram,
    toolTotal,
    toolOutcomes,
    unattributedErrors,
    filesTouched,
    errorCount,
    sidechainTurns,
    subAgentCount: subAgents.size,
    categoryTotals,
    authoritative,
    context: buildComposition(turns, promptUsageExact),
    loadedAt: Date.now(),
  };
}

