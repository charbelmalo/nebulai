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
  cumOutput: number; // cumulative output tokens through this turn (monotonic)

  /** First ≤240 chars of the user prompt this turn was serving (the latest
   *  real user text before it — tool_result lines don't count). null when the
   *  transcript had none; undefined on records persisted before this field. */
  promptPreview?: string | null;
  /** First ≤240 chars of this turn's visible assistant prose. */
  textPreview?: string | null;
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
  filesTouched: [string, number][]; // path → touch count, descending
  errorCount: number; // tool_result blocks flagged is_error
  sidechainTurns: number; // assistant turns from sub-agents
  subAgentCount: number; // distinct sub-agent spawns (parent_tool_use_id values)
  categoryTotals: Record<ToolCategory, number>; // turns per dominant category

  authoritative: SessionAuthoritative | null; // result-line ground truth, if any
  loadedAt: number;
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
 *  when the turn made no tool call at all. */
function dominantCategory(tools: string[]): ToolCategory {
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

function parseTs(s: string | undefined): number | null {
  if (!s) return null;
  const t = Date.parse(s);
  return Number.isFinite(t) ? t : null;
}

function num(u: Record<string, number> | undefined, k: string): number {
  const v = u?.[k];
  return typeof v === "number" && Number.isFinite(v) ? v : 0;
}

/** Count tool failures on one line: the audit format's top-level
 *  `tool_use_result.is_error`, plus the transcript format's user-message
 *  `tool_result` blocks flagged `is_error`. (The result line's session-level
 *  `is_error` is captured separately as authoritative.isError, not here.) */
function countErrors(o: RawLine): number {
  let n = 0;
  if (o.type !== "result" && o.tool_use_result && o.tool_use_result.is_error) n++;
  const content = o.message?.content;
  if (Array.isArray(content)) {
    for (const block of content) {
      if (block && typeof block === "object") {
        const b = block as Record<string, unknown>;
        if (b.type === "tool_result" && b.is_error) n++;
      }
    }
  }
  return n;
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
    errorCount += countErrors(o);

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
            acc.tools.push(b.name);
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
      cumOutput,
      promptPreview: acc.promptPreview,
      textPreview: acc.textPreview,
    });
  });

  const toolHistogram = [...toolCounts.entries()].sort((a, b) => b[1] - a[1]);
  const filesTouched = [...fileCounts.entries()].sort((a, b) => b[1] - a[1]);
  const toolTotal = toolHistogram.reduce((s, [, n]) => s + n, 0);
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
    filesTouched,
    errorCount,
    sidechainTurns,
    subAgentCount: subAgents.size,
    categoryTotals,
    authoritative,
    loadedAt: Date.now(),
  };
}

