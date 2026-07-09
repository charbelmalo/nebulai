/**
 * session-report — reconstruct a Sessions-tab analysis from the terminal.
 *
 * Runs the exact same parser the viewer uses (`src/chrome/sessionlog.ts`) over a
 * session log and prints the derived analysis: format, response/agent counts, the
 * per-agent context geometry (the honesty signature behind the split-path plot),
 * and the audit result-line ground truth. No data leaves your machine.
 *
 *   npm run report                       # newest *.jsonl in viewer/local-sessions/
 *   npm run report -- path/to/audit.jsonl
 *   npm run report -- "~/Downloads/.../local_<id>/audit.jsonl"
 *
 * Requires Node ≥ 22.6 (uses --experimental-strip-types; wired in package.json).
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseSessionTranscript, type SessionAnalysis } from "../src/chrome/sessionlog.ts";

const HERE = fileURLToPath(new URL(".", import.meta.url));
const DROP = resolve(HERE, "../local-sessions");

function expand(p: string): string {
  return p.startsWith("~") ? join(homedir(), p.slice(1)) : p;
}

/** Newest *.jsonl in the gitignored drop folder, else null. */
function newestDrop(): string | null {
  let files: string[];
  try {
    files = readdirSync(DROP).filter((f) => f.endsWith(".jsonl"));
  } catch {
    return null;
  }
  if (!files.length) return null;
  files.sort(
    (a, b) => statSync(join(DROP, b)).mtimeMs - statSync(join(DROP, a)).mtimeMs,
  );
  return join(DROP, files[0]!);
}

const arg = process.argv[2];
const file = arg ? resolve(expand(arg)) : newestDrop();
if (!file) {
  console.error(
    `No log given and none found in ${DROP}\n` +
      `Drop an audit.jsonl / transcript there, or pass a path:\n` +
      `  npm run report -- path/to/audit.jsonl`,
  );
  process.exit(1);
}

const raw = readFileSync(file, "utf8");
const a: SessionAnalysis = parseSessionTranscript(raw, file.split("/").pop() ?? "session");

const n = (x: number | null | undefined) => (x == null ? "—" : x.toLocaleString());
const dur = (ms: number | null) =>
  ms == null ? "—" : ms < 1000 ? `${ms}ms` : ms < 60_000 ? `${(ms / 1000).toFixed(1)}s` : `${(ms / 60_000).toFixed(1)}m`;

console.log(`\n▌ ${a.name}`);
console.log(`  file      ${file}`);
console.log(
  `  format    ${a.format}   ·   events ${a.events} (${a.nAssistant} assistant / ${a.nUser} user)   ·   span ${dur(a.spanSec * 1000)}`,
);
console.log(
  `  responses ${a.turns.length}   ·   sub-agents ${a.subAgentCount} (${a.sidechainTurns} sidechain turns)   ·   tools ${a.toolTotal}   ·   errors ${a.errorCount}`,
);
console.log(
  `  context   peak ${n(a.contextPeak)} tok (Y axis)   ·   cache-write peak ${n(a.cacheWritePeak)} tok (Z axis)   ·   output ${n(a.totalOutput)}${a.outputReliable ? "" : " ✧ (result-line total; per-turn output is streamed-partial)"}`,
);

// ── authoritative result-line ground truth (audit only) ──────────────────────
const au = a.authoritative;
if (au) {
  console.log(`\n  ▸ result-line ground truth (authoritative)`);
  console.log(
    `    SDK num_turns ${n(au.numTurns)}${au.numTurns != null && au.numTurns !== a.turns.length ? `  (≠ ${a.turns.length} responses — SDK's own turn metric, do not equate)` : ""}`,
  );
  console.log(
    `    output ${n(au.outputTokens)}   input ${n(au.inputTokens)}   cache-read ${n(au.cacheReadTokens)}   cache-write ${n(au.cacheCreationTokens)}`,
  );
  console.log(
    `    cost $${au.costUsd?.toFixed(4) ?? "—"}   wall ${dur(au.durationMs)}   api ${dur(au.apiDurationMs)}   models ${au.models.join(", ") || "—"}${au.isError ? "   ⚠ isError" : ""}`,
  );
}

// ── per-agent context geometry (the split-path honesty signature) ────────────
type Agg = { crStart: number; crPeak: number; crEnd: number; nn: number; isSub: boolean };
const byAgent = new Map<string, Agg>();
for (const t of a.turns) {
  const id = t.agentId ?? "main";
  let e = byAgent.get(id);
  if (!e) {
    e = { crStart: t.cacheRead, crPeak: t.cacheRead, crEnd: t.cacheRead, nn: 0, isSub: t.isSidechain };
    byAgent.set(id, e);
  }
  e.crPeak = Math.max(e.crPeak, t.cacheRead);
  e.crEnd = t.cacheRead;
  e.nn++;
}
const main = byAgent.get("main");
console.log(`\n  ▸ per-agent context geometry  (each agent = one polyline; sub-agents start fresh)`);
console.log(`    agent                          turns  role        ctx@start     ctx@peak   %ofMainPeak`);
for (const [id, e] of byAgent) {
  const pct = main && main.crPeak > 0 ? `${((100 * e.crStart) / main.crPeak).toFixed(1)}%` : "—";
  console.log(
    `    ${(id === "main" ? "main" : id.slice(0, 26)).padEnd(30)} ${String(e.nn).padStart(3)}   ${(e.isSub ? "sub-agent" : "MAIN").padEnd(10)} ${n(e.crStart).padStart(10)}   ${n(e.crPeak).padStart(10)}   ${pct.padStart(8)}`,
  );
}
if (main && a.subAgentCount > 0) {
  const maxSubStart = Math.max(
    0,
    ...[...byAgent].filter(([id]) => id !== "main").map(([, e]) => e.crStart),
  );
  const honest = maxSubStart <= main.crPeak * 0.5;
  console.log(
    `\n  ${honest ? "✓" : "✗"} ${honest ? "Sub-agents each start well below the main spine's peak → separate fresh windows (split path is honest)." : "A sub-agent starts near the main peak — inspect before trusting the split."}`,
  );
}

// ── dominant-category & top tools ────────────────────────────────────────────
const cats = Object.entries(a.categoryTotals)
  .filter(([, c]) => c > 0)
  .sort((x, y) => y[1] - x[1]);
console.log(`\n  ▸ turns by dominant category   ${cats.map(([k, c]) => `${k} ${c}`).join("  ·  ")}`);
console.log(`  ▸ top tools                    ${a.toolHistogram.slice(0, 8).map(([k, c]) => `${k}×${c}`).join("  ·  ")}\n`);
