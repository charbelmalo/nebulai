/** Snapshot Map pipeline — parses JSON conversation logs and extracts
 *  per-turn keyword co-occurrence for a topic filter. Everything runs
 *  in-memory client-side; the raw log never leaves the browser.
 *
 *  Log formats accepted (auto-detected):
 *   1. `[{ role, content }, …]`           — OpenAI-style
 *   2. `[{ role, text }, …]`              — trivial dumps
 *   3. `{ messages: [{ role, content }, …] }` — wrapped
 *   4. `{ transcript: [ … ] }` / `{ turns: […] }`
 *   5. Claude-Code JSONL transcripts     — one JSON per line, each
 *      `{ type: "user"|"assistant", message: { role, content: string | Array }, timestamp? }`
 *
 *  `content` may itself be a string OR an array of `{ type: "text", text }`
 *  blocks — both flatten to a single text blob. Non-text blocks are ignored.
 */

import type { ConversationTurn, SnapshotLog, TopicPreset } from "../app/store";

/** Parse a raw string that could be JSON or JSONL, returning a normalized
 *  turn list. Throws with a friendly message on unrecognized shapes. */
export function parseConversationText(raw: string, name: string): SnapshotLog {
  const trimmed = raw.trim();
  if (!trimmed) throw new Error("empty input");

  let messages: unknown[] = [];

  // JSONL: newline-separated JSON objects
  if (trimmed.includes("\n") && !trimmed.startsWith("[") && !trimmed.startsWith("{")) {
    messages = trimmed
      .split(/\r?\n/)
      .filter((l) => l.trim())
      .map((l) => {
        try {
          return JSON.parse(l);
        } catch {
          return null;
        }
      })
      .filter((m): m is object => !!m);
  } else {
    const parsed = JSON.parse(trimmed);
    if (Array.isArray(parsed)) {
      messages = parsed;
    } else if (parsed && typeof parsed === "object") {
      const p = parsed as Record<string, unknown>;
      const candidate =
        (Array.isArray(p.messages) && p.messages) ||
        (Array.isArray(p.transcript) && p.transcript) ||
        (Array.isArray(p.turns) && p.turns) ||
        (Array.isArray(p.log) && p.log) ||
        (Array.isArray(p.history) && p.history) ||
        null;
      if (!candidate)
        throw new Error("no messages/transcript/turns array found in object");
      messages = candidate;
    } else {
      throw new Error("expected an array or an object with a messages array");
    }
  }

  const turns: ConversationTurn[] = [];
  for (const raw of messages) {
    if (!raw || typeof raw !== "object") continue;
    const m = raw as Record<string, unknown>;
    // Claude-Code JSONL: { type: "user"|"assistant", message: {...} }
    const inner =
      m.message && typeof m.message === "object"
        ? (m.message as Record<string, unknown>)
        : m;
    const role =
      (typeof inner.role === "string" && inner.role) ||
      (typeof m.type === "string" && m.type) ||
      "unknown";
    const content = inner.content ?? inner.text ?? m.text ?? "";
    const text = flattenContent(content);
    if (!text) continue;
    const tsRaw = m.timestamp ?? inner.timestamp;
    const ts =
      typeof tsRaw === "number"
        ? tsRaw
        : typeof tsRaw === "string"
          ? Date.parse(tsRaw) || undefined
          : undefined;
    turns.push({ role, text, ts });
  }

  if (turns.length === 0) throw new Error("parsed 0 turns — check the file format");

  return {
    id: `log-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
    name,
    turns,
    loadedAt: Date.now(),
  };
}

function flattenContent(c: unknown): string {
  if (typeof c === "string") return c;
  if (Array.isArray(c))
    return c
      .map((block) => {
        if (typeof block === "string") return block;
        if (block && typeof block === "object") {
          const b = block as Record<string, unknown>;
          if (typeof b.text === "string") return b.text;
          if (typeof b.content === "string") return b.content;
        }
        return "";
      })
      .filter(Boolean)
      .join(" ");
  if (c && typeof c === "object") {
    const b = c as Record<string, unknown>;
    if (typeof b.text === "string") return b.text;
  }
  return "";
}

// ── extraction ─────────────────────────────────────────────────────────────

export interface KeywordHit {
  keyword: string;
  turnIndex: number;
  role: string;
  count: number;
}

export interface CoEdge {
  a: string;
  b: string;
  weight: number; // number of turns both appeared in, up to turnIndex
  lastTurn: number;
}

export interface SnapshotAnalysis {
  hits: KeywordHit[]; // all hits across all turns (for the current topic)
  perTurn: Map<number, Set<string>>; // turnIndex → set of matched keywords
  totals: Map<string, number>; // keyword → total hit count across visible turns
  edges: CoEdge[]; // co-occurrence edges up to visible turnIndex
  activeKeywords: string[]; // keywords with ≥1 hit up to turnIndex, ranked
}

/** Count case-insensitive substring hits for every keyword in every turn.
 *  Keywords are compared as lowercased phrases; `"safe triangle"` matches
 *  the exact phrase, `"tsl"` matches word-boundary. */
export function analyzeSnapshot(
  log: SnapshotLog | null,
  topic: TopicPreset | null,
  turnIndex: number,
): SnapshotAnalysis {
  const empty: SnapshotAnalysis = {
    hits: [],
    perTurn: new Map(),
    totals: new Map(),
    edges: [],
    activeKeywords: [],
  };
  if (!log || !topic || topic.keywords.length === 0) return empty;

  const kws = topic.keywords.map((k) => k.trim()).filter(Boolean);
  const patterns = kws.map((k) => ({
    k,
    re: new RegExp(k.length <= 4 ? `\\b${escapeRe(k)}\\b` : escapeRe(k), "gi"),
  }));

  const hits: KeywordHit[] = [];
  const perTurn = new Map<number, Set<string>>();
  const totals = new Map<string, number>();

  const cap = Math.min(turnIndex, log.turns.length - 1);
  for (let i = 0; i <= cap; i++) {
    const turn = log.turns[i];
    const matched = new Set<string>();
    for (const { k, re } of patterns) {
      re.lastIndex = 0;
      const m = turn.text.match(re);
      const c = m ? m.length : 0;
      if (c > 0) {
        matched.add(k);
        hits.push({ keyword: k, turnIndex: i, role: turn.role, count: c });
        totals.set(k, (totals.get(k) ?? 0) + c);
      }
    }
    if (matched.size > 0) perTurn.set(i, matched);
  }

  // co-occurrence edges — cumulative up to turnIndex
  const edgeMap = new Map<string, CoEdge>();
  for (const [i, set] of perTurn) {
    const arr = [...set].sort();
    for (let a = 0; a < arr.length; a++) {
      for (let b = a + 1; b < arr.length; b++) {
        const key = `${arr[a]}::${arr[b]}`;
        const existing = edgeMap.get(key);
        if (existing) {
          existing.weight++;
          existing.lastTurn = Math.max(existing.lastTurn, i);
        } else {
          edgeMap.set(key, { a: arr[a], b: arr[b], weight: 1, lastTurn: i });
        }
      }
    }
  }

  const activeKeywords = [...totals.entries()]
    .sort((x, y) => y[1] - x[1])
    .map(([k]) => k);

  return { hits, perTurn, totals, edges: [...edgeMap.values()], activeKeywords };
}

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Radial layout for a keyword set — deterministic (order-based) so the
 *  positions stay stable as the turn scrubber advances. */
export function layoutRadial(
  keywords: string[],
  cx: number,
  cy: number,
  radius: number,
): Map<string, { x: number; y: number }> {
  const out = new Map<string, { x: number; y: number }>();
  if (keywords.length === 0) return out;
  const step = (Math.PI * 2) / keywords.length;
  for (let i = 0; i < keywords.length; i++) {
    const theta = -Math.PI / 2 + i * step;
    out.set(keywords[i], {
      x: cx + Math.cos(theta) * radius,
      y: cy + Math.sin(theta) * radius,
    });
  }
  return out;
}
