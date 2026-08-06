/** The agent path — requested feature #2.
 *
 *  The audit asked for "a node→edge trace with numbered steps and per-step
 *  timing" in place of a cloud. The trace and the numbers were the easy part.
 *  Two things here are load-bearing and are what these tests hold:
 *
 *  1. **A step is numbered and timed within its own agent.** A sub-agent runs
 *     in its own context; its third step is its third step regardless of what
 *     the session around it was doing. Numbering by session position would make
 *     an interrupted parent's steps skip, and differencing wall-clock across an
 *     agent boundary would time the gap between two unrelated conversations.
 *
 *  2. **An unmeasurable gap is null, never 0.** `tSec` falls back to 0 when a
 *     line carries no timestamp, so it cannot be differenced safely; `tMs` is
 *     null in exactly that case and is what the gap reads. A 0 would assert
 *     that two responses landed simultaneously.
 *
 *  Provenance note worth keeping with the code: the multi-agent behaviour below
 *  is unexercised by any real recording on this machine. A sweep of all 239
 *  transcripts under `~/.claude/projects/…/nebulai` found 0 of 10,391 turns
 *  with `isSidechain` true and 0 with a non-null `parent_tool_use_id`. These
 *  fixtures are synthetic on purpose, and the doc says so.
 */

import { describe, expect, it } from "vitest";
import { buildAgentGraph, type SessionTurn } from "../../src/chrome/sessionlog";

/** A turn carrying only the fields the agent graph reads. */
function turn(p: Partial<SessionTurn> & { index: number }): SessionTurn {
  return {
    requestId: `r${p.index}`,
    tMs: 1_000_000 + p.index * 1000,
    tSec: p.index,
    isSidechain: false,
    agentId: "main",
    model: "claude-fable-5",
    inputTokens: 0,
    outputTokens: 0,
    cacheRead: 0,
    cacheWrite: 0,
    iterations: null,
    tools: [],
    files: [],
    thinkingBlocks: 0,
    textLen: 0,
    category: "reflect",
    cumOutput: 0,
    ...p,
  };
}

describe("buildAgentGraph", () => {
  it("numbers a single-agent session 1..n in transcript order", () => {
    const g = buildAgentGraph([turn({ index: 0 }), turn({ index: 1 }), turn({ index: 2 })]);
    expect(g.paths).toHaveLength(1);
    expect(g.paths[0]!.agentId).toBe("main");
    expect(g.paths[0]!.steps.map((s) => s.step)).toEqual([1, 2, 3]);
    expect(g.paths[0]!.steps.every((s) => s.ofSteps === 3)).toBe(true);
  });

  it("leaves an agent's first gap null rather than calling it zero", () => {
    const g = buildAgentGraph([turn({ index: 0, tMs: 5000 }), turn({ index: 1, tMs: 8000 })]);
    expect(g.byTurn.get(0)!.gapSec).toBeNull();
    expect(g.byTurn.get(1)!.gapSec).toBe(3);
  });

  it("times gaps from tMs, not from the tSec that falls back to zero", () => {
    // A line with no timestamp gets tSec 0. Differencing that against a real
    // tSec would manufacture a gap the size of the whole session so far.
    const g = buildAgentGraph([
      turn({ index: 0, tMs: 1000, tSec: 1 }),
      turn({ index: 1, tMs: null, tSec: 0 }),
      turn({ index: 2, tMs: 9000, tSec: 9 }),
    ]);
    expect(g.byTurn.get(1)!.gapSec).toBeNull(); // into the untimed turn
    expect(g.byTurn.get(2)!.gapSec).toBeNull(); // and out of it
    // One missing clock in the middle blanks BOTH adjacent gaps, so nothing in
    // this session is timed even though two of its three turns carry a stamp.
    // `timed` reports whether any GAP was measured, which is the question the
    // dashes raise — not whether any turn had a timestamp.
    expect(g.timed).toBe(false);
  });

  it("is timed as soon as one gap is measurable", () => {
    const g = buildAgentGraph([
      turn({ index: 0, tMs: 1000 }),
      turn({ index: 1, tMs: 3000 }),
      turn({ index: 2, tMs: null }),
    ]);
    expect(g.timed).toBe(true);
    expect(g.byTurn.get(1)!.gapSec).toBe(2);
  });

  it("says the session was never timed when no turn carries a clock", () => {
    const g = buildAgentGraph([turn({ index: 0, tMs: null }), turn({ index: 1, tMs: null })]);
    expect(g.timed).toBe(false);
    expect(g.paths[0]!.spanSec).toBeNull();
  });

  it("gives each agent its own numbering, not the session's", () => {
    // main, main, SUB, SUB, main — the parent's third step is its third step.
    const g = buildAgentGraph([
      turn({ index: 0 }),
      turn({ index: 1 }),
      turn({ index: 2, agentId: "toolu_a", isSidechain: true }),
      turn({ index: 3, agentId: "toolu_a", isSidechain: true }),
      turn({ index: 4 }),
    ]);
    expect(g.paths.map((p) => p.agentId)).toEqual(["main", "toolu_a"]);
    expect(g.byTurn.get(4)!.step).toBe(3); // NOT 5
    expect(g.byTurn.get(4)!.ofSteps).toBe(3);
    expect(g.byTurn.get(3)!.step).toBe(2); // the sub-agent's own second step
    expect(g.byTurn.get(3)!.ofSteps).toBe(2);
    expect(g.paths[1]!.isSidechain).toBe(true);
  });

  it("closes a parent's path over the sub-agent that interrupted it", () => {
    // The parent's step 2→3 edge spans turns 1→4 in session order. That edge is
    // real: the parent was inside one tool call for the whole interruption, and
    // its gap must cover that time, not be reset by it.
    const g = buildAgentGraph([
      turn({ index: 0, tMs: 0 }),
      turn({ index: 1, tMs: 1000 }),
      turn({ index: 2, tMs: 2000, agentId: "toolu_a", isSidechain: true }),
      turn({ index: 3, tMs: 3000, agentId: "toolu_a", isSidechain: true }),
      turn({ index: 4, tMs: 9000 }),
    ]);
    const main = g.paths[0]!;
    expect(main.steps.map((s) => s.turn)).toEqual([0, 1, 4]);
    expect(g.byTurn.get(4)!.gapSec).toBe(8); // 9000 − 1000, across the sub-agent
    expect(g.byTurn.get(2)!.gapSec).toBeNull(); // the sub-agent's own first step
    expect(g.byTurn.get(3)!.gapSec).toBe(1);
  });

  it("never emits an edge between two different agents", () => {
    // The defect this replaced: one walk over `turns` with a single `prev`
    // emits main→sub and sub→main, asserting a continuity neither agent has.
    const g = buildAgentGraph([
      turn({ index: 0 }),
      turn({ index: 1, agentId: "toolu_a", isSidechain: true }),
      turn({ index: 2 }),
      turn({ index: 3, agentId: "toolu_b", isSidechain: true }),
    ]);
    for (const p of g.paths) {
      for (const s of p.steps) expect(s.agentId).toBe(p.agentId);
    }
    const edges = g.paths.flatMap((p) =>
      p.steps.slice(1).map((s, i) => [p.steps[i]!.agentId, s.agentId] as const),
    );
    expect(edges).toEqual([["main", "main"]]); // one edge, and it is within main
  });

  it("orders paths by first appearance, so the main agent leads", () => {
    const g = buildAgentGraph([
      turn({ index: 0, agentId: "toolu_z", isSidechain: true }),
      turn({ index: 1 }),
      turn({ index: 2, agentId: "toolu_a", isSidechain: true }),
    ]);
    expect(g.paths.map((p) => p.agentId)).toEqual(["toolu_z", "main", "toolu_a"]);
  });

  it("measures each agent's span over its own first and last step", () => {
    const g = buildAgentGraph([
      turn({ index: 0, tMs: 0 }),
      turn({ index: 1, tMs: 4000, agentId: "toolu_a", isSidechain: true }),
      turn({ index: 2, tMs: 7000, agentId: "toolu_a", isSidechain: true }),
      turn({ index: 3, tMs: 20000 }),
    ]);
    expect(g.paths[0]!.spanSec).toBe(20); // main: 0 → 20000
    expect(g.paths[1]!.spanSec).toBe(3); // sub: 4000 → 7000
  });

  it("indexes every turn exactly once", () => {
    const turns = [
      turn({ index: 0 }),
      turn({ index: 1, agentId: "toolu_a", isSidechain: true }),
      turn({ index: 2 }),
    ];
    const g = buildAgentGraph(turns);
    expect(g.byTurn.size).toBe(turns.length);
    for (const t of turns) expect(g.byTurn.get(t.index)!.turn).toBe(t.index);
  });

  it("survives a session with no turns", () => {
    const g = buildAgentGraph([]);
    expect(g.paths).toEqual([]);
    expect(g.byTurn.size).toBe(0);
    expect(g.timed).toBe(false);
  });
});
