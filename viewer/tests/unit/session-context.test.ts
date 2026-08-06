/** The context-window decomposition — requested feature #4.
 *
 *  Two things are worth locking here, and neither is the arithmetic.
 *
 *  The first is what the decomposition REFUSES. The feature asked for a
 *  system / user / output / retrieved split; a transcript meters tokens per
 *  request, not per content block, so no such split is available. The
 *  attractive substitute — subtract the model's last output from the window's
 *  growth and call the difference "what the tools contributed" — is wrong, and
 *  wrong in a way that only shows up on real data: on a 731-turn transcript it
 *  sums to −351k tokens, because compactions dominate it and because thinking
 *  is billed as output and then never re-fed. These tests hold the boundary:
 *  `growth` and `priorOutput` are two measurements, and `residual` is named as
 *  a residual so nothing downstream can quietly promote it to an attribution.
 *
 *  The second is the three-state discipline the rest of this codebase already
 *  runs on: a quantity that cannot be known is `null`, never 0. The first turn
 *  has no growth because there is nothing to difference it against, and a 0
 *  there would read as "the window did not grow", which is a different claim.
 */

import { describe, expect, it } from "vitest";
import {
  buildComposition,
  parseSessionTranscript,
  type SessionTurn,
} from "../../src/chrome/sessionlog";

/** A turn carrying only the fields the composition reads. */
function turn(p: Partial<SessionTurn> & { index: number }): SessionTurn {
  return {
    requestId: `r${p.index}`,
    tMs: null,
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

describe("buildComposition", () => {
  it("splits each prompt into three disjoint parts that sum to it", () => {
    const c = buildComposition(
      [turn({ index: 0, cacheRead: 8000, cacheWrite: 1200, inputTokens: 43 })],
      true,
    );
    const s = c.slices[0]!;
    expect(s.reused).toBe(8000);
    expect(s.written).toBe(1200);
    expect(s.fresh).toBe(43);
    expect(s.prompt).toBe(9243);
    expect(s.reused + s.written + s.fresh).toBe(s.prompt);
  });

  it("leaves the first turn's growth null rather than calling it zero", () => {
    const c = buildComposition([turn({ index: 0, cacheRead: 5000, outputTokens: 700 })], true);
    const s = c.slices[0]!;
    expect(s.growth).toBeNull();
    expect(s.priorOutput).toBeNull();
    expect(s.residual).toBeNull();
  });

  it("reports growth and the model's last output as two separate measurements", () => {
    const c = buildComposition(
      [
        turn({ index: 0, cacheRead: 5000, outputTokens: 700 }),
        turn({ index: 1, cacheRead: 6000, outputTokens: 400 }),
      ],
      true,
    );
    const s = c.slices[1]!;
    expect(s.growth).toBe(1000); // the window
    expect(s.priorOutput).toBe(700); // what the model wrote just before it
    expect(s.residual).toBe(300);
  });

  it("lets the residual go negative when the model outwrote the window's growth", () => {
    // Real and common: thinking tokens are billed as output and then dropped
    // from the next prompt, so the window can grow by less than the model just
    // wrote. Measured on a real 731-turn transcript: 8 such steps. Clamping
    // this to zero would hide the one thing the residual is good for.
    const c = buildComposition(
      [
        turn({ index: 0, cacheRead: 70000, outputTokens: 3993 }),
        turn({ index: 1, cacheRead: 72296 }),
      ],
      true,
    );
    expect(c.slices[1]!.growth).toBe(2296);
    expect(c.slices[1]!.priorOutput).toBe(3993);
    expect(c.slices[1]!.residual).toBe(-1697);
  });

  it("names no field after a source it cannot measure", () => {
    // The refusal, stated as a test. There is no `fromTools`, no `fromUser`,
    // no `system`, no `retrieved` — a transcript does not meter content blocks,
    // and `residual` mixes at least two effects that cannot be separated from
    // it. A later change that adds such a field has to add a real source for
    // it, not a ratio.
    const c = buildComposition(
      [turn({ index: 0, cacheRead: 100 }), turn({ index: 1, cacheRead: 900 })],
      true,
    );
    expect(Object.keys(c.slices[1]!).filter((k) => /user|tool|system|retriev/i.test(k))).toEqual([]);
  });

  it("reports a shrinking window as compaction instead of clamping it", () => {
    const c = buildComposition(
      [
        turn({ index: 0, cacheRead: 90000, outputTokens: 500 }),
        turn({ index: 1, cacheRead: 12000, outputTokens: 500 }),
        turn({ index: 2, cacheRead: 14000 }),
      ],
      true,
    );
    expect(c.slices[1]!.growth).toBe(-78000);
    expect(c.compactedAt).toEqual([1]);
    // and the turn after a compaction still differences against the NEW window
    expect(c.slices[2]!.growth).toBe(2000);
  });

  it("drops sub-agent turns and says how many", () => {
    // A sub-agent runs in its own window. Differencing across it would measure
    // the gap between two unrelated conversations, which is not growth.
    const c = buildComposition(
      [
        turn({ index: 0, cacheRead: 5000, outputTokens: 100 }),
        turn({ index: 1, cacheRead: 300, isSidechain: true, agentId: "toolu_1" }),
        turn({ index: 2, cacheRead: 5400 }),
      ],
      true,
    );
    expect(c.slices.map((s) => s.turn)).toEqual([0, 2]);
    expect(c.excludedSidechain).toBe(1);
    // 5400 − 5000, NOT 5400 − 300
    expect(c.slices[1]!.growth).toBe(400);
    expect(c.slices[1]!.priorOutput).toBe(100);
  });

  it("takes the peak over prompts, not over cache reads", () => {
    const c = buildComposition(
      [
        turn({ index: 0, cacheRead: 9000, cacheWrite: 0, inputTokens: 10 }),
        turn({ index: 1, cacheRead: 8000, cacheWrite: 2000, inputTokens: 10 }),
      ],
      true,
    );
    expect(c.peakPrompt).toBe(10010); // turn 1, whose cacheRead is the smaller
  });

  it("survives a session with no turns", () => {
    const c = buildComposition([], true);
    expect(c.slices).toEqual([]);
    expect(c.peakPrompt).toBe(0);
    expect(c.compactedAt).toEqual([]);
    expect(c.excludedSidechain).toBe(0);
  });

  it("carries the exactness flag through", () => {
    expect(buildComposition([turn({ index: 0 })], false).exact).toBe(false);
    expect(buildComposition([turn({ index: 0 })], true).exact).toBe(true);
  });
});

/* ---- the flag, end to end ---------------------------------------------- */

let seq = 0;
const stamp = () => new Date(Date.UTC(2026, 0, 1, 0, 0, seq++)).toISOString();

/** One assistant line. Several lines sharing an `id` are ONE response. */
function line(id: string, usage: Record<string, number>): string {
  return JSON.stringify({
    type: "assistant",
    uuid: `u-${id}-${seq}`,
    timestamp: stamp(),
    message: { id, model: "claude-fable-5", usage, content: [{ type: "text", text: "hi" }] },
  });
}

describe("composition exactness", () => {
  it("is exact when every line of a response reports the same prompt", () => {
    // The transcript format repeats identical usage per content block, so the
    // per-field max IS that response's usage and the parts sum to its prompt.
    const u = { input_tokens: 12, cache_read_input_tokens: 4000, cache_creation_input_tokens: 90, output_tokens: 30 };
    const a = parseSessionTranscript([line("m1", u), line("m1", u)].join("\n"), "s");
    expect(a.context!.exact).toBe(true);
    expect(a.context!.slices[0]!.prompt).toBe(4102);
  });

  it("is not exact when one response reports two different prompts", () => {
    // The audit format streams partial snapshots; the max then draws its fields
    // from different lines and their sum is an upper bound on the prompt, not
    // any single request's prompt. The chart has to say so.
    const a = parseSessionTranscript(
      [
        line("m1", { input_tokens: 12, cache_read_input_tokens: 0, cache_creation_input_tokens: 4000 }),
        line("m1", { input_tokens: 12, cache_read_input_tokens: 4000, cache_creation_input_tokens: 0 }),
      ].join("\n"),
      "s",
    );
    expect(a.context!.exact).toBe(false);
  });

  it("does not blame output_tokens for the inexactness", () => {
    // output_tokens varies line to line BY DESIGN — the first chunk, then the
    // total. That says nothing about whether the prompt was reported
    // consistently, and treating it as disagreement would mark every audit
    // session inexact for a reason that isn't one.
    const a = parseSessionTranscript(
      [
        line("m1", { input_tokens: 12, cache_read_input_tokens: 4000, output_tokens: 5 }),
        line("m1", { input_tokens: 12, cache_read_input_tokens: 4000, output_tokens: 800 }),
      ].join("\n"),
      "s",
    );
    expect(a.context!.exact).toBe(true);
    expect(a.context!.slices[0]!.output).toBe(800);
  });
});
