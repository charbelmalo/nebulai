/** Thoughts — folded by identity, never by addition.
 *
 *  A reasoning fragment is not a delta, but it carries the same trap: Codex
 *  re-emits one `item.reasoning` as its text grows, so five events describe one
 *  thought and summing `chars` across them would triple it. Claude emits a
 *  single completed event per thinking block, so the same sum would be correct
 *  there — which is exactly the shape of bug that is invisible until you
 *  compare two agents.
 *
 *  The other property under test is the split between the two honest states.
 *  `native` means the run was captured with `--keep-reasoning` and we have the
 *  words; `dropped_by_policy` means the agent told us and we declined. Neither
 *  may collapse into the other, and neither may collapse into "nothing here".
 */

import { describe, expect, it } from "vitest";
import { LiveModel } from "../../src/seer/live";
import type { SeerEvent } from "../../src/seer/contract";

const RUN = "run_test";

let seq = 0;

function ev(over: Partial<SeerEvent> & { event_type: string }): SeerEvent {
  return {
    schema_version: "1.0",
    event_id: over.event_id ?? `e${seq++}`,
    ts: 100,
    mono_ns: 0,
    source: {
      agent: "codex",
      agent_version: "0.144.6",
      adapter: "codex",
      adapter_version: "1",
      capture_mode: "driven",
      fidelity: "native",
      ...(over.source ?? {}),
    },
    run_id: RUN,
    session_id: "s1",
    payload: {},
    privacy: {},
    ...over,
  } as SeerEvent;
}

/** A reasoning fragment as `codex.py` emits it: spanned, and re-sent as the
 *  item grows. `done` picks the completing event type. */
function codexThought(
  over: { ts: number; span: string; done?: boolean; text?: string; chars?: number },
): SeerEvent {
  const kept = over.text != null;
  return ev({
    event_type: over.done ? "model.request_completed" : "model.request_started",
    ts: over.ts,
    span_id: over.span,
    native_type: "item.reasoning",
    source: {
      agent: "codex",
      agent_version: "0.144.6",
      adapter: "codex",
      adapter_version: "1",
      capture_mode: "driven",
      fidelity: kept ? "native" : "dropped_by_policy",
    },
    payload: kept
      ? { kind: "reasoning", text: over.text }
      : { kind: "reasoning", chars: over.chars ?? 0, text_retained: false },
  } as Partial<SeerEvent> & { event_type: string });
}

/** A thinking block as `claude.py` emits it: unspanned, and only ever the
 *  completed event — the block is reported once it is finished. */
function claudeThought(over: { ts: number; id: string; text?: string; chars?: number }): SeerEvent {
  const kept = over.text != null;
  return ev({
    event_type: "model.request_completed",
    event_id: over.id,
    ts: over.ts,
    native_type: "assistant.thinking",
    source: {
      agent: "claude",
      agent_version: "2.0",
      adapter: "claude",
      adapter_version: "1",
      capture_mode: "driven",
      fidelity: kept ? "native" : "dropped_by_policy",
    },
    payload: kept
      ? { kind: "reasoning", text: over.text }
      : { kind: "reasoning", chars: over.chars ?? 0, text_retained: false },
  } as Partial<SeerEvent> & { event_type: string });
}

// ── the fold ─────────────────────────────────────────────────────────────────

describe("one stream is one thought", () => {
  it("collapses a codex stream's events into a single thought", () => {
    const m = new LiveModel();
    m.ingest(codexThought({ ts: 10, span: "sp1", text: "let me" }));
    m.ingest(codexThought({ ts: 11, span: "sp1", text: "let me check the" }));
    m.ingest(codexThought({ ts: 13, span: "sp1", done: true, text: "let me check the reducer" }));

    const ts = m.thoughts(RUN);
    expect(ts).toHaveLength(1);
    expect(ts[0]!.text).toBe("let me check the reducer");
    expect(ts[0]!.startedAt).toBe(10);
    expect(ts[0]!.endedAt).toBe(13);
    expect(ts[0]!.observedStart).toBe(true);
  });

  it("never sums chars across the events of one stream", () => {
    const m = new LiveModel();
    m.ingest(codexThought({ ts: 10, span: "sp1", chars: 40 }));
    m.ingest(codexThought({ ts: 11, span: "sp1", chars: 190 }));
    m.ingest(codexThought({ ts: 12, span: "sp1", done: true, chars: 512 }));

    // 40 + 190 + 512 = 742, which is the number a delta-shaped bug would print.
    expect(m.thoughts(RUN)[0]!.chars).toBe(512);
  });

  it("keeps two spans apart", () => {
    const m = new LiveModel();
    m.ingest(codexThought({ ts: 10, span: "sp1", done: true, chars: 40 }));
    m.ingest(codexThought({ ts: 20, span: "sp2", done: true, chars: 60 }));
    expect(m.thoughts(RUN)).toHaveLength(2);
  });

  it("gives an unspanned thought its own identity per event", () => {
    // Claude's thinking blocks carry no span. Keying them all under one bucket
    // would fold a whole session's thinking into a single row.
    const m = new LiveModel();
    m.ingest(claudeThought({ ts: 10, id: "a", chars: 100 }));
    m.ingest(claudeThought({ ts: 20, id: "b", chars: 200 }));
    expect(m.thoughts(RUN)).toHaveLength(2);
  });

  it("ignores everything that is not a reasoning fragment", () => {
    const m = new LiveModel();
    m.ingest(ev({ event_type: "tool.started", span_id: "s", ts: 10 }));
    m.ingest(ev({ event_type: "model.request_completed", ts: 11, payload: { chars: 12 } }));
    m.ingest(ev({ event_type: "message.assistant_completed", ts: 12, payload: { text: "hi" } }));
    expect(m.thoughts(RUN)).toHaveLength(0);
  });
});

// ── the two states ───────────────────────────────────────────────────────────

describe("kept and dropped are different facts", () => {
  it("carries the words when the run was captured with --keep-reasoning", () => {
    const m = new LiveModel();
    m.ingest(codexThought({ ts: 10, span: "sp1", done: true, text: "the reducer owns the fold" }));
    const t = m.thoughts(RUN)[0]!;
    expect(t.fidelity).toBe("native");
    expect(t.text).toBe("the reducer owns the fold");
    // No second number: `text.length` is the honest size, and two figures for
    // one thought could only disagree.
    expect(t.chars).toBeNull();
  });

  it("carries the size, and no text, when policy dropped it", () => {
    const m = new LiveModel();
    m.ingest(codexThought({ ts: 10, span: "sp1", done: true, chars: 1284 }));
    const t = m.thoughts(RUN)[0]!;
    expect(t.fidelity).toBe("dropped_by_policy");
    expect(t.text).toBeNull();
    expect(t.chars).toBe(1284);
  });

  it("distinguishes a kept-but-empty thought from a dropped one", () => {
    // `""` means the agent emitted a thought with no words. `null` means we
    // never had them. A rail that printed both as blank would erase the
    // difference between the agent's silence and our policy.
    const m = new LiveModel();
    m.ingest(codexThought({ ts: 10, span: "sp1", done: true, text: "" }));
    expect(m.thoughts(RUN)[0]!.text).toBe("");
    expect(m.thoughts(RUN)[0]!.fidelity).toBe("native");
  });
});

// ── what may be called a duration ────────────────────────────────────────────

describe("an interval needs two observed ends", () => {
  it("reports no observed start for a thought that only ever completed", () => {
    const m = new LiveModel();
    m.ingest(claudeThought({ ts: 42, id: "a", chars: 90 }));
    const t = m.thoughts(RUN)[0]!;
    expect(t.observedStart).toBe(false);
    // Start and end are the same instant, which is not a duration — the rail
    // reads `observedStart` rather than differencing these two.
    expect(t.startedAt).toBe(42);
    expect(t.endedAt).toBe(42);
  });

  it("leaves an unfinished stream open rather than stamping an end", () => {
    const m = new LiveModel();
    m.ingest(codexThought({ ts: 10, span: "sp1", text: "still going" }));
    expect(m.thoughts(RUN)[0]!.endedAt).toBeNull();
  });

  it("does not close a thought when the run finishes", () => {
    // `finish` clears open spans because a finished run has none. A thought is
    // different: we never saw its end, and inventing one would turn a gap in
    // the capture into a measurement.
    const m = new LiveModel();
    m.ingest(codexThought({ ts: 10, span: "sp1", text: "still going" }));
    m.finish(RUN);
    expect(m.thoughts(RUN)[0]!.endedAt).toBeNull();
  });

  it("closes on model.request_failed too", () => {
    const m = new LiveModel();
    m.ingest(codexThought({ ts: 10, span: "sp1", text: "…" }));
    m.ingest(
      ev({
        event_type: "model.request_failed",
        ts: 14,
        span_id: "sp1",
        payload: { kind: "reasoning", text: "…" },
      }),
    );
    expect(m.thoughts(RUN)[0]!.endedAt).toBe(14);
  });
});

// ── backfill ─────────────────────────────────────────────────────────────────

describe("reading the log back", () => {
  it("folds thoughts out of a history without touching the leading edge", () => {
    const m = new LiveModel();
    const before = m.marks(RUN).length;
    const n = m.ingestHistory(RUN, [
      ev({ event_type: "tool.started", ts: 1, span_id: "t1" }),
      codexThought({ ts: 2, span: "sp1", done: true, chars: 300 }),
      ev({ event_type: "tool.completed", ts: 3, span_id: "t1" }),
    ]);
    expect(n).toBe(1);
    expect(m.thoughts(RUN)).toHaveLength(1);
    // History is not the leading edge: it must not appear as marks, and it must
    // not inflate `droppedMarks`, which means something else entirely.
    expect(m.marks(RUN).length).toBe(before);
    expect(m.droppedMarks(RUN)).toBe(0);
    expect(m.openSpans(RUN)).toHaveLength(0);
  });

  it("refuses a second read of the same run", () => {
    const m = new LiveModel();
    expect(m.isBackfilled(RUN)).toBe(false);
    m.ingestHistory(RUN, [codexThought({ ts: 2, span: "sp1", done: true, chars: 300 })]);
    expect(m.isBackfilled(RUN)).toBe(true);
    expect(m.ingestHistory(RUN, [codexThought({ ts: 9, span: "sp2", done: true, chars: 1 })])).toBe(
      -1,
    );
    expect(m.thoughts(RUN)).toHaveLength(1);
  });

  it("cannot walk a live thought backwards", () => {
    // The stream showed the finished text; the backfill then hands us the same
    // stream's earlier, shorter events. Last-writer-wins by timestamp is what
    // keeps the rail from flickering back to a partial thought.
    const m = new LiveModel();
    m.ingest(codexThought({ ts: 13, span: "sp1", done: true, text: "the whole thought" }));
    m.ingestHistory(RUN, [
      codexThought({ ts: 10, span: "sp1", text: "the" }),
      codexThought({ ts: 11, span: "sp1", text: "the whole" }),
    ]);
    const t = m.thoughts(RUN)[0]!;
    expect(t.text).toBe("the whole thought");
    expect(t.endedAt).toBe(13);
    // The beginning, however, is new information and does move.
    expect(t.startedAt).toBe(10);
    expect(t.observedStart).toBe(true);
  });

  it("is idempotent when a backfilled event also arrives live", () => {
    const m = new LiveModel();
    const e = codexThought({ ts: 10, span: "sp1", done: true, chars: 512 });
    m.ingestHistory(RUN, [e]);
    m.ingest(e);
    expect(m.thoughts(RUN)).toHaveLength(1);
    expect(m.thoughts(RUN)[0]!.chars).toBe(512);
  });
});

// ── bounds ───────────────────────────────────────────────────────────────────

describe("the rail is bounded, and says so", () => {
  it("drops the oldest thoughts past the ceiling and counts them", () => {
    const m = new LiveModel({ maxThoughts: 3 });
    for (let i = 0; i < 5; i++) {
      m.ingest(codexThought({ ts: 10 + i, span: `sp${i}`, done: true, chars: i }));
    }
    const ts = m.thoughts(RUN);
    expect(ts).toHaveLength(3);
    expect(ts[0]!.startedAt).toBe(12);
    expect(m.droppedThoughts(RUN)).toBe(2);
  });

  it("does not evict a thought just because it left the mark window", () => {
    // Thoughts are bounded by count, not by the 120s window: a rail that only
    // showed the last two minutes of thinking would be empty for most of a run.
    const m = new LiveModel({ windowS: 5 });
    m.ingest(codexThought({ ts: 10, span: "sp1", done: true, chars: 100 }));
    for (let i = 0; i < 30; i++) {
      m.ingest(ev({ event_type: "tool.started", ts: 100 + i, span_id: `t${i}` }));
    }
    expect(m.thoughts(RUN)).toHaveLength(1);
    expect(m.droppedThoughts(RUN)).toBe(0);
  });

  it("forgets a run's thoughts with the run", () => {
    const m = new LiveModel();
    m.ingest(codexThought({ ts: 10, span: "sp1", done: true, chars: 1 }));
    m.forget(RUN);
    expect(m.thoughts(RUN)).toHaveLength(0);
    expect(m.isBackfilled(RUN)).toBe(false);
  });
});
