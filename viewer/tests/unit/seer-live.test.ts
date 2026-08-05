/** LiveModel — the leading edge, and the guarantee that it is only that.
 *
 *  The property this file exists for is the first one: **events never touch a
 *  figure**. `SeerPage` refuses to fold in TS because a second implementation
 *  of the Python reducer would drift from it invisibly, and the live view gets
 *  no exemption just because it has to animate. Every other test here is a
 *  smaller consequence of the same rule.
 */

import { describe, expect, it } from "vitest";
import { LiveModel } from "../../src/seer/live";
import type { Mark } from "../../src/seer/live";
import type { RunView } from "../../src/seer/client";
import type { SeerEvent } from "../../src/seer/contract";

const RUN = "run_test";

function ev(over: Partial<SeerEvent> & { event_type: string }): SeerEvent {
  return {
    schema_version: "1.0",
    event_id: over.event_id ?? `e${Math.random()}`,
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

/** A plausible snapshot. Only the fields the assertions read matter; the model
 *  never looks inside it at all, which is the point. */
function view(over: Partial<RunView> = {}): RunView {
  return {
    run_id: RUN,
    agent: "codex",
    state: "model_running",
    n_events: 3,
    n_turns: 1,
    action_counts: { edit: 2 },
    time_in_state: { model_running: 4.5 },
    spans: [],
    usage: {},
    ...over,
  } as RunView;
}

// ── the rule ─────────────────────────────────────────────────────────────────

describe("events never touch a figure", () => {
  it("hands back the adopted view untouched after a thousand events", () => {
    const m = new LiveModel();
    const snapshot = view();
    const before = JSON.parse(JSON.stringify(snapshot));
    m.adopt(RUN, snapshot);

    for (let i = 0; i < 1000; i++) {
      m.ingest(
        ev({
          event_id: `e${i}`,
          event_type: i % 3 === 0 ? "tool.started" : "tool.completed",
          span_id: `sp${i}`,
          action: "edit",
          effect: "state_changed",
          ts: 100 + i * 0.01,
        }),
      );
    }

    // Not merely equal — the same object, unmutated. A model that recomputed
    // `n_events` from the stream would pass a looser assertion.
    expect(m.figures(RUN)).toBe(snapshot);
    expect(m.figures(RUN)).toEqual(before);
  });

  it("has no figures at all before a view is adopted", () => {
    const m = new LiveModel();
    m.ingest(ev({ event_type: "tool.started", span_id: "sp1", action: "edit" }));
    // A run whose events we have seen but whose view we have not fetched has
    // no numbers we are entitled to state.
    expect(m.figures(RUN)).toBeNull();
    expect(m.marks(RUN)).toHaveLength(1);
  });

  it("replaces the snapshot on re-adopt rather than merging into it", () => {
    const m = new LiveModel();
    m.adopt(RUN, view({ n_events: 3 }));
    const second = view({ n_events: 90 });
    m.adopt(RUN, second);
    expect(m.figures(RUN)).toBe(second);
  });
});

// ── deltas ───────────────────────────────────────────────────────────────────

describe("deltas animate and never count", () => {
  it("produces no mark for a delta", () => {
    const m = new LiveModel();
    m.ingest(ev({ event_type: "tool.started", span_id: "sp1", action: "execute", ts: 10 }));
    for (let i = 0; i < 50; i++) {
      m.ingest(ev({ event_id: `d${i}`, event_type: "tool.output_delta", span_id: "sp1", ts: 11 }));
    }
    // 50 deltas, one mark. Nothing downstream can tally them by accident,
    // which matters because `payload.chars` means a fragment length to one
    // adapter and a cumulative length to another.
    expect(m.marks(RUN)).toHaveLength(1);
  });

  it("moves the span's activity pulse to the newest delta", () => {
    const m = new LiveModel();
    m.ingest(ev({ event_type: "tool.started", span_id: "sp1", action: "execute", ts: 10 }));
    expect(m.openSpans(RUN)[0]!.producingUntil).toBeNull();

    m.ingest(ev({ event_type: "tool.output_delta", span_id: "sp1", ts: 12 }));
    m.ingest(ev({ event_type: "tool.output_delta", span_id: "sp1", ts: 11 }));
    // A timestamp, not a count, and it only ever moves forward.
    expect(m.openSpans(RUN)[0]!.producingUntil).toBe(12);
  });

  it("ignores a delta for a span it never saw open", () => {
    const m = new LiveModel();
    m.ingest(ev({ event_type: "tool.output_delta", span_id: "ghost", ts: 12 }));
    expect(m.openSpans(RUN)).toHaveLength(0);
    expect(m.marks(RUN)).toHaveLength(0);
  });
});

// ── spans ────────────────────────────────────────────────────────────────────

describe("open spans", () => {
  it("opens on tool.started and closes on completion or failure", () => {
    const m = new LiveModel();
    m.ingest(ev({ event_type: "tool.started", span_id: "a", action: "execute", ts: 1 }));
    m.ingest(ev({ event_type: "tool.started", span_id: "b", action: "edit", ts: 2 }));
    expect(m.openSpans(RUN).map((s) => s.spanId)).toEqual(["a", "b"]);

    m.ingest(ev({ event_type: "tool.completed", span_id: "a", ts: 3 }));
    m.ingest(ev({ event_type: "tool.failed", span_id: "b", ts: 4 }));
    expect(m.openSpans(RUN)).toHaveLength(0);
  });

  it("does not treat run.started or session.started as spans", () => {
    // They end in the same word as `tool.started` and are not spans; matching
    // on the suffix would leave a bar growing for the life of the run.
    const m = new LiveModel();
    m.ingest(ev({ event_type: "run.started", span_id: "r", ts: 1 }));
    m.ingest(ev({ event_type: "session.started", span_id: "s", ts: 1 }));
    expect(m.openSpans(RUN)).toHaveLength(0);
  });

  it("keeps the first beginning when a span opens twice", () => {
    const m = new LiveModel();
    m.ingest(ev({ event_type: "tool.started", span_id: "a", ts: 5 }));
    m.ingest(ev({ event_id: "x2", event_type: "tool.started", span_id: "a", ts: 9 }));
    expect(m.openSpans(RUN)[0]!.startedAt).toBe(5);
  });

  it("carries the reasoning flag from our own payload, not from native_type", () => {
    const m = new LiveModel();
    m.ingest(
      ev({
        event_type: "tool.started",
        span_id: "th",
        ts: 1,
        payload: { kind: "reasoning" },
        native_type: "item.reasoning",
      }),
    );
    expect(m.openSpans(RUN)[0]!.reasoning).toBe(true);
    expect(m.marks(RUN)[0]!.reasoning).toBe(true);
  });

  it("closes everything when the run finishes but keeps the marks", () => {
    const m = new LiveModel();
    m.ingest(ev({ event_type: "tool.started", span_id: "a", ts: 1 }));
    m.finish(RUN);
    expect(m.openSpans(RUN)).toHaveLength(0);
    // A run that ends two seconds after you looked at it should not have its
    // last two seconds blink out of the picture.
    expect(m.marks(RUN)).toHaveLength(1);
  });

  it("closes a span even when the close arrives outside the mark window", () => {
    const m = new LiveModel({ windowS: 10 });
    m.ingest(ev({ event_type: "tool.started", span_id: "a", ts: 1 }));
    m.ingest(ev({ event_type: "tool.started", span_id: "b", ts: 100 }));
    m.ingest(ev({ event_type: "tool.completed", span_id: "a", ts: 2 }));
    expect(m.openSpans(RUN).map((s) => s.spanId)).toEqual(["b"]);
  });
});

// ── the stream ───────────────────────────────────────────────────────────────

describe("marks", () => {
  it("projects fields straight off the event", () => {
    const m = new LiveModel();
    m.ingest(
      ev({
        event_id: "one",
        event_type: "tool.completed",
        span_id: "sp",
        action: "verify",
        effect: "new_information",
        ts: 42,
        source: { fidelity: "estimated" } as SeerEvent["source"],
      }),
    );
    const got = m.marks(RUN)[0] as Mark;
    expect(got).toMatchObject({
      eventId: "one",
      ts: 42,
      action: "verify",
      effect: "new_information",
      fidelity: "estimated",
      spanId: "sp",
    });
  });

  it("drops a replayed event rather than drawing it twice", () => {
    const m = new LiveModel();
    const e = ev({ event_id: "dup", event_type: "tool.completed", ts: 1 });
    m.ingest(e);
    m.ingest(e);
    // A reconnect replays. Two particles for one action would misrepresent how
    // busy the run was.
    expect(m.marks(RUN)).toHaveLength(1);
  });

  it("trims to the window and reports what it dropped", () => {
    const m = new LiveModel({ windowS: 10 });
    for (let i = 0; i < 30; i++) {
      m.ingest(ev({ event_id: `e${i}`, event_type: "tool.completed", ts: i }));
    }
    const kept = m.marks(RUN);
    expect(kept[0]!.ts).toBeGreaterThanOrEqual(19);
    expect(m.droppedMarks(RUN)).toBe(30 - kept.length);
    // A view that bounds what it shows has to be able to say so.
    expect(m.droppedMarks(RUN)).toBeGreaterThan(0);
  });

  it("measures the window from the newest mark, not the wall clock", () => {
    // A reconciled import lands with month-old timestamps all at once.
    // Trimming that against `now` would throw away everything it just received.
    const old = 1_700_000_000;
    const m = new LiveModel({ windowS: 60 });
    for (let i = 0; i < 5; i++) {
      m.ingest(ev({ event_id: `r${i}`, event_type: "tool.completed", ts: old + i }));
    }
    expect(m.marks(RUN)).toHaveLength(5);
    expect(m.droppedMarks(RUN)).toBe(0);
  });

  it("honours the mark ceiling under a burst", () => {
    const m = new LiveModel({ windowS: 10_000, maxMarks: 50 });
    for (let i = 0; i < 500; i++) {
      m.ingest(ev({ event_id: `b${i}`, event_type: "tool.completed", ts: 1 + i * 0.001 }));
    }
    expect(m.marks(RUN)).toHaveLength(50);
    expect(m.droppedMarks(RUN)).toBe(450);
  });

  it("tracks the newest event timestamp without accumulating anything", () => {
    const m = new LiveModel();
    m.ingest(ev({ event_type: "tool.completed", ts: 10 }));
    m.ingest(ev({ event_type: "tool.completed", ts: 30 }));
    m.ingest(ev({ event_type: "tool.completed", ts: 20 }));
    expect(m.lastEventAt(RUN)).toBe(30);
    expect(m.lastEventAt("nobody")).toBeNull();
  });
});

// ── lifecycle ────────────────────────────────────────────────────────────────

describe("forget", () => {
  it("drops every trace of a deleted run", () => {
    const m = new LiveModel();
    m.adopt(RUN, view());
    m.ingest(ev({ event_type: "tool.started", span_id: "a", ts: 1 }));
    m.forget(RUN);
    expect(m.figures(RUN)).toBeNull();
    expect(m.marks(RUN)).toHaveLength(0);
    expect(m.openSpans(RUN)).toHaveLength(0);
    expect(m.runIds()).not.toContain(RUN);
  });

  it("keeps runs apart", () => {
    const m = new LiveModel();
    m.ingest(ev({ event_type: "tool.started", span_id: "a", ts: 1 }));
    m.ingest(ev({ run_id: "other", event_type: "tool.started", span_id: "a", ts: 1 }));
    expect(m.runIds()).toEqual([RUN, "other"]);
    m.forget(RUN);
    expect(m.openSpans("other")).toHaveLength(1);
  });
});
