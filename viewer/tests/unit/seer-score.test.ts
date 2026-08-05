/** ScoreDriver — the drawing rules that are claims about the data.
 *
 *  Most of what a chart does is taste and cannot be tested. Three things here
 *  are not taste, and this file pins them:
 *
 *  * a span with no clock is never drawn as a bar of any length,
 *  * a span that has not ended is never given a cap,
 *  * an effect the server grew and we have not shipped an encoding for is
 *    drawn as unknown, not as "changed nothing".
 *
 *  The canvas is stubbed rather than mocked wholesale: the driver draws through
 *  a real 2D-context shape and every call is recorded, so the assertions are
 *  about what was actually asked of the canvas.
 */

import { beforeAll, describe, expect, it, vi } from "vitest";
import { ScoreDriver, capFor, hexA, niceStep } from "../../src/scene/seer/ScoreDriver";
import type { ScoreInput } from "../../src/scene/seer/ScoreDriver";
import { EFFECT_CAP } from "../../src/seer/encoding";
import type { SpanRecord } from "../../src/seer/client";
import type { Mark, OpenSpan } from "../../src/seer/live";

// The suite runs in node, with no DOM. Only three globals are needed, and the
// page is reported hidden so the driver's own rAF loop never draws: every test
// calls `draw()` once, explicitly, and nothing races it.
beforeAll(() => {
  vi.stubGlobal("requestAnimationFrame", () => 0);
  vi.stubGlobal("cancelAnimationFrame", () => {});
  vi.stubGlobal("document", { hidden: true });
});

// ── a recording canvas ───────────────────────────────────────────────────────

interface Call {
  op: string;
  args: number[];
  fillStyle: unknown;
  strokeStyle: unknown;
}

function stubCanvas(): { canvas: HTMLCanvasElement; calls: Call[] } {
  const calls: Call[] = [];
  const state = { fillStyle: "", strokeStyle: "", lineWidth: 1, font: "" };
  const rec =
    (op: string) =>
    (...args: number[]) => {
      calls.push({ op, args, fillStyle: state.fillStyle, strokeStyle: state.strokeStyle });
    };
  const ctx = {
    get fillStyle() {
      return state.fillStyle;
    },
    set fillStyle(v: string) {
      state.fillStyle = v;
    },
    get strokeStyle() {
      return state.strokeStyle;
    },
    set strokeStyle(v: string) {
      state.strokeStyle = v;
    },
    lineWidth: 1,
    font: "",
    textAlign: "left",
    textBaseline: "top",
    setTransform: rec("setTransform"),
    clearRect: rec("clearRect"),
    fillRect: rec("fillRect"),
    strokeRect: rec("strokeRect"),
    fillText: vi.fn(),
    beginPath: rec("beginPath"),
    closePath: rec("closePath"),
    moveTo: rec("moveTo"),
    lineTo: rec("lineTo"),
    rect: rec("rect"),
    clip: rec("clip"),
    fill: rec("fill"),
    stroke: rec("stroke"),
    save: rec("save"),
    restore: rec("restore"),
    setLineDash: vi.fn(),
    createLinearGradient: () => ({ addColorStop: vi.fn() }),
  };
  const canvas = {
    width: 0,
    height: 0,
    style: {} as CSSStyleDeclaration,
    getContext: () => ctx,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 240 }),
  } as unknown as HTMLCanvasElement;
  return { canvas, calls };
}

/** Mounts a driver without starting its rAF loop, so a test draws exactly one
 *  frame and nothing races it. */
function mount(input: ScoreInput): { d: ScoreDriver; calls: Call[] } {
  const { canvas, calls } = stubCanvas();
  const d = new ScoreDriver();
  d.init(canvas);
  d.resize(800, 240, 1);
  d.setSource(() => input);
  calls.length = 0;
  return { d, calls };
}

function span(over: Partial<SpanRecord> = {}): SpanRecord {
  return {
    span_id: "sp1",
    action: "edit",
    native_type: null,
    started_at: 1000,
    ended_at: 1002,
    duration_s: 2,
    native_duration_s: null,
    duration_fidelity: "deterministic",
    synthetic_start: false,
    effect: "state_changed",
    failed: false,
    detail: "src/a.ts",
    ...over,
  };
}

function inputWith(spans: SpanRecord[], open: OpenSpan[] = [], marks: Mark[] = []): ScoreInput {
  return {
    view: { run_id: "r", spans } as unknown as ScoreInput["view"],
    openSpans: open,
    marks,
  };
}

/** Every rectangle wide enough to read as a duration. Caps are narrower than
 *  5px; lane backgrounds run the full plot width, which no fixture's bar does.
 *  Both are excluded so the assertions are only ever about bars. */
const LANE_ROW_W = 500;

function bars(calls: Call[]): Call[] {
  return calls.filter(
    (c) => c.op === "fillRect" && c.args[2]! > 5 && c.args[2]! < LANE_ROW_W && c.args[3]! < 30,
  );
}

/** Lane backgrounds, in draw order. The baseline rail is the same width and is
 *  excluded by height — it is a few pixels tall, a lane is not. */
function laneRows(calls: Call[]): Call[] {
  return calls.filter((c) => c.op === "fillRect" && c.args[2]! >= LANE_ROW_W && c.args[3]! > 10);
}

// ── the rules ────────────────────────────────────────────────────────────────

describe("a span with no clock is not drawn as a length", () => {
  it("draws an unclocked span as a hollow diamond, not a bar", () => {
    // Reconciled history: thread files carry no per-item clock, so the span
    // ended but nobody timed it and `duration_s` is 0 by construction.
    const { d, calls } = mount(
      inputWith([
        span({
          started_at: 1000,
          ended_at: 1000,
          duration_s: 0,
          duration_fidelity: "missing",
          synthetic_start: true,
        }),
      ]),
    );
    d.setFollow(false);
    (d as unknown as { edge: number }).edge = 1010;
    d.draw();

    expect(bars(calls)).toHaveLength(0);
    // A closed unfilled diamond: four line segments and a stroke.
    expect(calls.filter((c) => c.op === "lineTo").length).toBeGreaterThanOrEqual(3);
    expect(calls.some((c) => c.op === "fill")).toBe(false);
  });

  it("draws a measured span of the same span_id as a bar", () => {
    // The control: identical geometry, a real clock, and now a length is fair.
    const { d, calls } = mount(inputWith([span({ started_at: 1000, ended_at: 1004 })]));
    d.setFollow(false);
    (d as unknown as { edge: number }).edge = 1010;
    d.draw();
    expect(bars(calls).length).toBeGreaterThan(0);
  });

  it("never gives an open span a cap", () => {
    const open: OpenSpan[] = [
      { runId: "r", spanId: "o", action: "execute", startedAt: 1000, reasoning: false, producingUntil: null },
    ];
    const { d, calls } = mount(inputWith([], open));
    d.setFollow(false);
    (d as unknown as { edge: number }).edge = 1010;
    d.draw();
    // Caps are 3–4px wide marks at the end of a bar. An open span gets a
    // gradient instead, because there is no end to cap.
    const caps = calls.filter((c) => c.op === "fillRect" && c.args[2]! <= 4 && c.args[2]! >= 3);
    expect(caps).toHaveLength(0);
  });
});

describe("lanes", () => {
  it("keeps a lane for every action even when the run never used it", () => {
    // An empty `verify` lane is the finding. A chart that only draws lanes it
    // has data for hides "this run never ran a test".
    const { d, calls } = mount(inputWith([]));
    d.draw();
    // nine actions + thinking + unclassified
    expect(laneRows(calls).length).toBeGreaterThanOrEqual(11);
  });

  it("gives unclassified work its own lane rather than a plausible one", () => {
    const { d, calls } = mount(
      inputWith([span({ action: null, started_at: 1000, ended_at: 1004 })]),
    );
    d.setFollow(false);
    (d as unknown as { edge: number }).edge = 1010;
    d.draw();
    const drawn = bars(calls);
    expect(drawn).toHaveLength(1);
    // Last lane, not folded up into `execute` or `inspect`.
    const rows = laneRows(calls);
    const lastLaneY = rows[rows.length - 1]!.args[1]!;
    expect(drawn[0]!.args[1]!).toBeGreaterThanOrEqual(lastLaneY);
  });
});

describe("the clock", () => {
  it("puts the leading edge at the newest event when the browser is behind", () => {
    // The collector stamps events; the browser draws them. If the two machines
    // disagree and we trusted the browser, the newest events would sit past the
    // right edge and vanish exactly when someone is watching.
    const future = Date.now() / 1000 + 600;
    const marks: Mark[] = [
      {
        runId: "r",
        eventId: "e",
        ts: future,
        eventType: "tool.completed",
        action: "edit",
        effect: null,
        fidelity: "native",
        spanId: null,
        reasoning: false,
      },
    ];
    const { d } = mount(inputWith([], [], marks));
    d.draw();
    expect((d as unknown as { now(): number }).now()).toBeGreaterThanOrEqual(future);
  });
});

describe("transport", () => {
  it("leaves follow mode when panned and reports it once", () => {
    const { d } = mount(inputWith([]));
    const seen: boolean[] = [];
    d.onFollowChange = (f) => seen.push(f);
    d.panBy(-40);
    d.panBy(-40);
    expect(d.isFollowing).toBe(false);
    expect(seen).toEqual([false]);
  });

  it("keeps the time under the cursor fixed while zooming", () => {
    const { d } = mount(inputWith([]));
    d.setFollow(false);
    (d as unknown as { edge: number }).edge = 1000;
    const before = d.windowS;
    // Cursor at the right edge: the time there is the window edge, so it must
    // not move no matter how the window is scaled around it.
    d.zoomBy(2, 800);
    expect(d.windowS).toBe(before * 2);
    expect((d as unknown as { edge: number }).edge).toBeCloseTo(1000, 6);
  });

  it("clamps the window rather than letting it collapse or run away", () => {
    const { d } = mount(inputWith([]));
    for (let i = 0; i < 40; i++) d.zoomBy(0.5, 400);
    expect(d.windowS).toBeGreaterThanOrEqual(5);
    for (let i = 0; i < 40; i++) d.zoomBy(2, 400);
    expect(d.windowS).toBeLessThanOrEqual(3600);
  });
});

// ── helpers ──────────────────────────────────────────────────────────────────

describe("capFor", () => {
  it("maps every known effect through the shared encoding", () => {
    expect(capFor("state_changed")).toBe(EFFECT_CAP.state_changed);
    expect(capFor("failed")).toBe(EFFECT_CAP.failed);
  });

  it("degrades an unknown or absent effect to hollow, not to a definite cap", () => {
    // A server that grows an effect ahead of the viewer must not have it drawn
    // as "changed nothing" — that is a claim, and we do not have it.
    expect(capFor(null)).toBe(EFFECT_CAP.unknown);
    expect(capFor("teleported")).toBe(EFFECT_CAP.unknown);
  });
});

describe("niceStep", () => {
  it("snaps to intervals a human reads off an axis", () => {
    expect(niceStep(0.3)).toBe(1);
    expect(niceStep(7)).toBe(10);
    expect(niceStep(40)).toBe(60);
    expect(niceStep(100000)).toBe(3600);
  });
});

describe("hexA", () => {
  it("carries a palette hex into a canvas alpha colour", () => {
    expect(hexA("#62d9c0", 0.5)).toBe("rgba(98,217,192,0.5)");
    expect(hexA("#000000", 1)).toBe("rgba(0,0,0,1)");
  });
});
