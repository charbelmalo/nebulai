import { describe, expect, it } from "vitest";
import { GestureRecognizer, type GesturePointer, type PinchEvent } from "../../src/scene/gestures";

/** Vitest here runs in plain Node (no jsdom) — the recognizer's handlers take
 *  a structural `GesturePointer`, not a DOM `PointerEvent`, so tests drive
 *  `pointerDown`/`pointerMove`/`pointerUp`/`pointerCancel` directly with plain
 *  objects instead of going through `attach()`. */

function touch(id: number, x: number, y: number): GesturePointer {
  return { pointerId: id, pointerType: "touch", clientX: x, clientY: y };
}

function mouse(id: number, x: number, y: number): GesturePointer {
  return { pointerId: id, pointerType: "mouse", clientX: x, clientY: y };
}

interface Recorder {
  pan: [number, number][];
  pinch: PinchEvent[];
  dragStart: number;
  dragEnd: number;
  tap: number;
}

function makeRecorder(): Recorder {
  return { pan: [], pinch: [], dragStart: 0, dragEnd: 0, tap: 0 };
}

function makeRec(rec: Recorder, opts?: { dragThresholdPx?: number; twistDeadzoneRad?: number }) {
  return new GestureRecognizer(
    {
      onPan: (dx, dy) => rec.pan.push([dx, dy]),
      onPinch: (e) => rec.pinch.push(e),
      onDragStart: () => (rec.dragStart += 1),
      onDragEnd: () => (rec.dragEnd += 1),
      onTap: () => (rec.tap += 1),
    },
    opts,
  );
}

describe("GestureRecognizer", () => {
  it("spreading two fingers gives scale > 1; pinching in gives scale < 1", () => {
    const rec = makeRecorder();
    const g = makeRec(rec);

    // two fingers land 100px apart
    g.pointerDown(touch(1, 0, 0));
    g.pointerDown(touch(2, 100, 0));
    // spread apart to 200px
    g.pointerMove(touch(1, -50, 0));
    g.pointerMove(touch(2, 150, 0));

    expect(rec.pinch.length).toBeGreaterThan(0);
    const spreadEvent = rec.pinch.at(-1)!;
    expect(spreadEvent.scale).toBeGreaterThan(1);

    // now pinch back in
    g.pointerMove(touch(1, 0, 0));
    g.pointerMove(touch(2, 100, 0));
    const pinchEvent = rec.pinch.at(-1)!;
    expect(pinchEvent.scale).toBeLessThan(1);
  });

  it("twist deadzone: a pure zoom with no rotation leaks no azimuth, and once past it stays engaged", () => {
    const rec = makeRecorder();
    const g = makeRec(rec, { twistDeadzoneRad: 0.06 });

    // baseline: horizontal pair, 100px apart
    g.pointerDown(touch(1, 0, 0));
    g.pointerDown(touch(2, 100, 0));

    // pure spread, dead straight along x — no rotation at all
    g.pointerMove(touch(1, -20, 0));
    g.pointerMove(touch(2, 120, 0));
    let e = rec.pinch.at(-1)!;
    expect(e.dAngle).toBe(0);
    expect(e.angleEngaged).toBe(false);

    // now twist hard past the deadzone (rotate the pair ~90deg CCW-ish via y offset)
    g.pointerMove(touch(1, -20, -60));
    g.pointerMove(touch(2, 120, 60));
    e = rec.pinch.at(-1)!;
    expect(e.angleEngaged).toBe(true);

    // twist dips back toward parallel — engaged must stay latched true
    g.pointerMove(touch(1, -20, -1));
    g.pointerMove(touch(2, 120, 1));
    e = rec.pinch.at(-1)!;
    expect(e.angleEngaged).toBe(true);
  });

  it("second finger down cancels the 1-finger drag: onPan stops, onPinch starts", () => {
    const rec = makeRecorder();
    const g = makeRec(rec);

    g.pointerDown(touch(1, 0, 0));
    g.pointerMove(touch(1, 20, 0)); // past the drag threshold
    expect(rec.dragStart).toBe(1);
    expect(rec.pan.length).toBe(1);

    // second finger lands — the 1-finger drag ends and a pinch begins
    g.pointerDown(touch(2, 100, 0));
    expect(rec.dragEnd).toBe(1);

    const panCountBeforePinch = rec.pan.length;
    g.pointerMove(touch(1, -10, 0));
    g.pointerMove(touch(2, 110, 0));
    expect(rec.pinch.length).toBeGreaterThan(0);
    // no further onPan fired once the pinch took over
    expect(rec.pan.length).toBe(panCountBeforePinch);
  });

  it("2->1 finger transition re-baselines and the final pointerup fires no onTap", () => {
    const rec = makeRecorder();
    const g = makeRec(rec);

    g.pointerDown(touch(1, 0, 0));
    g.pointerDown(touch(2, 100, 0));
    g.pointerMove(touch(1, -10, 0));
    g.pointerMove(touch(2, 110, 0));
    expect(rec.pinch.length).toBeGreaterThan(0);

    // lift finger 2 — finger 1 continues as a drag, not a fresh tap candidate
    g.pointerUp(touch(2, 110, 0));
    expect(rec.tap).toBe(0);

    // moving the remaining finger should now drive onPan (continuing drag)
    g.pointerMove(touch(1, 5, 0));
    expect(rec.pan.length).toBeGreaterThan(0);

    // lifting the last finger must not fire onTap — it was suppressed as
    // continuing a pinch-derived drag
    g.pointerUp(touch(1, 5, 0));
    expect(rec.tap).toBe(0);
  });

  it("pointercancel mid-pinch fires no onTap", () => {
    const rec = makeRecorder();
    const g = makeRec(rec);

    g.pointerDown(touch(1, 0, 0));
    g.pointerDown(touch(2, 100, 0));
    g.pointerMove(touch(1, -10, 0));
    g.pointerMove(touch(2, 110, 0));

    g.pointerCancel(touch(1, -10, 0));
    g.pointerCancel(touch(2, 110, 0));
    expect(rec.tap).toBe(0);
  });

  it("onPan never fires for pointerType mouse", () => {
    const rec = makeRecorder();
    const g = makeRec(rec);

    g.pointerDown(mouse(1, 0, 0));
    g.pointerMove(mouse(1, 50, 50));
    g.pointerUp(mouse(1, 50, 50));

    expect(rec.pan.length).toBe(0);
    expect(rec.dragStart).toBe(0);
    expect(rec.dragEnd).toBe(0);
    expect(rec.tap).toBe(0);
  });

  it("a down->up under the drag threshold fires onTap; over it fires onDragStart/onPan/onDragEnd and no onTap", () => {
    const rec = makeRecorder();
    const g = makeRec(rec, { dragThresholdPx: 3 });

    // tap: barely moves, stays under threshold
    g.pointerDown(touch(1, 0, 0));
    g.pointerMove(touch(1, 1, 0));
    g.pointerUp(touch(1, 1, 0));
    expect(rec.tap).toBe(1);
    expect(rec.dragStart).toBe(0);
    expect(rec.pan.length).toBe(0);
    expect(rec.dragEnd).toBe(0);

    // drag: crosses the threshold
    g.pointerDown(touch(2, 0, 0));
    g.pointerMove(touch(2, 10, 0));
    g.pointerMove(touch(2, 20, 0));
    g.pointerUp(touch(2, 20, 0));
    expect(rec.dragStart).toBe(1);
    expect(rec.pan.length).toBeGreaterThan(0);
    expect(rec.dragEnd).toBe(1);
    expect(rec.tap).toBe(1); // unchanged from the earlier tap
  });
});
