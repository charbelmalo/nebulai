/** Shared multi-touch gesture recognizer for the scene drivers.
 *
 *  Every 3-D driver hand-rolled the same single-pointer drag and left zoom on
 *  the wheel — which has no touch equivalent, so on a phone those views could
 *  be rotated but never zoomed. Worse, each tracked its drag in ONE `last`
 *  {x,y} with no pointerId key, so a second finger (or a palm) stomped the
 *  other's baseline and the camera moved at double speed.
 *
 *  This recognizer owns the touch half of the input and leaves the mouse half
 *  alone: every handler returns immediately unless `pointerType === "touch"`,
 *  so a driver adopts it by *adding* it next to the code it already has. The
 *  gesture model is the one AtlasDriver proved (and still runs privately,
 *  because its pinch is entangled with morph state and the id-picker):
 *
 *    1 finger   → onPan(dx, dy)
 *    2 fingers  → onPinch({ scale, midpoint drag, twist })
 *    down+up    → onTap, when it never crossed the drag threshold
 *
 *  Pure enough to unit-test in Node: the handlers take a structural
 *  `GesturePointer`, not the DOM `PointerEvent` class, and `attach()` is the
 *  only part that touches an EventTarget (vitest here runs without jsdom).
 */

/** The subset of PointerEvent the recognizer reads. A real PointerEvent
 *  satisfies this structurally, and so does a plain object in a test. */
export interface GesturePointer {
  pointerId: number;
  pointerType: string;
  clientX: number;
  clientY: number;
}

export interface PinchEvent {
  /** Distance ratio since the previous frame: >1 = fingers spreading = zoom in.
   *  Multiplicative, so a driver applies it as `dist /= scale` regardless of
   *  whether its zoom is a frustum width, an orbit radius or a time window. */
  scale: number;
  /** Current midpoint in client px — the anchor a cursor-anchored zoom needs. */
  cx: number;
  cy: number;
  /** Midpoint travel since the previous frame, client px. */
  dcx: number;
  dcy: number;
  /** Twist since the previous frame, radians, folded into (−π, π]. */
  dAngle: number;
  /** True once the accumulated twist cleared the deadzone. Stays true for the
   *  rest of the gesture: a pinch is never perfectly parallel, so without the
   *  latch every zoom would also drift the rotation. */
  angleEngaged: boolean;
}

export interface GestureCallbacks {
  /** One-finger drag delta in client px. Never fires while two fingers are down. */
  onPan?(dx: number, dy: number, p: GesturePointer): void;
  onPinch?(e: PinchEvent): void;
  onDragStart?(p: GesturePointer): void;
  onDragEnd?(p: GesturePointer): void;
  /** A down→up that never became a drag or a pinch. */
  onTap?(p: GesturePointer): void;
}

export interface GestureOptions {
  /** Movement below this reads as a tap, not a drag. */
  dragThresholdPx?: number;
  /** Twist below this is treated as pinch wobble, not rotation. */
  twistDeadzoneRad?: number;
}

interface PinchBaseline {
  dist: number;
  angle: number;
  cx: number;
  cy: number;
  twist: number;
  twistOn: boolean;
}

const DEFAULT_DRAG_THRESHOLD_PX = 3;
const DEFAULT_TWIST_DEADZONE = 0.06; // matches TOUCH_TWIST_DEADZONE in AtlasDriver

export class GestureRecognizer {
  private readonly cb: GestureCallbacks;
  private readonly dragThreshold: number;
  private readonly twistDeadzone: number;

  /** Live contacts, keyed by pointerId — the fix for the shared-`last` bug. */
  private touches = new Map<number, { x: number; y: number }>();
  private pinch: PinchBaseline | null = null;

  /** The one finger currently driving onPan, and where it was last seen. */
  private panId: number | null = null;
  private panStart: { x: number; y: number } | null = null;
  private panLast: { x: number; y: number } | null = null;
  private dragging = false;
  /** Set when a finger is left over from a pinch: it continues the gesture as a
   *  drag, so its eventual pointerup must not read as a tap. */
  private tapSuppressed = false;

  private target: EventTarget | null = null;
  private detach: (() => void) | null = null;

  constructor(cb: GestureCallbacks, opts: GestureOptions = {}) {
    this.cb = cb;
    this.dragThreshold = opts.dragThresholdPx ?? DEFAULT_DRAG_THRESHOLD_PX;
    this.twistDeadzone = opts.twistDeadzoneRad ?? DEFAULT_TWIST_DEADZONE;
  }

  /** True while at least one finger is down — drivers gate their mouse-only
   *  hover work on this so a touch drag never leaves a stale hover behind. */
  get isTouchActive(): boolean {
    return this.touches.size > 0;
  }

  /** Wire the touch listeners. Pass the driver's AbortSignal for cleanup; the
   *  listeners are non-passive because a gesture must be able to preventDefault
   *  even on a canvas that forgot its `touch-action: none`. */
  attach(target: EventTarget, signal?: AbortSignal): void {
    this.target = target;
    const down = (e: Event) => this.pointerDown(e as unknown as GesturePointer);
    const move = (e: Event) => this.pointerMove(e as unknown as GesturePointer);
    const up = (e: Event) => this.pointerUp(e as unknown as GesturePointer);
    const cancel = (e: Event) => this.pointerCancel(e as unknown as GesturePointer);
    const opts: AddEventListenerOptions = { signal, passive: false };
    target.addEventListener("pointerdown", down, opts);
    target.addEventListener("pointermove", move, opts);
    target.addEventListener("pointerup", up, opts);
    target.addEventListener("pointercancel", cancel, opts);
    this.detach = () => {
      target.removeEventListener("pointerdown", down);
      target.removeEventListener("pointermove", move);
      target.removeEventListener("pointerup", up);
      target.removeEventListener("pointercancel", cancel);
    };
  }

  dispose(): void {
    this.detach?.();
    this.detach = null;
    this.target = null;
    this.reset();
  }

  // ── handlers (public so tests can drive them without a DOM) ───────────────

  pointerDown(e: GesturePointer): void {
    if (e.pointerType !== "touch") return;
    this.touches.set(e.pointerId, { x: e.clientX, y: e.clientY });

    if (this.touches.size >= 2) {
      // a second finger ends whatever the first was doing and starts the pinch
      // from a fresh baseline — carrying the old one over would jump the camera
      if (this.dragging && this.panId !== null) this.endDrag(e);
      this.panId = null;
      this.panStart = null;
      this.panLast = null;
      this.dragging = false;
      this.tapSuppressed = false;
      this.beginPinch();
      return;
    }

    this.panId = e.pointerId;
    this.panStart = { x: e.clientX, y: e.clientY };
    this.panLast = { x: e.clientX, y: e.clientY };
    this.dragging = false;
    this.tapSuppressed = false;
  }

  pointerMove(e: GesturePointer): void {
    if (e.pointerType !== "touch") return;
    if (!this.touches.has(e.pointerId)) return;
    this.touches.set(e.pointerId, { x: e.clientX, y: e.clientY });

    if (this.touches.size >= 2) {
      this.stepPinch();
      return;
    }

    if (e.pointerId !== this.panId || !this.panStart || !this.panLast) return;
    if (!this.dragging) {
      const dx = e.clientX - this.panStart.x;
      const dy = e.clientY - this.panStart.y;
      if (Math.hypot(dx, dy) <= this.dragThreshold) {
        this.panLast = { x: e.clientX, y: e.clientY };
        return;
      }
      this.dragging = true;
      this.cb.onDragStart?.(e);
    }
    const dx = e.clientX - this.panLast.x;
    const dy = e.clientY - this.panLast.y;
    this.panLast = { x: e.clientX, y: e.clientY };
    if (dx !== 0 || dy !== 0) this.cb.onPan?.(dx, dy, e);
  }

  pointerUp(e: GesturePointer): void {
    if (e.pointerType !== "touch") return;
    if (!this.touches.delete(e.pointerId)) return;

    if (this.pinch) {
      // lifting out of a pinch: re-baseline if two fingers remain, and never let
      // the leftover contact fall through to the tap path — it is mid-gesture
      this.pinch = null;
      if (this.touches.size >= 2) {
        this.beginPinch();
        return;
      }
      const [id, only] = [...this.touches.entries()][0] ?? [];
      if (id !== undefined && only) {
        this.panId = id;
        this.panStart = { x: only.x, y: only.y };
        this.panLast = { x: only.x, y: only.y };
        this.dragging = true; // continue as a drag, not a fresh tap
        this.tapSuppressed = true;
      }
      return;
    }

    if (e.pointerId !== this.panId) return;
    const wasDragging = this.dragging;
    const suppressed = this.tapSuppressed;
    this.panId = null;
    this.panStart = null;
    this.panLast = null;
    this.dragging = false;
    this.tapSuppressed = false;
    if (wasDragging) {
      this.cb.onDragEnd?.(e);
      return;
    }
    if (!suppressed) this.cb.onTap?.(e);
  }

  pointerCancel(e: GesturePointer): void {
    if (e.pointerType !== "touch") return;
    if (!this.touches.delete(e.pointerId)) return;
    // a cancelled pointer never delivers pointerup, so state has to be dropped
    // here or the next gesture inherits a baseline naming a finger that is gone
    this.pinch = null;
    if (this.touches.size >= 2) {
      this.beginPinch();
      return;
    }
    if (e.pointerId === this.panId) {
      this.panId = null;
      this.panStart = null;
      this.panLast = null;
      this.dragging = false;
      this.tapSuppressed = false;
    }
  }

  /** Drop all gesture state without firing callbacks (view switches, dispose). */
  reset(): void {
    this.touches.clear();
    this.pinch = null;
    this.panId = null;
    this.panStart = null;
    this.panLast = null;
    this.dragging = false;
    this.tapSuppressed = false;
  }

  // ── pinch math ────────────────────────────────────────────────────────────

  private beginPinch(): void {
    const [a, b] = [...this.touches.values()];
    if (!a || !b) return;
    this.pinch = {
      dist: Math.max(Math.hypot(b.x - a.x, b.y - a.y), 1),
      angle: Math.atan2(b.y - a.y, b.x - a.x),
      cx: (a.x + b.x) / 2,
      cy: (a.y + b.y) / 2,
      twist: 0,
      twistOn: false,
    };
  }

  private stepPinch(): void {
    const prev = this.pinch;
    if (!prev) return;
    const [a, b] = [...this.touches.values()];
    if (!a || !b) return;

    const dist = Math.max(Math.hypot(b.x - a.x, b.y - a.y), 1);
    const angle = Math.atan2(b.y - a.y, b.x - a.x);
    const cx = (a.x + b.x) / 2;
    const cy = (a.y + b.y) / 2;

    // atan2 wraps at ±π; fold the delta back into (−π, π] so crossing the seam
    // can't fling the camera a full turn
    let dAngle = angle - prev.angle;
    if (dAngle > Math.PI) dAngle -= Math.PI * 2;
    else if (dAngle < -Math.PI) dAngle += Math.PI * 2;

    const twist = prev.twist + dAngle;
    const engaged = prev.twistOn || Math.abs(twist) > this.twistDeadzone;

    this.pinch = { dist, angle, cx, cy, twist, twistOn: engaged };

    this.cb.onPinch?.({
      scale: dist / prev.dist,
      cx,
      cy,
      dcx: cx - prev.cx,
      dcy: cy - prev.cy,
      dAngle: engaged ? dAngle : 0,
      angleEngaged: engaged,
    });
  }

  private endDrag(e: GesturePointer): void {
    this.cb.onDragEnd?.(e);
  }
}
