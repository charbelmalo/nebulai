/** ScoreDriver — the run as a score: time across, action down.
 *
 *  The first of the live view's projections. Its geometry is `x = time,
 *  y = action lane`, which is also the geometry the fleet strip and the span
 *  tree use with a different `y` — so the morph between them (L2, L3) is a
 *  change of one axis, not a change of view. Everything it draws it draws
 *  through `seer/encoding.ts`, so a green bar here means what a green particle
 *  means in the field.
 *
 *  ## Where each bar comes from, and why it matters
 *
 *  A **closed** span is drawn from `view.spans` — Python's record. A span that
 *  is still **open** is drawn from `LiveModel.openSpans()` — our leading edge.
 *  That split is not an implementation detail: `LiveModel` drops a span the
 *  instant it closes precisely so there is no second, drifting copy of the
 *  record in TS. The consequence is visible here — the moment a tool finishes,
 *  its bar stops being ours and becomes the reducer's, and the two must line up
 *  because they were never two measurements in the first place.
 *
 *  ## What the drawing is not allowed to imply
 *
 *  · **An open bar is not a measurement.** It has no end yet, so it is drawn
 *    with a dissolving leading edge rather than a cap. Anything with a cap
 *    reads as finished, and a length the eye can compare reads as measured.
 *  · **A span with no clock is not a zero-length span.** Reconciled runs carry
 *    `synthetic_start` and an absent `duration_fidelity`: the call happened,
 *    nobody timed it. Those are drawn as a hollow diamond at the moment they
 *    ended, never as a bar of any length — see `isProvisional`.
 *  · **An empty lane is a fact.** All nine actions are always drawn, in the
 *    contract's own order. A lane that appears only once work lands in it makes
 *    "this run never ran a test" invisible, and that is usually the finding.
 */

import type { Action, Effect, Fidelity } from "../../seer/contract";
import { ACTIONS } from "../../seer/contract";
import type { RunView, SpanRecord } from "../../seer/client";
import type { Mark, OpenSpan } from "../../seer/live";
import {
  ACTION_COLOR,
  EFFECT_CAP,
  FIDELITY_TEXTURE,
  NEUTRAL_INK,
  isProvisional,
  markInk,
  type EffectCap,
  type Texture,
} from "../../seer/encoding";

export interface ScoreInput {
  /** Python's record. Null before the first fetch, which is honest: no bars. */
  view: RunView | null;
  openSpans: readonly OpenSpan[];
  marks: readonly Mark[];
}

/** What sits under the cursor. `provisional` is carried so the readout can say
 *  "still open" or "not timed" instead of printing a length. */
export interface ScoreHover {
  label: string;
  action: Action | null;
  detail: string | null;
  startedAt: number;
  endedAt: number | null;
  durationS: number | null;
  fidelity: Fidelity;
  provisional: boolean;
  x: number;
  y: number;
}

const EMPTY_INPUT: ScoreInput = { view: null, openSpans: [], marks: [] };

const GUTTER = 78;
const AXIS_H = 15;
const RAIL_H = 9;
const LANE_MIN = 11;
const LANE_MAX = 22;
const MIN_WINDOW_S = 5;
const MAX_WINDOW_S = 3600;
/** Bars thinner than this are still drawn — a 3ms tool call happened, and a run
 *  made of hundreds of them should look busy rather than empty. */
const MIN_BAR_PX = 2;

type LaneKey = Action | "thinking" | "unclassified";

/** Lane order: thinking on top, then the contract's own action order — which
 *  roughly follows a healthy run's path, so a trajectory reads top-to-bottom —
 *  and `unclassified` last.
 *
 *  That last lane is not a rounding error. Work the adapter declined to
 *  classify has to land somewhere it can be counted by eye; folding it into a
 *  plausible neighbour would invent the classification the adapter refused to
 *  make, and dropping it would make an adapter that classifies nothing look
 *  like an agent that did nothing. A run with a fat `unclassified` lane is
 *  telling you about the adapter, and that is worth seeing. */
const LANES: LaneKey[] = ["thinking", ...ACTIONS, "unclassified"];

interface HitRect {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  hover: Omit<ScoreHover, "x" | "y">;
}

export class ScoreDriver {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private raf = 0;
  private w = 0;
  private h = 0;
  private dpr = 1;

  /** Pulled once per frame rather than pushed on change.
   *
   *  `LiveModel` is not a signal and deliberately does not become one: a busy
   *  agent lands dozens of events a second, and a re-render per event would
   *  make the page the slowest thing watching the agent — the same reason
   *  `markDirty` coalesces its refetch. The chart already redraws every frame,
   *  so reading the model there costs nothing and is never stale. */
  private source: () => ScoreInput = () => EMPTY_INPUT;
  private input: ScoreInput = EMPTY_INPUT;
  private hits: HitRect[] = [];
  private cursor: { x: number; y: number } | null = null;
  private reducedMotion = false;

  /** Seconds visible across the canvas. */
  windowS = 90;
  /** Right edge of the window, in event-clock seconds. Only meaningful when
   *  not following. */
  private edge = 0;
  private following = true;

  onHover?: (h: ScoreHover | null) => void;
  onFollowChange?: (following: boolean) => void;
  onWindowChange?: (windowS: number) => void;

  init(canvas: HTMLCanvasElement): void {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    canvas.addEventListener("pointermove", this.onPointerMove);
    canvas.addEventListener("pointerleave", this.onPointerLeave);
    canvas.addEventListener("pointerdown", this.onPointerDown);
    canvas.addEventListener("wheel", this.onWheel, { passive: false });
    this.loop();
  }

  setSource(source: () => ScoreInput): void {
    this.source = source;
  }

  setReducedMotion(on: boolean): void {
    this.reducedMotion = on;
  }

  resize(width: number, height: number, dpr: number): void {
    const c = this.canvas;
    if (!c) return;
    this.w = width;
    this.h = height;
    this.dpr = dpr;
    c.width = Math.max(1, Math.round(width * dpr));
    c.height = Math.max(1, Math.round(height * dpr));
    c.style.width = `${width}px`;
    c.style.height = `${height}px`;
  }

  // ── transport ──────────────────────────────────────────────────────────

  get isFollowing(): boolean {
    return this.following;
  }

  setFollow(on: boolean): void {
    if (this.following === on) return;
    this.following = on;
    if (on) this.edge = this.now();
    this.onFollowChange?.(on);
  }

  /** Pan by a pixel delta. Dragging left moves forward in time. Any pan leaves
   *  follow mode — a view that snapped back to the live edge mid-drag would
   *  fight the hand holding it. */
  panBy(dxPx: number): void {
    if (this.w <= GUTTER) return;
    this.setFollow(false);
    this.edge += (dxPx / (this.w - GUTTER)) * this.windowS;
  }

  /** Zoom about a pixel position, so whatever is under the cursor stays there. */
  zoomBy(factor: number, atPx: number): void {
    const plot = this.w - GUTTER;
    if (plot <= 0) return;
    const next = clamp(this.windowS * factor, MIN_WINDOW_S, MAX_WINDOW_S);
    if (next === this.windowS) return;
    const edge = this.currentEdge();
    // Time under the cursor before and after; shift the edge to keep it fixed.
    const frac = clamp((atPx - GUTTER) / plot, 0, 1);
    const tAt = edge - (1 - frac) * this.windowS;
    this.windowS = next;
    this.onWindowChange?.(next);
    const newEdge = tAt + (1 - frac) * next;
    if (this.following && newEdge < this.now()) {
      // Zooming out at the live edge keeps following; zooming into the past
      // does not, because it has stopped being the live edge.
      this.setFollow(false);
    }
    this.edge = newEdge;
  }

  dispose(): void {
    cancelAnimationFrame(this.raf);
    this.raf = 0;
    const c = this.canvas;
    if (c) {
      c.removeEventListener("pointermove", this.onPointerMove);
      c.removeEventListener("pointerleave", this.onPointerLeave);
      c.removeEventListener("pointerdown", this.onPointerDown);
      c.removeEventListener("wheel", this.onWheel);
    }
    this.canvas = null;
    this.ctx = null;
  }

  // ── clock ──────────────────────────────────────────────────────────────

  /** "Now" on the event clock.
   *
   *  Events are stamped by the collector's clock and the window is drawn
   *  against the browser's. Those disagree by however far the two machines have
   *  drifted, and if the browser is behind, the newest events sit past the right
   *  edge and vanish. Taking the later of the two keeps the leading edge at or
   *  ahead of the newest thing we have heard about, which is the property the
   *  drawing actually needs. */
  private now(): number {
    const wall = Date.now() / 1000;
    const newest = this.input.marks.length
      ? this.input.marks[this.input.marks.length - 1]!.ts
      : 0;
    let open = 0;
    for (const s of this.input.openSpans) open = Math.max(open, s.startedAt);
    return Math.max(wall, newest, open);
  }

  private currentEdge(): number {
    return this.following ? this.now() : this.edge;
  }

  // ── input ──────────────────────────────────────────────────────────────

  private onPointerMove = (e: PointerEvent): void => {
    const c = this.canvas;
    if (!c) return;
    const r = c.getBoundingClientRect();
    this.cursor = { x: e.clientX - r.left, y: e.clientY - r.top };
    if (e.buttons & 1) {
      this.panBy(-e.movementX);
      this.onHover?.(null);
      return;
    }
    this.emitHover();
  };

  private onPointerLeave = (): void => {
    this.cursor = null;
    this.onHover?.(null);
  };

  private onPointerDown = (e: PointerEvent): void => {
    this.canvas?.setPointerCapture(e.pointerId);
  };

  private onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    const c = this.canvas;
    if (!c) return;
    const r = c.getBoundingClientRect();
    this.zoomBy(e.deltaY > 0 ? 1.15 : 1 / 1.15, e.clientX - r.left);
  };

  private emitHover(): void {
    if (!this.onHover) return;
    const p = this.cursor;
    if (!p) return this.onHover(null);
    for (let i = this.hits.length - 1; i >= 0; i--) {
      const hit = this.hits[i]!;
      if (p.x >= hit.x0 - 2 && p.x <= hit.x1 + 2 && p.y >= hit.y0 && p.y <= hit.y1) {
        this.onHover({ ...hit.hover, x: p.x, y: p.y });
        return;
      }
    }
    this.onHover(null);
  }

  // ── the frame ──────────────────────────────────────────────────────────

  private loop = (): void => {
    this.raf = requestAnimationFrame(this.loop);
    if (document.hidden) return;
    this.draw();
  };

  /** Exposed for tests and for the frozen-frame verification path: draws one
   *  frame synchronously without the rAF loop. */
  draw(): void {
    const ctx = this.ctx;
    if (!ctx || this.w <= 0 || this.h <= 0) return;
    this.input = this.source();

    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    this.hits = [];

    const edge = this.currentEdge();
    const t0 = edge - this.windowS;
    const plot = Math.max(1, this.w - GUTTER);
    const laneH = clamp(
      (this.h - AXIS_H - RAIL_H) / LANES.length,
      LANE_MIN,
      LANE_MAX,
    );
    const x = (t: number): number => GUTTER + ((t - t0) / this.windowS) * plot;

    this.drawAxis(ctx, edge, t0, x, plot);
    this.drawLanes(ctx, laneH);

    const view = this.input.view;
    if (view) {
      for (const s of view.spans) this.drawClosedSpan(ctx, s, x, laneH, t0, edge);
    }
    for (const s of this.input.openSpans) this.drawOpenSpan(ctx, s, x, laneH, edge);
    this.drawRail(ctx, x, laneH, t0, edge);
  }

  private laneY(key: LaneKey, laneH: number): number {
    return AXIS_H + LANES.indexOf(key) * laneH;
  }

  private drawAxis(
    ctx: CanvasRenderingContext2D,
    edge: number,
    t0: number,
    x: (t: number) => number,
    plot: number,
  ): void {
    ctx.save();
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textBaseline = "top";

    // A tick roughly every 90px, snapped to a readable interval.
    const target = (this.windowS / plot) * 90;
    const step = niceStep(target);
    ctx.strokeStyle = "rgba(255,255,255,0.055)";
    ctx.fillStyle = "rgba(255,255,255,0.30)";
    ctx.lineWidth = 1;
    const first = Math.ceil(t0 / step) * step;
    for (let t = first; t <= edge; t += step) {
      const px = Math.round(x(t)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(px, AXIS_H);
      ctx.lineTo(px, this.h);
      ctx.stroke();
      const rel = Math.round(t - edge);
      if (rel !== 0) ctx.fillText(`${rel}s`, px + 3, 2);
    }

    // The live edge. Labelled "now" only while following: parked in the past it
    // is the right edge of the window, which is a different claim.
    const ex = Math.round(x(edge)) + 0.5;
    ctx.strokeStyle = this.following ? "rgba(92,199,237,0.55)" : "rgba(255,255,255,0.18)";
    ctx.beginPath();
    ctx.moveTo(ex, AXIS_H);
    ctx.lineTo(ex, this.h);
    ctx.stroke();
    if (this.following) {
      ctx.fillStyle = "rgba(92,199,237,0.85)";
      ctx.textAlign = "right";
      ctx.fillText("now", ex - 3, 2);
    }
    ctx.restore();
  }

  private drawLanes(ctx: CanvasRenderingContext2D, laneH: number): void {
    ctx.save();
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textBaseline = "middle";
    for (const key of LANES) {
      const y = this.laneY(key, laneH);
      const special = key === "thinking" || key === "unclassified";
      ctx.fillStyle = special ? "rgba(255,255,255,0.022)" : "rgba(255,255,255,0.012)";
      ctx.fillRect(GUTTER, y, this.w - GUTTER, laneH - 1);
      ctx.fillStyle =
        key === "thinking"
          ? "rgba(176,166,240,0.72)"
          : key === "unclassified"
            ? hexA(NEUTRAL_INK, 0.85)
            : hexA(ACTION_COLOR[key], 0.78);
      ctx.fillText(key, 6, y + laneH / 2);
    }
    ctx.restore();
  }

  // ── marks ──────────────────────────────────────────────────────────────

  private drawClosedSpan(
    ctx: CanvasRenderingContext2D,
    s: SpanRecord,
    x: (t: number) => number,
    laneH: number,
    t0: number,
    edge: number,
  ): void {
    const action = (s.action as Action | null) ?? null;
    const end = s.ended_at ?? s.started_at;
    if (end < t0 || s.started_at > edge) return;
    const y = this.laneY(action ?? "unclassified", laneH) + 2;
    const barH = Math.max(3, laneH - 6);
    const fid = s.duration_fidelity;
    const ink = s.failed ? "#ff5c7a" : markInk(action, fid);
    const hover = {
      label: s.detail || action || "unclassified",
      action,
      detail: s.detail,
      startedAt: s.started_at,
      endedAt: s.ended_at,
    };

    if (isProvisional({ endedAt: s.ended_at, fidelity: fid }) || s.synthetic_start) {
      // It ended, nobody timed it. A hollow diamond at the moment it ended —
      // never a bar, because a bar of any length is a duration claim.
      const cx = x(end);
      this.diamond(ctx, cx, y + barH / 2, Math.min(5, barH / 2 + 1), ink);
      this.hits.push({
        x0: cx - 5,
        x1: cx + 5,
        y0: y,
        y1: y + barH,
        hover: { ...hover, durationS: null, fidelity: fid, provisional: true },
      });
      return;
    }

    const xa = Math.max(x(s.started_at), GUTTER);
    const xb = Math.min(x(end), this.w);
    if (xb <= GUTTER) return;
    const w = Math.max(MIN_BAR_PX, xb - xa);
    this.bar(ctx, xa, y, w, barH, ink, FIDELITY_TEXTURE[fid]);
    this.cap(ctx, xa + w, y, barH, ink, capFor(s.effect));
    this.hits.push({
      x0: xa,
      x1: xa + w,
      y0: y,
      y1: y + barH,
      hover: { ...hover, durationS: s.duration_s, fidelity: fid, provisional: false },
    });
  }

  private drawOpenSpan(
    ctx: CanvasRenderingContext2D,
    s: OpenSpan,
    x: (t: number) => number,
    laneH: number,
    edge: number,
  ): void {
    const key: LaneKey = s.reasoning ? "thinking" : (s.action ?? "unclassified");
    const y = this.laneY(key, laneH) + 2;
    const barH = Math.max(3, laneH - 6);
    const xa = Math.max(x(s.startedAt), GUTTER);
    const xb = Math.min(x(edge), this.w);
    if (xb <= GUTTER) return;
    const ink = s.reasoning ? "#b0a6f0" : s.action ? ACTION_COLOR[s.action] : NEUTRAL_INK;

    // No cap, and the leading edge dissolves: this has not ended, so nothing
    // about it may read as a finished length.
    const g = ctx.createLinearGradient(Math.max(xa, xb - 42), 0, xb, 0);
    g.addColorStop(0, hexA(ink, 0.75));
    g.addColorStop(1, hexA(ink, 0.12));
    ctx.fillStyle = hexA(ink, 0.75);
    ctx.fillRect(xa, y, Math.max(0, xb - xa - 42), barH);
    ctx.fillStyle = g;
    ctx.fillRect(Math.max(xa, xb - 42), y, Math.min(42, xb - xa), barH);

    // A pulse while output is actually streaming. Steady when it is not, so
    // "thinking quietly" and "producing" are distinguishable at a glance.
    const producing = s.producingUntil != null && edge - s.producingUntil < 1.5;
    if (producing && !this.reducedMotion) {
      const a = 0.45 + 0.35 * Math.sin(Date.now() / 140);
      ctx.fillStyle = hexA(ink, a);
      ctx.fillRect(xb - 2, y - 1, 2, barH + 2);
    } else if (producing) {
      ctx.fillStyle = hexA(ink, 0.8);
      ctx.fillRect(xb - 2, y - 1, 2, barH + 2);
    }

    this.hits.push({
      x0: xa,
      x1: xb,
      y0: y,
      y1: y + barH,
      hover: {
        label: s.reasoning ? "thinking" : (s.action ?? "tool"),
        action: s.action,
        detail: null,
        startedAt: s.startedAt,
        endedAt: null,
        durationS: null,
        fidelity: "deterministic",
        provisional: true,
      },
    });
  }

  /** Every mark as a tick on a baseline rail.
   *
   *  The lanes only hold classified work. Session and turn boundaries, model
   *  requests, approvals and warnings have no action and would otherwise be
   *  invisible — and "the run is emitting nothing at all" is exactly the state
   *  someone watching needs to be able to see. */
  private drawRail(
    ctx: CanvasRenderingContext2D,
    x: (t: number) => number,
    laneH: number,
    t0: number,
    edge: number,
  ): void {
    const y = AXIS_H + LANES.length * laneH + 2;
    ctx.save();
    ctx.fillStyle = "rgba(255,255,255,0.03)";
    ctx.fillRect(GUTTER, y, this.w - GUTTER, RAIL_H - 3);
    for (const m of this.input.marks) {
      if (m.ts < t0 || m.ts > edge) continue;
      ctx.fillStyle = hexA(markInk(m.action, m.fidelity), 0.62);
      ctx.fillRect(Math.round(x(m.ts)), y, 1, RAIL_H - 3);
    }
    ctx.restore();
  }

  // ── primitives ─────────────────────────────────────────────────────────

  private bar(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    ink: string,
    texture: Texture,
  ): void {
    ctx.save();
    if (texture === "outline" || texture === "policy") {
      // Unfilled. The whole point of both is that there is no magnitude to
      // claim, and a fill of any density is a claim.
      ctx.strokeStyle = hexA(ink, 0.85);
      ctx.lineWidth = 1;
      if (texture === "policy") ctx.setLineDash([2, 2]);
      ctx.strokeRect(x + 0.5, y + 0.5, Math.max(1, w - 1), h - 1);
      ctx.restore();
      return;
    }
    ctx.fillStyle = texture === "solid" ? ink : hexA(ink, 0.4);
    ctx.fillRect(x, y, w, h);
    if (texture === "hatched" || texture === "dotted") {
      ctx.beginPath();
      ctx.rect(x, y, w, h);
      ctx.clip();
      ctx.strokeStyle = hexA(ink, 0.95);
      ctx.lineWidth = 1;
      if (texture === "dotted") ctx.setLineDash([1, 3]);
      for (let i = -h; i < w + h; i += 4) {
        ctx.beginPath();
        ctx.moveTo(x + i, y + h);
        ctx.lineTo(x + i + h, y);
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  /** The effect, at the end of the mark, where the eye arrives last. */
  private cap(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    h: number,
    ink: string,
    cap: EffectCap,
  ): void {
    const m = y + h / 2;
    ctx.save();
    ctx.fillStyle = ink;
    ctx.strokeStyle = ink;
    ctx.lineWidth = 1;
    switch (cap) {
      case "wedge":
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + 4, m);
        ctx.lineTo(x, y + h);
        ctx.closePath();
        ctx.fill();
        break;
      case "block":
        ctx.fillRect(x, y - 1, 3, h + 2);
        break;
      case "tick":
        ctx.fillRect(x, m - 0.5, 4, 1);
        break;
      case "flat":
        ctx.fillRect(x, y, 1, h);
        break;
      case "cross":
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + 4, y + h);
        ctx.moveTo(x + 4, y);
        ctx.lineTo(x, y + h);
        ctx.stroke();
        break;
      case "hollow":
        ctx.strokeRect(x + 0.5, y + 0.5, 3, h - 1);
        break;
    }
    ctx.restore();
  }

  private diamond(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    r: number,
    ink: string,
  ): void {
    ctx.save();
    ctx.strokeStyle = ink;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, cy - r);
    ctx.lineTo(cx + r, cy);
    ctx.lineTo(cx, cy + r);
    ctx.lineTo(cx - r, cy);
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
  }
}

// ── helpers ──────────────────────────────────────────────────────────────────

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** The cap for a span's effect. `null` and anything we do not recognise both
 *  land on `unknown`'s hollow cap rather than on a definite-looking one: a
 *  server that grew an effect we have not shipped yet must not have it drawn as
 *  "changed nothing". */
export function capFor(effect: string | null): EffectCap {
  if (!effect) return EFFECT_CAP.unknown;
  return EFFECT_CAP[effect as Effect] ?? EFFECT_CAP.unknown;
}

/** Snap a target interval to something a human reads off an axis. */
export function niceStep(target: number): number {
  const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
  for (const s of steps) if (target <= s) return s;
  return 3600;
}

/** `#rrggbb` + alpha → `rgba(...)`. The palette is hex because that is what a
 *  designer hands over; the canvas wants alpha separately. */
export function hexA(hex: string, a: number): string {
  const h = hex.replace("#", "");
  const n = parseInt(h, 16);
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  return `rgba(${r},${g},${b},${a})`;
}
