/** LiveDriver — one surface, several meanings of `y`.
 *
 *  The live view is not a set of charts. It is one event stream drawn several
 *  ways, and the ways that share a time axis share this driver: **x is always
 *  time**, and the mode decides what `y` means.
 *
 *    score      y = action lane     what kind of work, over time
 *    fleet      y = run             who is doing it, over time
 *
 *  Switching between them is a *morph*, not a redraw. Every mark keeps its
 *  identity and travels from its old row to its new one, because the point
 *  being made is that these are the same events regrouped — a cross-fade
 *  between two pictures would say the opposite. Only the lane chrome
 *  cross-fades, because a label is a name rather than a datum, and a name
 *  sliding to a row it does not belong to would be the one lie in the picture.
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
  stateInk,
  type EffectCap,
  type Texture,
} from "../../seer/encoding";

/** One run's contribution to the surface. `view` is Python's snapshot and may
 *  be null before the first fetch, which is honest: no bars yet. */
export interface LiveRun {
  runId: string;
  label: string;
  state: string | null;
  view: RunView | null;
  openSpans: readonly OpenSpan[];
  marks: readonly Mark[];
}

export interface LiveInput {
  runs: readonly LiveRun[];
}

export type YMode = "score" | "fleet";

/** What sits under the cursor. `provisional` is carried so the readout can say
 *  "still open" or "not timed" instead of printing a length. */
export interface LiveHover {
  runId: string;
  runLabel: string;
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

const EMPTY_INPUT: LiveInput = { runs: [] };

const GUTTER = 82;
const AXIS_H = 15;
const RAIL_H = 9;
const LANE_MIN = 11;
const LANE_MAX = 22;
const MIN_WINDOW_S = 5;
const MAX_WINDOW_S = 3600;
/** Bars thinner than this are still drawn — a 3ms tool call happened, and a run
 *  made of hundreds of them should look busy rather than empty. */
const MIN_BAR_PX = 2;
/** Seconds for a full mode morph. Long enough that the eye can follow a mark
 *  from one row to another, which is the entire argument the morph is making. */
const MORPH_S = 0.55;

type ScoreLane = Action | "thinking" | "unclassified";

/** Score-mode lane order: thinking on top, then the contract's own action order
 *  — which roughly follows a healthy run's path, so a trajectory reads
 *  top-to-bottom — and `unclassified` last.
 *
 *  That last lane is not a rounding error. Work the adapter declined to
 *  classify has to land somewhere it can be counted by eye; folding it into a
 *  plausible neighbour would invent the classification the adapter refused to
 *  make, and dropping it would make an adapter that classifies nothing look
 *  like an agent that did nothing. A run with a fat `unclassified` lane is
 *  telling you about the adapter, and that is worth seeing. */
export const SCORE_LANES: ScoreLane[] = ["thinking", ...ACTIONS, "unclassified"];

interface HitRect {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  hover: Omit<LiveHover, "x" | "y">;
}

interface Geometry {
  /** 0 = score, 1 = fleet. */
  t: number;
  scoreH: number;
  fleetH: number;
  nRuns: number;
}

export class LiveDriver {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private raf = 0;
  private lastFrame = 0;
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
  private source: () => LiveInput = () => EMPTY_INPUT;
  private input: LiveInput = EMPTY_INPUT;
  private hits: HitRect[] = [];
  private cursor: { x: number; y: number } | null = null;
  private reducedMotion = false;

  private morph = 0;
  private morphTarget = 0;

  /** Seconds visible across the canvas. */
  windowS = 90;
  /** Right edge of the window, in event-clock seconds. Only meaningful when
   *  not following. */
  private edge = 0;
  private following = true;

  onHover?: (h: LiveHover | null) => void;
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

  setSource(source: () => LiveInput): void {
    this.source = source;
  }

  setReducedMotion(on: boolean): void {
    this.reducedMotion = on;
  }

  /** Switch what `y` means. Under reduced motion the morph is skipped rather
   *  than slowed: someone who has asked for no motion is not helped by a slow
   *  version of the thing they turned off. */
  setMode(mode: YMode): void {
    this.morphTarget = mode === "fleet" ? 1 : 0;
    if (this.reducedMotion) this.morph = this.morphTarget;
  }

  get mode(): YMode {
    return this.morphTarget === 1 ? "fleet" : "score";
  }

  /** How far through the morph we are. Exposed for tests and verification. */
  get morphT(): number {
    return this.morph;
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

  /** Frame the window on everything currently on the surface.
   *
   *  The control that makes fleet mode usable at all: several runs captured at
   *  different times have no common window, and "follow the live edge" shows an
   *  empty chart when none of them is live. */
  fitAll(): void {
    // Pulls the source rather than trusting `this.input`, which is only
    // refreshed by `draw()`. A fit issued before the first frame — the obvious
    // way to open on an old run — would otherwise silently do nothing.
    this.input = this.source();
    let lo = Infinity;
    let hi = -Infinity;
    for (const r of this.input.runs) {
      for (const s of r.view?.spans ?? []) {
        lo = Math.min(lo, s.started_at);
        hi = Math.max(hi, s.ended_at ?? s.started_at);
      }
      for (const s of r.openSpans) {
        lo = Math.min(lo, s.startedAt);
        hi = Math.max(hi, this.now());
      }
      for (const m of r.marks) {
        lo = Math.min(lo, m.ts);
        hi = Math.max(hi, m.ts);
      }
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return;
    const pad = Math.max(1, (hi - lo) * 0.06);
    this.windowS = clamp(hi - lo + pad * 2, MIN_WINDOW_S, MAX_WINDOW_S);
    this.onWindowChange?.(this.windowS);
    this.setFollow(false);
    this.edge = hi + pad;
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
    let newest = Date.now() / 1000;
    for (const r of this.input.runs) {
      const m = r.marks;
      if (m.length) newest = Math.max(newest, m[m.length - 1]!.ts);
      for (const s of r.openSpans) newest = Math.max(newest, s.startedAt);
    }
    return newest;
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

  /** Draws one frame synchronously. `dtS` is passed in by tests so a morph can
   *  be stepped deterministically; live frames measure it themselves. */
  draw(dtS?: number): void {
    const ctx = this.ctx;
    if (!ctx || this.w <= 0 || this.h <= 0) return;
    this.input = this.source();
    this.advanceMorph(dtS);

    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    this.hits = [];

    const edge = this.currentEdge();
    const t0 = edge - this.windowS;
    const plot = Math.max(1, this.w - GUTTER);
    const x = (t: number): number => GUTTER + ((t - t0) / this.windowS) * plot;
    const geo = this.geometry();

    this.drawAxis(ctx, edge, t0, x, plot);
    this.drawLanes(ctx, geo);

    for (let i = 0; i < this.input.runs.length; i++) {
      const run = this.input.runs[i]!;
      for (const s of run.view?.spans ?? []) {
        this.drawClosedSpan(ctx, run, i, s, x, geo, t0, edge);
      }
      for (const s of run.openSpans) this.drawOpenSpan(ctx, run, i, s, x, geo, edge);
    }
    this.drawRail(ctx, x, geo, t0, edge);
  }

  private advanceMorph(dtS?: number): void {
    const now = typeof performance !== "undefined" ? performance.now() : Date.now();
    const dt = dtS ?? (this.lastFrame ? Math.min(0.1, (now - this.lastFrame) / 1000) : 0);
    this.lastFrame = now;
    if (this.morph === this.morphTarget) return;
    const step = dt / MORPH_S;
    const d = this.morphTarget - this.morph;
    this.morph = Math.abs(d) <= step ? this.morphTarget : this.morph + Math.sign(d) * step;
  }

  // ── geometry ───────────────────────────────────────────────────────────

  /** Both lane layouts, every frame, regardless of mode.
   *
   *  A mark's `y` is the interpolation of its position in each. Holding only
   *  the current layout would force the morph to be a cross-fade, which asserts
   *  that these are two pictures rather than one regrouped. */
  private geometry(): Geometry {
    const avail = Math.max(1, this.h - AXIS_H - RAIL_H);
    const nRuns = Math.max(1, this.input.runs.length);
    return {
      t: this.morph,
      scoreH: clamp(avail / SCORE_LANES.length, LANE_MIN, LANE_MAX),
      fleetH: clamp(avail / nRuns, LANE_MIN, LANE_MAX),
      nRuns,
    };
  }

  private scoreRowY(geo: Geometry, lane: ScoreLane): number {
    return AXIS_H + SCORE_LANES.indexOf(lane) * geo.scoreH;
  }

  private fleetRowY(geo: Geometry, runIdx: number): number {
    return AXIS_H + runIdx * geo.fleetH;
  }

  /** Where a mark sits: its score row and its fleet row, interpolated. */
  private laneY(geo: Geometry, lane: ScoreLane, runIdx: number): number {
    const a = this.scoreRowY(geo, lane);
    const b = this.fleetRowY(geo, runIdx);
    return a + (b - a) * geo.t;
  }

  private laneH(geo: Geometry): number {
    return geo.scoreH + (geo.fleetH - geo.scoreH) * geo.t;
  }

  private railY(geo: Geometry): number {
    const a = AXIS_H + SCORE_LANES.length * geo.scoreH;
    const b = AXIS_H + geo.nRuns * geo.fleetH;
    return a + (b - a) * geo.t + 2;
  }

  // ── chrome ─────────────────────────────────────────────────────────────

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

  /** Lane backgrounds and names for both layouts, each faded by how far the
   *  morph has travelled away from it.
   *
   *  The chrome stays *put* and fades; only marks travel. A row labelled
   *  `verify` sliding down to become the row for `run_a6d2` would be the one
   *  outright lie in the picture — the marks are the same events regrouped, the
   *  names are not the same names. */
  private drawLanes(ctx: CanvasRenderingContext2D, geo: Geometry): void {
    ctx.save();
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textBaseline = "middle";

    if (geo.t < 1) {
      const a = 1 - geo.t;
      for (const lane of SCORE_LANES) {
        const y = this.scoreRowY(geo, lane);
        ctx.fillStyle = `rgba(255,255,255,${0.014 * a})`;
        ctx.fillRect(GUTTER, y, this.w - GUTTER, geo.scoreH - 1);
        ctx.fillStyle = laneInk(lane, 0.78 * a);
        ctx.fillText(lane, 6, y + geo.scoreH / 2);
      }
    }

    if (geo.t > 0) {
      for (let i = 0; i < this.input.runs.length; i++) {
        const run = this.input.runs[i]!;
        const y = this.fleetRowY(geo, i);
        ctx.fillStyle = `rgba(255,255,255,${0.014 * geo.t})`;
        ctx.fillRect(GUTTER, y, this.w - GUTTER, geo.fleetH - 1);
        // The row's name is inked by the run's state, so a fleet of rows shows
        // who is stalled without anyone reading a word of it.
        ctx.fillStyle = hexA(stateInk(run.state), 0.85 * geo.t);
        ctx.fillText(run.label.slice(0, 12), 6, y + geo.fleetH / 2);
      }
    }
    ctx.restore();
  }

  // ── marks ──────────────────────────────────────────────────────────────

  private drawClosedSpan(
    ctx: CanvasRenderingContext2D,
    run: LiveRun,
    runIdx: number,
    s: SpanRecord,
    x: (t: number) => number,
    geo: Geometry,
    t0: number,
    edge: number,
  ): void {
    const action = (s.action as Action | null) ?? null;
    const end = s.ended_at ?? s.started_at;
    if (end < t0 || s.started_at > edge) return;
    const y = this.laneY(geo, action ?? "unclassified", runIdx) + 2;
    const barH = Math.max(3, this.laneH(geo) - 6);
    const fid = s.duration_fidelity;
    const ink = s.failed ? "#ff5c7a" : markInk(action, fid);
    const hover = {
      runId: run.runId,
      runLabel: run.label,
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
    run: LiveRun,
    runIdx: number,
    s: OpenSpan,
    x: (t: number) => number,
    geo: Geometry,
    edge: number,
  ): void {
    const lane: ScoreLane = s.reasoning ? "thinking" : (s.action ?? "unclassified");
    const y = this.laneY(geo, lane, runIdx) + 2;
    const barH = Math.max(3, this.laneH(geo) - 6);
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
    if (producing) {
      const a = this.reducedMotion ? 0.8 : 0.45 + 0.35 * Math.sin(Date.now() / 140);
      ctx.fillStyle = hexA(ink, a);
      ctx.fillRect(xb - 2, y - 1, 2, barH + 2);
    }

    this.hits.push({
      x0: xa,
      x1: xb,
      y0: y,
      y1: y + barH,
      hover: {
        runId: run.runId,
        runLabel: run.label,
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
    geo: Geometry,
    t0: number,
    edge: number,
  ): void {
    const y = this.railY(geo);
    ctx.save();
    ctx.fillStyle = "rgba(255,255,255,0.03)";
    ctx.fillRect(GUTTER, y, this.w - GUTTER, RAIL_H - 3);
    for (const run of this.input.runs) {
      for (const m of run.marks) {
        if (m.ts < t0 || m.ts > edge) continue;
        ctx.fillStyle = hexA(markInk(m.action, m.fidelity), 0.62);
        ctx.fillRect(Math.round(x(m.ts)), y, 1, RAIL_H - 3);
      }
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

function laneInk(lane: ScoreLane, alpha: number): string {
  if (lane === "thinking") return `rgba(176,166,240,${alpha})`;
  if (lane === "unclassified") return hexA(NEUTRAL_INK, alpha);
  return hexA(ACTION_COLOR[lane], alpha);
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
