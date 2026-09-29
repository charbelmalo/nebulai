/** handOverlay.ts — the operator's view of what the hand rig currently thinks.
 *
 *  Drawn on a 2D canvas over the mirrored camera preview rather than built out
 *  of DOM nodes. That is not a performance reflex: the skeleton is 21 points and
 *  21 bones per hand redrawn at display cadence, and the DOM version of it would
 *  be 42 elements whose transforms are rewritten every frame — which is both
 *  slower and, more importantly, invisible to the one thing this panel exists
 *  for. The panel is a debugging instrument first and a decoration second: when
 *  a gesture will not fire, the answer is almost always visible here (the wrong
 *  role colour, a charge ring that never fills, a skeleton that snaps to a
 *  different hand), and that only works if what is drawn is exactly what the
 *  channel layer read.
 *
 *  Every colour comes from the shared chart theme, which is itself sampled from
 *  tokens.css. Nothing here invents a literal.
 */

import {
  ACCENT,
  HOT,
  SUCCESS,
  TEXT_DIM_RGB,
  TEXT_FAINT_RGB,
  TEXT_RGB,
  rampRgb,
  type RGB,
} from "@psychix/viz/chart-theme";
import type { HandOverlay } from "../hands/tracking";
import { HAND_CONNECTIONS, SPELL_LABEL, type HandRole, type SpellId } from "../hands/types";

/** Seconds a cast ring stays up. Must match the tracker's own retention. */
const CAST_SECONDS = 0.9;

/** How far a cast ring travels, as a fraction of the preview's width. */
const CAST_MAX_RADIUS = 0.42;

function rgba(color: RGB, alpha: number): string {
  return `rgba(${color[0]},${color[1]},${color[2]},${alpha})`;
}

/**
 * Role → colour.
 *
 * The colours are the point of the skeleton. A hand whose bones are grey is a
 * hand the channel layer has decided is doing nothing, which is the single most
 * common reason a gesture "does not work" — and it is unanswerable from a
 * monochrome skeleton.
 */
const ROLE_COLOR: Record<HandRole, RGB> = {
  light: ACCENT,
  charge: HOT,
  // `grab` is a recognised-and-inert role: a pinch drives nothing, and is
  // classified only so it is not mistaken for an open palm. It keeps a colour of
  // its own rather than sharing `idle`'s grey, because "the rig can see you are
  // pinching and that does nothing here" and "the rig cannot see your hand" are
  // different answers to the operator's question.
  grab: SUCCESS,
  point: rampRgb(0.55),
  hold: TEXT_DIM_RGB,
  idle: TEXT_FAINT_RGB,
};

const SPELL_COLOR: Record<SpellId, RGB> = {
  shockwave: HOT,
  snap: rampRgb(0.25),
};

export interface OverlayStyle {
  /** Suppresses the expanding cast rings and the charge sweep's motion. */
  reducedMotion: boolean;
}

/**
 * Resizes the backing store to the element's CSS box, in device pixels.
 *
 * Returns the CSS-pixel size, which is the space everything below draws in: the
 * transform set here means the drawing code never multiplies by the pixel ratio,
 * so a stroke width of 2 is 2 CSS pixels on every display.
 */
function fit(
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
): { width: number; height: number } {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const backingWidth = Math.max(1, Math.round(width * ratio));
  const backingHeight = Math.max(1, Math.round(height * ratio));
  if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
    canvas.width = backingWidth;
    canvas.height = backingHeight;
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { width, height };
}

export function drawHandOverlay(
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  overlay: HandOverlay,
  style: OverlayStyle,
): void {
  const { width, height } = fit(canvas, ctx);
  if (width < 2 || height < 2) return;

  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  for (const hand of overlay.hands) {
    const color = ROLE_COLOR[hand.role];
    const points = hand.points;

    ctx.strokeStyle = rgba(color, 0.7);
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (const [from, to] of HAND_CONNECTIONS) {
      const a = points[from];
      const b = points[to];
      if (!a || !b) continue;
      ctx.moveTo(a.x * width, a.y * height);
      ctx.lineTo(b.x * width, b.y * height);
    }
    ctx.stroke();

    ctx.fillStyle = rgba(color, 0.95);
    for (const point of points) {
      ctx.beginPath();
      ctx.arc(point.x * width, point.y * height, 2.2, 0, Math.PI * 2);
      ctx.fill();
    }

    // The palm ring sits on landmark 9 — the middle-finger knuckle, which is
    // what `features.ts` reports as the palm centre. Drawing it anywhere else,
    // however visually nicer, would make the ring a lie about where the rig
    // thinks the hand is, and every velocity and every spell origin is measured
    // from that exact point.
    const palm = points[9];
    if (palm) {
      ctx.strokeStyle = rgba(color, 0.5);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(palm.x * width, palm.y * height, 13, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // The stroke being drawn right now, so a lasso can be aimed while it is drawn
  // rather than judged after it closes.
  if (overlay.stroke.length > 1) {
    ctx.strokeStyle = rgba(rampRgb(0.55), 0.85);
    ctx.lineWidth = 2;
    ctx.beginPath();
    const first = overlay.stroke[0];
    if (first) {
      ctx.moveTo(first.x * width, first.y * height);
      for (const point of overlay.stroke) ctx.lineTo(point.x * width, point.y * height);
      ctx.stroke();
    }
  }

  drawCharge(ctx, overlay, width, height);
  drawCasts(ctx, overlay, width, height, style);
}

/**
 * The charge arc.
 *
 * An arc rather than a bar because charge is gathered *at the hand*, and a
 * readout somewhere else in the panel makes the operator look away from the
 * thing they are holding still. The crosshairs at full are the commit signal:
 * charge saturates at 1 and then simply stops changing, so without a distinct
 * mark for "full" the last 10% and the plateau look identical.
 */
function drawCharge(
  ctx: CanvasRenderingContext2D,
  overlay: HandOverlay,
  width: number,
  height: number,
): void {
  const at = overlay.chargeAt;
  if (!at || overlay.charge <= 0.01) return;

  const x = at.x * width;
  const y = at.y * height;
  const radius = 22;
  const full = overlay.charge >= 0.995;

  ctx.strokeStyle = rgba(HOT, 0.22);
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.stroke();

  ctx.strokeStyle = rgba(HOT, full ? 1 : 0.85);
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(x, y, radius, -Math.PI / 2, -Math.PI / 2 + overlay.charge * Math.PI * 2);
  ctx.stroke();

  if (!full) return;
  ctx.strokeStyle = rgba(HOT, 0.9);
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x - radius - 6, y);
  ctx.lineTo(x - radius + 4, y);
  ctx.moveTo(x + radius - 4, y);
  ctx.lineTo(x + radius + 6, y);
  ctx.moveTo(x, y - radius - 6);
  ctx.lineTo(x, y - radius + 4);
  ctx.moveTo(x, y + radius - 4);
  ctx.lineTo(x, y + radius + 6);
  ctx.stroke();
}

/**
 * Expanding rings for each recent cast, named.
 *
 * The name is not decoration. Six of the seven spells change something outside
 * the preview — a cluster, a toggle, the camera — and an operator who fires one
 * by accident has no way to know *which* one fired unless the rig says so at the
 * moment and place it happened.
 */
function drawCasts(
  ctx: CanvasRenderingContext2D,
  overlay: HandOverlay,
  width: number,
  height: number,
  style: OverlayStyle,
): void {
  ctx.font = "600 10px var(--font-ui, system-ui), system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  for (const cast of overlay.casts) {
    const life = Math.min(1, Math.max(0, cast.age / CAST_SECONDS));
    const color = SPELL_COLOR[cast.id];
    const x = cast.x * width;
    const y = cast.y * height;

    // With motion suppressed the ring holds at its mid radius and only fades:
    // the event still announces itself, without something sweeping outward
    // across the operator's own image of their hand.
    const grow = style.reducedMotion ? 0.5 : life;
    const radius = width * CAST_MAX_RADIUS * grow * (0.35 + 0.65 * cast.power);
    const alpha = (1 - life) * (0.35 + 0.65 * cast.power);

    ctx.strokeStyle = rgba(color, alpha);
    ctx.lineWidth = 2 + 2 * cast.power * (1 - life);
    ctx.beginPath();
    ctx.arc(x, y, Math.max(4, radius), 0, Math.PI * 2);
    ctx.stroke();

    const label = SPELL_LABEL[cast.id];
    const labelY = Math.max(10, y - Math.max(4, radius) - 8);
    ctx.fillStyle = rgba(TEXT_RGB, Math.min(1, alpha * 1.6));
    ctx.fillText(label, Math.min(width - 4, Math.max(4, x)), labelY);
  }
}
