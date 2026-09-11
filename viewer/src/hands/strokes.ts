import { clamp } from "./filters";

/**
 * Air-drawn stroke capture, and its resolution into a lasso.
 *
 * The reference rig this is ported from resolved a stroke through a $P
 * point-cloud recognizer against a fixed template vocabulary, because its host
 * had a closed list of six specimens and drawing a circle meant "show me the
 * rosette". The atlas has no such list: what a stroke *means* here is a region
 * of the map, and the useful thing to do with a fingertip path is to enclose
 * points with it. So the capture machinery below is the reference's verbatim —
 * the spacing floor, the extent floor, the stillness terminator and the
 * timeout are all load-bearing and all documented — and the template matcher is
 * replaced by a closed polygon plus a point-in-polygon test.
 *
 * Everything is in mirrored-viewport coordinates, both axes 0..1, matching
 * `HandFeatures.indexTip`. The consumer converts to whatever space its picking
 * lives in; doing that conversion here would tie the recognizer to one renderer.
 */

export interface StrokePoint {
  x: number;
  y: number;
}

/** Samples below this spacing are dropped as tremor rather than motion. */
const MIN_SAMPLE_SPACING = 0.004;

/** A stroke shorter than this diagonal is a twitch, not a drawing. */
const MIN_STROKE_EXTENT = 0.1;

/** Fewer points than this cannot enclose an area worth selecting. */
const MIN_STROKE_POINTS = 10;

/** A stroke this long has stopped being a gesture. */
const MAX_STROKE_SECONDS = 4;

export interface StrokeCapture {
  active: boolean;
  points: StrokePoint[];
  elapsed: number;
  /** Seconds the fingertip has been effectively stationary. */
  still: number;
}

export function createStrokeCapture(): StrokeCapture {
  return { active: false, points: [], elapsed: 0, still: 0 };
}

export function beginStroke(capture: StrokeCapture, point: StrokePoint): void {
  capture.active = true;
  capture.points = [{ x: point.x, y: point.y }];
  capture.elapsed = 0;
  capture.still = 0;
}

export function appendStroke(capture: StrokeCapture, point: StrokePoint, dt: number): void {
  if (!capture.active) return;
  capture.elapsed += dt;

  const last = capture.points[capture.points.length - 1];
  if (!last) {
    capture.points.push({ x: point.x, y: point.y });
    return;
  }

  const moved = Math.hypot(point.x - last.x, point.y - last.y);

  if (moved < MIN_SAMPLE_SPACING) {
    capture.still += dt;
    return;
  }

  capture.still = 0;
  capture.points.push({ x: point.x, y: point.y });
}

export function strokeExtent(capture: StrokeCapture): number {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (const point of capture.points) {
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
  }

  if (!Number.isFinite(minX)) return 0;
  return Math.hypot(maxX - minX, maxY - minY);
}

export function strokeExpired(capture: StrokeCapture, stillSeconds: number): boolean {
  return (
    capture.elapsed > MAX_STROKE_SECONDS ||
    (capture.points.length > 1 && capture.still >= stillSeconds)
  );
}

export function strokeIsDrawable(capture: StrokeCapture): boolean {
  return capture.points.length >= MIN_STROKE_POINTS && strokeExtent(capture) >= MIN_STROKE_EXTENT;
}

export function endStroke(capture: StrokeCapture): StrokePoint[] {
  const points = capture.points;
  capture.active = false;
  capture.points = [];
  capture.elapsed = 0;
  capture.still = 0;
  return points;
}

// ---------------------------------------------------------------------------
// Lasso
// ---------------------------------------------------------------------------

export interface Lasso {
  /** The closed polygon, mirrored-viewport coordinates. */
  polygon: readonly StrokePoint[];
  /** Axis-aligned bounds, so a consumer can reject most points with two compares. */
  bounds: { minX: number; minY: number; maxX: number; maxY: number };
  /**
   * How closed the drawn path was, 0..1 — 1 when the finger came back to where
   * it started. Reported rather than enforced: a lasso the operator did not
   * quite close is still the region they meant, and refusing it would make the
   * gesture feel unreliable for a reason they cannot see. The consumer can use
   * it to decide how confidently to announce the selection.
   */
  closure: number;
}

/**
 * Turns a raw fingertip path into a lasso.
 *
 * The path is closed by implication rather than by appending the first point:
 * `pointInLasso` wraps its edge walk, so the closing segment exists in the test
 * without existing in the drawn geometry, and the overlay draws exactly what
 * the operator drew.
 */
export function strokeToLasso(points: readonly StrokePoint[]): Lasso | null {
  if (points.length < MIN_STROKE_POINTS) return null;

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const point of points) {
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
  }
  if (!Number.isFinite(minX)) return null;

  const first = points[0];
  const last = points[points.length - 1];
  if (!first || !last) return null;

  const diagonal = Math.hypot(maxX - minX, maxY - minY);
  const gap = Math.hypot(last.x - first.x, last.y - first.y);
  const closure = diagonal > 1e-6 ? clamp(1 - gap / diagonal, 0, 1) : 0;

  return {
    polygon: points.map((point) => ({ x: point.x, y: point.y })),
    bounds: { minX, minY, maxX, maxY },
    closure,
  };
}

/**
 * Even-odd point-in-polygon (the standard crossing-number test).
 *
 * Even-odd rather than winding on purpose: an air-drawn loop crosses itself
 * constantly — the finger overshoots the start and doubles back — and a winding
 * rule would count those overlaps as *more* inside, which is invisible for a
 * convex loop and wrong for the self-intersecting one an operator actually
 * draws.
 */
export function pointInLasso(lasso: Lasso, x: number, y: number): boolean {
  const { bounds, polygon } = lasso;
  if (x < bounds.minX || x > bounds.maxX || y < bounds.minY || y > bounds.maxY) return false;

  let inside = false;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const a = polygon[index];
    const b = polygon[previous];
    if (!a || !b) continue;
    const straddles = a.y > y !== b.y > y;
    if (!straddles) continue;
    const crossingX = ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x;
    if (x < crossingX) inside = !inside;
  }
  return inside;
}
