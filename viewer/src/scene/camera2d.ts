/** 2D map camera: center + world-units-per-pixel, cursor-anchored zoom, eased
 *  flyTo. Pure math (no three) so the tween/projection behavior is unit-testable;
 *  AtlasDriver copies this state into its OrthographicCamera each frame. The
 *  HTML/SVG overlays project through worldToScreen so they can never drift from
 *  the GPU scene. */

export interface CameraTween {
  fromX: number;
  fromY: number;
  fromWpp: number;
  toX: number;
  toY: number;
  toWpp: number;
  start: number;
  duration: number;
}

export function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

export class Camera2D {
  /** world coords at the viewport center */
  cx = 0;
  cy = 0;
  /** world units per CSS pixel (zoom; smaller = closer) */
  wpp = 0.01;

  viewportW = 1;
  viewportH = 1;

  /** syncatlas moves are cinematic; reduced motion keeps them near-instant */
  flyMs = 450;
  reducedFlyMs = 150;
  reducedMotion = false;

  minWpp = 1e-5;
  maxWpp = 10;

  private tween: CameraTween | null = null;

  setViewport(w: number, h: number): void {
    this.viewportW = Math.max(w, 1);
    this.viewportH = Math.max(h, 1);
  }

  /** Frame a world-space AABB with paddingPx of margin on every side. */
  fitBounds(minX: number, minY: number, maxX: number, maxY: number, paddingPx = 48): void {
    const w = Math.max(maxX - minX, 1e-9);
    const h = Math.max(maxY - minY, 1e-9);
    const availW = Math.max(this.viewportW - paddingPx * 2, 1);
    const availH = Math.max(this.viewportH - paddingPx * 2, 1);
    this.cx = (minX + maxX) / 2;
    this.cy = (minY + maxY) / 2;
    this.wpp = this.clampWpp(Math.max(w / availW, h / availH));
    this.tween = null;
  }

  panPixels(dxPx: number, dyPx: number): void {
    // screen y grows downward, world y grows upward
    this.cx -= dxPx * this.wpp;
    this.cy += dyPx * this.wpp;
    this.tween = null;
  }

  /** Zoom by `factor` keeping the world point under (sx, sy) fixed on screen. */
  zoomAt(sx: number, sy: number, factor: number): void {
    const [wx, wy] = this.screenToWorld(sx, sy);
    this.wpp = this.clampWpp(this.wpp * factor);
    const [nx, ny] = this.screenToWorld(sx, sy);
    this.cx += wx - nx;
    this.cy += wy - ny;
    this.tween = null;
  }

  flyTo(cx: number, cy: number, wpp: number, now: number, duration?: number): void {
    this.tween = {
      fromX: this.cx,
      fromY: this.cy,
      fromWpp: this.wpp,
      toX: cx,
      toY: cy,
      toWpp: this.clampWpp(wpp),
      start: now,
      duration: duration ?? (this.reducedMotion ? this.reducedFlyMs : this.flyMs),
    };
  }

  /** Advance any active tween. Returns true while the camera is moving. */
  update(now: number): boolean {
    const tw = this.tween;
    if (!tw) return false;
    const t = Math.min((now - tw.start) / tw.duration, 1);
    const e = easeInOutCubic(t);
    this.cx = tw.fromX + (tw.toX - tw.fromX) * e;
    this.cy = tw.fromY + (tw.toY - tw.fromY) * e;
    // interpolate zoom in log space so the motion feels uniform
    this.wpp = Math.exp(
      Math.log(tw.fromWpp) + (Math.log(tw.toWpp) - Math.log(tw.fromWpp)) * e,
    );
    if (t >= 1) this.tween = null;
    return true;
  }

  get isFlying(): boolean {
    return this.tween !== null;
  }

  worldToScreen(x: number, y: number): [number, number] {
    return [
      this.viewportW / 2 + (x - this.cx) / this.wpp,
      this.viewportH / 2 - (y - this.cy) / this.wpp,
    ];
  }

  screenToWorld(sx: number, sy: number): [number, number] {
    return [
      this.cx + (sx - this.viewportW / 2) * this.wpp,
      this.cy - (sy - this.viewportH / 2) * this.wpp,
    ];
  }

  /** Ortho frustum half-extents in world units, for the render camera. */
  halfExtents(): [number, number] {
    return [(this.viewportW / 2) * this.wpp, (this.viewportH / 2) * this.wpp];
  }

  private clampWpp(wpp: number): number {
    return Math.min(Math.max(wpp, this.minWpp), this.maxWpp);
  }
}
