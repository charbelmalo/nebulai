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

/** CSS px of the viewport hidden behind chrome panels on each side. Fits and
 *  fly-tos frame the free rectangle between them instead of the whole canvas,
 *  so a map is never centred under the rail that covers it. */
export interface ViewInsets {
  l: number;
  r: number;
  t: number;
  b: number;
}

export const NO_INSETS: ViewInsets = Object.freeze({ l: 0, r: 0, t: 0, b: 0 });

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

  /** Frame a world-space AABB with paddingPx of margin on every side of the
   *  free rectangle the insets leave. */
  fitBounds(
    minX: number,
    minY: number,
    maxX: number,
    maxY: number,
    paddingPx = 48,
    insets: ViewInsets = NO_INSETS,
  ): void {
    const [cx, cy, wpp] = this.fitFor(minX, minY, maxX, maxY, paddingPx, insets);
    this.cx = cx;
    this.cy = cy;
    this.wpp = wpp;
    this.tween = null;
  }

  /** The camera (cx, cy, wpp) that fitBounds would settle on — for flyTo. */
  fitFor(
    minX: number,
    minY: number,
    maxX: number,
    maxY: number,
    paddingPx = 48,
    insets: ViewInsets = NO_INSETS,
  ): [number, number, number] {
    const w = Math.max(maxX - minX, 1e-9);
    const h = Math.max(maxY - minY, 1e-9);
    const availW = Math.max(this.freeW(insets) - paddingPx * 2, 1);
    const availH = Math.max(this.freeH(insets) - paddingPx * 2, 1);
    const wpp = this.clampWpp(Math.max(w / availW, h / availH));
    const [cx, cy] = this.centerFor((minX + maxX) / 2, (minY + maxY) / 2, wpp, insets);
    return [cx, cy, wpp];
  }

  freeW(insets: ViewInsets = NO_INSETS): number {
    return Math.max(this.viewportW - insets.l - insets.r, 1);
  }

  freeH(insets: ViewInsets = NO_INSETS): number {
    return Math.max(this.viewportH - insets.t - insets.b, 1);
  }

  /** The camera centre that shows world (x, y) at the middle of the free
   *  rectangle at zoom wpp (screen y grows down, world y up). */
  centerFor(x: number, y: number, wpp: number, insets: ViewInsets = NO_INSETS): [number, number] {
    const dx = (insets.l - insets.r) / 2;
    const dy = (insets.t - insets.b) / 2;
    return [x - dx * wpp, y + dy * wpp];
  }

  panPixels(dxPx: number, dyPx: number): void {
    // screen y grows downward, world y grows upward
    this.cx -= dxPx * this.wpp;
    this.cy += dyPx * this.wpp;
    this.tween = null;
  }

  /** Zoom by `factor` keeping the world point under (sx, sy) fixed on screen.
   *  The flat-map case of `zoomAtInFrame`, which it reduces to term for term. */
  zoomAt(sx: number, sy: number, factor: number): void {
    this.zoomAtInFrame(sx, sy, factor, 0, 0);
  }

  /** Zoom by `factor` keeping the **ground** point under (sx, sy) fixed on
   *  screen, for a map orbited to (az, el).
   *
   *  `zoomAt`'s screenToWorld round-trip assumes the screen axes are the world
   *  axes, which stops being true the moment the orbit carries an azimuth: the
   *  point it holds still is then not the point under the cursor, so every step
   *  slides the view — and a wheel tick drains ~10 steps over ~120 ms, so it
   *  compounds. Measured over one tick at the default tilt: 15 px of drift at
   *  az = 0, up to 280 px once the azimuth is round the far side, with the map
   *  visibly creeping away from whatever the user was zooming into.
   *
   *  Rotate the cursor offset into the camera's ground frame instead.
   *  Camera-right is (cos az, sin az, 0) — exactly in the ground plane at every
   *  azimuth — and the vertical offset picks up a 1/cos el from the
   *  foreshortening of a tilted view, the same factor `panScreen` and
   *  `applyOrbitPivot` already apply. `k` is the scale change that actually
   *  happened rather than the one requested, so a step the wpp clamp swallows
   *  anchors on the real movement instead of overshooting the center. */
  zoomAtInFrame(sx: number, sy: number, factor: number, az: number, el: number): void {
    const prev = this.wpp;
    this.wpp = this.clampWpp(this.wpp * factor);
    const k = prev - this.wpp;
    const ox = sx - this.viewportW / 2;
    // screen y grows downward, world y grows upward — hence the sign flip on
    // the cy term rather than here. el is clamped short of 90° by the caller's
    // orbit frame (at 90° the ground plane is edge-on and there is no answer).
    const oy = (sy - this.viewportH / 2) / Math.cos(el);
    this.cx += k * (ox * Math.cos(az) + oy * Math.sin(az));
    this.cy += k * (ox * Math.sin(az) - oy * Math.cos(az));
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

/** The ground-plane camera center that puts a 3-D world point at the viewport
 *  center, for a map orbited to (az, el).
 *
 *  In the flythrough `cx`/`cy` are not "where the camera looks" — they are the
 *  point on the z = 0 plane the orbit frame is built around, with the camera
 *  itself off along the (az, el) direction. So a point with z != 0 does *not*
 *  land at the viewport center when cx/cy equal its xy: the tilt carries it
 *  up-screen by its own height. Aiming a fly-to straight at a pos3 xy left
 *  search results a median of 365 px off-center and up to 2,844 px, which on an
 *  800 px-tall viewport put 15 of 50 sampled arrivals off-screen entirely.
 *
 *  Camera-right is (cos az, sin az, 0) and camera-up is
 *  (−cos el·sin az, cos el·cos az, sin el); setting both of the point's offsets
 *  along those axes to zero and solving for the center gives the two terms
 *  below. It is closed-form rather than iterative because the projection is
 *  orthographic, and it is exact — checked against a from-scratch rebuild of
 *  the render camera's basis over 140 frames in the unit tests, and to 1e-11 px
 *  through the live camera matrix. At el = 0 it returns the point's own xy, so
 *  a flat map is untouched.
 *
 *  `el` must be clamped short of 90°, as every orbit frame in the app is. */
export function centerForTarget(
  px: number,
  py: number,
  pz: number,
  az: number,
  el: number,
): [number, number] {
  const lift = pz * Math.tan(el);
  return [px - lift * Math.sin(az), py + lift * Math.cos(az)];
}
