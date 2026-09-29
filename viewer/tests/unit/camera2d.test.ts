import { describe, expect, it } from "vitest";
import { Camera2D, centerForTarget, easeInOutCubic } from "../../src/scene/camera2d";

function makeCam(): Camera2D {
  const cam = new Camera2D();
  cam.setViewport(1000, 800);
  return cam;
}

describe("Camera2D", () => {
  it("round-trips world ↔ screen", () => {
    const cam = makeCam();
    cam.cx = 3;
    cam.cy = -2;
    cam.wpp = 0.05;
    const [sx, sy] = cam.worldToScreen(4.2, -1.1);
    const [wx, wy] = cam.screenToWorld(sx, sy);
    expect(wx).toBeCloseTo(4.2, 10);
    expect(wy).toBeCloseTo(-1.1, 10);
    // viewport center maps to (cx, cy)
    expect(cam.screenToWorld(500, 400)).toEqual([3, -2]);
  });

  it("fitBounds centers and fits the larger axis with padding", () => {
    const cam = makeCam();
    cam.fitBounds(-10, -5, 10, 5, 100);
    expect(cam.cx).toBe(0);
    expect(cam.cy).toBe(0);
    // width 20 over (1000 - 200) px vs height 10 over (800 - 200) px
    expect(cam.wpp).toBeCloseTo(Math.max(20 / 800, 10 / 600), 10);
    const [sx] = cam.worldToScreen(-10, 0);
    expect(sx).toBeGreaterThanOrEqual(99.9); // padding respected
  });

  it("zoomAt keeps the world point under the cursor fixed", () => {
    const cam = makeCam();
    cam.fitBounds(-10, -10, 10, 10);
    const [wx, wy] = cam.screenToWorld(200, 650);
    cam.zoomAt(200, 650, 0.5);
    const [wx2, wy2] = cam.screenToWorld(200, 650);
    expect(wx2).toBeCloseTo(wx, 10);
    expect(wy2).toBeCloseTo(wy, 10);
    expect(cam.wpp).toBeCloseTo((20 / (800 - 96)) * 0.5, 10);
  });

  it("panPixels moves the view opposite to the drag, y flipped", () => {
    const cam = makeCam();
    cam.wpp = 0.01;
    cam.panPixels(50, -30); // drag right+up
    expect(cam.cx).toBeCloseTo(-0.5, 10);
    expect(cam.cy).toBeCloseTo(-0.3, 10);
  });

  it("flyTo eases to the target and reports completion", () => {
    const cam = makeCam();
    cam.cx = 0;
    cam.cy = 0;
    cam.wpp = 0.1;
    cam.flyTo(10, 20, 0.01, 1000, 400);
    expect(cam.isFlying).toBe(true);

    cam.update(1200); // halfway
    expect(cam.cx).toBeCloseTo(5, 10); // easeInOutCubic(0.5) = 0.5
    expect(cam.cy).toBeCloseTo(10, 10);
    // zoom interpolates in log space
    expect(cam.wpp).toBeCloseTo(Math.sqrt(0.1 * 0.01), 10);

    const moving = cam.update(1400);
    expect(moving).toBe(true);
    expect(cam.isFlying).toBe(false);
    expect(cam.cx).toBe(10);
    expect(cam.cy).toBe(20);
    expect(cam.wpp).toBeCloseTo(0.01, 10);
    expect(cam.update(1500)).toBe(false);
  });

  it("interaction cancels an active tween", () => {
    const cam = makeCam();
    cam.flyTo(10, 10, 0.01, 0);
    cam.panPixels(1, 1);
    expect(cam.isFlying).toBe(false);
  });

  it("clamps zoom to the wpp bounds", () => {
    const cam = makeCam();
    cam.wpp = cam.minWpp;
    cam.zoomAt(500, 400, 0.01);
    expect(cam.wpp).toBe(cam.minWpp);
    cam.wpp = cam.maxWpp;
    cam.zoomAt(500, 400, 100);
    expect(cam.wpp).toBe(cam.maxWpp);
  });

  it("easeInOutCubic hits the anchor values", () => {
    expect(easeInOutCubic(0)).toBe(0);
    expect(easeInOutCubic(0.5)).toBe(0.5);
    expect(easeInOutCubic(1)).toBe(1);
    // symmetric: e(t) + e(1-t) = 1
    expect(easeInOutCubic(0.2) + easeInOutCubic(0.8)).toBeCloseTo(1, 10);
  });
});

/** The orbit frame, rebuilt from scratch.
 *
 *  These take the two expressions AtlasDriver#frame literally sets on the
 *  three camera — its position on the (az, el) sphere and its up vector — and
 *  derive the screen basis the way three's own lookAt does, by cross products.
 *  Nothing here is imported from camera2d, so a slip shared with the code under
 *  test cannot make the two agree for the wrong reason: if `centerForTarget` or
 *  `zoomAtInFrame` disagrees with the matrix the GPU actually renders with,
 *  these fail. `camDist` is arbitrary — the projection is orthographic, so the
 *  distance along the view axis cannot reach the screen position. */
const CAM_DIST = 30;

type V3 = [number, number, number];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const norm = (a: V3): V3 => {
  const L = Math.hypot(...a);
  return [a[0] / L, a[1] / L, a[2] / L];
};

/** Where the GPU puts world point `p`, in CSS pixels, for a camera centered on
 *  (cam.cx, cam.cy) and orbited to (az, el). */
function projectOrbit(cam: Camera2D, p: V3, az: number, el: number): [number, number] {
  const target: V3 = [cam.cx, cam.cy, 0];
  const pos: V3 = [
    cam.cx + Math.sin(az) * Math.sin(el) * CAM_DIST,
    cam.cy - Math.cos(az) * Math.sin(el) * CAM_DIST,
    Math.cos(el) * CAM_DIST,
  ];
  const up: V3 = [
    -Math.cos(el) * Math.sin(az),
    Math.cos(el) * Math.cos(az),
    Math.sin(el),
  ];
  const z = norm(sub(pos, target)); // three's camera looks down -Z
  const x = norm(cross(up, z));
  const y = cross(z, x);
  const d = sub(p, target);
  return [
    cam.viewportW / 2 + dot(d, x) / cam.wpp,
    cam.viewportH / 2 - dot(d, y) / cam.wpp,
  ];
}

/** A spread of frames worth checking: az past every quadrant boundary and a
 *  sign change, el from flat to the app's ~83° cap. */
const AZ = [0, 0.4, 1.2, 2.5, 3.9, 5.6, -1.1];
const EL = [0, 0.35, 0.663, 1.1, 1.45];

describe("Camera2D in the orbit frame", () => {
  it("centerForTarget lands the point on the viewport center at every frame", () => {
    for (const az of AZ) {
      for (const el of EL) {
        for (const p of [[3, -2, 1.7], [-8, 4, -3.1], [0, 0, 9], [5, 5, 0]] as V3[]) {
          const cam = makeCam();
          cam.wpp = 0.02;
          [cam.cx, cam.cy] = centerForTarget(p[0], p[1], p[2], az, el);
          const [sx, sy] = projectOrbit(cam, p, az, el);
          expect(sx).toBeCloseTo(500, 6);
          expect(sy).toBeCloseTo(400, 6);
        }
      }
    }
  });

  it("centerForTarget is the identity on a flat map", () => {
    for (const az of AZ) {
      expect(centerForTarget(3, -2, 99, az, 0)).toEqual([3, -2]);
    }
  });

  it("aiming at the raw xy is what left fly-to off-screen", () => {
    // The bug this replaced: cam.cx/cy name a point on the z = 0 plane, so a
    // point pz above it is carried up-screen by pz·sin(el) world units. What
    // made that fatal rather than untidy is the fly-to's own framing — it zooms
    // to a window 6% of the cloud's extent, so a point a fifth of the extent
    // deep is displaced by more than a viewport. On the real atlas the worst of
    // five sampled nodes came out 2,144 px off center, and three of the five
    // landed off-screen entirely.
    const extent3 = 100;
    const az = 1.2;
    const el = 0.663; // the app's default tilt at morph = 1, no user elevation
    const cam = makeCam();
    cam.wpp = (extent3 * 0.06) / (800 * 0.55); // flyToPoint's own framing
    const p: V3 = [4, 1, extent3 * 0.2];
    [cam.cx, cam.cy] = [p[0], p[1]]; // the old, tilt-blind target
    const [, sy] = projectOrbit(cam, p, az, el);
    expect(400 - sy).toBeCloseTo((p[2] * Math.sin(el)) / cam.wpp, 6);
    expect(400 - sy).toBeGreaterThan(800); // more than a full viewport of it
    expect(sy).toBeLessThan(0); // off the top edge
    // and the fix puts it dead center at that same framing
    [cam.cx, cam.cy] = centerForTarget(p[0], p[1], p[2], az, el);
    expect(projectOrbit(cam, p, az, el)[1]).toBeCloseTo(400, 6);
  });

  it("zoomAtInFrame holds the ground point under the cursor at every frame", () => {
    for (const az of AZ) {
      for (const el of EL) {
        for (const factor of [0.5, 0.87, 1.0, 1.3, 2.4]) {
          const cam = makeCam();
          cam.wpp = 0.02;
          cam.cx = -1.5;
          cam.cy = 3;
          const [sx, sy] = [317, 612];
          // the ground point under the cursor, read off the real projection by
          // inverting it on the z=0 plane
          const before = groundUnder(cam, sx, sy, az, el);
          cam.zoomAtInFrame(sx, sy, factor, az, el);
          const [ax, ay] = projectOrbit(cam, [...before, 0] as V3, az, el);
          expect(ax).toBeCloseTo(sx, 6);
          expect(ay).toBeCloseTo(sy, 6);
        }
      }
    }
  });

  it("zoomAtInFrame does not accumulate drift over a wheel drain", () => {
    // A wheel tick drains as ~10 partial steps over 120 ms; the old path drifted
    // a little on each one, so the map crept away mid-gesture.
    const az = 2.5;
    const el = 0.9;
    const cam = makeCam();
    cam.wpp = 0.02;
    const [sx, sy] = [880, 140];
    const anchor = groundUnder(cam, sx, sy, az, el);
    for (let i = 0; i < 10; i++) cam.zoomAtInFrame(sx, sy, Math.exp(-0.06), az, el);
    const [ax, ay] = projectOrbit(cam, [...anchor, 0] as V3, az, el);
    expect(ax).toBeCloseTo(sx, 6);
    expect(ay).toBeCloseTo(sy, 6);
  });

  it("reduces to zoomAt at az = el = 0", () => {
    const a = makeCam();
    const b = makeCam();
    for (const [sx, sy, f] of [[200, 650, 0.5], [1000, 0, 1.7], [0, 800, 0.93]]) {
      a.zoomAt(sx!, sy!, f!);
      b.zoomAtInFrame(sx!, sy!, f!, 0, 0);
      expect(b.cx).toBeCloseTo(a.cx, 12);
      expect(b.cy).toBeCloseTo(a.cy, 12);
      expect(b.wpp).toBe(a.wpp);
    }
  });

  it("a clamped step anchors on the movement that happened, not the one asked for", () => {
    const cam = makeCam();
    cam.wpp = cam.minWpp;
    const before = { cx: cam.cx, cy: cam.cy };
    cam.zoomAtInFrame(0, 0, 0.01, 1.2, 0.7); // all of it swallowed by the clamp
    expect(cam.wpp).toBe(cam.minWpp);
    expect(cam.cx).toBe(before.cx); // k = 0, so no pan at all
    expect(cam.cy).toBe(before.cy);
  });

  it("zoomAtInFrame cancels an active tween", () => {
    const cam = makeCam();
    cam.flyTo(10, 10, 0.01, 0);
    cam.zoomAtInFrame(100, 100, 0.9, 1.2, 0.7);
    expect(cam.isFlying).toBe(false);
  });
});

/** Invert `projectOrbit` onto the z = 0 plane: the world point the cursor is
 *  over. Solved from the basis, not from camera2d's own algebra. */
function groundUnder(
  cam: Camera2D,
  sx: number,
  sy: number,
  az: number,
  el: number,
): [number, number] {
  const a = (sx - cam.viewportW / 2) * cam.wpp;
  const b = (cam.viewportH / 2 - sy) * cam.wpp;
  // a = d·right, b = d·up with d = (dx, dy, 0): two equations, two unknowns
  const cosAz = Math.cos(az);
  const sinAz = Math.sin(az);
  const bb = b / Math.cos(el);
  return [cam.cx + a * cosAz - bb * sinAz, cam.cy + a * sinAz + bb * cosAz];
}
