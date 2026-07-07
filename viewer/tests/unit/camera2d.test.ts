import { describe, expect, it } from "vitest";
import { Camera2D, easeInOutCubic } from "../../src/scene/camera2d";

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
