/** Picking, both rungs. 2D: exact kdbush lookup over pos2 — replaces the old
 *  compare viewer's O(N) hover loop. 3D (morphed positions): GPU id-buffer —
 *  the points layer's id companion sprite is rendered into an offscreen
 *  target and the pixel under the cursor decoded back to an instance index.
 *  kdbush can't serve 3D because the morph happens on the GPU. */

import * as THREE from "three/webgpu";
import KDBush from "kdbush";

export class PointPicker {
  private index: KDBush;

  constructor(
    private pos2: Float32Array,
    private count: number,
  ) {
    this.index = new KDBush(count);
    for (let i = 0; i < count; i++) {
      this.index.add(pos2[i * 2]!, pos2[i * 2 + 1]!);
    }
    this.index.finish();
  }

  /** Nearest point within worldRadius of (wx, wy), or -1. */
  nearest(wx: number, wy: number, worldRadius: number): number {
    let best = -1;
    let bestD2 = worldRadius * worldRadius;
    for (const i of this.index.within(wx, wy, worldRadius)) {
      const dx = this.pos2[i * 2]! - wx;
      const dy = this.pos2[i * 2 + 1]! - wy;
      const d2 = dx * dx + dy * dy;
      if (d2 <= bestD2) {
        bestD2 = d2;
        best = i;
      }
    }
    return best;
  }
}

/** GPU id-buffer picker for the 3D flythrough. Owns a private scene holding
 *  the id sprite and a 1×1 render target; `pick` renders one id frame with the
 *  caller's camera and async-reads the single pixel under the cursor.
 *  Callers throttle (~30Hz) and guard staleness — a pick that resolves after
 *  a dataset switch must be dropped.
 *
 *  The target is 1×1 rather than viewport-sized on purpose. Only one pixel is
 *  ever read, so rasterizing the whole frame is pure waste — measured at 1.6ms
 *  vs 1.0ms per pick for a 49k-point cloud at 1594×834 on an M4, and it scales
 *  with fill rate, so a weaker GPU pays far more. `camera.setViewOffset` scales
 *  the projection so the 1×1 target *is* the cursor pixel: the sub-rect keeps
 *  the exact projection the main render uses, so the id under the cursor is
 *  identical to what the visual pass drew there. */
export class IdPicker {
  private scene = new THREE.Scene();
  private rt: THREE.RenderTarget;
  /** full viewport in CSS px — the frame `setViewOffset` subdivides */
  private viewW = 1;
  private viewH = 1;
  /** readback failed (backend without readRenderTargetPixelsAsync support) —
   *  callers should stop asking */
  broken = false;

  constructor(
    private renderer: THREE.WebGPURenderer,
    idObject: THREE.Object3D,
  ) {
    this.scene.add(idObject);
    this.rt = new THREE.RenderTarget(1, 1, { depthBuffer: false });
  }

  setSize(w: number, h: number): void {
    this.viewW = Math.max(Math.round(w), 1);
    this.viewH = Math.max(Math.round(h), 1);
  }

  /** Instance index under CSS pixel (sx, sy), or -1 for background. */
  async pick(camera: THREE.Camera, sx: number, sy: number): Promise<number> {
    if (this.broken) return -1;
    const x = Math.min(Math.max(Math.round(sx), 0), this.viewW - 1);
    const y = Math.min(Math.max(Math.round(sy), 0), this.viewH - 1);

    // Narrow the projection to the single cursor pixel. setViewOffset uses a
    // top-left origin, matching the CSS coords callers hand us. Restoring the
    // previous view (rather than clearViewOffset) keeps this transparent to a
    // caller that is itself using a view offset.
    const view = (camera as THREE.PerspectiveCamera).view;
    const prevView = view?.enabled ? { ...view } : null;
    (camera as THREE.PerspectiveCamera).setViewOffset(this.viewW, this.viewH, x, y, 1, 1);

    const prevTarget = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(this.rt);
    this.renderer.render(this.scene, camera);
    this.renderer.setRenderTarget(prevTarget);

    // Restore before the await so the main render pass — which runs later in
    // this same frame — never sees the 1×1 projection.
    if (prevView) {
      (camera as THREE.PerspectiveCamera).setViewOffset(
        prevView.fullWidth,
        prevView.fullHeight,
        prevView.offsetX,
        prevView.offsetY,
        prevView.width,
        prevView.height,
      );
    } else {
      (camera as THREE.PerspectiveCamera).clearViewOffset();
    }

    try {
      // The target is the cursor pixel, so the readback origin is (0,0) on both
      // backends — the old bottom-left/top-left flip no longer applies.
      const px = (await this.renderer.readRenderTargetPixelsAsync(this.rt, 0, 0, 1, 1)) as Uint8Array;
      return px[0]! + px[1]! * 256 + px[2]! * 65536 - 1;
    } catch {
      this.broken = true;
      return -1;
    }
  }

  dispose(): void {
    this.rt.dispose();
    this.scene.clear();
  }
}
