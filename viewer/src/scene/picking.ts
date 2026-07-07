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
 *  the id sprite and a CSS-pixel-sized render target; `pick` renders one id
 *  frame with the caller's camera and async-reads the pixel under the cursor.
 *  Callers throttle (~30Hz) and guard staleness — a pick that resolves after
 *  a dataset switch must be dropped. */
export class IdPicker {
  private scene = new THREE.Scene();
  private rt: THREE.RenderTarget;
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
    this.rt.setSize(Math.max(Math.round(w), 1), Math.max(Math.round(h), 1));
  }

  /** Instance index under CSS pixel (sx, sy), or -1 for background. */
  async pick(camera: THREE.Camera, sx: number, sy: number): Promise<number> {
    if (this.broken) return -1;
    const x = Math.min(Math.max(Math.round(sx), 0), this.rt.width - 1);
    // readback origin: top-left on the WebGPU backend, bottom-left (GL
    // convention) on the forceWebGL rung — verified empirically on both
    const isGL = (this.renderer.backend as { isWebGLBackend?: boolean }).isWebGLBackend === true;
    const yTop = Math.min(Math.max(Math.round(sy), 0), this.rt.height - 1);
    const y = isGL ? this.rt.height - 1 - yTop : yTop;
    const prev = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(this.rt);
    this.renderer.render(this.scene, camera);
    this.renderer.setRenderTarget(prev);
    try {
      const px = (await this.renderer.readRenderTargetPixelsAsync(this.rt, x, y, 1, 1)) as Uint8Array;
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
