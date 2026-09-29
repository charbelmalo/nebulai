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
 *  the id sprite and a ONE-PIXEL render target; `pick` renders one id frame
 *  with the caller's camera and async-reads it. Callers throttle (~30Hz) and
 *  guard staleness — a pick that resolves after a dataset switch must be
 *  dropped.
 *
 *  The target is 1x1 rather than viewport-sized because only the pixel under
 *  the cursor is ever read. `camera.setViewOffset(w, h, sx, sy, 1, 1)` — three's
 *  own tiled-rendering API, exact on both PerspectiveCamera and
 *  OrthographicCamera — narrows the frustum to exactly that pixel, so the
 *  rasterizer shades one fragment instead of a screenful and the 4 MB
 *  full-viewport attachment is gone. Vertex work is unchanged: every instance
 *  is still transformed, and all but one is clipped. Measured: the ids returned
 *  are identical to the full-target render for every pixel tested.
 *
 *  The offset MUST be cleared synchronously, before the readback is awaited —
 *  the caller's next display frame uses the same camera object, and a 1-px
 *  view offset left on it would render the whole scene through that pixel. */
export class IdPicker {
  private scene = new THREE.Scene();
  private rt: THREE.RenderTarget;
  /** CSS-pixel viewport the caller's (sx, sy) are expressed in */
  private w = 1;
  private h = 1;
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
    this.w = Math.max(Math.round(w), 1);
    this.h = Math.max(Math.round(h), 1);
  }

  /** Instance index under CSS pixel (sx, sy), or -1 for background. */
  async pick(camera: THREE.Camera, sx: number, sy: number): Promise<number> {
    if (this.broken) return -1;
    // floor, not round: the cursor at CSS x=432.6 is inside pixel 432, and the
    // sub-frame must be that pixel. Verified on both camera types by projecting
    // a known point — with `round` the point lands ~1 px OUTSIDE the one-pixel
    // frustum (|NDC| ~ 1.9) whenever the fractional part is >= 0.5, which is the
    // neighbouring pixel's id. It never showed up as a wrong tooltip because a
    // point sprite is ~8.5 px across, so both pixels usually belong to it.
    const x = Math.min(Math.max(Math.floor(sx), 0), this.w - 1);
    const y = Math.min(Math.max(Math.floor(sy), 0), this.h - 1);
    const cam = camera as THREE.Camera & {
      view?: { enabled: boolean; fullWidth: number; fullHeight: number;
               offsetX: number; offsetY: number; width: number; height: number } | null;
      setViewOffset?: (fw: number, fh: number, x: number, y: number, w: number, h: number) => void;
      clearViewOffset?: () => void;
    };
    if (typeof cam.setViewOffset !== "function") {
      this.broken = true; // not a projection camera — nothing honest to read
      return -1;
    }
    // a view offset the caller set for its own reasons must come back exactly
    const prevView = cam.view?.enabled === true ? { ...cam.view } : null;
    cam.setViewOffset(this.w, this.h, x, y, 1, 1);
    const prevTarget = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(this.rt);
    this.renderer.render(this.scene, camera);
    this.renderer.setRenderTarget(prevTarget);
    if (prevView) {
      cam.setViewOffset(
        prevView.fullWidth, prevView.fullHeight,
        prevView.offsetX, prevView.offsetY, prevView.width, prevView.height,
      );
    } else {
      cam.clearViewOffset?.();
    }
    try {
      // one pixel, so there is no readback origin to get wrong: the WebGPU and
      // forceWebGL backends disagree about the y axis, and (0, 0) is the same
      // texel in both conventions.
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
