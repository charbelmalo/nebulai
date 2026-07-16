/** Selection marker: a quiet additive glow at the selected anchor. Toned to
 *  sit UNDER the beams in visual weight — the connection pattern is the hero,
 *  the flare only says "you are here". Pre-allocated; retargeted on selection
 *  change, hidden when nothing is selected. */

import * as THREE from "three/webgpu";
import { mix, uniform, uv, vec3 } from "three/tsl";

const FLARE_Z = 0.06; // topmost plane

export class FlareLayer {
  readonly group = new THREE.Group();

  private uCenter = uniform(new THREE.Vector2(0, 0));
  private uSize = uniform(1); // flare diameter in world units
  private uColor = uniform(new THREE.Color(1, 1, 1));
  private uActive = uniform(0);

  private flare: THREE.Sprite;
  private flareMat: THREE.SpriteNodeMaterial;

  constructor() {
    const flareMat = new THREE.SpriteNodeMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
    });
    flareMat.positionNode = vec3(this.uCenter, FLARE_Z);
    flareMat.scaleNode = this.uSize;

    // radial core + a faint wide skirt — no streaks, no sparkles. Kept well
    // below the beams in brightness: the selection marker must not outshine
    // the connection pattern it anchors
    const d = uv().sub(0.5).length();
    const core = d.smoothstep(0.0, 0.32).oneMinus();
    const halo = d.smoothstep(0.05, 0.5).oneMinus().mul(0.12);
    // whiten the core slightly so bloom still picks it out of the ramp hues
    flareMat.colorNode = mix(this.uColor, vec3(1, 0.98, 0.94), core.mul(0.5));
    flareMat.opacityNode = core.mul(0.55).add(halo).mul(this.uActive);

    this.flareMat = flareMat;
    this.flare = new THREE.Sprite(flareMat);
    this.flare.count = 1;
    this.flare.frustumCulled = false;
    this.flare.renderOrder = 3;

    this.group.add(this.flare);
    this.group.visible = false;
  }

  /** Point the glow at a world position. `size` = flare diameter (world). */
  setTarget(x: number, y: number, size: number, color: [number, number, number]): void {
    (this.uCenter.value as THREE.Vector2).set(x, y);
    this.uSize.value = size;
    (this.uColor.value as THREE.Color).setRGB(color[0], color[1], color[2]);
    this.uActive.value = 1;
    this.group.visible = true;
  }

  clearTarget(): void {
    this.uActive.value = 0;
    this.group.visible = false;
  }

  set visible(v: boolean) {
    this.group.visible = v && this.uActive.value === 1;
  }

  dispose(): void {
    // the sprite quad geometry is shared module-wide by THREE.Sprite — never
    // dispose it (see PointsLayer.dispose). Materials own the per-sprite nodes.
    this.flareMat.dispose();
  }
}
