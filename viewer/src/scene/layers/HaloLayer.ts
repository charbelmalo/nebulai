/** Pulsing halo rings on hub clusters (the video's "important node" tell).
 *  Hubs = highest summed edge weight in 10-D cluster space, so the pulse is
 *  data-driven, not decorative. One instanced sprite draw, ring SDF mask. */

import * as THREE from "three/webgpu";
import {
  instancedBufferAttribute,
  uniform,
  uv,
  vec3,
} from "three/tsl";

const RING_ALPHA = 0.22;

export interface Halo {
  pos: [number, number];
  /** ring radius in world units (hull radius) */
  radius: number;
  color: [number, number, number];
}

export class HaloLayer {
  readonly object: THREE.Sprite;
  readonly uTime = uniform(0);
  /** 1 = breathing pulse, 0 = static ring (reduced motion) */
  readonly uMotion = uniform(1);
  /** halos mark hubs on the flat map — the dimension morph fades them */
  readonly uFade = uniform(1);

  private material: THREE.SpriteNodeMaterial;

  constructor(halos: Halo[]) {
    const n = halos.length;
    const pos = new Float32Array(n * 2);
    const radius = new Float32Array(n);
    const color = new Float32Array(n * 3);
    const phase = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const h = halos[i]!;
      pos[i * 2] = h.pos[0];
      pos[i * 2 + 1] = h.pos[1];
      radius[i] = h.radius;
      color[i * 3] = h.color[0];
      color[i * 3 + 1] = h.color[1];
      color[i * 3 + 2] = h.color[2];
      phase[i] = (i * 2.399963) % (Math.PI * 2); // golden-angle stagger
    }

    const iPos = instancedBufferAttribute<"vec2">(new THREE.InstancedBufferAttribute(pos, 2), "vec2");
    const iRadius = instancedBufferAttribute<"float">(new THREE.InstancedBufferAttribute(radius, 1), "float");
    const iColor = instancedBufferAttribute<"vec3">(new THREE.InstancedBufferAttribute(color, 3), "vec3");
    const iPhase = instancedBufferAttribute<"float">(new THREE.InstancedBufferAttribute(phase, 1), "float");

    const material = new THREE.SpriteNodeMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
    });

    material.positionNode = vec3(iPos, 0.02);

    // breathe: ±12% scale, staggered per hub so they don't pulse in unison
    const breathe = this.uTime.mul(1.4).add(iPhase).sin().mul(0.12).mul(this.uMotion);
    material.scaleNode = iRadius.mul(2.6).mul(breathe.add(1));

    // ring SDF: bright band at r≈0.4 of the quad + faint interior glow
    const d = uv().sub(0.5).length();
    const band = d.sub(0.4).abs();
    const ring = band.smoothstep(0.004, 0.05).oneMinus();
    const glow = d.smoothstep(0.12, 0.42).oneMinus().mul(0.12);
    const alphaPulse = this.uTime.mul(1.4).add(iPhase).cos().mul(0.35).mul(this.uMotion).add(1);
    material.colorNode = iColor;
    material.opacityNode = ring.mul(RING_ALPHA).add(glow).mul(alphaPulse).mul(this.uFade);

    this.material = material;
    this.object = new THREE.Sprite(material);
    this.object.count = n;
    this.object.frustumCulled = false;
    this.object.renderOrder = 1;
    if (n === 0) this.object.visible = false;
  }

  set visible(v: boolean) {
    this.object.visible = v && this.object.count > 0;
  }

  dispose(): void {
    this.material.dispose();
    this.object.geometry.dispose();
  }
}
