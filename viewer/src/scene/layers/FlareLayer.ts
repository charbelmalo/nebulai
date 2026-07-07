/** Selection "sun": an additive flare sprite at the selected anchor plus a
 *  small orbiting sparkle particle system. The flare is what the bloom pass
 *  amplifies into the video's signature glow. Pre-allocated; retargeted on
 *  selection change, hidden when nothing is selected. */

import * as THREE from "three/webgpu";
import {
  float,
  instancedBufferAttribute,
  mix,
  uniform,
  uv,
  vec2,
  vec3,
} from "three/tsl";

const SPARKLE_COUNT = 48;
const FLARE_Z = 0.06; // topmost plane

export class FlareLayer {
  readonly group = new THREE.Group();
  readonly uTime = uniform(0);
  /** 1 = orbiting sparkle + shimmer, 0 = static glow (reduced motion) */
  readonly uMotion = uniform(1);

  private uCenter = uniform(new THREE.Vector2(0, 0));
  private uSize = uniform(1); // flare diameter in world units
  private uColor = uniform(new THREE.Color(1, 1, 1));
  private uActive = uniform(0);

  private flare: THREE.Sprite;
  private sparkle: THREE.Sprite;
  private flareMat: THREE.SpriteNodeMaterial;
  private sparkleMat: THREE.SpriteNodeMaterial;

  constructor() {
    // ── the sun: radial core + 4-point star streaks ──────────────────────
    const flareMat = new THREE.SpriteNodeMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
    });
    flareMat.positionNode = vec3(this.uCenter, FLARE_Z);
    flareMat.scaleNode = this.uSize;

    const q = uv().sub(0.5);
    const d = q.length();
    const core = d.smoothstep(0.0, 0.32).oneMinus(); // hot center
    const halo = d.smoothstep(0.05, 0.5).oneMinus().mul(0.35); // wide skirt
    // star streaks along the quad axes, shimmering slightly with time
    const shimmer = this.uTime.mul(5).sin().mul(0.15).mul(this.uMotion).add(1);
    const streakX = q.y.abs().smoothstep(0.0, 0.012).oneMinus().mul(q.x.abs().mul(2).oneMinus().max(0));
    const streakY = q.x.abs().smoothstep(0.0, 0.012).oneMinus().mul(q.y.abs().mul(2).oneMinus().max(0));
    const streaks = streakX.add(streakY).mul(0.55).mul(shimmer);
    // whiten the core so bloom reads it as the brightest thing on screen
    flareMat.colorNode = mix(this.uColor, vec3(1, 0.98, 0.94), core.mul(0.8));
    flareMat.opacityNode = core.mul(1.4).add(halo).add(streaks).mul(this.uActive);

    this.flareMat = flareMat;
    this.flare = new THREE.Sprite(flareMat);
    this.flare.count = 1;
    this.flare.frustumCulled = false;
    this.flare.renderOrder = 3;

    // ── sparkle: tiny particles orbiting/breathing around the center ─────
    const seeds = new Float32Array(SPARKLE_COUNT);
    for (let i = 0; i < SPARKLE_COUNT; i++) seeds[i] = i / SPARKLE_COUNT;
    const iSeed = instancedBufferAttribute<"float">(new THREE.InstancedBufferAttribute(seeds, 1), "float");

    const sparkleMat = new THREE.SpriteNodeMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
    });

    // per-particle: fixed angle from seed, radius cycles outward over a
    // staggered life; reduced motion pins t=0 → a static ring of dust
    const angle = iSeed.mul(Math.PI * 2 * 7.13); // wraps several turns
    const life = iSeed.mul(9.7).add(this.uTime.mul(0.35).mul(this.uMotion)).fract();
    const r = mix(float(0.18), float(0.85), life).mul(this.uSize);
    const px = angle.cos().mul(r);
    const py = angle.sin().mul(r);
    sparkleMat.positionNode = vec3(this.uCenter.add(vec2(px, py)), FLARE_Z);
    sparkleMat.scaleNode = this.uSize.mul(mix(float(0.02), float(0.045), iSeed.mul(13.7).fract()));

    const sd = uv().sub(0.5).length();
    const dot = sd.smoothstep(0.1, 0.5).oneMinus();
    const fade = life.smoothstep(0, 0.15).mul(life.oneMinus().smoothstep(0, 0.35));
    sparkleMat.colorNode = this.uColor;
    sparkleMat.opacityNode = dot.mul(fade).mul(0.85).mul(this.uActive);

    this.sparkleMat = sparkleMat;
    this.sparkle = new THREE.Sprite(sparkleMat);
    this.sparkle.count = SPARKLE_COUNT;
    this.sparkle.frustumCulled = false;
    this.sparkle.renderOrder = 3;

    this.group.add(this.flare, this.sparkle);
    this.group.visible = false;
  }

  /** Point the sun at a world position. `size` = flare diameter (world). */
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
    this.sparkleMat.dispose();
  }
}
