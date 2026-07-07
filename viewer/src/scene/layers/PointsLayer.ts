/** All ~50K points in one instanced sprite draw. Positions morph between pos2
 *  and pos3 on the GPU (TSL mix); colors are CPU-precomputed from the shared
 *  ramp (cluster hue via golden-ratio scramble, noise as dim dust) so the node
 *  graph stays tiny and transpiles cleanly to WebGL. Opacity = confidence —
 *  the honesty rule — with a fixed faint floor for noise so the "dust" reads. */

import * as THREE from "three/webgpu";
import {
  float,
  instanceIndex,
  instancedBufferAttribute,
  mix,
  select,
  uniform,
  uv,
  vec3,
} from "three/tsl";
import type { Columns } from "../../data/columns";
import { rampColor } from "../../styles/tokens";

// additive blending: dense cores must not saturate to white, so alphas stay low
const NOISE_RGB: [number, number, number] = [0.42, 0.38, 0.47];
const NOISE_ALPHA = 0.06;
const MIN_ALPHA = 0.07;
const MAX_ALPHA = 0.38;

/** Deterministic cluster hue: golden-ratio scramble so neighbors differ. */
export function clusterColor(cid: number): [number, number, number] {
  return rampColor((cid * 0.61803398875) % 1);
}

export class PointsLayer {
  readonly object: THREE.Sprite;
  /** world units per instance quad — driver sets this to px × wpp each frame */
  readonly uSize = uniform(0.01);
  /** 0 = pos2 map, 1 = pos3 flythrough */
  readonly uMorph = uniform(0);
  /** hovered instance index (float compare; -1 = none) */
  readonly uHover = uniform(-1);
  /** user point-scale multiplier (Additional tab) */
  readonly uScale = uniform(1);
  /** 1 = noise dust visible, 0 = hidden (toggle) */
  readonly uNoiseVis = uniform(1);
  /** hide clustered points whose confidence is below this (0–1) */
  readonly uConfFloor = uniform(0);

  private material: THREE.SpriteNodeMaterial;
  private idSprite: THREE.Sprite | null = null;

  // attribute nodes kept so the id-pick material can share the exact same
  // per-instance data (and therefore the exact same positions/visibility)
  private iPos2!: ReturnType<typeof instancedBufferAttribute<"vec2">>;
  private iPos3!: ReturnType<typeof instancedBufferAttribute<"vec3">>;
  private iNoise!: ReturnType<typeof instancedBufferAttribute<"float">>;
  private iConf!: ReturnType<typeof instancedBufferAttribute<"float">>;
  private count: number;

  constructor(columns: Columns) {
    const n = columns.count;

    const color = new Float32Array(n * 3);
    const alpha = new Float32Array(n);
    const noise = new Float32Array(n); // 1 = noise point
    const conf = new Float32Array(n); // normalized confidence for floor cut
    for (let i = 0; i < n; i++) {
      const cid = columns.clusterId[i]!;
      conf[i] = columns.confidence[i]! / 255;
      if (cid < 0) {
        color[i * 3] = NOISE_RGB[0];
        color[i * 3 + 1] = NOISE_RGB[1];
        color[i * 3 + 2] = NOISE_RGB[2];
        alpha[i] = NOISE_ALPHA;
        noise[i] = 1;
      } else {
        const [r, g, b] = clusterColor(cid);
        color[i * 3] = r;
        color[i * 3 + 1] = g;
        color[i * 3 + 2] = b;
        alpha[i] = MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * conf[i]!;
      }
    }

    const iPos2 = instancedBufferAttribute<"vec2">(new THREE.InstancedBufferAttribute(columns.pos2, 2), "vec2");
    const iPos3 = instancedBufferAttribute<"vec3">(new THREE.InstancedBufferAttribute(columns.pos3, 3), "vec3");
    const iColor = instancedBufferAttribute<"vec3">(new THREE.InstancedBufferAttribute(color, 3), "vec3");
    const iAlpha = instancedBufferAttribute<"float">(new THREE.InstancedBufferAttribute(alpha, 1), "float");
    const iNoise = instancedBufferAttribute<"float">(new THREE.InstancedBufferAttribute(noise, 1), "float");
    const iConf = instancedBufferAttribute<"float">(new THREE.InstancedBufferAttribute(conf, 1), "float");
    this.iPos2 = iPos2;
    this.iPos3 = iPos3;
    this.iNoise = iNoise;
    this.iConf = iConf;
    this.count = n;

    const material = new THREE.SpriteNodeMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
    });

    material.positionNode = mix(vec3(iPos2, 0), iPos3, this.uMorph);

    const hovered = instanceIndex.toFloat().equal(this.uHover);
    material.scaleNode = this.uSize.mul(this.uScale).mul(select(hovered, float(2.2), float(1)));

    // soft disc mask on the quad; hover pops to near-solid
    const d = uv().sub(0.5).length();
    const disc = d.smoothstep(0.18, 0.5).oneMinus();
    material.colorNode = iColor;
    // visibility gates: noise toggle kills dust; confidence floor cuts weak
    // clustered points (noise is exempt so the two controls stay orthogonal)
    const gate = select(iNoise.greaterThan(0.5), this.uNoiseVis, iConf.step(this.uConfFloor));
    material.opacityNode = disc.mul(select(hovered, float(1), iAlpha)).mul(gate);

    this.material = material;
    this.object = new THREE.Sprite(material);
    this.object.count = n;
    this.object.frustumCulled = false;
  }

  setHover(index: number | null): void {
    this.uHover.value = index ?? -1;
  }

  /** Companion sprite that renders every point's instance index as a 24-bit
   *  RGB id (offset by 1; 0 = background) — the id-buffer 3D picker renders
   *  this into an offscreen target and reads one pixel. Shares this layer's
   *  attribute + uniform nodes so pick positions and visibility gates can
   *  never drift from what's on screen. */
  createIdMesh(): THREE.Sprite {
    const material = new THREE.SpriteNodeMaterial({ transparent: false });

    material.positionNode = mix(vec3(this.iPos2, 0), this.iPos3, this.uMorph);
    // slightly fatter than the visual point so hover is finger-friendly
    material.scaleNode = this.uSize.mul(this.uScale).mul(1.8);

    const id = instanceIndex.add(1).toFloat();
    const r = id.mod(256);
    const g = id.div(256).floor().mod(256);
    const b = id.div(65536).floor();
    material.colorNode = vec3(r, g, b).div(255);

    // hard disc + the same visibility gates as the visual layer; alphaTest
    // discards instead of blending so ids never mix
    const d = uv().sub(0.5).length();
    const gate = select(this.iNoise.greaterThan(0.5), this.uNoiseVis, this.iConf.step(this.uConfFloor));
    material.opacityNode = select(d.lessThan(0.45), float(1), float(0)).mul(gate);
    material.alphaTest = 0.5;

    const sprite = new THREE.Sprite(material);
    sprite.count = this.count;
    sprite.frustumCulled = false;
    this.idSprite = sprite;
    return sprite;
  }

  dispose(): void {
    // NB: never dispose `object.geometry` — THREE.Sprite shares ONE module-level
    // quad geometry across every sprite (points, flare, halos, id-mesh). Freeing
    // it here destroys that buffer for all of them → "Buffer used in submit while
    // destroyed" every frame and a blank atlas after a dataset switch. The
    // per-instance data lives on the material's TSL nodes, freed by dispose().
    this.material.dispose();
    if (this.idSprite) {
      (this.idSprite.material as THREE.Material).dispose();
      this.idSprite = null;
    }
  }
}
