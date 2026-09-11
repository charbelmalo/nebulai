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
import { rampColor } from "@psychix/viz/tokens";

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
  /** 1 = a keyword search is active: non-matches dim to a faint ghost */
  readonly uSearchMode = uniform(0);
  /** hand-rig shockwave: (origin.x, origin.y, front radius, amplitude), all in
   *  world units. amplitude 0 — the rest value — makes the whole term vanish. */
  readonly uPulse = uniform(new THREE.Vector4(0, 0, 0, 0));
  /** Gaussian half-width of the shockwave packet, world units. */
  readonly uPulseWidth = uniform(1);

  private material: THREE.SpriteNodeMaterial;
  private idSprite: THREE.Sprite | null = null;

  // attribute nodes kept so the id-pick material can share the exact same
  // per-instance data (and therefore the exact same positions/visibility)
  private iPos2!: ReturnType<typeof instancedBufferAttribute<"vec2">>;
  private iPos3!: ReturnType<typeof instancedBufferAttribute<"vec3">>;
  private iNoise!: ReturnType<typeof instancedBufferAttribute<"float">>;
  private iConf!: ReturnType<typeof instancedBufferAttribute<"float">>;
  private count: number;
  // search-match flags: CPU-writable so setMatches() can flip them per query
  private matchArray: Float32Array;
  private matchAttr: THREE.InstancedBufferAttribute;

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
    this.matchArray = new Float32Array(n);
    this.matchAttr = new THREE.InstancedBufferAttribute(this.matchArray, 1);
    this.matchAttr.setUsage(THREE.DynamicDrawUsage);
    const iMatch = instancedBufferAttribute<"float">(this.matchAttr, "float");
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

    material.positionNode = this.positionExpression();

    const hovered = instanceIndex.toFloat().equal(this.uHover);
    // matches grow slightly while a search is live so they read at map zoom
    const searchScale = mix(float(1), mix(float(1), float(1.5), iMatch), this.uSearchMode);
    material.scaleNode = this.uSize
      .mul(this.uScale)
      .mul(select(hovered, float(2.2), float(1)))
      .mul(searchScale);

    // soft disc mask on the quad; hover pops to near-solid
    const d = uv().sub(0.5).length();
    const disc = d.smoothstep(0.18, 0.5).oneMinus();
    material.colorNode = iColor;
    // visibility gates: noise toggle kills dust; confidence floor cuts weak
    // clustered points (noise is exempt so the two controls stay orthogonal)
    const gate = select(iNoise.greaterThan(0.5), this.uNoiseVis, iConf.step(this.uConfFloor));
    // search dim: non-matches ghost to 5% (still pickable — the id mesh does
    // not apply this) so matches carry the frame without hiding the map
    const searchDim = mix(float(1), mix(float(0.05), float(1), iMatch), this.uSearchMode);
    material.opacityNode = disc.mul(select(hovered, float(1), iAlpha)).mul(gate).mul(searchDim);

    this.material = material;
    this.object = new THREE.Sprite(material);
    this.object.count = n;
    this.object.frustumCulled = false;
  }

  /** The one position expression, shared by the visual sprite and the id mesh.
   *
   *  Base is the 2D↔3D morph. On top of it rides the hand rig's shockwave: a
   *  radial Gabor packet — a Gaussian envelope on a single sine cycle, so one
   *  compression front is followed by one rarefaction — displacing each point
   *  along the ray from the wave's origin. A bare bump would read as a
   *  travelling smudge; the signed pair is what makes it read as a wave.
   *
   *  This is deliberately a displacement of the geometry rather than an overlay
   *  drawn on top of it. Because `createIdMesh()` builds from the same method,
   *  the id-buffer picker sees the displaced positions too — a point caught in
   *  the front is genuinely where it appears to be, and stays clickable there.
   *  Anything that only touched `colorNode`/`opacityNode` would be a decal that
   *  the picker could see through.
   *
   *  `uPulse.w` is 0 at rest, which zeroes the whole term, so a session that
   *  never casts renders the identical expression it did before the rig existed.
   */
  private positionExpression(): ReturnType<typeof vec3> {
    const base = mix(vec3(this.iPos2, 0), this.iPos3, this.uMorph);
    const offset = base.xy.sub(this.uPulse.xy);
    const dist = offset.length();
    // The ray outward from the origin. Guarded because a point sitting exactly
    // on the origin has no direction, and 0/0 poisons the whole vertex.
    const ray = offset.div(dist.max(float(1e-6)));
    const ring = dist.sub(this.uPulse.z);
    const envelope = ring
      .mul(ring)
      .div(this.uPulseWidth.mul(this.uPulseWidth).mul(2))
      .negate()
      .exp();
    const wave = ring.div(this.uPulseWidth).mul(2).sin();
    const strain = this.uPulse.w.mul(envelope).mul(wave);
    return vec3(base.xy.add(ray.mul(strain)), base.z);
  }

  setHover(index: number | null): void {
    this.uHover.value = index ?? -1;
  }

  /** Highlight a search-match set (null = no active query, everything back to
   *  normal). A live query with zero matches dims the whole map — honest:
   *  "nothing here matches". */
  setMatches(ids: Uint32Array | null): void {
    this.matchArray.fill(0);
    if (ids) for (let i = 0; i < ids.length; i++) this.matchArray[ids[i]!] = 1;
    this.matchAttr.needsUpdate = true;
    this.uSearchMode.value = ids ? 1 : 0;
  }

  /** Companion sprite that renders every point's instance index as a 24-bit
   *  RGB id (offset by 1; 0 = background) — the id-buffer 3D picker renders
   *  this into an offscreen target and reads one pixel. Shares this layer's
   *  attribute + uniform nodes so pick positions and visibility gates can
   *  never drift from what's on screen. */
  createIdMesh(): THREE.Sprite {
    const material = new THREE.SpriteNodeMaterial({ transparent: false });

    material.positionNode = this.positionExpression();
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
