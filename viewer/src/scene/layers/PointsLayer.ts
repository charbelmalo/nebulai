/** All ~50K points in one instanced sprite draw. Positions morph between pos2
 *  and pos3 on the GPU (TSL mix); colors are CPU-precomputed from the shared
 *  ramp (cluster hue via golden-ratio scramble, noise as dim dust) so the node
 *  graph stays tiny and transpiles cleanly to WebGL. Opacity = confidence —
 *  the honesty rule — with a fixed faint floor for noise so the "dust" reads.
 *
 *  ## The vertex-buffer budget (read before adding an attribute)
 *
 *  WebGPU's `maxVertexBuffers` is **8** on every desktop adapter we target, and
 *  exceeding it does not throw — the pipeline is rejected and the draw silently
 *  disappears. `RenderObject.getVertexBuffers()` counts one buffer per distinct
 *  `attribute.data`, so `THREE.Sprite`'s interleaved position+uv quad costs one,
 *  and every `instancedBufferAttribute` costs one more.
 *
 *  Before the glitch lens this material bound exactly **8**: the quad, plus
 *  `iPos2 iPos3 iColor iAlpha iNoise iConf iMatch`. There was no free slot, so
 *  the three per-instance *scalars* are now packed into one `vec4`:
 *
 *      iFlags = (noise, confidence, searchMatch, channelValue)
 *
 *  which brings the sprite to **6** buffers and leaves room for the Phase-1
 *  axis lane (`iAxis`, also a packed vec4) with one slot still spare. Any new
 *  per-instance scalar belongs in a spare lane of an existing vec4, not in a
 *  new attribute — and a simplified diagnostic will not warn you, because TSL
 *  tree-shakes attributes the node graph does not reference, so a cut-down
 *  shader binds fewer buffers than the real one.
 *
 *  `iFlags` is `DynamicDrawUsage`: both the search lane and the channel lane are
 *  rewritten from the CPU (~800 KB per re-upload at 50K points, once per query
 *  or channel switch — not per frame).
 */

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
import { RAMP, hexToRgb01, rampColor } from "@psychix/viz/tokens";

// additive blending: dense cores must not saturate to white, so alphas stay low
const NOISE_RGB: [number, number, number] = [0.42, 0.38, 0.47];
const NOISE_ALPHA = 0.06;
const MIN_ALPHA = 0.07;
const MAX_ALPHA = 0.38;

/** The channel lane's "this point has no value" sentinel.
 *
 *  A per-instance attribute cannot carry `null`, and it must not carry `0` — a
 *  token whose row was never read would then draw inside the low-norm knot,
 *  which is the exact false finding the channel exists to test for. NaN is worse
 *  than useless here: every GPU comparison against NaN is false, so a NaN would
 *  silently take the *same* branch as "out of range" while also poisoning the
 *  ramp. So absence is a magnitude no measurement can reach, and the shader
 *  tests for it explicitly and paints those points in the not-measured grey. */
export const CHANNEL_MISSING = -3.4e38;

/** Points with no value for the active channel. Deliberately off-ramp and
 *  desaturated: it can be mistaken for neither end of the scale. */
const NOT_MEASURED_RGB: [number, number, number] = [0.34, 0.35, 0.40];

/** Deterministic cluster hue: golden-ratio scramble so neighbors differ. */
export function clusterColor(cid: number): [number, number, number] {
  return rampColor((cid * 0.61803398875) % 1);
}

/** The shared 5-stop ramp as a TSL node — the same piecewise-linear walk
 *  `rampColor()` does on the CPU, so the rail's legend and the points agree.
 *  Chained `mix`es rather than a sampled texture: no resource to allocate,
 *  dispose or re-upload on a dataset switch, and nothing to leak. */
function rampNode(t: THREE.Node<"float">): THREE.Node<"vec3"> {
  const x = t.clamp(0, 1).mul(RAMP.length - 1);
  let acc: THREE.Node<"vec3"> = vec3(...hexToRgb01(RAMP[0]!));
  for (let i = 1; i < RAMP.length; i++) {
    acc = mix(acc, vec3(...hexToRgb01(RAMP[i]!)), x.sub(i - 1).clamp(0, 1));
  }
  return acc;
}

/** Where a raw channel value lands on the 0–1 ramp, given the ramp's ends.
 *
 *  The CPU mirror of the shader's ramp coordinate, exported so
 *  `tests/unit/channels.test.ts` can assert the legend the UI draws and the
 *  colour the GPU picks are the same function. Returns null for "not measured",
 *  which has no position on the scale at all. */
export function channelRampT(value: number, lo: number, hi: number): number | null {
  if (!Number.isFinite(value) || value <= CHANNEL_MISSING) return null;
  const span = hi - lo;
  if (!(Math.abs(span) > 0)) return 0;
  return Math.min(Math.max((value - lo) / span, 0), 1);
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
  /** 1 = a channel lens is active (colour by value, dim outside the range). */
  readonly uChannelMode = uniform(0);
  /** (filterLo, filterHi, rampLo, rampHi) in the channel's own raw units.
   *  The filter window and the colour scale are separate on purpose: narrowing
   *  the window to the knot must not re-stretch the ramp under it, or every
   *  screenshot would show a "full range" of colour whatever it contained. */
  readonly uChannel = uniform(new THREE.Vector4(0, 1, 0, 1));

  private material: THREE.SpriteNodeMaterial;
  private idSprite: THREE.Sprite | null = null;

  // attribute nodes kept so the id-pick material can share the exact same
  // per-instance data (and therefore the exact same positions/visibility)
  private iPos2!: ReturnType<typeof instancedBufferAttribute<"vec2">>;
  private iPos3!: ReturnType<typeof instancedBufferAttribute<"vec3">>;
  private iFlags!: ReturnType<typeof instancedBufferAttribute<"vec4">>;
  private count: number;
  // (noise, confidence, searchMatch, channelValue) — see the buffer-budget note
  // at the top of this file. CPU-writable: setMatches() rewrites lane z and
  // setChannel() rewrites lane w.
  private flagsArray: Float32Array;
  private flagsAttr: THREE.InstancedBufferAttribute;

  constructor(columns: Columns) {
    const n = columns.count;

    const color = new Float32Array(n * 3);
    const alpha = new Float32Array(n);
    // (noise, confidence, searchMatch, channelValue) per instance
    const flags = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      const cid = columns.clusterId[i]!;
      const conf = columns.confidence[i]! / 255;
      flags[i * 4 + 1] = conf;
      flags[i * 4 + 3] = CHANNEL_MISSING; // no channel until one is chosen
      if (cid < 0) {
        color[i * 3] = NOISE_RGB[0];
        color[i * 3 + 1] = NOISE_RGB[1];
        color[i * 3 + 2] = NOISE_RGB[2];
        alpha[i] = NOISE_ALPHA;
        flags[i * 4] = 1;
      } else {
        const [r, g, b] = clusterColor(cid);
        color[i * 3] = r;
        color[i * 3 + 1] = g;
        color[i * 3 + 2] = b;
        alpha[i] = MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * conf;
      }
    }

    const iPos2 = instancedBufferAttribute<"vec2">(new THREE.InstancedBufferAttribute(columns.pos2, 2), "vec2");
    const iPos3 = instancedBufferAttribute<"vec3">(new THREE.InstancedBufferAttribute(columns.pos3, 3), "vec3");
    const iColor = instancedBufferAttribute<"vec3">(new THREE.InstancedBufferAttribute(color, 3), "vec3");
    const iAlpha = instancedBufferAttribute<"float">(new THREE.InstancedBufferAttribute(alpha, 1), "float");
    this.flagsArray = flags;
    this.flagsAttr = new THREE.InstancedBufferAttribute(flags, 4);
    this.flagsAttr.setUsage(THREE.DynamicDrawUsage);
    const iFlags = instancedBufferAttribute<"vec4">(this.flagsAttr, "vec4");
    const iNoise = iFlags.x;
    const iConf = iFlags.y;
    const iMatch = iFlags.z;
    const iValue = iFlags.w;
    this.iPos2 = iPos2;
    this.iPos3 = iPos3;
    this.iFlags = iFlags;
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

    // ── the channel lens ──────────────────────────────────────────────────
    // A point with no value is not at either end of the scale, so it is tested
    // for FIRST and painted in the not-measured grey. Everything else takes its
    // colour from the shared ramp at its own value.
    const missing = iValue.lessThanEqual(float(CHANNEL_MISSING * 0.5));
    const t = iValue
      .sub(this.uChannel.z)
      .div(this.uChannel.w.sub(this.uChannel.z).max(float(1e-9)))
      .clamp(0, 1);
    const lensColor = select(missing, vec3(...NOT_MEASURED_RGB), rampNode(t));
    material.colorNode = mix(iColor, lensColor, this.uChannelMode);

    // visibility gates: noise toggle kills dust; confidence floor cuts weak
    // clustered points (noise is exempt so the two controls stay orthogonal)
    const gate = select(iNoise.greaterThan(0.5), this.uNoiseVis, iConf.step(this.uConfFloor));
    // search dim: non-matches ghost to 5% (still pickable — the id mesh does
    // not apply this) so matches carry the frame without hiding the map
    const searchDim = mix(float(1), mix(float(0.05), float(1), iMatch), this.uSearchMode);
    // channel dim: same idiom, and unmeasured points are OUTSIDE every window —
    // they are dimmed with the out-of-range points rather than kept in frame as
    // though the filter had found them.
    const inWindow = select(
      missing,
      float(0),
      select(iValue.greaterThanEqual(this.uChannel.x).and(iValue.lessThanEqual(this.uChannel.y)), float(1), float(0)),
    );
    const channelDim = mix(float(1), mix(float(0.05), float(1), inWindow), this.uChannelMode);
    material.opacityNode = disc
      .mul(select(hovered, float(1), iAlpha))
      .mul(gate)
      .mul(searchDim)
      .mul(channelDim);

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
    const f = this.flagsArray;
    for (let i = 2; i < f.length; i += 4) f[i] = 0;
    if (ids) for (let i = 0; i < ids.length; i++) f[ids[i]! * 4 + 2] = 1;
    this.flagsAttr.needsUpdate = true;
    this.uSearchMode.value = ids ? 1 : 0;
  }

  /** Light the channel lens, or (with `null`) put it away.
   *
   *  `values` is one raw number per point; anything non-finite is written as
   *  `CHANNEL_MISSING` and renders as not-measured grey, outside every filter
   *  window. `ramp` is the colour scale's ends and `window` the filter's — kept
   *  apart so zooming the filter onto the knot does not re-stretch the ramp
   *  beneath it and make a sliver look like a spectrum. */
  setChannel(
    values: Float32Array | null,
    ramp?: readonly [number, number],
    window?: readonly [number, number],
  ): void {
    const f = this.flagsArray;
    if (!values) {
      for (let i = 3; i < f.length; i += 4) f[i] = CHANNEL_MISSING;
      this.flagsAttr.needsUpdate = true;
      this.uChannelMode.value = 0;
      return;
    }
    const n = Math.min(this.count, values.length);
    for (let i = 0; i < this.count; i++) {
      const v = i < n ? values[i]! : Number.NaN;
      f[i * 4 + 3] = Number.isFinite(v) ? v : CHANNEL_MISSING;
    }
    this.flagsAttr.needsUpdate = true;
    const [rLo, rHi] = ramp ?? [0, 1];
    const [wLo, wHi] = window ?? [rLo, rHi];
    this.uChannel.value.set(wLo, wHi, rLo, rHi);
    this.uChannelMode.value = 1;
  }

  /** Move only the filter window (a rail drag), leaving the ramp alone. */
  setChannelWindow(lo: number, hi: number): void {
    this.uChannel.value.x = lo;
    this.uChannel.value.y = hi;
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
    const gate = select(
      this.iFlags.x.greaterThan(0.5),
      this.uNoiseVis,
      this.iFlags.y.step(this.uConfFloor),
    );
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
