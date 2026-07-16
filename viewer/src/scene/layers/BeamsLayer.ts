/** The hero effect: thin gradient beams from a selected hub to its neighbors.
 *  Minimal by design (Embedding-Projector / Cosmograph register): weight maps
 *  to opacity, no dashes, no motion — the connection pattern carries the
 *  frame, not the stroke styling. Pre-allocated instanced quads — never
 *  re-created, just re-written and re-counted on selection change. Geometry
 *  is built in the vertex stage from per-instance start/end points so a beam
 *  is one instance, not a polyline. Weights come from 10-D cluster space
 *  (honesty guardrail). */

import * as THREE from "three/webgpu";
import {
  float,
  instancedDynamicBufferAttribute,
  mix,
  positionGeometry,
  texture,
  uniform,
  uv,
  vec2,
  vec3,
} from "three/tsl";
import { rampTextureData } from "../../styles/tokens";

export const MAX_BEAMS = 64;

const WIDTH_SRC_PX = 2.5; // beam width at the source, in CSS px
const WIDTH_DST_PX = 1.25; // tapers toward the target
const BEAM_Z = 0.04; // above territories (-0.05) and the point plane (0)

export interface Beam {
  start: [number, number];
  end: [number, number];
  /** similarity weight ∈ (0,1] — ramp position + alpha */
  weight: number;
}

export class BeamsLayer {
  readonly object: THREE.InstancedMesh;
  /** world units per CSS px — driver syncs with the camera each frame */
  readonly uWpp = uniform(0.01);
  /** user width multiplier (Settings → Appearance → Atlas: beamWidth) */
  readonly uWidthScale = uniform(1);

  private startAttr: THREE.InstancedBufferAttribute;
  private endAttr: THREE.InstancedBufferAttribute;
  private weightAttr: THREE.InstancedBufferAttribute;
  private material: THREE.MeshBasicNodeMaterial;
  private rampTex: THREE.DataTexture;

  constructor() {
    const starts = new Float32Array(MAX_BEAMS * 2);
    const ends = new Float32Array(MAX_BEAMS * 2);
    const weights = new Float32Array(MAX_BEAMS);
    this.startAttr = new THREE.InstancedBufferAttribute(starts, 2);
    this.endAttr = new THREE.InstancedBufferAttribute(ends, 2);
    this.weightAttr = new THREE.InstancedBufferAttribute(weights, 1);

    const aStart = instancedDynamicBufferAttribute<"vec2">(this.startAttr, "vec2");
    const aEnd = instancedDynamicBufferAttribute<"vec2">(this.endAttr, "vec2");
    const aWeight = instancedDynamicBufferAttribute<"float">(this.weightAttr, "float");

    this.rampTex = new THREE.DataTexture(rampTextureData(), 256, 1, THREE.RGBAFormat);
    this.rampTex.needsUpdate = true;

    const material = new THREE.MeshBasicNodeMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });

    // build the ribbon: t runs source→target, `across` spans the width
    const t = uv().x;
    const across = positionGeometry.y; // plane geometry: -0.5 … 0.5
    const dir = aEnd.sub(aStart);
    const len = dir.length().max(1e-6);
    const perp = vec2(dir.y.negate(), dir.x).div(len);
    const widthWorld = mix(float(WIDTH_SRC_PX), float(WIDTH_DST_PX), t)
      .mul(this.uWidthScale)
      .mul(this.uWpp);
    // gentle bow (links arc slightly so overlapping edges stay separable)
    const bow = t.mul(Math.PI).sin().mul(len).mul(0.03);
    const p = aStart
      .add(dir.mul(t))
      .add(perp.mul(across.mul(widthWorld).add(bow)));
    material.positionNode = vec3(p, BEAM_Z);

    // gradient along the beam through the shared 5-stop ramp
    material.colorNode = texture(this.rampTex, vec2(t, 0.5)).rgb;

    // soft ribbon: fade across the width and at both ends; weight IS the
    // opacity — one visual channel, honestly mapped, nothing animated
    const edgeFade = across.abs().mul(2).smoothstep(0.25, 1).oneMinus();
    const endFade = t.smoothstep(0, 0.05).mul(t.oneMinus().smoothstep(0, 0.08));
    material.opacityNode = edgeFade.mul(endFade).mul(mix(float(0.3), float(0.95), aWeight));

    this.material = material;
    // 1×1 plane, its vertices fully recomputed by positionNode
    this.object = new THREE.InstancedMesh(new THREE.PlaneGeometry(1, 1, 24, 1), material, MAX_BEAMS);
    this.object.count = 0;
    this.object.frustumCulled = false;
    this.object.renderOrder = 2; // after points so beams glow on top
  }

  setBeams(beams: Beam[]): void {
    const n = Math.min(beams.length, MAX_BEAMS);
    const s = this.startAttr.array as Float32Array;
    const e = this.endAttr.array as Float32Array;
    const w = this.weightAttr.array as Float32Array;
    for (let i = 0; i < n; i++) {
      const b = beams[i]!;
      s[i * 2] = b.start[0];
      s[i * 2 + 1] = b.start[1];
      e[i * 2] = b.end[0];
      e[i * 2 + 1] = b.end[1];
      w[i] = b.weight;
    }
    this.startAttr.needsUpdate = true;
    this.endAttr.needsUpdate = true;
    this.weightAttr.needsUpdate = true;
    this.object.count = n;
  }

  clear(): void {
    this.object.count = 0;
  }

  set visible(v: boolean) {
    this.object.visible = v;
  }

  dispose(): void {
    this.material.dispose();
    this.object.geometry.dispose();
    this.rampTex.dispose();
  }
}
