/** LiveField — the atmosphere under the live chart.
 *
 *  A second canvas, behind `LiveDriver`'s: additive motes on the exact rows the
 *  chart draws its bars on, so a busy stretch of a run glows and a quiet one
 *  goes dark. It is the layer that makes a fleet *feel* like it is running,
 *  which a chart of rectangles never quite does.
 *
 *  ## What the light is allowed to mean
 *
 *  One mote per event, and **an event has no magnitude**. The stream tells us
 *  that a thing happened, what kind it was and when — never how big. So no term
 *  in this shader reads a value: brightness is *recency*, and everything else
 *  bright on screen is **crowding**, which additive blending gives for free and
 *  which is a real property of the stream rather than a number we invented for
 *  it. Where events pile up, light accumulates; where they do not, it does not.
 *
 *  That is a narrower claim than `SessionFieldDriver` makes, and deliberately.
 *  There, glow carries output tokens and has to be rank-normalised before it
 *  touches a bloom curve, because raw magnitude through bloom is a lie with a
 *  nice finish. Here there is no magnitude to rank, so the honest move is to
 *  spend the channel on time instead — and to say so on the card, because a
 *  glowing field that meant nothing would be the most persuasive lie in the
 *  whole subsystem.
 *
 *  ## Why it is allowed to be missing
 *
 *  The field is the only part of the live view that needs a GPU, and it is the
 *  only part that carries no figure. Those two facts are the same design
 *  decision: on a machine with no WebGPU, or with bloom switched off, the field
 *  simply is not there and the chart above it is exactly as legible. Nothing a
 *  researcher reads off this page depends on it.
 *
 *  Positions come from `LiveDriver.field()`. This file never touches the time
 *  window, the lane geometry or the encoding — see `FieldSample`.
 */

import * as THREE from "three/webgpu";
import {
  float,
  instancedDynamicBufferAttribute,
  mix,
  uniform,
  uv,
  vec3,
} from "three/tsl";
import { appStore } from "../../app/store";
import { createBloomPipeline, type BloomPipeline } from "../post/bloom";
import type { FieldSample } from "./LiveDriver";

/** Capacity of the instance buffers. Matches `FIELD_CAP` in `LiveDriver`; the
 *  driver clamps, this allocates. */
const CAP = 6000;

/** Mote radius in CSS pixels. Constant, and that is a decision: size is a
 *  magnitude channel and there is no magnitude here, so varying it would be
 *  inventing one. */
const MOTE_PX = 9;

/** Emissive at the live edge and at the back of the window. The near end
 *  exceeds 1 so the bloom threshold has something to key on — only the newest
 *  events are meant to flare. */
const GLOW_NEW = 1.55;
const GLOW_OLD = 0.1;

/** Alpha at each end of the same ramp. The old end is not zero: an event that
 *  happened is still an event, and fading it out of existence would say the
 *  window is emptier than it is. */
const ALPHA_NEW = 0.85;
const ALPHA_OLD = 0.16;

export class LiveField {
  private renderer: THREE.WebGPURenderer | null = null;
  private scene = new THREE.Scene();
  private camera = new THREE.OrthographicCamera(0, 1, 0, 1, -1, 1);
  private sprite: THREE.Sprite | null = null;
  private bloomPipe: BloomPipeline | null = null;
  private webgpu = false;
  private disposed = false;
  private ready = false;

  private xy = new THREE.InstancedBufferAttribute(new Float32Array(CAP * 2), 2);
  private rgb = new THREE.InstancedBufferAttribute(new Float32Array(CAP * 3), 3);
  private age = new THREE.InstancedBufferAttribute(new Float32Array(CAP), 1);

  private w = 1;
  private h = 1;
  private dpr = 1;

  /** True once there is something on screen. The card asks, so it can say the
   *  field is off rather than leave a reader wondering why it is dark. */
  get isRunning(): boolean {
    return this.ready && !this.disposed;
  }

  get isWebGPU(): boolean {
    return this.webgpu;
  }

  /** What the GPU was last asked to draw. Browser verification needs a number
   *  it can read back — a WebGPU canvas has no `getImageData`, so "the field is
   *  on" can otherwise only be checked by looking at a screenshot. */
  get lastCount(): number {
    return this.sprite?.count ?? 0;
  }

  async init(canvas: HTMLCanvasElement): Promise<void> {
    const tier = appStore.getState().capabilities?.tier ?? "webgpu";
    this.webgpu = tier === "webgpu";

    const renderer = new THREE.WebGPURenderer({
      canvas,
      antialias: false,
      alpha: true,
      forceWebGL: !this.webgpu,
    });
    await renderer.init();
    // A slow adapter can take long enough that the card is gone by the time it
    // answers. Dropping the renderer here is the difference between a torn-down
    // page and a leaked device.
    if (this.disposed) {
      renderer.dispose();
      return;
    }
    this.renderer = renderer;
    renderer.setClearColor(0x000000, 0);

    this.build();
    // Bloom rides only the real WebGPU rung, as everywhere else in the viewer.
    if (this.webgpu && appStore.getState().settings.bloom) {
      // A high threshold on purpose: this field is dense along a row, and at
      // the default the summed halos of a busy run wash the action hues out to
      // white — the same failure the compare field hit. Only the newest motes,
      // which are the ones above 1.0, are meant to flare.
      this.bloomPipe = createBloomPipeline(renderer, this.scene, this.camera, "full", {
        threshold: 0.75,
        strength: 0.7,
        radius: 0.5,
      });
    }
    this.ready = true;
  }

  private build(): void {
    for (const a of [this.xy, this.rgb, this.age]) a.setUsage(THREE.DynamicDrawUsage);
    const iXY = instancedDynamicBufferAttribute<"vec2">(this.xy, "vec2");
    const iRGB = instancedDynamicBufferAttribute<"vec3">(this.rgb, "vec3");
    const iAge = instancedDynamicBufferAttribute<"float">(this.age, "float");

    // World units are CSS pixels — the camera below is set up so they are — so
    // a position from the chart needs no conversion at all. That is the whole
    // reason the layers cannot drift.
    const material = new THREE.SpriteNodeMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
    });
    material.positionNode = vec3(iXY.x, iXY.y, 0);
    material.scaleNode = uniform(MOTE_PX);

    // Recency, and only recency. `age` is 0 at the live edge and 1 at the back
    // of the window, and it is the sole input to both ramps below.
    const fresh = iAge.oneMinus();
    material.colorNode = iRGB.mul(mix(float(GLOW_OLD), float(GLOW_NEW), fresh.pow(2.2)));

    // Soft radial falloff, no hard edge, so overlapping motes accumulate into a
    // density rather than occluding each other into a flat disc.
    const d = uv().sub(0.5).length();
    const disc = d.smoothstep(0.04, 0.5).oneMinus();
    material.opacityNode = disc.mul(mix(float(ALPHA_OLD), float(ALPHA_NEW), fresh.pow(1.6)));

    const sprite = new THREE.Sprite(material);
    sprite.count = 0;
    sprite.frustumCulled = false;
    this.sprite = sprite;
    this.scene.add(sprite);
  }

  resize(w: number, h: number, dpr: number): void {
    this.w = Math.max(1, w);
    this.h = Math.max(1, h);
    this.dpr = dpr;
    if (!this.renderer) return;
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(this.w, this.h, false);
    // Pixel space, y down, matching the 2D canvas exactly: (0,0) top-left.
    this.camera.left = 0;
    this.camera.right = this.w;
    this.camera.top = 0;
    this.camera.bottom = this.h;
    this.camera.updateProjectionMatrix();
  }

  /** Draw the frame the chart just drew. Called from `LiveDriver.onFrame`, so
   *  the two layers are never a frame apart. */
  render(sample: FieldSample): void {
    const r = this.renderer;
    const sprite = this.sprite;
    if (!r || !sprite || this.disposed) return;
    if (sample.w !== this.w || sample.h !== this.h) this.resize(sample.w, sample.h, this.dpr);

    const n = Math.min(sample.n, CAP);
    if (n > 0) {
      (this.xy.array as Float32Array).set(sample.xy.subarray(0, n * 2));
      (this.rgb.array as Float32Array).set(sample.rgb.subarray(0, n * 3));
      (this.age.array as Float32Array).set(sample.age.subarray(0, n));
      this.xy.needsUpdate = true;
      this.rgb.needsUpdate = true;
      this.age.needsUpdate = true;
    }
    sprite.count = n;

    if (this.bloomPipe) this.bloomPipe.post.render();
    else r.render(this.scene, this.camera);
  }

  dispose(): void {
    this.disposed = true;
    this.ready = false;
    this.bloomPipe?.dispose();
    this.bloomPipe = null;
    this.sprite?.material.dispose();
    this.sprite = null;
    this.scene.clear();
    this.renderer?.dispose();
    this.renderer = null;
  }
}
