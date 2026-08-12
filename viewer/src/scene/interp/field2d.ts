/** Emissive particle field for the STATIC 2D interp views.
 *
 *  The sessions and compare fields (`sessions/SessionFieldDriver`,
 *  `drivers/CompareDriver`) are perspective orbit cameras with their own rAF
 *  and playback clocks. The interp scatter views are none of those things: they
 *  are fixed orthographic charts that redraw on demand (`animated = false`, so
 *  InterpPage never gives them a rAF). So this shares the part that actually
 *  generalises — the TSL appearance graph, the tone-map + bloom chain, the
 *  isometric fit, and the "never dispose a Sprite's geometry" teardown — and
 *  leaves the orbit camera to the two drivers that have one.
 *
 *  WHY IT LOOKS DIFFERENT FROM A SCATTERPLOT: a deck.gl ScatterplotLayer draws
 *  hard-edged opaque discs, so N overlapping points read as ONE disc — the
 *  density that a 36k-point field is mostly about is thrown away at the top of
 *  the pipe. Here every point is a soft radial sprite under additive blending,
 *  so overlap SUMS and density becomes brightness. Two consequences worth
 *  naming, because both are load-bearing:
 *
 *    - Draw order stops mattering. Addition is commutative, so there is no
 *      painter's-order bias to defeat — a caller that shuffled its points to
 *      avoid one can stop.
 *    - `NeutralToneMapping` becomes mandatory. Additive sums run past 1.0 by
 *      design; with the default (none) they hard-clip PER CHANNEL, so a dense
 *      blue core lands at (1,1,1) and the colour encoding is gone. Measured on
 *      the compare field: 84% of lit pixels below 0.15 saturation before,
 *      ~10% after.
 *
 *  Chrome (hover rings, crosshairs, labels) deliberately does NOT live in this
 *  scene — it belongs in the caller's DOM overlay, projected through
 *  `worldToScreen`. Anything in the scene is data and will bloom; a hover ring
 *  that blooms reads as a measurement.
 */

import * as THREE from "three/webgpu";
import { float, instancedBufferAttribute, mix, uniform, uv, vec3 } from "three/tsl";

import { appStore } from "../../app/store";
import type { GpuTier } from "@psychix/viz/capabilities";
import { Camera2D } from "../camera2d";
import { PointPicker } from "../picking";
import { createBloomPipeline, type BloomOptions, type BloomPipeline } from "../post/bloom";
import { BG } from "@psychix/viz/tokens";

/** Per-view look. Deliberately all required except bloom: these are not taste
 *  knobs with sane defaults — the right value depends on how densely the view's
 *  hot content packs, which is a property of the data, not of the effect. A
 *  value copied from a sparse view will wash a dense one to white. Tune by
 *  measuring pixel saturation on the view's DENSEST state. */
export interface Field2DLook {
  /** Sprite quad diameter in CSS pixels at `radius = 1`.
   *
   *  Pixels, not world units, so the encoding survives a resize and matches
   *  what these views already do (`radiusUnits: "pixels"`). The most sensitive
   *  number here: it sets how often sprites overlap, and additive blending sums
   *  overlaps, so doubling it does not double brightness — it clips the core to
   *  white. The quad is bigger than the visible core: the radial falloff needs
   *  room to spill, and that spill is the glow. */
  pointPx: number;
  /** What the dimmest point still carries, so nothing is invisible. */
  emissiveMin: number;
  /** Above 1.0 on purpose — that headroom is what the bloom threshold keys on. */
  emissiveMax: number;
  /** Gamma on the rank. Higher keeps the bulk dark and sharpens the spikes;
   *  below ~1.4 it washes back into a flat scatter. */
  glowGamma: number;
  /** Opacity of the dimmest motes. */
  moteFloor: number;
  /** Brightness retained by points the caller has pushed to the background
   *  (`active = 0`) — the field equivalent of the old dim scatter layer. */
  dimLevel: number;
  bloom?: BloomOptions;
}

export interface Field2DData {
  count: number;
  /** world xy, 2 floats per point */
  pos: Float32Array;
  /** linear rgb 0..1, 3 floats per point */
  color: Float32Array;
  /** 0..1, drives glow. Rank-normalise upstream (see `rankNormalize`) rather
   *  than passing a raw magnitude — the quantities these views plot are
   *  long-tailed, and then one outlier owns the whole ramp. */
  rank: Float32Array;
  /** per-point size multiplier on `look.pointPx`, 1 = base */
  radius: Float32Array;
  /** 1 = foreground, 0 = pushed to `look.dimLevel`. Omit for all-foreground. */
  active?: Float32Array;
}

/** Asymmetric px gutters around the plot area, so axis captions and chip rows
 *  keep the room they already reserve. */
export interface Inset {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

/** Map values onto 0..1 by RANK, not by magnitude.
 *
 *  Every quantity these views plot (embedding norms, ‖w_out‖) is long-tailed,
 *  so a linear min→max ramp puts almost all the mass in the bottom decile and
 *  hands the entire visible range to a handful of outliers. Ranking spends the
 *  ramp where the points actually are. Ties share a rank, so equal values can
 *  never be drawn differently.
 *
 *  Rank is monotone in the input, so a rank-driven channel and a magnitude-
 *  driven channel of the SAME quantity can never disagree about which point is
 *  larger — which is what makes it safe to drive glow by rank while size stays
 *  linear in the real value. */
export function rankNormalize(values: ArrayLike<number>): Float32Array {
  const n = values.length;
  const out = new Float32Array(n);
  if (n === 0) return out;
  if (n === 1) {
    out[0] = 1;
    return out;
  }
  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((a, b) => (values[a] as number) - (values[b] as number));
  const denom = n - 1;
  let i = 0;
  while (i < n) {
    // ties share the mean rank of the run, so equal inputs get equal output
    let j = i + 1;
    while (j < n && (values[order[j]!] as number) === (values[order[i]!] as number)) j++;
    const share = (i + j - 1) / 2 / denom;
    for (let k = i; k < j; k++) out[order[k]!] = share;
    i = j;
  }
  return out;
}

/** Hover chrome — crosshair guides plus the LED diamond that marks the focused
 *  point — as DOM, not scene objects.
 *
 *  It used to be a LineLayer + SolidPolygonLayer inside the plot. That cannot
 *  come along to the emissive field: everything in this scene is additive and
 *  passes through bloom, so a marker drawn there would glow exactly like a
 *  dense cluster of real points and read as a measurement. Keeping chrome in
 *  the overlay (projected through `EmissiveField2D.worldToScreen`, so it cannot
 *  drift) keeps the rule visible: if it bloomed, it was data. */
export class FieldMarker {
  private root: HTMLElement;
  private h: HTMLElement;
  private v: HTMLElement;
  private dot: HTMLElement;

  constructor(parent: HTMLElement) {
    const mk = (cls: string, into: HTMLElement) => {
      const el = document.createElement("div");
      el.className = cls;
      into.appendChild(el);
      return el;
    };
    this.root = document.createElement("div");
    this.root.className = "interp-field-marker";
    this.h = mk("interp-fm-h", this.root);
    this.v = mk("interp-fm-v", this.root);
    this.dot = mk("interp-fm-dot", this.root);
    parent.appendChild(this.root);
  }

  /** `box` is the data bbox in CSS px: the guides stop at the data, as they did
   *  in the deck.gl version, rather than running the full canvas. */
  show(
    sx: number,
    sy: number,
    rPx: number,
    box: { x0: number; y0: number; x1: number; y1: number },
  ): void {
    this.root.classList.add("is-on");
    const left = Math.min(box.x0, box.x1);
    const right = Math.max(box.x0, box.x1);
    const top = Math.min(box.y0, box.y1);
    const bottom = Math.max(box.y0, box.y1);
    this.h.style.translate = `${left.toFixed(1)}px ${sy.toFixed(1)}px`;
    this.h.style.width = `${Math.max(0, right - left).toFixed(1)}px`;
    this.v.style.translate = `${sx.toFixed(1)}px ${top.toFixed(1)}px`;
    this.v.style.height = `${Math.max(0, bottom - top).toFixed(1)}px`;
    const d = Math.max(7, rPx * 2);
    this.dot.style.width = `${d.toFixed(1)}px`;
    this.dot.style.height = `${d.toFixed(1)}px`;
    // `translate`, never `transform` — the rotation that makes the diamond
    // lives in `transform`, and writing position there would clobber it (the
    // same trap the shared tooltip hit).
    this.dot.style.translate = `${(sx - d / 2).toFixed(1)}px ${(sy - d / 2).toFixed(1)}px`;
  }

  hide(): void {
    this.root.classList.remove("is-on");
  }

  dispose(): void {
    this.root.remove();
  }
}

interface FitBox {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  inset: Inset;
  fit: number;
}

export class EmissiveField2D {
  private renderer: THREE.WebGPURenderer | null = null;
  private scene = new THREE.Scene();
  private camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
  private cam = new Camera2D();
  private sprite: THREE.Sprite | null = null;
  private material: THREE.SpriteNodeMaterial | null = null;
  private bloomPipe: BloomPipeline | null = null;
  private bloomOn = false;
  private disposed = false;
  private unsubscribe: (() => void) | null = null;

  private uPointScale = uniform(1);
  /** world units per CSS pixel — sprite sizes are authored in px and scaled by
   *  this, so the encoding is resize-stable */
  private uWpp = uniform(0.01);
  /** global brightness multiplier: the whole field steps back while a single
   *  point is focused, the same "defer to the marker" move the deck.gl version
   *  made with layer opacity */
  private uDim = uniform(1);

  /** (x, y, rank, radius) and (r, g, b, active) — two vec4s, so with the
   *  Sprite's own quad this material binds 4 vertex buffers. WebGPU's limit is
   *  8 and it fails SILENTLY (an invalid pipeline draws zero fragments), so
   *  packing is not premature. */
  private attrA: THREE.InstancedBufferAttribute | null = null;
  private attrB: THREE.InstancedBufferAttribute | null = null;

  private picker: PointPicker | null = null;
  private count = 0;
  private box: FitBox | null = null;

  constructor(private look: Field2DLook) {}

  async init(canvas: HTMLCanvasElement, tier: GpuTier): Promise<void> {
    const webgpu = tier === "webgpu";
    const renderer = new THREE.WebGPURenderer({
      canvas,
      antialias: true,
      // TSL lowers the same node graph to GLSL, so the fallback rung is free —
      // minus post-processing, which is why bloom is gated below.
      forceWebGL: !webgpu,
    });
    await renderer.init();
    if (this.disposed) {
      renderer.dispose();
      return;
    }
    renderer.setClearColor(new THREE.Color(BG), 1);
    renderer.toneMapping = THREE.NeutralToneMapping; // see the header — mandatory
    this.renderer = renderer;

    this.camera.position.set(0, 0, 10);
    this.camera.lookAt(0, 0, 0);

    const s0 = appStore.getState();
    this.uPointScale.value = s0.settings.pointScale;
    this.bloomOn = webgpu && s0.settings.bloom;
    if (webgpu) {
      this.bloomPipe = createBloomPipeline(
        renderer,
        this.scene,
        this.camera,
        "full",
        this.look.bloom ?? {},
      );
    }
    // no rAF here, so a settings change has to redraw itself
    this.unsubscribe = appStore.subscribe((s) => {
      const scale = s.settings.pointScale;
      const wantBloom = webgpu && s.settings.bloom;
      if (scale === this.uPointScale.value && wantBloom === this.bloomOn) return;
      this.uPointScale.value = scale;
      this.bloomOn = wantBloom;
      this.render();
    });
  }

  /** Build (or rebuild) the field. Safe to call on every dataset change. */
  setData(d: Field2DData): void {
    this.teardownField();
    this.count = d.count;
    if (d.count === 0) return;

    const a = new Float32Array(d.count * 4);
    const b = new Float32Array(d.count * 4);
    for (let i = 0; i < d.count; i++) {
      a[i * 4] = d.pos[i * 2] ?? 0;
      a[i * 4 + 1] = d.pos[i * 2 + 1] ?? 0;
      a[i * 4 + 2] = d.rank[i] ?? 0;
      a[i * 4 + 3] = d.radius[i] ?? 1;
      b[i * 4] = d.color[i * 3] ?? 1;
      b[i * 4 + 1] = d.color[i * 3 + 1] ?? 1;
      b[i * 4 + 2] = d.color[i * 3 + 2] ?? 1;
      b[i * 4 + 3] = d.active ? (d.active[i] ?? 1) : 1;
    }
    this.attrA = new THREE.InstancedBufferAttribute(a, 4);
    this.attrB = new THREE.InstancedBufferAttribute(b, 4);
    const iA = instancedBufferAttribute<"vec4">(this.attrA, "vec4");
    const iB = instancedBufferAttribute<"vec4">(this.attrB, "vec4");

    const L = this.look;
    const glow = iA.z.pow(float(L.glowGamma));
    // `active` lerps a point toward the background instead of hiding it: a
    // dimmed point is still evidence, and a view that deletes its non-selected
    // mass is lying about how much of it there is. Brightness only — under
    // additive blending the visible contribution is colour × opacity, so
    // scaling one channel is a clean linear dim and scaling both squares it.
    const fg = mix(float(L.dimLevel), float(1), iB.w).mul(this.uDim);

    const material = new THREE.SpriteNodeMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
    });
    material.positionNode = vec3(iA.x, iA.y, float(0));
    // size stays whatever the caller's real quantity says (px, via uWpp); the
    // rank only widens it gently, so a hot core gets a halo — that halo is what
    // reads as glow — without compounding the magnitude into the radius twice
    material.scaleNode = this.uPointScale
      .mul(float(L.pointPx))
      .mul(this.uWpp)
      .mul(iA.w)
      .mul(float(0.8).add(glow.mul(0.5)));
    material.colorNode = iB.xyz.mul(mix(float(L.emissiveMin), float(L.emissiveMax), glow)).mul(fg);

    // soft radial falloff — no hard edge, so overlapping points accumulate
    // rather than occlude. This is the whole difference from a scatter disc.
    // Edges must INCREASE: reversed edges silently work on GLSL, not on WGSL.
    const dist = uv().sub(0.5).length();
    const disc = dist.smoothstep(0.06, 0.5).oneMinus();
    material.opacityNode = disc.mul(mix(float(L.moteFloor), float(1), glow));

    this.material = material;
    const sprite = new THREE.Sprite(material);
    sprite.count = d.count;
    sprite.frustumCulled = false;
    this.sprite = sprite;
    this.scene.add(sprite);

    this.picker = new PointPicker(d.pos, d.count);
  }

  /** Re-upload just the foreground mask — no rebuild, no picker churn. */
  setActive(active: Float32Array | null): void {
    const attr = this.attrB;
    if (!attr) return;
    const arr = attr.array as Float32Array;
    for (let i = 0; i < this.count; i++) arr[i * 4 + 3] = active ? (active[i] ?? 1) : 1;
    attr.needsUpdate = true;
  }

  /** Global brightness, for stepping the field back behind a focused point. */
  setDim(level: number): void {
    this.uDim.value = level;
  }

  /** Frame a world box inside asymmetric px gutters at one isometric scale
   *  (equal px per world unit on both axes), so on-screen distances stay
   *  faithful and the axis captions keep their room. Remembered and reapplied
   *  on resize. */
  fitInset(minX: number, minY: number, maxX: number, maxY: number, inset: Inset, fit = 1): void {
    this.box = { minX, minY, maxX, maxY, inset, fit };
    this.applyFit();
  }

  resize(w: number, h: number, dpr: number): void {
    const r = this.renderer;
    if (!r) return;
    r.setPixelRatio(Math.min(dpr, 2));
    r.setSize(w, h, false);
    this.cam.setViewport(w, h);
    this.applyFit();
  }

  /** Project world → CSS px, so the caller's DOM overlay can never drift from
   *  the GPU scene. */
  worldToScreen(wx: number, wy: number): [number, number] {
    return this.cam.worldToScreen(wx, wy);
  }

  /** Nearest point to a CSS-pixel position, or -1. `pxRadius` is a screen-space
   *  tolerance, so the hit area does not change with the fit. Exact and
   *  synchronous (kdbush) — no GPU id-buffer readback needed, because these
   *  positions never morph on the GPU. */
  pickAt(sx: number, sy: number, pxRadius = 6): number {
    if (!this.picker) return -1;
    const [wx, wy] = this.cam.screenToWorld(sx, sy);
    return this.picker.nearest(wx, wy, pxRadius * this.cam.wpp);
  }

  /** On demand — these views are static, so InterpPage gives them no rAF.
   *  Every input that changes what is on screen must call this. */
  render(): void {
    const r = this.renderer;
    if (!r || this.disposed) return;
    if (this.bloomOn && this.bloomPipe) this.bloomPipe.post.render();
    else r.render(this.scene, this.camera);
  }

  dispose(): void {
    this.disposed = true;
    this.unsubscribe?.();
    this.unsubscribe = null;
    this.teardownField();
    this.bloomPipe?.dispose();
    this.bloomPipe = null;
    this.renderer?.dispose();
    this.renderer = null;
  }

  private applyFit(): void {
    const box = this.box;
    if (!box) return;
    const { inset } = box;
    const cssW = this.cam.viewportW;
    const cssH = this.cam.viewportH;
    const availW = Math.max(1, cssW - inset.left - inset.right);
    const availH = Math.max(1, cssH - inset.top - inset.bottom);
    const spanX = Math.max(1e-3, box.maxX - box.minX);
    const spanY = Math.max(1e-3, box.maxY - box.minY);
    // one scale for both axes: px per world unit
    const z = Math.max(1, Math.min(availW / spanX, availH / spanY) * box.fit);
    const wpp = 1 / z;
    const dataCX = (box.minX + box.maxX) / 2;
    const dataCY = (box.minY + box.maxY) / 2;
    // the data box centres on the GUTTERED area, not on the viewport, so the
    // camera centre carries that offset — this is the inverse of worldToScreen,
    // and is exactly the offset the deck.gl viewState used to carry
    const drawCX = inset.left + availW / 2;
    const drawCY = inset.top + availH / 2;
    this.cam.wpp = wpp;
    this.cam.cx = dataCX + (cssW / 2 - drawCX) * wpp;
    this.cam.cy = dataCY + (drawCY - cssH / 2) * wpp;
    this.uWpp.value = wpp;
    this.syncCamera();
  }

  private syncCamera(): void {
    const [hx, hy] = this.cam.halfExtents();
    const c = this.camera;
    c.left = -hx;
    c.right = hx;
    c.top = hy;
    c.bottom = -hy;
    c.position.set(this.cam.cx, this.cam.cy, 10);
    c.updateProjectionMatrix();
    c.updateMatrixWorld();
  }

  private teardownField(): void {
    if (this.sprite) {
      this.scene.remove(this.sprite);
      // NEVER sprite.geometry.dispose() — every THREE.Sprite in the app shares
      // ONE module-level quad, so disposing it here blanks the atlas, the
      // sessions field and every other sprite the moment this view unmounts.
      this.sprite = null;
    }
    this.material?.dispose();
    this.material = null;
    this.attrA = null;
    this.attrB = null;
    this.picker = null;
    this.count = 0;
  }
}
