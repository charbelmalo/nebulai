/** ChartStage — the 3-D counterpart to `field2d.ts`.
 *
 *  `field2d` gave the flat views one shared emissive stack: a WebGPU renderer
 *  with a free WebGL2 rung, a TSL appearance graph, the bloom chain, and a
 *  world→screen projector so a DOM overlay can never drift from the GPU scene.
 *  This is the same stack under a PERSPECTIVE orbit camera, drawing extruded
 *  columns on a lattice instead of billboarded motes.
 *
 *  WHAT IT WILL AND WILL NOT DRAW
 *  ------------------------------
 *  Discrete extrusion, yes. Interpolated surface, no. Every column is one
 *  measured value with a footprint you can point at; nothing is drawn between
 *  two samples. A smooth mesh through the same points would invent a reading at
 *  every pixel between them, and at a glancing camera angle those invented
 *  pixels are most of the frame. That constraint is why this is a bar stage and
 *  not a heightfield, and it is not negotiable per-consumer.
 *
 *  DIFFERENCES FROM field2d THAT ARE NOT TASTE
 *    - Columns are OPAQUE and depth-tested. Additive motes may pile up because
 *      overlapping evidence should accumulate; overlapping solids must occlude,
 *      or the depth ordering — the only reason to be in 3-D — is a lie.
 *    - `NeutralToneMapping` is still mandatory. Emissive drive runs past 1.0 on
 *      purpose so the bloom threshold has something to key on, and without a
 *      tone map that headroom clips to flat white.
 *    - Picking is a real raycast against the instanced boxes, so a column
 *      hidden behind a taller one in front is NOT pickable. That is the honest
 *      answer: you cannot click what you cannot see.
 *
 *  The stage owns the renderer, camera, frame, probe and bars. Anything a
 *  consumer adds to `scene` is the consumer's to dispose.
 */

import * as THREE from "three/webgpu";
import {
  float,
  instancedDynamicBufferAttribute,
  mix,
  normalGeometry,
  uniform,
  vec3,
} from "three/tsl";
import { appStore } from "../../app/store";
import type { GpuTier } from "@psychix/viz/capabilities";
import { GestureRecognizer } from "../gestures";
import { createBloomPipeline, type BloomOptions, type BloomPipeline } from "../post/bloom";
import { BG } from "@psychix/viz/tokens";
import type { Inset } from "./field2d";

/** Per-view look. Required except where marked, for the same reason
 *  `Field2DLook` is: these are encoding decisions, and a default would let a
 *  consumer ship one without having made it. */
export interface ChartStageLook {
  /** Cage + floor grid. Dim enough to locate the data without competing. */
  frameColor: number;
  frameOpacity: number;
  /** Emissive multiplier at glow 0 and glow 1. Above 1.0 on purpose — that
   *  headroom is what the bloom threshold keys on. */
  emissiveMin: number;
  emissiveMax: number;
  /** Brightness a de-emphasised column falls back to. Never 0: a dimmed column
   *  is still evidence, and a view that deletes its unselected mass is lying
   *  about how much of it there is. */
  dimLevel: number;
  /** Default orbit. `elevation` is clamped so the camera never crosses the
   *  floor plane — from underneath, a bar chart reads inverted. */
  azimuth: number;
  elevation: number;
  /** Vertical field of view, degrees. Narrow keeps the lattice from splaying. */
  fov: number;
  bloom?: BloomOptions;
}

/** One column per entry. All arrays are indexed together and must agree on
 *  `count`; a short array is read as its default rather than throwing, because
 *  a half-drawn chart is easier to diagnose than a blank one. */
export interface BarData {
  count: number;
  /** Lattice position of the column's centre, 2 per bar: (x, z) in world units. */
  pos: Float32Array;
  /** Column height in world units. Clamped to `MIN_H` — see `writeMatrices`. */
  height: Float32Array;
  /** Linear rgb 0..1, 3 per bar. */
  color: Float32Array;
  /** 0..1 emissive drive. Usually the same normalized magnitude as `height`,
   *  so colour and extrusion cannot disagree about one measurement. */
  glow: Float32Array;
  /** 1 = foreground, 0 = dimmed to `dimLevel`. Optional; absent means all 1. */
  active?: Float32Array;
  /** Footprint of one column, world units. */
  cellX: number;
  cellZ: number;
}

/** The lattice the frame is drawn around. Separate from `BarData` because the
 *  cage is a property of the axes, not of which cells happen to be occupied —
 *  #23 draws only the causal lower triangle but must still be caged by the full
 *  T×T grid, or the empty half reads as "off the chart" instead of "excluded". */
export interface Lattice {
  /** Half-extent in x and z; the lattice is centred on the origin. */
  halfX: number;
  halfZ: number;
  /** Top of the cage in world units — the height axis's 1.0. */
  cageY: number;
  /** Grid line count along each axis, excluding the border. */
  divX: number;
  divZ: number;
}

/** A zero-scale instance matrix is singular, and `InstancedMesh.raycast`
 *  inverts it — so a value of exactly 0 would produce NaN and silently break
 *  picking for the whole mesh. Clamping leaves a visible floor plate, which is
 *  the right reading anyway: the cell exists and measured ~0. */
const MIN_H = 1e-4;
const EL_MIN = 0.08; // never cross the floor plane
const EL_MAX = 1.45; // and never quite reach straight down

/** Height-axis post placement — see `heightPost`. The minimum offset clears the
 *  ±0.5 name labels the data axes hang off the cage edges; the gap is how far
 *  outside the silhouette the post has to land before it counts as clear. */
const POST_OUT_MIN = 0.55;
const POST_GAP_PX = 14;
/** Four steps of ~0.22·half is a little under one cage radius of travel. Past
 *  that the ticks are so far off the lattice they stop reading as its axis — a
 *  post the eye cannot connect to the columns measures nothing, so we take the
 *  best gap we found rather than keep walking into open space. */
const POST_WALK_STEPS = 4;

/** Distance from a point to a segment, in screen px. */
function segDist(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const vx = bx - ax;
  const vy = by - ay;
  const len = vx * vx + vy * vy;
  const t = len > 0 ? Math.max(0, Math.min(1, ((px - ax) * vx + (py - ay) * vy) / len)) : 0;
  return Math.hypot(px - (ax + t * vx), py - (ay + t * vy));
}

/** Fixed key light, in world space. The lattice never rotates — only the camera
 *  does — so a fixed direction gives every column the same top/side/front
 *  ramp from any angle, which is what makes the extrusion legible. A
 *  camera-relative light would flatten the columns as you orbit. */
const KEY = new THREE.Vector3(0.38, 0.86, 0.34).normalize();
/** Floor of the lambert term. Faces pointing away are shaded, never black —
 *  a column's dark side still has to show its colour. */
const AMBIENT = 0.42;

export class ChartStage {
  private renderer: THREE.WebGPURenderer | null = null;
  readonly scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private bloomPipe: BloomPipeline | null = null;
  private bloomOn = false;
  private disposed = false;
  private unsubscribe: (() => void) | null = null;

  private bars: THREE.InstancedMesh | null = null;
  private barMat: THREE.MeshBasicNodeMaterial | null = null;
  private frame: THREE.LineSegments | null = null;
  private frameMat: THREE.LineBasicNodeMaterial | null = null;
  private probe: THREE.LineSegments | null = null;
  private probeMat: THREE.LineBasicNodeMaterial | null = null;

  /** (r, g, b, glow) and (active, —, —, —). With BoxGeometry's
   *  position/normal/uv and the instance matrix that is 6 vertex buffers of
   *  WebGPU's 8. The limit fails SILENTLY — an invalid pipeline draws zero
   *  fragments — so the headroom is deliberate, not slack. */
  private attrA: THREE.InstancedBufferAttribute | null = null;
  private attrB: THREE.InstancedBufferAttribute | null = null;
  private count = 0;
  private cell = { x: 1, z: 1 };
  private lastBars: BarData | null = null;

  private uDim = uniform(1);

  // orbit state
  private az: number;
  private el: number;
  private dist = 4;
  private baseDist = 4; // the fit distance; zoom is a multiple of it
  private zoom = 1;
  private pan = new THREE.Vector2(); // camera-local right/up offset, world units
  private lattice: Lattice | null = null;
  private inset: Inset = { left: 0, top: 0, right: 0, bottom: 0 };
  private fitSlack = 1;

  private cssW = 1;
  private cssH = 1;
  private dragging = false;
  private dragMoved = false;
  private last: { x: number; y: number } | null = null;
  private abort = new AbortController();
  private raycaster = new THREE.Raycaster();
  /** touch orbit/zoom — see src/scene/gestures.ts; the mouse path is untouched.
   *  Shared by every ChartStage consumer (AttentionRolloutDriver,
   *  ResidualRibbonDriver), so wiring it once here gives both touch for free. */
  private gestures: GestureRecognizer | null = null;

  /** Fired after any camera change, so a consumer can relayout its DOM overlay
   *  in the same tick the pixels moved. */
  onCamera: (() => void) | null = null;
  /** Fired on a pointerdown/up pair that did not drag — a click, not an orbit. */
  onClick: ((sx: number, sy: number) => void) | null = null;

  constructor(private look: ChartStageLook) {
    this.az = look.azimuth;
    this.el = Math.min(EL_MAX, Math.max(EL_MIN, look.elevation));
    this.camera = new THREE.PerspectiveCamera(look.fov, 1, 0.02, 200);
  }

  async init(canvas: HTMLCanvasElement, tier: GpuTier): Promise<void> {
    const webgpu = tier === "webgpu";
    const renderer = new THREE.WebGPURenderer({
      canvas,
      antialias: true,
      // TSL lowers the same node graph to GLSL, so the fallback rung is free —
      // minus post-processing, which is why bloom is gated on `webgpu` below.
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
    this.syncCamera();

    this.bloomOn = webgpu && appStore.getState().settings.bloom;
    if (webgpu) {
      this.bloomPipe = createBloomPipeline(
        renderer,
        this.scene,
        this.camera,
        "full",
        this.look.bloom ?? {},
      );
    }
    // no rAF here — these views are static between inputs, so a settings
    // change has to redraw itself
    this.unsubscribe = appStore.subscribe((s) => {
      const want = webgpu && s.settings.bloom;
      if (want === this.bloomOn) return;
      this.bloomOn = want;
      this.render();
    });

    this.attachPointer(canvas);
  }

  // ── data ──────────────────────────────────────────────────────────────────

  /** Build or update the columns. Reuses the buffers when `count` is unchanged,
   *  which is the common case: scrubbing an ordinal axis rewrites every value
   *  but never the cell set. */
  setBars(d: BarData): void {
    this.cell = { x: d.cellX, z: d.cellZ };
    if (d.count !== this.count || !this.bars) {
      this.teardownBars();
      this.count = d.count;
      if (d.count === 0) return;
      this.buildBars(d.count);
    }
    // After the teardown, which clears it. Kept for `heightPost`, which has to
    // know where the columns actually are. Safe to hold: consumers reuse these
    // buffers and re-enter setBars on every change, so the reference is never a
    // stale copy of a live array.
    this.lastBars = d;
    this.writeMatrices(d);
    this.writeAppearance(d);
  }

  /** Re-upload just the foreground mask — no rebuild, no matrix churn. */
  setActive(active: Float32Array | null): void {
    const attr = this.attrB;
    if (!attr) return;
    const arr = attr.array as Float32Array;
    for (let i = 0; i < this.count; i++) arr[i * 4] = active ? (active[i] ?? 1) : 1;
    attr.needsUpdate = true;
  }

  /** Global brightness, for stepping the whole lattice back behind a focus. */
  setDim(level: number): void {
    this.uDim.value = level;
  }

  private buildBars(count: number): void {
    // Unit column standing ON the floor: the box is translated so y ∈ [0, 1],
    // which makes the instance matrix a plain scale — no origin correction, and
    // no chance of a bar sinking half its height through the grid.
    const geo = new THREE.BoxGeometry(1, 1, 1);
    geo.translate(0, 0.5, 0);

    const a = new Float32Array(count * 4);
    const b = new Float32Array(count * 4);
    this.attrA = new THREE.InstancedBufferAttribute(a, 4);
    this.attrB = new THREE.InstancedBufferAttribute(b, 4);
    this.attrA.setUsage(THREE.DynamicDrawUsage);
    this.attrB.setUsage(THREE.DynamicDrawUsage);
    // DYNAMIC on both: scrubbing an ordinal axis rewrites every value every
    // step, and the static node caches its buffer — the columns would freeze at
    // whatever the first frame happened to hold.
    const iA = instancedDynamicBufferAttribute<"vec4">(this.attrA, "vec4");
    const iB = instancedDynamicBufferAttribute<"vec4">(this.attrB, "vec4");

    const L = this.look;
    // The lattice never rotates, so the geometry normal IS the world normal and
    // the key light can be a constant. Edges must INCREASE in TSL smoothsteps;
    // this one is a plain dot, but the same rule bit the 2-D field.
    const lambert = normalGeometry.dot(vec3(KEY.x, KEY.y, KEY.z)).max(0);
    const shade = float(AMBIENT).add(lambert.mul(1 - AMBIENT));
    const emissive = mix(float(L.emissiveMin), float(L.emissiveMax), iA.w);
    const fg = mix(float(L.dimLevel), float(1), iB.x).mul(this.uDim);

    const material = new THREE.MeshBasicNodeMaterial({
      transparent: false,
      depthWrite: true,
      depthTest: true,
    });
    material.colorNode = iA.xyz.mul(shade).mul(emissive).mul(fg);
    this.barMat = material;

    // The attribute NODES own their buffers — deliberately not also
    // `geometry.setAttribute`, which would bind each one a second time and
    // spend two more of the eight vertex-buffer slots for nothing.
    const mesh = new THREE.InstancedMesh(geo, material, count);
    mesh.count = count;
    mesh.frustumCulled = false;
    mesh.renderOrder = 2;
    this.bars = mesh;
    this.scene.add(mesh);
  }

  private tmpM = new THREE.Matrix4();

  private writeMatrices(d: BarData): void {
    const mesh = this.bars;
    if (!mesh) return;
    for (let i = 0; i < d.count; i++) {
      const h = Math.max(MIN_H, d.height[i] ?? 0);
      this.tmpM.makeScale(d.cellX, h, d.cellZ);
      this.tmpM.setPosition(d.pos[i * 2] ?? 0, 0, d.pos[i * 2 + 1] ?? 0);
      mesh.setMatrixAt(i, this.tmpM);
    }
    mesh.instanceMatrix.needsUpdate = true;
    // raycast tests this sphere first; a stale one silently kills picking the
    // moment a column grows past the old radius
    mesh.computeBoundingSphere();
  }

  private writeAppearance(d: BarData): void {
    const a = this.attrA?.array as Float32Array | undefined;
    const b = this.attrB?.array as Float32Array | undefined;
    if (!a || !b) return;
    for (let i = 0; i < d.count; i++) {
      a[i * 4] = d.color[i * 3] ?? 1;
      a[i * 4 + 1] = d.color[i * 3 + 1] ?? 1;
      a[i * 4 + 2] = d.color[i * 3 + 2] ?? 1;
      a[i * 4 + 3] = d.glow[i] ?? 0;
      b[i * 4] = d.active ? (d.active[i] ?? 1) : 1;
    }
    if (this.attrA) this.attrA.needsUpdate = true;
    if (this.attrB) this.attrB.needsUpdate = true;
  }

  // ── frame ─────────────────────────────────────────────────────────────────

  /** The cage: floor grid, floor border, four corner posts and a top ring. The
   *  posts and ring are what a height tick has to hang off — without them the
   *  vertical axis has no visible extent and a column's height is unreadable. */
  setLattice(l: Lattice): void {
    this.lattice = l;
    const pts: number[] = [];
    const { halfX, halfZ, cageY, divX, divZ } = l;
    for (let i = 1; i < divX; i++) {
      const x = -halfX + (i / divX) * halfX * 2;
      pts.push(x, 0, -halfZ, x, 0, halfZ);
    }
    for (let i = 1; i < divZ; i++) {
      const z = -halfZ + (i / divZ) * halfZ * 2;
      pts.push(-halfX, 0, z, halfX, 0, z);
    }
    // border
    pts.push(-halfX, 0, -halfZ, halfX, 0, -halfZ);
    pts.push(halfX, 0, -halfZ, halfX, 0, halfZ);
    pts.push(halfX, 0, halfZ, -halfX, 0, halfZ);
    pts.push(-halfX, 0, halfZ, -halfX, 0, -halfZ);
    // corner posts + top ring
    for (const sx of [-halfX, halfX]) {
      for (const sz of [-halfZ, halfZ]) pts.push(sx, 0, sz, sx, cageY, sz);
    }
    pts.push(-halfX, cageY, -halfZ, halfX, cageY, -halfZ);
    pts.push(halfX, cageY, -halfZ, halfX, cageY, halfZ);
    pts.push(halfX, cageY, halfZ, -halfX, cageY, halfZ);
    pts.push(-halfX, cageY, halfZ, -halfX, cageY, -halfZ);

    this.frame?.geometry.dispose();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    if (this.frame) {
      this.frame.geometry = geo;
    } else {
      const mat = new THREE.LineBasicNodeMaterial({
        color: new THREE.Color(this.look.frameColor),
        transparent: true,
        opacity: this.look.frameOpacity,
        depthWrite: false,
      });
      this.frameMat = mat;
      const seg = new THREE.LineSegments(geo, mat);
      seg.frustumCulled = false;
      seg.renderOrder = 1;
      this.frame = seg;
      this.scene.add(seg);
    }
    this.applyFit();
  }

  /** Rails from one column's top to the floor and to both axis walls. This is
   *  how a value is read off the cage in 3-D — the flat views used a crosshair,
   *  which has no meaning once the camera can rotate. Pass null to clear. */
  setProbe(x: number, y: number, z: number): void;
  setProbe(x: null): void;
  setProbe(x: number | null, y = 0, z = 0): void {
    const l = this.lattice;
    if (!this.probe) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(6 * 3), 3));
      const mat = new THREE.LineBasicNodeMaterial({
        color: 0xff5c7a, // --danger, the app's marker hue — never a ramp colour
        transparent: true,
        opacity: 0.7,
        depthWrite: false,
        depthTest: false,
      });
      this.probeMat = mat;
      const seg = new THREE.LineSegments(geo, mat);
      seg.frustumCulled = false;
      seg.renderOrder = 4;
      seg.visible = false;
      this.probe = seg;
      this.scene.add(seg);
    }
    if (x === null || !l) {
      this.probe.visible = false;
      return;
    }
    const attr = this.probe.geometry.getAttribute("position") as THREE.BufferAttribute;
    (attr.array as Float32Array).set([
      x, y, z, x, 0, z,            // down to the floor
      x, y, z, -l.halfX, y, z,     // across to the left wall, at height
      x, y, z, x, y, -l.halfZ,     // across to the back wall, at height
    ]);
    attr.needsUpdate = true;
    this.probe.visible = true;
  }

  // ── camera ────────────────────────────────────────────────────────────────

  /** Frame the lattice inside asymmetric px gutters. The gutters are honoured
   *  by PANNING the camera (translating eye and target together), not by
   *  skewing the frustum — an off-centre frustum would shear the columns, and
   *  a sheared bar is a wrong bar. */
  fitInset(inset: Inset, fit = 1): void {
    this.inset = inset;
    this.fitSlack = fit;
    this.applyFit();
  }

  /** Current orbit, for a consumer that wants to persist or report it. */
  get orbit(): { az: number; el: number; dist: number } {
    return { az: this.az, el: this.el, dist: this.dist };
  }

  resize(w: number, h: number, dpr: number): void {
    this.cssW = Math.max(1, w);
    this.cssH = Math.max(1, h);
    const r = this.renderer;
    if (r) {
      r.setPixelRatio(Math.min(dpr, 2));
      r.setSize(w, h, false);
    }
    this.camera.aspect = this.cssW / this.cssH;
    this.applyFit();
  }

  /** The eight corners of the cage. Nothing the stage draws reaches outside
   *  them — columns are clamped to `cageY` by their own normalization — so
   *  these bound the whole silhouette. */
  private cageCorners(l: Lattice): THREE.Vector3[] {
    const out: THREE.Vector3[] = [];
    for (const x of [-l.halfX, l.halfX])
      for (const y of [0, l.cageY])
        for (const z of [-l.halfZ, l.halfZ]) out.push(new THREE.Vector3(x, y, z));
    return out;
  }

  /** Screen-space bounding box of a point set, or null if any point is behind
   *  the camera (in which case a fit computed from it would be nonsense). */
  private projectBox(
    pts: THREE.Vector3[],
  ): { cx: number; cy: number; w: number; h: number } | null {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const p of pts) {
      const s = this.project(p.x, p.y, p.z);
      if (!s) return null;
      minX = Math.min(minX, s[0]);
      maxX = Math.max(maxX, s[0]);
      minY = Math.min(minY, s[1]);
      maxY = Math.max(maxY, s[1]);
    }
    return { cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, w: maxX - minX, h: maxY - minY };
  }

  /** Where to hang the height axis: a vertical post outside the cage, picked by
   *  MEASUREMENT rather than by choosing a corner and hoping.
   *
   *  The obvious choice is the one corner neither data axis occupies, and on
   *  #23's square lattice that is right — the ticks land over the acausal half,
   *  which by construction has nothing standing on it. On #8's 13×11 grid the
   *  same corner is the top of the tallest column in the chart, and walking it
   *  outward moves it almost straight UP the screen, so it never escapes.
   *
   *  A bounding box cannot tell those two apart: both posts project inside the
   *  silhouette rectangle, because one tall column at the far edge stretches
   *  that rectangle over ground it does not occupy. So clearance is measured
   *  against the COLUMNS — the smallest screen distance from the post's tick
   *  band to any column top. That is the question actually being asked, and it
   *  answers it for any lattice shape at any orbit.
   *
   *  `side` is measured too, so orbiting can never swing the labels back inward
   *  over the data.
   *
   *  @param busyX sign of the x edge a data axis already occupies
   *  @param busyZ sign of the z edge a data axis already occupies */
  heightPost(busyX: -1 | 1, busyZ: -1 | 1): { x: number; z: number; side: "left" | "right" } | null {
    const l = this.lattice;
    if (!l) return null;
    const centre = this.project(0, l.cageY / 2, 0);
    if (!centre) return null;
    const sx = -busyX;
    const step = Math.max(l.halfX, l.halfZ) * 0.22;

    // Project every column top ONCE — heightPost runs on every camera change,
    // and re-projecting the whole lattice per candidate per walk step would put
    // a few thousand matrix multiplies inside a drag.
    const tops: number[] = [];
    const d = this.lastBars;
    if (d) {
      for (let i = 0; i < d.count; i++) {
        const p = this.project(
          d.pos[i * 2] ?? 0,
          Math.max(MIN_H, d.height[i] ?? 0),
          d.pos[i * 2 + 1] ?? 0,
        );
        if (p) tops.push(p[0], p[1]);
      }
    }

    const at = (sz: number, out: number) => ({ x: sx * (l.halfX + out), z: sz * (l.halfZ + out) });
    /** Smallest screen distance from this candidate's tick band to any column
     *  top. -Infinity if the post is off the canvas, where no label is any use
     *  however clear of the data it is. */
    const clearance = (sz: number, out: number): number => {
      const c = at(sz, out);
      const a = this.project(c.x, 0, c.z);
      const b = this.project(c.x, l.cageY, c.z);
      if (!a || !b) return -Infinity;
      // Both ends, both axes. Checking x alone let a candidate win on gap while
      // its top decade sat above the canvas, where the label host hides it —
      // an axis clear of every column and missing its largest tick.
      for (const q of [a, b]) {
        if (q[0] < 0 || q[0] > this.cssW || q[1] < 0 || q[1] > this.cssH) return -Infinity;
      }
      let min = Math.max(this.cssW, this.cssH);
      for (let i = 0; i < tops.length; i += 2) {
        const q = segDist(tops[i]!, tops[i + 1]!, a[0], a[1], b[0], b[1]);
        if (q < min) min = q;
      }
      return min;
    };
    // Both candidates are walked outward TOGETHER, one step at a time, and the
    // first that clears wins. Walking one to exhaustion first would let it
    // "succeed" by sheer distance — on #8 that parked the axis six units past
    // the cage, where the ticks float with no post under them and measure
    // nothing. Nearest-that-clears is the property worth having; ties go to the
    // corner neither data axis touches.
    let sz = -busyZ;
    let out = POST_OUT_MIN;
    let best = { sz, out, gap: -Infinity };
    outer: for (let i = 0; i <= POST_WALK_STEPS; i++) {
      const o = POST_OUT_MIN + i * step;
      for (const s of [-busyZ, busyZ]) {
        const gap = clearance(s, o);
        if (gap > best.gap) best = { sz: s, out: o, gap };
        if (gap >= POST_GAP_PX) {
          sz = s;
          out = o;
          break outer;
        }
      }
      if (i === POST_WALK_STEPS) {
        sz = best.sz;
        out = best.out;
      }
    }
    const c = at(sz, out);
    const p = this.project(c.x, l.cageY / 2, c.z);
    return { x: c.x, z: c.z, side: !p || p[0] <= centre[0] ? "left" : "right" };
  }

  /** Fit by MEASURING the projected silhouette, not by bounding-sphere algebra.
   *  A sphere that contains the cage is 30–40% larger than the cage's actual
   *  screen footprint — the cage is a slab and its corners come nowhere near
   *  filling the sphere — so the algebraic fit parks the camera a third too far
   *  back and the chart floats in a sea of margin. Distance and projection are
   *  coupled (screen size ≈ k/dist), so a few ratio corrections converge; the
   *  loop measures rather than assuming the relationship is exact.
   *
   *  The centring pass is separate and comes after, because a rigid pan shifts
   *  near and far points by different screen amounts — the true silhouette
   *  centre is not the world centre, and only iterating finds it. */
  private applyFit(): void {
    const l = this.lattice;
    if (!l) return;
    const inset = this.inset;
    const availW = Math.max(1, this.cssW - inset.left - inset.right);
    const availH = Math.max(1, this.cssH - inset.top - inset.bottom);
    const drawCX = inset.left + availW / 2;
    const drawCY = inset.top + availH / 2;
    const vFov = (this.camera.fov * Math.PI) / 180;
    const corners = this.cageCorners(l);

    this.silent = true;
    this.pan.set(0, 0);
    // seed well back, so the first projection is guaranteed to be in front of
    // the camera and the loop can only tighten
    let dist = Math.hypot(l.halfX, l.cageY, l.halfZ) / Math.sin(vFov / 2);
    for (let pass = 0; pass < 6; pass++) {
      this.dist = dist;
      this.syncCamera();
      const b = this.projectBox(corners);
      if (!b || b.w < 1e-3 || b.h < 1e-3) break;
      const grow = Math.max(b.w / availW, b.h / availH);
      dist *= grow;
      if (Math.abs(grow - 1) < 0.003) break;
    }
    this.baseDist = dist / this.fitSlack;
    this.dist = this.baseDist * this.zoom;
    this.silent = false;
    this.recentre();
  }

  /** Put the silhouette's centre on the drawable box's centre. Split out from
   *  `applyFit` because orbiting must NOT change the distance: a chart that
   *  rescales itself while you turn it cannot be compared with itself one
   *  moment earlier. Re-centring is free of that problem and keeps the lattice
   *  from wandering under the gutters as it rotates. */
  private recentre(): void {
    const l = this.lattice;
    if (!l) return;
    const inset = this.inset;
    const availW = Math.max(1, this.cssW - inset.left - inset.right);
    const availH = Math.max(1, this.cssH - inset.top - inset.bottom);
    const drawCX = inset.left + availW / 2;
    const drawCY = inset.top + availH / 2;
    const vFov = (this.camera.fov * Math.PI) / 180;
    const corners = this.cageCorners(l);
    this.silent = true;
    this.pan.set(0, 0);
    for (let pass = 0; pass < 4; pass++) {
      this.syncCamera();
      const b = this.projectBox(corners);
      if (!b) break;
      const dx = b.cx - drawCX;
      const dy = b.cy - drawCY;
      if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5) break;
      // world units per CSS px at the target plane — the conversion is exact
      // only at that depth, which is why this is a loop and not a formula
      const wpp = (2 * this.dist * Math.tan(vFov / 2)) / this.cssH;
      // move the camera the way we want the CONTENT to go on x, and the
      // opposite way on y — screen y runs down, the camera's up axis runs up
      this.pan.x += dx * wpp;
      this.pan.y -= dy * wpp;
    }
    this.silent = false;
    this.syncCamera();
  }

  private silent = false;

  private syncCamera(): void {
    const l = this.lattice;
    const cy = l ? l.cageY / 2 : 0;
    const ce = Math.cos(this.el);
    const eye = new THREE.Vector3(
      Math.sin(this.az) * ce * this.dist,
      Math.sin(this.el) * this.dist + cy,
      Math.cos(this.az) * ce * this.dist,
    );
    const target = new THREE.Vector3(0, cy, 0);
    // pan in the camera's own basis, applied to eye AND target so the view
    // translates rather than rotates
    const fwd = target.clone().sub(eye).normalize();
    const right = fwd.clone().cross(new THREE.Vector3(0, 1, 0)).normalize();
    const up = right.clone().cross(fwd).normalize();
    const off = right.multiplyScalar(this.pan.x).add(up.multiplyScalar(this.pan.y));
    eye.add(off);
    target.add(off);
    this.camera.position.copy(eye);
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(target);
    this.camera.updateProjectionMatrix();
    this.camera.updateMatrixWorld();
    // `applyFit` moves the camera several times to converge; firing the
    // consumer's relayout on each intermediate pose would lay out labels
    // against poses that were never on screen
    if (!this.silent) this.onCamera?.();
  }

  // ── interaction ───────────────────────────────────────────────────────────

  private attachPointer(el: HTMLCanvasElement): void {
    const sig = this.abort.signal;

    // touch is handled once, by the shared recognizer (see gestures.ts) — the
    // pointer handlers below early-return on pointerType "touch" so the two
    // paths never fight over the same drag.
    this.gestures = new GestureRecognizer({
      onPan: (dx, dy) => {
        this.az -= dx * 0.006;
        this.el = Math.min(EL_MAX, Math.max(EL_MIN, this.el + dy * 0.006));
        this.recentre(); // pan only — turning the lattice must not rescale it
        this.render();
      },
      onPinch: (e) => {
        this.zoom = Math.min(2.4, Math.max(0.35, this.zoom / e.scale));
        this.az -= e.dcx * 0.006;
        this.el = Math.min(EL_MAX, Math.max(EL_MIN, this.el + e.dcy * 0.006));
        this.applyFit(); // the pan is distance-dependent, so it re-converges too
        this.render();
      },
    });
    this.gestures.attach(el, sig);

    el.addEventListener(
      "pointerdown",
      (e) => {
        if (e.pointerType === "touch") return;
        this.dragging = true;
        this.dragMoved = false;
        this.last = { x: e.clientX, y: e.clientY };
        el.setPointerCapture(e.pointerId);
      },
      { signal: sig },
    );
    el.addEventListener(
      "pointermove",
      (e) => {
        if (e.pointerType === "touch") return;
        if (!this.dragging || !this.last) return;
        const dx = e.clientX - this.last.x;
        const dy = e.clientY - this.last.y;
        if (Math.abs(dx) + Math.abs(dy) > 2) this.dragMoved = true;
        this.az -= dx * 0.006;
        this.el = Math.min(EL_MAX, Math.max(EL_MIN, this.el + dy * 0.006));
        this.last = { x: e.clientX, y: e.clientY };
        this.recentre(); // pan only — turning the lattice must not rescale it
        this.render();
      },
      { signal: sig },
    );
    const end = (e: PointerEvent) => {
      if (e.pointerType === "touch") return;
      const wasClick = this.dragging && !this.dragMoved;
      this.dragging = false;
      this.last = null;
      if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
      if (wasClick) {
        const r = el.getBoundingClientRect();
        this.onClick?.(e.clientX - r.left, e.clientY - r.top);
      }
    };
    el.addEventListener("pointerup", end, { signal: sig });
    el.addEventListener("pointercancel", end, { signal: sig });
    el.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.zoom = Math.min(2.4, Math.max(0.35, this.zoom * Math.exp(e.deltaY * 0.0012)));
        this.applyFit(); // the pan is distance-dependent, so it re-converges too
        this.render();
      },
      { signal: sig, passive: false },
    );
  }

  /** Whether the pointer is mid-orbit. A consumer must not treat a drag as a
   *  hover — the tooltip would chase the camera instead of the data. */
  get isDragging(): boolean {
    return this.dragging;
  }

  /** Index of the frontmost column under a CSS-pixel position, or -1. A real
   *  raycast, so occlusion is respected: a column you cannot see is not a
   *  column you can pick. */
  pickAt(sx: number, sy: number): number {
    const mesh = this.bars;
    if (!mesh || this.count === 0) return -1;
    this.raycaster.setFromCamera(
      new THREE.Vector2((sx / this.cssW) * 2 - 1, -(sy / this.cssH) * 2 + 1),
      this.camera,
    );
    const hits = this.raycaster.intersectObject(mesh, false);
    const id = hits[0]?.instanceId;
    return id === undefined ? -1 : id;
  }

  /** Project world → CSS px, so a DOM overlay can never drift from the GPU
   *  scene. Null when the point is behind the camera — the caller must hide
   *  that label rather than draw it at a wrapped-around position. */
  project(x: number, y: number, z: number): [number, number] | null {
    const p = new THREE.Vector3(x, y, z).project(this.camera);
    if (p.z > 1) return null;
    return [(p.x * 0.5 + 0.5) * this.cssW, (-p.y * 0.5 + 0.5) * this.cssH];
  }

  /** On demand — these views are static between inputs, so the host gives them
   *  no rAF. Every input that changes what is on screen must call this. */
  render(): void {
    const r = this.renderer;
    if (!r || this.disposed || this.cssW < 2 || this.cssH < 2) return;
    if (this.bloomOn && this.bloomPipe) this.bloomPipe.post.render();
    else r.render(this.scene, this.camera);
  }

  dispose(): void {
    this.disposed = true;
    this.abort.abort();
    this.gestures?.dispose();
    this.unsubscribe?.();
    this.unsubscribe = null;
    this.teardownBars();
    if (this.frame) {
      this.scene.remove(this.frame);
      // Safe to dispose here, unlike a Sprite's quad: this BufferGeometry is
      // built in setLattice and owned by exactly one stage.
      this.frame.geometry.dispose();
      this.frame = null;
    }
    this.frameMat?.dispose();
    this.frameMat = null;
    if (this.probe) {
      this.scene.remove(this.probe);
      this.probe.geometry.dispose();
      this.probe = null;
    }
    this.probeMat?.dispose();
    this.probeMat = null;
    this.bloomPipe?.dispose();
    this.bloomPipe = null;
    this.renderer?.dispose();
    this.renderer = null;
  }

  private teardownBars(): void {
    if (this.bars) {
      this.scene.remove(this.bars);
      this.bars.geometry.dispose(); // this stage's own BoxGeometry, not a shared quad
      this.bars.dispose();
      this.bars = null;
    }
    this.barMat?.dispose();
    this.barMat = null;
    this.attrA = null;
    this.attrB = null;
    this.count = 0;
    this.lastBars = null;
  }
}
