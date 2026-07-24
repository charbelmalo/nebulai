/** SessionFieldDriver — the agent-session plot as an emissive particle field.
 *
 *  This replaces the deck.gl scatter plot. The data and the axes are the same;
 *  what changes is that every turn no longer shouts. The old plot drew 1600
 *  equal-weight opaque discs in seven saturated hues, which is a texture, not a
 *  chart — there was no figure and no ground, so the eye had nowhere to land.
 *
 *  Here magnitude drives LUMINANCE, and that single change is what separates
 *  signal from field: most turns are dim motes, a few dozen blaze, and the
 *  bright ones are the ones worth looking at. Category still sets hue — a quiet
 *  turn keeps most of its colour and simply goes dim, because dimness alone
 *  already does the figure-ground work and a desaturated field would throw
 *  away the identification the hue exists for. Points are additive with a soft
 *  radial falloff, so where turns crowd together their light ACCUMULATES into a
 *  readable density instead of occluding each other into mush. The brightest
 *  cores exceed 1.0 and are picked up by the shared bloom pipeline — the same
 *  one the atlas uses, so the two views finally look like one product.
 *
 *    X = wall-clock time      Y = context window (cache-read)
 *    Z = new context this turn (cache-write)
 *    glow = output tokens, ranked (see outputRank)   size = tools this turn
 *    hue = dominant tool category
 *    red = a tool call failed here    faded = sub-agent turn
 *
 *  All three axes are asinh-scaled (see scales.ts): linear near zero, log in
 *  the tail, defined AT zero, monotone, and exactly invertible — so every tick
 *  is labelled by inverting the scale and reads a true value, and the tooltip
 *  always shows the raw number. That is what makes the idle-gap and long-tail
 *  problem go away without a piecewise axis or a lie.
 *
 *  Drop-lines exist only for the turn you're pointing at. One per node is a
 *  curtain that hides the trajectory it exists to locate.
 *
 *  Renders through three/webgpu + TSL, so one shader source serves both the
 *  WebGPU and the forceWebGL rungs (bloom rides only on the WebGPU rung, as in
 *  AtlasDriver).
 */

import * as THREE from "three/webgpu";
import {
  cameraPosition,
  float,
  instanceIndex,
  instancedBufferAttribute,
  instancedDynamicBufferAttribute,
  mix,
  positionGeometry,
  select,
  uniform,
  uv,
  vec3,
} from "three/tsl";
import { appStore } from "../../app/store";
import {
  dominantCategory,
  type SessionAnalysis,
  type SessionTurn,
  type ToolCategory,
} from "../../chrome/sessionlog";
import { createBloomPipeline, type BloomPipeline } from "../post/bloom";
import { IdPicker } from "../picking";
import { asinhScale, suggestK, type AxisScale } from "./scales";

/** Cube side in world units; the cube is centred on the origin. */
const CUBE = 1;
const HALF = CUBE / 2;

/** Category hue at full strength, 0–1 per channel. Quiet turns are washed
 *  toward NEUTRAL — see the saturation ramp in buildField(). */
const CATEGORY_RGB: Record<ToolCategory, [number, number, number]> = {
  orient: [0.36, 0.78, 0.93],
  plan: [0.96, 0.75, 0.36],
  edit: [0.49, 0.87, 0.59],
  exec: [0.78, 0.51, 0.94],
  deliver: [0.94, 0.47, 0.59],
  reflect: [0.59, 0.62, 0.71],
};
/** What a low-magnitude turn desaturates toward: cool, dim, recedes. Kept close
 *  in luminance to the hues above so the wash costs saturation, not identity. */
const NEUTRAL: [number, number, number] = [0.42, 0.52, 0.72];
/** The one place red is allowed. */
const ERROR_RGB: [number, number, number] = [1, 0.33, 0.3];

// ── tuning ───────────────────────────────────────────────────────────────────
const POINT_SIZE = 0.0105; // world units, before the glow ramp
/** Applied to the output-token RANK. 1.7 puts the median turn at 0.31 and the
 *  top decile above 0.84 — a dim field with a scattering of hot points, on any
 *  distribution. Below ~1.4 the field washes back out to the flat scatter this
 *  replaced; above ~2.4 the mid-range turns go black and the density read that
 *  makes a busy stretch legible goes with them. */
const GLOW_GAMMA = 1.7;
const BASE_ALPHA = 0.17;
const PEAK_ALPHA = 1;
/** Emissive multiplier at the two ends of the glow ramp. The ceiling is
 *  deliberately modest: additive blending already sums overlapping sprites, so
 *  a high ceiling clips every dense region to white and throws away the hue
 *  that carries category identity. Only the top of the ramp clears the bloom
 *  threshold (0.55), which is what makes a spike glow instead of everything. */
const EMISSIVE_MIN = 0.34;
const EMISSIVE_MAX = 2.1;
/** How much of its category hue a quiet turn keeps. A floor this high means
 *  dimness alone does the figure-ground work; the wash only takes the edge off,
 *  so a 1-tool `edit` still reads green rather than as generic haze. */
const SAT_FLOOR = 0.6;
const TRAIL_ALPHA = 0.028; // ambient: structure, not subject
const TRAIL_FOCUS_ALPHA = 0.5; // the run around the hovered/pinned turn
const TRAIL_FOCUS_SPAN = 14; // turns either side of it
const TRAIL_WIDTH = 0.0016;
const DIM_ALPHA = 0.16; // a category dimmed via the legend — never hidden
const FRAME_RGB = 0x2c3446;

/** The legend's swatch colour for a category, as CSS 0–255 — the single source
 *  of truth, so a legend chip and a node can never disagree. */
export const CATEGORY_CSS: Record<ToolCategory, string> = Object.fromEntries(
  (Object.keys(CATEGORY_RGB) as ToolCategory[]).map((c) => [
    c,
    `rgb(${CATEGORY_RGB[c].map((v) => Math.round(v * 255)).join(",")})`,
  ]),
) as Record<ToolCategory, string>;

export interface TurnRef {
  sessionId: string;
  index: number;
}

interface FieldNode {
  sessionId: string;
  sessionName: string;
  index: number;
  turn: SessionTurn;
  pos: THREE.Vector3;
}

interface Axes {
  x: AxisScale;
  y: AxisScale;
  z: AxisScale;
}

/** Per-axis asinh bend. 0 = plain linear. `null` = pick one from the data. */
export interface ScaleK {
  x: number | null;
  y: number | null;
  z: number | null;
}

export class SessionFieldDriver {
  onSelect: ((sel: TurnRef | null) => void) | null = null;
  onHover: ((sel: TurnRef | null) => void) | null = null;

  private canvas!: HTMLCanvasElement;
  private overlay!: HTMLElement;
  private renderer!: THREE.WebGPURenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(40, 1, 0.05, 60);
  private bloomPipe: BloomPipeline | null = null;
  private webgpu = false;

  // ── data ───────────────────────────────────────────────────────────────
  private analyses: SessionAnalysis[] = [];
  private nodes: FieldNode[] = [];
  private axes: Axes = { x: asinhScale(1, 0), y: asinhScale(1, 0), z: asinhScale(1, 0) };
  private scaleK: ScaleK = { x: null, y: null, z: null };

  // ── gpu objects ────────────────────────────────────────────────────────
  private field: THREE.Sprite | null = null;
  private fieldMat: THREE.SpriteNodeMaterial | null = null;
  private idSprite: THREE.Sprite | null = null;
  private trail: THREE.InstancedMesh | null = null;
  private trailMat: THREE.MeshBasicNodeMaterial | null = null;
  private frame3: THREE.LineSegments | null = null;
  private probe: THREE.LineSegments | null = null;
  private picker: IdPicker | null = null;

  private uHover = uniform(-1);
  private uSelected = uniform(-1);
  private uSize = uniform(POINT_SIZE);
  /** per-instance legend dimming, CPU-written */
  private visArray = new Float32Array(0);
  private visAttr: THREE.InstancedBufferAttribute | null = null;
  /** per-trail-segment emphasis, CPU-written on hover/pin */
  private focusArray = new Float32Array(0);
  private focusAttr: THREE.InstancedBufferAttribute | null = null;
  /** which (session, destination turn) each trail segment belongs to */
  private trailKeys: TurnRef[] = [];

  // ── camera / interaction ───────────────────────────────────────────────
  private az = -0.62;
  private el = 0.42;
  private dist = 2.35;
  private static readonly HOME = { az: -0.62, el: 0.42, dist: 2.35 };
  private static readonly EL_MIN = -1.35;
  private static readonly EL_MAX = 1.35;
  private static readonly DIST_MIN = 0.8;
  private static readonly DIST_MAX = 6;
  private cameraDirty = true;
  private cssW = 1;
  private cssH = 1;

  private dragging = false;
  private dragMoved = false;
  private last: { x: number; y: number } | null = null;
  private mouse: { x: number; y: number } | null = null;
  private hovered: number | null = null;
  private selected: number | null = null;
  private pickBusy = false;
  private lastPickAt = 0;

  private hiddenCats = new Set<ToolCategory>();
  private raf = 0;
  private disposed = false;
  private labels: HTMLElement[] = [];
  private tooltipEl: HTMLElement | null = null;
  private abort = new AbortController();

  async init(canvas: HTMLCanvasElement, overlay: HTMLElement): Promise<void> {
    this.canvas = canvas;
    this.overlay = overlay;
    const tier = appStore.getState().capabilities?.tier ?? "webgpu";
    this.webgpu = tier === "webgpu";

    this.renderer = new THREE.WebGPURenderer({
      canvas,
      antialias: true,
      alpha: true, // the page's own backdrop shows through
      forceWebGL: !this.webgpu,
    });
    await this.renderer.init();
    if (this.disposed) return;
    this.renderer.setClearColor(0x000000, 0);

    this.buildFrame();
    this.buildProbe();

    // bloom rides only the real WebGPU rung, exactly as in AtlasDriver
    if (this.webgpu) {
      this.bloomPipe = createBloomPipeline(this.renderer, this.scene, this.camera, "full");
    }

    this.tooltipEl = document.createElement("div");
    this.tooltipEl.className = "session-tooltip";
    this.tooltipEl.style.display = "none";
    this.overlay.appendChild(this.tooltipEl);

    this.attachPointer();
    this.loop();
  }

  // ── data in ──────────────────────────────────────────────────────────────

  setSessions(analyses: SessionAnalysis[]): void {
    this.analyses = analyses;
    this.rebuild();
  }

  /** Override the asinh bend on one or more axes (null = auto from the data).
   *  Display-only: it changes spacing, never a value. */
  setScaleK(k: Partial<ScaleK>): void {
    this.scaleK = { ...this.scaleK, ...k };
    this.rebuild();
  }

  setSelected(sel: TurnRef | null): void {
    const i = sel ? this.nodes.findIndex((n) => n.sessionId === sel.sessionId && n.index === sel.index) : -1;
    this.selected = i >= 0 ? i : null;
    this.uSelected.value = this.selected ?? -1;
    this.updateProbe();
    this.updateTrailFocus();
    this.cameraDirty = true;
  }

  setCategoryFilter(hidden: ToolCategory[]): void {
    this.hiddenCats = new Set(hidden);
    this.applyVisibility();
  }

  resetCamera(): void {
    this.az = SessionFieldDriver.HOME.az;
    this.el = SessionFieldDriver.HOME.el;
    this.dist = SessionFieldDriver.HOME.dist;
    this.cameraDirty = true;
  }

  /** Debug/e2e handle: what the driver believes it is drawing. */
  describe() {
    return {
      nodes: this.nodes.length,
      webgpu: this.webgpu,
      bloom: this.bloomPipe !== null,
      k: { x: this.axes.x.k, y: this.axes.y.k, z: this.axes.z.k },
      curved: { x: this.axes.x.curved, y: this.axes.y.curved, z: this.axes.z.curved },
      camera: { az: this.az, el: this.el, dist: this.dist },
    };
  }

  // ── build ────────────────────────────────────────────────────────────────

  private rebuild(): void {
    this.clearData();
    if (this.analyses.length === 0) {
      this.layoutLabels();
      this.cameraDirty = true;
      return;
    }

    // axis maxima across every visible session, so overlaid sessions stay
    // comparable (a longer/heavier one genuinely reads bigger)
    const xs: number[] = [];
    const ys: number[] = [];
    const zs: number[] = [];
    for (const a of this.analyses) {
      for (const t of a.turns) {
        xs.push(t.tSec);
        ys.push(t.cacheRead);
        zs.push(t.cacheWrite);
      }
    }
    const mk = (vals: number[], forced: number | null): AxisScale => {
      const max = vals.reduce((m, v) => (v > m ? v : m), 0);
      return asinhScale(max, forced ?? suggestK(vals, max));
    };
    this.axes = {
      x: mk(xs, this.scaleK.x),
      y: mk(ys, this.scaleK.y),
      z: mk(zs, this.scaleK.z),
    };

    let maxTools = 1;
    for (const a of this.analyses) {
      for (const t of a.turns) if (t.tools.length > maxTools) maxTools = t.tools.length;
    }

    const trailSegs: {
      a: THREE.Vector3;
      b: THREE.Vector3;
      rgb: [number, number, number];
      sessionId: string;
      index: number;
    }[] = [];
    for (let ai = 0; ai < this.analyses.length; ai++) {
      const a = this.analyses[ai]!;
      const hue = sessionHue(ai);
      let prev: THREE.Vector3 | null = null;
      for (const raw of a.turns) {
        // a stored analysis can carry a category this build has no colour for
        // (written by another build; the raw transcript is never persisted, so
        // it can't be re-parsed) — re-derive from the tools that WERE stored
        const t =
          raw.category in CATEGORY_RGB ? raw : { ...raw, category: dominantCategory(raw.tools) };
        const pos = new THREE.Vector3(
          this.axes.x.toUnit(t.tSec) * CUBE - HALF,
          this.axes.y.toUnit(t.cacheRead) * CUBE - HALF,
          this.axes.z.toUnit(t.cacheWrite) * CUBE - HALF,
        );
        this.nodes.push({ sessionId: a.id, sessionName: a.name, index: t.index, turn: t, pos });
        if (prev) trailSegs.push({ a: prev, b: pos, rgb: hue, sessionId: a.id, index: t.index });
        prev = pos;
      }
    }

    this.buildField(maxTools);
    this.buildTrail(trailSegs);
    this.applyVisibility();
    this.layoutLabels();
    this.cameraDirty = true;
  }

  private buildField(maxTools: number): void {
    const n = this.nodes.length;
    const pos = new Float32Array(n * 3);
    const color = new Float32Array(n * 3);
    const glow = new Float32Array(n);
    const size = new Float32Array(n);
    const sub = new Float32Array(n);
    this.visArray = new Float32Array(n).fill(1);

    const rank = outputRank(this.nodes);

    for (let i = 0; i < n; i++) {
      const node = this.nodes[i]!;
      const t = node.turn;
      pos[i * 3] = node.pos.x;
      pos[i * 3 + 1] = node.pos.y;
      pos[i * 3 + 2] = node.pos.z;

      // Brightness carries output tokens, RANKED within the loaded sessions —
      // see outputRank(). The gamma is what keeps the bulk of the field dim:
      // the median turn lands near 0.22, the top decile above 0.79, whatever
      // the underlying distribution looks like. That contrast IS the
      // figure-ground the old flat scatter never had.
      const g = Math.pow(rank[i]!, GLOW_GAMMA);
      const failed = (t.errors ?? 0) > 0;
      glow[i] = failed ? Math.max(g, 0.85) : g;
      // Size is a SECOND channel on a second attribute: tool calls this turn.
      // Deliberately a narrow range — it separates a 5-tool turn from a
      // 0-tool one without competing with brightness for attention.
      size[i] = 0.55 + 0.45 * Math.min(t.tools.length / maxTools, 1);
      sub[i] = t.isSidechain ? 1 : 0;

      const cat = CATEGORY_RGB[t.category] ?? NEUTRAL;
      const base = failed ? ERROR_RGB : cat;
      // saturation ramps with magnitude: quiet turns wash toward neutral so
      // they read as field, loud ones carry their category at full strength
      const sat = failed ? 1 : SAT_FLOOR + (1 - SAT_FLOOR) * glow[i]!;
      for (let c = 0; c < 3; c++) {
        const n0 = NEUTRAL[c] ?? 0;
        color[i * 3 + c] = n0 + ((base[c] ?? 0) - n0) * sat;
      }
    }

    const iPos = instancedBufferAttribute<"vec3">(new THREE.InstancedBufferAttribute(pos, 3), "vec3");
    const iColor = instancedBufferAttribute<"vec3">(new THREE.InstancedBufferAttribute(color, 3), "vec3");
    const iGlow = instancedBufferAttribute<"float">(new THREE.InstancedBufferAttribute(glow, 1), "float");
    const iSize = instancedBufferAttribute<"float">(new THREE.InstancedBufferAttribute(size, 1), "float");
    const iSub = instancedBufferAttribute<"float">(new THREE.InstancedBufferAttribute(sub, 1), "float");
    this.visAttr = new THREE.InstancedBufferAttribute(this.visArray, 1);
    this.visAttr.setUsage(THREE.DynamicDrawUsage);
    const iVis = instancedDynamicBufferAttribute<"float">(this.visAttr, "float");

    const material = new THREE.SpriteNodeMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
    });

    const hovered = instanceIndex.toFloat().equal(this.uHover);
    const picked = instanceIndex.toFloat().equal(this.uSelected);
    const emphasis = select(picked, float(2.6), select(hovered, float(1.9), float(1)));

    material.positionNode = iPos;
    // the bright turns also spread a little wider — the two channels reinforce
    // rather than fight, and a hot core with a wide halo is what reads as glow
    material.scaleNode = this.uSize.mul(iSize).mul(float(0.7).add(iGlow.mul(1.1))).mul(emphasis);

    // emissive: the top of the range exceeds 1.0 on purpose — that headroom is
    // what the bloom threshold (0.55) keys on, so only real spikes glow
    material.colorNode = iColor.mul(mix(float(EMISSIVE_MIN), float(EMISSIVE_MAX), iGlow));

    // soft radial falloff — no hard edge, so overlapping points accumulate
    // rather than occlude
    const d = uv().sub(0.5).length();
    const disc = d.smoothstep(0.06, 0.5).oneMinus();
    const alpha = mix(float(BASE_ALPHA), float(PEAK_ALPHA), iGlow);
    const subDim = mix(float(1), float(0.5), iSub);
    material.opacityNode = disc
      .mul(select(picked.or(hovered), float(1), alpha))
      .mul(subDim)
      .mul(iVis);

    this.fieldMat = material;
    const sprite = new THREE.Sprite(material);
    sprite.count = n;
    sprite.frustumCulled = false;
    sprite.renderOrder = 3;
    this.field = sprite;
    this.scene.add(sprite);

    // id companion for GPU picking — shares the same attribute nodes so a pick
    // can never drift from what is on screen
    const idMat = new THREE.SpriteNodeMaterial({ transparent: false });
    idMat.positionNode = iPos;
    idMat.scaleNode = this.uSize.mul(iSize).mul(float(0.7).add(iGlow.mul(1.1))).mul(2.4); // finger-friendly
    const id = instanceIndex.add(1).toFloat();
    idMat.colorNode = vec3(id.mod(256), id.div(256).floor().mod(256), id.div(65536).floor()).div(255);
    idMat.opacityNode = select(d.lessThan(0.45), float(1), float(0)).mul(iVis.step(0.5).oneMinus().oneMinus());
    idMat.alphaTest = 0.5;
    const idSprite = new THREE.Sprite(idMat);
    idSprite.count = n;
    idSprite.frustumCulled = false;
    this.idSprite = idSprite;
    this.picker = new IdPicker(this.renderer, idSprite);
    this.picker.setSize(this.cssW, this.cssH);
  }

  /** The trajectory, as thin additive ribbons between consecutive turns.
   *
   *  A 1,600-turn session ordered by time on X but scattered on Y/Z produces
   *  1,599 segments that criss-cross the whole cube — drawn at any readable
   *  alpha they are the single largest contributor to visual noise, and they
   *  bury the field they are supposed to annotate. So the whole path sits at a
   *  near-subliminal ambient alpha (you can see there IS a path), and the run
   *  around whatever turn you point at lights up. Nothing is hidden or
   *  resampled: every real segment is drawn, the emphasis just follows the
   *  cursor. */
  private buildTrail(
    segs: {
      a: THREE.Vector3;
      b: THREE.Vector3;
      rgb: [number, number, number];
      sessionId: string;
      index: number;
    }[],
  ): void {
    if (segs.length === 0) return;
    const n = segs.length;
    this.trailKeys = segs.map((s) => ({ sessionId: s.sessionId, index: s.index }));
    const starts = new Float32Array(n * 3);
    const ends = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const s = segs[i]!;
      starts[i * 3] = s.a.x;
      starts[i * 3 + 1] = s.a.y;
      starts[i * 3 + 2] = s.a.z;
      ends[i * 3] = s.b.x;
      ends[i * 3 + 1] = s.b.y;
      ends[i * 3 + 2] = s.b.z;
      colors[i * 3] = s.rgb[0];
      colors[i * 3 + 1] = s.rgb[1];
      colors[i * 3 + 2] = s.rgb[2];
    }
    const aStart = instancedBufferAttribute<"vec3">(new THREE.InstancedBufferAttribute(starts, 3), "vec3");
    const aEnd = instancedBufferAttribute<"vec3">(new THREE.InstancedBufferAttribute(ends, 3), "vec3");
    const aColor = instancedBufferAttribute<"vec3">(new THREE.InstancedBufferAttribute(colors, 3), "vec3");
    this.focusArray = new Float32Array(n);
    this.focusAttr = new THREE.InstancedBufferAttribute(this.focusArray, 1);
    this.focusAttr.setUsage(THREE.DynamicDrawUsage);
    const aFocus = instancedDynamicBufferAttribute<"float">(this.focusAttr, "float");

    const material = new THREE.MeshBasicNodeMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });

    // ribbon built in the vertex stage from the two endpoints, billboarded
    // toward the camera so it stays face-on at any orbit angle (same trick as
    // the atlas beams)
    const t = uv().x;
    const across = positionGeometry.y;
    const dir = aEnd.sub(aStart);
    const mid = aStart.add(aEnd).mul(0.5);
    const perpRaw = cameraPosition.sub(mid).cross(dir);
    const perp = perpRaw.div(perpRaw.length().max(1e-6));
    material.positionNode = aStart.add(dir.mul(t)).add(perp.mul(across.mul(TRAIL_WIDTH)));
    material.colorNode = aColor;
    const edgeFade = across.abs().mul(2).smoothstep(0.2, 1).oneMinus();
    material.opacityNode = edgeFade.mul(mix(float(TRAIL_ALPHA), float(TRAIL_FOCUS_ALPHA), aFocus));

    this.trailMat = material;
    const mesh = new THREE.InstancedMesh(new THREE.PlaneGeometry(1, 1, 1, 1), material, n);
    mesh.count = n;
    mesh.frustumCulled = false;
    mesh.renderOrder = 2;
    this.trail = mesh;
    this.scene.add(mesh);
  }

  /** The containing cube and its floor grid — dim, so it locates the data
   *  without competing with it. */
  private buildFrame(): void {
    const pts: number[] = [];
    const c = [-HALF, HALF];
    // 12 edges
    for (const a of c) {
      for (const b of c) {
        pts.push(-HALF, a, b, HALF, a, b);
        pts.push(a, -HALF, b, a, HALF, b);
        pts.push(a, b, -HALF, a, b, HALF);
      }
    }
    // floor grid
    const DIV = 8;
    for (let i = 1; i < DIV; i++) {
      const u = -HALF + (i / DIV) * CUBE;
      pts.push(u, -HALF, -HALF, u, -HALF, HALF);
      pts.push(-HALF, -HALF, u, HALF, -HALF, u);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    const mat = new THREE.LineBasicNodeMaterial({
      color: FRAME_RGB,
      transparent: true,
      opacity: 0.5,
      depthWrite: false,
    });
    this.frame3 = new THREE.LineSegments(geo, mat);
    this.frame3.frustumCulled = false;
    this.frame3.renderOrder = 1;
    this.scene.add(this.frame3);
  }

  /** Drop-lines for the ONE turn under the cursor / pinned. Three rails to the
   *  floor and two walls, so a single point's coordinates are readable without
   *  drawing 1600 of them. */
  private buildProbe(): void {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(6 * 3), 3));
    const mat = new THREE.LineBasicNodeMaterial({
      color: 0xf5c33b,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      depthTest: false,
    });
    this.probe = new THREE.LineSegments(geo, mat);
    this.probe.frustumCulled = false;
    this.probe.visible = false;
    this.probe.renderOrder = 4;
    this.scene.add(this.probe);
  }

  private updateProbe(): void {
    const i = this.selected ?? this.hovered;
    if (!this.probe) return;
    if (i === null || i < 0 || i >= this.nodes.length) {
      this.probe.visible = false;
      return;
    }
    const p = this.nodes[i]!.pos;
    const arr = (this.probe.geometry.getAttribute("position") as THREE.BufferAttribute)
      .array as Float32Array;
    const rails: number[] = [
      p.x, p.y, p.z, p.x, -HALF, p.z, // to the floor
      p.x, p.y, p.z, -HALF, p.y, p.z, // to the time wall
      p.x, p.y, p.z, p.x, p.y, -HALF, // to the new-context wall
    ];
    arr.set(rails);
    (this.probe.geometry.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
    this.probe.visible = true;
  }

  /** Light the run of segments around the active turn and let the rest fall back
   *  to ambient. A linear ramp over TRAIL_FOCUS_SPAN turns, so the emphasis has
   *  a direction you can follow rather than a hard edge. */
  private updateTrailFocus(): void {
    if (!this.focusAttr) return;
    const i = this.selected ?? this.hovered;
    const active = i === null ? null : this.nodes[i];
    for (let s = 0; s < this.trailKeys.length; s++) {
      const key = this.trailKeys[s]!;
      if (!active || key.sessionId !== active.sessionId) {
        this.focusArray[s] = 0;
        continue;
      }
      const d = Math.abs(key.index - active.index);
      this.focusArray[s] = d > TRAIL_FOCUS_SPAN ? 0 : 1 - d / TRAIL_FOCUS_SPAN;
    }
    this.focusAttr.needsUpdate = true;
    this.cameraDirty = true;
  }

  private applyVisibility(): void {
    if (!this.visAttr) return;
    for (let i = 0; i < this.nodes.length; i++) {
      this.visArray[i] = this.hiddenCats.has(this.nodes[i]!.turn.category) ? DIM_ALPHA : 1;
    }
    this.visAttr.needsUpdate = true;
    this.cameraDirty = true;
  }

  // ── frame loop ───────────────────────────────────────────────────────────

  private loop = (): void => {
    if (this.disposed) return;
    this.raf = requestAnimationFrame(this.loop);
    if (this.cssW < 2 || this.cssH < 2) return;

    if (this.cameraDirty) {
      const ce = Math.cos(this.el);
      this.camera.position.set(
        Math.sin(this.az) * ce * this.dist,
        Math.sin(this.el) * this.dist,
        Math.cos(this.az) * ce * this.dist,
      );
      this.camera.lookAt(0, 0, 0);
      this.camera.updateProjectionMatrix();
      this.camera.updateMatrixWorld();
      this.layoutLabels();
      this.positionTooltip();
      this.cameraDirty = false;
    }

    this.maybePick();

    if (this.bloomPipe) this.bloomPipe.post.render();
    else this.renderer.render(this.scene, this.camera);
  };

  resize(width: number, height: number, dpr: number): void {
    if (width < 2 || height < 2) return;
    this.cssW = width;
    this.cssH = height;
    this.renderer.setPixelRatio(Math.min(dpr, 2));
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.picker?.setSize(width, height);
    this.cameraDirty = true;
  }

  // ── interaction ──────────────────────────────────────────────────────────

  private attachPointer(): void {
    const sig = this.abort.signal;
    const el = this.canvas;
    el.addEventListener(
      "pointerdown",
      (e) => {
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
        const r = el.getBoundingClientRect();
        this.mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
        if (this.dragging && this.last) {
          const dx = e.clientX - this.last.x;
          const dy = e.clientY - this.last.y;
          if (Math.abs(dx) + Math.abs(dy) > 2) this.dragMoved = true;
          this.az -= dx * 0.006;
          this.el = clamp(this.el + dy * 0.006, SessionFieldDriver.EL_MIN, SessionFieldDriver.EL_MAX);
          this.last = { x: e.clientX, y: e.clientY };
          this.cameraDirty = true;
        }
      },
      { signal: sig },
    );
    const end = (e: PointerEvent) => {
      if (this.dragging && !this.dragMoved) this.clickSelect();
      this.dragging = false;
      this.last = null;
      if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
    };
    el.addEventListener("pointerup", end, { signal: sig });
    el.addEventListener("pointercancel", end, { signal: sig });
    el.addEventListener(
      "pointerleave",
      () => {
        this.mouse = null;
        this.setHovered(null);
      },
      { signal: sig },
    );
    el.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.dist = clamp(
          this.dist * Math.exp(e.deltaY * 0.0012),
          SessionFieldDriver.DIST_MIN,
          SessionFieldDriver.DIST_MAX,
        );
        this.cameraDirty = true;
      },
      { signal: sig, passive: false },
    );
  }

  private clickSelect(): void {
    const i = this.hovered;
    if (i === null) {
      this.selected = null;
      this.uSelected.value = -1;
      this.updateProbe();
      this.updateTrailFocus();
      this.onSelect?.(null);
      return;
    }
    const n = this.nodes[i]!;
    this.selected = i;
    this.uSelected.value = i;
    this.updateProbe();
    this.updateTrailFocus();
    this.onSelect?.({ sessionId: n.sessionId, index: n.index });
  }

  private maybePick(): void {
    if (!this.picker || this.picker.broken || this.pickBusy || !this.mouse || this.dragging) return;
    const now = performance.now();
    if (now - this.lastPickAt < 33) return; // ~30 Hz, as in the atlas
    this.lastPickAt = now;
    this.pickBusy = true;
    const { x, y } = this.mouse;
    const generation = this.nodes.length;
    this.picker
      .pick(this.camera, x, y)
      .then((idx) => {
        if (this.disposed || generation !== this.nodes.length) return;
        this.setHovered(idx >= 0 && idx < this.nodes.length ? idx : null);
      })
      .catch(() => {})
      .finally(() => {
        this.pickBusy = false;
      });
  }

  private setHovered(i: number | null): void {
    if (this.hovered === i) return;
    this.hovered = i;
    this.uHover.value = i ?? -1;
    this.updateProbe();
    this.updateTrailFocus();
    this.showTooltip(i);
    this.canvas.style.cursor = i === null ? "grab" : "pointer";
    const n = i === null ? null : this.nodes[i]!;
    this.onHover?.(n ? { sessionId: n.sessionId, index: n.index } : null);
    this.cameraDirty = true;
  }

  // ── html overlay: ticks + tooltip ────────────────────────────────────────

  private ensureLabels(count: number): void {
    while (this.labels.length < count) {
      const el = document.createElement("div");
      el.className = "session-axis-label";
      this.overlay.appendChild(el);
      this.labels.push(el);
    }
  }

  /** Ticks are labelled by INVERTING the scale, so each one reads a true value
   *  even though the spacing is non-uniform. */
  private layoutLabels(): void {
    if (!this.overlay) return;
    const specs: { pos: THREE.Vector3; text: string }[] = [];
    if (this.nodes.length > 0) {
      const TICKS = 4;
      for (const { u, value } of this.axes.x.ticks(TICKS)) {
        specs.push({ pos: new THREE.Vector3(u * CUBE - HALF, -HALF, HALF), text: fmtSecs(value) });
      }
      for (const { u, value } of this.axes.y.ticks(TICKS)) {
        specs.push({ pos: new THREE.Vector3(-HALF, u * CUBE - HALF, HALF), text: fmtTokens(value) });
      }
      for (const { u, value } of this.axes.z.ticks(TICKS)) {
        specs.push({ pos: new THREE.Vector3(-HALF, -HALF, u * CUBE - HALF), text: fmtTokens(value) });
      }
      specs.push({ pos: new THREE.Vector3(0, -HALF - 0.08, HALF), text: "TIME →" });
      specs.push({ pos: new THREE.Vector3(-HALF - 0.06, 0, HALF), text: "CONTEXT ↑" });
      specs.push({ pos: new THREE.Vector3(-HALF - 0.06, -HALF - 0.05, 0), text: "NEW CTX ↗" });
    }
    this.ensureLabels(specs.length);
    for (let i = 0; i < this.labels.length; i++) {
      const el = this.labels[i]!;
      const spec = specs[i];
      if (!spec) {
        el.style.display = "none";
        continue;
      }
      const p = this.project(spec.pos);
      if (!p) {
        el.style.display = "none";
        continue;
      }
      el.textContent = spec.text;
      el.style.display = "";
      el.style.translate = `${Math.round(p[0])}px ${Math.round(p[1])}px`;
    }
  }

  private project(v: THREE.Vector3): [number, number] | null {
    const p = v.clone().project(this.camera);
    if (p.z > 1) return null; // behind the camera
    return [(p.x * 0.5 + 0.5) * this.cssW, (-p.y * 0.5 + 0.5) * this.cssH];
  }

  private showTooltip(i: number | null): void {
    const el = this.tooltipEl;
    if (!el) return;
    if (i === null) {
      el.style.display = "none";
      return;
    }
    const n = this.nodes[i]!;
    const t = n.turn;
    const err = t.errors;
    el.innerHTML = "";
    const head = document.createElement("div");
    head.className = "session-tt-head";
    const name = document.createElement("span");
    name.className = "session-tt-name";
    name.textContent = n.sessionName;
    const turn = document.createElement("span");
    turn.className = "session-tt-turn";
    turn.textContent = `#${t.index + 1}`;
    head.append(name, turn);
    el.appendChild(head);
    const cat = document.createElement("span");
    cat.className = "session-tt-cat";
    cat.dataset["cat"] = t.category;
    cat.textContent = t.category;
    el.appendChild(cat);
    const grid = document.createElement("dl");
    grid.className = "session-tt-grid";
    const rows: [string, string][] = [
      ["Time", fmtSecs(t.tSec)],
      ["Output", t.outputTokens.toLocaleString()],
      ["Context", t.cacheRead.toLocaleString()],
      ["New context", t.cacheWrite.toLocaleString()],
      ["Tools", String(t.tools.length)],
    ];
    if (err !== undefined && err > 0) rows.push(["Failures", String(err)]);
    for (const [k, v] of rows) {
      const dt = document.createElement("dt");
      dt.textContent = k;
      const dd = document.createElement("dd");
      dd.textContent = v;
      grid.append(dt, dd);
    }
    el.appendChild(grid);
    el.style.display = "";
    this.positionTooltip();
  }

  private positionTooltip(): void {
    const el = this.tooltipEl;
    if (!el || el.style.display === "none" || this.hovered === null) return;
    const p = this.project(this.nodes[this.hovered]!.pos);
    if (!p) {
      el.style.display = "none";
      return;
    }
    // `translate`, never `transform` — the entrance animation owns transform
    el.style.translate = `${Math.round(p[0] + 14)}px ${Math.round(p[1] + 10)}px`;
  }

  // ── teardown ─────────────────────────────────────────────────────────────

  private clearData(): void {
    if (this.field) {
      this.scene.remove(this.field);
      this.field = null;
    }
    this.fieldMat?.dispose();
    this.fieldMat = null;
    if (this.idSprite) {
      (this.idSprite.material as THREE.Material).dispose();
      this.idSprite = null;
    }
    this.picker?.dispose();
    this.picker = null;
    if (this.trail) {
      this.scene.remove(this.trail);
      this.trail.geometry.dispose();
      this.trail = null;
    }
    this.trailMat?.dispose();
    this.trailMat = null;
    this.nodes = [];
    this.hovered = null;
    this.selected = null;
    this.uHover.value = -1;
    this.uSelected.value = -1;
    this.visAttr = null;
    this.focusAttr = null;
    this.trailKeys = [];
    if (this.probe) this.probe.visible = false;
    this.tooltipEl && (this.tooltipEl.style.display = "none");
  }

  dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    this.abort.abort();
    this.clearData();
    // NB: never dispose a Sprite's geometry — THREE.Sprite shares ONE
    // module-level quad across every sprite in the app (see PointsLayer).
    if (this.frame3) {
      this.scene.remove(this.frame3);
      this.frame3.geometry.dispose();
      (this.frame3.material as THREE.Material).dispose();
      this.frame3 = null;
    }
    if (this.probe) {
      this.scene.remove(this.probe);
      this.probe.geometry.dispose();
      (this.probe.material as THREE.Material).dispose();
      this.probe = null;
    }
    this.bloomPipe?.dispose();
    this.bloomPipe = null;
    for (const el of this.labels) el.remove();
    this.labels = [];
    this.tooltipEl?.remove();
    this.tooltipEl = null;
    this.renderer?.dispose();
  }
}

// ── helpers ─────────────────────────────────────────────────────────────────

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Hue 0 (pure red) is reserved for error states, so per-session trail hues
 *  walk the arc that excludes the red band. */
const HUE_RESERVED = 16;
const HUE_SPAN = 360 - 2 * HUE_RESERVED;
const HUE_START = 206;

export function sessionHue(i: number): [number, number, number] {
  const h = (HUE_RESERVED + ((HUE_START + i * 137.508) % HUE_SPAN)) / 360;
  const s = 0.6;
  const l = 0.62;
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const f = (t: number) => {
    let tt = t;
    if (tt < 0) tt += 1;
    if (tt > 1) tt -= 1;
    if (tt < 1 / 6) return p + (q - p) * 6 * tt;
    if (tt < 1 / 2) return q;
    if (tt < 2 / 3) return p + (q - p) * (2 / 3 - tt) * 6;
    return p;
  };
  return [f(h + 1 / 3), f(h), f(h - 1 / 3)];
}

/** Each node's output-token rank in [0,1] across every loaded session.
 *
 *  Brightness has to separate turns on *any* transcript, and a raw
 *  value/max ratio can't: tokens-per-turn is heavy-tailed on some sessions and
 *  near-uniform on others, and tool-count is an integer that often spans only
 *  0–3, which collapses the ramp to four levels. The rank always uses the full
 *  range, so the field looks the same shape regardless.
 *
 *  It stays honest because it is MONOTONE in the raw value — brighter always
 *  means more output, never less — and equal values get an equal (mid-)rank so
 *  two identical turns can never be drawn differently. What it does not claim
 *  is proportionality: twice as bright is not twice the tokens, which is why
 *  the tooltip prints the raw count and the legend says "ranked".
 */
export function outputRank(nodes: { turn: { outputTokens: number } }[]): Float32Array {
  const n = nodes.length;
  const out = new Float32Array(n);
  if (n < 2) return out.fill(n === 1 ? 1 : 0);
  const order = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => nodes[a]!.turn.outputTokens - nodes[b]!.turn.outputTokens,
  );
  const denom = n - 1;
  let i = 0;
  while (i < n) {
    // midrank over the run of ties, so equal token counts glow equally
    let j = i + 1;
    const v = nodes[order[i]!]!.turn.outputTokens;
    while (j < n && nodes[order[j]!]!.turn.outputTokens === v) j++;
    const mid = (i + j - 1) / 2 / denom;
    for (let k = i; k < j; k++) out[order[k]!] = mid;
    i = j;
  }
  return out;
}

function fmtTokens(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k`;
  return String(Math.round(n));
}

function fmtSecs(s: number): string {
  if (s >= 3600) return `${(s / 3600).toFixed(1)}h`;
  if (s >= 60) return `${Math.round(s / 60)}m`;
  return `${Math.round(s)}s`;
}
