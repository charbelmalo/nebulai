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
  buildAgentGraph,
  CATEGORY_ORDER,
  dominantCategory,
  type AgentStep,
  type SessionAnalysis,
  type SessionTurn,
  type ToolCategory,
} from "../../chrome/sessionlog";
import { createBloomPipeline, type BloomPipeline } from "../post/bloom";
import { IdPicker } from "../picking";
import { asinhScale, suggestK, type AxisScale } from "./scales";
import {
  DEFAULT_SESSIONS_APPEARANCE,
  hexToRgb,
  orderedCategoryRgb,
  type SessionsAppearance,
  type SessionsAxisMode,
} from "./appearance";

/** Cube side in world units; the cube is centred on the origin. */
const CUBE = 1;
const HALF = CUBE / 2;

// ── fixed tuning (not user-facing) ───────────────────────────────────────────
const POINT_SIZE = 0.0105; // world-unit base; the pointSize knob multiplies it
const PEAK_ALPHA = 1; // the top of the mote-alpha ramp
/** Emissive multiplier at the BOTTOM of the glow ramp. The ceiling is the
 *  user-facing `glowStrength`; this floor stays fixed — it only sets how dark
 *  the quietest lit mote is, and the bloom read keys on the ceiling. */
const EMISSIVE_MIN = 0.34;

// ── timeline playback ────────────────────────────────────────────────────────
/** Wall-clock seconds for one full start→finish sweep at 1×. The sweep advances
 *  linearly in axis-space, so on an eased time axis idle gaps compress and the
 *  replay never stalls; on a linear axis it plays in true proportional time. */
const PLAY_BASE_SECONDS = 22;
/** Axis-space width of the ignite fade — a turn eases in over this band as the
 *  playhead reaches its timestamp rather than popping on. */
const REVEAL_FADE = 0.015;
/** Axis-space width of the wavefront glow trailing just behind the playhead. */
const PULSE_WIDTH = 0.06;
/** Extra emissive (×) a turn carries at the instant the wavefront passes it. */
const PULSE_AMT = 1.6;
/** Extra trail alpha at the growing tip of the path during playback. */
const TRAIL_PULSE = 0.6;

/** Which categories this build can colour directly; a stored analysis carrying
 *  anything else is re-derived from its tools (see rebuild()). */
const VALID_CATEGORIES = new Set<ToolCategory>(CATEGORY_ORDER);
/** Category → its index in CATEGORY_ORDER, the slot its colour uniform lives in. */
const CATEGORY_INDEX: Record<ToolCategory, number> = Object.fromEntries(
  CATEGORY_ORDER.map((c, i) => [c, i]),
) as Record<ToolCategory, number>;

export interface TurnRef {
  sessionId: string;
  index: number;
}

/** A snapshot of the timeline transport, emitted to the transport UI. The
 *  playhead is a pure position in [0,1] along the honest time axis; `tSec` is
 *  its exact inverse (axes.x.toValue), so the readout never lies about where in
 *  wall-clock time the sweep is. */
export interface PlaybackState {
  playing: boolean;
  /** 0..1 sweep position along the time axis */
  playhead: number;
  /** wall-clock seconds at the playhead — the honest inverse of `playhead` */
  tSec: number;
  /** wall-clock seconds at the end of the run (the time-axis maximum) */
  totalSec: number;
  /** turns whose timestamp the wavefront has reached */
  revealed: number;
  /** turns in total across every loaded session */
  total: number;
  /** replay rate multiplier (0.5 / 1 / 2 / 4) */
  speed: number;
  hasData: boolean;
}

interface FieldNode {
  sessionId: string;
  sessionName: string;
  index: number;
  turn: SessionTurn;
  pos: THREE.Vector3;
  /** This turn's place in its OWN agent's path — step n of m, and the gap
   *  since that agent's previous step. Undefined only if the turn somehow
   *  escaped the graph, which the builder makes impossible. */
  step?: AgentStep;
  /** How many distinct agents ran in this turn's session. 1 for every real
   *  transcript on record; the tooltip suppresses its per-agent row at 1, where
   *  the agent step and the session turn number are the same number twice. */
  agentCount: number;
}

interface Axes {
  x: AxisScale;
  y: AxisScale;
  z: AxisScale;
}

export class SessionFieldDriver {
  onSelect: ((sel: TurnRef | null) => void) | null = null;
  onHover: ((sel: TurnRef | null) => void) | null = null;
  onPlayback: ((state: PlaybackState) => void) | null = null;

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
  /** The live look. Every value here is either a uniform (updated in place, no
   *  rebuild) or a cheap CPU pass; only the three axis modes force a geometry
   *  rebuild, because they change where the motes sit. */
  private cfg: SessionsAppearance = { ...DEFAULT_SESSIONS_APPEARANCE };
  /** honours the global Settings → bloom toggle (webgpu rung only). */
  private bloomOn = true;

  // ── gpu objects ────────────────────────────────────────────────────────
  private field: THREE.Sprite | null = null;
  private fieldMat: THREE.SpriteNodeMaterial | null = null;
  private idSprite: THREE.Sprite | null = null;
  private trail: THREE.InstancedMesh | null = null;
  private trailMat: THREE.MeshBasicNodeMaterial | null = null;
  private frame3: THREE.LineSegments | null = null;
  private frameMat: THREE.LineBasicNodeMaterial | null = null;
  private probe: THREE.LineSegments | null = null;
  private sweep: THREE.LineSegments | null = null;
  private sweepMat: THREE.LineBasicNodeMaterial | null = null;
  private picker: IdPicker | null = null;

  // ── live uniforms (see cfg) ──────────────────────────────────────────────
  private uHover = uniform(-1);
  private uSelected = uniform(-1);
  private uSize = uniform(POINT_SIZE * DEFAULT_SESSIONS_APPEARANCE.pointSize);
  private uGamma = uniform(DEFAULT_SESSIONS_APPEARANCE.glowContrast);
  private uMoteFloor = uniform(DEFAULT_SESSIONS_APPEARANCE.moteFloor);
  private uGlowMax = uniform(DEFAULT_SESSIONS_APPEARANCE.glowStrength);
  private uSatFloor = uniform(DEFAULT_SESSIONS_APPEARANCE.saturation);
  private uHoverEmph = uniform(DEFAULT_SESSIONS_APPEARANCE.hoverEmphasis);
  private uSelectEmph = uniform(DEFAULT_SESSIONS_APPEARANCE.selectEmphasis);
  private uSubOpacity = uniform(DEFAULT_SESSIONS_APPEARANCE.subAgentOpacity);
  private uMarkFail = uniform(DEFAULT_SESSIONS_APPEARANCE.markFailures ? 1 : 0);
  private uFailFloor = uniform(DEFAULT_SESSIONS_APPEARANCE.failureGlow);
  private uNeutral = uniform(new THREE.Vector3(...hexToRgb(DEFAULT_SESSIONS_APPEARANCE.neutralColor)));
  private uError = uniform(new THREE.Vector3(...hexToRgb(DEFAULT_SESSIONS_APPEARANCE.errorColor)));
  private uCats = orderedCategoryRgb(DEFAULT_SESSIONS_APPEARANCE).map(
    (rgb) => uniform(new THREE.Vector3(...rgb)),
  );
  private uTrailAlpha = uniform(DEFAULT_SESSIONS_APPEARANCE.trailOpacity);
  private uTrailFocusAlpha = uniform(DEFAULT_SESSIONS_APPEARANCE.trailFocusOpacity);
  private uTrailWidth = uniform(DEFAULT_SESSIONS_APPEARANCE.trailWidth);
  // ── timeline playback (uniform-driven; no rebuild) ───────────────────────
  /** wavefront position along the time axis, 0..1. Rest = 1 (all revealed). */
  private uPlayhead = uniform(1);
  private uRevealFade = uniform(REVEAL_FADE);
  private uPulse = uniform(0); // wavefront glow amount; 0 unless playing
  private uPulseWidth = uniform(PULSE_WIDTH);
  private uTrailPulse = uniform(0); // growing-tip trail boost; 0 unless playing
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

  // ── playback runtime state ───────────────────────────────────────────────
  private playing = false;
  /** mirror of uPlayhead; the loop advances this and pushes it to the uniform */
  private playhead = 1;
  private playSpeed = 1;
  private playReducedMotion = false;
  /** every node's seq (unit time position), sorted ascending, for a cheap
   *  revealed-count binary search */
  private seqSorted = new Float32Array(0);
  private lastClock = 0;
  private lastEmit = 0;
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
    this.buildSweep();
    this.applyScaffold();

    // bloom rides only the real WebGPU rung, exactly as in AtlasDriver. It is
    // built once here and toggled by the render-path choice in loop(), so the
    // global Settings → bloom switch flips live with no teardown.
    this.bloomOn = appStore.getState().settings.bloom;
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

  /** Apply a full appearance config. Most knobs are uniforms or a cheap CPU
   *  pass and update in place; only an axis-mode change reshapes the field, so
   *  only that path rebuilds the geometry. */
  setAppearance(cfg: SessionsAppearance): void {
    const prev = this.cfg;
    this.cfg = cfg;
    this.applyUniforms();
    this.applyScaffold();
    this.applyVisibility(); // dimmedOpacity may have changed
    this.updateTrailFocus(); // trailFocusSpan may have changed
    this.layoutLabels(); // showLabels may have changed
    const axisChanged =
      prev.axisTime !== cfg.axisTime ||
      prev.axisContext !== cfg.axisContext ||
      prev.axisNewContext !== cfg.axisNewContext;
    if (axisChanged) this.rebuild();
    else this.cameraDirty = true;
  }

  /** Honour the global Settings → bloom switch (webgpu rung only). */
  setBloom(on: boolean): void {
    this.bloomOn = on;
    this.cameraDirty = true;
  }

  /** Push every live-tunable value from cfg into its uniform. */
  private applyUniforms(): void {
    const c = this.cfg;
    this.uSize.value = POINT_SIZE * c.pointSize;
    this.uGamma.value = c.glowContrast;
    this.uMoteFloor.value = c.moteFloor;
    this.uGlowMax.value = c.glowStrength;
    this.uSatFloor.value = c.saturation;
    this.uHoverEmph.value = c.hoverEmphasis;
    this.uSelectEmph.value = c.selectEmphasis;
    this.uSubOpacity.value = c.subAgentOpacity;
    this.uMarkFail.value = c.markFailures ? 1 : 0;
    this.uFailFloor.value = c.failureGlow;
    this.uNeutral.value.set(...hexToRgb(c.neutralColor));
    this.uError.value.set(...hexToRgb(c.errorColor));
    const cats = orderedCategoryRgb(c);
    for (let i = 0; i < this.uCats.length; i++) this.uCats[i]!.value.set(...cats[i]!);
    this.uTrailAlpha.value = c.trailOpacity;
    this.uTrailFocusAlpha.value = c.trailFocusOpacity;
    this.uTrailWidth.value = c.trailWidth;
  }

  /** Frame / grid / probe / trail visibility straight from cfg. */
  private applyScaffold(): void {
    const c = this.cfg;
    if (this.frame3) this.frame3.visible = c.showFrame;
    if (this.frameMat) {
      this.frameMat.opacity = c.frameOpacity;
      this.frameMat.color.set(c.frameColor);
    }
    if (this.trail) this.trail.visible = c.showTrails;
    if (!c.showProbe && this.probe) this.probe.visible = false;
    else this.updateProbe();
  }

  /** Resolve an axis mode to an asinh bend against the data. Honest in all
   *  three modes — see appearance.ts. */
  private axisK(mode: SessionsAxisMode, vals: number[], max: number): number {
    if (mode === "linear") return 0;
    return suggestK(vals, max, mode === "eased");
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

  // ── timeline transport ─────────────────────────────────────────────────────

  /** Start (or resume) the sweep. Replays from the top if it was parked at the
   *  end. No-op with no data loaded. */
  play(): void {
    if (this.nodes.length === 0) return;
    if (this.playhead >= 0.9999) this.playhead = 0;
    this.playing = true;
    this.lastClock = performance.now();
    this.updatePlayback();
  }

  pause(): void {
    this.playing = false;
    this.updatePlayback();
  }

  togglePlay(): void {
    if (this.playing) this.pause();
    else this.play();
  }

  /** Jump to the top and play from there. */
  restart(): void {
    this.playhead = 0;
    this.playing = this.nodes.length > 0;
    this.lastClock = performance.now();
    this.updatePlayback();
  }

  /** Move the playhead to a unit position without changing play/pause state —
   *  the transport scrubber pauses first, then seeks. */
  seek(u: number): void {
    this.playhead = clamp(u, 0, 1);
    this.updatePlayback();
  }

  setSpeed(mult: number): void {
    this.playSpeed = mult > 0 ? mult : 1;
    this.lastClock = performance.now(); // so the new rate applies from now, not retroactively
    this.emitPlayback(true);
  }

  /** Reduced motion kills the wavefront flare but keeps the progressive reveal;
   *  the sweep still plays, it just doesn't pulse. */
  setReducedMotion(on: boolean): void {
    this.playReducedMotion = on;
    this.updatePlayback(false);
  }

  /** Park the sweep at fully-revealed and stop — the resting state, applied on
   *  every dataset/axis rebuild so loading data shows everything. */
  private resetPlayback(): void {
    this.playing = false;
    this.playhead = 1;
    this.uPlayhead.value = 1;
    this.uPulse.value = 0;
    this.uTrailPulse.value = 0;
    this.updateSweep();
    this.emitPlayback(true);
  }

  /** Push playhead + derived pulse amounts to the uniforms and refresh the
   *  sweep plane. The pulse only lives while actively playing (and not under
   *  reduced motion), so a paused mid-run scrub is calm. */
  private updatePlayback(emit = true): void {
    this.uPlayhead.value = this.playhead;
    const motion = this.playing && !this.playReducedMotion;
    this.uPulse.value = motion ? PULSE_AMT : 0;
    this.uTrailPulse.value = motion ? TRAIL_PULSE : 0;
    this.updateSweep();
    if (emit) this.emitPlayback(true);
  }

  /** # of turns whose seq ≤ playhead — a binary search over the sorted column. */
  private revealedCount(): number {
    const a = this.seqSorted;
    let lo = 0;
    let hi = a.length;
    while (lo < hi) {
      const m = (lo + hi) >> 1;
      if (a[m]! <= this.playhead) lo = m + 1;
      else hi = m;
    }
    return lo;
  }

  private playbackState(): PlaybackState {
    const total = this.nodes.length;
    return {
      playing: this.playing,
      playhead: this.playhead,
      // honest inverse: the time axis maps the unit playhead back to real seconds
      tSec: total > 0 ? this.axes.x.toValue(clamp(this.playhead, 0, 1)) : 0,
      totalSec: total > 0 ? this.axes.x.max : 0,
      revealed: this.revealedCount(),
      total,
      speed: this.playSpeed,
      hasData: total > 0,
    };
  }

  /** Emit at most ~18 Hz while playing; discrete transport events force it. */
  private emitPlayback(force = false): void {
    if (!this.onPlayback) return;
    const now = performance.now();
    if (!force && now - this.lastEmit < 55) return;
    this.lastEmit = now;
    this.onPlayback(this.playbackState());
  }

  /** Debug/e2e handle: what the driver believes it is drawing. */
  describe() {
    return {
      nodes: this.nodes.length,
      webgpu: this.webgpu,
      bloom: this.bloomPipe !== null && this.bloomOn,
      k: { x: this.axes.x.k, y: this.axes.y.k, z: this.axes.z.k },
      curved: { x: this.axes.x.curved, y: this.axes.y.curved, z: this.axes.z.curved },
      camera: { az: this.az, el: this.el, dist: this.dist },
      playback: {
        playing: this.playing,
        playhead: this.playhead,
        revealed: this.revealedCount(),
        speed: this.playSpeed,
      },
    };
  }

  // ── build ────────────────────────────────────────────────────────────────

  private rebuild(): void {
    this.clearData();
    if (this.analyses.length === 0) {
      this.seqSorted = new Float32Array(0);
      this.resetPlayback();
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
    const mk = (vals: number[], mode: SessionsAxisMode): AxisScale => {
      const max = vals.reduce((m, v) => (v > m ? v : m), 0);
      return asinhScale(max, this.axisK(mode, vals, max));
    };
    this.axes = {
      x: mk(xs, this.cfg.axisTime),
      y: mk(ys, this.cfg.axisContext),
      z: mk(zs, this.cfg.axisNewContext),
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
      const graph = buildAgentGraph(a.turns);
      const posOf = new Map<number, THREE.Vector3>();
      for (const raw of a.turns) {
        // a stored analysis can carry a category this build has no colour for
        // (written by another build; the raw transcript is never persisted, so
        // it can't be re-parsed) — re-derive from the tools that WERE stored
        const t = VALID_CATEGORIES.has(raw.category)
          ? raw
          : { ...raw, category: dominantCategory(raw.tools) };
        const pos = new THREE.Vector3(
          this.axes.x.toUnit(t.tSec) * CUBE - HALF,
          this.axes.y.toUnit(t.cacheRead) * CUBE - HALF,
          this.axes.z.toUnit(t.cacheWrite) * CUBE - HALF,
        );
        this.nodes.push({
          sessionId: a.id,
          sessionName: a.name,
          index: t.index,
          turn: t,
          pos,
          step: graph.byTurn.get(t.index),
          agentCount: graph.paths.length,
        });
        posOf.set(t.index, pos);
      }
      // One polyline PER AGENT, not one per session. A single walk down
      // `a.turns` emits an edge wherever the agent changes — parent→sub on the
      // way in and sub→parent on the way out — and neither is a step either
      // agent took. Walking each agent's own steps also means the parent's path
      // closes over the sub-agent that interrupted it, which is the true shape:
      // the parent was inside that one tool call the whole time.
      for (const path of graph.paths) {
        let prev: THREE.Vector3 | null = null;
        for (const s of path.steps) {
          const pos = posOf.get(s.turn);
          if (!pos) continue;
          if (prev) trailSegs.push({ a: prev, b: pos, rgb: hue, sessionId: a.id, index: s.turn });
          prev = pos;
        }
      }
    }

    this.buildField(maxTools);
    this.buildTrail(trailSegs);
    this.applyVisibility();
    this.layoutLabels();

    // a fresh dataset always shows fully-revealed; playback is opt-in from the
    // transport. Cache the sorted seq column for the revealed-count search.
    const seqs = this.nodes.map((nd) => (nd.pos.x + HALF) / CUBE);
    seqs.sort((a, b) => a - b);
    this.seqSorted = Float32Array.from(seqs);
    this.resetPlayback();

    this.cameraDirty = true;
  }

  private buildField(maxTools: number): void {
    const n = this.nodes.length;
    const pos = new Float32Array(n * 3);
    // RAW per-turn attributes only — every mapping (gamma, saturation, hue,
    // emissive) lives in the shader against uniforms, so the appearance knobs
    // retune the field in place without rebuilding a buffer. The five scalars
    // are PACKED into two attributes and swizzled in TSL: WebGPU caps a material
    // at 8 vertex buffers and rejects the ninth SILENTLY (see the ChordDriver
    // note in the skill), and adding the playback `seq` as its own attribute
    // would have tipped iPos+5 scalars+iVis+uv over that edge.
    //   iA = (rank, toolFrac, catIndex, fail)      iB = (sub, seq)
    // seq is the turn's position along the honest time axis in [0,1] — the exact
    // coordinate the playback wavefront (uPlayhead) sweeps through.
    const packA = new Float32Array(n * 4);
    const packB = new Float32Array(n * 2);
    this.visArray = new Float32Array(n).fill(1);

    const rank = outputRank(this.nodes);

    for (let i = 0; i < n; i++) {
      const node = this.nodes[i]!;
      const t = node.turn;
      pos[i * 3] = node.pos.x;
      pos[i * 3 + 1] = node.pos.y;
      pos[i * 3 + 2] = node.pos.z;
      packA[i * 4] = rank[i]!; // output-token rank, 0..1 (see outputRank)
      packA[i * 4 + 1] = Math.min(t.tools.length / maxTools, 1); // tools / max
      packA[i * 4 + 2] = CATEGORY_INDEX[t.category] ?? 0; // category slot, 0..5
      packA[i * 4 + 3] = (t.errors ?? 0) > 0 ? 1 : 0; // failure flag
      packB[i * 2] = t.isSidechain ? 1 : 0; // sub-agent flag
      packB[i * 2 + 1] = (node.pos.x + HALF) / CUBE; // seq: unit time position
    }

    const iPos = instancedBufferAttribute<"vec3">(new THREE.InstancedBufferAttribute(pos, 3), "vec3");
    const iA = instancedBufferAttribute<"vec4">(new THREE.InstancedBufferAttribute(packA, 4), "vec4");
    const iB = instancedBufferAttribute<"vec2">(new THREE.InstancedBufferAttribute(packB, 2), "vec2");
    this.visAttr = new THREE.InstancedBufferAttribute(this.visArray, 1);
    this.visAttr.setUsage(THREE.DynamicDrawUsage);
    const iVis = instancedDynamicBufferAttribute<"float">(this.visAttr, "float");

    // unpack — every downstream term reads these exactly as before
    const iRank = iA.x;
    const iTool = iA.y;
    const iCat = iA.z;
    const iFail = iA.w;
    const iSub = iB.x;
    const iSeq = iB.y;

    // ── shared shader terms ──
    // Brightness carries output tokens, RANKED (see outputRank), shaped by the
    // glow-contrast gamma: higher gamma keeps the bulk of the field dark and
    // sharpens the spikes. A failed turn (when markFailures is on) is floored
    // to failureGlow so it always reads.
    const glow0 = iRank.pow(this.uGamma);
    const failOn = iFail.mul(this.uMarkFail); // 0 or 1
    const glow = mix(glow0, glow0.max(this.uFailFloor), failOn);
    // Size is a SECOND channel: tool calls this turn, over a deliberately
    // narrow range so it never competes with brightness.
    const toolSize = float(0.55).add(iTool.mul(0.45));

    const hovered = instanceIndex.toFloat().equal(this.uHover);
    const picked = instanceIndex.toFloat().equal(this.uSelected);
    const emphasis = select(picked, this.uSelectEmph, select(hovered, this.uHoverEmph, float(1)));

    // category hue: pick the matching per-category uniform by index. iCat holds
    // exact integers 0..5 in CATEGORY_ORDER; the 0.5 thresholds bracket each.
    const catColor = select(
      iCat.lessThan(0.5),
      this.uCats[0]!,
      select(
        iCat.lessThan(1.5),
        this.uCats[1]!,
        select(
          iCat.lessThan(2.5),
          this.uCats[2]!,
          select(
            iCat.lessThan(3.5),
            this.uCats[3]!,
            select(iCat.lessThan(4.5), this.uCats[4]!, this.uCats[5]!),
          ),
        ),
      ),
    );
    const base = mix(catColor, this.uError, failOn);
    // saturation ramps with magnitude: quiet turns wash toward neutral, loud
    // ones carry full category hue; a marked failure is always full saturation.
    const satNormal = this.uSatFloor.add(float(1).sub(this.uSatFloor).mul(glow));
    const sat = mix(satNormal, float(1), failOn);
    const rgb = mix(this.uNeutral, base, sat);

    // ── timeline reveal + wavefront glow ──
    // A turn ignites exactly as the wavefront reaches its timestamp, easing in
    // over REVEAL_FADE just ahead of it (smoothstep edges must increase on WGSL,
    // so the fade band is seq-fade → seq). At rest uPlayhead = 1 ≥ every seq, so
    // reveal = 1 for all and the plot is identical to a static render.
    const reveal = this.uPlayhead.smoothstep(iSeq.sub(this.uRevealFade), iSeq);
    // extra emissive right at the wavefront, decaying over uPulseWidth behind it.
    // uPulse is 0 unless actively playing (and stays 0 under reduced motion), so
    // this whole term vanishes at rest.
    const behind = this.uPlayhead.sub(iSeq).max(0);
    const pulse = behind
      .smoothstep(float(0), this.uPulseWidth)
      .oneMinus()
      .mul(reveal)
      .mul(this.uPulse);

    const material = new THREE.SpriteNodeMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
    });

    material.positionNode = iPos;
    // the bright turns also spread a little wider — the two channels reinforce
    // rather than fight, and a hot core with a wide halo is what reads as glow.
    // The wavefront gives a small extra pop as it ignites each turn.
    material.scaleNode = this.uSize
      .mul(toolSize)
      .mul(float(0.7).add(glow.mul(1.1)))
      .mul(emphasis)
      .mul(float(1).add(pulse.mul(0.45)));
    // emissive: the ceiling (glowStrength) exceeds 1.0 on purpose — that
    // headroom is what the bloom threshold keys on, so only real spikes glow.
    // The wavefront pulse pushes an igniting turn transiently over that ceiling.
    material.colorNode = rgb
      .mul(mix(float(EMISSIVE_MIN), this.uGlowMax, glow))
      .mul(float(1).add(pulse));

    // soft radial falloff — no hard edge, so overlapping points accumulate
    // rather than occlude
    const d = uv().sub(0.5).length();
    const disc = d.smoothstep(0.06, 0.5).oneMinus();
    const alpha = mix(this.uMoteFloor, float(PEAK_ALPHA), glow);
    const subDim = mix(float(1), this.uSubOpacity, iSub);
    material.opacityNode = disc
      .mul(select(picked.or(hovered), float(1), alpha))
      .mul(subDim)
      .mul(iVis)
      .mul(reveal); // unrevealed turns are fully transparent until the sweep arrives

    this.fieldMat = material;
    const sprite = new THREE.Sprite(material);
    sprite.count = n;
    sprite.frustumCulled = false;
    sprite.renderOrder = 3;
    this.field = sprite;
    this.scene.add(sprite);

    // id companion for GPU picking — shares the same attribute nodes so a pick
    // can never drift from what is on screen. Its base scale matches the field
    // (minus the hover/select emphasis) so the hit target tracks the mote.
    const idMat = new THREE.SpriteNodeMaterial({ transparent: false });
    idMat.positionNode = iPos;
    idMat.scaleNode = this.uSize.mul(toolSize).mul(float(0.7).add(glow.mul(1.1))).mul(2.4); // finger-friendly
    const id = instanceIndex.add(1).toFloat();
    idMat.colorNode = vec3(id.mod(256), id.div(256).floor().mod(256), id.div(65536).floor()).div(255);
    // only fully-visible, already-revealed motes are pickable — a legend-dimmed
    // category (iVis < 1) or a turn the wavefront has not reached (reveal < 1)
    // drops out of picking, so a hover can never land on something unseen.
    idMat.opacityNode = select(
      d.lessThan(0.45),
      select(iVis.greaterThanEqual(0.999).and(reveal.greaterThanEqual(0.999)), float(1), float(0)),
      float(0),
    );
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
    // growth: how far the wavefront has advanced across this segment's time span,
    // 0..1. Clamping the along-ribbon parameter `t` to `frac` collapses the
    // not-yet-drawn tail onto the wavefront point (zero area → invisible), so the
    // path draws itself turn by turn. At rest uPlayhead = 1 ≥ seqEnd, frac = 1,
    // and the full ribbon is drawn exactly as before.
    //
    // The endpoints' seq is DERIVED from their X rather than carried in its own
    // attribute — X is the time axis, so this is the same number, and an extra
    // attribute here would be the ninth vertex buffer. An InstancedMesh on a
    // PlaneGeometry already binds position+normal+uv+instanceMatrix (4) before
    // aStart/aEnd/aColor/aFocus (4), which is exactly the WebGPU max of 8; the
    // ninth is rejected SILENTLY at pipeline creation and the trail vanishes.
    const seqStart = aStart.x.add(HALF).div(CUBE);
    const seqEnd = aEnd.x.add(HALF).div(CUBE);
    const frac = this.uPlayhead.sub(seqStart).div(seqEnd.sub(seqStart).max(1e-4)).clamp(0, 1);
    const grown = t.min(frac);
    material.positionNode = aStart.add(dir.mul(grown)).add(perp.mul(across.mul(this.uTrailWidth)));
    material.colorNode = aColor;
    const edgeFade = across.abs().mul(2).smoothstep(0.2, 1).oneMinus();
    // a brief brightening while a segment is actively growing (0 < frac < 1)
    const activeTip = frac.smoothstep(0, 0.12).mul(frac.smoothstep(0.88, 1).oneMinus()).mul(this.uTrailPulse);
    material.opacityNode = edgeFade.mul(
      mix(this.uTrailAlpha, this.uTrailFocusAlpha, aFocus).add(activeTip),
    );

    this.trailMat = material;
    const mesh = new THREE.InstancedMesh(new THREE.PlaneGeometry(1, 1, 1, 1), material, n);
    mesh.count = n;
    mesh.frustumCulled = false;
    mesh.renderOrder = 2;
    mesh.visible = this.cfg.showTrails;
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
      color: new THREE.Color(this.cfg.frameColor),
      transparent: true,
      opacity: this.cfg.frameOpacity,
      depthWrite: false,
    });
    this.frameMat = mat;
    this.frame3 = new THREE.LineSegments(geo, mat);
    this.frame3.frustumCulled = false;
    this.frame3.renderOrder = 1;
    this.frame3.visible = this.cfg.showFrame;
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
    if (!this.cfg.showProbe || i === null || i < 0 || i >= this.nodes.length) {
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

  /** The wavefront marker — a bright rectangle in the context/new-context plane
   *  that rides the playhead across the time axis. It reads as the "now" line of
   *  the replay and blooms softly on the WebGPU rung. Hidden at rest. */
  private buildSweep(): void {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(8 * 3), 3));
    const mat = new THREE.LineBasicNodeMaterial({
      color: new THREE.Color(0x4d8dff), // --accent
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
      depthTest: false,
    });
    this.sweepMat = mat;
    this.sweep = new THREE.LineSegments(geo, mat);
    this.sweep.frustumCulled = false;
    this.sweep.renderOrder = 5;
    this.sweep.visible = false;
    this.scene.add(this.sweep);
  }

  private updateSweep(): void {
    if (!this.sweep) return;
    const active = this.nodes.length > 0 && (this.playing || this.playhead < 0.9999);
    if (!active) {
      this.sweep.visible = false;
      return;
    }
    const x = this.playhead * CUBE - HALF;
    const arr = (this.sweep.geometry.getAttribute("position") as THREE.BufferAttribute)
      .array as Float32Array;
    // rectangle in the Y/Z plane at x, as four edges (8 vertices)
    arr.set([
      x, -HALF, -HALF, x, HALF, -HALF,
      x, HALF, -HALF, x, HALF, HALF,
      x, HALF, HALF, x, -HALF, HALF,
      x, -HALF, HALF, x, -HALF, -HALF,
    ]);
    (this.sweep.geometry.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
    this.sweep.visible = true;
  }

  /** Light the run of segments around the active turn and let the rest fall back
   *  to ambient. A linear ramp over trailFocusSpan turns, so the emphasis has
   *  a direction you can follow rather than a hard edge. */
  private updateTrailFocus(): void {
    if (!this.focusAttr) return;
    const span = Math.max(1, this.cfg.trailFocusSpan);
    const i = this.selected ?? this.hovered;
    const active = i === null ? null : this.nodes[i];
    for (let s = 0; s < this.trailKeys.length; s++) {
      const key = this.trailKeys[s]!;
      if (!active || key.sessionId !== active.sessionId) {
        this.focusArray[s] = 0;
        continue;
      }
      const d = Math.abs(key.index - active.index);
      this.focusArray[s] = d > span ? 0 : 1 - d / span;
    }
    this.focusAttr.needsUpdate = true;
    this.cameraDirty = true;
  }

  private applyVisibility(): void {
    if (!this.visAttr) return;
    const dim = this.cfg.dimmedOpacity;
    for (let i = 0; i < this.nodes.length; i++) {
      this.visArray[i] = this.hiddenCats.has(this.nodes[i]!.turn.category) ? dim : 1;
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

    this.advancePlayback();
    this.maybePick();

    // bloom rides the webgpu rung and only when the global toggle is on;
    // otherwise render straight so the switch is live with no teardown
    if (this.bloomPipe && this.bloomOn) this.bloomPipe.post.render();
    else this.renderer.render(this.scene, this.camera);
  };

  /** Advance the playhead by wall-clock time while playing. The sweep moves
   *  linearly in axis space, so on an eased time axis idle gaps compress and the
   *  replay never stalls; PLAY_BASE_SECONDS is one full pass at 1×. */
  private advancePlayback(): void {
    if (!this.playing || this.nodes.length === 0) return;
    const now = performance.now();
    // cap dt so returning from a backgrounded tab doesn't teleport the sweep
    const dt = Math.min((now - this.lastClock) / 1000, 0.1);
    this.lastClock = now;
    const dur = PLAY_BASE_SECONDS / Math.max(this.playSpeed, 1e-3);
    this.playhead += dt / dur;
    if (this.playhead >= 1) {
      this.playhead = 1;
      this.playing = false;
      this.uPlayhead.value = 1;
      this.uPulse.value = 0;
      this.uTrailPulse.value = 0;
      this.updateSweep();
      this.emitPlayback(true);
      return;
    }
    this.uPlayhead.value = this.playhead;
    this.updateSweep();
    this.emitPlayback(false);
  }

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
    if (this.cfg.showLabels && this.nodes.length > 0) {
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
    const st = n.step;
    const rows: [string, string][] = [
      ["Time", fmtSecs(t.tSec)],
      // Elapsed since the PREVIOUS step of this same agent. Em-dash on an
      // agent's first step and on any pair missing a timestamp — a gap that
      // could not be measured is not a gap of zero. The x axis already carries
      // this geometrically; the number is here because that axis is asinh, so
      // eye-reading a span off it is not the same as knowing it.
      ["Since prev", st?.gapSec != null ? fmtGap(st.gapSec) : "—"],
      ["Output", t.outputTokens.toLocaleString()],
      ["Context", t.cacheRead.toLocaleString()],
      ["New context", t.cacheWrite.toLocaleString()],
      ["Tools", String(t.tools.length)],
    ];
    if (err !== undefined && err > 0) rows.push(["Failures", String(err)]);
    // Only when the session actually ran more than one agent. On a single-agent
    // session `step` equals the `#N` already in the header, and a row that
    // repeats the header teaches nothing.
    if (st && n.agentCount > 1) {
      rows.push([st.agentId === "main" ? "Main step" : "Sub-agent step", `${st.step} of ${st.ofSteps}`]);
    }
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
      this.frameMat = null;
    }
    if (this.probe) {
      this.scene.remove(this.probe);
      this.probe.geometry.dispose();
      (this.probe.material as THREE.Material).dispose();
      this.probe = null;
    }
    if (this.sweep) {
      this.scene.remove(this.sweep);
      this.sweep.geometry.dispose();
      (this.sweep.material as THREE.Material).dispose();
      this.sweep = null;
      this.sweepMat = null;
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

/** Like `fmtSecs` but keeps a decimal under 10s. Step gaps live down there —
 *  rounding a 0.4s gap to "0s" would print a duration that did not happen. */
function fmtGap(s: number): string {
  if (s >= 3600) return `${(s / 3600).toFixed(1)}h`;
  if (s >= 60) return `${Math.round(s / 60)}m`;
  if (s >= 10) return `${Math.round(s)}s`;
  return `${s.toFixed(1)}s`;
}
