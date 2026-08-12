/** Cross-model comparison — three.js WebGPU + TSL emissive field.
 *
 *  Was a bespoke raw-WGSL driver (a line-for-line port of the retired
 *  backend/viewer.py). It moved onto the three-TSL rung when the view gained
 *  bloom: the routing table in the nebulai-viz skill sends "instanced point
 *  clouds, morphs, bloom, and a WebGL fallback" to three+TSL, and hand-rolling
 *  a bright-pass/blur/composite chain in WGSL would have duplicated
 *  `scene/post/bloom.ts` for no gain. The port also drops the WebGPU-only
 *  restriction — TSL lowers the same node graph to GLSL, so the view now has a
 *  forceWebGL rung (without bloom, exactly as AtlasDriver degrades).
 *
 *  WHAT IT DRAWS. 837 cluster concepts from N models, each carrying its
 *  position in four layout states. The states are blended on the GPU from a
 *  single weight vec4, so a transition is one uniform write and no rebuild.
 *
 *  ENCODING (the legend says all of this):
 *    hue   = source model — the identity channel, never desaturated, because
 *            telling the models apart IS the job of this view.
 *    size  = cluster token count, sqrt-scaled (area tracks the count).
 *    glow  = the same token count, RANKED. Deliberately redundant with size:
 *            837 equal-weight discs read as texture, and the rank is what puts
 *            the big concepts above the bloom threshold so a figure emerges
 *            from the ground. Monotone in the raw count; the tooltip prints it.
 *
 *  At ~840 points the CPU projection hover loop is exact and cheap; that
 *  pattern does NOT scale to the atlas's 50K (which is why AtlasDriver uses
 *  kdbush / GPU id-buffer picking). */

import * as THREE from "three/webgpu";
import { float, instancedBufferAttribute, mix, select, uniform, uv, vec3 } from "three/tsl";

import type { CompareTourState } from "../../app/actions";
import { appStore, type CompareUI } from "../../app/store";
import {
  compareAttributes,
  type CompareAttributes,
  type CompareData,
} from "../../data/compare";
import { createBloomPipeline, type BloomPipeline } from "../post/bloom";
import { GestureRecognizer } from "../gestures";
import { BG } from "@psychix/viz/tokens";

const STATE_TWEEN_MS = 900;
const REDUCED_TWEEN_MS = 150;

// ── field look ───────────────────────────────────────────────────────────────
/** World-unit base radius; the pointScale setting multiplies it.
 *
 *  The single most sensitive constant in this file, and not for the reason you
 *  expect: it controls how often sprites OVERLAP, and additive blending sums
 *  overlaps. Doubling it does not make the field twice as bright, it makes the
 *  dense core clip to white and takes the model hue with it. */
const POINT_SIZE = 0.38;
/** Emissive floor — what the dimmest concept still carries, so nothing is lost.
 *  Kept high enough that the bulk of the field reads as a coloured haze rather
 *  than a sparse sparkle: most concepts are small, and a view about model
 *  overlap has to show the bulk, not just the outliers. */
const EMISSIVE_MIN = 0.52;
/** Emissive ceiling. Above 1.0 on purpose: that headroom is what the bloom
 *  threshold keys on, so only genuine spikes glow.
 *
 *  Held well below the sessions field's 2.1 on purpose. Additive blending SUMS
 *  overlapping sprites, and 837 concepts land in a much tighter blob than a
 *  session's flight path — at 2.0 the cluster core saturated all three channels
 *  and the whole field read white, which destroys the one encoding this view
 *  exists for (model identity is the hue). Anything that raises this must be
 *  checked against the densest layout ("semantic"), not the sparsest. */
const EMISSIVE_MAX = 1.6;
/** Gamma on the size rank. Higher keeps the bulk of the field dark and sharpens
 *  the spikes; below ~1.4 it washes back to a flat scatter. */
const GLOW_GAMMA = 1.9;
/** Opacity of the dimmest motes. */
const MOTE_FLOOR = 0.42;
/** How much a shared concept is lifted toward white — a tint, NOT brightness,
 *  so it stays readable as a separate channel from the size ramp. */
const SHARED_TINT = 0.16;

// Bloom, tuned for THIS view's density rather than the shared defaults
// (0.85 / 0.4 / 0.55). The high threshold is the important one: it keeps the
// glow on the handful of genuinely large concepts instead of letting 837
// overlapping halos sum to white and erase the model hue. Re-tune by measuring
// mean pixel saturation on the "semantic" layout, not by eye on a sparse one.
const BLOOM_STRENGTH = 0.85;
const BLOOM_RADIUS = 0.4;
const BLOOM_THRESHOLD = 0.6;

// ── layout tour (playback) ───────────────────────────────────────────────────
/** Wall-clock seconds for one full native → semantic → by-model → by-concept
 *  pass at 1×, dwells included. */
const TOUR_BASE_SECONDS = 19;
/** Relative weight of a dwell (parked on one layout) in the tour timeline. */
const DWELL_W = 1;
/** Relative weight of a transition. Longer than a dwell — the morph between
 *  layouts is the thing worth watching. */
const TRANS_W = 1.5;
/** Extra emissive (×) at the midpoint of a transition, so the field flares
 *  while it is in motion and settles when it lands. */
const FLARE_AMT = 0.85;

interface Blend {
  from: number;
  to: number;
  mix: number;
  /** Optional 4-weight origin, overriding `from`. A hand-driven tween starts
   *  from whatever the field is showing — which may be a morph in flight, and
   *  so may not be any single layout. Without this the field has to round to
   *  the nearest layout first, and that round is a visible jump. */
  fromW?: readonly number[];
}

export class CompareDriver {
  /** Emitted as the tour advances (~18Hz while playing, forced on transport
   *  events). main.ts forwards it to the chrome signal. */
  onTour: ((s: CompareTourState) => void) | null = null;

  private canvas!: HTMLCanvasElement;
  private renderer!: THREE.WebGPURenderer;
  private scene = new THREE.Scene();
  // 40°, not the 57.3° (=1 rad) this started at. A wide FOV magnifies whatever
  // is nearest the camera by ~1.7×, which on an oblique orbit throws the near
  // half of the cloud outside a fit computed from world-space radii. 40° keeps
  // enough depth cue to read the 3D structure without that distortion.
  private camera = new THREE.PerspectiveCamera(40, 1, 0.1, 600);
  private webgpu = true;
  private bloomPipe: BloomPipeline | null = null;
  private bloomOn = true;
  private disposed = false;

  private field: THREE.Sprite | null = null;
  private fieldMat: THREE.SpriteNodeMaterial | null = null;

  // ── uniforms ──────────────────────────────────────────────────────────────
  /** Per-state blend weights. One vec4 replaces the old from/to/t branch: the
   *  shader is `p0*w.x + p1*w.y + p2*w.z + p3*w.w`, which is exact for a dwell
   *  (a single 1) and for a transition (two weights summing to 1), and costs no
   *  branching at all. */
  private uWeights = uniform(new THREE.Vector4(0, 1, 0, 0));
  private uPointScale = uniform(1);
  private uVisA = uniform(new THREE.Vector4(1, 1, 1, 1));
  private uVisB = uniform(new THREE.Vector4(1, 1, 1, 1));
  private uSharedOnly = uniform(0);
  private uFlare = uniform(0);

  private data: CompareData | null = null;
  /** The packed + normalised attribute arrays — the exact positions the GPU
   *  draws, which is what CPU hover must project. */
  private attrs: CompareAttributes | null = null;
  private count = 0;

  // orbit camera. Angles are the user's; the radius is fitted to the data in
  // setData. Every state is normalised around the origin, so there is no pan.
  private theta = 0.7;
  private phi = 0.35;
  private radius = 46;
  /** Common world radius every layout state is normalised to (STATE_EXTENT). */
  private extent = 10;

  // manual state tween (the radio buttons)
  private curState = 1;
  private fromState = 1;
  private toState = 1;
  private tAnim = 1;
  private animStart = 0;
  private animing = false;
  /** Where the current hand-driven tween started, as the full 4-weight vector
   *  (see Blend.fromW). Null once the tween lands on a single layout. */
  private fromWeights: number[] | null = null;

  // tour state
  /** The tour CLOCK is advancing. */
  private playing = false;
  /** The field's position comes from `progress` rather than the manual tween.
   *
   *  Split from `playing` because pausing and scrubbing are not "leaving the
   *  tour". With one flag, pause() fell back to the manual tween and rounded
   *  the field to the nearest layout — so pausing mid-morph jumped, and the
   *  1000-position scrubber collapsed to the 4 dwell positions, which is most
   *  of what a scrubber is for. Only a radio pick actually leaves the tour. */
  private tourEngaged = false;
  private progress = 0;
  private tourSpeed = 1;
  private lastClock = 0;
  private lastEmit = 0;
  private reducedMotion = false;

  private sharedOnly = 0;
  private hidden: number[] = [];

  private cssW = 2;
  private cssH = 2;

  private tooltip: HTMLElement | null = null;
  private unsubscribe: (() => void) | null = null;
  private abort = new AbortController();
  /** touch pan/pinch — see src/scene/gestures.ts; the mouse path below is untouched */
  private gestures: GestureRecognizer | null = null;

  async init(canvas: HTMLCanvasElement): Promise<void> {
    this.canvas = canvas;
    const tier = appStore.getState().capabilities?.tier ?? "webgpu";
    this.webgpu = tier === "webgpu";

    this.renderer = new THREE.WebGPURenderer({
      canvas,
      antialias: true,
      forceWebGL: !this.webgpu,
    });
    await this.renderer.init();
    if (this.disposed) return;
    this.renderer.setClearColor(new THREE.Color(BG), 1);
    // The one thing that makes a saturated additive field survive bloom.
    // With no tone mapping (what every other driver here does) values above 1
    // HARD-CLIP per channel, so a hot blue concept plus a bloom halo lands at
    // (1,1,1) and the model hue — the only encoding this view has — is gone.
    // Khronos PBR Neutral rolls highlights off while holding hue, which is
    // exactly the failure it was designed for. Measured on the "semantic"
    // layout: 84% of lit pixels below 0.15 saturation before, ~15% after.
    this.renderer.toneMapping = THREE.NeutralToneMapping;

    this.bloomOn = appStore.getState().settings.bloom;
    if (this.webgpu) {
      // see BloomOptions — the compare field is far denser than the atlas, so
      // it needs a higher threshold or the summed halos wash the hue out
      this.bloomPipe = createBloomPipeline(this.renderer, this.scene, this.camera, "full", {
        strength: BLOOM_STRENGTH,
        radius: BLOOM_RADIUS,
        threshold: BLOOM_THRESHOLD,
      });
    }

    this.reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.initInput();
    this.initTooltip();

    // mirror store → driver (chrome writes the store; we only read). The tour
    // deliberately does NOT write back — it reports through onTour instead, so
    // the store stays chrome-owned and there is no feedback loop.
    // -1 so the first apply is always treated as a layout pick (and settles the
    // driver on whatever the permalink/store says) without a spurious tween
    let lastPick = -1;
    const apply = (c: CompareUI) => {
      this.sharedOnly = c.sharedOnly ? 1 : 0;
      this.uSharedOnly.value = this.sharedOnly;
      this.hidden = c.hiddenModels;
      const vis = [1, 1, 1, 1, 1, 1, 1, 1];
      for (let i = 0; i < 8; i++) vis[i] = c.hiddenModels.includes(i) ? 0 : 1;
      this.uVisA.value.set(vis[0]!, vis[1]!, vis[2]!, vis[3]!);
      this.uVisB.value.set(vis[4]!, vis[5]!, vis[6]!, vis[7]!);
      // a layout pick cancels the tour — the user took the wheel. Compared
      // against the last STORE value, not the driver's, so toggling a model or
      // the shared filter mid-tour doesn't stop it.
      if (c.state !== lastPick) {
        const first = lastPick < 0;
        lastPick = c.state;
        if (this.tourEngaged) this.stopTour();
        if (first) {
          this.curState = clampIdx(c.state);
          this.fromState = this.curState;
          this.toState = this.curState;
          this.progress = this.stops()[this.curState] ?? 0;
        } else {
          this.goto(c.state);
        }
      }
    };
    apply(appStore.getState().compare);
    let prev = appStore.getState().compare;
    this.unsubscribe = appStore.subscribe((s) => {
      if (s.compare !== prev) {
        prev = s.compare;
        apply(s.compare);
      }
      this.uPointScale.value = s.settings.pointScale;
      this.bloomOn = s.settings.bloom;
    });
  }

  setData(data: CompareData): void {
    this.data = data;
    this.count = data.points.length;
    this.clearField();
    if (this.count === 0) return;

    const at = compareAttributes(data);
    this.attrs = at;
    const iA = instancedBufferAttribute<"vec4">(new THREE.InstancedBufferAttribute(at.a, 4), "vec4");
    const iB = instancedBufferAttribute<"vec4">(new THREE.InstancedBufferAttribute(at.b, 4), "vec4");
    const iC = instancedBufferAttribute<"vec4">(new THREE.InstancedBufferAttribute(at.c, 4), "vec4");
    const iD = instancedBufferAttribute<"vec4">(new THREE.InstancedBufferAttribute(at.d, 4), "vec4");
    const iColor = instancedBufferAttribute<"vec3">(
      new THREE.InstancedBufferAttribute(at.color, 3),
      "vec3",
    );

    this.fitCamera(at.extent);

    const iRadius = iA.w;
    const iSrc = iB.w;
    const iShared = iC.w;
    const iRank = iD.w;

    // ── position: a weighted sum of the four layout states ──
    const world = iA.xyz
      .mul(this.uWeights.x)
      .add(iB.xyz.mul(this.uWeights.y))
      .add(iC.xyz.mul(this.uWeights.z))
      .add(iD.xyz.mul(this.uWeights.w));

    // ── visibility: model toggles + shared-only, both pure uniforms ──
    // srcIdx holds exact integers 0..7; the .5 thresholds bracket each slot.
    const vis = select(
      iSrc.lessThan(0.5),
      this.uVisA.x,
      select(
        iSrc.lessThan(1.5),
        this.uVisA.y,
        select(
          iSrc.lessThan(2.5),
          this.uVisA.z,
          select(
            iSrc.lessThan(3.5),
            this.uVisA.w,
            select(
              iSrc.lessThan(4.5),
              this.uVisB.x,
              select(
                iSrc.lessThan(5.5),
                this.uVisB.y,
                select(iSrc.lessThan(6.5), this.uVisB.z, this.uVisB.w),
              ),
            ),
          ),
        ),
      ),
    );
    // sharedOnly = 1 → the factor becomes the shared flag; = 0 → it stays 1
    const shownShared = mix(float(1), iShared, this.uSharedOnly);
    const shown = vis.mul(shownShared);

    // ── brightness: the size rank, gamma-shaped ──
    const glow = iRank.pow(float(GLOW_GAMMA));
    // flare while a transition is in flight — the field is hottest mid-morph
    const flare = float(1).add(this.uFlare);

    const material = new THREE.SpriteNodeMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
    });
    material.positionNode = world;
    // radius carries the token count directly; the rank adds only a gentle
    // widening so a hot core gets a halo (that is what reads as glow) without
    // compounding size into the radius twice
    material.scaleNode = this.uPointScale
      .mul(float(POINT_SIZE))
      .mul(iRadius)
      .mul(float(0.8).add(glow.mul(0.5)))
      .mul(float(1).add(this.uFlare.mul(0.22)));
    // hue is never touched — dimming a saturated colour keeps the model
    // identifiable, so no saturation floor is needed here. Shared concepts get
    // a small lift toward white: a tint channel, independent of the size ramp.
    const tinted = mix(iColor, vec3(1, 1, 1), iShared.mul(float(SHARED_TINT)));
    material.colorNode = tinted
      .mul(mix(float(EMISSIVE_MIN), float(EMISSIVE_MAX), glow))
      .mul(flare);

    // soft radial falloff — no hard edge, so overlapping concepts accumulate
    // rather than occlude each other
    const d = uv().sub(0.5).length();
    const disc = d.smoothstep(0.06, 0.5).oneMinus();
    material.opacityNode = disc.mul(mix(float(MOTE_FLOOR), float(1), glow)).mul(shown);

    this.fieldMat = material;
    const sprite = new THREE.Sprite(material);
    sprite.count = this.count;
    sprite.frustumCulled = false;
    this.field = sprite;
    this.scene.add(sprite);
    // the tour geometry only becomes knowable once the state list arrives, so
    // park the scrubber on the layout the store asked for and tell the panel
    this.progress = this.stops()[clampIdx(this.curState)] ?? 0;
    this.applyBlend(this.currentBlend());
    this.emitTour(true);
  }

  /** Frame the camera on the data instead of a hard-coded radius.
   *
   *  This is one line only because compareAttributes normalises every state to
   *  the same sphere (see normaliseState) — so one fit frames all four and the
   *  camera never has to pan or dolly through a morph, which is exactly when
   *  the viewer is trying to track individual concepts moving. The 46 this
   *  replaces framed the semantic layout at ~28% of the stage. */
  private fitCamera(extent?: number): void {
    if (extent !== undefined) this.extent = extent;
    // The stage canvas is full-bleed and the chrome floats ON TOP of it: the
    // settings card covers the left edge, the legend rail the right, the top
    // bar and the transport the top and bottom. Fitting to the canvas puts
    // content under the panels — "by model" strings its four clusters along a
    // horizontal band, and the last one landed behind the rail. So the fit
    // targets the clear middle, and takes whichever axis binds first. The
    // margin also absorbs the sprite radii (the fit only sees centres) and the
    // ~2% of points outside the p98 sphere.
    const VFILL = 0.72;
    const HFILL = 0.44;
    const aspect = Math.max(this.camera.aspect || 1, 0.2);
    const fill = Math.min(VFILL, HFILL * aspect);
    this.radius =
      Math.max(this.extent, 1e-3) / (Math.tan((this.camera.fov * Math.PI) / 360) * fill);
  }

  resize(width: number, height: number, dpr: number): void {
    if (width < 2 || height < 2) return;
    this.cssW = width;
    this.cssH = height;
    this.renderer.setPixelRatio(Math.min(dpr, 2));
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.fitCamera(); // the fit depends on aspect, so it has to follow a resize
  }

  frame(_dt: number, _t: number): void {
    if (this.disposed || this.count === 0) return;

    if (this.playing) this.advanceTour();
    else if (this.animing) {
      const dur = this.reducedMotion ? REDUCED_TWEEN_MS : STATE_TWEEN_MS;
      this.tAnim = Math.min(1, (performance.now() - this.animStart) / dur);
      const done = this.tAnim >= 1;
      if (done) {
        this.animing = false;
        this.curState = this.toState;
        this.fromState = this.toState;
        this.fromWeights = null; // landed on a single layout
      }
      this.emitTour(done); // the readout follows a hand-driven morph too
    }
    this.applyBlend(this.currentBlend());

    const ce = Math.cos(this.phi);
    this.camera.position.set(
      this.radius * ce * Math.cos(this.theta),
      this.radius * Math.sin(this.phi),
      this.radius * ce * Math.sin(this.theta),
    );
    this.camera.lookAt(0, 0, 0);
    this.camera.updateMatrixWorld();

    if (this.bloomPipe && this.bloomOn) this.bloomPipe.post.render();
    else this.renderer.render(this.scene, this.camera);
  }

  // ── layout blending ────────────────────────────────────────────────────────

  /** The blend the field should show right now — from the tour if it is
   *  running, otherwise from the manual radio tween. */
  private currentBlend(): Blend {
    if (this.tourEngaged) return this.tourAt(this.progress);
    if (this.animing) {
      return {
        from: this.fromState,
        to: this.toState,
        mix: smoothstep01(this.tAnim),
        fromW: this.fromWeights ?? undefined,
      };
    }
    return { from: this.curState, to: this.curState, mix: 1 };
  }

  /** Push a blend into the weight vec4 and the flare uniform. */
  private applyBlend(b: Blend): void {
    const w = this.uWeights.value;
    w.set(0, 0, 0, 0);
    const from = clampIdx(b.from);
    const to = clampIdx(b.to);
    const comps: ("x" | "y" | "z" | "w")[] = ["x", "y", "z", "w"];
    const inv = 1 - b.mix;
    if (b.fromW) for (let i = 0; i < 4; i++) w[comps[i]!] += (b.fromW[i] ?? 0) * inv;
    else w[comps[from]!] += inv;
    w[comps[to]!] += b.mix;
    // flare peaks mid-transition and vanishes at both ends; reduced motion and
    // the parked state both leave it at 0. "In flight" has to be measured
    // against the real origin: a vector origin that is already the target
    // (a re-pick of the current layout) is not a transition.
    const inFlight = b.fromW ? (b.fromW[to] ?? 0) < 0.999 : from !== to;
    const motion = inFlight && !this.reducedMotion;
    this.uFlare.value = motion ? Math.sin(Math.PI * b.mix) * FLARE_AMT : 0;
  }

  /** A layout radio pick: leave the tour and morph to that layout.
   *
   *  Separate from the store subscription because the store diff cannot see
   *  this click — see CompareTourCommand's "pick". The store write still
   *  arrives right after; goto short-circuits the duplicate. */
  pickState(s: number): void {
    if (this.tourEngaged) this.stopTour();
    this.goto(s);
  }

  private goto(s: number): void {
    if (this.animing && this.toState === clampIdx(s)) return; // already heading there
    // Start from what is on screen, not from a layout index. When idle the
    // uniform is exactly one-hot so this is identical to the old behaviour;
    // when a morph is in flight (a tour paused mid-transition, or a second
    // radio pick during a tween) it is the difference between a continuous
    // move and a jump to the nearest layout first.
    const w = this.uWeights.value;
    const cur = [w.x, w.y, w.z, w.w];
    // "already there" has to mean the FIELD is there, not just the bookkeeping:
    // leaving a paused mid-morph for its own nearest layout is still a move.
    const settled = (cur[clampIdx(this.curState)] ?? 0) > 0.999;
    if (s === this.curState && !this.animing && settled) return;
    this.fromWeights = cur;
    this.fromState = this.animing ? this.toState : this.curState;
    this.curState = this.fromState;
    this.toState = s;
    this.animStart = performance.now();
    this.animing = true;
    // park the tour scrubber on the layout the user just picked, so the thumb
    // never claims a position the field isn't showing
    this.progress = this.stops()[clampIdx(s)] ?? this.progress;
    this.emitTour(true);
  }

  // ── layout tour transport ──────────────────────────────────────────────────

  /** Segment weights across the whole tour: dwell, transition, dwell, … The
   *  dwells are what make it readable — a continuous morph never lets you see
   *  any one layout. */
  private segments(): number[] {
    const s = this.stageCount();
    const segs: number[] = [];
    for (let i = 0; i < s; i++) {
      segs.push(DWELL_W);
      if (i < s - 1) segs.push(TRANS_W);
    }
    return segs;
  }

  private stageCount(): number {
    return Math.max(this.data?.states.length ?? 0, 1);
  }

  /** Map tour progress → which layouts are blended and by how much. Even
   *  segment indices are dwells (parked, mix 0), odd ones are transitions. */
  private tourAt(u: number): Blend {
    const segs = this.segments();
    const total = segs.reduce((a, b) => a + b, 0);
    let acc = 0;
    const target = clamp(u, 0, 1) * total;
    for (let i = 0; i < segs.length; i++) {
      const w = segs[i]!;
      if (target <= acc + w || i === segs.length - 1) {
        const f = w > 0 ? clamp((target - acc) / w, 0, 1) : 0;
        const stage = i >> 1;
        if ((i & 1) === 0) return { from: stage, to: stage, mix: 1 }; // dwell
        return { from: stage, to: stage + 1, mix: smoothstep01(f) }; // transition
      }
      acc += w;
    }
    const last = this.stageCount() - 1;
    return { from: last, to: last, mix: 1 };
  }

  play(): void {
    if (this.count === 0) return;
    if (this.progress >= 0.9999) this.progress = 0;
    this.playing = true;
    this.tourEngaged = true;
    this.animing = false;
    this.lastClock = performance.now();
    this.syncStateFromTour();
    this.emitTour(true);
  }

  pause(): void {
    if (!this.playing) return;
    this.playing = false;
    // tourEngaged stays true: the field holds exactly where the tour was,
    // mid-morph included.
    this.syncStateFromTour();
    this.applyBlend(this.currentBlend());
    this.emitTour(true);
  }

  togglePlay(): void {
    if (this.playing) this.pause();
    else this.play();
  }

  restart(): void {
    if (this.count === 0) return;
    this.progress = 0;
    this.playing = true;
    this.tourEngaged = true;
    this.animing = false;
    this.lastClock = performance.now();
    this.syncStateFromTour();
    this.emitTour(true);
  }

  /** Scrub without changing play/pause state — the transport pauses first.
   *  Re-engages the tour, so dragging the scrubber shows the morphs and not
   *  just the four settled layouts. */
  seek(u: number): void {
    this.progress = clamp(u, 0, 1);
    this.tourEngaged = true;
    this.syncStateFromTour();
    this.applyBlend(this.currentBlend());
    this.emitTour(true);
  }

  setSpeed(mult: number): void {
    this.tourSpeed = mult > 0 ? mult : 1;
    this.lastClock = performance.now(); // the new rate applies from now, not retroactively
    this.emitTour(true);
  }

  /** Reduced motion keeps the tour but kills the mid-transition flare. */
  setReducedMotion(on: boolean): void {
    this.reducedMotion = on;
    this.applyBlend(this.currentBlend());
  }

  /** Leave the tour, parking the manual tween on whichever layout is nearest so
   *  the radio buttons and the field agree the moment the tour stops. */
  private stopTour(): void {
    // sync BEFORE dropping tourEngaged: the manual tween is about to become the
    // source of truth and it needs to start from where the field actually is,
    // not from whatever layout was current when play was pressed
    this.syncStateFromTour();
    this.playing = false;
    this.tourEngaged = false;
    this.emitTour(true);
  }

  /** Park the manual tween's "current layout" on whichever layout the tour is
   *  nearest. Only matters when the tour is abandoned (a radio pick): the
   *  manual tween can only interpolate between two layout indices, so a morph
   *  in flight has to round to one before a hand-driven tween can start from
   *  it. Pausing and scrubbing do NOT round — they keep tourEngaged. */
  private syncStateFromTour(): void {
    const b = this.tourAt(this.progress);
    const landed = b.mix >= 0.5 ? b.to : b.from;
    this.curState = landed;
    this.fromState = landed;
    this.toState = landed;
    this.tAnim = 1;
  }

  private advanceTour(): void {
    const now = performance.now();
    // cap dt so returning from a backgrounded tab doesn't teleport the tour
    const dt = Math.min((now - this.lastClock) / 1000, 0.1);
    this.lastClock = now;
    const dur = TOUR_BASE_SECONDS / Math.max(this.tourSpeed, 1e-3);
    this.progress += dt / dur;
    if (this.progress >= 1) {
      this.progress = 1;
      this.playing = false; // tourEngaged holds: it rests on the last layout
      this.syncStateFromTour();
      this.applyBlend(this.currentBlend());
      this.emitTour(true);
      return;
    }
    this.emitTour(false);
  }

  /** Progress at which each layout is fully settled — the MIDPOINT of its
   *  dwell, which is where the field is unambiguously showing that one layout.
   *  Handed to the UI so the scrub ticks land on the truth. */
  private stops(): number[] {
    const segs = this.segments();
    const total = segs.reduce((a, b) => a + b, 0) || 1;
    const out: number[] = [];
    let acc = 0;
    for (let i = 0; i < segs.length; i++) {
      if ((i & 1) === 0) out.push((acc + segs[i]! / 2) / total);
      acc += segs[i]!;
    }
    return out;
  }

  private tourState(): CompareTourState {
    // deliberately currentBlend(), NOT tourAt(progress): the readout must
    // describe what the field is actually showing, which during a manual radio
    // tween is the hand-driven blend and has nothing to do with the tour clock
    const b = this.currentBlend();
    const names = this.data?.states ?? [];
    return {
      playing: this.playing,
      progress: this.progress,
      stage: b.from,
      stageName: names[b.from] ?? "",
      blend: b.from === b.to ? 0 : b.mix,
      stages: this.stageCount(),
      stops: this.stops(),
      speed: this.tourSpeed,
      hasData: this.count > 0,
    };
  }

  /** Emit at most ~18Hz while playing; discrete transport events force it. */
  private emitTour(force = false): void {
    if (!this.onTour) return;
    const now = performance.now();
    if (!force && now - this.lastEmit < 55) return;
    this.lastEmit = now;
    this.onTour(this.tourState());
  }

  /** Debug/e2e handle: what the driver believes it is drawing. */
  describe() {
    const b = this.currentBlend();
    const w = this.uWeights.value;
    return {
      points: this.count,
      webgpu: this.webgpu,
      bloom: this.bloomPipe !== null && this.bloomOn,
      states: this.data?.states ?? [],
      curState: this.curState,
      hiddenModels: this.hidden,
      sharedOnly: this.sharedOnly === 1,
      camera: { theta: this.theta, phi: this.phi, radius: this.radius },
      weights: [w.x, w.y, w.z, w.w],
      flare: this.uFlare.value,
      tour: {
        playing: this.playing,
        progress: this.progress,
        stage: b.from,
        blend: b.from === b.to ? 0 : b.mix,
        speed: this.tourSpeed,
      },
    };
  }

  // ── input: orbit drag + wheel zoom + CPU hover ─────────────────────────────
  private initInput(): void {
    let dragging = false;
    let lx = 0;
    let ly = 0;
    const sig = this.abort.signal;

    // touch is handled once, by the shared recognizer (see gestures.ts) — the
    // pointer handlers below early-return on pointerType "touch" so the two
    // paths never fight over the same drag.
    this.gestures = new GestureRecognizer({
      onPan: (dx, dy) => {
        this.theta -= dx * 0.006;
        this.phi = Math.max(-1.5, Math.min(1.5, this.phi + dy * 0.006));
      },
      onPinch: (e) => {
        this.radius = clamp(this.radius / e.scale, 8, 180);
        this.theta -= e.dcx * 0.006;
        this.phi = Math.max(-1.5, Math.min(1.5, this.phi + e.dcy * 0.006));
      },
    });
    this.gestures.attach(this.canvas, sig);

    this.canvas.addEventListener(
      "pointerdown",
      (e) => {
        if (e.pointerType === "touch") return;
        dragging = true;
        lx = e.clientX;
        ly = e.clientY;
        this.canvas.setPointerCapture(e.pointerId);
      },
      { signal: sig },
    );
    this.canvas.addEventListener(
      "pointerup",
      (e) => {
        if (e.pointerType === "touch") return;
        dragging = false;
      },
      { signal: sig },
    );
    this.canvas.addEventListener(
      "pointermove",
      (e) => {
        if (e.pointerType === "touch") return;
        if (dragging) {
          this.theta -= (e.clientX - lx) * 0.006;
          this.phi = Math.max(-1.5, Math.min(1.5, this.phi + (e.clientY - ly) * 0.006));
          lx = e.clientX;
          ly = e.clientY;
        }
        const rect = this.canvas.getBoundingClientRect();
        this.updateTooltip(e.clientX - rect.left, e.clientY - rect.top);
      },
      { signal: sig },
    );
    this.canvas.addEventListener("pointerleave", () => this.hideTooltip(), { signal: sig });
    this.canvas.addEventListener(
      "wheel",
      (e) => {
        this.radius = Math.max(8, Math.min(180, this.radius * Math.exp(e.deltaY * 0.0012)));
      },
      { passive: true, signal: sig },
    );
  }

  /** The world position a point currently occupies — the same weighted sum the
   *  vertex shader computes, so hover can never drift from what is drawn. */
  private worldOf(i: number): THREE.Vector3 | null {
    // Reads the PACKED attribute arrays, not data.points[].positions — those
    // are the raw export, and compareAttributes normalises each state before it
    // reaches the GPU. Blending the raw ones here would put the hover target
    // somewhere the concept is not drawn, which is the one thing this method
    // exists to prevent.
    const at = this.attrs;
    if (!at) return null;
    const w = this.uWeights.value;
    const ws = [w.x, w.y, w.z, w.w];
    const bufs = [at.a, at.b, at.c, at.d];
    let x = 0;
    let y = 0;
    let z = 0;
    for (let s = 0; s < 4; s++) {
      const k = ws[s]!;
      if (k === 0) continue;
      const buf = bufs[s]!;
      x += buf[i * 4]! * k;
      y += buf[i * 4 + 1]! * k;
      z += buf[i * 4 + 2]! * k;
    }
    return new THREE.Vector3(x, y, z);
  }

  /** world → CSS px through the same camera the GPU rendered with. */
  private project(v: THREE.Vector3): [number, number] | null {
    const p = v.clone().project(this.camera);
    if (p.z > 1) return null; // behind the near plane
    return [(p.x * 0.5 + 0.5) * this.cssW, (1 - (p.y * 0.5 + 0.5)) * this.cssH];
  }

  private initTooltip(): void {
    const overlay = document.getElementById("overlay-html");
    if (!overlay) return;
    this.tooltip = document.createElement("div");
    this.tooltip.className = "point-tooltip compare-tooltip";
    this.tooltip.style.visibility = "hidden";
    overlay.appendChild(this.tooltip);
  }

  private updateTooltip(mx: number, my: number): void {
    const data = this.data;
    if (!data || !this.tooltip) return;
    let best = -1;
    let bd = 14;
    for (let i = 0; i < this.count; i++) {
      const pt = data.points[i]!;
      if (this.hidden.includes(pt.source_idx)) continue;
      if (this.sharedOnly && !pt.shared) continue;
      const w = this.worldOf(i);
      if (!w) continue;
      const sc = this.project(w);
      if (!sc) continue;
      const d = Math.hypot(sc[0] - mx, sc[1] - my);
      if (d < bd) {
        bd = d;
        best = i;
      }
    }
    if (best < 0) {
      this.hideTooltip();
      return;
    }
    const pt = data.points[best]!;
    this.tooltip.textContent = "";
    const t = document.createElement("div");
    t.className = "point-tooltip-label";
    t.textContent = pt.title;
    const m = document.createElement("div");
    m.className = "point-tooltip-cluster";
    m.textContent = `${pt.source} · ${pt.shared ? "shared concept" : "unique"} · ${pt.size} tokens`;
    this.tooltip.append(t, m);
    this.tooltip.style.visibility = "visible";
    const w = this.tooltip.offsetWidth;
    const h = this.tooltip.offsetHeight;
    const x = Math.min(Math.max(mx + 14, 8), this.cssW - w - 8);
    const y = Math.min(Math.max(my + 14, 8), this.cssH - h - 8);
    this.tooltip.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
  }

  private hideTooltip(): void {
    if (this.tooltip) this.tooltip.style.visibility = "hidden";
  }

  /** Drop the field mesh. NEVER dispose the geometry: every THREE.Sprite shares
   *  one module-level quad, and destroying it blanks every sprite in the app. */
  private clearField(): void {
    if (this.field) {
      this.scene.remove(this.field);
      this.field = null;
    }
    this.fieldMat?.dispose();
    this.fieldMat = null;
    // NEVER dispose the geometry — every THREE.Sprite shares one module-level
    // quad, so disposing it here would blank every sprite in the app.
    this.attrs = null; // hover must not project positions that are gone
  }

  dispose(): void {
    this.disposed = true;
    this.abort.abort();
    this.gestures?.dispose();
    this.unsubscribe?.();
    this.tooltip?.remove();
    this.clearField();
    this.bloomPipe?.dispose();
    this.bloomPipe = null;
    this.renderer?.dispose();
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function clampIdx(i: number): number {
  return Math.max(0, Math.min(3, Math.round(i)));
}

/** The classic smoothstep ease the WGSL shader used, kept on the CPU now that
 *  the blend weights are computed there. */
function smoothstep01(t: number): number {
  const x = clamp(t, 0, 1);
  return x * x * (3 - 2 * x);
}
