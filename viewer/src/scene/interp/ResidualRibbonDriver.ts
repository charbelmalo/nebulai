/** #8 Residual-Stream Ribbon — the L2 norm of each token's residual stream as it
 *  flows through the network, layer by layer. Real quantity, straight from
 *  trace_*.json → resid_norm[layer][pos]: the Euclidean norm ‖x_ℓ(t)‖₂ of the
 *  residual-stream vector for token t at the output of layer ℓ (ℓ=0 is the
 *  token+position embedding, ℓ=1..n_layer is after each transformer block).
 *
 *  DRAWN AS EXTRUDED COLUMNS ON `ChartStage` (three/webgpu + TSL + bloom), one
 *  column per (layer, token) pair. Nothing about that is decoration: the data
 *  IS a rectangular grid of scalars, and the flat version had to overlay every
 *  token's polyline in one plane. Eleven trajectories through the same box
 *  cross constantly, so the chart could only be read by hovering one at a time
 *  and dimming the other ten. Given the grid its own second axis, every
 *  measurement has somewhere to stand and the whole surface is legible at once.
 *
 *  Three encoding rules, all inherited from #23:
 *
 *  1. Height is a LOG₁₀ axis with an absolute base at ‖x‖₂ = 1, ticked in
 *     decades. The norm grows roughly geometrically with depth (≈10 → ≈3000
 *     over 12 layers on GPT-2) and one token usually balloons into a "massive
 *     activation" that dwarfs the rest; on a linear axis every other token lies
 *     flat on the floor. The base is a real value (log₁₀ 1 = 0), not the
 *     smallest observed norm — a bar chart whose baseline is the data minimum
 *     exaggerates every difference on it.
 *  2. HUE is the token's position in the sequence and NOTHING else; height and
 *     glow both carry the norm. Two channels never describe one quantity
 *     differently, and the one channel that describes a different quantity says
 *     so on its own axis.
 *  3. No interpolated surface between columns. The 2-D version could fill the
 *     area under a curve because the curve was a path through measured points;
 *     a sheet stretched over this grid would put shaded pixels at (layer,
 *     token) pairs that were never sampled. */

import type { GpuTier } from "@psychix/viz/capabilities";
import { loadTrace, type TraceBundle } from "../../data/interp";
import { ChartStage, type BarData, type ChartStageLook } from "./chart-stage";
import { decadeOn, logSpan } from "@psychix/viz/logscale";
import { InterpTooltip } from "@psychix/viz/chart-tooltip";
import type { RGB } from "@psychix/viz/chart-theme";
import type { InterpDriver } from "./InterpDriver";
import type { StatTile } from "@psychix/viz/StatStrip";

/** World layout: one lattice cell is 1 unit, the footprint leaves a gap so
 *  neighbouring columns read as separate measurements rather than a ridge. */
const CELL = 0.74;
/** Cage height, world units, scaled off the larger lattice axis and clamped so
 *  a 4-token trace is not a pillar and a 64-token one is not a pancake. */
const cageHeight = (n: number) => Math.min(13, Math.max(5, n * 0.4));

/** px gutters the fit must keep clear: token labels left, layer ticks and the
 *  axis caption below. No transport strip — every layer is on screen at once,
 *  which is the whole point of giving depth its own axis.
 *
 *  The right gutter is wide because the key card docks there and this lattice
 *  is wider than it is deep, which pushes the height axis out to the FRONT-right
 *  corner rather than the back one (see `ChartStage.heightPost`). It costs less
 *  than it looks: the fit on this aspect is bound by height, not width, so the
 *  gutter mostly eats margin the chart was not using. */
const INSET = { left: 72, top: 34, right: 316, bottom: 66 };

/** Minimum projected spacing between two adjacent axis labels before the axis
 *  starts skipping them. Measured against the real projection, so the stride
 *  adapts to the orbit instead of assuming one. */
const LABEL_MIN_PX = 11;

/** token-position hue: a cool→warm ramp on t = pos/(T-1). Encodes sequence
 *  order — a real ordered attribute — and nothing else. Magnitude lives
 *  entirely on the height axis and its glow. */
const POS_RAMP: Array<[number, [number, number, number]]> = [
  [0.0, [70, 200, 235]], // cyan — earliest token
  [0.5, [245, 195, 59]], // gold — middle
  [1.0, [234, 79, 134]], // magenta — latest token
];

const LOOK: ChartStageLook = {
  frameColor: 0x3d4560,
  frameOpacity: 0.75,
  // The glow range here is NARROW and measured, not guessed: on a GPT-2 trace
  // the norms are 4.7 … 3112, which on the 1 … 10000 axis is glow 0.17 … 0.87
  // with the interquartile band only 0.45 … 0.55. Most (layer, token) pairs
  // genuinely do sit at the same magnitude — that IS the finding, and the
  // outlier is the story. A high emissive floor compresses that already-narrow
  // band into one pastel wash where nothing recedes; the floor is therefore low
  // and the ceiling high, so the ridge separates from the plateau.
  emissiveMin: 0.22,
  emissiveMax: 1.35,
  dimLevel: 0.2,
  azimuth: -0.72,
  elevation: 0.58,
  fov: 38,
  bloom: { strength: 0.45, radius: 0.3, threshold: 0.86 },
};

export class ResidualRibbonDriver implements InterpDriver {
  readonly animated = false; // static per trace — redraws on hover, click, orbit
  private stage = new ChartStage(LOOK);
  private canvas!: HTMLCanvasElement;
  private tooltip!: InterpTooltip;
  private labelRoot!: HTMLElement;

  private bundle: TraceBundle | null = null;
  private T = 0;
  private nLayer = 0; // blocks; resid_norm has nLayer + 1 rows (0..nLayer)
  private nStage = 0; // plotted stages along x = nLayer + 1
  private baseExp = 0; // log₁₀ decade at the cage floor
  private topExp = 1; // log₁₀ decade at the cage top
  private selTok = -1; // isolated token row, or -1 for the whole grid
  private hover: { layer: number; tok: number } | null = null;
  private tokRGB: Array<[number, number, number]> = [];
  /** `ChartStage.onClick` reports only a screen point, not the originating
   *  PointerEvent, so a touch tap can't be told apart from a mouse click there.
   *  A second, independent pointerdown listener (below) just records the type
   *  — it never calls preventDefault/setPointerCapture, so the stage's own
   *  click-vs-drag handling is untouched. */
  private lastPointerType = "mouse";

  // reused per-instance buffers — hover and selection rewrite appearance but
  // never the cell set, so these are allocated once per bundle
  private bufPos = new Float32Array(0);
  private bufH = new Float32Array(0);
  private bufC = new Float32Array(0);
  private bufG = new Float32Array(0);
  private bufA = new Float32Array(0);

  private cssW = 1;
  private cssH = 1;
  private disposers: Array<() => void> = [];
  private labels: HTMLElement[] = [];

  async init(canvas: HTMLCanvasElement, tier: GpuTier, overlay: HTMLElement): Promise<void> {
    this.canvas = canvas;
    await this.stage.init(canvas, tier);
    this.stage.onCamera = () => this.positionLabels();
    this.stage.onClick = (sx, sy) => this.onClick(sx, sy);

    this.tooltip = new InterpTooltip(overlay);
    this.labelRoot = document.createElement("div");
    this.labelRoot.className = "interp-rs-labels";
    overlay.appendChild(this.labelRoot);

    const onMove = (e: PointerEvent) => this.onPointerMove(e);
    const onLeave = () => this.hideTip();
    const onPointerDown = (e: PointerEvent) => {
      this.lastPointerType = e.pointerType;
    };
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    canvas.addEventListener("pointerdown", onPointerDown);
    this.disposers.push(() => {
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("pointerdown", onPointerDown);
    });
  }

  async setModel(model: string, trace?: string): Promise<void> {
    if (!trace) throw new Error("no forward trace selected");
    const b = await loadTrace(model, trace);
    this.bundle = b;
    this.T = b.meta.T;
    this.nLayer = b.meta.n_layer;
    this.nStage = b.resid_norm.length; // nLayer + 1 in every bundle we ship
    this.selTok = -1;
    this.hover = null;

    // Axis bounds snap OUTWARD to whole decades. The ticks are then at real
    // round numbers (‖x‖ = 10, 100, 1000) instead of at whatever the extremes
    // of this particular prompt happened to be, and the same axis serves every
    // prompt in the selector — so switching prompts moves the columns, not the
    // ruler they are measured against.
    let hi = -Infinity;
    for (const row of b.resid_norm) for (const v of row) if (v > 0 && Math.log10(v) > hi) hi = Math.log10(v);
    this.baseExp = 0; // ‖x‖₂ = 1. A real reference, not the data minimum.
    this.topExp = Number.isFinite(hi) ? Math.max(1, Math.ceil(hi)) : 1;

    this.tokRGB = [];
    for (let t = 0; t < this.T; t++) {
      this.tokRGB.push(ramp(POS_RAMP, this.T > 1 ? t / (this.T - 1) : 0));
    }

    const n = this.nStage * this.T;
    this.bufPos = new Float32Array(n * 2);
    this.bufH = new Float32Array(n);
    this.bufC = new Float32Array(n * 3);
    this.bufG = new Float32Array(n);
    this.bufA = new Float32Array(n);
    const halfX = this.nStage / 2;
    const halfZ = this.T / 2;
    for (let k = 0; k < n; k++) {
      const l = (k / this.T) | 0;
      const t = k % this.T;
      // x = depth, increasing away from the embedding — the same left-to-right
      // reading the flat version had. z = token, position 0 at the far edge so
      // the sequence runs toward the viewer.
      this.bufPos[k * 2] = l + 0.5 - halfX;
      this.bufPos[k * 2 + 1] = t + 0.5 - halfZ;
    }

    this.stage.setLattice({
      halfX,
      halfZ,
      cageY: cageHeight(Math.max(this.nStage, this.T)),
      divX: this.nStage,
      divZ: this.T,
    });
    this.stage.fitInset(INSET);
    this.pushBars();
    this.positionLabels();
  }

  private normAt(layer: number, tok: number): number {
    return this.bundle?.resid_norm[layer]?.[tok] ?? 0;
  }

  /** The one normalization in this view. Height and glow both read it, so a
   *  tall column is a bright column by construction. */
  private norm(v: number): number {
    return logSpan(v, this.baseExp, this.topExp);
  }

  private pushBars(): void {
    const n = this.nStage * this.T;
    if (n === 0) return;
    const cage = cageHeight(Math.max(this.nStage, this.T));
    for (let k = 0; k < n; k++) {
      const l = (k / this.T) | 0;
      const t = k % this.T;
      const y = this.norm(this.normAt(l, t));
      this.bufH[k] = y * cage;
      this.bufG[k] = y;
      const [r, g, b] = this.tokRGB[t] ?? [255, 255, 255];
      this.bufC[k * 3] = r / 255;
      this.bufC[k * 3 + 1] = g / 255;
      this.bufC[k * 3 + 2] = b / 255;
      const isSel = this.selTok < 0 || t === this.selTok;
      const isHover = this.hover !== null && this.hover.layer === l && this.hover.tok === t;
      this.bufA[k] = isSel || isHover ? 1 : 0;
    }
    const d: BarData = {
      count: n,
      pos: this.bufPos,
      height: this.bufH,
      color: this.bufC,
      glow: this.bufG,
      active: this.bufA,
      cellX: CELL,
      cellZ: CELL,
    };
    this.stage.setBars(d);
    this.syncProbe();
    this.stage.render();
  }

  /** Rails from the hovered column's top down to the floor and out to both
   *  axis walls — how a height is read off the cage once the camera can turn. */
  private syncProbe(): void {
    const h = this.hover;
    if (!h) {
      this.stage.setProbe(null);
      return;
    }
    const cage = cageHeight(Math.max(this.nStage, this.T));
    this.stage.setProbe(
      h.layer + 0.5 - this.nStage / 2,
      this.norm(this.normAt(h.layer, h.tok)) * cage,
      h.tok + 0.5 - this.T / 2,
    );
  }

  // ---- labels ---------------------------------------------------------------

  private ensureLabels(n: number): void {
    while (this.labels.length < n) {
      const el = document.createElement("div");
      el.className = "interp-stage-lab";
      this.labelRoot.appendChild(el);
      this.labels.push(el);
    }
  }

  /** Layer ticks on the near cage edge, token names down the left edge, decade
   *  ticks up the one free corner post, and three captions. Every anchor is a
   *  real world point run through the stage's projector, so a label cannot
   *  drift from the column it names. */
  private positionLabels(): void {
    const b = this.bundle;
    if (!b || !this.labelRoot) return;
    const halfX = this.nStage / 2;
    const halfZ = this.T / 2;
    const cage = cageHeight(Math.max(this.nStage, this.T));

    const stride = (a: [number, number] | null, c: [number, number] | null, n: number) => {
      if (!a || !c) return 1;
      const gap = Math.hypot(c[0] - a[0], c[1] - a[1]);
      return gap < 0.01 ? n : Math.max(1, Math.ceil(LABEL_MIN_PX / gap));
    };
    const layStride = stride(
      this.stage.project(0.5 - halfX, 0, halfZ),
      this.stage.project(1.5 - halfX, 0, halfZ),
      this.nStage,
    );
    const tokStride = stride(
      this.stage.project(-halfX, 0, 0.5 - halfZ),
      this.stage.project(-halfX, 0, 1.5 - halfZ),
      this.T,
    );

    type Spec = { p: [number, number] | null; text: string; cls: string };
    const specs: Spec[] = [];
    for (let l = 0; l < this.nStage; l += layStride) {
      specs.push({
        p: this.stage.project(l + 0.5 - halfX, 0, halfZ + 0.5),
        // "emb" is not layer zero of the stack, it is the token+position
        // embedding the stack starts from. Numbering it 0 alongside the blocks
        // would claim the model has a block there.
        text: l === 0 ? "emb" : String(l),
        cls: "interp-stage-lab is-col",
      });
    }
    for (let t = 0; t < this.T; t += tokStride) {
      specs.push({
        p: this.stage.project(-halfX - 0.5, 0, t + 0.5 - halfZ),
        text: fmtTok(b.token_strs[t] ?? ""),
        cls: `interp-stage-lab is-row${t === this.selTok ? " is-sel" : ""}`,
      });
    }
    // Decade ticks up a post the stage places for us: the token axis owns the
    // whole x = -halfX edge and the layer axis the whole z = +halfZ edge, and
    // on this 13×11 lattice the corner they leave free projects into the middle
    // of the chart — see `ChartStage.heightPost`.
    const post = this.stage.heightPost(-1, 1);
    if (post) {
      const cls = `interp-stage-lab ${post.side === "left" ? "is-h" : "is-hl"}`;
      for (let e = this.baseExp; e <= this.topExp; e++) {
        specs.push({
          p: this.stage.project(post.x, decadeOn(e, this.baseExp, this.topExp) * cage, post.z),
          // exponent form past 100, matching #23's ticks — and, more to the
          // point, "10,000" is six glyphs hanging off the outermost corner of
          // the widest chart on the page, where the gutter it needs comes
          // straight out of the plot
          text: e <= 2 ? String(10 ** e) : `1e${e}`,
          cls,
        });
      }
      specs.push({
        p: this.stage.project(post.x, cage + 0.9, post.z),
        text: "‖x‖₂ ↑",
        cls: "interp-stage-cap",
      });
    }
    specs.push({
      p: this.stage.project(0, 0, halfZ + 1.7),
      text: `layer: embedding → block ${this.nLayer} →`,
      cls: "interp-stage-cap",
    });
    specs.push({
      p: this.stage.project(-halfX - 2.1, 0, 0),
      text: "token →",
      cls: "interp-stage-cap",
    });

    this.ensureLabels(specs.length);
    for (let k = 0; k < this.labels.length; k++) {
      const el = this.labels[k]!;
      const s = specs[k];
      // Off the canvas, hide it. Every anchor sits OUTSIDE the cage by design,
      // so at some orbits one slides past the edge — and an axis label clamped
      // back inside would point at something other than what it names.
      if (!s || !s.p || s.p[0] < 2 || s.p[1] < 2 || s.p[0] > this.cssW - 2 || s.p[1] > this.cssH - 2) {
        el.style.display = "none";
        continue;
      }
      el.className = s.cls;
      el.textContent = s.text;
      el.style.display = "";
      // `translate`, never `transform` — the anchoring offset lives in CSS
      // `transform`, and writing both here would clobber it
      el.style.translate = `${Math.round(s.p[0])}px ${Math.round(s.p[1])}px`;
    }
  }

  // ---- interaction ----------------------------------------------------------

  private pick(sx: number, sy: number): { layer: number; tok: number } | null {
    const idx = this.stage.pickAt(sx, sy);
    if (idx < 0 || this.T === 0) return null;
    return { layer: (idx / this.T) | 0, tok: idx % this.T };
  }

  private onClick(sx: number, sy: number): void {
    const c = this.pick(sx, sy);
    if (!c) {
      // tap/click on empty space used to bare-return, leaving a stale token
      // isolation stuck — clear it, same as clicking the isolated column again would
      if (this.selTok >= 0) {
        this.selTok = -1;
        this.pushBars();
        this.positionLabels();
      }
      this.tooltip.pinned = false;
      this.hideTip();
      return;
    }
    const wasSel = this.selTok === c.tok;
    this.selTok = wasSel ? -1 : c.tok; // toggle token isolation
    this.pushBars();
    this.positionLabels();
    // touch has no hover to read the isolated column from — pin the readout open
    if (this.lastPointerType === "touch" && !wasSel) {
      this.hover = c;
      this.pushBars();
      this.tooltip.pinned = true;
      this.showTooltipFor(c, sx, sy);
    }
  }

  private onPointerMove(e: PointerEvent): void {
    if (this.tooltip.pinned) return;
    if (!this.bundle) return;
    // mid-orbit the pointer is steering the camera, not reading the data
    if (this.stage.isDragging) {
      this.hideTip();
      return;
    }
    const rect = this.canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const c = this.pick(sx, sy);
    const same = c && this.hover && c.layer === this.hover.layer && c.tok === this.hover.tok;
    if (!same) {
      this.hover = c;
      this.pushBars();
    }
    if (!c) {
      this.hideTip();
      return;
    }
    this.showTooltipFor(c, sx, sy);
  }

  private showTooltipFor(c: { layer: number; tok: number }, sx: number, sy: number): void {
    const b = this.bundle;
    if (!b) return;
    const v = this.normAt(c.layer, c.tok);
    const embed = this.normAt(0, c.tok);
    const final = this.normAt(this.nStage - 1, c.tok);
    let peak = 0;
    let peakL = 0;
    for (let l = 0; l < this.nStage; l++) {
      const q = this.normAt(l, c.tok);
      if (q > peak) {
        peak = q;
        peakL = l;
      }
    }
    const growth = embed > 0 ? final / embed : 0;
    const rgb = this.tokRGB[c.tok] ?? [255, 255, 255];
    const swatch: RGB = [rgb[0], rgb[1], rgb[2]];
    this.tooltip.show([
      {
        kind: "label",
        text: `“${fmtTok(b.token_strs[c.tok] ?? "")}” · ${c.layer === 0 ? "embedding" : `after block ${c.layer}`}`,
        swatch,
      },
      { text: "‖x‖₂", value: v.toFixed(2), hot: true },
      {
        text: `embed ${embed.toFixed(1)} → final ${final.toFixed(1)} (×${growth.toFixed(1)}) · peak ${peak.toFixed(0)} @ ${peakL === 0 ? "emb" : `L${peakL}`}`,
      },
    ]);
    this.tooltip.move(sx, sy, this.cssW, this.cssH);
    this.canvas.style.cursor = "crosshair";
  }

  private hideTip(): void {
    // pointerleave fires the instant a touch finger lifts — must not clobber
    // a tap-pinned readout (the whole reason a pin exists)
    if (this.tooltip?.pinned) return;
    this.tooltip?.hide();
    this.canvas.style.cursor = "";
    if (this.hover) {
      this.hover = null;
      this.pushBars();
    }
  }

  /** Footer strip. Growth and peak are the per-token quantities already spelled
   *  out in the hover line; the strip reports their maxima over tokens. Layer
   *  count is stated as blocks, matching the axis caption — the plot has
   *  nLayer+1 stages because the embedding is one of them, and calling that
   *  "13 layers" would misdescribe the model. The span tile names the axis that
   *  is actually drawn, not the extremes of the data, because the two are
   *  deliberately different: the cage snaps outward to whole decades. */
  stats(): StatTile[] {
    if (!this.bundle || this.T === 0) return [];
    let maxGrowth = 0;
    let peak = 0;
    for (let t = 0; t < this.T; t++) {
      const embed = this.normAt(0, t);
      const final = this.normAt(this.nStage - 1, t);
      if (embed > 0) maxGrowth = Math.max(maxGrowth, final / embed);
      for (let l = 0; l < this.nStage; l++) peak = Math.max(peak, this.normAt(l, t));
    }
    return [
      { label: "tokens", value: String(this.T) },
      {
        label: "blocks",
        value: String(this.nLayer),
        title: `${this.nStage} plotted stages — embedding, then ${this.nLayer} blocks`,
      },
      {
        label: "columns",
        value: (this.nStage * this.T).toLocaleString("en-US"),
        title: "one extruded column per (layer, token) measurement",
      },
      {
        label: "max growth",
        value: `×${maxGrowth.toFixed(1)}`,
        title: "largest final ÷ embedding norm ratio over tokens",
      },
      {
        label: "peak ‖x‖₂",
        value: peak.toLocaleString("en-US", { maximumFractionDigits: 0 }),
        title: `height axis spans ‖x‖ ${(10 ** this.baseExp).toLocaleString("en-US")} … ${(10 ** this.topExp).toLocaleString("en-US")} (${this.topExp - this.baseExp} decades)`,
      },
    ];
  }

  frame(_dt: number, _t: number): void {
    // static per trace — the stage renders on demand, no data-bearing motion
  }

  resize(width: number, height: number, dpr: number): void {
    this.cssW = width;
    this.cssH = height;
    this.stage.resize(width, height, dpr);
    this.positionLabels();
    // `animated` is false, so the host gives this driver no rAF: a resize that
    // does not redraw leaves the last frame stretched, or — on the first one,
    // which arrives after setModel — leaves the stage empty forever.
    this.stage.render();
  }

  dispose(): void {
    for (const d of this.disposers) d();
    this.disposers = [];
    this.tooltip?.dispose();
    this.labelRoot?.remove();
    this.labels = [];
    this.stage.dispose();
  }
}

function ramp(stops: Array<[number, [number, number, number]]>, t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t));
  for (let s = 1; s < stops.length; s++) {
    const [t1, c1] = stops[s]!;
    if (x <= t1) {
      const [t0, c0] = stops[s - 1]!;
      const f = t1 === t0 ? 0 : (x - t0) / (t1 - t0);
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * f),
        Math.round(c0[1] + (c1[1] - c0[1]) * f),
        Math.round(c0[2] + (c1[2] - c0[2]) * f),
      ];
    }
  }
  return stops[stops.length - 1]![1];
}

function fmtTok(s: string): string {
  return s.replace(/^ /, "␣").replace(/\n/g, "⏎") || "∅";
}
