/** #23 Attention-Rollout Waterfall — attention rollout (Abnar & Zuidema, 2020):
 *  account for how information mixes across depth by taking the cumulative matrix
 *  product of the (residual-augmented, head-averaged) attention maps. Real
 *  quantity, computed in-browser from trace_*.json → attn[layer][head][i][j]:
 *
 *    A_l   = mean_h attn[l][h]                 (head-averaged, still row-stochastic)
 *    Ã_l   = 0.5·A_l + 0.5·I, row-normalized   (0.5 weight to the residual stream)
 *    R_d   = Ã_d · Ã_{d-1} · … · Ã_0           (cumulative through layer d)
 *
 *  R_d[i][j] is how much source token j contributes to destination token i's
 *  representation after layers 0..d — a proper distribution (each row sums to 1)
 *  and strictly causal (j ≤ i). Scrub / play the depth to watch the "waterfall":
 *  local structure at shallow depth cascading onto the first-token attention SINK
 *  by full depth (a known, honest property of rollout — not a bug).
 *
 *  DRAWN AS EXTRUDED COLUMNS ON `ChartStage` (three/webgpu + TSL + bloom), one
 *  column per causal (source, destination) pair. Two things follow from that,
 *  and neither is decoration:
 *
 *  1. The causal half is EMPTY GROUND, not a flat sheet of zeros. The cage still
 *     spans the full T×T lattice, so the missing half reads as excluded rather
 *     than as measured-and-tiny.
 *  2. Height and colour carry the SAME log₁₀ normalization, so the two channels
 *     cannot disagree about one number. The vertical axis is therefore a log
 *     axis and is ticked as one — decade labels at 1e-4 … 1. A linear height
 *     would leave everything but the attention sink flat on the floor, which is
 *     precisely the structure this view exists to show.
 *
 *  No interpolated surface is drawn between columns: every pixel of every bar
 *  belongs to one measured pair. */

import type { GpuTier } from "../../app/capabilities";
import { loadTrace, type TraceBundle } from "../../data/interp";
import { ChartStage, type BarData, type ChartStageLook } from "./chart-stage";
import { causalCells, computeRollouts, decadeAt, logNorm, type RollCell } from "./rollout";
import { InterpTooltip } from "./chart-tooltip";
import type { InterpDriver } from "./InterpDriver";
import type { StatTile } from "../../chrome/StatStrip";

const LOG_FLOOR = 1e-4; // colour + height floor: values at or below map to 0
const STEP_MS = 640; // auto-play cadence (one real layer per step — no interpolation)

/** World layout. One lattice cell is 1 unit; the footprint leaves a gap so
 *  neighbouring columns read as separate measurements rather than a ridge. */
const CELL = 0.78;
/** Cage height, world units, as a fraction of the lattice span — clamped so a
 *  4-token trace is not a pillar and a 64-token one is not a pancake. */
const cageHeight = (T: number) => Math.min(12, Math.max(4, T * 0.3));

/** px gutters the fit must keep clear: token labels left and front, the depth
 *  transport and axis caption below. */
const INSET = { left: 84, top: 34, right: 44, bottom: 96 };

/** Minimum projected spacing between two adjacent token labels before the axis
 *  starts skipping them. Measured on the actual projection, so the stride
 *  adapts to the orbit instead of assuming one. */
const LABEL_MIN_PX = 11;

/** perceptual dark → gold ramp, evaluated on the log-normalized weight. The
 *  lowest step is deliberately lifted off the page background so faint (but
 *  real) columns stay visible as solids rather than blending into the void.
 *
 *  The top stop is GOLD, not the near-white the flat version used. At full
 *  depth the whole j=0 wall sits at t ≈ 0.95, so the top stop is not a rare
 *  peak — it is most of what you look at. Cream plus bloom drove that entire
 *  wall to flat white: the hue that IS the encoding, gone, exactly the failure
 *  the compare field hit. Gold keeps its identity and lets the bloom halo be
 *  the thing that says "hot". */
const RAMP: Array<[number, [number, number, number]]> = [
  [0.0, [30, 38, 66]],
  [0.35, [50, 66, 120]],
  [0.6, [70, 150, 214]],
  [0.82, [232, 160, 60]],
  [1.0, [250, 208, 112]],
];

const LOOK: ChartStageLook = {
  // bright enough to be seen through the bloom haze the sink wall throws: the
  // acausal half of the lattice has to read as EXCLUDED, and it can only do
  // that if its floor grid is actually visible
  frameColor: 0x3d4560,
  frameOpacity: 0.75,
  // just past 1.0: the headroom is what the bloom threshold keys on, and the
  // threshold is raised to match, so only the sink and its neighbours glow
  emissiveMin: 0.62,
  emissiveMax: 1.2,
  dimLevel: 0.22,
  azimuth: -0.72,
  elevation: 0.62,
  fov: 38,
  // Tight radius on purpose. At full depth the ENTIRE j=0 wall clears the
  // threshold, and a wide blur over an area that large stops reading as a glow
  // on the wall and becomes a wash over the whole stage — which then hides the
  // cage the acausal half needs to be legible.
  bloom: { strength: 0.45, radius: 0.25, threshold: 0.88 },
};

export class AttentionRolloutDriver implements InterpDriver {
  readonly animated = false; // static per depth; the play timer re-renders on step
  private stage = new ChartStage(LOOK);
  private canvas!: HTMLCanvasElement;
  private tooltip!: InterpTooltip;
  private labelRoot!: HTMLElement;
  private ctrlRoot!: HTMLElement;

  private bundle: TraceBundle | null = null;
  private T = 0;
  private nLayer = 0;
  private rollouts: Float64Array[][] = []; // rollouts[d] = R_d (T rows of length T)
  private cells: RollCell[] = [];
  private depth = 0; // current cumulative depth d (0..nLayer-1)
  private selRow = -1; // isolated destination row, or -1 for the whole matrix
  private hover: RollCell | null = null;
  private playing = false;
  private timer = 0;
  // ChartStage's onClick callback is (sx, sy) only — it does not forward the
  // originating PointerEvent — so this is captured on the raw pointerdown to
  // tell a touch tap from a mouse click by the time onClick fires
  private lastPointerType = "mouse";

  // reused per-instance buffers — a depth step rewrites every value but never
  // the cell set, so these are allocated once per bundle
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
    this.labelRoot.className = "interp-roll-labels";
    overlay.appendChild(this.labelRoot);
    this.ctrlRoot = document.createElement("div");
    this.ctrlRoot.className = "transport interp-transport"; // buildControls fills it
    overlay.appendChild(this.ctrlRoot);

    const onMove = (e: PointerEvent) => this.onPointerMove(e);
    const onLeave = () => this.hideTip();
    // read-only: just remembers pointerType for the stage's own pointerdown/up
    // click-pair detection (chart-stage.ts, not owned here) to consult later
    const onDown = (e: PointerEvent) => {
      this.lastPointerType = e.pointerType;
    };
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    canvas.addEventListener("pointerdown", onDown);
    this.disposers.push(() => {
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("pointerdown", onDown);
    });
  }

  async setModel(model: string, trace?: string): Promise<void> {
    if (!trace) throw new Error("no forward trace selected");
    const b = await loadTrace(model, trace);
    this.bundle = b;
    this.T = b.meta.T;
    this.nLayer = b.meta.n_layer;
    this.rollouts = computeRollouts(b.attn, this.T, b.meta.n_head, this.nLayer);
    this.selRow = -1; // no row isolated by default — show the whole matrix
    this.hover = null;
    this.stopPlay();
    this.depth = 0; // start at the top of the cascade and auto-play down the stack

    // the causal lower triangle: rollout only ever flows from j ≤ i
    this.cells = causalCells(this.T);

    const n = this.cells.length;
    this.bufPos = new Float32Array(n * 2);
    this.bufH = new Float32Array(n);
    this.bufC = new Float32Array(n * 3);
    this.bufG = new Float32Array(n);
    this.bufA = new Float32Array(n);
    const half = this.T / 2;
    for (let k = 0; k < n; k++) {
      const c = this.cells[k]!;
      // x = source, z = destination. Row 0 sits at the FAR edge, so the causal
      // staircase descends toward the viewer in the same direction the sequence
      // reads — the 2-D version put row 0 at the top for the same reason.
      this.bufPos[k * 2] = c.j + 0.5 - half;
      this.bufPos[k * 2 + 1] = c.i + 0.5 - half;
    }

    this.stage.setLattice({
      halfX: half,
      halfZ: half,
      cageY: cageHeight(this.T),
      divX: this.T,
      divZ: this.T,
    });
    this.stage.fitInset(INSET);
    this.buildControls();
    this.pushBars();
    this.positionLabels();
    this.startPlay(); // the waterfall: R_0 → R_{L-1}, resting at the full rollout
  }

  private valOf(i: number, j: number): number {
    return this.rollouts[this.depth]?.[i]?.[j] ?? 0;
  }

  /** The one normalization in this view. Colour AND height both read it, so a
   *  tall column is a bright column by construction — they can never tell two
   *  different stories about the same weight. */
  private norm(v: number): number {
    return logNorm(v, LOG_FLOOR);
  }

  private pushBars(): void {
    const n = this.cells.length;
    if (n === 0) return;
    const cage = cageHeight(this.T);
    for (let k = 0; k < n; k++) {
      const c = this.cells[k]!;
      const t = this.norm(this.valOf(c.i, c.j));
      this.bufH[k] = t * cage;
      this.bufG[k] = t;
      const [r, g, b] = ramp(t);
      this.bufC[k * 3] = r / 255;
      this.bufC[k * 3 + 1] = g / 255;
      this.bufC[k * 3 + 2] = b / 255;
      const isSel = this.selRow < 0 || c.i === this.selRow;
      const isHover = this.hover !== null && this.hover.i === c.i && this.hover.j === c.j;
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
    const half = this.T / 2;
    const t = this.norm(this.valOf(h.i, h.j));
    this.stage.setProbe(h.j + 0.5 - half, t * cageHeight(this.T), h.i + 0.5 - half);
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

  /** Token axes on the two near cage edges, decade ticks up the height post,
   *  and three captions. Every anchor is a real world point run through the
   *  stage's projector, so a label cannot drift from the column it names. */
  private positionLabels(): void {
    const b = this.bundle;
    if (!b || !this.labelRoot) return;
    const T = this.T;
    const half = T / 2;
    const cage = cageHeight(T);

    // Stride is MEASURED, not assumed: project two adjacent anchors and see how
    // far apart they actually land at this orbit. A fixed stride would either
    // waste the axis when the camera is close or overlap when it is not.
    const stride = (a: [number, number] | null, c: [number, number] | null) => {
      if (!a || !c) return 1;
      const gap = Math.hypot(c[0] - a[0], c[1] - a[1]);
      return gap < 0.01 ? T : Math.max(1, Math.ceil(LABEL_MIN_PX / gap));
    };
    const srcStride = stride(
      this.stage.project(0.5 - half, 0, half),
      this.stage.project(1.5 - half, 0, half),
    );
    const dstStride = stride(
      this.stage.project(-half, 0, 0.5 - half),
      this.stage.project(-half, 0, 1.5 - half),
    );

    type Spec = { p: [number, number] | null; text: string; cls: string };
    const specs: Spec[] = [];
    for (let j = 0; j < T; j += srcStride) {
      specs.push({
        p: this.stage.project(j + 0.5 - half, 0, half + 0.5),
        text: fmtTok(b.token_strs[j] ?? ""),
        cls: "interp-stage-lab is-col",
      });
    }
    for (let i = 0; i < T; i += dstStride) {
      specs.push({
        p: this.stage.project(-half - 0.5, 0, i + 0.5 - half),
        text: fmtTok(b.token_strs[i] ?? ""),
        cls: `interp-stage-lab is-row${i === this.selRow ? " is-sel" : ""}`,
      });
    }
    // Decade ticks — the height axis is log₁₀, and saying so with the tick
    // VALUES is the only way the extrusion is readable as a measurement.
    //
    // They hang off a post the stage places: destination owns the whole
    // x = -half edge and source the whole z = +half edge, and which of the
    // remaining corners actually lands outside the silhouette depends on the
    // lattice aspect and the orbit — see `ChartStage.heightPost`.
    const post = this.stage.heightPost(-1, 1);
    if (post) {
      const cls = `interp-stage-lab ${post.side === "left" ? "is-h" : "is-hl"}`;
      for (let e = -4; e <= 0; e++) {
        specs.push({
          p: this.stage.project(post.x, decadeAt(e, LOG_FLOOR) * cage, post.z),
          text: e === 0 ? "1" : `1e${e}`,
          cls,
        });
      }
      specs.push({
        p: this.stage.project(post.x, cage + 0.9, post.z),
        text: "rollout ↑",
        cls: "interp-stage-cap",
      });
    }
    specs.push({
      p: this.stage.project(0, 0, half + 1.6),
      text: "source →",
      cls: "interp-stage-cap",
    });
    specs.push({
      p: this.stage.project(-half - 1.9, 0, 0),
      text: "destination →",
      cls: "interp-stage-cap",
    });

    this.ensureLabels(specs.length);
    for (let k = 0; k < this.labels.length; k++) {
      const el = this.labels[k]!;
      const s = specs[k];
      // Off the canvas, hide it. Every anchor here sits OUTSIDE the cage by
      // design, so at some orbits one slides past the edge — and an axis label
      // clipped by the stat strip or clamped back inside would be pointing at
      // something other than what it names.
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

  /** The depth transport. Built imperatively because that is this page's
   *  contract — InterpPage hands every driver an `overlay` element and each one
   *  owns the DOM inside it — but it uses the SHARED `.transport` / `.tp-*`
   *  classes, so it is the same control the sessions playhead and the compare
   *  layout tour use.
   *
   *  HONESTY: depth is an ORDINAL layer axis, not time. The readout names the
   *  layer it has accumulated through and counts layers; a seconds display
   *  would describe the animation, not the model. Layer numbers stay
   *  0-indexed, matching how the model names them (L0 … L{n-1}). */
  private buildControls(): void {
    this.ctrlRoot.textContent = "";
    this.ctrlRoot.className = "transport interp-transport";
    this.ctrlRoot.setAttribute("role", "group");
    this.ctrlRoot.setAttribute("aria-label", "Rollout depth");

    const play = document.createElement("button");
    play.type = "button";
    play.className = "tp-btn tp-play";
    play.addEventListener("click", () => this.togglePlay());

    const restart = document.createElement("button");
    restart.type = "button";
    restart.className = "tp-btn";
    restart.textContent = "↺";
    restart.title = "Restart the cascade from L0";
    restart.setAttribute("aria-label", "Restart the cascade from the first layer");
    restart.addEventListener("click", () => {
      this.stopPlay();
      this.setDepth(0);
      this.startPlay();
    });

    const slider = document.createElement("input");
    slider.type = "range";
    slider.className = "tp-scrub";
    slider.min = "0";
    slider.max = String(Math.max(0, this.nLayer - 1));
    slider.step = "1";
    slider.value = String(this.depth);
    slider.setAttribute("aria-label", "Cumulative depth");
    slider.addEventListener("input", () => {
      // read the dragged value FIRST: stopPlay → syncControls rewrites
      // slider.value back to the current depth, so reading after it would
      // discard the drag and snap to where playback happened to be.
      const want = Number(slider.value);
      this.stopPlay(); // scrubbing takes the wheel, exactly as in the other two
      this.setDepth(want);
    });

    const read = document.createElement("span");
    read.className = "tp-stage";
    read.setAttribute("aria-hidden", "true");
    const now = document.createElement("span");
    now.className = "tp-now";
    read.appendChild(now);

    const count = document.createElement("span");
    count.className = "tp-count";
    count.title = "layers accumulated, of the whole stack";

    this.ctrlRoot.append(play, restart, slider, read, count);
    this.ctrlEls = { play, slider, now, count };
    this.syncControls();
  }
  private ctrlEls: {
    play: HTMLButtonElement;
    slider: HTMLInputElement;
    now: HTMLElement;
    count: HTMLElement;
  } | null = null;

  /** Single writer for every control's text — the readout can only ever say
   *  what `depth`/`playing` actually are. */
  private syncControls(): void {
    const c = this.ctrlEls;
    if (!c) return;
    const last = Math.max(0, this.nLayer - 1);
    c.play.textContent = this.playing ? "⏸" : "▶";
    c.play.setAttribute("aria-label", this.playing ? "Pause the cascade" : "Play the cascade");
    c.slider.value = String(this.depth);
    c.slider.style.setProperty(
      "--tp-progress",
      `${last > 0 ? ((this.depth / last) * 100).toFixed(1) : 0}%`,
    );
    c.slider.setAttribute(
      "aria-valuetext",
      `cumulative through layer ${this.depth} of ${last}`,
    );
    c.now.textContent = `through L${this.depth}`;
    c.count.textContent = String(this.depth);
    const total = document.createElement("i");
    total.textContent = `/${last}`;
    c.count.appendChild(total);
  }

  private setDepth(d: number): void {
    this.depth = Math.max(0, Math.min(this.nLayer - 1, d | 0));
    this.syncControls();
    this.pushBars();
  }

  private togglePlay(): void {
    if (this.playing) this.stopPlay();
    else this.startPlay();
  }
  private startPlay(): void {
    if (this.playing || this.nLayer < 2) return;
    this.playing = true;
    this.syncControls();
    if (this.depth >= this.nLayer - 1) this.setDepth(0); // restart from the top
    this.timer = window.setInterval(() => {
      if (this.depth >= this.nLayer - 1) {
        this.stopPlay(); // stop at full depth (the complete rollout)
        return;
      }
      this.setDepth(this.depth + 1);
    }, STEP_MS);
  }
  private stopPlay(): void {
    this.playing = false;
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = 0;
    }
    this.syncControls();
  }

  private pick(sx: number, sy: number): RollCell | null {
    const idx = this.stage.pickAt(sx, sy);
    return idx < 0 ? null : (this.cells[idx] ?? null);
  }

  private onClick(sx: number, sy: number): void {
    const cell = this.pick(sx, sy);
    if (!cell) {
      // touch has no hover, so a tap in empty ground is the only way to
      // release a tap-pinned tooltip — and since row isolation is itself a
      // "pin one row" gesture, the same tap should release it too, closing
      // the gap where a miss previously did nothing for mouse or touch
      if (this.lastPointerType === "touch") {
        this.tooltip.pinned = false;
        this.tooltip.hide();
      }
      if (this.selRow !== -1) {
        this.selRow = -1;
        this.pushBars();
        this.positionLabels();
      }
      return;
    }
    this.selRow = this.selRow === cell.i ? -1 : cell.i; // toggle row isolation
    if (this.lastPointerType === "touch") {
      this.tooltip.pinned = true;
      this.showTooltipFor(cell, sx, sy);
    }
    this.pushBars();
    this.positionLabels();
  }

  private onPointerMove(e: PointerEvent): void {
    // a tap-pinned tooltip (touch has no hover) survives a stray move
    if (this.tooltip.pinned) return;
    if (!this.bundle) return;
    // mid-orbit the pointer is steering the camera, not reading the data — a
    // tooltip here would chase the drag across every column it swept
    if (this.stage.isDragging) {
      this.hideTip();
      return;
    }
    const rect = this.canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const cell = this.pick(sx, sy);
    const same = cell && this.hover && cell.i === this.hover.i && cell.j === this.hover.j;
    if (!same) {
      this.hover = cell;
      this.pushBars();
    }
    if (!cell) {
      this.hideTip();
      return;
    }
    this.showTooltipFor(cell, sx, sy);
    this.canvas.style.cursor = "crosshair";
  }

  /** Row-building + placement, shared by the hover path and a touch tap-to-pin. */
  private showTooltipFor(cell: RollCell, sx: number, sy: number): void {
    const b = this.bundle;
    if (!b) return;
    const v = this.valOf(cell.i, cell.j);
    const dst = fmtTok(b.token_strs[cell.i] ?? "");
    const src = fmtTok(b.token_strs[cell.j] ?? "");
    // swatch = the exact log-normalized ramp colour this column was drawn with
    const [sr, sg, sb] = ramp(this.norm(v));
    this.tooltip.show([
      { kind: "label", text: `dst “${dst}” ← src “${src}”`, swatch: [sr, sg, sb] },
      { text: "rollout", value: v.toFixed(4), hot: true },
      { text: "through", value: `L${this.depth}` },
      { text: "pos", value: `${cell.j}→${cell.i}` },
    ]);
    this.tooltip.move(sx, sy, this.cssW, this.cssH);
  }

  private hideTip(): void {
    // a tap-pinned tooltip must survive the pointer leaving/dragging — touch
    // has no hover state to interrupt in the first place
    if (this.tooltip.pinned) return;
    this.tooltip?.hide();
    this.canvas.style.cursor = "";
    if (this.hover) {
      this.hover = null;
      this.pushBars();
    }
  }

  /** Footer strip. Every tile is depth-INDEPENDENT: the host reads this once
   *  when the bundle lands, so a value keyed to the scrub position would go
   *  stale the moment the user moved it. The peak is taken at full depth over
   *  exactly the cells that get drawn (lower triangle), not over the padded
   *  matrix — the strip and the canvas must agree on what "max" means.
   *
   *  It also EXCLUDES the diagonal, which is not a detail. R[0][0] is 1.0000 in
   *  every rollout of every prompt — the first token has nothing else to attend
   *  to — so a plain max prints a structural constant dressed as a measurement,
   *  identically for every model the viewer can load. The off-diagonal max is
   *  the one that actually varies with the prompt. */
  stats(): StatTile[] {
    if (!this.bundle) return [];
    const full = this.rollouts[this.nLayer - 1];
    let peak = 0;
    let at = "";
    if (full) {
      for (const c of this.cells) {
        if (c.i === c.j) continue;
        const v = full[c.i]?.[c.j] ?? 0;
        if (v > peak) {
          peak = v;
          at = `${c.j}→${c.i}`;
        }
      }
    }
    return [
      { label: "tokens", value: String(this.T), title: "sequence length T" },
      { label: "layers", value: String(this.nLayer) },
      { label: "heads", value: String(this.bundle.meta.n_head) },
      {
        label: "columns",
        value: this.cells.length.toLocaleString("en-US"),
        title: "lower-triangular (source → destination) pairs extruded",
      },
      {
        label: "peak off-diag",
        value: full ? peak.toFixed(4) : "—",
        title: at
          ? `largest source→destination rollout at full depth (L${this.nLayer}), at pos ${at}. ` +
            "Self-attention is excluded: R[0][0] is 1.0 by construction."
          : undefined,
      },
    ];
  }

  frame(_dt: number, _t: number): void {
    // depth is stepped by the play timer and the camera by the stage's own
    // pointer handlers — both redraw on demand, so there is no RAF work here
  }

  resize(width: number, height: number, dpr: number): void {
    this.cssW = width;
    this.cssH = height;
    this.stage.resize(width, height, dpr);
    this.positionLabels();
    this.stage.render();
  }

  dispose(): void {
    this.stopPlay();
    for (const d of this.disposers) d();
    this.disposers = [];
    this.tooltip?.dispose();
    this.labelRoot?.remove();
    this.ctrlRoot?.remove();
    this.labels = [];
    this.stage.dispose();
  }
}

function ramp(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t));
  for (let s = 1; s < RAMP.length; s++) {
    const [t1, c1] = RAMP[s]!;
    if (x <= t1) {
      const [t0, c0] = RAMP[s - 1]!;
      const f = t1 === t0 ? 0 : (x - t0) / (t1 - t0);
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * f),
        Math.round(c0[1] + (c1[1] - c0[1]) * f),
        Math.round(c0[2] + (c1[2] - c0[2]) * f),
      ];
    }
  }
  return RAMP[RAMP.length - 1]![1];
}

function fmtTok(s: string): string {
  return s.replace(/^ /, "␣").replace(/\n/g, "⏎") || "∅";
}
