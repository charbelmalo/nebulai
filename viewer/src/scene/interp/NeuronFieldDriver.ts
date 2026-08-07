/** #6 Neuron Write-Direction Field — all 36,864 MLP neurons (12 layers × 3072),
 *  each placed at the exact PCA score of its WRITE DIRECTION: the row of
 *  mlp.c_proj that the neuron adds to the residual stream (scaled by its
 *  activation). Nothing is a synthetic layout: positions are real PC scores of
 *  the mean-centered write-direction matrix, dot size is the real ‖w_out‖₂, and
 *  color is the neuron's layer (viridis ramp).
 *
 *  Each neuron also carries its direct-path logit readout — the token its write
 *  direction most promotes and most suppresses through the model's own final-LN
 *  gain + tied unembedding: ℓ = ((w − mean(w)) ⊙ γ_f)·W_Eᵀ. That is the DIRECT
 *  path only (no downstream-layer effects, positive activation assumed) and the
 *  view says so. Honest findings this exposes: median write norm grows
 *  monotonically with depth (≈2.2 → 5.2 — late layers write hardest), and
 *  PC1+PC2 explain only ~3.3% of variance — a low-D shadow of a 768-D space.
 *
 *  RENDERING: three/webgpu + TSL emissive field (`field2d.ts`), not a deck.gl
 *  scatter. 36,864 hard-edged discs overplot into one flat silhouette, so the
 *  density structure — which is most of what a field this size has to say — was
 *  being discarded before it reached the screen. Additive soft sprites let
 *  overlap sum, so crowded regions read bright and sparse ones dark.
 *
 *  WHAT CHANGED IN THE ENCODING, stated plainly because it matters: brightness
 *  used to be a pure function of layer (viridis luminance rises with depth, so
 *  "brighter = deeper" was readable). It no longer is. Brightness now carries
 *  the RANK of ‖w_out‖₂, and layer is carried by HUE — which viridis varies
 *  independently of luminance (blue → teal → green → yellow), so the two
 *  channels stay separable and hover always gives the exact layer. Size is
 *  still linear in the real norm; rank is monotone in the same quantity, so the
 *  two norm channels can never disagree about which neuron writes harder.
 *  Rank rather than magnitude because the norm distribution is long-tailed.
 *
 *  Draw order no longer needs the deterministic shuffle the deck.gl version
 *  used to defeat layer-on-top-of-layer painting: addition is commutative, so
 *  an additive field has no painter's-order bias to correct.
 *
 *  Camera off, static (redraws on hover / layer isolate).
 *  Source: neurons.json — PCA computed offline in float64 (768×768 covariance
 *  eigendecomposition), readout in float32 through the tied W_E. */

import type { GpuTier } from "../../app/capabilities";
import { type NeuronsBundle, loadNeurons } from "../../data/interp";
import { EmissiveField2D, FieldMarker, rankNormalize, type Field2DLook } from "./field2d";
import { InterpTooltip, type TipRow } from "./chart-tooltip";
import type { InterpDriver } from "./InterpDriver";

const GL = 60; // px gutters (axis captions clear of the data)
const GR = 60;
const GT = 78;
const GB = 88; // extra room for the layer-isolate chip row
const FIT = 0.94;

/** Viridis sampled at t = 0.25 + 0.75·L/11 (low end clipped so layer 0 stays
 *  legible on the dark stage). HUE advances monotonically with layer — that is
 *  the channel the reader uses now that brightness carries the write norm (see
 *  the header); hover always gives the exact layer. */
export const LAYER_COLORS: [number, number, number][] = [
  [59, 82, 138],
  [51, 100, 141],
  [44, 117, 142],
  [37, 134, 141],
  [33, 150, 138],
  [34, 166, 133],
  [54, 181, 120],
  [83, 195, 104],
  [119, 208, 83],
  [165, 218, 53],
  [209, 226, 38],
  [253, 231, 37],
];

/** Tuned by measurement against the DENSEST state — all 12 layers on, where PC1
 *  and PC2 carry 2.4% and 0.9% of the variance, so 36864 neurons pile into one
 *  small blob at the origin and only a handful of outliers escape it.
 *
 *  That density is why this view needs a far WEAKER bloom than the embedding
 *  constellation next door (strength 0.14/radius 0.08 here vs 0.45/0.32 there),
 *  and why raising the threshold — the lever that fixed the embedding view —
 *  does almost nothing here: the additive core sums well past 4.0 in linear
 *  space, so it clears any threshold worth setting. Strength and radius are the
 *  only honest levers left, and the thing they have to protect is FILL, not
 *  extent: measured against a bloom-off frame, bloom barely moves the p95 radius
 *  of lit pixels (56 px vs 58 px — the outlier neurons genuinely reach that far)
 *  but it multiplies the count of lit pixels 12×, from 623 to 7470. In a field
 *  where brightness IS density, glow poured into the gap between the core and
 *  those outliers reads as neurons that are not there.
 *
 *  Measured on the all-layers state (fraction of lit pixels, max channel >0.12):
 *  before, lit 65.8% / mean L 0.273 / 9.5% blown — a fog with no structure.
 *  After, lit 12.2% / L 0.060 / 5.6% blown, and only 1.3% of lit pixels fall
 *  below 0.15 saturation, i.e. the viridis layer hue survives essentially
 *  everywhere except the true core. */
const LOOK: Field2DLook = {
  pointPx: 1.8,
  emissiveMin: 0.14,
  emissiveMax: 1.5,
  glowGamma: 2.2,
  moteFloor: 0.12,
  // isolate mode: the other 11 layers stay as faint context rather than
  // disappearing — a view that deletes its non-selected mass is lying about how
  // much of it there is. Far lower than it looks: 33792 dimmed points ADD, so
  // the background's brightness is its density and the level has to be set
  // against the sum, not against one point. At 0.03 the residue still out-summed
  // the 3072 isolated ones and the core stayed the all-layers yellow; at 0.012
  // the mean lit pixel measures (78,102,64) with L3 isolated — the layer's own
  // viridis green — while the context cloud is still plainly visible.
  dimLevel: 0.012,
  bloom: { strength: 0.14, radius: 0.08, threshold: 2.0 },
};

/** Global brightness while a neuron is focused. */
const HOVER_DIM = 0.42;

interface NeuronPt {
  x: number;
  y: number;
  z: number; // PC3 (hover only)
  norm: number; // exact ‖w_out‖₂
  layer: number;
  idx: number; // neuron index within its layer
  topTok: string;
  topVal: number;
  botTok: string;
  botVal: number;
  id: number; // global index (layer·d_mlp + idx)
}

export class NeuronFieldDriver implements InterpDriver {
  readonly animated = false;
  private field = new EmissiveField2D(LOOK);
  private canvas!: HTMLCanvasElement;
  private tooltip!: InterpTooltip;
  private marker!: FieldMarker;
  private labelRoot!: HTMLElement;
  private chipRoot!: HTMLElement;

  private bundle: NeuronsBundle | null = null;
  private pts: NeuronPt[] = [];
  private anchors: NeuronPt[] = [];
  private isolate: number | null = null; // layer to isolate, null = all
  private minX = 0;
  private maxX = 1;
  private minY = 0;
  private maxY = 1;
  private normMin = 0;
  private normMax = 1;
  private hover: NeuronPt | null = null;

  private cssW = 1;
  private cssH = 1;
  private disposers: Array<() => void> = [];

  async init(canvas: HTMLCanvasElement, tier: GpuTier, overlay: HTMLElement): Promise<void> {
    this.canvas = canvas;
    await this.field.init(canvas, tier);

    this.tooltip = new InterpTooltip(overlay);
    this.marker = new FieldMarker(overlay);
    this.labelRoot = document.createElement("div");
    this.labelRoot.className = "interp-neuron-labels";
    overlay.appendChild(this.labelRoot);
    this.chipRoot = document.createElement("div");
    this.chipRoot.className = "interp-neuron-chips";
    overlay.appendChild(this.chipRoot);

    const onMove = (e: PointerEvent) => this.onPointerMove(e);
    const onLeave = () => this.onLeave();
    const onClick = (e: PointerEvent) => this.onClick(e);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    canvas.addEventListener("click", onClick);
    this.disposers.push(() => {
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("click", onClick);
    });
  }

  async setModel(model: string, _trace?: string): Promise<void> {
    const b = await loadNeurons(model);
    this.bundle = b;
    const n = b.n;
    const dMlp = b.meta.d_mlp;
    const c = b.coords;
    const pts: NeuronPt[] = new Array(n);
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    let nmin = Infinity;
    let nmax = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = c[i * 2] ?? 0;
      const y = c[i * 2 + 1] ?? 0;
      const nm = b.norm[i] ?? 0;
      pts[i] = {
        x,
        y,
        z: b.z[i] ?? 0,
        norm: nm,
        layer: Math.floor(i / dMlp),
        idx: i % dMlp,
        topTok: b.top_tok[i] ?? "",
        topVal: b.top_val[i] ?? 0,
        botTok: b.bot_tok[i] ?? "",
        botVal: b.bot_val[i] ?? 0,
        id: i,
      };
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (nm < nmin) nmin = nm;
      if (nm > nmax) nmax = nm;
    }
    this.pts = pts;
    this.minX = minX;
    this.maxX = maxX;
    this.minY = minY;
    this.maxY = maxY;
    this.normMin = nmin;
    this.normMax = nmax;
    this.hover = null;
    this.isolate = null;
    this.marker.hide();
    this.field.setDim(1);

    // landmarks: the REAL extremes only (PC1/PC2 range ends + max write norm),
    // each labelled with its layer and the token its direction most promotes.
    const set = new Map<number, NeuronPt>();
    if (pts.length) {
      const extreme = (f: (p: NeuronPt) => number, sign: number): NeuronPt => {
        let best = pts[0] as NeuronPt;
        for (const p of pts) if (sign * f(p) > sign * f(best)) best = p;
        return best;
      };
      for (const p of [
        extreme((p) => p.x, +1),
        extreme((p) => p.x, -1),
        extreme((p) => p.y, +1),
        extreme((p) => p.y, -1),
        extreme((p) => p.norm, +1),
      ]) {
        set.set(p.id, p);
      }
    }
    this.anchors = [...set.values()];

    this.buildChips();
    this.pushField();
    this.field.fitInset(minX, minY, maxX, maxY, this.inset(), FIT);
    this.field.render();
    this.positionLabels();
  }

  private radiusOf(norm: number): number {
    // linear in the real norm (min–max over all neurons) — outliers stay outliers
    const t = (norm - this.normMin) / Math.max(1e-6, this.normMax - this.normMin);
    return 0.9 + t * 2.8;
  }

  private colorOf(p: NeuronPt): [number, number, number] {
    return LAYER_COLORS[p.layer] ?? [205, 210, 224];
  }

  private inset() {
    return { left: GL, right: GR, top: GT, bottom: GB };
  }

  /** Upload the field once per dataset. Unlike the deck.gl version there is no
   *  per-hover or per-isolate layer rebuild: isolate flips one float per point
   *  (`setActive`) and hover only moves DOM chrome, so 36k attributes are built
   *  exactly once. */
  private pushField(): void {
    const n = this.pts.length;
    if (!n) return;
    const pos = new Float32Array(n * 2);
    const color = new Float32Array(n * 3);
    const radius = new Float32Array(n);
    const norms = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const p = this.pts[i] as NeuronPt;
      pos[i * 2] = p.x;
      pos[i * 2 + 1] = p.y;
      const [r, g, b] = this.colorOf(p);
      color[i * 3] = r / 255;
      color[i * 3 + 1] = g / 255;
      color[i * 3 + 2] = b / 255;
      radius[i] = this.radiusOf(p.norm);
      norms[i] = p.norm;
    }
    this.field.setData({ count: n, pos, color, rank: rankNormalize(norms), radius });
    this.applyIsolate();
  }

  private applyIsolate(): void {
    if (this.isolate === null) {
      this.field.setActive(null);
      return;
    }
    const n = this.pts.length;
    const active = new Float32Array(n);
    for (let i = 0; i < n; i++) active[i] = (this.pts[i] as NeuronPt).layer === this.isolate ? 1 : 0;
    this.field.setActive(active);
  }

  private dataBoxPx(): { x0: number; y0: number; x1: number; y1: number } {
    const [x0, y1] = this.field.worldToScreen(this.minX, this.minY);
    const [x1, y0] = this.field.worldToScreen(this.maxX, this.maxY);
    return { x0, y0, x1, y1 };
  }

  private buildChips(): void {
    this.chipRoot.textContent = "";
    if (!this.bundle) return;
    const mk = (label: string, layer: number | null) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "interp-neuron-chip";
      btn.textContent = label;
      btn.setAttribute("aria-pressed", String(this.isolate === layer));
      if (this.isolate === layer) btn.classList.add("is-active");
      if (layer !== null) {
        const [r, g, b] = LAYER_COLORS[layer] ?? [205, 210, 224];
        btn.style.setProperty("--chip-dot", `rgb(${r},${g},${b})`);
      }
      btn.addEventListener("click", () => {
        this.isolate = this.isolate === layer ? null : layer;
        this.hover = null;
        this.marker.hide();
        this.tooltip.hide();
        this.field.setDim(1);
        this.buildChips();
        this.applyIsolate();
        this.field.render();
      });
      this.chipRoot.appendChild(btn);
    };
    mk("all", null);
    for (let l = 0; l < this.bundle.meta.n_layer; l++) mk(`L${l}`, l);
  }

  private positionLabels(): void {
    this.labelRoot.textContent = "";
    if (!this.bundle) return;
    const evr = this.bundle.explained_variance_ratio;
    const pc1 = ((evr[0] ?? 0) * 100).toFixed(1);
    const pc2 = ((evr[1] ?? 0) * 100).toFixed(1);

    const cap = (cls: string, text: string, sx: number, sy: number) => {
      const el = document.createElement("div");
      el.className = cls;
      el.textContent = text;
      el.style.transform = `translate(${sx.toFixed(1)}px, ${sy.toFixed(1)}px)`;
      this.labelRoot.appendChild(el);
    };
    // PC1 caption at the bottom-right of the data bbox (right-mid would sit
    // under the top-right legend card); PC2 caption above the data top.
    const dataCX = (this.minX + this.maxX) / 2;
    const dataCY = (this.minY + this.maxY) / 2;
    const [rx] = this.field.worldToScreen(this.maxX, dataCY);
    const [, by] = this.field.worldToScreen(dataCX, this.minY);
    cap("interp-neuron-axis", `PC1 → · ${pc1}% var`, rx - 96, by + 14);
    const [tx, ty] = this.field.worldToScreen(dataCX, this.maxY);
    cap("interp-neuron-axis", `PC2 ↑ · ${pc2}% var`, tx + 8, ty + 2);

    // anchor labels: real extreme neurons, tinted by their layer's exact color.
    // Labels flip to the left of their point when they'd clip the right edge.
    for (const p of this.anchors) {
      const [sx, sy] = this.field.worldToScreen(p.x, p.y);
      const [r, g, b] = this.colorOf(p);
      const el = document.createElement("div");
      el.className = "interp-neuron-anchor";
      el.textContent = `L${p.layer} ↑${fmtTok(p.topTok)}`;
      el.style.color = `rgb(${r},${g},${b})`;
      this.labelRoot.appendChild(el);
      const w = el.offsetWidth;
      const x = sx + 6 + w > this.cssW - 8 ? sx - w - 6 : sx + 6;
      el.style.transform = `translate(${x.toFixed(1)}px, ${(sy - 8).toFixed(1)}px)`;
    }
  }

  private onPointerMove(e: PointerEvent): void {
    if (this.tooltip.pinned) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    if (!this.showTooltipFor(x, y)) {
      this.tooltip.hide();
      this.canvas.style.cursor = "";
    }
  }

  /** Touch-only: a tap pins the readout so it survives past the finger lifting
   *  (touch has no hover, so pointerleave would otherwise hide it instantly).
   *  A tap on empty space clears the pin and the hover highlight rather than
   *  leaving a stale one stuck. Mouse pointers no-op — the hover path serves them. */
  private onClick(e: PointerEvent): void {
    if (e.pointerType !== "touch") return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    if (this.showTooltipFor(x, y)) {
      this.tooltip.pinned = true;
    } else {
      this.tooltip.pinned = false;
      this.tooltip.hide();
      this.canvas.style.cursor = "";
    }
  }

  /** Pick the neuron at (x, y), update the hover highlight + marker, and show
   *  the tooltip there. Returns whether a neuron was hit. */
  private showTooltipFor(x: number, y: number): boolean {
    let p: NeuronPt | null = null;
    const hit = this.field.pickAt(x, y, 6);
    if (hit >= 0) {
      const cand = (this.pts[hit] as NeuronPt) ?? null;
      // isolate mode dims the other layers but keeps them on screen as context;
      // context is not pickable, exactly as the dim scatter layer wasn't
      p = cand && (this.isolate === null || cand.layer === this.isolate) ? cand : null;
    }
    if ((p?.id ?? -1) !== (this.hover?.id ?? -1)) {
      this.hover = p;
      this.field.setDim(p ? HOVER_DIM : 1);
      if (p) {
        const [sx, sy] = this.field.worldToScreen(p.x, p.y);
        this.marker.show(sx, sy, this.radiusOf(p.norm) + 2, this.dataBoxPx());
      } else {
        this.marker.hide();
      }
      this.field.render();
    }
    if (!p) return false;
    const lc = this.colorOf(p);
    const rows: TipRow[] = [
      { kind: "label", text: `L${p.layer} · neuron ${p.idx}`, swatch: [lc[0], lc[1], lc[2]] },
      {
        text: `PC1 ${p.x.toFixed(2)} · PC2 ${p.y.toFixed(2)} · PC3 ${p.z.toFixed(2)}`,
      },
      { text: "‖w_out‖₂", value: p.norm.toFixed(2), hot: true },
      { text: `promotes "${fmtTok(p.topTok)}" · Δlogit +${p.topVal.toFixed(2)}` },
      { text: `suppresses "${fmtTok(p.botTok)}" · Δlogit ${p.botVal.toFixed(2)}` },
    ];
    this.tooltip.show(rows);
    this.tooltip.move(x, y, this.cssW, this.cssH);
    this.canvas.style.cursor = "crosshair";
    return true;
  }

  private onLeave(): void {
    if (this.tooltip.pinned) return;
    if (this.hover) {
      this.hover = null;
      this.marker.hide();
      this.field.setDim(1);
      this.field.render();
    }
    this.tooltip.hide();
    this.canvas.style.cursor = "";
  }

  frame(_dt: number, _t: number): void {
    // static — one fixed projection of the write directions, no data-bearing motion
  }

  resize(width: number, height: number, dpr: number): void {
    this.cssW = width;
    this.cssH = height;
    this.field.resize(width, height, dpr); // re-applies the stored fit itself
    // the marker and tooltip are anchored in screen px, so a resize invalidates
    // both — drop the hover rather than leave chrome pointing at nothing
    this.hover = null;
    this.marker.hide();
    this.tooltip.hide();
    this.field.setDim(1);
    this.field.render();
    this.positionLabels();
  }

  dispose(): void {
    for (const d of this.disposers) d();
    this.disposers = [];
    this.tooltip?.dispose();
    this.marker?.dispose();
    this.labelRoot?.remove();
    this.chipRoot?.remove();
    this.field.dispose();
  }
}

function fmtTok(s: string): string {
  return s.replace(/^ /, "␣").replace(/\n/g, "⏎") || "∅";
}
