/** #2 Head Fingerprints — all 144 attention heads (12 layers × 12 heads) on two
 *  honest, mathematically bounded axes:
 *    x — mean attention paid to the PREVIOUS token, measured over real forward
 *        passes of the bundled prompts (a stated sample: meta.n_rows query
 *        rows), ∈ [0, 1].
 *    y — OV copying score Σ Re λ / Σ |λ| over the eigenvalues of the head's
 *        residual-space OV map diag(γ₁)·W_V·W_O (ln₁ gain folded, biases
 *        excluded), ∈ [−1, +1]. +1: the head writes back every direction it
 *        reads with positive sign (copying); −1: it systematically inverts.
 *  Unlike the PCA constellations these axes are DIFFERENT quantities, so each
 *  is scaled independently to its full mathematical range — no distance claims
 *  across axes. Dot size is the real ‖OV‖_F (how strongly the head writes),
 *  color is the layer (same viridis ramp as the neuron field). This view
 *  surfaces textbook GPT-2-small structure from raw data: L4H11 is the
 *  previous-token head (prev ≈ 1.0, copying +0.96), and near-pure copying
 *  heads concentrate in layers 9–11.
 *
 *  deck.gl (WebGL2), camera off, static (redraws on hover / layer isolate).
 *  Source: heads.json — circuits in float64 via the d_head×d_head
 *  eigendecomposition (eig(AB) = eig(BA)), behavior from unrounded attention. */

import type { Deck, OrthographicView, PickingInfo } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { type HeadsBundle, loadHeads } from "../../data/interp";
import type { InterpDriver } from "./InterpDriver";
import { LAYER_COLORS } from "./NeuronFieldDriver";

type LayersModule = typeof import("@deck.gl/layers");

const GL = 76; // px gutters — room for y tick labels
const GR = 44;
const GT = 78;
const GB = 96; // x tick labels + the layer-isolate chip row

const DIM_RGBA: [number, number, number, number] = [118, 126, 158, 30];
const GRID_RGBA: [number, number, number, number] = [118, 126, 158, 26];
const ZERO_RGBA: [number, number, number, number] = [166, 173, 200, 64];

const X_TICKS = [0, 0.25, 0.5, 0.75, 1];
const Y_TICKS = [-1, -0.5, 0, 0.5, 1];

interface HeadPt {
  position: [number, number]; // CSS px (recomputed on resize)
  prev: number; // mean attention to previous token (measured)
  copying: number; // OV eigenvalue copying score
  sink: number;
  selfAttn: number;
  entropy: number;
  froOv: number;
  sigmaQk: number;
  eig1Re: number;
  eig1Im: number;
  layer: number;
  head: number;
  id: number;
}

interface GridLine {
  source: [number, number];
  target: [number, number];
  major: boolean;
}

export class HeadFingerprintDriver implements InterpDriver {
  readonly animated = false;
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private canvas!: HTMLCanvasElement;
  private tooltip!: HTMLElement;
  private labelRoot!: HTMLElement;
  private chipRoot!: HTMLElement;

  private bundle: HeadsBundle | null = null;
  private pts: HeadPt[] = [];
  private byLayer: HeadPt[][] = [];
  private anchors: Array<{ p: HeadPt; tag: string }> = [];
  private isolate: number | null = null;
  private dimPts: HeadPt[] = [];
  private froMin = 0;
  private froMax = 1;
  private hover: HeadPt | null = null;

  private cssW = 1;
  private cssH = 1;
  private dpr = 1;
  private disposers: Array<() => void> = [];

  async init(canvas: HTMLCanvasElement, _tier: GpuTier, overlay: HTMLElement): Promise<void> {
    this.canvas = canvas;
    const [core, layers] = await Promise.all([import("@deck.gl/core"), import("@deck.gl/layers")]);
    this.layersMod = layers;
    // flipY:true + zoom 0 → world units ARE css pixels; the driver lays the
    // plot out in pixel space directly (two independent axis scales, so the
    // constellations' equal-px-per-unit contract does not apply here).
    this.deck = new core.Deck({
      canvas,
      views: [new core.OrthographicView({ id: "ortho", flipY: true })],
      viewState: this.viewState(),
      controller: false,
      useDevicePixels: Math.min(this.dpr, 2),
      layers: [],
      width: this.cssW,
      height: this.cssH,
    }) as unknown as Deck<OrthographicView[]>;

    this.tooltip = document.createElement("div");
    this.tooltip.className = "point-tooltip interp-tooltip";
    this.tooltip.style.visibility = "hidden";
    overlay.appendChild(this.tooltip);
    this.labelRoot = document.createElement("div");
    this.labelRoot.className = "interp-neuron-labels";
    overlay.appendChild(this.labelRoot);
    this.chipRoot = document.createElement("div");
    this.chipRoot.className = "interp-neuron-chips";
    overlay.appendChild(this.chipRoot);

    const onMove = (e: PointerEvent) => this.onPointerMove(e);
    const onLeave = () => this.onLeave();
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    this.disposers.push(() => {
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
    });
  }

  async setModel(model: string, _trace?: string): Promise<void> {
    const b = await loadHeads(model);
    this.bundle = b;
    const H = b.meta.n_head;
    const pts: HeadPt[] = new Array(b.n);
    const byLayer: HeadPt[][] = Array.from({ length: b.meta.n_layer }, () => []);
    let fmin = Infinity;
    let fmax = -Infinity;
    for (let i = 0; i < b.n; i++) {
      const fro = b.fro_ov[i] ?? 0;
      const p: HeadPt = {
        position: [0, 0], // set by layoutPoints()
        prev: b.prev[i] ?? 0,
        copying: b.copying[i] ?? 0,
        sink: b.sink[i] ?? 0,
        selfAttn: b.self[i] ?? 0,
        entropy: b.entropy[i] ?? 0,
        froOv: fro,
        sigmaQk: b.sigma_qk[i] ?? 0,
        eig1Re: b.eig1_re[i] ?? 0,
        eig1Im: b.eig1_im[i] ?? 0,
        layer: Math.floor(i / H),
        head: i % H,
        id: i,
      };
      pts[i] = p;
      byLayer[p.layer]?.push(p);
      if (fro < fmin) fmin = fro;
      if (fro > fmax) fmax = fro;
    }
    this.pts = pts;
    this.byLayer = byLayer;
    this.froMin = fmin;
    this.froMax = fmax;
    this.hover = null;
    this.isolate = null;
    this.dimPts = [];

    // landmarks: the real extremes on each encoded quantity, tagged with WHICH
    // extreme they are so the label is a factual pointer, not a name.
    if (pts.length) {
      const extreme = (f: (p: HeadPt) => number, sign: number): HeadPt => {
        let best = pts[0] as HeadPt;
        for (const p of pts) if (sign * f(p) > sign * f(best)) best = p;
        return best;
      };
      const seen = new Set<number>();
      this.anchors = [];
      for (const [p, tag] of [
        [extreme((p) => p.prev, +1), "↑prev"],
        [extreme((p) => p.copying, +1), "↑copy"],
        [extreme((p) => p.copying, -1), "↓copy"],
        [extreme((p) => p.sink, +1), "↑sink"],
      ] as Array<[HeadPt, string]>) {
        if (!seen.has(p.id)) {
          seen.add(p.id);
          this.anchors.push({ p, tag });
        }
      }
    }

    this.layoutPoints();
    this.buildChips();
    this.deck?.setProps({ viewState: this.viewState() });
    this.pushLayers();
    this.positionLabels();
  }

  // ---- pixel-space layout: two independent, fully labelled linear axes ------
  private plotW(): number {
    return Math.max(1, this.cssW - GL - GR);
  }
  private plotH(): number {
    return Math.max(1, this.cssH - GT - GB);
  }
  private xPx(prev: number): number {
    return GL + prev * this.plotW();
  }
  private yPx(copying: number): number {
    return GT + (1 - (copying + 1) / 2) * this.plotH();
  }
  private layoutPoints(): void {
    for (const p of this.pts) {
      p.position = [this.xPx(p.prev), this.yPx(p.copying)];
    }
  }
  private viewState() {
    return {
      ortho: {
        target: [this.cssW / 2, this.cssH / 2, 0] as [number, number, number],
        zoom: 0,
      },
    };
  }

  private radiusOf(fro: number): number {
    // linear in the real ‖OV‖_F (min–max over heads) — outliers stay outliers
    const t = (fro - this.froMin) / Math.max(1e-6, this.froMax - this.froMin);
    return 3 + t * 4.5;
  }

  private colorOf(p: HeadPt): [number, number, number] {
    return LAYER_COLORS[p.layer] ?? [205, 210, 224];
  }

  private gridLines(): GridLine[] {
    const lines: GridLine[] = [];
    for (const t of X_TICKS) {
      const x = this.xPx(t);
      lines.push({ source: [x, this.yPx(1)], target: [x, this.yPx(-1)], major: false });
    }
    for (const t of Y_TICKS) {
      const y = this.yPx(t);
      lines.push({ source: [this.xPx(0), y], target: [this.xPx(1), y], major: t === 0 });
    }
    return lines;
  }

  private pushLayers(): void {
    if (!this.deck || !this.pts.length) return;
    const { ScatterplotLayer, LineLayer } = this.layersMod;
    const active = this.isolate === null ? this.pts : (this.byLayer[this.isolate] ?? []);

    this.deck.setProps({
      layers: [
        new LineLayer<GridLine>({
          id: "head-grid",
          data: this.gridLines(),
          getSourcePosition: (l) => [l.source[0], l.source[1], 0],
          getTargetPosition: (l) => [l.target[0], l.target[1], 0],
          getColor: (l) => (l.major ? ZERO_RGBA : GRID_RGBA),
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        new ScatterplotLayer<HeadPt>({
          id: "head-dim",
          data: this.dimPts,
          getPosition: (p) => [p.position[0], p.position[1], 0],
          getFillColor: DIM_RGBA,
          getRadius: (p) => this.radiusOf(p.froOv),
          radiusUnits: "pixels",
          pickable: false,
        }),
        new ScatterplotLayer<HeadPt>({
          id: "head-active",
          data: active,
          getPosition: (p) => [p.position[0], p.position[1], 0],
          getFillColor: (p) => {
            const [r, g, bl] = this.colorOf(p);
            return [r, g, bl, 190];
          },
          getRadius: (p) => this.radiusOf(p.froOv),
          radiusUnits: "pixels",
          pickable: true,
        }),
        new ScatterplotLayer<HeadPt>({
          id: "head-hover",
          data: this.hover ? [this.hover] : [],
          getPosition: (p) => [p.position[0], p.position[1], 0],
          getFillColor: [255, 255, 255, 235],
          getLineColor: [12, 14, 22, 235],
          stroked: true,
          lineWidthUnits: "pixels",
          getLineWidth: 1.4,
          getRadius: (p) => this.radiusOf(p.froOv) + 1.8,
          radiusUnits: "pixels",
          pickable: false,
        }),
      ],
    });
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
        this.dimPts = this.isolate === null ? [] : this.pts.filter((p) => p.layer !== this.isolate);
        this.hover = null;
        this.tooltip.style.visibility = "hidden";
        this.buildChips();
        this.pushLayers();
      });
      this.chipRoot.appendChild(btn);
    };
    mk("all", null);
    for (let l = 0; l < this.bundle.meta.n_layer; l++) mk(`L${l}`, l);
  }

  private positionLabels(): void {
    this.labelRoot.textContent = "";
    if (!this.bundle) return;

    const cap = (cls: string, text: string, sx: number, sy: number, color?: string) => {
      const el = document.createElement("div");
      el.className = cls;
      el.textContent = text;
      if (color) el.style.color = color;
      el.style.transform = `translate(${sx.toFixed(1)}px, ${sy.toFixed(1)}px)`;
      this.labelRoot.appendChild(el);
      return el;
    };

    // tick labels — the axes are real scales, so every gridline is numbered
    for (const t of X_TICKS) {
      const el = cap("interp-neuron-axis", t.toString(), 0, 0);
      el.style.transform = `translate(${(this.xPx(t) - el.offsetWidth / 2).toFixed(1)}px, ${(this.yPx(-1) + 8).toFixed(1)}px)`;
    }
    for (const t of Y_TICKS) {
      const el = cap("interp-neuron-axis", t === 0 ? "0" : t.toFixed(1), 0, 0);
      el.style.transform = `translate(${(this.xPx(0) - el.offsetWidth - 10).toFixed(1)}px, ${(this.yPx(t) - 8).toFixed(1)}px)`;
    }
    // axis titles
    const xt = cap(
      "interp-neuron-axis",
      `mean attention to previous token → · measured, ${this.bundle.meta.prompts.length} prompts`,
      0,
      0,
    );
    xt.style.transform = `translate(${(this.xPx(0.5) - xt.offsetWidth / 2).toFixed(1)}px, ${(this.yPx(-1) + 26).toFixed(1)}px)`;
    cap("interp-neuron-axis", "OV copying ↑ (weights)", this.xPx(0) - 10, this.yPx(1) - 22);

    // anchor labels: the real extremes, tagged with which extreme they are
    for (const { p, tag } of this.anchors) {
      const [r, g, b] = this.colorOf(p);
      const el = document.createElement("div");
      el.className = "interp-neuron-anchor";
      el.textContent = `L${p.layer}H${p.head} ${tag}`;
      el.style.color = `rgb(${r},${g},${b})`;
      this.labelRoot.appendChild(el);
      const w = el.offsetWidth;
      const [sx, sy] = p.position;
      const x = sx + 8 + w > this.cssW - 8 ? sx - w - 8 : sx + 8;
      el.style.transform = `translate(${x.toFixed(1)}px, ${(sy - 9).toFixed(1)}px)`;
    }
  }

  private onPointerMove(e: PointerEvent): void {
    if (!this.deck) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const info = this.deck.pickObject({ x, y, radius: 6, layerIds: ["head-active"] }) as
      | PickingInfo
      | null;
    const p = (info?.object as HeadPt | undefined) ?? null;
    const changed = (p?.id ?? -1) !== (this.hover?.id ?? -1);
    if (changed) {
      this.hover = p;
      this.pushLayers();
    }
    if (!p) {
      this.tooltip.style.visibility = "hidden";
      this.canvas.style.cursor = "";
      return;
    }
    const b = this.bundle;
    this.tooltip.innerHTML = "";
    const add = (cls: string, text: string) => {
      const el = document.createElement("div");
      el.className = cls;
      el.textContent = text;
      this.tooltip.appendChild(el);
    };
    const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
    const im = p.eig1Im;
    const eig1 = im === 0 ? p.eig1Re.toFixed(2) : `${p.eig1Re.toFixed(2)}${im > 0 ? "+" : "−"}${Math.abs(im).toFixed(2)}i`;
    add("point-tooltip-label", `L${p.layer} · head ${p.head}`);
    add("point-tooltip-conf", `OV copying ${p.copying >= 0 ? "+" : ""}${p.copying.toFixed(3)} · λ₁ = ${eig1}`);
    add("point-tooltip-conf", `‖OV‖_F ${p.froOv.toFixed(2)} · σ_max QK ${p.sigmaQk.toFixed(2)}`);
    add(
      "point-tooltip-conf",
      `attends: prev ${pct(p.prev)} · first ${pct(p.sink)} · self ${pct(p.selfAttn)}`,
    );
    add("point-tooltip-conf", `entropy ${p.entropy.toFixed(3)} (0 = peaked, 1 = uniform)`);
    if (b) {
      add(
        "point-tooltip-conf",
        `measured over ${b.meta.prompts.length} prompts · ${b.meta.n_rows} query rows`,
      );
    }
    this.tooltip.style.visibility = "visible";
    const px = Math.min(x + 14, this.cssW - 280);
    const py = Math.min(y + 14, this.cssH - 130);
    this.tooltip.style.transform = `translate(${px.toFixed(1)}px, ${py.toFixed(1)}px)`;
    this.canvas.style.cursor = "crosshair";
  }

  private onLeave(): void {
    if (this.hover) {
      this.hover = null;
      this.pushLayers();
    }
    this.tooltip.style.visibility = "hidden";
    this.canvas.style.cursor = "";
  }

  frame(_dt: number, _t: number): void {
    // static — fixed axes, no data-bearing motion
  }

  resize(width: number, height: number, dpr: number): void {
    this.cssW = width;
    this.cssH = height;
    this.dpr = dpr;
    this.layoutPoints();
    if (this.isolate !== null) {
      // dim array holds the same objects; positions were updated in place
      this.dimPts = this.dimPts.slice();
    }
    this.deck?.setProps({
      width,
      height,
      useDevicePixels: Math.min(dpr, 2),
      viewState: this.viewState(),
    });
    // positions changed in place — new array refs force attribute regeneration
    this.pts = this.pts.slice();
    this.byLayer = this.byLayer.map((l) => l.slice());
    this.pushLayers();
    this.positionLabels();
  }

  dispose(): void {
    for (const d of this.disposers) d();
    this.disposers = [];
    this.tooltip?.remove();
    this.labelRoot?.remove();
    this.chipRoot?.remove();
    this.deck?.finalize();
    this.deck = null;
  }
}
