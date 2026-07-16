/** #6 Neuron Write-Direction Field — all 36,864 MLP neurons (12 layers × 3072),
 *  each placed at the exact PCA score of its WRITE DIRECTION: the row of
 *  mlp.c_proj that the neuron adds to the residual stream (scaled by its
 *  activation). Nothing is a synthetic layout: positions are real PC scores of
 *  the mean-centered write-direction matrix, dot size is the real ‖w_out‖₂, and
 *  color is the neuron's layer (viridis ramp, luminance strictly ↑ with depth).
 *
 *  Each neuron also carries its direct-path logit readout — the token its write
 *  direction most promotes and most suppresses through the model's own final-LN
 *  gain + tied unembedding: ℓ = ((w − mean(w)) ⊙ γ_f)·W_Eᵀ. That is the DIRECT
 *  path only (no downstream-layer effects, positive activation assumed) and the
 *  view says so. Honest findings this exposes: median write norm grows
 *  monotonically with depth (≈2.2 → 5.2 — late layers write hardest), and
 *  PC1+PC2 explain only ~3.3% of variance — a low-D shadow of a 768-D space.
 *
 *  deck.gl (WebGL2), camera off, static (redraws on hover / layer isolate).
 *  Source: neurons.json — PCA computed offline in float64 (768×768 covariance
 *  eigendecomposition), readout in float32 through the tied W_E. */

import type { Deck, OrthographicView, PickingInfo } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { type NeuronsBundle, loadNeurons } from "../../data/interp";
import {
  ACCENT,
  crosshair,
  MARKER_HOT,
  markerPoly,
  type Seg as ThemeSeg,
  type Vec2,
  withAlpha,
} from "./chart-theme";
import { InterpTooltip, type TipRow } from "./chart-tooltip";
import type { InterpDriver } from "./InterpDriver";

type LayersModule = typeof import("@deck.gl/layers");

const GL = 60; // px gutters (axis captions clear of the data)
const GR = 60;
const GT = 78;
const GB = 88; // extra room for the layer-isolate chip row
const FIT = 0.94;

/** Viridis sampled at t = 0.25 + 0.75·L/11 (low end clipped so layer 0 stays
 *  legible on the dark stage). Luminance is strictly increasing with layer, so
 *  "brighter = deeper" reads truthfully; hover always gives the exact layer. */
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
const DIM_RGBA: [number, number, number, number] = [118, 126, 158, 22];

interface NeuronPt {
  position: [number, number];
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
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private canvas!: HTMLCanvasElement;
  private tooltip!: InterpTooltip;
  private labelRoot!: HTMLElement;
  private chipRoot!: HTMLElement;

  private bundle: NeuronsBundle | null = null;
  private pts: NeuronPt[] = [];
  /** draw-order copy, deterministically shuffled — in layer order L11 would
   *  always paint over L0..L10 in dense regions, a systematic z-order bias. */
  private drawPts: NeuronPt[] = [];
  private byLayer: NeuronPt[][] = [];
  private anchors: NeuronPt[] = [];
  private isolate: number | null = null; // layer to isolate, null = all
  /** cached complement of the isolated layer — recomputed only on isolate
   *  change, so hover-driven pushLayers reuses stable data refs (deck skips
   *  attribute regeneration when the array identity is unchanged). */
  private dimPts: NeuronPt[] = [];
  private minX = 0;
  private maxX = 1;
  private minY = 0;
  private maxY = 1;
  private normMin = 0;
  private normMax = 1;
  private hover: NeuronPt | null = null;

  private cssW = 1;
  private cssH = 1;
  private dpr = 1;
  private disposers: Array<() => void> = [];

  async init(canvas: HTMLCanvasElement, _tier: GpuTier, overlay: HTMLElement): Promise<void> {
    this.canvas = canvas;
    const [core, layers] = await Promise.all([import("@deck.gl/core"), import("@deck.gl/layers")]);
    this.layersMod = layers;
    this.deck = new core.Deck({
      canvas,
      views: [new core.OrthographicView({ id: "ortho", flipY: false })],
      viewState: this.viewState(),
      controller: false,
      useDevicePixels: Math.min(this.dpr, 2),
      layers: [],
      width: this.cssW,
      height: this.cssH,
    }) as unknown as Deck<OrthographicView[]>;

    this.tooltip = new InterpTooltip(overlay);
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
    const b = await loadNeurons(model);
    this.bundle = b;
    const n = b.n;
    const dMlp = b.meta.d_mlp;
    const c = b.coords;
    const pts: NeuronPt[] = new Array(n);
    const byLayer: NeuronPt[][] = Array.from({ length: b.meta.n_layer }, () => []);
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
      const layer = Math.floor(i / dMlp);
      const p: NeuronPt = {
        position: [x, y],
        z: b.z[i] ?? 0,
        norm: nm,
        layer,
        idx: i % dMlp,
        topTok: b.top_tok[i] ?? "",
        topVal: b.top_val[i] ?? 0,
        botTok: b.bot_tok[i] ?? "",
        botVal: b.bot_val[i] ?? 0,
        id: i,
      };
      pts[i] = p;
      byLayer[layer]?.push(p);
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (nm < nmin) nmin = nm;
      if (nm > nmax) nmax = nm;
    }
    this.pts = pts;
    this.drawPts = shuffled(pts);
    this.byLayer = byLayer;
    this.minX = minX;
    this.maxX = maxX;
    this.minY = minY;
    this.maxY = maxY;
    this.normMin = nmin;
    this.normMax = nmax;
    this.hover = null;
    this.isolate = null;
    this.dimPts = [];

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
        extreme((p) => p.position[0], +1),
        extreme((p) => p.position[0], -1),
        extreme((p) => p.position[1], +1),
        extreme((p) => p.position[1], -1),
        extreme((p) => p.norm, +1),
      ]) {
        set.set(p.id, p);
      }
    }
    this.anchors = [...set.values()];

    this.buildChips();
    this.deck?.setProps({ viewState: this.viewState() });
    this.pushLayers();
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

  private pushLayers(): void {
    if (!this.deck || !this.pts.length) return;
    const { ScatterplotLayer, LineLayer, SolidPolygonLayer } = this.layersMod;
    const active = this.isolate === null ? this.drawPts : (this.byLayer[this.isolate] ?? []);
    const dimmed = this.dimPts;

    // markers/crosshair live in world (PC) space, so pixel-authored sizes are
    // scaled by world-units-per-pixel (mirrors the WeightSpectrum template).
    const wpp = 1 / this.zoomPx();
    const bounds = { x0: this.minX, y0: this.minY, x1: this.maxX, y1: this.maxY };
    // crosshair guides snap onto the hovered neuron (req 4)
    const cross: ThemeSeg[] = this.hover
      ? crosshair(this.hover.position[0], this.hover.position[1], bounds, 3 * wpp, 4 * wpp)
      : [];
    // hover LED diamond (translucent glow under a full-alpha core) replaces the
    // old white outline ring (req 4)
    interface Marker {
      poly: Vec2[];
      color: [number, number, number, number];
    }
    const mr = this.hover ? (this.radiusOf(this.hover.norm) + 2) * wpp : 0;
    const marks: Marker[] = this.hover
      ? [
          {
            poly: markerPoly(this.hover.position[0], this.hover.position[1], mr * 2.1),
            color: withAlpha(MARKER_HOT, 0.22),
          },
          {
            poly: markerPoly(this.hover.position[0], this.hover.position[1], mr),
            color: withAlpha(MARKER_HOT, 1),
          },
        ]
      : [];

    this.deck.setProps({
      layers: [
        // isolate mode: the other 11 layers stay as faint context, unpickable
        new ScatterplotLayer<NeuronPt>({
          id: "neuron-dim",
          data: dimmed,
          getPosition: (p) => [p.position[0], p.position[1], 0],
          getFillColor: DIM_RGBA,
          getRadius: (p) => this.radiusOf(p.norm),
          radiusUnits: "pixels",
          pickable: false,
        }),
        new ScatterplotLayer<NeuronPt>({
          id: "neuron-active",
          data: active,
          // the field dims to defer to the focused marker on hover (req 3)
          opacity: this.hover ? 0.38 : 1,
          getPosition: (p) => [p.position[0], p.position[1], 0],
          getFillColor: (p) => {
            const [r, g, bl] = this.colorOf(p);
            return [r, g, bl, 150];
          },
          getRadius: (p) => this.radiusOf(p.norm),
          radiusUnits: "pixels",
          pickable: true,
        }),
        new LineLayer<ThemeSeg>({
          id: "neuron-crosshair",
          data: cross,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: withAlpha(ACCENT, 0.5),
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        new SolidPolygonLayer<Marker>({
          id: "neuron-marker",
          data: marks,
          getPolygon: (m) => m.poly,
          getFillColor: (m) => m.color,
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
        this.dimPts =
          this.isolate === null ? [] : this.drawPts.filter((p) => p.layer !== this.isolate);
        this.hover = null;
        this.tooltip.hide();
        this.buildChips();
        this.pushLayers();
      });
      this.chipRoot.appendChild(btn);
    };
    mk("all", null);
    for (let l = 0; l < this.bundle.meta.n_layer; l++) mk(`L${l}`, l);
  }

  // ---- isometric layout: equal px per PC unit so distances stay faithful -----
  private availW(): number {
    return Math.max(1, this.cssW - GL - GR);
  }
  private availH(): number {
    return Math.max(1, this.cssH - GT - GB);
  }
  private spanX(): number {
    return Math.max(1e-3, this.maxX - this.minX);
  }
  private spanY(): number {
    return Math.max(1e-3, this.maxY - this.minY);
  }
  private zoomPx(): number {
    return Math.max(1, Math.min(this.availW() / this.spanX(), this.availH() / this.spanY()) * FIT);
  }
  private dataCX(): number {
    return (this.minX + this.maxX) / 2;
  }
  private dataCY(): number {
    return (this.minY + this.maxY) / 2;
  }
  private drawCX(): number {
    return GL + this.availW() / 2;
  }
  private drawCY(): number {
    return GT + this.availH() / 2;
  }
  private worldToScreen(wx: number, wy: number): [number, number] {
    const z = this.zoomPx();
    return [this.drawCX() + (wx - this.dataCX()) * z, this.drawCY() - (wy - this.dataCY()) * z];
  }
  private viewState() {
    const z = this.zoomPx();
    return {
      ortho: {
        target: [
          this.dataCX() + (this.cssW / 2 - this.drawCX()) / z,
          this.dataCY() + (this.drawCY() - this.cssH / 2) / z,
          0,
        ] as [number, number, number],
        zoom: Math.log2(z),
      },
    };
  }

  private positionLabels(): void {
    this.labelRoot.textContent = "";
    if (!this.bundle) return;
    const evr = this.bundle.explained_variance_ratio;
    const pc1 = ((evr[0] ?? 0) * 100).toFixed(1);
    const pc2 = ((evr[1] ?? 0) * 100).toFixed(1);

    const cap = (cls: string, text: string, sx: number, sy: number, color?: string) => {
      const el = document.createElement("div");
      el.className = cls;
      el.textContent = text;
      if (color) el.style.color = color;
      el.style.transform = `translate(${sx.toFixed(1)}px, ${sy.toFixed(1)}px)`;
      this.labelRoot.appendChild(el);
    };
    // PC1 caption at the bottom-right of the data bbox (right-mid would sit
    // under the top-right legend card); PC2 caption above the data top.
    const [rx] = this.worldToScreen(this.maxX, this.dataCY());
    const [, by] = this.worldToScreen(this.dataCX(), this.minY);
    cap("interp-neuron-axis", `PC1 → · ${pc1}% var`, rx - 96, by + 14);
    const [tx, ty] = this.worldToScreen(this.dataCX(), this.maxY);
    cap("interp-neuron-axis", `PC2 ↑ · ${pc2}% var`, tx + 8, ty + 2);

    // anchor labels: real extreme neurons, tinted by their layer's exact color.
    // Labels flip to the left of their point when they'd clip the right edge.
    for (const p of this.anchors) {
      const [sx, sy] = this.worldToScreen(p.position[0], p.position[1]);
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
    if (!this.deck) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const info = this.deck.pickObject({ x, y, radius: 5, layerIds: ["neuron-active"] }) as
      | PickingInfo
      | null;
    const p = (info?.object as NeuronPt | undefined) ?? null;
    const changed = (p?.id ?? -1) !== (this.hover?.id ?? -1);
    if (changed) {
      this.hover = p;
      this.pushLayers();
    }
    if (!p) {
      this.tooltip.hide();
      this.canvas.style.cursor = "";
      return;
    }
    const lc = this.colorOf(p);
    const rows: TipRow[] = [
      { kind: "label", text: `L${p.layer} · neuron ${p.idx}`, swatch: [lc[0], lc[1], lc[2]] },
      {
        text: `PC1 ${p.position[0].toFixed(2)} · PC2 ${p.position[1].toFixed(2)} · PC3 ${p.z.toFixed(2)}`,
      },
      { text: "‖w_out‖₂", value: p.norm.toFixed(2), hot: true },
      { text: `promotes “${fmtTok(p.topTok)}” · Δlogit +${p.topVal.toFixed(2)}` },
      { text: `suppresses “${fmtTok(p.botTok)}” · Δlogit ${p.botVal.toFixed(2)}` },
    ];
    this.tooltip.show(rows);
    this.tooltip.move(x, y, this.cssW, this.cssH);
    this.canvas.style.cursor = "crosshair";
  }

  private onLeave(): void {
    if (this.hover) {
      this.hover = null;
      this.pushLayers();
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
    this.dpr = dpr;
    this.deck?.setProps({
      width,
      height,
      useDevicePixels: Math.min(dpr, 2),
      viewState: this.viewState(),
    });
    this.positionLabels();
  }

  dispose(): void {
    for (const d of this.disposers) d();
    this.disposers = [];
    this.tooltip?.dispose();
    this.labelRoot?.remove();
    this.chipRoot?.remove();
    this.deck?.finalize();
    this.deck = null;
  }
}

function fmtTok(s: string): string {
  return s.replace(/^ /, "␣").replace(/\n/g, "⏎") || "∅";
}

/** Deterministic Fisher–Yates (mulberry32) — same order every load, but no
 *  systematic layer-on-top-of-layer painting in overplotted regions. */
function shuffled<T>(arr: T[]): T[] {
  const out = arr.slice();
  let seed = 0x9e3779b9;
  const rand = () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = out[i] as T;
    out[i] = out[j] as T;
    out[j] = tmp;
  }
  return out;
}
