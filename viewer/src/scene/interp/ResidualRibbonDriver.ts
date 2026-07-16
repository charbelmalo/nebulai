/** #8 Residual-Stream Ribbon — the L2 norm of each token's residual stream as it
 *  flows through the network, layer by layer. Real quantity, straight from
 *  trace_*.json → resid_norm[layer][pos]: the Euclidean norm ‖x_ℓ(t)‖₂ of the
 *  residual-stream vector for token t at the output of layer ℓ (ℓ=0 is the
 *  token+position embedding, ℓ=1..n_layer is after each transformer block).
 *
 *  One ribbon per token, drawn left→right across depth. y is log₁₀‖x‖₂ — the norm
 *  grows roughly geometrically with depth (≈10 → ≈3000 over 12 layers on GPT-2),
 *  and one token (usually the first) balloons into a "massive activation" that
 *  dwarfs the rest, so a linear axis would flatten every other token onto the
 *  floor. Log-y keeps every trajectory legible AND preserves order; decade
 *  gridlines (‖x‖=10, 100, 1000) are labeled so the magnitude is readable, not
 *  merely relative. The filled area under each curve is decoration toward the
 *  axis floor — the bright top line and the node dots ARE the numbers. Hover any
 *  node for the exact norm at that (token, layer), plus the token's embed→final
 *  growth factor. deck.gl (WebGL2), camera off (framing from canvas size). */

import type { Deck, OrthographicView, PickingInfo } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { loadTrace, type TraceBundle } from "../../data/interp";
import {
  GRID_RGBA,
  MARKER_HOT,
  markerPoly,
  type RGB,
  type Vec2,
  withAlpha,
  dashedSegment,
} from "./chart-theme";
import { InterpTooltip, type TipRow } from "./chart-tooltip";
import type { InterpDriver } from "./InterpDriver";

type LayersModule = typeof import("@deck.gl/layers");

const SPAN_X = 2.6; // world width of the plot box
const SPAN_Y = 1.5; // world height
const GL = 76; // px left gutter — log₁₀ norm decade labels
const GR = 132; // px right gutter — per-token end labels
const GT = 44; // px top gutter
const GB = 74; // px bottom gutter — layer ticks + x caption
const LOG_PAD = 0.06; // decade padding above/below the data so lines clear the edges

// token-position hue: a cool→warm ramp on t = pos/(T-1). Encodes sequence order
// (a real attribute), nothing more; magnitude lives entirely on the y-axis.
const POS_RAMP: Array<[number, [number, number, number]]> = [
  [0.0, [70, 200, 235]], // cyan — earliest token
  [0.5, [245, 195, 59]], // gold — middle
  [1.0, [234, 79, 134]], // magenta — latest token
];

interface Ribbon {
  token: number;
  tokenStr: string;
  rgb: [number, number, number];
  line: [number, number][];
  area: [number, number][];
  embedNorm: number;
  finalNorm: number;
  peakNorm: number;
  peakLayer: number;
}
interface RNode {
  token: number;
  layer: number;
  norm: number;
  x: number;
  y: number;
  rgb: [number, number, number];
}

export class ResidualRibbonDriver implements InterpDriver {
  readonly animated = false; // static per trace — deck redraws only on hover/resize
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private makeView!: () => OrthographicView;
  private canvas!: HTMLCanvasElement;
  private overlay!: HTMLElement;
  private tooltip!: InterpTooltip;
  private labelRoot!: HTMLElement;
  private axisRoot!: HTMLElement;

  private bundle: TraceBundle | null = null;
  private T = 0;
  private nLayer = 0; // number of blocks; resid_norm has nLayer+1 rows (0..nLayer)
  private ribbons: Ribbon[] = [];
  private nodes: RNode[] = [];
  private logMin = 0;
  private logMax = 1;
  private hoverTok = -1;
  private hoverMark: Vec2 | null = null; // world point of the hovered (token,layer) node

  private cssW = 1;
  private cssH = 1;
  private dpr = 1;
  private disposers: Array<() => void> = [];

  async init(canvas: HTMLCanvasElement, _tier: GpuTier, overlay: HTMLElement): Promise<void> {
    this.canvas = canvas;
    this.overlay = overlay;
    const [core, layers] = await Promise.all([import("@deck.gl/core"), import("@deck.gl/layers")]);
    this.layersMod = layers;
    this.makeView = () => new core.OrthographicView({ id: "ortho", flipY: false });
    this.deck = new core.Deck({
      canvas,
      views: [this.makeView()],
      viewState: this.viewState(),
      controller: false,
      useDevicePixels: Math.min(this.dpr, 2),
      layers: [],
      width: this.cssW,
      height: this.cssH,
    }) as unknown as Deck<OrthographicView[]>;

    this.tooltip = new InterpTooltip(overlay);
    this.labelRoot = document.createElement("div");
    this.labelRoot.className = "interp-rs-labels";
    overlay.appendChild(this.labelRoot);
    this.axisRoot = document.createElement("div");
    this.axisRoot.className = "interp-axis";
    overlay.appendChild(this.axisRoot);

    const onMove = (e: PointerEvent) => this.onPointerMove(e);
    const onLeave = () => this.onLeave();
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    this.disposers.push(() => {
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
    });
  }

  async setModel(model: string, trace?: string): Promise<void> {
    if (!trace) throw new Error("no forward trace selected");
    const b = await loadTrace(model, trace);
    this.bundle = b;
    this.T = b.meta.T;
    this.nLayer = b.meta.n_layer; // resid_norm rows = nLayer + 1
    this.hoverTok = -1;

    // global log₁₀ range over all positive norms → shared, honest y-axis
    let hi = -Infinity;
    let lo = Infinity;
    for (const row of b.resid_norm) {
      for (const v of row) {
        if (v <= 0) continue;
        const l = Math.log10(v);
        if (l > hi) hi = l;
        if (l < lo) lo = l;
      }
    }
    this.logMin = lo - LOG_PAD;
    this.logMax = hi + LOG_PAD;

    const nAxis = b.resid_norm.length - 1; // 0..nLayer along x
    this.ribbons = [];
    this.nodes = [];
    for (let t = 0; t < this.T; t++) {
      const rgb = ramp(POS_RAMP, this.T > 1 ? t / (this.T - 1) : 0);
      const line: [number, number][] = [];
      let peak = -Infinity;
      let peakLayer = 0;
      for (let l = 0; l <= nAxis; l++) {
        const norm = b.resid_norm[l]![t]!;
        const x = this.xAt(l, nAxis);
        const y = this.yAt(norm);
        line.push([x, y]);
        this.nodes.push({ token: t, layer: l, norm, x, y, rgb });
        if (norm > peak) {
          peak = norm;
          peakLayer = l;
        }
      }
      // area: the curve, then closed down to the axis floor (world y = 0)
      const area: [number, number][] = [
        [line[0]![0], 0],
        ...line,
        [line[line.length - 1]![0], 0],
      ];
      this.ribbons.push({
        token: t,
        tokenStr: fmtTok(b.token_strs[t] ?? ""),
        rgb,
        line,
        area,
        embedNorm: b.resid_norm[0]![t]!,
        finalNorm: b.resid_norm[nAxis]![t]!,
        peakNorm: peak,
        peakLayer,
      });
    }
    this.buildAxis();
    this.pushLayers();
    this.positionLabels();
  }

  private xAt(layer: number, nAxis: number): number {
    return (layer / Math.max(nAxis, 1)) * SPAN_X;
  }
  private yAt(norm: number): number {
    const l = norm > 0 ? Math.log10(norm) : this.logMin;
    const t = (l - this.logMin) / Math.max(this.logMax - this.logMin, 1e-6);
    return t * SPAN_Y;
  }

  private pushLayers(): void {
    if (!this.deck) return;
    const { PathLayer, PolygonLayer, ScatterplotLayer, SolidPolygonLayer } = this.layersMod;
    const hv = this.hoverTok;
    const wpp = this.worldPerPx();
    // decade gridlines at norm = 10^k inside the data range — subtle DASHED
    // hairlines now (req 5); dash geometry authored in px, scaled to world.
    const grid: { path: [number, number][] }[] = [];
    const kLo = Math.ceil(this.logMin);
    const kHi = Math.floor(this.logMax);
    for (let k = kLo; k <= kHi; k++) {
      const y = this.yAt(10 ** k);
      for (const s of dashedSegment([0, y], [SPAN_X, y], 3 * wpp, 6 * wpp)) {
        grid.push({ path: [s.source, s.target] });
      }
    }
    // per-layer vertical guides, likewise dashed
    const nAxis = this.nLayer;
    const vguides: { path: [number, number][] }[] = [];
    for (let l = 0; l <= nAxis; l++) {
      const x = this.xAt(l, nAxis);
      for (const s of dashedSegment([x, 0], [x, SPAN_Y], 3 * wpp, 6 * wpp)) {
        vguides.push({ path: [s.source, s.target] });
      }
    }
    // sharp LED marker locked onto the hovered (token, layer) node (req 4)
    const marks = this.hoverMark
      ? [
          { poly: markerPoly(this.hoverMark[0], this.hoverMark[1], 8 * wpp), color: withAlpha(MARKER_HOT, 0.22) },
          { poly: markerPoly(this.hoverMark[0], this.hoverMark[1], 4 * wpp), color: withAlpha(MARKER_HOT, 1) },
        ]
      : [];

    // Fill only the hovered ribbon — overlapping translucent areas would merge
    // into a muddy mass and misrepresent individual values. By default the chart
    // is clean, hue-coded trajectory lines; the "ribbon" (filled area to the
    // floor) reveals itself on hover to emphasize the one token being read.
    const areaAlpha = (d: Ribbon) => (d.token === hv ? 58 : 0);
    const lineAlpha = (d: Ribbon) => (hv < 0 ? 235 : d.token === hv ? 255 : 45);
    const lineWidth = (d: Ribbon) => (d.token === hv ? 3 : 1.9);

    this.deck.setProps({
      layers: [
        new PathLayer<{ path: [number, number][] }>({
          id: "rs-vguides",
          data: vguides,
          getPath: (d) => d.path,
          getColor: GRID_RGBA,
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        new PathLayer<{ path: [number, number][] }>({
          id: "rs-grid",
          data: grid,
          getPath: (d) => d.path,
          getColor: GRID_RGBA,
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        new PolygonLayer<Ribbon>({
          id: "rs-areas",
          data: this.ribbons,
          getPolygon: (d) => d.area,
          getFillColor: (d) => [d.rgb[0], d.rgb[1], d.rgb[2], areaAlpha(d)],
          stroked: false,
          filled: true,
          pickable: false,
          updateTriggers: { getFillColor: hv },
        }),
        new PathLayer<Ribbon>({
          id: "rs-lines",
          data: this.ribbons,
          getPath: (d) => d.line,
          getColor: (d) => [d.rgb[0], d.rgb[1], d.rgb[2], lineAlpha(d)],
          getWidth: lineWidth,
          widthUnits: "pixels",
          jointRounded: true,
          capRounded: true,
          pickable: true,
          updateTriggers: { getColor: hv, getWidth: hv },
        }),
        new ScatterplotLayer<RNode>({
          id: "rs-nodes",
          data: this.nodes,
          getPosition: (d) => [d.x, d.y, 0],
          getFillColor: (d) => {
            const dim = hv >= 0 && d.token !== hv;
            const a = dim ? 40 : 235;
            return [d.rgb[0], d.rgb[1], d.rgb[2], a];
          },
          getRadius: (d) => (d.token === hv ? 3.2 : 2),
          radiusUnits: "pixels",
          stroked: false,
          pickable: true,
          updateTriggers: { getFillColor: hv, getRadius: hv },
        }),
        new SolidPolygonLayer<{ poly: Vec2[]; color: [number, number, number, number] }>({
          id: "rs-marker",
          data: marks,
          getPolygon: (d) => d.poly,
          getFillColor: (d) => d.color,
          pickable: false,
        }),
      ],
    });
  }

  /** World units per screen pixel — pixel-authored dashes/markers scaled to world. */
  private worldPerPx(): number {
    return 1 / this.zoomPx();
  }

  // ---- layout (asymmetric gutters; world box centered in the plot area) ------
  private zoomPx(): number {
    const availW = this.cssW - GL - GR;
    const availH = this.cssH - GT - GB;
    return Math.max(4, Math.min(availW / SPAN_X, availH / SPAN_Y));
  }
  private centerX(): number {
    return GL + (this.cssW - GL - GR) / 2;
  }
  private centerY(): number {
    return GT + (this.cssH - GT - GB) / 2;
  }
  private worldToScreen(wx: number, wy: number): [number, number] {
    const z = this.zoomPx();
    return [this.centerX() + (wx - SPAN_X / 2) * z, this.centerY() - (wy - SPAN_Y / 2) * z];
  }
  private viewState() {
    const z = this.zoomPx();
    return {
      ortho: {
        target: [
          SPAN_X / 2 + (this.cssW / 2 - this.centerX()) / z,
          SPAN_Y / 2 + (this.centerY() - this.cssH / 2) / z,
          0,
        ] as [number, number, number],
        zoom: Math.log2(z),
      },
    };
  }

  private buildAxis(): void {
    this.axisRoot.textContent = "";
    // y decade labels
    const kLo = Math.ceil(this.logMin);
    const kHi = Math.floor(this.logMax);
    for (let k = kLo; k <= kHi; k++) {
      const el = document.createElement("div");
      el.className = "interp-axis-y";
      el.dataset.norm = String(10 ** k);
      el.textContent = `‖x‖=${(10 ** k).toLocaleString("en-US")}`;
      this.axisRoot.appendChild(el);
    }
    // x layer ticks (embed, 1..nLayer)
    for (let l = 0; l <= this.nLayer; l++) {
      const el = document.createElement("div");
      el.className = "interp-rs-xtick";
      el.dataset.layer = String(l);
      el.textContent = l === 0 ? "emb" : String(l);
      this.axisRoot.appendChild(el);
    }
    const yCap = document.createElement("div");
    yCap.className = "interp-axis-x interp-rs-ycap";
    yCap.textContent = "residual-stream L2 norm (log₁₀) →";
    this.axisRoot.appendChild(yCap);
    const xCap = document.createElement("div");
    xCap.className = "interp-axis-x";
    xCap.textContent = "layer (embedding → block 12) →";
    this.axisRoot.appendChild(xCap);
    this.positionAxis();
  }

  private positionAxis(): void {
    for (const el of Array.from(this.axisRoot.querySelectorAll<HTMLElement>(".interp-axis-y"))) {
      const norm = Number(el.dataset.norm);
      const [sx, sy] = this.worldToScreen(0, this.yAt(norm));
      el.style.transform = `translate(${(sx - 8).toFixed(1)}px, ${(sy - 8).toFixed(1)}px) translateX(-100%)`;
    }
    const nAxis = this.nLayer;
    for (const el of Array.from(this.axisRoot.querySelectorAll<HTMLElement>(".interp-rs-xtick"))) {
      const l = Number(el.dataset.layer);
      const [sx, sy] = this.worldToScreen(this.xAt(l, nAxis), 0);
      el.style.transform = `translate(${sx.toFixed(1)}px, ${(sy + 8).toFixed(1)}px) translateX(-50%)`;
    }
    const yCap = this.axisRoot.querySelector<HTMLElement>(".interp-rs-ycap");
    if (yCap) {
      const [, sy] = this.worldToScreen(0, SPAN_Y / 2);
      yCap.style.transform = `translate(16px, ${sy.toFixed(1)}px) rotate(-90deg) translateX(50%)`;
    }
    const xCap = this.axisRoot.querySelector<HTMLElement>(".interp-axis-x:not(.interp-rs-ycap)");
    if (xCap) {
      const [sx, sy] = this.worldToScreen(SPAN_X / 2, 0);
      xCap.style.transform = `translate(${sx.toFixed(1)}px, ${(sy + 30).toFixed(1)}px) translateX(-50%)`;
    }
  }

  /** Token end-labels tracking each ribbon's final-layer point, with a vertical
   *  de-collision pass — many tokens land at near-identical final norms, so a
   *  naive placement overlaps them into an unreadable stack. */
  private positionLabels(): void {
    this.labelRoot.textContent = "";
    const nAxis = this.nLayer;
    const sx0 = this.worldToScreen(this.xAt(nAxis, nAxis), 0)[0];
    const placed = this.ribbons
      .map((rb) => ({ rb, y: this.worldToScreen(0, this.yAt(rb.finalNorm))[1] }))
      .sort((a, b) => a.y - b.y);
    const GAP = 12.5;
    for (let i = 1; i < placed.length; i++) {
      if (placed[i]!.y - placed[i - 1]!.y < GAP) placed[i]!.y = placed[i - 1]!.y + GAP;
    }
    for (const { rb, y } of placed) {
      const el = document.createElement("div");
      el.className = `interp-rs-tok${rb.token === this.hoverTok ? " is-hot" : ""}`;
      el.textContent = rb.tokenStr;
      el.style.color = `rgb(${rb.rgb[0]},${rb.rgb[1]},${rb.rgb[2]})`;
      el.style.transform = `translate(${(sx0 + 8).toFixed(1)}px, ${(y - 6).toFixed(1)}px)`;
      this.labelRoot.appendChild(el);
    }
  }

  private onPointerMove(e: PointerEvent): void {
    if (!this.deck || !this.bundle) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const info = this.deck.pickObject({ x, y, radius: 8, layerIds: ["rs-nodes", "rs-lines"] }) as
      | PickingInfo
      | null;
    if (!info?.object) {
      this.onLeave();
      return;
    }
    let tok: number;
    let layer: number;
    let norm: number;
    if (info.layer?.id === "rs-nodes") {
      const n = info.object as RNode;
      tok = n.token;
      layer = n.layer;
      norm = n.norm;
    } else {
      const rb = info.object as Ribbon;
      tok = rb.token;
      // nearest layer to the cursor's world x
      const wx = info.coordinate ? info.coordinate[0]! : 0;
      layer = Math.max(0, Math.min(this.nLayer, Math.round((wx / SPAN_X) * this.nLayer)));
      norm = this.bundle.resid_norm[layer]![tok]!;
    }
    const mark: Vec2 = [this.xAt(layer, this.nLayer), this.yAt(norm)];
    const markChanged =
      !this.hoverMark || this.hoverMark[0] !== mark[0] || this.hoverMark[1] !== mark[1];
    if (tok !== this.hoverTok) {
      this.hoverTok = tok;
      this.hoverMark = mark;
      this.pushLayers();
      this.positionLabels();
    } else if (markChanged) {
      this.hoverMark = mark;
      this.pushLayers();
    }
    const rb = this.ribbons[tok]!;
    const growth = rb.embedNorm > 0 ? rb.finalNorm / rb.embedNorm : 0;
    const swatch: RGB = [rb.rgb[0], rb.rgb[1], rb.rgb[2]];
    this.tooltip.show([
      {
        kind: "label",
        text: `“${rb.tokenStr}” · ${layer === 0 ? "embedding" : `after block ${layer}`}`,
        swatch,
      },
      { text: "‖x‖₂", value: norm.toFixed(2), hot: true },
      {
        text: `embed ${rb.embedNorm.toFixed(1)} → final ${rb.finalNorm.toFixed(1)} (×${growth.toFixed(1)}) · peak ${rb.peakNorm.toFixed(0)} @ ${rb.peakLayer === 0 ? "emb" : `L${rb.peakLayer}`}`,
      },
    ]);
    this.tooltip.move(x, y, this.cssW, this.cssH);
    this.canvas.style.cursor = "crosshair";
  }

  private onLeave(): void {
    if (this.hoverTok !== -1) {
      this.hoverTok = -1;
      this.hoverMark = null;
      this.pushLayers();
      this.positionLabels();
    }
    this.tooltip?.hide();
    this.canvas.style.cursor = "";
  }

  frame(_dt: number, _t: number): void {
    // static per trace — deck redraws on hover/resize, no data-bearing motion
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
    this.positionAxis();
    this.positionLabels();
  }

  dispose(): void {
    for (const d of this.disposers) d();
    this.disposers = [];
    this.tooltip?.dispose();
    this.labelRoot?.remove();
    this.axisRoot?.remove();
    this.deck?.finalize();
    this.deck = null;
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
