/** #3 Logit-Lens Tunnel — the logit lens (nostalgebraist 2020): decode the
 *  residual stream at the LAST position through the model's own unembedding at
 *  every layer, and watch the next-token prediction sharpen with depth. Real
 *  quantity (trace_*.json → logit_lens_last[layer].topk): the softmax-over-vocab
 *  top-k tokens and their probabilities at each layer 0 (raw embedding) …
 *  n_layer (final), straight from a real forward pass. Nothing is normalized,
 *  smoothed, or interpolated — each row is one layer's actual distribution.
 *
 *  The "tunnel" is the stack of layers you look down: layer 0 at the bottom, the
 *  final layer at the top. Each row is a horizontal probability bar — the top-k
 *  tokens laid left→right, every segment's WIDTH equal to its probability p
 *  (a faithful 0..1 length encoding). The full 0..1 track is drawn behind each
 *  bar so the unfilled remainder is visibly the tail mass NOT in the top-k. The
 *  final layer's top-1 token is painted gold wherever it appears, so you can see
 *  exactly which layer the answer emerges at (the crossover is marked). Hover any
 *  cell for the exact token, probability, and rank. deck.gl (WebGL2), camera off. */

import type { Deck, OrthographicView } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { loadTrace, type TraceBundle } from "../../data/interp";
import type { InterpDriver } from "./InterpDriver";

type LayersModule = typeof import("@deck.gl/layers");

const X0 = -0.6; // world x at probability 0
const X1 = 0.6; // world x at probability 1
const PROBSPAN = X1 - X0;
const YTOP = 0.84; // final layer (top of the tunnel)
const YBOT = -0.84; // layer 0 / embedding (bottom)
const BAR_FRAC = 0.62; // fraction of the row pitch the bar height fills

const LGUT = 60; // px reserved left for layer labels
const RGUT = 176; // px reserved right for the per-layer prediction readout
const PADTOP = 64;
const PADBOT = 132; // room below the bars for the axis + the docked legend

const GOLD: [number, number, number] = [245, 195, 59]; // final answer token
const STEEL: [number, number, number] = [96, 165, 224]; // any other top-k token
const TICKS = [0, 0.25, 0.5, 0.75, 1];

interface Seg {
  r: number; // row index (0 = layer 0 at bottom)
  layer: number; // true layer id from the bundle
  rank: number; // 0-based rank within this layer's top-k
  token: string;
  prob: number;
  c0: number; // cumulative probability before this segment
  isAns: boolean; // token === final layer's top-1
}

export class LogitLensTunnelDriver implements InterpDriver {
  readonly animated = false; // static distribution — redraws on hover only
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private makeView!: () => OrthographicView;
  private canvas!: HTMLCanvasElement;
  private overlay!: HTMLElement;
  private tooltip!: HTMLElement;
  private labelRoot!: HTMLElement;

  private bundle: TraceBundle | null = null;
  private nRows = 0;
  private k = 0;
  private segs: Seg[] = [];
  private finalAns = "";
  private crossover = -1; // first row (ascending) whose top-1 IS the final answer
  private hoverKey: string | null = null;

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

    this.tooltip = document.createElement("div");
    this.tooltip.className = "point-tooltip interp-tooltip";
    this.tooltip.style.visibility = "hidden";
    overlay.appendChild(this.tooltip);
    this.labelRoot = document.createElement("div");
    this.labelRoot.className = "interp-lens-labels";
    overlay.appendChild(this.labelRoot);

    const onMove = (e: PointerEvent) => this.onPointerMove(e);
    const onLeave = () => this.hideTip();
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
    const lens = b.logit_lens_last;
    this.nRows = lens.length;
    this.k = lens.reduce((m, e) => Math.max(m, e.topk.length), 0);
    this.finalAns = lens[lens.length - 1]?.topk[0]?.[0] ?? "";
    this.hoverKey = null;

    // Build the stacked-bar segments. Row r renders lens[r] (layer r, layer 0 at
    // the bottom). Segments are laid left→right by rank; each carries its true
    // cumulative offset so widths sum to the captured top-k mass (≤ 1).
    this.segs = [];
    this.crossover = -1;
    for (let r = 0; r < this.nRows; r++) {
      const entry = lens[r]!;
      let cum = 0;
      for (let rank = 0; rank < entry.topk.length; rank++) {
        const [token, prob] = entry.topk[rank]!;
        this.segs.push({
          r,
          layer: entry.layer,
          rank,
          token,
          prob,
          c0: cum,
          isAns: token === this.finalAns,
        });
        cum += prob;
      }
      if (this.crossover < 0 && entry.topk[0]?.[0] === this.finalAns) this.crossover = r;
    }
    this.pushLayers();
    this.positionLabels();
  }

  private rowY(r: number): number {
    if (this.nRows <= 1) return 0;
    return YBOT + (r / (this.nRows - 1)) * (YTOP - YBOT);
  }
  private barHalf(): number {
    const pitch = this.nRows > 1 ? (YTOP - YBOT) / (this.nRows - 1) : YTOP - YBOT;
    return (pitch * BAR_FRAC) / 2;
  }
  private xOf(p: number): number {
    return X0 + p * PROBSPAN;
  }

  private cellRect(d: Seg): [number, number][] {
    const h = this.barHalf();
    const y = this.rowY(d.r);
    const xa = this.xOf(d.c0);
    const xb = this.xOf(d.c0 + d.prob);
    return [
      [xa, y - h],
      [xb, y - h],
      [xb, y + h],
      [xa, y + h],
    ];
  }

  private fillOf = (d: Seg): [number, number, number, number] => {
    const [rr, gg, bb] = d.isAns ? GOLD : STEEL;
    // alpha steps down with rank so abutting same-color cells stay legible; the
    // final-answer cell is always the most opaque (it's what you're tracking).
    let a = d.isAns ? 235 : Math.max(95, 215 - d.rank * 20);
    if (this.hoverKey) a = this.hoverKey === `${d.r}:${d.rank}` ? 255 : Math.round(a * 0.4);
    return [rr, gg, bb, a];
  };

  private pushLayers(): void {
    if (!this.deck) return;
    const { PolygonLayer, LineLayer } = this.layersMod;
    const rows = Array.from({ length: this.nRows }, (_, r) => r);
    const h = this.barHalf();
    this.deck.setProps({
      layers: [
        // faint full-width track (probability 0..1) — the unfilled part is the
        // tail mass NOT captured by the top-k, shown honestly rather than hidden.
        new PolygonLayer<number>({
          id: "ll-track",
          data: rows,
          getPolygon: (r) => {
            const y = this.rowY(r);
            return [
              [X0, y - h],
              [X1, y - h],
              [X1, y + h],
              [X0, y + h],
            ];
          },
          getFillColor: [255, 255, 255, 8],
          stroked: true,
          filled: true,
          getLineColor: [255, 255, 255, 26],
          lineWidthUnits: "pixels",
          getLineWidth: 1,
          pickable: false,
        }),
        // probability gridlines at 0 / .25 / .5 / .75 / 1
        new LineLayer<number>({
          id: "ll-grid",
          data: TICKS,
          getSourcePosition: (t) => [this.xOf(t), YBOT - 0.05],
          getTargetPosition: (t) => [this.xOf(t), YTOP + 0.05],
          getColor: (t) => [255, 255, 255, t === 0 || t === 1 ? 60 : 26],
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        // the top-k probability cells — width == probability
        new PolygonLayer<Seg>({
          id: "ll-cells",
          data: this.segs,
          getPolygon: (d) => this.cellRect(d),
          getFillColor: this.fillOf,
          stroked: true,
          filled: true,
          getLineColor: [10, 13, 22, 210],
          lineWidthUnits: "pixels",
          getLineWidth: 1,
          lineWidthMinPixels: 1,
          pickable: true,
          updateTriggers: { getFillColor: this.hoverKey ?? "none" },
        }),
      ],
    });
  }

  // ---- layout: fit the bar band into the center, gutters for labels ----------
  private zoomPx(): number {
    const availW = this.cssW - LGUT - RGUT;
    const availH = this.cssH - PADTOP - PADBOT;
    return Math.max(30, Math.min(availW / PROBSPAN, availH / (YTOP - YBOT)));
  }
  private targetX(): number {
    // shift the world origin so the (symmetric) bar band centers in the space
    // between the asymmetric left/right gutters.
    return (RGUT - LGUT) / (2 * this.zoomPx());
  }
  private centerY(): number {
    // vertical center of the available band (PADTOP ≠ PADBOT), in world units.
    return -(PADBOT - PADTOP) / (2 * this.zoomPx());
  }
  private worldToScreen(wx: number, wy: number): [number, number] {
    const z = this.zoomPx();
    return [this.cssW / 2 + (wx - this.targetX()) * z, this.cssH / 2 - (wy - this.centerY()) * z];
  }
  private viewState() {
    return {
      ortho: {
        target: [this.targetX(), this.centerY(), 0] as [number, number, number],
        zoom: Math.log2(this.zoomPx()),
      },
    };
  }

  private positionLabels(): void {
    const b = this.bundle;
    if (!b) return;
    this.labelRoot.textContent = "";
    const lens = b.logit_lens_last;

    for (let r = 0; r < this.nRows; r++) {
      const entry = lens[r]!;
      // left: layer id (embed for layer 0)
      const lab = document.createElement("div");
      lab.className = "interp-lens-row";
      lab.textContent = r === 0 ? "embed" : `L${entry.layer}`;
      const [lx, ly] = this.worldToScreen(X0, this.rowY(r));
      lab.style.transform = `translate(${(lx - 10).toFixed(1)}px, ${ly.toFixed(1)}px)`;
      this.labelRoot.appendChild(lab);

      // right: this layer's argmax prediction + its probability
      const [tok, p] = entry.topk[0] ?? ["", 0];
      const pred = document.createElement("div");
      pred.className = `interp-lens-pred${tok === this.finalAns ? " is-ans" : ""}`;
      pred.innerHTML = `<b>${escapeHtml(fmtTok(tok))}</b> ${p.toFixed(3)}`;
      const [rx, ry] = this.worldToScreen(X1, this.rowY(r));
      pred.style.transform = `translate(${(rx + 10).toFixed(1)}px, ${ry.toFixed(1)}px)`;
      this.labelRoot.appendChild(pred);
    }

    // emergence marker: the layer where the answer first becomes the top-1
    if (this.crossover >= 0) {
      const tag = document.createElement("div");
      tag.className = "interp-lens-emerge";
      tag.textContent = "◂ answer emerges";
      const [ex, ey] = this.worldToScreen(X1, this.rowY(this.crossover));
      tag.style.transform = `translate(${(ex + 92).toFixed(1)}px, ${ey.toFixed(1)}px)`;
      this.labelRoot.appendChild(tag);
    }

    // probability axis ticks + caption along the bottom
    for (const t of TICKS) {
      const tk = document.createElement("div");
      tk.className = "interp-lens-tick";
      tk.textContent = t === 0 ? "0" : t === 1 ? "1" : String(t).replace(/^0/, "");
      const [tx, ty] = this.worldToScreen(this.xOf(t), YBOT - 0.05);
      tk.style.transform = `translate(${tx.toFixed(1)}px, ${(ty + 8).toFixed(1)}px)`;
      this.labelRoot.appendChild(tk);
    }
    const cap = document.createElement("div");
    cap.className = "interp-lens-axis";
    cap.textContent = "P(next token) — width of each cell · top-k shown, rest is tail";
    const [cx, cy] = this.worldToScreen(0, YBOT - 0.05);
    cap.style.transform = `translate(${cx.toFixed(1)}px, ${(cy + 26).toFixed(1)}px)`;
    this.labelRoot.appendChild(cap);
  }

  private onPointerMove(e: PointerEvent): void {
    if (!this.deck || !this.bundle) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const info = this.deck.pickObject({ x, y, radius: 2, layerIds: ["ll-cells"] });
    const seg = info?.object as Seg | undefined;
    const key = seg ? `${seg.r}:${seg.rank}` : null;
    if (key !== this.hoverKey) {
      this.hoverKey = key;
      this.pushLayers();
    }
    if (!seg) {
      this.hideTip();
      return;
    }
    this.tooltip.innerHTML = "";
    const l1 = document.createElement("div");
    l1.className = "point-tooltip-label";
    l1.textContent = `${seg.r === 0 ? "embed" : `layer ${seg.layer}`} · “${fmtTok(seg.token)}”`;
    const l2 = document.createElement("div");
    l2.className = "point-tooltip-conf";
    l2.textContent = `p = ${seg.prob.toFixed(4)} · rank ${seg.rank + 1} of ${this.k}${
      seg.isAns ? " · final answer" : ""
    }`;
    this.tooltip.append(l1, l2);
    this.tooltip.style.visibility = "visible";
    const px = Math.min(x + 14, this.cssW - 260);
    const py = Math.min(y + 14, this.cssH - 54);
    this.tooltip.style.transform = `translate(${px.toFixed(1)}px, ${py.toFixed(1)}px)`;
    this.canvas.style.cursor = "crosshair";
  }

  private hideTip(): void {
    if (this.tooltip) this.tooltip.style.visibility = "hidden";
    this.canvas.style.cursor = "";
  }

  frame(_dt: number, _t: number): void {
    // static — no data-bearing motion
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
    this.tooltip?.remove();
    this.labelRoot?.remove();
    this.deck?.finalize();
    this.deck = null;
  }
}

/** compact, printable token string: ␣ for a leading space, ⏎ for newline. */
function fmtTok(s: string): string {
  return s.replace(/^ /, "␣").replace(/\n/g, "⏎") || "∅";
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>]/g, (c) => (c === "&" ? "&amp;" : c === "<" ? "&lt;" : "&gt;"));
}
