/** #7 Attention-Head Flow Graph — the post-softmax attention pattern of ONE
 *  head, drawn as the canonical two-column flow view. Real quantity
 *  (trace_*.json → attn[layer][head][i][j]): the probability query token i puts
 *  on key token j, straight from a real forward pass. Rows are a causal softmax
 *  so they sum to 1 and j ≤ i only — nothing is normalized, smoothed, or faked.
 *
 *  Left column = tokens as queries (reading order, top→bottom); right column =
 *  the same tokens as keys. A line runs from query i to key j with OPACITY equal
 *  to attn[i][j] (a probability, so opacity is a faithful 0..1 encoding). Hover a
 *  line for the exact weight and the two token strings; hover a token to isolate
 *  its distribution.
 *
 *  A 12×12 layer×head grid picks the head. Each cell is tinted by that head's
 *  attention FOCUS = 1 − mean normalized entropy of its rows — a real summary
 *  (bright = a sharp, few-keys head; dim = a diffuse one), so the grid is itself
 *  a small map of the model's heads. deck.gl (WebGL2), camera off. */

import type { Deck, OrthographicView } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { loadTrace, type TraceBundle } from "../../data/interp";
import type { InterpDriver } from "./InterpDriver";

type LayersModule = typeof import("@deck.gl/layers");

const XCOL = 0.58; // column x (world units), symmetric about 0
const YTOP = 0.9;
const YBOT = -0.9;
const GUTTER = 248; // px reserved each side for the head-grid / legend panels
const LABEL = 112; // px reserved for a token label beyond each column
const PADY = 92;
const MIN_W = 0.008; // don't draw lines below this weight (declutter, stated)

const HOT: [number, number, number] = [245, 195, 59]; // query→key line
const KEYC: [number, number, number] = [70, 200, 235]; // key-side accent

interface Line {
  i: number; // query index
  j: number; // key index
  w: number; // attn[i][j]
}

export class AttentionFlowDriver implements InterpDriver {
  readonly animated = false; // static pattern — redraws on hover / head change
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private makeView!: () => OrthographicView;
  private canvas!: HTMLCanvasElement;
  private overlay!: HTMLElement;
  private tooltip!: HTMLElement;
  private labelRoot!: HTMLElement;
  private gridRoot!: HTMLElement;

  private bundle: TraceBundle | null = null;
  private T = 0;
  private nLayer = 0;
  private nHead = 0;
  private layer = 0;
  private head = 0;
  private focus: number[][] = []; // [layer][head] in [0,1]
  private lines: Line[] = [];
  private hover: { col: "q" | "k"; idx: number } | null = null;

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
    this.labelRoot.className = "interp-attn-labels";
    overlay.appendChild(this.labelRoot);
    this.gridRoot = document.createElement("div");
    this.gridRoot.className = "interp-headgrid";
    overlay.appendChild(this.gridRoot);

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
    this.T = b.meta.T;
    this.nLayer = b.meta.n_layer;
    this.nHead = b.meta.n_head;
    this.layer = 0;
    this.head = 0;
    this.hover = null;
    this.computeFocus();
    this.buildGrid();
    this.rebuild();
  }

  /** per-head focus = 1 − mean_i normalized entropy of the causal attn row. */
  private computeFocus(): void {
    const b = this.bundle!;
    this.focus = [];
    for (let l = 0; l < this.nLayer; l++) {
      const row: number[] = [];
      for (let h = 0; h < this.nHead; h++) {
        let acc = 0;
        let cnt = 0;
        for (let i = 1; i < this.T; i++) {
          const p = b.attn[l]![h]![i]!;
          let ent = 0;
          for (let j = 0; j <= i; j++) {
            const v = p[j]!;
            if (v > 0) ent -= v * Math.log2(v);
          }
          const maxEnt = Math.log2(i + 1); // uniform over the i+1 legal keys
          if (maxEnt > 0) {
            acc += ent / maxEnt;
            cnt += 1;
          }
        }
        const meanNormEnt = cnt > 0 ? acc / cnt : 0;
        row.push(1 - meanNormEnt);
      }
      this.focus.push(row);
    }
  }

  private yOf(i: number): number {
    if (this.T <= 1) return 0;
    return YTOP - (i / (this.T - 1)) * (YTOP - YBOT);
  }

  private rebuild(): void {
    const b = this.bundle;
    if (!b || !this.deck) return;
    const head = b.attn[this.layer]![this.head]!;
    this.lines = [];
    for (let i = 0; i < this.T; i++) {
      const p = head[i]!;
      for (let j = 0; j <= i; j++) {
        const w = p[j]!;
        if (w >= MIN_W) this.lines.push({ i, j, w });
      }
    }
    this.pushLayers();
    this.positionLabels();
  }

  private lineColor = (d: Line): [number, number, number, number] => {
    let a = 12 + d.w * 230; // opacity == attention (probability) + a faint floor
    if (this.hover) {
      const on = this.hover.col === "q" ? d.i === this.hover.idx : d.j === this.hover.idx;
      a = on ? Math.max(60, 12 + d.w * 243) : a * 0.12;
    }
    return [HOT[0], HOT[1], HOT[2], Math.round(Math.max(0, Math.min(255, a)))];
  };

  private pushLayers(): void {
    if (!this.deck) return;
    const { LineLayer, ScatterplotLayer } = this.layersMod;
    const dots = [];
    for (let i = 0; i < this.T; i++) {
      dots.push({ pos: [-XCOL, this.yOf(i)] as [number, number], col: HOT });
      dots.push({ pos: [XCOL, this.yOf(i)] as [number, number], col: KEYC });
    }
    this.deck.setProps({
      layers: [
        new LineLayer<Line>({
          id: "af-lines",
          data: this.lines,
          getSourcePosition: (d) => [-XCOL, this.yOf(d.i)],
          getTargetPosition: (d) => [XCOL, this.yOf(d.j)],
          getColor: this.lineColor,
          getWidth: (d) => 0.6 + d.w * 3.4,
          widthUnits: "pixels",
          pickable: true,
          updateTriggers: { getColor: this.hover ? `${this.hover.col}${this.hover.idx}` : "none" },
        }),
        new ScatterplotLayer<{ pos: [number, number]; col: [number, number, number] }>({
          id: "af-dots",
          data: dots,
          getPosition: (d) => d.pos,
          getFillColor: (d) => [d.col[0], d.col[1], d.col[2], 220],
          getRadius: 3,
          radiusUnits: "pixels",
          pickable: false,
        }),
      ],
    });
  }

  private buildGrid(): void {
    this.gridRoot.textContent = "";
    const title = document.createElement("div");
    title.className = "interp-headgrid-title";
    title.textContent = "layer × head";
    this.gridRoot.appendChild(title);
    const grid = document.createElement("div");
    grid.className = "interp-headgrid-cells";
    grid.style.gridTemplateColumns = `repeat(${this.nHead}, 1fr)`;
    for (let l = 0; l < this.nLayer; l++) {
      for (let h = 0; h < this.nHead; h++) {
        const cell = document.createElement("button");
        cell.type = "button";
        cell.className = "interp-headcell";
        cell.dataset.l = String(l);
        cell.dataset.h = String(h);
        const f = this.focus[l]?.[h] ?? 0;
        cell.style.background = `rgba(245,195,59,${(0.08 + 0.92 * f).toFixed(3)})`;
        cell.title = `L${l} H${h} · focus ${f.toFixed(2)}`;
        cell.addEventListener("click", () => this.selectHead(l, h));
        grid.appendChild(cell);
      }
    }
    this.gridRoot.appendChild(grid);
    this.markActiveCell();
  }

  private markActiveCell(): void {
    for (const el of Array.from(this.gridRoot.querySelectorAll<HTMLElement>(".interp-headcell"))) {
      const on = Number(el.dataset.l) === this.layer && Number(el.dataset.h) === this.head;
      el.classList.toggle("is-active", on);
    }
  }

  private selectHead(l: number, h: number): void {
    this.layer = l;
    this.head = h;
    this.hover = null;
    this.markActiveCell();
    this.rebuild();
  }

  /** side gutter for the panels — full width on desktop, shrinking on narrow
   *  screens so the diagram never collapses (panels overlay, blurred, when tight). */
  private gutterPx(): number {
    return Math.min(GUTTER, Math.max(120, this.cssW * 0.235));
  }
  private zoomPx(): number {
    // fit the columns + their labels into the CENTER band, leaving a gutter each
    // side for the head-grid (left) and legend (right) panels.
    const halfW = (this.cssW - 2 * this.gutterPx()) / 2 - LABEL;
    const halfH = (this.cssH - PADY) / 2;
    return Math.max(40, Math.min(halfW / XCOL, halfH / YTOP));
  }
  private worldToScreen(wx: number, wy: number): [number, number] {
    const z = this.zoomPx();
    return [this.cssW / 2 + wx * z, this.cssH / 2 - wy * z];
  }

  private positionLabels(): void {
    const b = this.bundle;
    if (!b) return;
    this.labelRoot.textContent = "";
    for (let i = 0; i < this.T; i++) {
      const tok = fmtTok(b.token_strs[i] ?? "");
      for (const col of ["q", "k"] as const) {
        const el = document.createElement("div");
        el.className = `interp-attn-tok ${col === "q" ? "is-q" : "is-k"}`;
        el.textContent = tok;
        el.dataset.i = String(i);
        el.dataset.col = col;
        const [sx, sy] = this.worldToScreen(col === "q" ? -XCOL : XCOL, this.yOf(i));
        const offX = col === "q" ? -10 : 10;
        el.style.transform = `translate(${(sx + offX).toFixed(1)}px, ${sy.toFixed(1)}px)`;
        el.addEventListener("pointerenter", () => this.setHover(col, i));
        el.addEventListener("pointerleave", () => this.setHover(null, 0));
        this.labelRoot.appendChild(el);
      }
    }
  }

  private setHover(col: "q" | "k" | null, idx: number): void {
    this.hover = col ? { col, idx } : null;
    for (const el of Array.from(this.labelRoot.querySelectorAll<HTMLElement>(".interp-attn-tok"))) {
      const on = !!col && el.dataset.col === col && Number(el.dataset.i) === idx;
      el.classList.toggle("is-hover", on);
    }
    this.pushLayers();
  }

  private onPointerMove(e: PointerEvent): void {
    if (!this.deck || !this.bundle) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const info = this.deck.pickObject({ x, y, radius: 4, layerIds: ["af-lines"] });
    const line = info?.object as Line | undefined;
    if (!line) {
      this.hideTip();
      return;
    }
    const b = this.bundle;
    const qi = fmtTok(b.token_strs[line.i] ?? "");
    const kj = fmtTok(b.token_strs[line.j] ?? "");
    this.tooltip.innerHTML = "";
    const l1 = document.createElement("div");
    l1.className = "point-tooltip-label";
    l1.textContent = `query “${qi}” → key “${kj}”`;
    const l2 = document.createElement("div");
    l2.className = "point-tooltip-conf";
    l2.textContent = `attn = ${line.w.toFixed(4)} · pos ${line.i}→${line.j} · L${this.layer} H${this.head}`;
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

  private viewState() {
    return {
      ortho: { target: [0, 0, 0] as [number, number, number], zoom: Math.log2(this.zoomPx()) },
    };
  }

  frame(_dt: number, _t: number): void {
    // static — no data-bearing motion
  }

  resize(width: number, height: number, dpr: number): void {
    this.cssW = width;
    this.cssH = height;
    this.dpr = dpr;
    this.deck?.setProps({ width, height, useDevicePixels: Math.min(dpr, 2), viewState: this.viewState() });
    this.positionLabels();
  }

  dispose(): void {
    for (const d of this.disposers) d();
    this.disposers = [];
    this.tooltip?.remove();
    this.labelRoot?.remove();
    this.gridRoot?.remove();
    this.deck?.finalize();
    this.deck = null;
  }
}

/** compact, printable token string: show ␣ for a leading space, ⏎ for newline. */
function fmtTok(s: string): string {
  return s.replace(/^ /, "␣").replace(/\n/g, "⏎") || "∅";
}
