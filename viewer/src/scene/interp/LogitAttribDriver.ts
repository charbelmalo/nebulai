/** #13 Direct Logit Attribution — who wrote the final prediction?
 *
 *  The final residual decomposes exactly into everything ever added to it:
 *  emb + Σ_L(Σ_h head_out + b_o + mlp_out). Each component is projected into
 *  the model's top-1 vs runner-up next-token logit margin through the final
 *  LayerNorm with σ frozen at the forward's actual value (standard DLA), so
 *  the grid cells are ADDITIVE: they sum to the real margin, and the measured
 *  float32 drift between Σ and the true margin is printed, never hidden.
 *  Rows = layers, columns = 12 heads + MLP + the out-proj bias b_o (which
 *  belongs to no head — smearing it across heads would be a lie). The right
 *  gutter accumulates the margin layer by layer: watch the decision form.
 *  Per head, hover shows the argmax-attended token at the final row — what
 *  the head read, not a causal claim.
 *
 *  deck.gl (WebGL2), camera off, static. Source: attrib.json. */

import type { Deck, OrthographicView, PickingInfo } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { appStore, type InterpSelection } from "../../app/store";
import { type AttribBundle, type AttribTrace, isLiveTrace, loadAttrib } from "../../data/interp";
import {
  ACCENT,
  AXIS_RGBA,
  crosshair,
  dashedSegment,
  MARKER_HOT,
  type RGB,
  type Seg as ThemeSeg,
  withAlpha,
} from "./chart-theme";
import { InterpTooltip, type TipRow } from "./chart-tooltip";
import type { InterpDriver } from "./InterpDriver";

type LayersModule = typeof import("@deck.gl/layers");

const GL = 56; // px — layer labels
const GR = 190; // px — cumulative-margin gutter (bars + values)
const GT = 96; // px — header summary + column labels, below the prompt tracebar
const GB = 92; // px — clears the collapsed legend pill even on short stages
// (the cumulative gutter runs to the last row; a 56px band let the pill
// cover the final two Σ values exactly where the decision lands)
const COL_GAP = 10; // px — separates heads from the MLP/bias columns

// diverging ramp from a shared dark base — a ~0 cell is a distinct
// near-invisible grey so "didn't matter" never reads as "mattered a little"
const LOW: [number, number, number] = [64, 66, 96];
const POS: [number, number, number] = [245, 195, 59]; // pushes top-1
const NEG: [number, number, number] = [96, 150, 255]; // pushes runner-up
const ZERO_RGBA: [number, number, number, number] = [118, 126, 158, 14];

interface Cell {
  poly: [number, number][];
  v: number; // exact exported contribution to the margin
  layer: number;
  col: number; // 0..H-1 heads, H = MLP, H+1 = b_o
}

interface Bar {
  poly: [number, number][];
  cum: number;
  layer: number;
}

export class LogitAttribDriver implements InterpDriver {
  readonly animated = false;
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private canvas!: HTMLCanvasElement;
  private tooltip!: InterpTooltip;
  private labelRoot!: HTMLElement;

  private bundle: AttribBundle | null = null;
  private tr: AttribTrace | null = null;
  private nL = 12;
  private nH = 12;
  private cells: Cell[] = [];
  private bars: Bar[] = [];
  private cums: number[] = []; // cumulative margin after each layer
  private scale = 1; // symmetric color clamp = max |contribution| this trace
  private hover: Cell | null = null;
  /** head cell the global cross-view selection outlines */
  private linked: { layer: number; head: number } | null = null;

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
      views: [new core.OrthographicView({ id: "ortho", flipY: true })],
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

  async setModel(model: string, trace?: string): Promise<void> {
    if (!this.bundle || this.bundle.meta.model !== model) {
      this.bundle = await loadAttrib(model);
    }
    this.nL = this.bundle.meta.n_layer;
    this.nH = this.bundle.meta.n_head;
    const found = this.bundle.traces.find((t) => t.slug === trace);
    // typed prompts have no precomputed attribution — never silently swap in
    // a bundled prompt under a live slug
    if (!found && trace && isLiveTrace(trace))
      throw new Error(
        "logit attribution is precomputed per bundled prompt — custom prompts " +
          "cover the forward-trace views only",
      );
    const tr = found ?? this.bundle.traces[0];
    if (!tr) throw new Error("attrib.json has no traces");
    this.tr = tr;
    this.hover = null;

    // symmetric clamp over every grid cell — stated in the header
    this.scale = Math.max(
      1e-9,
      ...tr.heads.map(Math.abs),
      ...tr.mlp.map(Math.abs),
      ...tr.bias.map(Math.abs),
    );
    // cumulative margin: emb, then + each layer's total (heads + MLP + b_o).
    // Built from the exported 4dp values, so the endpoint is sum_check − β,
    // consistent with the printed drift line.
    this.cums = [];
    let cum = tr.emb;
    for (let L = 0; L < this.nL; L++) {
      for (let h = 0; h < this.nH; h++) cum += tr.heads[L * this.nH + h] ?? 0;
      cum += (tr.mlp[L] ?? 0) + (tr.bias[L] ?? 0);
      this.cums.push(cum);
    }

    this.layout();
    this.deck?.setProps({ viewState: this.viewState() });
    this.pushLayers();
    this.positionLabels();
  }

  // ---- pixel-space layout ---------------------------------------------------
  private plotW(): number {
    return Math.max(1, this.cssW - GL - GR);
  }
  private nCols(): number {
    return this.nH + 2; // heads + MLP + b_o
  }
  private cellW(): number {
    return (this.plotW() - COL_GAP) / this.nCols();
  }
  private rowH(): number {
    return Math.min(44, Math.max(10, (this.cssH - GT - GB) / this.nL));
  }
  private colX(c: number): number {
    return GL + c * this.cellW() + (c >= this.nH ? COL_GAP : 0);
  }
  private viewState() {
    return {
      ortho: {
        target: [this.cssW / 2, this.cssH / 2, 0] as [number, number, number],
        zoom: 0,
      },
    };
  }

  private layout(): void {
    const tr = this.tr;
    if (!tr) return;
    const cw = this.cellW();
    const rh = this.rowH();
    const cells: Cell[] = [];
    for (let L = 0; L < this.nL; L++) {
      const y0 = GT + L * rh;
      for (let c = 0; c < this.nCols(); c++) {
        const v =
          c < this.nH ? (tr.heads[L * this.nH + c] ?? 0)
          : c === this.nH ? (tr.mlp[L] ?? 0)
          : (tr.bias[L] ?? 0);
        const x0 = this.colX(c);
        cells.push({
          poly: [
            [x0 + 0.5, y0 + 0.5],
            [x0 + cw - 0.5, y0 + 0.5],
            [x0 + cw - 0.5, y0 + rh - 1],
            [x0 + 0.5, y0 + rh - 1],
          ],
          v,
          layer: L,
          col: c,
        });
      }
    }
    this.cells = cells;

    // cumulative bars: shared linear axis over [min, max] of everything shown,
    // zero axis included so bar direction is meaningful
    const vals = [0, tr.emb, tr.margin, ...this.cums];
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const bx0 = GL + this.plotW() + 14;
    const bw = GR - 84; // leave room for printed values
    const x = (v: number) => bx0 + ((v - lo) / Math.max(1e-9, hi - lo)) * bw;
    this.bars = this.cums.map((cum, L) => {
      const y0 = GT + L * rh + rh / 2 - 3;
      const xa = Math.min(x(0), x(cum));
      const xb = Math.max(x(0), x(cum));
      return {
        poly: [
          [xa, y0],
          [Math.max(xb, xa + 1), y0],
          [Math.max(xb, xa + 1), y0 + 6],
          [xa, y0 + 6],
        ] as [number, number][],
        cum,
        layer: L,
      };
    });
  }

  private colorOf(v: number): [number, number, number, number] {
    if (Math.abs(v) < 5e-5) return ZERO_RGBA; // below the 4dp export floor
    const t = Math.min(1, Math.abs(v) / this.scale);
    const end = v > 0 ? POS : NEG;
    const r = Math.round(LOW[0] + t * (end[0] - LOW[0]));
    const g = Math.round(LOW[1] + t * (end[1] - LOW[1]));
    const b = Math.round(LOW[2] + t * (end[2] - LOW[2]));
    return [r, g, b, Math.round(70 + 185 * t)];
  }

  private pushLayers(): void {
    if (!this.deck || !this.cells.length) return;
    const { SolidPolygonLayer, LineLayer } = this.layersMod;
    const hover = this.hover;
    const rh = this.rowH();
    const cw = this.cellW();
    // hovered cell: keep the crisp outline but recolor it to the danger LED, and
    // snap an ACCENT row/column crosshair onto the cell centre (req 4)
    const edges = hover
      ? hover.poly.map((p, i) => ({
          source: p,
          target: hover.poly[(i + 1) % hover.poly.length] as [number, number],
        }))
      : [];
    const plotBounds = { x0: GL, y0: GT, x1: GL + this.plotW(), y1: GT + this.nL * rh };
    const cross: ThemeSeg[] = hover
      ? crosshair(this.colX(hover.col) + cw / 2, GT + hover.layer * rh + rh / 2, plotBounds)
      : [];
    // zero axis of the cumulative gutter — a DASHED hairline now (req 5)
    const tr = this.tr;
    const vals = tr ? [0, tr.emb, tr.margin, ...this.cums] : [0];
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const bx0 = GL + this.plotW() + 14;
    const zx = bx0 + ((0 - lo) / Math.max(1e-9, hi - lo)) * (GR - 84);
    const axis: ThemeSeg[] = dashedSegment([zx, GT], [zx, GT + this.nL * rh]);
    // cross-view linked head — a steady ACCENT outline on its cell
    const linkEdges: ThemeSeg[] = [];
    if (this.linked && this.linked.layer < this.nL && this.linked.head < this.nH) {
      const lx = this.colX(this.linked.head);
      const ly = GT + this.linked.layer * rh;
      linkEdges.push(
        { source: [lx, ly], target: [lx + cw, ly] },
        { source: [lx + cw, ly], target: [lx + cw, ly + rh] },
        { source: [lx + cw, ly + rh], target: [lx, ly + rh] },
        { source: [lx, ly + rh], target: [lx, ly] },
      );
    }

    this.deck.setProps({
      layers: [
        new SolidPolygonLayer<Cell>({
          id: "attr-cells",
          data: this.cells,
          getPolygon: (c) => c.poly,
          getFillColor: (c) => this.colorOf(c.v),
          pickable: true,
        }),
        new SolidPolygonLayer<Bar>({
          id: "attr-cum",
          data: this.bars,
          getPolygon: (b) => b.poly,
          getFillColor: (b) => this.colorOf(b.cum >= 0 ? this.scale : -this.scale),
          pickable: false,
        }),
        new LineLayer<ThemeSeg>({
          id: "attr-axis",
          data: axis,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: AXIS_RGBA,
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        new LineLayer<ThemeSeg>({
          id: "attr-link",
          data: linkEdges,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: withAlpha(ACCENT, 0.95),
          getWidth: 1.6,
          widthUnits: "pixels",
          pickable: false,
        }),
        new LineLayer<ThemeSeg>({
          id: "attr-crosshair",
          data: cross,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: withAlpha(ACCENT, 0.5),
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        new LineLayer<{ source: [number, number]; target: [number, number] }>({
          id: "attr-hover",
          data: edges,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: withAlpha(MARKER_HOT, 0.86),
          getWidth: 1.2,
          widthUnits: "pixels",
          pickable: false,
        }),
      ],
    });
  }

  private positionLabels(): void {
    this.labelRoot.textContent = "";
    const tr = this.tr;
    if (!tr) return;
    const cw = this.cellW();
    const rh = this.rowH();

    const cap = (text: string, cls = "interp-neuron-axis") => {
      const el = document.createElement("div");
      el.className = cls;
      el.textContent = text;
      this.labelRoot.appendChild(el);
      return el;
    };
    const place = (el: HTMLElement, x: number, y: number) => {
      el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
    };

    // header: the model's real prediction and the margin being attributed
    const h1 = cap(
      `next “${vis(tr.top1[0])}” p ${tr.top1[2].toFixed(3)} · runner-up “${vis(tr.top2[0])}” ` +
        `p ${tr.top2[2].toFixed(3)} · margin Δlogit ${sgn(tr.margin, 4)}`,
    );
    h1.style.color = "rgb(245,195,59)";
    place(h1, GL, GT - 44);
    // honesty line: additivity drift, the two off-grid terms, the color clamp
    const drift = Math.abs(tr.sum_check - tr.margin);
    place(
      cap(
        `Σ cells + emb + β = ${sgn(tr.sum_check, 4)} (float32 drift ${drift.toFixed(4)}) · ` +
          `emb ${sgn(tr.emb, 2)} · ln_f β ${sgn(tr.lnf_bias, 2)} · color ±${this.scale.toFixed(2)} · ` +
          `stream recon rel ${tr.recon_rel}`,
      ),
      GL,
      GT - 30,
    );

    // column labels
    for (let c = 0; c < this.nCols(); c++) {
      const name = c < this.nH ? `H${c}` : c === this.nH ? "MLP" : "b_o";
      const el = cap(name);
      if (c >= this.nH) el.title = c === this.nH ? "MLP block output" : "attention out-projection bias (belongs to no head)";
      place(el, this.colX(c) + Math.max(1, (cw - el.offsetWidth) / 2), GT - 16);
    }
    const cumHdr = cap("Σ→ margin so far");
    place(cumHdr, GL + this.plotW() + 14, GT - 16);

    // row labels
    for (let L = 0; L < this.nL; L++) {
      place(cap(`L${L}`), 8, GT + L * rh + rh / 2 - 8);
    }

    // cumulative values, printed exactly (2 dp — bars carry the shape)
    for (const b of this.bars) {
      const el = cap(sgn(b.cum, 2));
      place(el, GL + this.plotW() + GR - 62, GT + b.layer * rh + rh / 2 - 8);
    }
    // the endpoint: + β closes the sum to the printed margin. Bottom-LEFT —
    // the bottom-right band belongs to the collapsed legend pill.
    place(
      cap(`Σ after L${this.nL - 1} ${sgn(this.cums[this.nL - 1] ?? 0, 2)} + β ${sgn(tr.lnf_bias, 2)} → Δ ${sgn(tr.margin, 4)}`),
      GL,
      GT + this.nL * rh + 8,
    );

    // in-cell values for the strongest contributors (the anchors)
    if (rh >= 15 && cw >= 30) {
      for (const c of this.cells) {
        if (Math.abs(c.v) < 0.3 * this.scale) continue;
        const el = cap(sgn(c.v, 2));
        el.style.color = "rgba(255,255,255,0.92)";
        place(el, this.colX(c.col) + Math.max(1, (cw - el.offsetWidth) / 2), GT + c.layer * rh + rh / 2 - 8);
      }
    }
  }

  private onPointerMove(e: PointerEvent): void {
    if (!this.deck) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const info = this.deck.pickObject({ x, y, radius: 1, layerIds: ["attr-cells"] }) as
      | PickingInfo
      | null;
    const c = (info?.object as Cell | undefined) ?? null;
    if (c !== this.hover) {
      this.hover = c;
      this.pushLayers();
    }
    const tr = this.tr;
    if (!c || !tr) {
      this.tooltip.hide();
      this.canvas.style.cursor = "";
      return;
    }
    const kind =
      c.col < this.nH ? `head ${c.col}` : c.col === this.nH ? "MLP block" : "attn out-proj bias b_o";
    const cc = this.colorOf(c.v);
    const swatch: RGB = [cc[0], cc[1], cc[2]];
    const rows: TipRow[] = [
      { kind: "label", text: `L${c.layer} · ${kind}`, swatch },
      {
        text: `Δ(“${vis(tr.top1[0])}” − “${vis(tr.top2[0])}”)`,
        value: `${sgn(c.v, 4)} logits`,
        hot: true,
      },
    ];
    if (Math.abs(tr.margin) > 1e-6) {
      rows.push({ text: `${sgn((c.v / tr.margin) * 100, 0)}% of the ${sgn(tr.margin, 4)} margin` });
    }
    if (c.col < this.nH) {
      const i = c.layer * this.nH + c.col;
      rows.push({
        text: `reads “${vis(tr.attend_tok[i] ?? "")}” — ${(tr.attend_w[i] ?? 0).toFixed(2)} of final-row attention`,
      });
    }
    this.tooltip.show(rows);
    this.tooltip.move(x, y, this.cssW, this.cssH);
    this.canvas.style.cursor = "crosshair";
  }

  private onClick(e: PointerEvent): void {
    if (!this.deck) return;
    const rect = this.canvas.getBoundingClientRect();
    const info = this.deck.pickObject({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      radius: 1,
      layerIds: ["attr-cells"],
    }) as PickingInfo | null;
    const c = (info?.object as Cell | undefined) ?? null;
    if (!c || c.col >= this.nH) return; // MLP/b_o cells aren't heads
    // toggle publish as the global cross-view head selection
    const same = this.linked?.layer === c.layer && this.linked.head === c.col;
    appStore.getState().setInterpSelection(same ? null : { kind: "head", layer: c.layer, head: c.col });
  }

  /** Cross-view link: outline the selected head's cell. */
  setSelection(sel: InterpSelection | null): void {
    const next = sel?.kind === "head" ? { layer: sel.layer, head: sel.head } : null;
    const same =
      (next === null && this.linked === null) ||
      (next !== null && this.linked !== null && next.layer === this.linked.layer && next.head === this.linked.head);
    if (same) return;
    this.linked = next;
    this.pushLayers();
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
    // static — the grid only changes on trace switches
  }

  resize(width: number, height: number, dpr: number): void {
    this.cssW = width;
    this.cssH = height;
    this.dpr = dpr;
    this.layout(); // rebuilds arrays → fresh refs for deck
    this.hover = null;
    this.deck?.setProps({
      width,
      height,
      useDevicePixels: Math.min(dpr, 2),
      viewState: this.viewState(),
    });
    this.pushLayers();
    this.positionLabels();
  }

  dispose(): void {
    for (const d of this.disposers) d();
    this.disposers = [];
    this.tooltip?.dispose();
    this.labelRoot?.remove();
    this.deck?.finalize();
    this.deck = null;
  }
}

/** Signed fixed-point: +1.57, −0.42 — attribution values are directions. */
function sgn(v: number, dp: number): string {
  return `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(dp)}`;
}

/** Visible-escape a token for labels: leading space → ␣, newline → ⏎. */
function vis(s: string): string {
  const t = s.replace(/\n/g, "⏎").replace(/^ /, "␣");
  return t || "·";
}
