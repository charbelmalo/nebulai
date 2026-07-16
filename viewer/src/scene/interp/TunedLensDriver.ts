/** #20 Tuned-Lens Delta — a fitted affine translator vs the raw logit lens.
 *
 *  The logit lens reads every layer's residual through the final LN +
 *  unembedding; at early layers that readout is unfair because the residual
 *  basis drifts with depth. Here a per-layer affine translator (A_L, b_L),
 *  fit by exact least squares to the final residual on a disclosed corpus,
 *  is applied first — the honest, stated approximation of the tuned lens
 *  (Belrose et al. 2023; the paper trains on KL, this fit is residual MSE —
 *  said everywhere). Left: KL-to-final and top-1-agreement curves per layer
 *  for BOTH lenses, with IQR bands, on held-out positions. Right: the
 *  per-position prediction grid for the selected prompt — every cell is the
 *  lens's real top-1 token colored by its exact KL to the final distribution.
 *
 *  deck.gl (WebGL2), camera off, static. Source: tuned.json (+ trace slug). */

import type { Deck, OrthographicView, PickingInfo } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import {
  type TunedBundle,
  type TunedCurvePoint,
  type TunedGrid,
  isLiveTrace,
  loadTuned,
} from "../../data/interp";
import { ACCENT, dashedSegment, HOT, withAlpha } from "./chart-theme";
import { InterpTooltip, type TipRow } from "./chart-tooltip";
import type { InterpDriver } from "./InterpDriver";

type LayersModule = typeof import("@deck.gl/layers");

const GL = 50; // px — y tick labels
const GR = 16;
const GT = 96; // header lines under the tracebar
const GB = 92; // clears the collapsed legend pill

const TUNED: [number, number, number] = HOT; // amber (== --data-hot token)
const LOGIT: [number, number, number] = [96, 150, 255]; // blue
const GUIDE: [number, number, number] = [118, 126, 158];
const CELL_LO: [number, number, number] = [40, 42, 60]; // KL 0 (agrees with final)

interface Rect {
  x0: number;
  y0: number;
  w: number;
  h: number;
}

interface Cell {
  poly: [number, number][];
  layer: number; // 0..12 (12 = final residual)
  t: number; // position in the prompt
}

interface Seg {
  source: [number, number];
  target: [number, number];
  color: [number, number, number, number];
  width: number;
}

interface Dot {
  pos: [number, number];
  color: [number, number, number, number];
  r: number;
}

interface Band {
  poly: [number, number][];
  color: [number, number, number, number];
}

export class TunedLensDriver implements InterpDriver {
  readonly animated = false;
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private canvas!: HTMLCanvasElement;
  private tooltip!: InterpTooltip;
  private labelRoot!: HTMLElement;
  private chipRoot!: HTMLElement;

  private bundle: TunedBundle | null = null;
  private grid: TunedGrid | null = null;
  private nL = 12;
  private lens: "tuned" | "logit" = "tuned";
  private cells: Cell[] = [];
  private klClamp = 1; // grid color clamp = max cell KL across BOTH lenses (stated)
  private hoverCell: Cell | null = null;
  private hoverLayer = -1; // curve-panel hover (nearest layer), -1 = none

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

  async setModel(model: string, trace?: string): Promise<void> {
    if (!this.bundle || this.bundle.meta.model !== model) {
      this.bundle = await loadTuned(model);
    }
    this.nL = this.bundle.n_layer;
    const gFound = this.bundle.grids.find((x) => x.slug === trace);
    if (!gFound && trace && isLiveTrace(trace))
      throw new Error(
        "the tuned-lens translator grids are precomputed per bundled prompt — " +
          "custom prompts cover the forward-trace views only",
      );
    const g = gFound ?? this.bundle.grids[0];
    if (!g) throw new Error("tuned.json has no grids");
    this.grid = g;
    this.hoverCell = null;
    this.hoverLayer = -1;
    // shared clamp over BOTH lenses so toggling is comparable — but computed
    // over layers ≥ 1 AND positions ≥ 1: the pre-block-0 embedding row and the
    // massive-activation first position are both known outlier regimes (KL can
    // exceed 60–70 bits) and either would crush the whole ramp. Those cells
    // saturate instead; the header says so and the exact value is always in
    // the hover.
    let mx = 0;
    for (const rows of [g.logit, g.tuned]) {
      rows.forEach((row, L) => {
        if (L === 0) return;
        row.forEach((c, t) => {
          if (t > 0) mx = Math.max(mx, c[2]);
        });
      });
    }
    this.klClamp = Math.max(1e-9, mx);
    this.rebuild();
  }

  // ---- regions ---------------------------------------------------------------
  private narrow(): boolean {
    return this.cssW < 640;
  }
  private klRect(): Rect {
    const pw = this.cssW - GL - GR;
    const ph = this.cssH - GT - GB;
    if (this.narrow()) {
      const w = (pw - 42) / 2;
      return { x0: GL, y0: GT, w, h: Math.max(40, ph * 0.4 - 24) };
    }
    const w = Math.max(200, pw * 0.32);
    return { x0: GL, y0: GT, w, h: Math.max(60, ph * 0.56 - 20) };
  }
  private agRect(): Rect {
    const k = this.klRect();
    const ph = this.cssH - GT - GB;
    if (this.narrow()) {
      return { x0: k.x0 + k.w + 42, y0: GT, w: k.w, h: k.h };
    }
    return { x0: k.x0, y0: k.y0 + k.h + 34, w: k.w, h: Math.max(50, ph - k.h - 34) };
  }
  private gridRect(): Rect {
    const k = this.klRect();
    if (this.narrow()) {
      const y0 = k.y0 + k.h + 44;
      return { x0: GL - 14, y0, w: this.cssW - GL - GR + 14, h: Math.max(50, this.cssH - GB - y0) };
    }
    const x0 = k.x0 + k.w + 58;
    return { x0, y0: GT + 16, w: Math.max(60, this.cssW - GR - x0), h: this.cssH - GB - GT - 16 };
  }
  private viewState() {
    return {
      ortho: {
        target: [this.cssW / 2, this.cssH / 2, 0] as [number, number, number],
        zoom: 0,
      },
    };
  }

  /** Curve-panel y max: over layers ≥ 1 (both lenses) plus tuned L0. The raw
   *  logit lens at L0 (~72 bits) is drawn clamped at the top edge with its
   *  true value printed beside it — a stated off-scale point, not a hidden one. */
  private klMax(): number {
    const b = this.bundle;
    if (!b) return 1;
    let mx = 0;
    b.logit.forEach((cv, L) => {
      if (L > 0) mx = Math.max(mx, cv.p75, cv.mean);
    });
    for (const cv of b.tuned) mx = Math.max(mx, cv.p75, cv.mean);
    return Math.max(0.5, Math.ceil(mx));
  }
  private xOfL(r: Rect, L: number): number {
    return r.x0 + (L / this.nL) * r.w;
  }
  private yOfKL(r: Rect, v: number): number {
    return r.y0 + (1 - Math.min(1, v / this.klMax())) * r.h;
  }
  private yOfAg(r: Rect, v: number): number {
    return r.y0 + (1 - v) * r.h;
  }

  private rebuild(): void {
    const g = this.grid;
    if (!g) return;
    const r = this.gridRect();
    const cw = r.w / g.T;
    const rh = r.h / (this.nL + 1);
    const cells: Cell[] = [];
    for (let L = 0; L <= this.nL; L++) {
      const y0 = r.y0 + L * rh;
      for (let t = 0; t < g.T; t++) {
        const x0 = r.x0 + t * cw;
        cells.push({
          poly: [
            [x0 + 0.5, y0 + 0.5],
            [x0 + cw - 0.5, y0 + 0.5],
            [x0 + cw - 0.5, y0 + rh - 1],
            [x0 + 0.5, y0 + rh - 1],
          ],
          layer: L,
          t,
        });
      }
    }
    this.cells = cells;
    this.deck?.setProps({ viewState: this.viewState() });
    this.pushLayers();
    this.positionLabels();
    this.buildChips();
  }

  /** Sequential ramp for a KL cell: dark (0 = matches final) → amber (clamp). */
  private cellColor(kl: number): [number, number, number, number] {
    const t = Math.min(1, kl / this.klClamp);
    return [
      Math.round(CELL_LO[0] + t * (TUNED[0] - CELL_LO[0])),
      Math.round(CELL_LO[1] + t * (TUNED[1] - CELL_LO[1])),
      Math.round(CELL_LO[2] + t * (TUNED[2] - CELL_LO[2])),
      Math.round(70 + 185 * t),
    ];
  }

  private pushLayers(): void {
    const b = this.bundle;
    const g = this.grid;
    if (!this.deck || !b || !g) return;
    const { SolidPolygonLayer, LineLayer, ScatterplotLayer } = this.layersMod;
    const kr = this.klRect();
    const ar = this.agRect();

    const bands: Band[] = [];
    const segs: Seg[] = [];
    const dots: Dot[] = [];
    const curves: Array<["logit" | "tuned", TunedCurvePoint[], [number, number, number]]> = [
      ["logit", b.logit, LOGIT],
      ["tuned", b.tuned, TUNED],
    ];
    for (const [, cv, col] of curves) {
      // IQR band p25..p75 (translucent, drawn under the mean line)
      const top = cv.map((p, L) => [this.xOfL(kr, L), this.yOfKL(kr, p.p75)] as [number, number]);
      const bot = cv.map((p, L) => [this.xOfL(kr, L), this.yOfKL(kr, p.p25)] as [number, number]);
      bands.push({ poly: [...top, ...bot.reverse()], color: [col[0], col[1], col[2], 26] });
      for (let L = 0; L < this.nL; L++) {
        const a = cv[L];
        const nx = cv[L + 1];
        if (!a || !nx) continue;
        segs.push({
          source: [this.xOfL(kr, L), this.yOfKL(kr, a.mean)],
          target: [this.xOfL(kr, L + 1), this.yOfKL(kr, nx.mean)],
          color: [col[0], col[1], col[2], 235],
          width: 1.6,
        });
        segs.push({
          source: [this.xOfL(ar, L), this.yOfAg(ar, a.agree)],
          target: [this.xOfL(ar, L + 1), this.yOfAg(ar, nx.agree)],
          color: [col[0], col[1], col[2], 235],
          width: 1.6,
        });
      }
      for (let L = 0; L <= this.nL; L++) {
        const a = cv[L];
        if (!a) continue;
        dots.push({ pos: [this.xOfL(kr, L), this.yOfKL(kr, a.mean)], color: [col[0], col[1], col[2], 255], r: 2.2 });
        dots.push({ pos: [this.xOfL(ar, L), this.yOfAg(ar, a.agree)], color: [col[0], col[1], col[2], 255], r: 2.2 });
      }
    }
    // guides: KL 0 baseline, 50% agreement — DASHED hairlines now (req 5) so the
    // structure whispers under the curves.
    const dashGuide = (
      a: [number, number],
      b: [number, number],
      color: [number, number, number, number],
      width: number,
    ): Seg[] => dashedSegment(a, b).map((s) => ({ source: s.source, target: s.target, color, width }));
    const guides: Seg[] = [
      ...dashGuide(
        [kr.x0, this.yOfKL(kr, 0)],
        [kr.x0 + kr.w, this.yOfKL(kr, 0)],
        [GUIDE[0], GUIDE[1], GUIDE[2], 60],
        1,
      ),
      ...dashGuide(
        [ar.x0, this.yOfAg(ar, 0.5)],
        [ar.x0 + ar.w, this.yOfAg(ar, 0.5)],
        [GUIDE[0], GUIDE[1], GUIDE[2], 90],
        1,
      ),
    ];
    // crosshair locked onto the hovered layer column (req 4), in the chrome accent
    if (this.hoverLayer >= 0) {
      for (const r of [kr, ar]) {
        guides.push(
          ...dashGuide(
            [this.xOfL(r, this.hoverLayer), r.y0],
            [this.xOfL(r, this.hoverLayer), r.y0 + r.h],
            withAlpha(ACCENT, 0.5),
            1,
          ),
        );
      }
    }

    const rows = this.lens === "tuned" ? g.tuned : g.logit;
    const agreeEdges: Seg[] = [];
    for (const c of this.cells) {
      const cell = rows[c.layer]?.[c.t];
      const fin = g.final_top[c.t];
      if (!cell || !fin) continue;
      if (cell[0] === fin[0]) {
        // outline: this lens's top-1 already equals the final top-1
        const p = c.poly;
        for (let i = 0; i < 4; i++) {
          agreeEdges.push({
            source: p[i] as [number, number],
            target: p[(i + 1) % 4] as [number, number],
            color: [255, 255, 255, 105],
            width: 1,
          });
        }
      }
    }
    const hp = this.hoverCell?.poly;
    const hoverEdges: Seg[] = hp
      ? hp.map((p, i) => ({
          source: p,
          target: hp[(i + 1) % 4] as [number, number],
          color: [255, 255, 255, 230] as [number, number, number, number],
          width: 1.4,
        }))
      : [];

    this.deck.setProps({
      layers: [
        new SolidPolygonLayer<Band>({
          id: "tl-bands",
          data: bands,
          getPolygon: (d) => d.poly,
          getFillColor: (d) => d.color,
          pickable: false,
        }),
        new SolidPolygonLayer<Cell>({
          id: "tl-cells",
          data: this.cells,
          getPolygon: (c) => c.poly,
          getFillColor: (c) => this.cellColor(rows[c.layer]?.[c.t]?.[2] ?? 0),
          pickable: true,
          updateTriggers: { getFillColor: [this.lens, this.klClamp] },
        }),
        new LineLayer<Seg>({
          id: "tl-guides",
          data: guides,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: (e) => e.color,
          getWidth: (e) => e.width,
          widthUnits: "pixels",
          pickable: false,
        }),
        new LineLayer<Seg>({
          id: "tl-lines",
          data: [...segs, ...agreeEdges, ...hoverEdges],
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: (e) => e.color,
          getWidth: (e) => e.width,
          widthUnits: "pixels",
          pickable: false,
        }),
        new ScatterplotLayer<Dot>({
          id: "tl-dots",
          data: dots,
          getPosition: (d) => [d.pos[0], d.pos[1], 0],
          getFillColor: (d) => d.color,
          getRadius: (d) => d.r,
          radiusUnits: "pixels",
          pickable: false,
        }),
      ],
    });
  }

  // ---- labels ----------------------------------------------------------------
  private positionLabels(): void {
    this.labelRoot.textContent = "";
    const b = this.bundle;
    const g = this.grid;
    if (!b || !g) return;
    const narrow = this.narrow();
    const kr = this.klRect();
    const ar = this.agRect();
    const gr = this.gridRect();

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
    const fit = (variants: string[], maxPx: number): string =>
      variants.find((v) => v.length * 6.9 <= maxPx) ?? variants[variants.length - 1] ?? "";

    // header (3 lines)
    const h1 = cap(
      fit(
        [
          `tuned lens (least-squares translator) vs logit lens · KL(final ‖ lens) in bits · ${b.meta.n_eval_pos.toLocaleString()} held-out positions`,
          `tuned vs logit lens · KL(final ‖ lens) bits · ${b.meta.n_eval_pos.toLocaleString()} held-out pos`,
          `tuned vs logit lens · KL bits`,
        ],
        this.cssW - GL - GR,
      ),
    );
    h1.style.color = "rgb(245,195,59)";
    place(h1, GL, GT - 44);
    const r2a = (b.meta.r2_train[0] ?? 0).toFixed(2);
    const r2b = (b.meta.r2_train[this.nL - 1] ?? 0).toFixed(2);
    place(
      cap(
        fit(
          [
            `fit: ‖A·h_L + b − h_final‖² on ${b.meta.n_train_pos.toLocaleString()} Alice positions · R² L0 ${r2a} → L11 ${r2b} · NOT the KL-trained lens (Belrose 2023)`,
            `fit: least squares on ${b.meta.n_train_pos.toLocaleString()} positions · R² ${r2a}→${r2b} · not the KL-trained lens`,
            `least-squares fit · R² ${r2a}→${r2b}`,
          ],
          this.cssW - GL - GR,
        ),
      ),
      GL,
      GT - 30,
    );
    const h3 = cap(
      fit(
        [
          `grid: ${this.lens} lens top-1 per (layer, position) · color 0 → ${this.klClamp.toFixed(1)} bits KL (L0 row + pos 0 saturate) · outline = matches final`,
          `grid: ${this.lens} lens · color 0→${this.klClamp.toFixed(1)} bits (L0 + pos 0 saturate) · outline = final`,
          `${this.lens} · 0→${this.klClamp.toFixed(1)} bits (L0+pos0 sat)`,
          `${this.lens} · 0→${this.klClamp.toFixed(1)} bits`,
        ],
        this.cssW - GL - GR,
      ),
    );
    h3.style.color = "rgb(166,173,200)";
    place(h3, GL, GT - 16);

    // KL panel ticks (no panel title — h1 already names the quantity; a title
    // at kr.y0−14 collided with the h3 header line)
    const km = this.klMax();
    const kstep = km <= 4 ? 1 : Math.ceil(km / 4);
    for (let v = 0; v <= km + 1e-9; v += kstep) {
      const el = cap(v.toFixed(0));
      place(el, kr.x0 - 8 - v.toFixed(0).length * 6.9, this.yOfKL(kr, v) - 7);
    }
    // the raw logit lens at L0 sits far off this scale — say so at the point,
    // just under its clamped dot inside the panel (top-left region is the
    // clamped curve itself, so nudge right of the dot)
    const l0 = b.logit[0];
    if (l0 && l0.mean > km) {
      const off = cap(`← L0 ${l0.mean.toFixed(1)} off-scale`);
      off.style.color = `rgb(${LOGIT[0]},${LOGIT[1]},${LOGIT[2]})`;
      place(off, kr.x0 + kr.w / this.nL + 14, kr.y0 + 2);
    }
    // agreement panel ticks + title INSIDE the panel (its top-left is provably
    // empty: agreement rises left→right from near 0)
    const agTitle = cap(narrow ? "top-1 agree" : "top-1 agreement with final");
    place(agTitle, ar.x0 + 12, ar.y0 + 2);
    for (const v of [0, 0.5, 1]) {
      const el = cap(v === 0.5 ? ".5" : v.toFixed(0));
      place(el, ar.x0 - 8 - el.textContent!.length * 6.9, this.yOfAg(ar, v) - 7);
    }
    // shared layer x ticks under each curve panel
    const lstep = kr.w / this.nL < 26 ? 3 : 1;
    for (let L = 0; L <= this.nL; L += lstep) {
      const s = L === this.nL ? "fin" : `${L}`;
      const kx = this.xOfL(kr, L) - s.length * 3.45;
      place(cap(s), kx, kr.y0 + kr.h + 4);
      place(cap(s), this.xOfL(ar, L) - s.length * 3.45, ar.y0 + ar.h + 4);
    }
    const gl = cap("50%");
    place(gl, ar.x0 + ar.w - 26, this.yOfAg(ar, 0.5) - 14);

    // grid row + column labels
    const cw = gr.w / g.T;
    const rh = gr.h / (this.nL + 1);
    if (rh >= 9) {
      const rstep = rh < 13 ? 2 : 1;
      for (let L = 0; L <= this.nL; L += rstep) {
        const s = L === this.nL ? "fin" : `L${L}`;
        place(cap(s), gr.x0 - 8 - s.length * 6.9, gr.y0 + L * rh + rh / 2 - 7);
      }
    }
    const maxChars = Math.max(1, Math.floor((cw - 6) / 6.9));
    const cstep = cw < 16 ? Math.ceil(16 / cw) : 1;
    for (let t = 0; t < g.T; t += cstep) {
      const s = trunc(vis(g.token_strs[t] ?? ""), maxChars + (cstep > 1 ? 3 : 0));
      const el = cap(s);
      el.style.color = "rgb(166,173,200)";
      place(el, gr.x0 + t * cw + Math.max(1, (cw - s.length * 6.9) / 2), gr.y0 - 15);
    }
    // in-cell top-1 tokens where they fit
    const rows = this.lens === "tuned" ? g.tuned : g.logit;
    if (cw >= 26 && rh >= 12) {
      for (const c of this.cells) {
        const cell = rows[c.layer]?.[c.t];
        if (!cell) continue;
        const s = trunc(vis(b.tok_strs[cell[0]] ?? ""), maxChars);
        const el = cap(s);
        const t = Math.min(1, cell[2] / this.klClamp);
        el.style.color = t > 0.55 ? "rgba(20,20,28,0.95)" : "rgba(230,233,245,0.92)";
        place(el, gr.x0 + c.t * cw + Math.max(1, (cw - s.length * 6.9) / 2), gr.y0 + c.layer * rh + rh / 2 - 7);
      }
    }
    // honesty footer (bottom-left; the centered lens chips and the bottom-right
    // legend pill own the rest of the band — the footer must stop short of both)
    place(
      cap(
        fit(
          [
            `held-out Alice windows (${b.meta.split}) · row = stream entering block L`,
            `held-out eval (${b.meta.split})`,
            `held-out eval`,
          ],
          Math.max(100, this.cssW / 2 - 130 - GL),
        ),
      ),
      GL,
      this.cssH - GB + 40,
    );
  }

  private buildChips(): void {
    this.chipRoot.textContent = "";
    const lab = document.createElement("span");
    lab.className = "interp-neuron-axis";
    // the axis class is position:absolute for plot labels — restore flow here
    lab.style.position = "static";
    lab.style.alignSelf = "center";
    lab.style.marginRight = "2px";
    lab.textContent = "grid lens";
    this.chipRoot.appendChild(lab);
    for (const name of ["tuned", "logit"] as const) {
      const active = this.lens === name;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "interp-neuron-chip";
      if (active) btn.classList.add("is-active");
      btn.setAttribute("aria-pressed", String(active));
      const col = name === "tuned" ? TUNED : LOGIT;
      btn.style.setProperty("--chip-dot", `rgb(${col[0]},${col[1]},${col[2]})`);
      btn.textContent = name;
      btn.title =
        name === "tuned"
          ? "grid shows the tuned lens's top-1 token per (layer, position)"
          : "grid shows the raw logit lens's top-1 token per (layer, position)";
      btn.onclick = () => {
        if (this.lens !== name) {
          this.lens = name;
          this.buildChips();
          this.pushLayers();
          this.positionLabels();
        }
      };
      this.chipRoot.appendChild(btn);
    }
  }

  // ---- hover ------------------------------------------------------------------
  private onPointerMove(e: PointerEvent): void {
    if (!this.deck) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // curve panels: nearest-layer hover (manual hit test — cells own deck picking)
    const kr = this.klRect();
    const ar = this.agRect();
    let curveL = -1;
    for (const r of [kr, ar]) {
      if (x >= r.x0 - 6 && x <= r.x0 + r.w + 6 && y >= r.y0 - 4 && y <= r.y0 + r.h + 16) {
        curveL = Math.max(0, Math.min(this.nL, Math.round(((x - r.x0) / r.w) * this.nL)));
      }
    }
    const info =
      curveL < 0
        ? (this.deck.pickObject({ x, y, radius: 2, layerIds: ["tl-cells"] }) as PickingInfo | null)
        : null;
    const c = (info?.object as Cell | undefined) ?? null;
    if (c !== this.hoverCell || curveL !== this.hoverLayer) {
      this.hoverCell = c;
      this.hoverLayer = curveL;
      this.pushLayers();
    }
    const b = this.bundle;
    const g = this.grid;
    if ((!c && curveL < 0) || !b || !g) {
      this.tooltip.hide();
      this.canvas.style.cursor = "";
      return;
    }
    const lo0 = curveL >= 0 ? b.logit[curveL] : undefined;
    const tu0 = curveL >= 0 ? b.tuned[curveL] : undefined;
    const tuc = c ? g.tuned[c.layer]?.[c.t] : undefined;
    const loc = c ? g.logit[c.layer]?.[c.t] : undefined;
    const fin = c ? g.final_top[c.t] : undefined;
    let rows: TipRow[];
    if (lo0 && tu0) {
      rows = [
        {
          kind: "label",
          text: curveL === this.nL ? "final residual (layer 12)" : `entering block L${curveL}`,
        },
        {
          text: `tuned  KL ${tu0.mean.toFixed(3)} bits (p25 ${tu0.p25.toFixed(2)} · p50 ${tu0.p50.toFixed(2)} · p75 ${tu0.p75.toFixed(2)})`,
          swatch: TUNED,
        },
        {
          text: `logit  KL ${lo0.mean.toFixed(3)} bits (p25 ${lo0.p25.toFixed(2)} · p50 ${lo0.p50.toFixed(2)} · p75 ${lo0.p75.toFixed(2)})`,
          swatch: LOGIT,
        },
        {
          text: `top-1 agreement: tuned ${(tu0.agree * 100).toFixed(1)}% · logit ${(lo0.agree * 100).toFixed(1)}%`,
        },
        { text: `over ${b.meta.n_eval_pos.toLocaleString()} held-out positions` },
      ];
    } else if (c && tuc && loc && fin) {
      rows = [
        {
          kind: "label",
          text: `${c.layer === this.nL ? "final" : `L${c.layer}`} · pos ${c.t} “${vis(g.token_strs[c.t] ?? "")}”`,
        },
        {
          text: `tuned → “${vis(b.tok_strs[tuc[0]] ?? "")}” p ${tuc[1].toFixed(3)} · KL ${tuc[2].toFixed(3)} bits`,
          swatch: TUNED,
        },
        {
          text: `logit → “${vis(b.tok_strs[loc[0]] ?? "")}” p ${loc[1].toFixed(3)} · KL ${loc[2].toFixed(3)} bits`,
          swatch: LOGIT,
        },
        { text: `final → “${vis(b.tok_strs[fin[0]] ?? "")}” p ${fin[1].toFixed(3)}` },
      ];
    } else {
      this.tooltip.hide();
      return;
    }
    this.tooltip.show(rows);
    this.tooltip.move(x, y, this.cssW, this.cssH);
    this.canvas.style.cursor = "crosshair";
  }

  private onLeave(): void {
    if (this.hoverCell || this.hoverLayer >= 0) {
      this.hoverCell = null;
      this.hoverLayer = -1;
      this.pushLayers();
    }
    this.tooltip.hide();
    this.canvas.style.cursor = "";
  }

  frame(_dt: number, _t: number): void {
    // static — redraws only on lens/trace/hover changes
  }

  resize(width: number, height: number, dpr: number): void {
    this.cssW = width;
    this.cssH = height;
    this.dpr = dpr;
    this.hoverCell = null;
    this.hoverLayer = -1;
    this.deck?.setProps({
      width,
      height,
      useDevicePixels: Math.min(dpr, 2),
      viewState: this.viewState(),
    });
    this.rebuild();
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

/** Visible-escape a token: leading space → ␣, newline → ⏎, C0 → ␀-style. */
function vis(s: string): string {
  const t = s
    .replace(/\n/g, "\u23CE")
    .replace(/^ /, "\u2423")
    .replace(/[\x00-\x09\x0b-\x1f\x7f]/g, (c) =>
      String.fromCodePoint(0x2400 + (c === "\x7f" ? 0x21 : c.charCodeAt(0))),
    );
  return t || "\u00B7";
}

function trunc(s: string, n: number): string {
  return s.length <= n ? s : `${s.slice(0, Math.max(1, n - 1))}…`;
}
