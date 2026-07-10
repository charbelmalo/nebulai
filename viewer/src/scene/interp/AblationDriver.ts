/** #17 Ablation Ghosts — causal necessity of the induction circuit.
 *
 *  Same repeated random-token sequence as the Induction Microscope (#2d, same
 *  seed → identical tokens), but now each of the 144 heads is knocked out —
 *  one real ablated forward per (head, mode) — and the grid shows how much the
 *  second-repeat loss rises. Clicking a head draws its "ghost": the ablated
 *  per-position NLL curve overlaid on the baseline's in-context-learning
 *  drop. Combo chips ablate the top induction heads together, exposing the
 *  redundancy single-head ablation understates.
 *
 *  Closes the arc: #2c predicted the circuit from weights, #2d measured its
 *  behavior, this measures what breaks when you remove it.
 *
 *  deck.gl (WebGL2), camera off, static. Source: ablation.json. */

import type { Deck, OrthographicView, PickingInfo } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { type AblationBundle, loadAblation } from "../../data/interp";
import type { InterpDriver } from "./InterpDriver";

type LayersModule = typeof import("@deck.gl/layers");

const GL = 40; // px — grid row labels (L0…L11)
const GR = 14;
const GT = 96; // px — header
const GB = 92; // px — chip strip + collapsed legend pill

const LOW: [number, number, number] = [64, 66, 96];
const AMBER: [number, number, number] = [245, 195, 59];
const BLUE: [number, number, number] = [96, 165, 250];
const BASE_GREY: [number, number, number, number] = [205, 210, 228, 200];

type Mode = "zero" | "mean";
const MODES: Mode[] = ["zero", "mean"];

type Sel = { kind: "head"; layer: number; head: number } | { kind: "combo"; idx: number };

interface GridCell {
  poly: [number, number][];
  layer: number;
  head: number;
}

interface CurveCol {
  poly: [number, number][];
  j: number; // predicted-token index (1..T−1)
}

interface Seg {
  source: [number, number];
  target: [number, number];
}

interface Dot {
  position: [number, number];
  ghost: boolean;
}

export class AblationDriver implements InterpDriver {
  readonly animated = false;
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private canvas!: HTMLCanvasElement;
  private tooltip!: HTMLElement;
  private labelRoot!: HTMLElement;
  private chipRoot!: HTMLElement;

  private bundle: AblationBundle | null = null;
  private mode: Mode = "zero";
  private sel: Sel | null = null;
  private gridCells: GridCell[] = [];
  private curveCols: CurveCol[] = [];
  private baseSegs: Seg[] = [];
  private ghostSegs: Seg[] = [];
  private dots: Dot[] = [];
  private yMax = 1;
  private hover: GridCell | CurveCol | null = null;

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

  async setModel(model: string): Promise<void> {
    if (!this.bundle || this.bundle.meta.model !== model) {
      this.bundle = await loadAblation(model);
    }
    // default ghost = the largest combo (the redundancy headline: the top
    // induction heads ablated together cost ~3× the sum of their singles)
    this.sel = this.bundle.combos.length
      ? { kind: "combo", idx: this.bundle.combos.length - 1 }
      : (() => {
          const top = this.dMaxAbs(this.mode);
          return { kind: "head" as const, layer: top.layer, head: top.head };
        })();
    this.hover = null;
    this.layout();
    this.buildChips();
    this.deck?.setProps({ viewState: this.viewState() });
    this.pushLayers();
    this.positionLabels();
  }

  // ---- pixel-space layout ---------------------------------------------------
  private narrow(): boolean {
    return this.cssW < 640;
  }
  private gb(): number {
    return this.narrow() ? 150 : GB;
  }
  private plotH(): number {
    return Math.max(40, this.cssH - GT - this.gb());
  }
  private gcell(): number {
    const H = this.bundle?.meta.n_head ?? 12;
    const nL = this.bundle?.meta.n_layer ?? 12;
    // narrow keeps BOTH panels side by side (the #2d lesson: stacking starved
    // both) — grid ~42% of the width, the ghost-curve panel gets the rest
    return this.narrow()
      ? Math.min(10, (0.42 * (this.cssW - GL - GR)) / H, this.plotH() / nL)
      : Math.min(36, (0.38 * (this.cssW - GL - GR)) / H, this.plotH() / nL);
  }
  /** ghost-curve panel box — always right of the grid */
  private curveBox(): { x0: number; y0: number; cw: number; ch: number } {
    const g = this.gcell();
    const H = this.bundle?.meta.n_head ?? 12;
    const x0 = GL + H * g + (this.narrow() ? 34 : 70);
    return {
      x0,
      y0: GT,
      cw: Math.max(40, this.cssW - x0 - GR - 6),
      ch: Math.max(30, this.plotH() - 16),
    };
  }

  private viewState() {
    return {
      ortho: {
        target: [this.cssW / 2, this.cssH / 2, 0] as [number, number, number],
        zoom: 0,
      },
    };
  }

  // ---- data access ------------------------------------------------------------
  private dOf(layer: number, head: number, mode: Mode): number {
    const b = this.bundle;
    if (!b) return 0;
    return b[`d_${mode}` as const][layer * b.meta.n_head + head] ?? 0;
  }
  private indOf(layer: number, head: number): number {
    const b = this.bundle;
    if (!b) return 0;
    return b.ind[layer * b.meta.n_head + head] ?? 0;
  }
  /** largest |Δ| across the 144 single-head ablations of a mode (grid clamp) */
  private dMaxAbs(mode: Mode): { v: number; layer: number; head: number } {
    const b = this.bundle;
    if (!b) return { v: 1, layer: 0, head: 0 };
    let best = 0;
    let at = 0;
    b[`d_${mode}` as const].forEach((v, i) => {
      if (Math.abs(v) > Math.abs(best)) {
        best = v;
        at = i;
      }
    });
    return { v: best, layer: Math.floor(at / b.meta.n_head), head: at % b.meta.n_head };
  }
  private selLabel(sel: Sel): string {
    if (sel.kind === "head") return `L${sel.layer}H${sel.head}`;
    return this.bundle?.combos[sel.idx]?.label ?? "combo";
  }
  private selDelta(sel: Sel, mode: Mode): number {
    if (sel.kind === "head") return this.dOf(sel.layer, sel.head, mode);
    return this.bundle?.combos[sel.idx]?.[`d_${mode}` as const] ?? 0;
  }
  /** the selected ablation's full per-position NLL curve (length T−1) */
  private selCurve(sel: Sel, mode: Mode): number[] {
    const b = this.bundle;
    if (!b) return [];
    if (sel.kind === "combo") return b.combos[sel.idx]?.[`nll_${mode}` as const] ?? [];
    const n = b.meta.T - 1;
    const at = (sel.layer * b.meta.n_head + sel.head) * n;
    return b[`nll_${mode}` as const].slice(at, at + n);
  }

  private layout(): void {
    const b = this.bundle;
    if (!b) return;
    const g = this.gcell();
    const cells: GridCell[] = [];
    for (let l = 0; l < b.meta.n_layer; l++) {
      for (let h = 0; h < b.meta.n_head; h++) {
        const x0 = GL + h * g;
        const y0 = GT + l * g;
        cells.push({
          poly: [
            [x0 + 0.5, y0 + 0.5],
            [x0 + g - 0.5, y0 + 0.5],
            [x0 + g - 0.5, y0 + g - 0.5],
            [x0 + 0.5, y0 + g - 0.5],
          ],
          layer: l,
          head: h,
        });
      }
    }
    this.gridCells = cells;
    this.layoutCurves();
  }

  private xOf(j: number): number {
    const T = this.bundle?.meta.T ?? 97;
    const { x0, cw } = this.curveBox();
    return x0 + ((j - 1) / (T - 2)) * cw;
  }
  private yOf(v: number): number {
    const { y0, ch } = this.curveBox();
    return y0 + ch - (Math.min(v, this.yMax) / this.yMax) * ch;
  }

  private layoutCurves(): void {
    const b = this.bundle;
    this.baseSegs = [];
    this.ghostSegs = [];
    this.dots = [];
    this.curveCols = [];
    if (!b) return;
    const T = b.meta.T;
    const ghost = this.sel ? this.selCurve(this.sel, this.mode) : [];
    // axis ceiling = the data's own max, rounded up to a whole nat and stated
    this.yMax = Math.max(1, Math.ceil(Math.max(...b.nll_base, ...(ghost.length ? ghost : [0]))));
    const { y0, ch, cw } = this.curveBox();
    const seg = (arr: number[], j: number): Seg => ({
      source: [this.xOf(j), this.yOf(arr[j - 1] ?? 0)],
      target: [this.xOf(j + 1), this.yOf(arr[j] ?? 0)],
    });
    for (let j = 1; j < T - 1; j++) {
      this.baseSegs.push(seg(b.nll_base, j));
      if (ghost.length) this.ghostSegs.push(seg(ghost, j));
    }
    for (let j = 1; j <= T - 1; j++) {
      this.dots.push({ position: [this.xOf(j), this.yOf(b.nll_base[j - 1] ?? 0)], ghost: false });
      if (ghost.length)
        this.dots.push({ position: [this.xOf(j), this.yOf(ghost[j - 1] ?? 0)], ghost: true });
      const half = cw / (T - 2) / 2;
      this.curveCols.push({
        poly: [
          [this.xOf(j) - half, y0],
          [this.xOf(j) + half, y0],
          [this.xOf(j) + half, y0 + ch],
          [this.xOf(j) - half, y0 + ch],
        ],
        j,
      });
    }
  }

  private gridColor(layer: number, head: number): [number, number, number, number] {
    const v = this.dOf(layer, head, this.mode);
    const clamp = Math.abs(this.dMaxAbs(this.mode).v) || 1;
    const t = Math.min(1, Math.abs(v) / clamp);
    const hi = v >= 0 ? AMBER : BLUE; // + = ablation hurts, − = ablation helps
    return [
      Math.round(LOW[0] + t * (hi[0] - LOW[0])),
      Math.round(LOW[1] + t * (hi[1] - LOW[1])),
      Math.round(LOW[2] + t * (hi[2] - LOW[2])),
      Math.round(55 + 200 * t),
    ];
  }

  private pushLayers(): void {
    if (!this.deck || !this.bundle) return;
    const { SolidPolygonLayer, LineLayer, ScatterplotLayer } = this.layersMod;
    const b = this.bundle;
    const { x0, y0, cw, ch } = this.curveBox();
    const [w0, w1] = b.meta.window;

    const hover = this.hover;
    const hoverSegs: Seg[] = [];
    if (hover && "layer" in hover) {
      hover.poly.forEach((p, i) => {
        hoverSegs.push({
          source: p as [number, number],
          target: hover.poly[(i + 1) % hover.poly.length] as [number, number],
        });
      });
    } else if (hover) {
      hoverSegs.push({ source: [this.xOf(hover.j), y0], target: [this.xOf(hover.j), y0 + ch] });
    }
    const g = this.gcell();
    const selEdges: Seg[] = [];
    if (this.sel?.kind === "head") {
      const sx = GL + this.sel.head * g;
      const sy = GT + this.sel.layer * g;
      selEdges.push(
        { source: [sx, sy], target: [sx + g, sy] },
        { source: [sx + g, sy], target: [sx + g, sy + g] },
        { source: [sx + g, sy + g], target: [sx, sy + g] },
        { source: [sx, sy + g], target: [sx, sy] },
      );
    }
    // axes + the scored-window bounds (guides, not data)
    const guides: Seg[] = [
      { source: [x0, y0], target: [x0, y0 + ch] },
      { source: [x0, y0 + ch], target: [x0 + cw, y0 + ch] },
      { source: [this.xOf(w0), y0], target: [this.xOf(w0), y0 + ch] },
      { source: [this.xOf(w1), y0], target: [this.xOf(w1), y0 + ch] },
    ];

    this.deck.setProps({
      layers: [
        new SolidPolygonLayer<GridCell>({
          id: "agh-grid",
          data: this.gridCells,
          getPolygon: (c) => c.poly,
          getFillColor: (c) => this.gridColor(c.layer, c.head),
          updateTriggers: { getFillColor: this.mode },
          pickable: true,
        }),
        new SolidPolygonLayer<CurveCol>({
          id: "agh-cols",
          data: this.curveCols,
          getPolygon: (c) => c.poly,
          getFillColor: [0, 0, 0, 1], // effectively invisible; exists for picking
          pickable: true,
        }),
        new LineLayer<Seg>({
          id: "agh-guides",
          data: guides,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: [118, 126, 158, 110],
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        new LineLayer<Seg>({
          id: "agh-base",
          data: this.baseSegs,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: BASE_GREY,
          getWidth: 1.2,
          widthUnits: "pixels",
          pickable: false,
        }),
        new LineLayer<Seg>({
          id: "agh-ghost",
          data: this.ghostSegs,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: [AMBER[0], AMBER[1], AMBER[2], 235],
          getWidth: 1.6,
          widthUnits: "pixels",
          pickable: false,
        }),
        new ScatterplotLayer<Dot>({
          id: "agh-dots",
          data: this.dots,
          getPosition: (d) => [d.position[0], d.position[1], 0],
          getFillColor: (d) => (d.ghost ? [AMBER[0], AMBER[1], AMBER[2], 235] : BASE_GREY),
          getRadius: 1.6,
          radiusUnits: "pixels",
          pickable: false,
        }),
        new LineLayer<Seg>({
          id: "agh-sel",
          data: selEdges,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: [245, 195, 59, 230],
          getWidth: 1.4,
          widthUnits: "pixels",
          pickable: false,
        }),
        new LineLayer<Seg>({
          id: "agh-hover",
          data: hoverSegs,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: [255, 255, 255, 200],
          getWidth: 1.2,
          widthUnits: "pixels",
          pickable: false,
        }),
      ],
    });
  }

  // ---- chips: ablation mode + the top-head combos ----------------------------
  private buildChips(): void {
    const b = this.bundle;
    if (!b) return;
    this.chipRoot.textContent = "";
    this.chipRoot.style.bottom = this.narrow() ? "110px" : "";
    for (const mode of MODES) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "interp-neuron-chip";
      btn.textContent = this.narrow() ? mode : mode === "zero" ? "zero-ablate" : "mean-ablate";
      btn.title =
        mode === "zero"
          ? "delete the head's output entirely (off-distribution but standard)"
          : "replace the head's output with its per-position mean over this run";
      btn.setAttribute("aria-pressed", String(mode === this.mode));
      if (mode === this.mode) btn.classList.add("is-active");
      btn.addEventListener("click", () => {
        if (mode === this.mode) return;
        this.mode = mode;
        this.hover = null;
        this.tooltip.style.visibility = "hidden";
        this.layoutCurves();
        this.buildChips();
        this.pushLayers();
        this.positionLabels();
      });
      this.chipRoot.appendChild(btn);
    }
    b.combos.forEach((combo, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "interp-neuron-chip";
      btn.textContent = this.narrow() ? `top-${combo.sites.length}` : combo.label;
      btn.title =
        `ablate ${combo.label} together (the top-${combo.sites.length} induction heads of ` +
        "this run) — redundancy makes the combo exceed the sum of its singles";
      const active = this.sel?.kind === "combo" && this.sel.idx === idx;
      btn.setAttribute("aria-pressed", String(active));
      if (active) btn.classList.add("is-active");
      btn.addEventListener("click", () => {
        if (this.sel?.kind === "combo" && this.sel.idx === idx) return;
        this.sel = { kind: "combo", idx };
        this.hover = null;
        this.tooltip.style.visibility = "hidden";
        this.layoutCurves();
        this.buildChips();
        this.pushLayers();
        this.positionLabels();
      });
      this.chipRoot.appendChild(btn);
    });
  }

  // ---- labels -----------------------------------------------------------------
  private positionLabels(): void {
    this.labelRoot.textContent = "";
    const b = this.bundle;
    if (!b) return;
    const narrow = this.narrow();
    const g = this.gcell();
    const P = b.meta.period;
    const [w0, w1] = b.meta.window;

    const cap = (text: string, cls = "interp-neuron-axis") => {
      const el = document.createElement("div");
      el.className = cls;
      el.textContent = text;
      el.style.overflow = "hidden";
      el.style.textOverflow = "ellipsis";
      this.labelRoot.appendChild(el);
      return el;
    };
    const place = (el: HTMLElement, x: number, y: number) => {
      el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
    };
    const signed = (v: number, dp = 2) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(dp)}`;

    const top = this.dMaxAbs(this.mode);
    const compact = this.cssW < 1000;
    const hMax = `${(this.cssW - (narrow ? 12 : GL) - 12).toFixed(0)}px`;
    const h1 = cap(
      narrow
        ? `ablate heads · top ${this.mode} L${top.layer}H${top.head} Δ${signed(top.v)}`
        : compact
          ? `ablation ghosts · ${P} rand ×2 (seed ${b.meta.seed}) · top ${this.mode} ` +
              `L${top.layer}H${top.head} Δ${signed(top.v)} nats`
          : `ablation ghosts · same ${P} random tokens ×2 as #2d (seed ${b.meta.seed}) · ` +
              `top ${this.mode}-ablation Δ L${top.layer}H${top.head} ${signed(top.v, 4)} nats`,
    );
    h1.style.color = "rgb(245,195,59)";
    h1.style.maxWidth = hMax;
    place(h1, narrow ? 12 : GL, GT - 44);
    const h2 = cap(
      narrow
        ? `Δ NLL j ${w0}–${w1} · base ${b.meta.base_window.toFixed(3)} · ${b.meta.n_forward} fwd`
        : compact
          ? `Δ mean NLL, predicted j ${w0}–${w1} · base ${b.meta.base_window} ` +
              `(1st repeat ${b.meta.base_first}) · ${b.meta.n_forward} fwd · drift ${b.meta.ident_drift}`
          : `Δ = mean NLL (nats) over predicted j ${w0}–${w1} − baseline ${b.meta.base_window} · ` +
              `first repeat ${b.meta.base_first} → in-context drop ×${(b.meta.base_first / b.meta.base_window).toFixed(0)} · ` +
              `${b.meta.n_forward} real forwards · no-ablation drift ${b.meta.ident_drift}`,
    );
    h2.style.maxWidth = hMax;
    place(h2, narrow ? 12 : GL, GT - 30);

    // grid axes — thin by cell size (the #2d lesson)
    const step = g >= 15 ? 1 : g >= 12 ? 2 : 3;
    for (let h = 0; h < b.meta.n_head; h += step) {
      place(cap(`H${h}`), GL + h * g + 2, GT - 15);
    }
    for (let l = 0; l < b.meta.n_layer; l += step) {
      place(cap(`L${l}`), 10, GT + l * g + g / 2 - 8);
    }
    if (g >= 30) {
      for (const c of this.gridCells) {
        const v = this.dOf(c.layer, c.head, this.mode);
        if (Math.abs(v) < 0.05) continue;
        const el = cap(signed(v));
        el.style.color = "rgba(20,22,34,0.95)";
        place(el, GL + c.head * g + g / 2 - 13, GT + c.layer * g + g / 2 - 8);
      }
    }

    // ghost-curve panel
    const { x0, y0, cw, ch } = this.curveBox();
    const sel = this.sel;
    if (sel) {
      const d = this.selDelta(sel, this.mode);
      const shortSel =
        sel.kind === "combo"
          ? `top-${this.bundle?.combos[sel.idx]?.sites.length ?? 0} combo`
          : this.selLabel(sel);
      const st = cap(
        narrow
          ? `${shortSel} ${this.mode} Δ${signed(d)}`
          : compact
            ? `ghost ${this.selLabel(sel)} · ${this.mode} · Δ ${signed(d)} nats`
            : `ghost: ${this.selLabel(sel)} · ${this.mode}-ablated NLL vs baseline · Δ ${signed(d, 4)} nats over the window`,
        "interp-neuron-axis",
      );
      st.style.color = "rgb(245,195,59)";
      st.style.maxWidth = `${(this.cssW - x0 - GR).toFixed(0)}px`;
      place(st, x0, y0 - 15);
    }
    // y ticks (nats): 0 at the axis foot, the stated ceiling at the top
    place(cap(String(this.yMax)), Math.max(2, x0 - 24), y0 - 2);
    place(cap("0"), Math.max(2, x0 - 16), y0 + ch - 12);
    // x ticks: j=1 sits bottom-left (safe); the window-bound ticks live at the
    // TOP next to their guides — the bottom-right band belongs to the
    // collapsed legend pill (the #13/#2d lesson), which would cover them
    place(cap("j=1"), x0 - 4, y0 + ch + 2);
    place(cap(String(w0)), this.xOf(w0) - (w0 >= 10 ? 22 : 14), y0 + 2);
    place(cap(String(w1)), this.xOf(w1) - (w1 >= 10 ? 26 : 18), y0 + 2);
    // the scored window is where the baseline has already dropped low, so the
    // strip right of its start guide is empty — the caption lives there
    if (!narrow) {
      place(cap(compact ? "scored window" : `scored window: predicted j ${w0}–${w1}`), this.xOf(w0) + 6, y0 + 2);
    }
    if (!narrow && !compact) {
      place(cap("NLL (nats), linear · lines join 96 discrete values"), x0 + 4, y0 + ch - 14);
    }
  }

  // ---- interaction --------------------------------------------------------------
  private pick(e: PointerEvent): GridCell | CurveCol | null {
    if (!this.deck) return null;
    const rect = this.canvas.getBoundingClientRect();
    const info = this.deck.pickObject({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      radius: 1,
      layerIds: ["agh-grid", "agh-cols"],
    }) as PickingInfo | null;
    return (info?.object as GridCell | CurveCol | undefined) ?? null;
  }

  private onPointerMove(e: PointerEvent): void {
    const b = this.bundle;
    if (!b) return;
    const c = this.pick(e);
    if (c !== this.hover) {
      this.hover = c;
      this.pushLayers();
    }
    if (!c) {
      this.tooltip.style.visibility = "hidden";
      this.canvas.style.cursor = "";
      return;
    }
    this.tooltip.innerHTML = "";
    const add = (cls: string, text: string) => {
      const el = document.createElement("div");
      el.className = cls;
      el.textContent = text;
      this.tooltip.appendChild(el);
    };
    const signed = (v: number, dp = 4) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(dp)}`;
    if ("layer" in c) {
      const dz = this.dOf(c.layer, c.head, "zero");
      const dm = this.dOf(c.layer, c.head, "mean");
      const cur = this.mode === "zero" ? dz : dm;
      add(
        "point-tooltip-label",
        `L${c.layer}H${c.head} — ${this.mode}-ablation Δ ${signed(cur)} nats ` +
          `(window ${b.meta.base_window.toFixed(4)} → ${(b.meta.base_window + cur).toFixed(4)})`,
      );
      add(
        "point-tooltip-conf",
        `zero ${signed(dz)} · mean ${signed(dm)} · induction score ${this.indOf(c.layer, c.head).toFixed(4)} (this run)`,
      );
      add("point-tooltip-conf", "click to draw its ghost curve");
    } else {
      const base = b.nll_base[c.j - 1] ?? 0;
      const inWin = c.j >= b.meta.window[0] && c.j <= b.meta.window[1];
      add(
        "point-tooltip-label",
        `j=${c.j} “${vis(b.token_strs[c.j] ?? "")}” — baseline NLL ${base.toFixed(4)}`,
      );
      if (this.sel) {
        const gv = this.selCurve(this.sel, this.mode)[c.j - 1] ?? 0;
        add(
          "point-tooltip-conf",
          `${this.selLabel(this.sel)} ${this.mode}-ablated: ${gv.toFixed(4)} (Δ ${signed(gv - base)})`,
        );
      }
      add(
        "point-tooltip-conf",
        `−log p(s_j | s_<j), nats · ${inWin ? "inside" : "outside"} the scored window`,
      );
    }
    this.tooltip.style.visibility = "visible";
    const rect = this.canvas.getBoundingClientRect();
    const px = Math.min(e.clientX - rect.left + 14, this.cssW - 330);
    const py = Math.min(e.clientY - rect.top + 14, this.cssH - 96);
    this.tooltip.style.transform = `translate(${px.toFixed(1)}px, ${py.toFixed(1)}px)`;
    this.canvas.style.cursor = "layer" in c ? "pointer" : "crosshair";
  }

  private onClick(e: PointerEvent): void {
    const c = this.pick(e);
    if (!c || !("layer" in c)) return;
    if (this.sel?.kind === "head" && this.sel.layer === c.layer && this.sel.head === c.head) return;
    this.sel = { kind: "head", layer: c.layer, head: c.head };
    this.layoutCurves();
    this.buildChips();
    this.pushLayers();
    this.positionLabels();
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
    // static — redraws on mode/selection switches only
  }

  resize(width: number, height: number, dpr: number): void {
    this.cssW = width;
    this.cssH = height;
    this.dpr = dpr;
    this.layout();
    this.hover = null;
    this.deck?.setProps({
      width,
      height,
      useDevicePixels: Math.min(dpr, 2),
      viewState: this.viewState(),
    });
    this.pushLayers();
    if (this.bundle) this.buildChips();
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

/** Visible-escape a token for labels: leading space → ␣, newline → ⏎. */
function vis(s: string): string {
  const t = s.replace(/\n/g, "⏎").replace(/^ /, "␣");
  return t || "·";
}
