/** #2d Induction Microscope — behavioral proof of induction heads.
 *
 *  The published diagnostic (Olsson et al. 2022): feed <|endoftext|> + 48
 *  uniform-random tokens repeated twice. On the second repeat every token has
 *  occurred exactly once before, so a head implementing induction ("[A][B] …
 *  [A] → attend to [B]") must attend from position t to t−period+1. The grid
 *  scores all 144 heads on three structural targets (induction / duplicate /
 *  prev-token); clicking a head opens its REAL T×T attention pattern, where
 *  the induction stripe is visible as the off-diagonal at exactly −47.
 *
 *  This closes the loop #2c opened: K-composition predicted induction from
 *  weights alone (L4H11→L5H1/L5H5); this view measures it from behavior.
 *
 *  deck.gl (WebGL2), camera off, static. Source: induction.json. */

import type { Deck, OrthographicView, PickingInfo } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { appStore, type InterpSelection } from "../../app/store";
import { type InductionBundle, loadInduction } from "../../data/interp";
import { AXIS_RGBA, dashedSegment, HOT, MARKER_HOT, type RGB, withAlpha } from "./chart-theme";
import { InterpTooltip, type TipRow } from "./chart-tooltip";
import type { InterpDriver } from "./InterpDriver";

type LayersModule = typeof import("@deck.gl/layers");

const GL = 40; // px — grid row labels (L0…L11)
const GR = 14;
const GT = 96; // px — header
const GB = 92; // px — chip strip + collapsed legend pill

const LOW: [number, number, number] = [64, 66, 96];
const AMBER: [number, number, number] = [245, 195, 59];
const ZERO_RGBA: [number, number, number, number] = [118, 126, 158, 16];

type Metric = "ind" | "dup" | "prev";
const METRICS: Metric[] = ["ind", "dup", "prev"];

interface GridCell {
  poly: [number, number][];
  layer: number;
  head: number;
}

interface StripeCell {
  poly: [number, number][];
  from: number;
  to: number;
  v: number; // stored post-softmax attention, 4 dp
}

interface Seg {
  source: [number, number];
  target: [number, number];
}

export class InductionDriver implements InterpDriver {
  readonly animated = false;
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private canvas!: HTMLCanvasElement;
  private tooltip!: InterpTooltip;
  private labelRoot!: HTMLElement;
  private chipRoot!: HTMLElement;

  private bundle: InductionBundle | null = null;
  private metric: Metric = "ind";
  private sel: { layer: number; head: number } | null = null;
  private gridCells: GridCell[] = [];
  private stripeCells: StripeCell[] = [];
  private hover: GridCell | StripeCell | null = null;

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
      this.bundle = await loadInduction(model);
    }
    const top = this.bundle.patterns[0];
    this.sel = top ? { layer: top.layer, head: top.head } : null;
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
    // narrow stages keep BOTH panels beside each other (stacking starved both:
    // a 375×450 stage left 5px grid cells and a stripe under the chip strip) —
    // grid gets ~42% of the width, stripe the rest
    return this.narrow()
      ? Math.min(10, (0.42 * (this.cssW - GL - GR)) / H, this.plotH() / nL)
      : Math.min(36, (0.38 * (this.cssW - GL - GR)) / H, this.plotH() / nL);
  }
  /** stripe origin + cell size — always right of the grid; gutter shrinks narrow */
  private stripeBox(): { x0: number; y0: number; s: number } {
    const b = this.bundle;
    const T = b?.meta.T ?? 97;
    const g = this.gcell();
    const H = b?.meta.n_head ?? 12;
    const x0 = GL + H * g + (this.narrow() ? 30 : 64);
    const s = Math.max(1, Math.min(8, (this.cssW - x0 - GR) / T, this.plotH() / T));
    return { x0, y0: GT, s };
  }

  private viewState() {
    return {
      ortho: {
        target: [this.cssW / 2, this.cssH / 2, 0] as [number, number, number],
        zoom: 0,
      },
    };
  }

  private scoreOf(layer: number, head: number, metric: Metric, seedB = false): number {
    const b = this.bundle;
    if (!b) return 0;
    const arr = b[seedB ? (`${metric}_b` as const) : metric];
    return arr[layer * b.meta.n_head + head] ?? 0;
  }
  private metricMax(metric: Metric): { v: number; layer: number; head: number } {
    const b = this.bundle;
    if (!b) return { v: 0, layer: 0, head: 0 };
    let best = 0;
    let at = 0;
    b[metric].forEach((v, i) => {
      if (v > best) {
        best = v;
        at = i;
      }
    });
    return { v: best, layer: Math.floor(at / b.meta.n_head), head: at % b.meta.n_head };
  }
  private metricDrift(metric: Metric): number {
    const b = this.bundle;
    if (!b) return 0;
    const bb = b[`${metric}_b` as const];
    return b[metric].reduce((mx, v, i) => Math.max(mx, Math.abs(v - (bb[i] ?? 0))), 0);
  }
  /** target offset (key = query − offset) for each structural score */
  private offsetOf(metric: Metric): number {
    const P = this.bundle?.meta.period ?? 48;
    return metric === "ind" ? P - 1 : metric === "dup" ? P : 1;
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
    this.layoutStripe();
  }

  private layoutStripe(): void {
    const b = this.bundle;
    const sel = this.sel;
    this.stripeCells = [];
    if (!b || !sel) return;
    const pat = b.patterns.find((p) => p.layer === sel.layer && p.head === sel.head);
    if (!pat) return; // no exported pattern — positionLabels states why
    const T = b.meta.T;
    const { x0, y0, s } = this.stripeBox();
    const cells: StripeCell[] = [];
    for (let f = 0; f < T; f++) {
      for (let t = 0; t <= f; t++) {
        // causal: keys ≤ query only (upper triangle is exactly 0, verified)
        cells.push({
          poly: [
            [x0 + t * s, y0 + f * s],
            [x0 + (t + 1) * s, y0 + f * s],
            [x0 + (t + 1) * s, y0 + (f + 1) * s],
            [x0 + t * s, y0 + (f + 1) * s],
          ],
          from: f,
          to: t,
          v: pat.attn[f * T + t] ?? 0,
        });
      }
    }
    this.stripeCells = cells;
  }

  private gridColor(layer: number, head: number): [number, number, number, number] {
    const b = this.bundle;
    if (!b) return ZERO_RGBA;
    const v = this.scoreOf(layer, head, this.metric);
    const floor = b.meta.floor;
    if (v <= floor) return ZERO_RGBA; // at/below uniform-attention chance
    const max = this.metricMax(this.metric).v;
    const t = Math.min(1, (v - floor) / Math.max(1e-9, max - floor));
    return [
      Math.round(LOW[0] + t * (AMBER[0] - LOW[0])),
      Math.round(LOW[1] + t * (AMBER[1] - LOW[1])),
      Math.round(LOW[2] + t * (AMBER[2] - LOW[2])),
      Math.round(70 + 185 * t),
    ];
  }

  private pushLayers(): void {
    if (!this.deck || !this.bundle) return;
    const { SolidPolygonLayer, LineLayer } = this.layersMod;
    const b = this.bundle;
    const hover = this.hover;
    const hoverEdges: Seg[] = hover
      ? hover.poly.map((p, i) => ({
          source: p as [number, number],
          target: hover.poly[(i + 1) % hover.poly.length] as [number, number],
        }))
      : [];
    const g = this.gcell();
    const selEdges: Seg[] = [];
    if (this.sel) {
      const x0 = GL + this.sel.head * g;
      const y0 = GT + this.sel.layer * g;
      selEdges.push(
        { source: [x0, y0], target: [x0 + g, y0] },
        { source: [x0 + g, y0], target: [x0 + g, y0 + g] },
        { source: [x0 + g, y0 + g], target: [x0, y0 + g] },
        { source: [x0, y0 + g], target: [x0, y0] },
      );
    }
    // layout guide: where the second repeat begins (row/col period+1) —
    // DASHED hairlines now (req 5) so the structure whispers under the stripe
    const guides: Seg[] = [];
    if (this.stripeCells.length) {
      const { x0, y0, s } = this.stripeBox();
      const T = b.meta.T;
      const at = (b.meta.period + 1) * s;
      guides.push(
        ...dashedSegment([x0 + at, y0], [x0 + at, y0 + T * s]),
        ...dashedSegment([x0, y0 + at], [x0 + T * s, y0 + at]),
      );
    }

    this.deck.setProps({
      layers: [
        new SolidPolygonLayer<GridCell>({
          id: "imic-grid",
          data: this.gridCells,
          getPolygon: (c) => c.poly,
          getFillColor: (c) => this.gridColor(c.layer, c.head),
          updateTriggers: { getFillColor: this.metric },
          pickable: true,
        }),
        new SolidPolygonLayer<StripeCell>({
          id: "imic-stripe",
          data: this.stripeCells,
          getPolygon: (c) => c.poly,
          // linear alpha in the stored attention — no gamma, no smoothing
          getFillColor: (c) => [AMBER[0], AMBER[1], AMBER[2], Math.round(255 * c.v)],
          pickable: true,
        }),
        new LineLayer<Seg>({
          id: "imic-guides",
          data: guides,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: AXIS_RGBA,
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        new LineLayer<Seg>({
          id: "imic-sel",
          data: selEdges,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: withAlpha(HOT, 230 / 255),
          getWidth: 1.4,
          widthUnits: "pixels",
          pickable: false,
        }),
        new LineLayer<Seg>({
          id: "imic-hover",
          data: hoverEdges,
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

  // ---- chips: which structural target colors the grid ------------------------
  private metricLabel(metric: Metric, long: boolean): string {
    const off = this.offsetOf(metric);
    const base = metric === "ind" ? "induction" : metric === "dup" ? "duplicate" : "prev token";
    return long ? `${base} (t−${off})` : base;
  }

  private buildChips(): void {
    const b = this.bundle;
    if (!b) return;
    this.chipRoot.textContent = "";
    this.chipRoot.style.bottom = this.narrow() ? "110px" : "";
    for (const metric of METRICS) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "interp-neuron-chip";
      btn.textContent = this.metricLabel(metric, !this.narrow());
      btn.title = `color the head grid by mean attention to the ${this.metricLabel(metric, true)} target`;
      btn.setAttribute("aria-pressed", String(metric === this.metric));
      if (metric === this.metric) btn.classList.add("is-active");
      btn.addEventListener("click", () => {
        if (metric === this.metric) return;
        this.metric = metric;
        this.hover = null;
        this.tooltip.hide();
        this.buildChips();
        this.pushLayers();
        this.positionLabels();
      });
      this.chipRoot.appendChild(btn);
    }
  }

  // ---- labels -----------------------------------------------------------------
  private positionLabels(): void {
    this.labelRoot.textContent = "";
    const b = this.bundle;
    if (!b) return;
    const narrow = this.narrow();
    const g = this.gcell();
    const P = b.meta.period;
    const T = b.meta.T;

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

    const top = this.metricMax(this.metric);
    const drift = this.metricDrift(this.metric);
    const mLabel = this.metricLabel(this.metric, true);
    // header copy compresses below 1000px so the honesty tail (floor, seed-B
    // drift, forward count) is never the part an ellipsis eats
    const compact = this.cssW < 1000;
    const hMax = `${(this.cssW - (narrow ? 12 : GL) - 12).toFixed(0)}px`;
    const h1 = cap(
      narrow
        ? `${P} rand ×2 · top ${this.metric} L${top.layer}H${top.head} ${top.v.toFixed(2)} (${(top.v / b.meta.floor).toFixed(0)}× floor)`
        : compact
          ? `induction test · ${P} rand ×2 (seed ${b.meta.seed_a}) · top ${mLabel.split(" ")[0]} ` +
              `L${top.layer}H${top.head} ${top.v.toFixed(2)} = ${(top.v / b.meta.floor).toFixed(0)}× floor`
          : `induction microscope · <|endoftext|> + ${P} random tokens ×2 (seed ${b.meta.seed_a}) · ` +
              `top ${mLabel} L${top.layer}H${top.head} ${top.v.toFixed(4)} = ${(top.v / b.meta.floor).toFixed(1)}× floor`,
    );
    h1.style.color = "rgb(245,195,59)";
    h1.style.maxWidth = hMax;
    place(h1, narrow ? 12 : GL, GT - 44);
    const h2 = cap(
      narrow
        ? `attn[t,t−${this.offsetOf(this.metric)}] · floor ${b.meta.floor} · seedΔ≤${drift.toFixed(3)} · 2 fwd`
        : compact
          ? `mean attn[t, t−${this.offsetOf(this.metric)}] · floor ${b.meta.floor} · seed ${b.meta.seed_b} Δ≤${drift.toFixed(3)} · 2 fwd`
          : `score = mean attn[t, t−${this.offsetOf(this.metric)}] over 2nd repeat · floor ${b.meta.floor} = uniform · ` +
              `seed ${b.meta.seed_b} re-run max |Δ| ${drift.toFixed(4)} (hover shows both) · 2 real forwards`,
    );
    h2.style.maxWidth = hMax;
    place(h2, narrow ? 12 : GL, GT - 30);

    // grid axes — thin by cell size so labels never overlap into mush
    const step = g >= 15 ? 1 : g >= 12 ? 2 : 3;
    for (let h = 0; h < b.meta.n_head; h += step) {
      place(cap(`H${h}`), GL + h * g + 2, GT - 15);
    }
    for (let l = 0; l < b.meta.n_layer; l += step) {
      place(cap(`L${l}`), 10, GT + l * g + g / 2 - 8);
    }
    // in-cell scores where they fit and matter
    if (g >= 30) {
      for (const c of this.gridCells) {
        const v = this.scoreOf(c.layer, c.head, this.metric);
        if (v < 0.3) continue;
        const el = cap(v.toFixed(2));
        el.style.color = "rgba(20,22,34,0.95)";
        place(el, GL + c.head * g + g / 2 - 11, GT + c.layer * g + g / 2 - 8);
      }
    }

    // stripe panel
    const { x0, y0, s } = this.stripeBox();
    const sel = this.sel;
    if (sel && this.stripeCells.length) {
      const st = cap(
        narrow
          ? `L${sel.layer}H${sel.head} · from↓ to→`
          : this.cssW < 1000
            ? `L${sel.layer}H${sel.head} pattern · row=from, col=to`
            : `L${sel.layer}H${sel.head} pattern (seed ${b.meta.seed_a}) · row=from, col=to · alpha ∝ attn, linear`,
        "interp-neuron-axis",
      );
      st.style.color = "rgb(245,195,59)";
      st.style.maxWidth = `${(this.cssW - x0 - GR).toFixed(0)}px`;
      place(st, x0, y0 - 15);
      // row ticks on the left; col ticks for 49/96 + the boundary caption live
      // INSIDE the upper triangle — causally guaranteed empty, and the bottom
      // band (where they'd otherwise sit) is where the collapsed pill docks
      for (const tick of [0, P + 1, T - 1]) {
        place(cap(String(tick)), Math.max(2, x0 - 26), y0 + tick * s);
      }
      place(cap("0"), x0 + 2, y0 + T * s + 2);
      place(cap(String(P + 1)), x0 + (P + 1) * s + 4, y0 + 2);
      place(cap(String(T - 1)), x0 + (T - 1) * s - 16, y0 + 2);
      if (!narrow) {
        // full caption only when it actually fits between its anchor and the
        // T−1 tick at the pattern's right edge (≈6.9 px/char, mono)
        const capX = x0 + (P + 1) * s + 30;
        const full = `← col ${P + 1}, where the 2nd repeat begins`;
        const fits = capX + full.length * 6.9 < x0 + (T - 1) * s - 24;
        place(cap(!compact && fits ? full : "← 2nd repeat"), capX, y0 + 2);
      }
    } else if (sel) {
      const note = cap(
        `no exported pattern for L${sel.layer}H${sel.head} — full T×T patterns ship only for the ` +
          `top-${b.patterns.length} heads by score (144 would be ~8 MB). The grid score is still exact.`,
      );
      note.style.maxWidth = `${Math.min(260, this.cssW - x0 - 20).toFixed(0)}px`;
      note.style.whiteSpace = "normal";
      place(note, x0 + 8, y0 + 20);
    }
  }

  // ---- interaction --------------------------------------------------------------
  private pick(e: PointerEvent): GridCell | StripeCell | null {
    if (!this.deck) return null;
    const rect = this.canvas.getBoundingClientRect();
    const info = this.deck.pickObject({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      radius: 1,
      layerIds: ["imic-grid", "imic-stripe"],
    }) as PickingInfo | null;
    return (info?.object as GridCell | StripeCell | undefined) ?? null;
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
      this.tooltip.hide();
      this.canvas.style.cursor = "";
      return;
    }
    const rows: TipRow[] = [];
    if ("layer" in c) {
      const v = this.scoreOf(c.layer, c.head, this.metric);
      const gc = this.gridColor(c.layer, c.head);
      const swatch: RGB = [gc[0], gc[1], gc[2]];
      rows.push({
        kind: "label",
        text: `L${c.layer}H${c.head} — ${this.metricLabel(this.metric, true)} ${v.toFixed(4)} (${(v / b.meta.floor).toFixed(1)}× floor)`,
        swatch,
      });
      rows.push({
        text:
          `seed ${b.meta.seed_b}: ${this.scoreOf(c.layer, c.head, this.metric, true).toFixed(4)} · ` +
          `ind ${this.scoreOf(c.layer, c.head, "ind").toFixed(4)} · dup ${this.scoreOf(c.layer, c.head, "dup").toFixed(4)} · ` +
          `prev ${this.scoreOf(c.layer, c.head, "prev").toFixed(4)}`,
      });
      rows.push({
        text: b.patterns.some((p) => p.layer === c.layer && p.head === c.head)
          ? "click to inspect the full attention pattern"
          : `pattern not exported (top-${b.patterns.length} by score only) — click shows why`,
      });
    } else {
      const P = b.meta.period;
      rows.push({
        kind: "label",
        text: `attn ${c.v.toFixed(4)} — from ${c.from} “${vis(b.token_strs[c.from] ?? "")}” → ${c.to} “${vis(b.token_strs[c.to] ?? "")}”`,
        swatch: AMBER,
      });
      const tag =
        c.to === c.from - (P - 1)
          ? `induction target (t−${P - 1}): the token AFTER this one's previous occurrence`
          : c.to === c.from - P
            ? `duplicate target (t−${P}): this token's previous occurrence`
            : c.to === c.from - 1
              ? "previous token"
              : c.to === c.from
                ? "self"
                : c.to === 0
                  ? "<|endoftext|> — the attention sink"
                  : "off-target";
      rows.push({ text: tag });
      rows.push({ text: `post-softmax attention, stored at 4 dp · seed ${b.meta.seed_a}` });
    }
    this.tooltip.show(rows);
    const rect = this.canvas.getBoundingClientRect();
    this.tooltip.move(e.clientX - rect.left, e.clientY - rect.top, this.cssW, this.cssH);
    this.canvas.style.cursor = "layer" in c ? "pointer" : "crosshair";
  }

  private onClick(e: PointerEvent): void {
    const c = this.pick(e);
    if (!c || !("layer" in c)) return;
    if (this.sel && this.sel.layer === c.layer && this.sel.head === c.head) return;
    this.selectHead(c.layer, c.head);
    // publish the inspected head as the global cross-view selection
    appStore.getState().setInterpSelection({ kind: "head", layer: c.layer, head: c.head });
  }

  private selectHead(layer: number, head: number): void {
    this.sel = { layer, head };
    this.layoutStripe();
    this.pushLayers();
    this.positionLabels();
  }

  /** Cross-view link: follow a global head selection by inspecting that head.
   *  A cleared selection keeps the current head — this view always shows one. */
  setSelection(sel: InterpSelection | null): void {
    const b = this.bundle;
    if (!b || sel?.kind !== "head") return;
    if (sel.layer >= b.meta.n_layer || sel.head >= b.meta.n_head) return;
    if (this.sel && this.sel.layer === sel.layer && this.sel.head === sel.head) return;
    this.selectHead(sel.layer, sel.head);
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
    // static — redraws on metric/head switches only
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
    this.tooltip?.dispose();
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
