/** #24 Co-Firing Venn — which SAE features fire together, counted on a corpus.
 *
 *  One point per feature PAIR: x = the pair's decoder-direction cosine
 *  (geometry), y = PMI = log2(observed/expected co-firing) computed in here
 *  from the bundle's exact integers (c, n_i, n_j, N — nothing pre-rounded
 *  enters the axes). Counts come from running the res-jb encoder over every
 *  token of a disclosed public-domain corpus; the census is exhaustive for
 *  c ≥ the stated support threshold. The horizontal PMI=0 guide is
 *  independence; a seeded shuffle (exported) lands on it. The right panel is
 *  an area-true Venn of the pinned pair: circle areas ∝ marginal counts,
 *  overlap lens area ∝ the exact joint count (center distance solved by
 *  bisection on the lens-area formula).
 *
 *  deck.gl (WebGL2), camera off, static. Source: cofire.json ⋈ sae.json. */

import type { Deck, OrthographicView, PickingInfo } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import type { InterpSelection } from "../../app/store";
import {
  type CofireBundle,
  type CofireChip,
  type SAEBundle,
  loadCofire,
  loadSAE,
} from "../../data/interp";
import { ACCENT, AXIS_RGBA, crosshair, dashedSegment, MARKER_HOT, withAlpha } from "./chart-theme";
import { InterpTooltip, type TipRow } from "./chart-tooltip";
import type { InterpDriver } from "./InterpDriver";
import { LAYER_COLORS } from "./NeuronFieldDriver";

type LayersModule = typeof import("@deck.gl/layers");

const GL = 50; // px — y-axis (PMI) tick labels
const GR = 16;
const GT = 96; // px — header summary
const GB = 92; // px — chips + collapsed legend pill
const VW = 240; // px — venn panel width (wide layouts)
const VH = 150; // px — venn band height (narrow layouts)

const AMBER: [number, number, number] = [245, 195, 59];
const CIRC_A: [number, number, number] = [96, 165, 250];
const CIRC_B: [number, number, number] = [244, 114, 182];

interface Pt {
  k: number; // pair index
}

interface Seg {
  source: [number, number];
  target: [number, number];
}

interface Poly {
  ring: number[][];
  fill: [number, number, number, number];
  line: [number, number, number, number];
}

/** unified detail record for a pinned pair (scatter pair or chip-only pair) */
interface SelData {
  i: number;
  j: number;
  c: number;
  e: number;
  lift: number;
  cos: number;
  tok: string | null;
  share: number;
  shuf: number | null;
}

export class CofireDriver implements InterpDriver {
  readonly animated = false;
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private canvas!: HTMLCanvasElement;
  private tooltip!: InterpTooltip;
  private labelRoot!: HTMLElement;
  private chipRoot!: HTMLElement;

  private cof: CofireBundle | null = null;
  private sae: SAEBundle | null = null;
  private pts: Pt[] = []; // deterministic shuffled draw order
  private px: Float32Array = new Float32Array(0);
  private pmi: Float32Array = new Float32Array(0);
  private marg = new Map<number, number>();
  private pairIdx = new Map<string, number>();
  private layoutGen = 0;
  private hover: number | null = null; // pair index
  private sel: { i: number; j: number } | null = null;
  /** feature id the global cross-view selection pinned a pair for */
  private linked: number | null = null;
  private xMin = -0.2;
  private xMax = 1;
  private yMin = -2;
  private yMax = 10;
  private cMax = 1;

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
    if (!this.cof || this.cof.meta.model !== model) {
      const [cof, sae] = await Promise.all([loadCofire(model), loadSAE(model)]);
      this.cof = cof;
      this.sae = sae;
      const n = cof.c.length;
      this.marg = new Map(cof.f_ids.map((f, q) => [f, cof.f_n[q] ?? 0]));
      this.pairIdx = new Map(cof.pi.map((a, k) => [`${a}:${cof.pj[k]}`, k]));
      // exact PMI from the exported integers — the y axis is computed here,
      // not shipped rounded
      this.pmi = new Float32Array(n);
      for (let k = 0; k < n; k++) {
        const ni = this.marg.get(cof.pi[k] ?? -1) ?? 1;
        const nj = this.marg.get(cof.pj[k] ?? -1) ?? 1;
        this.pmi[k] = Math.log2(((cof.c[k] ?? 1) * cof.N) / (ni * nj));
      }
      let x0 = 1;
      let x1 = -1;
      let y0 = Number.POSITIVE_INFINITY;
      let y1 = Number.NEGATIVE_INFINITY;
      for (let k = 0; k < n; k++) {
        const cs = cof.cos[k] ?? 0;
        if (cs < x0) x0 = cs;
        if (cs > x1) x1 = cs;
        const p = this.pmi[k] ?? 0;
        if (p < y0) y0 = p;
        if (p > y1) y1 = p;
      }
      this.xMin = Math.min(0, Math.floor(x0 * 10) / 10);
      this.xMax = Math.max(0.4, Math.ceil(x1 * 10) / 10);
      this.yMin = Math.min(-0.5, Math.floor(y0 * 2) / 2);
      this.yMax = Math.ceil(y1 * 2) / 2 + 0.25;
      this.cMax = cof.c[0] ?? 1; // pairs are sorted by c desc
      // deterministic draw-order shuffle (the #6/#2b z-bias lesson); stride
      // kept coprime with n so the permutation is complete
      let stride = 5323;
      while (gcd(stride, n) > 1) stride++;
      this.pts = Array.from({ length: n }, (_, k) => ({ k: (k * stride) % n }));
      // default pin: the strongest above-independence pair — the "these two
      // features are one concept split in half" story
      const c0 = cof.chips.assoc[0];
      this.sel = c0 ? { i: c0.i, j: c0.j } : null;
    }
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
    return this.narrow() ? 128 : GB;
  }
  private plotW(): number {
    return Math.max(40, this.cssW - GL - GR - (this.narrow() ? 0 : VW + 16));
  }
  private plotH(): number {
    return Math.max(
      40,
      this.cssH - GT - this.gb() - 18 - (this.narrow() ? VH : 0),
    );
  }
  /** venn panel rect: right column (wide) or a band below the scatter (narrow) */
  private vennRect(): { x0: number; y0: number; w: number; h: number } {
    return this.narrow()
      ? { x0: GL, y0: GT + this.plotH() + 24, w: this.cssW - GL - GR, h: VH - 24 }
      : { x0: this.cssW - GR - VW, y0: GT, w: VW, h: this.plotH() };
  }
  private xOf(cos: number): number {
    return GL + ((cos - this.xMin) / (this.xMax - this.xMin)) * this.plotW();
  }
  private yOf(p: number): number {
    return GT + (1 - (p - this.yMin) / (this.yMax - this.yMin)) * this.plotH();
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
    const cof = this.cof;
    if (!cof) return;
    const n = cof.c.length;
    this.px = new Float32Array(n * 2);
    for (let k = 0; k < n; k++) {
      this.px[k * 2] = this.xOf(cof.cos[k] ?? 0);
      this.px[k * 2 + 1] = this.yOf(this.pmi[k] ?? 0);
    }
    this.layoutGen++;
  }

  /** viridis via the shared 12-stop ramp (#6/#22) — t in [0,1] */
  private ramp(t: number): [number, number, number] {
    const s = Math.max(0, Math.min(1, t)) * (LAYER_COLORS.length - 1);
    const q = Math.min(LAYER_COLORS.length - 2, Math.floor(s));
    const f = s - q;
    const a = LAYER_COLORS[q] ?? [205, 210, 224];
    const b = LAYER_COLORS[q + 1] ?? [205, 210, 224];
    return [
      Math.round(a[0] + (b[0] - a[0]) * f),
      Math.round(a[1] + (b[1] - a[1]) * f),
      Math.round(a[2] + (b[2] - a[2]) * f),
    ];
  }
  private countT(c: number): number {
    const cof = this.cof;
    if (!cof) return 0;
    const lo = Math.log10(cof.meta.min_count);
    const hi = Math.log10(this.cMax);
    return hi > lo ? (Math.log10(c) - lo) / (hi - lo) : 0;
  }

  /** area-true venn geometry: areas ∝ exact counts; on narrow the circles sit
   *  left in the band so the label column right of them never covers them */
  private vennGeom(d: SelData): {
    cx: number;
    cy: number;
    r1: number;
    r2: number;
    dist: number;
  } {
    const R = this.vennRect();
    const ni = this.marg.get(d.i) ?? 1;
    const nj = this.marg.get(d.j) ?? 1;
    const narrow = this.narrow();
    const rMax = narrow ? Math.min(R.h * 0.38, R.w * 0.16) : Math.min(R.h * 0.34, R.w * 0.24);
    const s = (Math.PI * rMax * rMax) / Math.max(ni, nj);
    const r1 = Math.sqrt((s * ni) / Math.PI);
    const r2 = Math.sqrt((s * nj) / Math.PI);
    const dist = d.c > 0 ? vennDist(r1, r2, s * d.c) : r1 + r2 + 6;
    return {
      cx: R.x0 + R.w * (narrow ? 0.28 : 0.5),
      cy: R.y0 + R.h * (narrow ? 0.44 : 0.42),
      r1,
      r2,
      dist,
    };
  }

  private selData(): SelData | null {
    const cof = this.cof;
    const s = this.sel;
    if (!cof || !s) return null;
    const k = this.pairIdx.get(`${s.i}:${s.j}`);
    if (k != null) {
      const ni = this.marg.get(s.i) ?? 1;
      const nj = this.marg.get(s.j) ?? 1;
      const c = cof.c[k] ?? 0;
      return {
        i: s.i,
        j: s.j,
        c,
        e: (ni * nj) / cof.N,
        lift: (c * cof.N) / (ni * nj),
        cos: cof.cos[k] ?? 0,
        tok: cof.ctok_strs[cof.tt[k] ?? 0] ?? null,
        share: cof.tshare[k] ?? 0,
        shuf: null,
      };
    }
    for (const list of [cof.chips.avoid, cof.chips.assoc, cof.chips.count]) {
      const ch = list.find((x) => x.i === s.i && x.j === s.j);
      if (ch) {
        return { ...ch, e: ch.e, shuf: ch.shuf };
      }
    }
    return null;
  }

  private pushLayers(): void {
    if (!this.deck || !this.cof) return;
    const { ScatterplotLayer, LineLayer, PolygonLayer } = this.layersMod;
    const cof = this.cof;

    // guides: PMI = 0 (independence — where the shuffle lands) and cos = 0 —
    // dashed hairlines now (req 5), the structure whispers under the points
    const guides: Seg[] = [...dashedSegment([GL, this.yOf(0)], [GL + this.plotW(), this.yOf(0)])];
    if (this.xMin < 0) {
      guides.push(...dashedSegment([this.xOf(0), GT], [this.xOf(0), GT + this.plotH()]));
    }
    // ACCENT crosshair snapping onto the hovered pair (req 4)
    const cross: Seg[] =
      this.hover != null
        ? crosshair(this.px[this.hover * 2] ?? 0, this.px[this.hover * 2 + 1] ?? 0, {
            x0: GL,
            y0: GT,
            x1: GL + this.plotW(),
            y1: GT + this.plotH(),
          })
        : [];

    const rings: { k: number; hov: boolean }[] = [];
    if (this.sel) {
      const k = this.pairIdx.get(`${this.sel.i}:${this.sel.j}`);
      if (k != null) rings.push({ k, hov: false });
    }
    if (this.hover != null) rings.push({ k: this.hover, hov: true });

    // area-true venn for the pinned pair: areas ∝ n_i, n_j; lens ∝ c (exact)
    const polys: Poly[] = [];
    const d = this.selData();
    if (d) {
      const { cx, cy, r1, r2, dist } = this.vennGeom(d);
      polys.push(
        { ring: circle(cx - dist / 2, cy, r1), fill: [...CIRC_A, 46], line: [...CIRC_A, 220] },
        { ring: circle(cx + dist / 2, cy, r2), fill: [...CIRC_B, 46], line: [...CIRC_B, 220] },
      );
    }

    this.deck.setProps({
      layers: [
        new LineLayer<Seg>({
          id: "cofire-guides",
          data: guides,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: AXIS_RGBA,
          getWidth: 1,
          widthUnits: "pixels",
          updateTriggers: { getSourcePosition: this.layoutGen, getTargetPosition: this.layoutGen },
          pickable: false,
        }),
        new LineLayer<Seg>({
          id: "cofire-crosshair",
          data: cross,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: withAlpha(ACCENT, 0.5),
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        new ScatterplotLayer<Pt>({
          id: "cofire-pts",
          data: this.pts,
          // the point field dims to defer to the focused pair on hover (req 3)
          opacity: this.hover != null ? 0.4 : 1,
          getPosition: (p) => [this.px[p.k * 2] ?? 0, this.px[p.k * 2 + 1] ?? 0, 0],
          getFillColor: (p) => {
            const [r, g, b] = this.ramp(this.countT(cof.c[p.k] ?? 1));
            return [r, g, b, 150];
          },
          getRadius: this.narrow() ? 1.4 : 1.7,
          radiusUnits: "pixels",
          updateTriggers: { getPosition: this.layoutGen },
          pickable: true,
        }),
        new ScatterplotLayer<{ k: number; hov: boolean }>({
          id: "cofire-rings",
          data: rings,
          getPosition: (r) => [this.px[r.k * 2] ?? 0, this.px[r.k * 2 + 1] ?? 0, 0],
          getFillColor: [0, 0, 0, 0],
          getLineColor: (r) =>
            r.hov
              ? [MARKER_HOT[0], MARKER_HOT[1], MARKER_HOT[2], 230]
              : [AMBER[0], AMBER[1], AMBER[2], 230],
          getRadius: 5,
          radiusUnits: "pixels",
          stroked: true,
          filled: false,
          getLineWidth: 1.4,
          lineWidthUnits: "pixels",
          updateTriggers: { getPosition: this.layoutGen },
          pickable: false,
        }),
        new PolygonLayer<Poly>({
          id: "cofire-venn",
          data: polys,
          getPolygon: (p) => p.ring,
          getFillColor: (p) => p.fill,
          getLineColor: (p) => p.line,
          getLineWidth: 1.4,
          lineWidthUnits: "pixels",
          stroked: true,
          filled: true,
          updateTriggers: {
            getPolygon: `${this.layoutGen}:${this.sel?.i}:${this.sel?.j}`,
          },
          pickable: false,
        }),
      ],
    });
  }

  // ---- chips: exemplar pairs ------------------------------------------------
  private buildChips(): void {
    const cof = this.cof;
    if (!cof) return;
    this.chipRoot.textContent = "";
    this.chipRoot.style.bottom = this.narrow() ? "110px" : "";
    const narrow = this.narrow();
    // one row only (the #22 lesson): group label ≈ 38px, chip ≈ 112px wide /
    // 60px narrow — compute how many chips per group actually fit
    const per = Math.max(
      1,
      Math.min(3, Math.floor((this.cssW - 40 - 3 * 38) / (3 * (narrow ? 60 : 112)))),
    );
    const groups: Array<[string, string, CofireChip[]]> = [
      ["G²↑", "strongest above-independence pairs by Dunning's G² (deduped by feature)", cof.chips.assoc],
      ["co↑", "most co-firing pairs by raw count", cof.chips.count],
      ["avoid", "largest expected co-count with observed far below it", cof.chips.avoid],
    ];
    for (const [tag, tip, list] of groups) {
      const lab = document.createElement("span");
      lab.className = "interp-neuron-axis";
      lab.textContent = tag;
      lab.title = tip;
      // the axis class is position:absolute for plot labels — restore flow
      // inside the chip row or all three group tags pile up at the corner
      lab.style.position = "static";
      lab.style.alignSelf = "center";
      lab.style.marginRight = "2px";
      this.chipRoot.appendChild(lab);
      for (const ch of list.slice(0, per)) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "interp-neuron-chip";
        const tok = ch.tok != null ? vis(ch.tok) : "—";
        btn.textContent =
          tag === "avoid"
            ? narrow
              ? `${ch.lift.toFixed(1)}×`
              : `${ch.lift.toFixed(2)}× e${ch.e.toFixed(0)}`
            : narrow
              ? tag === "G²↑"
                ? `${ch.lift.toFixed(0)}×`
                : `${ch.c}×`
              : tag === "G²↑"
                ? `${ch.lift.toFixed(ch.lift >= 100 ? 0 : 1)}× “${tok}”`
                : `${ch.c}co “${tok}”`;
        btn.title =
          `#${ch.i} × #${ch.j} — co-fire ${ch.c} (expected ${ch.e}, lift ${ch.lift}×) · ` +
          `shuffled ${ch.shuf} · decoder cos ${ch.cos}` +
          (ch.tok != null ? ` · top co-token “${tok}” ${(ch.share * 100).toFixed(1)}%` : "");
        const active = this.sel?.i === ch.i && this.sel?.j === ch.j;
        btn.setAttribute("aria-pressed", String(active));
        if (active) btn.classList.add("is-active");
        btn.addEventListener("click", () => {
          this.sel = active ? null : { i: ch.i, j: ch.j };
          this.buildChips();
          this.pushLayers();
          this.positionLabels();
        });
        this.chipRoot.appendChild(btn);
      }
    }
  }

  // ---- labels -----------------------------------------------------------------
  private positionLabels(): void {
    this.labelRoot.textContent = "";
    const cof = this.cof;
    const sae = this.sae;
    if (!cof || !sae) return;
    const narrow = this.narrow();
    const meta = cof.meta;

    const cap = (text: string, cls = "interp-neuron-axis") => {
      const el = document.createElement("div");
      el.className = cls;
      el.textContent = text;
      el.style.overflow = "hidden";
      el.style.textOverflow = "ellipsis";
      el.style.whiteSpace = "nowrap";
      this.labelRoot.appendChild(el);
      return el;
    };
    const place = (el: HTMLElement, x: number, y: number) => {
      el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
    };

    // measured ~6.74 px/char (6.9 with margin) — pick the longest fitting
    // variant (the #22 lesson: never ellipsize the honesty tail)
    const fit = (s: string) => s.length * 6.9 + (narrow ? 24 : GL + 12) <= this.cssW;
    const pick3 = (full: string, mid: string, short: string) =>
      fit(full) ? full : fit(mid) ? mid : short;
    const hMax = `${(this.cssW - (narrow ? 12 : GL) - 12).toFixed(0)}px`;

    const h1 = cap(
      pick3(
        `co-firing venn · top ${meta.n_pairs.toLocaleString()} of ${meta.n_support.toLocaleString()} ` +
          `pairs (c ≥ ${meta.min_count}) by G² · Alice in Wonderland, ${cof.N.toLocaleString()} positions · ` +
          `L0 mean ${meta.l0_mean}`,
        `co-firing · top ${meta.n_pairs.toLocaleString()} of ${meta.n_support.toLocaleString()} pairs by G² · ` +
          `${cof.N.toLocaleString()} positions`,
        `co-firing · top ${meta.n_pairs.toLocaleString()} pairs by G²`,
      ),
    );
    h1.style.color = "rgb(245,195,59)";
    h1.style.maxWidth = hMax;
    place(h1, narrow ? 12 : GL, GT - 44);
    const h2 = cap(
      pick3(
        `y = PMI = log₂(observed/expected) from exact counts · x = decoder cos · ` +
          `ALL-support-pairs r(cos,PMI) = ${meta.pearson_cos_pmi} · shuffle ratio ${meta.shuffle.agg_ratio} · ` +
          `recon cos ${meta.recon_cos_mean}`,
        `y = PMI (exact counts) · x = decoder cos · global r ${meta.pearson_cos_pmi} · shuffle ${meta.shuffle.agg_ratio}`,
        `y = PMI · x = decoder cos · r ${meta.pearson_cos_pmi}`,
      ),
    );
    h2.style.maxWidth = hMax;
    place(h2, narrow ? 12 : GL, GT - 30);
    const d = this.selData();
    if (d) {
      const ri = vis(sae.top_tok[d.i] ?? "");
      const rj = vis(sae.top_tok[d.j] ?? "");
      const h3 = cap(
        pick3(
          `#${d.i} ↑“${ri}” × #${d.j} ↑“${rj}” · co-fire ${d.c} · expected ${d.e.toFixed(1)} · ` +
            `lift ${d.lift.toFixed(d.lift >= 100 ? 0 : 1)}× · cos ${d.cos.toFixed(4)}` +
            (d.tok != null ? ` · top “${vis(d.tok)}” ${(d.share * 100).toFixed(0)}%` : ""),
          `#${d.i} × #${d.j} · c ${d.c} · e ${d.e.toFixed(1)} · lift ${d.lift.toFixed(1)}× · cos ${d.cos.toFixed(3)}`,
          `#${d.i}×#${d.j} · ${d.c} vs e${d.e.toFixed(0)}`,
        ),
      );
      h3.style.color = "rgb(245,195,59)";
      h3.style.maxWidth = hMax;
      place(h3, narrow ? 12 : GL, GT - 16);
    }

    // y ticks: integer PMI steps sized to the range
    const step = Math.max(1, Math.ceil((this.yMax - this.yMin) / 7));
    for (let v = Math.ceil(this.yMin); v <= this.yMax; v += step) {
      place(cap(v > 0 ? `+${v}` : `${v}`), Math.max(2, GL - 28), this.yOf(v) - 7);
    }
    const yTick = GT + this.plotH() + 4;
    for (let v = Math.round(this.xMin * 10); v <= this.xMax * 10 + 0.01; v += 2) {
      const cx = v / 10;
      place(cap(cx.toFixed(1).replace(/^(-?)0\./, "$1.")), this.xOf(cx) - 6, yTick);
    }

    // guide labels: PMI 0 = independence. Right end of the guide sits under
    // the venn panel column on wide, so the label never covers scatter data.
    const gIs = narrow ? "independence" : "PMI 0 = independence (shuffle lands here)";
    const gI = cap(gIs);
    gI.style.color = "rgb(118,126,158)";
    place(gI, GL + this.plotW() - gIs.length * 6.9, this.yOf(0) + 3);

    // venn panel labels — the panel is driver-owned space, always safe.
    // wide: labels above/below the circles; narrow: a text column to the right
    if (d) {
      const R = this.vennRect();
      const ni = this.marg.get(d.i) ?? 1;
      const nj = this.marg.get(d.j) ?? 1;
      // compact-but-exact on narrow: the label column is ~148px wide
      const cline = narrow
        ? `∩ ${d.c} · e ${d.e.toFixed(1)}`
        : `overlap ${d.c} — expected ${d.e.toFixed(1)}`;
      const l4s = narrow
        ? "areas ∝ counts"
        : `areas ∝ counts (exact)` + (d.shuf != null ? ` · shuffled: ${d.shuf}` : "");
      const la = cap(`#${d.i} fires ${ni}×`);
      la.style.color = `rgb(${CIRC_A[0]},${CIRC_A[1]},${CIRC_A[2]})`;
      const lb = cap(`#${d.j} fires ${nj}×`);
      lb.style.color = `rgb(${CIRC_B[0]},${CIRC_B[1]},${CIRC_B[2]})`;
      const lc = cap(cline);
      lc.style.color = "rgb(245,195,59)";
      const l4 = cap(l4s);
      if (narrow) {
        const tx = R.x0 + R.w * 0.52;
        const { cy } = this.vennGeom(d);
        place(la, tx, cy - 32);
        place(lb, tx, cy - 16);
        place(lc, tx, cy);
        if (d.shuf != null) {
          const ls = cap(`shuffled ${d.shuf}`);
          place(ls, tx, cy + 16);
        }
        place(l4, tx, cy + 32);
      } else {
        // two rows at top-left — right-anchoring the second label collides
        // with the first when both marginals are 4-5 digits
        place(la, R.x0 + 4, R.y0 + 2);
        place(lb, R.x0 + 4, R.y0 + 16);
        place(lc, R.x0 + Math.max(0, (R.w - cline.length * 6.9) / 2), R.y0 + R.h - 30);
        place(l4, R.x0 + Math.max(0, (R.w - l4s.length * 6.9) / 2), R.y0 + R.h - 16);
      }
    }

    // count-color legend — inside the venn panel (driver-owned, never on data)
    const R = this.vennRect();
    const leg = document.createElement("div");
    leg.className = "interp-neuron-axis";
    leg.style.display = "flex";
    leg.style.alignItems = "center";
    leg.style.gap = "3px";
    const mid = Math.round(Math.sqrt(cof.meta.min_count * this.cMax));
    for (const cv of [cof.meta.min_count, mid, this.cMax]) {
      const sw = document.createElement("span");
      const [r, g, b] = this.ramp(this.countT(cv));
      sw.style.width = "9px";
      sw.style.height = "9px";
      sw.style.borderRadius = "2px";
      sw.style.background = `rgb(${r},${g},${b})`;
      sw.title = `co-count ${cv}`;
      leg.appendChild(sw);
      const t = document.createElement("span");
      t.textContent = String(cv);
      t.style.marginRight = "3px";
      leg.appendChild(t);
    }
    const legTxt = document.createElement("span");
    legTxt.textContent = "co-count (log)"; // fits the 240px panel
    legTxt.style.marginLeft = "2px";
    leg.appendChild(legTxt);
    this.labelRoot.appendChild(leg);
    place(leg, R.x0 + 4, narrow ? R.y0 + R.h - 12 : R.y0 + R.h - 54);
  }

  // ---- interaction --------------------------------------------------------------
  private pick(e: PointerEvent): number | null {
    if (!this.deck) return null;
    const rect = this.canvas.getBoundingClientRect();
    const info = this.deck.pickObject({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      radius: 4,
      layerIds: ["cofire-pts"],
    }) as PickingInfo | null;
    const p = info?.object as Pt | undefined;
    return p ? p.k : null;
  }

  private onPointerMove(e: PointerEvent): void {
    const cof = this.cof;
    const sae = this.sae;
    if (!cof || !sae) return;
    const k = this.pick(e);
    if (k !== this.hover) {
      this.hover = k;
      this.pushLayers();
    }
    if (k == null) {
      this.tooltip.hide();
      this.canvas.style.cursor = "";
      return;
    }
    const i = cof.pi[k] ?? 0;
    const j = cof.pj[k] ?? 0;
    const ni = this.marg.get(i) ?? 1;
    const nj = this.marg.get(j) ?? 1;
    const c = cof.c[k] ?? 0;
    const ev = (ni * nj) / cof.N;
    const lift = (c * cof.N) / (ni * nj);
    // swatch = the exact count-ramp color this point was drawn with
    const [sr, sg, sb] = this.ramp(this.countT(cof.c[k] ?? 1));
    const rows: TipRow[] = [
      {
        kind: "label",
        text: `#${i} × #${j} — co-fire ${c}× (expected ${ev.toFixed(1)}, lift ${lift.toFixed(lift >= 100 ? 0 : 1)}×)`,
        swatch: [sr, sg, sb],
      },
      {
        text: "PMI",
        value: `${(this.pmi[k] ?? 0) >= 0 ? "+" : ""}${(this.pmi[k] ?? 0).toFixed(2)}`,
        hot: true,
      },
      {
        text: `G² ${g2Of(c, ni, nj, cof.N).toFixed(0)} · decoder cos ${(cof.cos[k] ?? 0).toFixed(4)}`,
      },
      {
        text: `top co-token “${vis(cof.ctok_strs[cof.tt[k] ?? 0] ?? "")}” — ${((cof.tshare[k] ?? 0) * 100).toFixed(1)}% of co-firings`,
      },
      { text: `#${i} ↑“${vis(sae.top_tok[i] ?? "")}” · fires ${ni}×` },
      { text: `#${j} ↑“${vis(sae.top_tok[j] ?? "")}” · fires ${nj}× · click to pin` },
    ];
    this.tooltip.show(rows);
    const rect = this.canvas.getBoundingClientRect();
    this.tooltip.move(e.clientX - rect.left, e.clientY - rect.top, this.cssW, this.cssH);
    this.canvas.style.cursor = "pointer";
  }

  private onClick(e: PointerEvent): void {
    const cof = this.cof;
    if (!cof) return;
    const k = this.pick(e);
    if (k == null) return;
    const i = cof.pi[k] ?? 0;
    const j = cof.pj[k] ?? 0;
    this.sel = this.sel?.i === i && this.sel?.j === j ? null : { i, j };
    this.buildChips();
    this.pushLayers();
    this.positionLabels();
  }

  /** Cross-view link (follow-only): pin the first exported co-firing pair
   *  containing the selected feature. A pair click stays local — publishing
   *  ONE of its two features would be an arbitrary choice. */
  setSelection(sel: InterpSelection | null): void {
    const cof = this.cof;
    if (!cof) return;
    const id = sel?.kind === "saeFeature" ? sel.id : null;
    if (id === this.linked) return;
    const prev = this.linked;
    this.linked = id;
    if (id !== null) {
      const k = cof.pi.findIndex((a, m) => a === id || cof.pj[m] === id);
      if (k < 0) return; // feature not in any exported pair — nothing to show
      const i = cof.pi[k] ?? 0;
      const j = cof.pj[k] ?? 0;
      if (this.sel?.i === i && this.sel?.j === j) return;
      this.sel = { i, j };
    } else if (prev !== null && (this.sel?.i === prev || this.sel?.j === prev)) {
      // only clear a pin the link itself created/mirrors
      this.sel = null;
    } else {
      return;
    }
    this.buildChips();
    this.pushLayers();
    this.positionLabels();
  }

  private onLeave(): void {
    if (this.hover != null) {
      this.hover = null;
      this.pushLayers();
    }
    this.tooltip.hide();
    this.canvas.style.cursor = "";
  }

  frame(_dt: number, _t: number): void {
    // static — redraws on hover/selection/resize only
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
    if (this.cof) this.buildChips();
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

function gcd(a: number, b: number): number {
  return b === 0 ? a : gcd(b, a % b);
}

/** Dunning's G² over the full 2×2 table — exact from the integer counts */
function g2Of(c: number, ni: number, nj: number, N: number): number {
  const cells: Array<[number, number, number]> = [
    [c, ni, nj],
    [ni - c, ni, N - nj],
    [nj - c, N - ni, nj],
    [N - ni - nj + c, N - ni, N - nj],
  ];
  let s = 0;
  for (const [k, row, col] of cells) {
    if (k > 0) s += k * Math.log((k * N) / (row * col));
  }
  return 2 * s;
}

/** 97-point polyline circle for PolygonLayer */
function circle(cx: number, cy: number, r: number): number[][] {
  return Array.from({ length: 97 }, (_, q) => {
    const a = (q / 96) * Math.PI * 2;
    return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
  });
}

/** area of the lens where two circles of radii r1, r2 at center distance d overlap */
function lensArea(r1: number, r2: number, d: number): number {
  if (d <= Math.abs(r1 - r2)) return Math.PI * Math.min(r1, r2) ** 2;
  if (d >= r1 + r2) return 0;
  const a1 = r1 * r1 * Math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1));
  const a2 = r2 * r2 * Math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2));
  const t =
    0.5 * Math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2));
  return a1 + a2 - t;
}

/** center distance so the lens area equals target — bisection (lens ↓ in d) */
function vennDist(r1: number, r2: number, target: number): number {
  const full = Math.PI * Math.min(r1, r2) ** 2;
  if (target <= 0) return r1 + r2;
  if (target >= full) return Math.abs(r1 - r2);
  let lo = Math.abs(r1 - r2);
  let hi = r1 + r2;
  for (let it = 0; it < 80; it++) {
    const mid = (lo + hi) / 2;
    if (lensArea(r1, r2, mid) > target) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

/** Visible-escape a token for labels: leading space → ␣, newline → ⏎, other
 *  C0 controls → their Unicode control pictures. */
function vis(s: string): string {
  const t = s
    .replace(/\n/g, "⏎")
    // whole leading run, not just the first space — multi-space code-indent
    // tokens otherwise render as an invisible gap between quotes
    .replace(/^ +/, (m) => "␣".repeat(m.length))
    .replace(/[\x00-\x09\x0b-\x1f\x7f]/g, (c) =>
      String.fromCodePoint(0x2400 + (c === "\x7f" ? 0x21 : c.charCodeAt(0))),
    );
  return t || "·";
}
