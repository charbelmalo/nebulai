/** #16 Grokking Clock — delayed generalization, trained from scratch.
 *
 *  A toy MLP (2·97 → 128 → 97, φ(z)=z², full-batch AdamW) learns
 *  c = (a+b) mod 97. The run is checkpointed every 100 steps and three real
 *  quantities are drawn: (1) train/test accuracy — train hits 100% thousands
 *  of steps before test leaves chance (grokking); (2) per-unit single-
 *  frequency purity — each hidden unit becomes a near-pure oscillator in
 *  lockstep with the test-accuracy jump; (3) the clock — token rows of the
 *  trained W1 projected onto a frequency's Fourier pair land on a circle
 *  traversed k times. NOT GPT-2: a separately trained toy model (meta says
 *  so, the header says so, /guide says so).
 *
 *  deck.gl (WebGL2), camera off, static. Source: grok.json. */

import type { Deck, OrthographicView, PickingInfo } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { type GrokBundle, loadGrok } from "../../data/interp";
import type { InterpDriver } from "./InterpDriver";

type LayersModule = typeof import("@deck.gl/layers");

const GL = 46; // px — y-axis labels
const GR = 16;
const GT = 96; // px — header summary
const GB = 92; // px — clock chips + collapsed legend pill

const AMBER: [number, number, number] = [245, 195, 59];
const SLATE: [number, number, number] = [138, 146, 178];
const CYAN: [number, number, number] = [70, 200, 235];
const GUIDE: [number, number, number, number] = [118, 126, 158, 130];

interface Seg {
  source: [number, number];
  target: [number, number];
}
interface CkCol {
  ci: number; // checkpoint index — invisible hover strip over the curve panel
}
interface HeatCell {
  ci: number;
  k: number; // frequency
}
interface ClockPt {
  a: number; // token 0..p-1
}

export class GrokClockDriver implements InterpDriver {
  readonly animated = false;
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private canvas!: HTMLCanvasElement;
  private tooltip!: HTMLElement;
  private labelRoot!: HTMLElement;
  private chipRoot!: HTMLElement;

  private g: GrokBundle | null = null;
  private clockIdx = 0; // selected clock face (ranked by measured circ)
  private hoverCk: number | null = null; // checkpoint index (curves)
  private hoverHeat: HeatCell | null = null;
  private hoverA: number | null = null; // clock token
  private layoutGen = 0;

  private heatCells: HeatCell[] = [];
  private ckCols: CkCol[] = [];
  private clockPts: ClockPt[] = [];
  private heatMax = 1; // stated clamp of the heat ramp

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
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    this.disposers.push(() => {
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
    });
  }

  async setModel(model: string): Promise<void> {
    if (!this.g) {
      this.g = await loadGrok(model);
      this.clockIdx = 0; // clocks[0] = highest measured circularity
    }
    this.hoverCk = this.hoverHeat = this.hoverA = null;
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
  private plotW(): number {
    return Math.max(40, this.cssW - GL - GR);
  }
  private plotH(): number {
    return Math.max(80, this.cssH - GT - this.gb());
  }

  /** panel geometry — wide: curves+heat stacked left (SHARED x), clock right;
   *  narrow (~450px-tall canvas): curves full-width on top, then heat and
   *  clock side by side — each panel carries its own x-axis there, and the
   *  heat keeps its own complete step ticks */
  private geom() {
    const pw = this.plotW();
    const ph = this.plotH();
    if (this.narrow()) {
      const curveH = Math.max(56, ph * 0.34);
      const gap = 26; // curve → row 2 (the heat caption row lives here)
      const row2Y = GT + curveH + gap;
      const row2H = Math.max(70, ph - curveH - gap);
      const heatW = Math.max(120, pw * 0.52);
      const heatH = Math.max(40, row2H - 18); // 18px = heat step-tick band
      const clockW = pw - heatW - 12;
      return {
        cx0: GL,
        curveW: pw,
        heatW,
        curveY: GT,
        curveH,
        heatY: row2Y,
        heatH,
        clockCX: GL + heatW + 12 + clockW / 2,
        clockCY: row2Y + heatH / 2,
        clockR: Math.max(28, Math.min(clockW, heatH) / 2 - 4),
      };
    }
    const lw = Math.max(120, pw * 0.56);
    const curveH = Math.max(60, ph * 0.52);
    const heatH = Math.max(50, ph - curveH - 34);
    const rw = pw - lw - 28;
    return {
      cx0: GL,
      curveW: lw,
      heatW: lw,
      curveY: GT,
      curveH,
      heatY: GT + curveH + 34,
      heatH,
      clockCX: GL + lw + 28 + rw / 2,
      clockCY: GT + ph / 2,
      clockR: Math.max(40, Math.min(rw, ph) / 2 - 26),
    };
  }

  /** x position of a training step within a panel of width `w` (curveW or
   *  heatW — identical on wide, different on narrow) */
  private xOfStep(step: number, w: number): number {
    const g = this.g;
    if (!g) return GL;
    return GL + (step / Math.max(1, g.meta.steps_run)) * w;
  }
  private yOfFrac(v: number): number {
    const gm = this.geom();
    return gm.curveY + (1 - Math.max(0, Math.min(1, v))) * gm.curveH;
  }

  private clockScale(): number {
    const g = this.g;
    const ck = g?.clocks[this.clockIdx];
    if (!g || !ck) return 1;
    let maxR = 1e-9;
    const p = g.meta.p;
    for (let a = 0; a < p; a++) {
      const x = ck.xy[a * 2] ?? 0;
      const y = ck.xy[a * 2 + 1] ?? 0;
      maxR = Math.max(maxR, Math.hypot(x, y));
    }
    return this.geom().clockR / maxR;
  }
  private clockXY(a: number): [number, number] {
    const g = this.g;
    const ck = g?.clocks[this.clockIdx];
    const gm = this.geom();
    if (!g || !ck) return [gm.clockCX, gm.clockCY];
    const s = this.clockScale();
    // flipY view: negate y so counter-clockwise math angles render as usual
    return [gm.clockCX + (ck.xy[a * 2] ?? 0) * s, gm.clockCY - (ck.xy[a * 2 + 1] ?? 0) * s];
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
    const g = this.g;
    if (!g) return;
    const nf = g.meta.n_freq;
    this.ckCols = g.steps.map((_, ci) => ({ ci }));
    this.heatCells = [];
    for (let ci = 0; ci < g.n_ckpt; ci++) {
      for (let k = 0; k < nf; k++) this.heatCells.push({ ci, k });
    }
    // ramp clamp = max NON-DC cell (linear to the DC max buries the real
    // 1–5% per-frequency structure; DC saturates and the caption says so)
    this.heatMax = 1e-9;
    for (let ci = 0; ci < g.n_ckpt; ci++) {
      for (let k = 1; k < nf; k++) {
        this.heatMax = Math.max(this.heatMax, g.fpower[ci * nf + k] ?? 0);
      }
    }
    this.clockPts = Array.from({ length: g.meta.p }, (_, a) => ({ a }));
    this.layoutGen++;
  }

  private series(vals: number[]): Seg[] {
    const g = this.g;
    if (!g) return [];
    const w = this.geom().curveW;
    const out: Seg[] = [];
    for (let i = 1; i < vals.length; i++) {
      out.push({
        source: [this.xOfStep(g.steps[i - 1] ?? 0, w), this.yOfFrac(vals[i - 1] ?? 0)],
        target: [this.xOfStep(g.steps[i] ?? 0, w), this.yOfFrac(vals[i] ?? 0)],
      });
    }
    return out;
  }

  private pushLayers(): void {
    const g = this.g;
    if (!this.deck || !g) return;
    const { ScatterplotLayer, LineLayer, SolidPolygonLayer } = this.layersMod;
    const gm = this.geom();
    const nf = g.meta.n_freq;
    const ck = g.clocks[this.clockIdx];

    // purity inter-quartile band — one polygon, honest quartiles from the bundle
    const band: [number, number][] = [];
    for (let i = 0; i < g.n_ckpt; i++) {
      band.push([this.xOfStep(g.steps[i] ?? 0, gm.curveW), this.yOfFrac(g.purity_q3[i] ?? 0)]);
    }
    for (let i = g.n_ckpt - 1; i >= 0; i--) {
      band.push([this.xOfStep(g.steps[i] ?? 0, gm.curveW), this.yOfFrac(g.purity_q1[i] ?? 0)]);
    }

    // vertical guides: train hits 100% / test hits 100% (the grok) — drawn
    // per panel, since curve and heat x-axes differ on narrow
    const guides: Seg[] = [];
    for (const s of [g.meta.tr100_step, g.meta.grok_step]) {
      guides.push({
        source: [this.xOfStep(s, gm.curveW), gm.curveY],
        target: [this.xOfStep(s, gm.curveW), gm.curveY + gm.curveH],
      });
      guides.push({
        source: [this.xOfStep(s, gm.heatW), gm.heatY],
        target: [this.xOfStep(s, gm.heatW), gm.heatY + gm.heatH],
      });
    }

    // clock successor path a -> a+1 — real pairs of computed points; the
    // k-fold winding of the polygon IS the algorithm
    const succ: Seg[] = [];
    const p = g.meta.p;
    for (let a = 0; a < p; a++) {
      succ.push({ source: this.clockXY(a), target: this.clockXY((a + 1) % p) });
    }

    const heatCellW = gm.heatW / g.n_ckpt;
    const heatCellH = gm.heatH / nf;

    const markers: { x: number; y: number; c: [number, number, number] }[] = [];
    if (this.hoverCk != null) {
      const i = this.hoverCk;
      const mx = this.xOfStep(g.steps[i] ?? 0, gm.curveW);
      markers.push(
        { x: mx, y: this.yOfFrac(g.train_acc[i] ?? 0), c: SLATE },
        { x: mx, y: this.yOfFrac(g.test_acc[i] ?? 0), c: AMBER },
        { x: mx, y: this.yOfFrac(g.purity_med[i] ?? 0), c: CYAN },
      );
    }

    this.deck.setProps({
      layers: [
        new SolidPolygonLayer<{ poly: [number, number][] }>({
          id: "grok-band",
          data: [{ poly: band }],
          getPolygon: (d) => d.poly,
          getFillColor: [CYAN[0], CYAN[1], CYAN[2], 36],
          updateTriggers: { getPolygon: this.layoutGen },
          pickable: false,
        }),
        new LineLayer<Seg>({
          id: "grok-guides",
          data: guides,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: GUIDE,
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        new LineLayer<Seg>({
          id: "grok-train",
          data: this.series(g.train_acc),
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: [SLATE[0], SLATE[1], SLATE[2], 210],
          getWidth: 1.4,
          widthUnits: "pixels",
          pickable: false,
        }),
        new LineLayer<Seg>({
          id: "grok-purity",
          data: this.series(g.purity_med),
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: [CYAN[0], CYAN[1], CYAN[2], 220],
          getWidth: 1.4,
          widthUnits: "pixels",
          pickable: false,
        }),
        new LineLayer<Seg>({
          id: "grok-test",
          data: this.series(g.test_acc),
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: [AMBER[0], AMBER[1], AMBER[2], 235],
          getWidth: 1.8,
          widthUnits: "pixels",
          pickable: false,
        }),
        // invisible hover strips — one per checkpoint over the curve panel
        new SolidPolygonLayer<CkCol>({
          id: "grok-curve-hit",
          data: this.ckCols,
          getPolygon: (d) => {
            const x0 = gm.cx0 + (d.ci / g.n_ckpt) * gm.curveW;
            const x1 = gm.cx0 + ((d.ci + 1) / g.n_ckpt) * gm.curveW;
            return [
              [x0, gm.curveY],
              [x1, gm.curveY],
              [x1, gm.curveY + gm.curveH],
              [x0, gm.curveY + gm.curveH],
            ];
          },
          getFillColor: [0, 0, 0, 1],
          updateTriggers: { getPolygon: this.layoutGen },
          pickable: true,
        }),
        new SolidPolygonLayer<HeatCell>({
          id: "grok-heat",
          data: this.heatCells,
          getPolygon: (d) => {
            const x0 = gm.cx0 + d.ci * heatCellW;
            const y0 = gm.heatY + d.k * heatCellH;
            return [
              [x0, y0],
              [x0 + heatCellW, y0],
              [x0 + heatCellW, y0 + heatCellH],
              [x0, y0 + heatCellH],
            ];
          },
          getFillColor: (d) => {
            const v = g.fpower[d.ci * nf + d.k] ?? 0;
            const t = Math.min(1, v / this.heatMax);
            return [
              20 + t * (AMBER[0] - 20),
              22 + t * (AMBER[1] - 22),
              34 + t * (AMBER[2] - 34),
              255,
            ];
          },
          updateTriggers: { getPolygon: this.layoutGen, getFillColor: this.layoutGen },
          pickable: true,
        }),
        new LineLayer<Seg>({
          id: "grok-succ",
          data: succ,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: [AMBER[0], AMBER[1], AMBER[2], 60],
          getWidth: 1,
          widthUnits: "pixels",
          updateTriggers: {
            getSourcePosition: [this.layoutGen, this.clockIdx],
            getTargetPosition: [this.layoutGen, this.clockIdx],
          },
          pickable: false,
        }),
        new ScatterplotLayer<ClockPt>({
          id: "grok-clock-pts",
          data: this.clockPts,
          getPosition: (d) => {
            const [x, y] = this.clockXY(d.a);
            return [x, y, 0];
          },
          getFillColor: [AMBER[0], AMBER[1], AMBER[2], 215],
          getRadius: this.narrow() ? 2.4 : 3,
          radiusUnits: "pixels",
          updateTriggers: { getPosition: [this.layoutGen, this.clockIdx] },
          pickable: true,
        }),
        new ScatterplotLayer<{ x: number; y: number; c: [number, number, number] }>({
          id: "grok-markers",
          data: markers,
          getPosition: (d) => [d.x, d.y, 0],
          getFillColor: [0, 0, 0, 0],
          getLineColor: (d) => [d.c[0], d.c[1], d.c[2], 255],
          getRadius: 4.5,
          radiusUnits: "pixels",
          stroked: true,
          filled: false,
          getLineWidth: 1.4,
          lineWidthUnits: "pixels",
          pickable: false,
        }),
        new ScatterplotLayer<{ a: number }>({
          id: "grok-clock-ring",
          data: this.hoverA != null ? [{ a: this.hoverA }] : [],
          getPosition: (d) => {
            const [x, y] = this.clockXY(d.a);
            return [x, y, 0];
          },
          getFillColor: [0, 0, 0, 0],
          getLineColor: [255, 255, 255, 230],
          getRadius: 6,
          radiusUnits: "pixels",
          stroked: true,
          filled: false,
          getLineWidth: 1.4,
          lineWidthUnits: "pixels",
          pickable: false,
        }),
      ],
    });
  }

  // ---- chips: clock faces ranked by measured circularity ------------------------
  private buildChips(): void {
    const g = this.g;
    if (!g) return;
    this.chipRoot.textContent = "";
    this.chipRoot.style.bottom = this.narrow() ? "110px" : "";
    g.clocks.forEach((ck, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "interp-neuron-chip";
      btn.textContent = this.narrow() ? `k${ck.k}` : `k=${ck.k} · circ ${ck.circ.toFixed(3)}`;
      btn.title =
        `frequency ${ck.k}: phase alignment ${ck.circ}, radius cv ${ck.radius_cv}, ` +
        `${ck.n_units} units, ${(ck.power_frac * 100).toFixed(1)}% of power — ` +
        "clocks ranked by measured circularity";
      const active = idx === this.clockIdx;
      btn.setAttribute("aria-pressed", String(active));
      if (active) btn.classList.add("is-active");
      btn.addEventListener("click", () => {
        this.clockIdx = idx;
        this.hoverA = null;
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
    const g = this.g;
    if (!g) return;
    const narrow = this.narrow();
    const compact = this.cssW < 1000;
    const gm = this.geom();
    const m = g.meta;

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
    const kk = (s: number) => (s >= 1000 ? `${(s / 1000).toFixed(1)}k` : `${s}`);
    const hMax = `${(this.cssW - (narrow ? 12 : GL) - 12).toFixed(0)}px`;

    const h1 = cap(
      narrow
        ? `grokking · toy MLP mod ${m.p} · NOT GPT-2 · grok @ ${kk(m.grok_step)}`
        : compact
          ? `grokking clock · toy MLP mod ${m.p} trained from scratch (NOT GPT-2) · ` +
            `train 100% @ ${kk(m.tr100_step)} · test 99.9% @ ${kk(m.grok_step)}`
          : `grokking clock · toy MLP (2·${m.p} → ${m.n_hidden} → ${m.p}, φ=z²) trained from ` +
            `scratch — NOT GPT-2 · train 100% @ ${kk(m.tr100_step)} · test 99.9% @ ` +
            `${kk(m.grok_step)} · gap ${kk(m.grok_step - m.tr100_step)} steps`,
    );
    h1.style.color = "rgb(245,195,59)";
    h1.style.maxWidth = hMax;
    place(h1, narrow ? 12 : GL, GT - 44);
    const h2 = cap(
      narrow
        ? `purity ${m.purity_at_memorized.toFixed(2)} → ${m.purity_final.toFixed(2)} at the grok`
        : compact
          ? `unit purity ${m.purity_at_memorized.toFixed(2)} → ${m.purity_final.toFixed(2)} in ` +
            `lockstep with test acc · aggregate spectrum stays spread`
          : `per-unit purity ${m.purity_init.toFixed(2)} init → ` +
            `${m.purity_at_memorized.toFixed(2)} memorized → ${m.purity_final.toFixed(2)} ` +
            `grokked, lockstep with test acc · aggregate spectrum stays spread ` +
            `(top-5 freqs ${(m.top5_mass_final * 100).toFixed(0)}%)`,
    );
    h2.style.maxWidth = hMax;
    place(h2, narrow ? 12 : GL, GT - 30);
    const h3 = cap(
      narrow
        ? `AdamW wd 0.3 · ${g.n_ckpt} ckpts · seed 0`
        : `full-batch AdamW · ${m.n_train.toLocaleString()} train / ${m.n_test.toLocaleString()} ` +
          `test pairs (${(m.train_frac * 100).toFixed(0)}%) · ${g.n_ckpt} checkpoints every ` +
          `${m.ckpt_every} steps · deterministic seed 0`,
    );
    h3.style.maxWidth = hMax;
    place(h3, narrow ? 12 : GL, GT - 16);

    // curve panel: y ticks + series labels + guide labels
    for (const v of [0, 0.5, 1]) {
      place(cap(v.toFixed(1)), Math.max(2, gm.cx0 - 26), this.yOfFrac(v) - 7);
    }
    const lt = cap("test acc");
    lt.style.color = "rgb(245,195,59)";
    place(lt, gm.cx0 + 4, gm.curveY + 2);
    const ltr = cap("train acc");
    ltr.style.color = "rgb(138,146,178)";
    place(ltr, gm.cx0 + 4, gm.curveY + 16);
    const lp = cap("unit purity (median + IQR)");
    lp.style.color = "rgb(70,200,235)";
    place(lp, gm.cx0 + 4, gm.curveY + 30);
    // narrow drops the in-panel guide labels — the same steps are printed as
    // x ticks under the heatmap, and at 375px they land on the curves
    if (!narrow) {
      const g1 = cap(`train 100% @ ${kk(m.tr100_step)}`);
      g1.style.color = "rgb(118,126,158)";
      place(g1, this.xOfStep(m.tr100_step, gm.curveW) + 4, gm.curveY + gm.curveH - 14);
      const g2 = cap(`grok @ ${kk(m.grok_step)}`);
      g2.style.color = "rgb(118,126,158)";
      place(g2, this.xOfStep(m.grok_step, gm.curveW) + 4, gm.curveY + gm.curveH - 14);
    }

    // heat panel: freq ticks + ramp caption + x (step) ticks under it
    const hLab = cap(
      narrow
        ? `W1 power/freq · 0→${(this.heatMax * 100).toFixed(1)}% · DC sat.`
        : compact
          ? `W1 power per freq · 0 → ${(this.heatMax * 100).toFixed(1)}% · DC saturates`
          : `W1 spectral power fraction per frequency · linear ramp 0 → ` +
            `${(this.heatMax * 100).toFixed(1)}% (max non-DC cell) · DC row saturates`,
    );
    hLab.style.color = "rgb(118,126,158)";
    place(hLab, gm.cx0, gm.heatY - 14);
    for (const k of narrow ? [0, 24, 48] : [0, 12, 24, 36, 48]) {
      const el = cap(k === 0 ? "DC" : `${k}`);
      place(el, Math.max(2, gm.cx0 - 26), gm.heatY + (k / m.n_freq) * gm.heatH - 2);
    }
    for (const s of narrow
      ? [0, m.tr100_step, m.grok_step]
      : [0, m.tr100_step, m.grok_step, m.steps_run]) {
      const el = cap(kk(s));
      place(el, this.xOfStep(s, gm.heatW) - (s === m.steps_run ? 22 : 2), gm.heatY + gm.heatH + 4);
    }
    if (!narrow) {
      const xt = cap("step →");
      xt.style.color = "rgb(118,126,158)";
      place(xt, gm.cx0 + gm.heatW - 44, gm.heatY + gm.heatH + 16);
    }

    // clock caption + a few token labels to show the winding
    const ck = g.clocks[this.clockIdx];
    if (ck) {
      const cLab = cap(
        narrow
          ? `k=${ck.k} · circ ${ck.circ.toFixed(3)}`
          : `clock k=${ck.k} · circ ${ck.circ.toFixed(4)} · ` +
            `radius cv ${ck.radius_cv.toFixed(2)} · ${ck.n_units} units`,
      );
      cLab.style.color = "rgb(245,195,59)";
      const cw2 = narrow
        ? this.cssW - (gm.clockCX - gm.clockR) - 8
        : this.cssW - (gm.clockCX - gm.clockR) - 12;
      cLab.style.maxWidth = `${Math.max(80, cw2).toFixed(0)}px`;
      // narrow: the clock sits beside the heat panel, so its caption goes in
      // the heat's tick row (below the clock); wide: above the clock
      place(
        cLab,
        gm.clockCX - gm.clockR,
        narrow ? gm.heatY + gm.heatH + 4 : gm.clockCY - gm.clockR - 30,
      );
      if (!narrow) {
        const cNote = cap(`angle(a) ≈ ±2πk·a/${m.p} · path a→a+1 winds k×`);
        cNote.style.color = "rgb(118,126,158)";
        cNote.style.maxWidth = cLab.style.maxWidth;
        place(cNote, gm.clockCX - gm.clockR, gm.clockCY - gm.clockR - 16);
      }
      for (const a of narrow ? [0] : [0, 1, 2]) {
        const [x, y] = this.clockXY(a);
        const el = cap(`a=${a}`);
        el.style.color = "rgb(205,210,224)";
        place(el, x + 6, y - 16);
      }
    }
  }

  // ---- interaction --------------------------------------------------------------
  private pick(e: PointerEvent): PickingInfo | null {
    if (!this.deck) return null;
    const rect = this.canvas.getBoundingClientRect();
    return this.deck.pickObject({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      radius: 4,
      layerIds: ["grok-clock-pts", "grok-curve-hit", "grok-heat"],
    }) as PickingInfo | null;
  }

  private onPointerMove(e: PointerEvent): void {
    const g = this.g;
    if (!g) return;
    const info = this.pick(e);
    const layer = info?.layer?.id;
    const prev = `${this.hoverCk}/${this.hoverHeat?.ci},${this.hoverHeat?.k}/${this.hoverA}`;
    this.hoverCk = layer === "grok-curve-hit" ? (info?.object as CkCol).ci : null;
    this.hoverHeat = layer === "grok-heat" ? (info?.object as HeatCell) : null;
    this.hoverA = layer === "grok-clock-pts" ? (info?.object as ClockPt).a : null;
    if (`${this.hoverCk}/${this.hoverHeat?.ci},${this.hoverHeat?.k}/${this.hoverA}` !== prev) {
      this.pushLayers();
    }

    this.tooltip.innerHTML = "";
    const add = (cls: string, text: string) => {
      const el = document.createElement("div");
      el.className = cls;
      el.textContent = text;
      this.tooltip.appendChild(el);
    };
    if (this.hoverCk != null) {
      const i = this.hoverCk;
      add("point-tooltip-label", `step ${g.steps[i]}`);
      add("point-tooltip-conf", `train acc ${(g.train_acc[i] ?? 0).toFixed(4)} · MSE ${(g.train_loss[i] ?? 0).toFixed(6)}`);
      add("point-tooltip-conf", `test acc ${(g.test_acc[i] ?? 0).toFixed(4)} · MSE ${(g.test_loss[i] ?? 0).toFixed(6)}`);
      add(
        "point-tooltip-conf",
        `unit purity median ${(g.purity_med[i] ?? 0).toFixed(4)} · ` +
          `IQR [${(g.purity_q1[i] ?? 0).toFixed(3)}, ${(g.purity_q3[i] ?? 0).toFixed(3)}]`,
      );
    } else if (this.hoverHeat) {
      const { ci, k } = this.hoverHeat;
      const v = g.fpower[ci * g.meta.n_freq + k] ?? 0;
      add("point-tooltip-label", `freq ${k === 0 ? "0 (DC)" : k} · step ${g.steps[ci]}`);
      add("point-tooltip-conf", `${(v * 100).toFixed(2)}% of ‖W1ₐ‖² spectral power`);
      add(
        "point-tooltip-conf",
        `linear ramp clamps at ${(this.heatMax * 100).toFixed(2)}% (max non-DC cell)`,
      );
    } else if (this.hoverA != null && g.clocks[this.clockIdx]) {
      const ck = g.clocks[this.clockIdx];
      if (!ck) return;
      const a = this.hoverA;
      const x = ck.xy[a * 2] ?? 0;
      const y = ck.xy[a * 2 + 1] ?? 0;
      const ang = (Math.atan2(y, x) * 180) / Math.PI;
      add("point-tooltip-label", `token a = ${a} · clock k = ${ck.k}`);
      add("point-tooltip-conf", `proj (${x.toFixed(4)}, ${y.toFixed(4)}) · angle ${ang.toFixed(1)}°`);
      add("point-tooltip-conf", `k·a mod ${g.meta.p} = ${(ck.k * a) % g.meta.p} — its slot on the dial`);
      add("point-tooltip-conf", `phase alignment of this clock: ${ck.circ}`);
    } else {
      this.tooltip.style.visibility = "hidden";
      this.canvas.style.cursor = "";
      return;
    }
    const rect = this.canvas.getBoundingClientRect();
    const lx = e.clientX - rect.left;
    const ly = e.clientY - rect.top;
    this.tooltip.style.visibility = "visible";
    this.tooltip.style.left = `${Math.min(lx + 14, this.cssW - 250)}px`;
    this.tooltip.style.top = `${Math.min(ly + 14, this.cssH - 110)}px`;
    this.canvas.style.cursor = "crosshair";
  }

  private onLeave(): void {
    if (this.hoverCk != null || this.hoverHeat || this.hoverA != null) {
      this.hoverCk = this.hoverHeat = null;
      this.hoverA = null;
      this.pushLayers();
    }
    this.tooltip.style.visibility = "hidden";
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
    this.hoverCk = this.hoverHeat = null;
    this.hoverA = null;
    this.deck?.setProps({
      width,
      height,
      useDevicePixels: Math.min(dpr, 2),
      viewState: this.viewState(),
    });
    this.pushLayers();
    if (this.g) this.buildChips();
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
