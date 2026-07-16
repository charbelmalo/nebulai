/** #1 Fourier Atlas — the discrete Fourier transform of GPT-2's LEARNED
 *  positional embeddings W_pos, taken along the position axis. Real quantity
 *  (fourier.json): P(f) = mean_d |rfft(W_pos − mean)[f, d]|², the mean power at
 *  frequency f in cycles per 1024-token context window, plus per_dim_dominant —
 *  the frequency at which each of the 768 embedding dimensions peaks.
 *
 *  Radial spectrogram: angle encodes frequency (f=1 at top, increasing
 *  clockwise; a labeled gap marks the f_max→f_min seam so the axis reads as
 *  non-cyclic). Gold spokes grow OUTWARD from the baseline ring with radius =
 *  log₁₀ P(f); cyan spokes grow INWARD with length = how many dims are dominant
 *  at f. Concentric rings are labeled power decades. The whole disc is a dial:
 *  hover anywhere to read the exact frequency, its period in tokens, the exact
 *  mean power, and the dominant-dim count — all real, all from the bundle.
 *
 *  This exposes the genuinely periodic structure of the learned positional code
 *  (low frequencies carry almost all the power) without any smoothing or
 *  decorative motion. deck.gl (WebGL2), camera off, framing from canvas size. */

import type { Deck, OrthographicView } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { loadFourier, type FourierBundle } from "../../data/interp";
import {
  dashedSegment,
  DIM_ALPHA,
  GRID_RGBA,
  MARKER_HOT,
  markerPoly,
  type Vec2,
  withAlpha,
} from "./chart-theme";
import { InterpTooltip, type TipRow } from "./chart-tooltip";
import type { InterpDriver } from "./InterpDriver";

type LayersModule = typeof import("@deck.gl/layers");

const INNER_R = 0.42; // baseline ring radius (world units)
const OUTER_R = 1.0; // max power radius
const DIM_R = 0.1; // innermost radius the dim-count spokes can reach
const GAP = 0.17; // angular gap (rad) marking the f_max→f_min seam
const START = Math.PI / 2; // f=1 at top
const SWEEP = Math.PI * 2 - GAP; // total angular span for the frequency axis
const PAD_PX = 80;

const GOLD: [number, number, number] = [245, 195, 59];
const CYAN: [number, number, number] = [70, 200, 235];

interface Spoke {
  f: number;
  path: [number, number][];
  /** normalized intensity in [0,1] — the SAME quantity the radius encodes
   *  (log power for gold, dim-count for cyan). Driving alpha + width from it too
   *  is redundant, not misleading: it stops the hundreds of near-zero
   *  high-frequency spokes from out-inking the few that carry the power. */
  v: number;
}

export class FourierAtlasDriver implements InterpDriver {
  readonly animated = false; // static spectrum — redraws only on hover/resize
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private makeView!: () => OrthographicView;
  private canvas!: HTMLCanvasElement;
  private tooltip!: InterpTooltip;
  private axisRoot!: HTMLElement;

  private bundle: FourierBundle | null = null;
  private counts: number[] = []; // dims dominant per frequency index
  private maxCount = 1;
  private logMin = 0;
  private logMax = 1;
  private nFreq = 0;
  private nCtx = 1024;
  private hoverF: number | null = null; // illuminated frequency under the cursor

  private cssW = 1;
  private cssH = 1;
  private dpr = 1;
  private disposers: Array<() => void> = [];

  async init(canvas: HTMLCanvasElement, _tier: GpuTier, overlay: HTMLElement): Promise<void> {
    this.canvas = canvas;
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
    this.axisRoot = document.createElement("div");
    this.axisRoot.className = "interp-axis";
    overlay.appendChild(this.axisRoot);

    const onMove = (e: PointerEvent) => this.onPointerMove(e);
    const onLeave = () => this.hideTip();
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    this.disposers.push(() => {
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
    });
  }

  async setModel(model: string): Promise<void> {
    const b = await loadFourier(model);
    this.bundle = b;
    this.nFreq = b.power_mean.length;
    this.nCtx = b.meta.n_ctx;

    // log power range over positive powers, excluding DC (f=0, ~0 after centering)
    let hi = -Infinity;
    let lo = Infinity;
    for (let f = 1; f < this.nFreq; f++) {
      const p = b.power_mean[f]!;
      if (p <= 0) continue;
      const l = Math.log10(p);
      if (l > hi) hi = l;
      if (l < lo) lo = l;
    }
    this.logMax = hi;
    this.logMin = lo;

    // histogram: how many of the 768 dims are dominant at each frequency
    this.counts = new Array(this.nFreq).fill(0);
    for (const d of b.per_dim_dominant) {
      if (d >= 0 && d < this.nFreq) this.counts[d]! += 1;
    }
    this.maxCount = Math.max(1, ...this.counts);

    this.buildAxis();
    this.pushLayers();
  }

  /** angle for frequency index f (1..nFreq-1), clockwise from the top. LOG
   *  spacing in f: the learned positional power is a steep power law (almost all
   *  of it at f=1..~16), so a linear angular axis would collapse every
   *  informative frequency into a thin sliver. log-f gives the low frequencies
   *  the angular room they carry — the axis label states this so it stays honest. */
  private angleOf(f: number): number {
    const t = Math.log(f) / Math.log(Math.max(this.nFreq - 1, 2));
    return START - t * SWEEP;
  }
  private powerRadius(p: number): number {
    const l = p > 0 ? Math.log10(p) : this.logMin;
    const t = (l - this.logMin) / Math.max(this.logMax - this.logMin, 1e-9);
    return INNER_R + Math.max(0, Math.min(1, t)) * (OUTER_R - INNER_R);
  }

  /** World units per screen pixel — pixel-authored dashes/markers scaled to world. */
  private worldPerPx(): number {
    return 1 / this.zoomPx();
  }

  private pushLayers(): void {
    if (!this.deck || !this.bundle) return;
    const { PathLayer, SolidPolygonLayer } = this.layersMod;
    const b = this.bundle;
    const wpp = this.worldPerPx();
    const hf = this.hoverF;

    const power: Spoke[] = [];
    const dims: Spoke[] = [];
    const span = OUTER_R - INNER_R;
    for (let f = 1; f < this.nFreq; f++) {
      const a = this.angleOf(f);
      const ca = Math.cos(a);
      const sa = Math.sin(a);
      const rp = this.powerRadius(b.power_mean[f]!);
      const vp = (rp - INNER_R) / span; // same normalized log power the radius shows
      power.push({ f, v: vp, path: [[INNER_R * ca, INNER_R * sa], [rp * ca, rp * sa]] });
      const cN = this.counts[f]! / this.maxCount;
      if (cN > 0) {
        const rd = INNER_R - cN * (INNER_R - DIM_R);
        dims.push({ f, v: cN, path: [[INNER_R * ca, INNER_R * sa], [rd * ca, rd * sa]] });
      }
    }

    // concentric power-decade rings + the baseline ring — subtle DASHED
    // hairlines now (req 5): the radius legend whispers, the spokes speak. Each
    // ring is broken into pixel-scaled dashes around its circumference.
    const rings: { path: [number, number][] }[] = [];
    const ringAt = (r: number) => {
      const pts: [number, number][] = [];
      for (let i = 0; i <= 96; i++) {
        const a = (i / 96) * Math.PI * 2;
        pts.push([r * Math.cos(a), r * Math.sin(a)]);
      }
      for (let i = 0; i < pts.length - 1; i++) {
        for (const s of dashedSegment(pts[i]!, pts[i + 1]!, 3 * wpp, 6 * wpp)) {
          rings.push({ path: [s.source, s.target] });
        }
      }
    };
    ringAt(INNER_R);
    const kLo = Math.ceil(this.logMin);
    const kHi = Math.floor(this.logMax);
    for (let k = kLo; k <= kHi; k++) ringAt(this.powerRadius(10 ** k));

    // focus/dim (req 3): on hover, the focused frequency's spokes illuminate to
    // full strength while every other frequency recedes to a faint trace.
    const powerAlpha = (d: Spoke) => {
      const base = Math.round(60 + 190 * d.v);
      if (hf == null) return base;
      return d.f === hf ? 255 : Math.round(base * DIM_ALPHA);
    };
    const dimAlpha = (d: Spoke) => {
      const base = Math.round(55 + 165 * d.v);
      if (hf == null) return base;
      return d.f === hf ? 255 : Math.round(base * DIM_ALPHA);
    };
    // sharp LED marker locked onto the hovered frequency's power-spoke tip (req 4)
    const marks: { poly: Vec2[]; color: [number, number, number, number] }[] = [];
    if (hf != null) {
      const a = this.angleOf(hf);
      const rp = this.powerRadius(b.power_mean[hf]!);
      const tx = rp * Math.cos(a);
      const ty = rp * Math.sin(a);
      marks.push(
        { poly: markerPoly(tx, ty, 8 * wpp), color: withAlpha(MARKER_HOT, 0.22) },
        { poly: markerPoly(tx, ty, 4 * wpp), color: withAlpha(MARKER_HOT, 1) },
      );
    }

    this.deck.setProps({
      layers: [
        new PathLayer<{ path: [number, number][] }>({
          id: "fa-rings",
          data: rings,
          getPath: (d) => d.path,
          getColor: GRID_RGBA,
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        new PathLayer<Spoke>({
          id: "fa-dims",
          data: dims,
          getPath: (d) => d.path,
          getColor: (d) => [CYAN[0], CYAN[1], CYAN[2], dimAlpha(d)],
          getWidth: (d) => (hf === d.f ? 2.2 : 1 + 1.1 * d.v),
          widthUnits: "pixels",
          updateTriggers: { getColor: hf, getWidth: hf },
          pickable: false,
        }),
        new PathLayer<Spoke>({
          id: "fa-power",
          data: power,
          getPath: (d) => d.path,
          getColor: (d) => [GOLD[0], GOLD[1], GOLD[2], powerAlpha(d)],
          getWidth: (d) => (hf === d.f ? 2.8 : 1 + 1.7 * d.v),
          widthUnits: "pixels",
          updateTriggers: { getColor: hf, getWidth: hf },
          pickable: false,
        }),
        new SolidPolygonLayer<{ poly: Vec2[]; color: [number, number, number, number] }>({
          id: "fa-marker",
          data: marks,
          getPolygon: (d) => d.poly,
          getFillColor: (d) => d.color,
          pickable: false,
        }),
      ],
    });
  }

  private buildAxis(): void {
    if (!this.axisRoot) return;
    this.axisRoot.textContent = "";
    // power decade labels, placed on the vertical (top) where spokes start
    const kLo = Math.ceil(this.logMin);
    const kHi = Math.floor(this.logMax);
    for (let k = kLo; k <= kHi; k++) {
      const el = document.createElement("div");
      el.className = "interp-axis-y";
      el.dataset.r = String(this.powerRadius(10 ** k));
      el.textContent = `P=10${sup(k)}`;
      this.axisRoot.appendChild(el);
    }
    // seam markers on either side of the gap
    const f1 = document.createElement("div");
    f1.className = "interp-axis-x";
    f1.dataset.role = "f1";
    f1.textContent = "f=1";
    this.axisRoot.appendChild(f1);
    const fN = document.createElement("div");
    fN.className = "interp-axis-x";
    fN.dataset.role = "fmax";
    fN.textContent = `f=${this.nFreq - 1}`;
    this.axisRoot.appendChild(fN);
    this.positionAxis();
  }

  private zoomPx(): number {
    return Math.min((this.cssW - PAD_PX) / (2 * OUTER_R), (this.cssH - PAD_PX) / (2 * OUTER_R));
  }
  private worldToScreen(wx: number, wy: number): [number, number] {
    const z = this.zoomPx();
    return [this.cssW / 2 + wx * z, this.cssH / 2 - wy * z];
  }

  private positionAxis(): void {
    if (!this.axisRoot) return;
    // decade rings are labeled along the empty seam wedge (top-left), where no
    // spoke can land — keeps the radius legend clear of the data.
    const aLbl = START + GAP / 2;
    const cl = Math.cos(aLbl);
    const sl = Math.sin(aLbl);
    for (const el of Array.from(this.axisRoot.querySelectorAll<HTMLElement>(".interp-axis-y"))) {
      const r = Number(el.dataset.r);
      const [sx, sy] = this.worldToScreen(r * cl, r * sl);
      el.style.transform = `translate(${(sx - 4).toFixed(1)}px, ${(sy - 8).toFixed(1)}px) translateX(-100%)`;
    }
    // seam markers sit just either side of the gap at the top: f=1 at the top
    // (frequency grows clockwise, so its label leans right); f_max lands one GAP
    // counter-clockwise of the top, its label leaning left into empty space.
    for (const el of Array.from(this.axisRoot.querySelectorAll<HTMLElement>(".interp-axis-x"))) {
      const isF1 = el.dataset.role === "f1";
      const a = START + (isF1 ? 0.0 : GAP);
      const rr = OUTER_R + 0.06;
      const [sx, sy] = this.worldToScreen(rr * Math.cos(a), rr * Math.sin(a));
      el.style.transform =
        `translate(${sx.toFixed(1)}px, ${(sy - 6).toFixed(1)}px) translateX(${isF1 ? "10%" : "-110%"})`;
    }
  }

  private onPointerMove(e: PointerEvent): void {
    if (!this.deck || !this.bundle) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const vp = this.deck.getViewports()[0];
    if (!vp) return;
    const world = vp.unproject([x, y]) as [number, number, number];
    const wx = world[0];
    const wy = world[1];
    const r = Math.hypot(wx, wy);
    if (r < DIM_R * 0.6 || r > OUTER_R * 1.08) {
      this.hideTip();
      return;
    }
    // angle → frequency (clockwise from top); reject the seam gap
    let d = START - Math.atan2(wy, wx); // 0 at top, grows clockwise
    d = ((d % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
    if (d > SWEEP) {
      this.hideTip();
      return;
    }
    // invert the LOG angle map: f = (nFreq-1)^(d/SWEEP)
    const t = d / SWEEP;
    const f = Math.max(
      1,
      Math.min(this.nFreq - 1, Math.round(Math.pow(this.nFreq - 1, t))),
    );
    const power = this.bundle.power_mean[f]!;
    const period = this.nCtx / f;
    const dimCount = this.counts[f]!;

    if (f !== this.hoverF) {
      this.hoverF = f;
      this.pushLayers();
    }
    const rows: TipRow[] = [
      { kind: "label", text: `f = ${f} cyc/window`, swatch: GOLD },
      { text: "P(f)", value: power.toPrecision(4), hot: true },
      { text: "period", value: `${period.toFixed(1)} tokens` },
      { text: `${dimCount} of ${this.bundle.per_dim_dominant.length} dims peak here` },
    ];
    this.tooltip.show(rows);
    this.tooltip.move(x, y, this.cssW, this.cssH);
    this.canvas.style.cursor = "crosshair";
  }

  private hideTip(): void {
    this.tooltip?.hide();
    this.canvas.style.cursor = "";
    if (this.hoverF != null) {
      this.hoverF = null;
      this.pushLayers();
    }
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
    this.deck?.setProps({
      width,
      height,
      useDevicePixels: Math.min(dpr, 2),
      viewState: this.viewState(),
    });
    this.positionAxis();
  }

  dispose(): void {
    for (const d of this.disposers) d();
    this.disposers = [];
    this.tooltip?.dispose();
    this.axisRoot?.remove();
    this.deck?.finalize();
    this.deck = null;
  }
}

/** superscript a (possibly negative) integer exponent for axis labels. */
function sup(k: number): string {
  const map: Record<string, string> = {
    "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
  };
  return String(k)
    .split("")
    .map((c) => map[c] ?? c)
    .join("");
}
