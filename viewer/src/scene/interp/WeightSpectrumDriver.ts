/** #21 Weight Spectrum — the singular-value spectrum of every weight matrix in
 *  the model, overlaid as log-scale decay curves. Real quantity: σ = svd(W),
 *  computed float64 offline (weights.json). One polyline per matrix; x = SV
 *  index, y = log10(σ), hue = matrix kind, brightness = layer depth. Hover a
 *  curve to read the exact σ at that index plus the matrix's stable rank,
 *  effective rank, and condition number — all real, all from the bundle.
 *
 *  Honest encodings only: the axes ARE the numbers. The log-y makes the
 *  power-law decay legible without distorting order; decade gridlines are
 *  labeled so magnitude is readable, not merely relative. deck.gl (WebGL2)
 *  with its built-in picking, camera off (framing derived from canvas size). */

import type { Deck, OrthographicView, PickingInfo } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { loadWeights, type SpectrumMatrix, type WeightsBundle } from "../../data/interp";
import {
  ACCENT,
  crosshair,
  dashedSegment,
  GRID_RGBA,
  MARKER_HOT,
  markerPoly,
  type Vec2,
  withAlpha,
} from "./chart-theme";
import { InterpTooltip } from "./chart-tooltip";
import type { InterpDriver } from "./InterpDriver";

type LayersModule = typeof import("@deck.gl/layers");

const SPAN_X = 2.4; // world width of the plot box
const SPAN_Y = 1.5; // world height
const PAD_PX = 64;

// matrix-kind palette — hue carries type, and the legend states it. Chosen for
// categorical distinctness in both themes (not a perceptual ramp on ordered data).
const KIND_RGB: Record<SpectrumMatrix["kind"], [number, number, number]> = {
  embed: [234, 79, 134], // W_E — magenta
  pos: [245, 195, 59], // W_pos — gold
  attn_qkv: [70, 200, 235], // attention QKV — cyan
  attn_out: [90, 230, 180], // attention out — teal
  mlp_in: [150, 130, 240], // MLP in — violet
  mlp_out: [139, 59, 240], // MLP out — purple
};

interface Curve {
  matrix: SpectrumMatrix;
  path: [number, number][];
  color: [number, number, number, number];
}

export class WeightSpectrumDriver implements InterpDriver {
  readonly animated = false; // static spectrum — deck redraws only on hover/resize
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private makeView!: () => OrthographicView;
  private canvas!: HTMLCanvasElement;
  private overlay!: HTMLElement;
  private tooltip!: InterpTooltip;
  private axisRoot!: HTMLElement;

  private bundle: WeightsBundle | null = null;
  private curves: Curve[] = [];
  private logMax = 1;
  private logMin = 0;
  // hover focus: the illuminated curve (by unique matrix name) and the world
  // point of the σ under the cursor (for the LED marker + crosshair).
  private hoverName: string | null = null;
  private hoverMark: Vec2 | null = null;

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
    const b = await loadWeights(model);
    this.bundle = b;
    // global log range across all stored singular values (positive only)
    let hi = -Infinity;
    let lo = Infinity;
    for (const m of b.matrices) {
      for (const s of m.singular_values) {
        if (s <= 0) continue;
        const l = Math.log10(s);
        if (l > hi) hi = l;
        if (l < lo) lo = l;
      }
    }
    this.logMax = hi;
    this.logMin = lo;
    const nLayers = Math.max(1, b.meta.n_layer);

    this.curves = b.matrices.map((m) => {
      const [r, g, bl] = KIND_RGB[m.kind];
      // brightness by layer depth: shallow → dim, deep → bright (embed/pos = mid)
      const depth = m.layer == null ? 0.5 : (m.layer + 0.5) / nLayers;
      const lift = 0.55 + 0.45 * depth;
      const n = m.singular_values.length;
      const path: [number, number][] = [];
      for (let i = 0; i < n; i++) {
        const s = m.singular_values[i]!;
        path.push([this.xAt(i, n), this.yAt(s)]);
      }
      return {
        matrix: m,
        path,
        color: [r * lift, g * lift, bl * lift, 205],
      };
    });
    this.buildAxis();
    this.pushLayers();
  }

  private xAt(i: number, n: number): number {
    return (i / Math.max(n - 1, 1)) * SPAN_X;
  }
  private yAt(sigma: number): number {
    const l = sigma > 0 ? Math.log10(sigma) : this.logMin;
    const t = (l - this.logMin) / Math.max(this.logMax - this.logMin, 1e-6);
    return t * SPAN_Y;
  }

  /** World units per screen pixel — dashes/markers are authored in pixels for a
   *  consistent on-screen size, then scaled into this driver's world space. */
  private worldPerPx(): number {
    const zoom = Math.min((this.cssW - PAD_PX) / SPAN_X, (this.cssH - PAD_PX) / SPAN_Y);
    return 1 / Math.max(zoom, 1e-6);
  }

  private pushLayers(): void {
    if (!this.deck) return;
    const { PathLayer, SolidPolygonLayer } = this.layersMod;
    const wpp = this.worldPerPx();
    // decade gridlines at σ = 10^k within [logMin, logMax] — subtle DASHED
    // hairlines now (req 5): the structure whispers, the curves speak.
    const grid: { path: [number, number][] }[] = [];
    const kLo = Math.ceil(this.logMin);
    const kHi = Math.floor(this.logMax);
    for (let k = kLo; k <= kHi; k++) {
      const y = this.yAt(10 ** k);
      for (const s of dashedSegment([0, y], [SPAN_X, y], 3 * wpp, 6 * wpp)) {
        grid.push({ path: [s.source, s.target] });
      }
    }
    // crosshair + LED marker locked onto the hovered σ point (req 4)
    const cross: { path: [number, number][] }[] = this.hoverMark
      ? crosshair(this.hoverMark[0], this.hoverMark[1], { x0: 0, y0: 0, x1: SPAN_X, y1: SPAN_Y }, 4 * wpp, 5 * wpp).map(
          (s) => ({ path: [s.source, s.target] }),
        )
      : [];
    const marks = this.hoverMark
      ? [
          { poly: markerPoly(this.hoverMark[0], this.hoverMark[1], 9 * wpp), color: withAlpha(MARKER_HOT, 0.22) },
          { poly: markerPoly(this.hoverMark[0], this.hoverMark[1], 4.5 * wpp), color: withAlpha(MARKER_HOT, 1) },
        ]
      : [];

    this.deck.setProps({
      layers: [
        new PathLayer<{ path: [number, number][] }>({
          id: "ws-grid",
          data: grid,
          getPath: (d) => d.path,
          getColor: GRID_RGBA,
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        new PathLayer<Curve>({
          id: "ws-curves",
          data: this.curves,
          getPath: (d) => d.path,
          // focus/dim (req 3): on hover, the focused curve illuminates to full
          // hue while every other curve recedes to a faint trace.
          getColor: (d) => {
            if (this.hoverName == null) return d.color;
            if (d.matrix.name === this.hoverName) {
              const [r, g, bl] = KIND_RGB[d.matrix.kind];
              return [r, g, bl, 245];
            }
            return [d.color[0], d.color[1], d.color[2], 26];
          },
          getWidth: (d) =>
            this.hoverName == null ? 1.6 : d.matrix.name === this.hoverName ? 2.6 : 1.0,
          widthUnits: "pixels",
          jointRounded: true,
          capRounded: true,
          updateTriggers: { getColor: this.hoverName, getWidth: this.hoverName },
          pickable: true,
        }),
        new PathLayer<{ path: [number, number][] }>({
          id: "ws-crosshair",
          data: cross,
          getPath: (d) => d.path,
          getColor: withAlpha(ACCENT, 0.5),
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        new SolidPolygonLayer<{ poly: Vec2[]; color: [number, number, number, number] }>({
          id: "ws-marker",
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
    // y decade labels + an x-axis caption, positioned in resize()
    const kLo = Math.ceil(this.logMin);
    const kHi = Math.floor(this.logMax);
    for (let k = kLo; k <= kHi; k++) {
      const el = document.createElement("div");
      el.className = "interp-axis-y";
      el.dataset.sigma = String(10 ** k);
      el.textContent = `σ=${10 ** k >= 1 ? (10 ** k).toLocaleString("en-US") : (10 ** k).toPrecision(1)}`;
      this.axisRoot.appendChild(el);
    }
    const xl = document.createElement("div");
    xl.className = "interp-axis-x";
    xl.textContent = "singular-value index →";
    this.axisRoot.appendChild(xl);
    this.positionAxis();
  }

  /** World → screen for axis label placement (mirrors the deck viewState). */
  private worldToScreen(wx: number, wy: number): [number, number] {
    const short = Math.max(Math.min(this.cssW, this.cssH), 1);
    const zoom = Math.min(
      (this.cssW - PAD_PX) / SPAN_X,
      (this.cssH - PAD_PX) / SPAN_Y,
    );
    void short;
    const cx = this.cssW / 2;
    const cy = this.cssH / 2;
    return [cx + (wx - SPAN_X / 2) * zoom, cy - (wy - SPAN_Y / 2) * zoom];
  }

  private positionAxis(): void {
    if (!this.axisRoot) return;
    for (const el of Array.from(this.axisRoot.querySelectorAll<HTMLElement>(".interp-axis-y"))) {
      const sigma = Number(el.dataset.sigma);
      const [sx, sy] = this.worldToScreen(0, this.yAt(sigma));
      // right-aligned at the plot edge; clamp so the widest label (σ=100)
      // can't cross the overlay's left boundary and get clipped
      const x = Math.max(sx - 6, el.offsetWidth + 4);
      el.style.transform = `translate(${x.toFixed(1)}px, ${(sy - 8).toFixed(1)}px) translateX(-100%)`;
    }
    const xEl = this.axisRoot.querySelector<HTMLElement>(".interp-axis-x");
    if (xEl) {
      const [sx, sy] = this.worldToScreen(SPAN_X / 2, 0);
      xEl.style.transform = `translate(${sx.toFixed(1)}px, ${(sy + 10).toFixed(1)}px) translateX(-50%)`;
    }
  }

  private onPointerMove(e: PointerEvent): void {
    if (!this.deck) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const info = this.deck.pickObject({ x, y, radius: 6 }) as PickingInfo | null;
    if (!info?.object || info.layer?.id !== "ws-curves") {
      this.hideTip();
      return;
    }
    const c = info.object as Curve;
    // exact SV index from the world x under the cursor (info.coordinate)
    const wx = info.coordinate ? info.coordinate[0]! : 0;
    const n = c.matrix.singular_values.length;
    const idx = Math.max(0, Math.min(n - 1, Math.round((wx / SPAN_X) * (n - 1))));
    const sigma = c.matrix.singular_values[idx]!;
    const m = c.matrix;
    // update focus + marker, then re-push (only 4 layers — cheap enough to stay
    // at 60fps): the marker must track the cursor even within one curve.
    this.hoverName = m.name;
    this.hoverMark = [this.xAt(idx, n), this.yAt(sigma)];
    this.pushLayers();
    this.tooltip.show([
      { kind: "label", text: m.name, swatch: KIND_RGB[m.kind] },
      { text: `σ[${idx}]`, value: sigma.toPrecision(5), hot: true },
      { text: "stable rank", value: String(m.stable_rank) },
      { text: "eff. rank", value: String(m.effective_rank) },
      { text: "κ", value: m.condition.toLocaleString("en-US") },
    ]);
    this.tooltip.move(x, y, this.cssW, this.cssH);
    this.canvas.style.cursor = "crosshair";
  }

  private hideTip(): void {
    this.tooltip?.hide();
    this.canvas.style.cursor = "";
    if (this.hoverName != null || this.hoverMark != null) {
      this.hoverName = null;
      this.hoverMark = null;
      this.pushLayers();
    }
  }

  private viewState() {
    const zoom = Math.log2(
      Math.min((this.cssW - PAD_PX) / SPAN_X, (this.cssH - PAD_PX) / SPAN_Y),
    );
    return {
      ortho: { target: [SPAN_X / 2, SPAN_Y / 2, 0] as [number, number, number], zoom },
    };
  }

  frame(_dt: number, _t: number): void {
    // deck renders on demand; the spectrum is static (no data-bearing motion)
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
