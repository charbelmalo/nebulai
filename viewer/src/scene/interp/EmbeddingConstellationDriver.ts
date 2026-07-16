/** #15 Embedding Constellation — every token embedding W_E[i] as a star, placed
 *  at its exact PCA score on the top-2 principal axes of the mean-centered
 *  embedding matrix. Nothing here is a synthetic layout: positions are real PC
 *  scores, star size is the real row L2 norm ‖W_E[i]‖₂, and color is a real
 *  orthographic property — whether the token string begins with a space —
 *  decoded per token. The honest finding it makes visible: GPT-2's leading
 *  embedding axes organize tokens largely by SURFACE FORM (leading space, case,
 *  digits, the commonest function words at the PC1 extreme), not meaning — and
 *  the top 2 PCs explain only ~2.6% of the variance, so this is deliberately
 *  shown as a low-dimensional shadow of a genuinely high-dimensional space.
 *
 *  Axes are drawn to a single isometric scale (equal px per PC unit) so on-screen
 *  distances are faithful; PC1 simply spans a wider range than PC2 because it
 *  carries more variance. deck.gl (WebGL2), camera off, static (redraw on hover).
 *
 *  Source: embed.json → PCA of W_E computed offline in float64 (eigendecomp of
 *  the 768×768 covariance; coords = Wc·V). */

import type { Deck, OrthographicView, PickingInfo } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { type EmbedBundle, loadEmbed } from "../../data/interp";
import {
  ACCENT,
  crosshair,
  MARKER_HOT,
  markerPoly,
  type Seg as ThemeSeg,
  type Vec2,
  withAlpha,
} from "./chart-theme";
import { InterpTooltip, type TipRow } from "./chart-tooltip";
import type { InterpDriver } from "./InterpDriver";

type LayersModule = typeof import("@deck.gl/layers");

const GL = 60; // px gutters (axis captions + anchor labels)
const GR = 60;
const GT = 78;
const GB = 64;
const FIT = 0.94; // leave a little breathing room around the data bbox

const SPACE: [number, number, number] = [245, 190, 92]; // leading-space token (warm)
const NOSPACE: [number, number, number] = [92, 198, 236]; // non-space token (cool)

interface Star {
  position: [number, number];
  z: number; // PC3 (hover only)
  norm: number;
  lead: number; // 1 = leading space
  str: string;
  id: number;
}

export class EmbeddingConstellationDriver implements InterpDriver {
  readonly animated = false;
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private makeView!: () => OrthographicView;
  private canvas!: HTMLCanvasElement;
  private tooltip!: InterpTooltip;
  private labelRoot!: HTMLElement;

  private bundle: EmbedBundle | null = null;
  private stars: Star[] = [];
  private anchors: Star[] = []; // a few extreme tokens, labelled to orient the eye
  private minX = 0;
  private maxX = 1;
  private minY = 0;
  private maxY = 1;
  private normMin = 0;
  private normMax = 1;
  private hover: Star | null = null;

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
    this.labelRoot = document.createElement("div");
    this.labelRoot.className = "interp-embed-labels";
    overlay.appendChild(this.labelRoot);

    const onMove = (e: PointerEvent) => this.onPointerMove(e);
    const onLeave = () => this.onLeave();
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    this.disposers.push(() => {
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
    });
  }

  async setModel(model: string, _trace?: string): Promise<void> {
    const b = await loadEmbed(model);
    this.bundle = b;
    const n = b.n;
    const c = b.coords; // flat 2n
    const stars: Star[] = new Array(n);
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    let nmin = Infinity;
    let nmax = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = c[i * 2] ?? 0;
      const y = c[i * 2 + 1] ?? 0;
      const nm = b.norm[i] ?? 0;
      stars[i] = { position: [x, y], z: b.z[i] ?? 0, norm: nm, lead: b.lead_space[i] ?? 0, str: b.strs[i] ?? "", id: i };
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (nm < nmin) nmin = nm;
      if (nm > nmax) nmax = nm;
    }
    this.stars = stars;
    this.minX = minX;
    this.maxX = maxX;
    this.minY = minY;
    this.maxY = maxY;
    this.normMin = nmin;
    this.normMax = nmax;
    this.hover = null;

    // orient the eye with a handful of REAL extremes (no cherry-picking of the
    // interior): the tokens at the PC1/PC2 range ends and the largest-norm star.
    const set = new Map<number, Star>();
    if (stars.length) {
      const extreme = (f: (s: Star) => number, sign: number): Star => {
        let best = stars[0] as Star;
        for (const s of stars) if (sign * f(s) > sign * f(best)) best = s;
        return best;
      };
      for (const s of [
        extreme((s) => s.position[0], +1),
        extreme((s) => s.position[0], -1),
        extreme((s) => s.position[1], +1),
        extreme((s) => s.position[1], -1),
        extreme((s) => s.norm, +1),
      ]) {
        set.set(s.id, s);
      }
    }
    this.anchors = [...set.values()];

    this.deck?.setProps({ viewState: this.viewState() });
    this.pushLayers();
    this.positionLabels();
  }

  private radiusOf(norm: number): number {
    const t = (norm - this.normMin) / Math.max(1e-6, this.normMax - this.normMin);
    return 1.2 + t * 2.4; // 1.2 .. 3.6 px, ∝ real embedding norm
  }

  private pushLayers(): void {
    if (!this.deck || !this.stars.length) return;
    const { ScatterplotLayer, LineLayer, SolidPolygonLayer } = this.layersMod;
    const colorOf = (s: Star): [number, number, number] => (s.lead ? SPACE : NOSPACE);

    // markers/crosshair live in world (PC) space, so pixel-authored sizes are
    // scaled by world-units-per-pixel (mirrors the WeightSpectrum template).
    const wpp = 1 / this.zoomPx();
    const bounds = { x0: this.minX, y0: this.minY, x1: this.maxX, y1: this.maxY };
    // crosshair guides snap onto the hovered star (req 4)
    const cross: ThemeSeg[] = this.hover
      ? crosshair(this.hover.position[0], this.hover.position[1], bounds, 3 * wpp, 4 * wpp)
      : [];
    // hover LED diamond (translucent glow under a full-alpha core) replaces the
    // old white outline ring (req 4)
    interface Marker {
      poly: Vec2[];
      color: [number, number, number, number];
    }
    const mr = this.hover ? (this.radiusOf(this.hover.norm) + 2) * wpp : 0;
    const marks: Marker[] = this.hover
      ? [
          {
            poly: markerPoly(this.hover.position[0], this.hover.position[1], mr * 2.1),
            color: withAlpha(MARKER_HOT, 0.22),
          },
          {
            poly: markerPoly(this.hover.position[0], this.hover.position[1], mr),
            color: withAlpha(MARKER_HOT, 1),
          },
        ]
      : [];

    this.deck.setProps({
      layers: [
        // soft glow: radius/alpha ∝ real norm — the "bright stars" are the
        // highest-magnitude embeddings, not a decorative pick.
        new ScatterplotLayer<Star>({
          id: "embed-halo",
          data: this.stars,
          getPosition: (s) => [s.position[0], s.position[1], 0],
          getFillColor: (s) => {
            const [r, g, bl] = colorOf(s);
            const t = (s.norm - this.normMin) / Math.max(1e-6, this.normMax - this.normMin);
            return [r, g, bl, Math.round(10 + t * 30)];
          },
          getRadius: (s) => this.radiusOf(s.norm) * 2.3,
          radiusUnits: "pixels",
          pickable: false,
        }),
        new ScatterplotLayer<Star>({
          id: "embed-stars",
          data: this.stars,
          // the field dims to defer to the focused marker on hover (req 3)
          opacity: this.hover ? 0.38 : 1,
          getPosition: (s) => [s.position[0], s.position[1], 0],
          getFillColor: (s) => {
            const [r, g, bl] = colorOf(s);
            return [r, g, bl, 165];
          },
          getRadius: (s) => this.radiusOf(s.norm),
          radiusUnits: "pixels",
          pickable: true,
        }),
        new LineLayer<ThemeSeg>({
          id: "embed-crosshair",
          data: cross,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: withAlpha(ACCENT, 0.5),
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        new SolidPolygonLayer<Marker>({
          id: "embed-marker",
          data: marks,
          getPolygon: (m) => m.poly,
          getFillColor: (m) => m.color,
          pickable: false,
        }),
      ],
    });
  }

  // ---- isometric layout: equal px per PC unit so distances stay faithful -----
  private availW(): number {
    return Math.max(1, this.cssW - GL - GR);
  }
  private availH(): number {
    return Math.max(1, this.cssH - GT - GB);
  }
  private spanX(): number {
    return Math.max(1e-3, this.maxX - this.minX);
  }
  private spanY(): number {
    return Math.max(1e-3, this.maxY - this.minY);
  }
  private zoomPx(): number {
    return Math.max(1, Math.min(this.availW() / this.spanX(), this.availH() / this.spanY()) * FIT);
  }
  private dataCX(): number {
    return (this.minX + this.maxX) / 2;
  }
  private dataCY(): number {
    return (this.minY + this.maxY) / 2;
  }
  private drawCX(): number {
    return GL + this.availW() / 2;
  }
  private drawCY(): number {
    return GT + this.availH() / 2;
  }
  private worldToScreen(wx: number, wy: number): [number, number] {
    const z = this.zoomPx();
    return [this.drawCX() + (wx - this.dataCX()) * z, this.drawCY() - (wy - this.dataCY()) * z];
  }
  private viewState() {
    const z = this.zoomPx();
    return {
      ortho: {
        target: [
          this.dataCX() + (this.cssW / 2 - this.drawCX()) / z,
          this.dataCY() + (this.drawCY() - this.cssH / 2) / z,
          0,
        ] as [number, number, number],
        zoom: Math.log2(z),
      },
    };
  }

  private positionLabels(): void {
    this.labelRoot.textContent = "";
    if (!this.bundle) return;
    const evr = this.bundle.explained_variance_ratio;
    const pc1 = ((evr[0] ?? 0) * 100).toFixed(1);
    const pc2 = ((evr[1] ?? 0) * 100).toFixed(1);

    const cap = (cls: string, html: string, sx: number, sy: number) => {
      const el = document.createElement("div");
      el.className = cls;
      el.innerHTML = html;
      el.style.transform = `translate(${sx.toFixed(1)}px, ${sy.toFixed(1)}px)`;
      this.labelRoot.appendChild(el);
    };
    // axis captions at the data extremes, on the isometric frame
    const [rx, ry] = this.worldToScreen(this.maxX, this.dataCY());
    cap("interp-embed-axis", `PC1 → · ${pc1}% var`, rx - 96, ry - 22);
    const [tx, ty] = this.worldToScreen(this.dataCX(), this.maxY);
    cap("interp-embed-axis is-v", `PC2 ↑ · ${pc2}% var`, tx + 8, ty + 2);

    // anchor labels: the real extreme tokens, so the cloud has landmarks
    for (const s of this.anchors) {
      const [sx, sy] = this.worldToScreen(s.position[0], s.position[1]);
      cap(
        `interp-embed-anchor${s.lead ? " is-space" : ""}`,
        escapeHtml(fmtTok(s.str)),
        sx + 6,
        sy - 8,
      );
    }
  }

  private onPointerMove(e: PointerEvent): void {
    if (!this.deck) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const info = this.deck.pickObject({ x, y, radius: 5, layerIds: ["embed-stars"] }) as
      | PickingInfo
      | null;
    const s = (info?.object as Star | undefined) ?? null;
    const changed = (s?.id ?? -1) !== (this.hover?.id ?? -1);
    if (changed) {
      this.hover = s;
      this.pushLayers();
    }
    if (!s) {
      this.tooltip.hide();
      this.canvas.style.cursor = "";
      return;
    }
    const [cr, cg, cb] = s.lead ? SPACE : NOSPACE;
    const rows: TipRow[] = [
      { kind: "label", text: `token “${fmtTok(s.str)}”`, swatch: [cr, cg, cb] },
      {
        text: `PC1 ${s.position[0].toFixed(2)} · PC2 ${s.position[1].toFixed(2)} · PC3 ${s.z.toFixed(2)}`,
      },
      {
        text: `‖W_E‖ = ${s.norm.toFixed(2)} · ${s.lead ? "leading space" : "no leading space"}`,
      },
    ];
    this.tooltip.show(rows);
    this.tooltip.move(x, y, this.cssW, this.cssH);
    this.canvas.style.cursor = "crosshair";
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
    // static — the constellation is one fixed projection, no data-bearing motion
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
    this.tooltip?.dispose();
    this.labelRoot?.remove();
    this.deck?.finalize();
    this.deck = null;
  }
}

function fmtTok(s: string): string {
  return s.replace(/^ /, "␣").replace(/\n/g, "⏎") || "∅";
}
function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
