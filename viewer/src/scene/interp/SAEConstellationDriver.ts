/** #5 SAE Decoder Constellation — all 24,576 features of an open sparse
 *  autoencoder trained on GPT-2's layer-8 residual stream (res-jb release),
 *  each placed at the exact PCA score of its DECODER DIRECTION: row i of W_dec,
 *  the vector the feature adds to the residual stream scaled by its activation.
 *  Positions are real PC scores (same _pca_rows math as the embedding and
 *  neuron constellations — the three views are directly comparable projections
 *  of W_E rows, W_out rows, and W_dec rows).
 *
 *  Decoder rows are unit-norm by construction (‖W_dec[i]‖₂ ∈ [0.9998, 1.0013]),
 *  so size/brightness encode the release's MEASURED log₁₀ firing sparsity — a
 *  real activation statistic over the SAE's evaluation set — never the norm.
 *  Hover gives the exact log₁₀ sparsity (features at the −10 floor are dead)
 *  and the direct-path unembedding readout, with the honest caveat that these
 *  directions enter at layer 8 and skip blocks 8–11 on the direct path.
 *
 *  deck.gl (WebGL2), camera off, static (redraws on hover / band isolate).
 *  Source: sae.json — PCA computed offline in float64; sparsity from the
 *  release's sparsity.safetensors; readout through the tied W_E. */

import type { Deck, OrthographicView, PickingInfo } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { type SAEBundle, loadSAE } from "../../data/interp";
import type { InterpDriver } from "./InterpDriver";

type LayersModule = typeof import("@deck.gl/layers");

const GL = 60;
const GR = 60;
const GT = 78;
const GB = 88; // room for the sparsity-band chip row
const FIT = 0.94;

/** Viridis stops (t = 0.25 + 0.75·k/11, low end clipped for the dark stage) —
 *  the SAME ramp the neuron field uses per layer, here interpolated
 *  continuously over normalized log₁₀ sparsity. Luminance strictly ↑ with
 *  firing rate: brighter = fires more often. */
const RAMP: [number, number, number][] = [
  [59, 82, 138],
  [51, 100, 141],
  [44, 117, 142],
  [37, 134, 141],
  [33, 150, 138],
  [34, 166, 133],
  [54, 181, 120],
  [83, 195, 104],
  [119, 208, 83],
  [165, 218, 53],
  [209, 226, 38],
  [253, 231, 37],
];

/** Decade thresholds for the isolate chips — stated in the chip labels. */
const FREQ_LOG = -2; // fires on ≥1% of tokens
const RARE_LOG = -4; // fires on <0.01% of tokens
type Band = "all" | "freq" | "mid" | "rare";

/** Fixed, stated color/size window: log₁₀ sparsity −6 … −1, clamped. The raw
 *  range runs to the −10 dead-feature floor, but only ~29 of 24,576 features
 *  sit below −6 — normalizing over the floor would compress the ramp onto the
 *  top fifth and render everything bright. An explicit decade window (declared
 *  in the legend note) keeps the encoding readable; hover always gives the
 *  exact value. */
const RAMP_LO = -6;
const RAMP_HI = -1;

interface FeaturePt {
  position: [number, number];
  z: number; // PC3 (hover only)
  logSp: number; // measured log10 firing fraction (−10 = floor/dead)
  norm: number; // ‖W_dec[i]‖₂ (≈1 by construction — hover only)
  topTok: string;
  topVal: number;
  botTok: string;
  botVal: number;
  id: number; // feature index
}

export class SAEConstellationDriver implements InterpDriver {
  readonly animated = false;
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private canvas!: HTMLCanvasElement;
  private tooltip!: HTMLElement;
  private labelRoot!: HTMLElement;
  private chipRoot!: HTMLElement;

  private bundle: SAEBundle | null = null;
  private pts: FeaturePt[] = [];
  private anchors: FeaturePt[] = [];
  private band: Band = "all";
  /** stable per-band arrays, computed once at load — hover-driven pushLayers
   *  must reuse identical refs so deck skips attribute regeneration. */
  private bandPts: Record<Band, FeaturePt[]> = { all: [], freq: [], mid: [], rare: [] };
  private bandDim: Record<Band, FeaturePt[]> = { all: [], freq: [], mid: [], rare: [] };
  private minX = 0;
  private maxX = 1;
  private minY = 0;
  private maxY = 1;
  private hover: FeaturePt | null = null;

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
      views: [new core.OrthographicView({ id: "ortho", flipY: false })],
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

  async setModel(model: string, _trace?: string): Promise<void> {
    const b = await loadSAE(model);
    this.bundle = b;
    const n = b.n;
    const c = b.coords;
    const pts: FeaturePt[] = new Array(n);
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = c[i * 2] ?? 0;
      const y = c[i * 2 + 1] ?? 0;
      const ls = b.log_sparsity[i] ?? -10;
      pts[i] = {
        position: [x, y],
        z: b.z[i] ?? 0,
        logSp: ls,
        norm: b.norm[i] ?? 1,
        topTok: b.top_tok[i] ?? "",
        topVal: b.top_val[i] ?? 0,
        botTok: b.bot_tok[i] ?? "",
        botVal: b.bot_val[i] ?? 0,
        id: i,
      };
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    this.pts = pts;
    this.minX = minX;
    this.maxX = maxX;
    this.minY = minY;
    this.maxY = maxY;
    this.hover = null;
    this.band = "all";

    // per-band splits (stable refs). PCA order carries no draw bias here —
    // feature index is arbitrary — so no shuffle is needed; still, bands are
    // filtered once so band switches and hovers never re-filter 24k points.
    const freq = pts.filter((p) => p.logSp >= FREQ_LOG);
    const mid = pts.filter((p) => p.logSp < FREQ_LOG && p.logSp >= RARE_LOG);
    const rare = pts.filter((p) => p.logSp < RARE_LOG);
    this.bandPts = { all: pts, freq, mid, rare };
    this.bandDim = {
      all: [],
      freq: pts.filter((p) => p.logSp < FREQ_LOG),
      mid: pts.filter((p) => !(p.logSp < FREQ_LOG && p.logSp >= RARE_LOG)),
      rare: pts.filter((p) => p.logSp >= RARE_LOG),
    };

    // landmarks: real extremes only — PC1/PC2 range ends + the most-firing
    // feature, labelled with the token its decoder direction most promotes.
    const set = new Map<number, FeaturePt>();
    if (pts.length) {
      const extreme = (f: (p: FeaturePt) => number, sign: number): FeaturePt => {
        let best = pts[0] as FeaturePt;
        for (const p of pts) if (sign * f(p) > sign * f(best)) best = p;
        return best;
      };
      for (const p of [
        extreme((p) => p.position[0], +1),
        extreme((p) => p.position[0], -1),
        extreme((p) => p.position[1], +1),
        extreme((p) => p.position[1], -1),
        extreme((p) => p.logSp, +1),
      ]) {
        set.set(p.id, p);
      }
    }
    this.anchors = [...set.values()];

    this.buildChips();
    this.deck?.setProps({ viewState: this.viewState() });
    this.pushLayers();
    this.positionLabels();
  }

  /** normalized firing rate ∈ [0,1] over the FIXED stated window [−6, −1] */
  private tOf(logSp: number): number {
    return Math.min(1, Math.max(0, (logSp - RAMP_LO) / (RAMP_HI - RAMP_LO)));
  }

  private radiusOf(p: FeaturePt): number {
    return 0.8 + this.tOf(p.logSp) * 2.6; // ∝ measured firing rate, NOT norm
  }

  private colorOf(p: FeaturePt): [number, number, number] {
    const t = this.tOf(p.logSp) * (RAMP.length - 1);
    const k = Math.min(RAMP.length - 2, Math.floor(t));
    const f = t - k;
    const a = RAMP[k] as [number, number, number];
    const b = RAMP[k + 1] as [number, number, number];
    return [
      Math.round(a[0] + (b[0] - a[0]) * f),
      Math.round(a[1] + (b[1] - a[1]) * f),
      Math.round(a[2] + (b[2] - a[2]) * f),
    ];
  }

  private pushLayers(): void {
    if (!this.deck || !this.pts.length) return;
    const { ScatterplotLayer } = this.layersMod;
    const active = this.bandPts[this.band];
    const dimmed = this.bandDim[this.band];

    this.deck.setProps({
      layers: [
        new ScatterplotLayer<FeaturePt>({
          id: "sae-dim",
          data: dimmed,
          getPosition: (p) => [p.position[0], p.position[1], 0],
          getFillColor: [118, 126, 158, 22],
          getRadius: (p) => this.radiusOf(p),
          radiusUnits: "pixels",
          pickable: false,
        }),
        new ScatterplotLayer<FeaturePt>({
          id: "sae-active",
          data: active,
          getPosition: (p) => [p.position[0], p.position[1], 0],
          getFillColor: (p) => {
            const [r, g, bl] = this.colorOf(p);
            return [r, g, bl, Math.round(70 + this.tOf(p.logSp) * 110)];
          },
          getRadius: (p) => this.radiusOf(p),
          radiusUnits: "pixels",
          pickable: true,
        }),
        new ScatterplotLayer<FeaturePt>({
          id: "sae-hover",
          data: this.hover ? [this.hover] : [],
          getPosition: (p) => [p.position[0], p.position[1], 0],
          getFillColor: [255, 255, 255, 235],
          getLineColor: [12, 14, 22, 235],
          stroked: true,
          lineWidthUnits: "pixels",
          getLineWidth: 1.4,
          getRadius: (p) => this.radiusOf(p) + 1.6,
          radiusUnits: "pixels",
          pickable: false,
        }),
      ],
    });
  }

  private buildChips(): void {
    this.chipRoot.textContent = "";
    if (!this.bundle) return;
    const defs: Array<{ band: Band; label: string }> = [
      { band: "all", label: "all" },
      { band: "freq", label: `≥10⁻² · ${this.bandPts.freq.length}` },
      { band: "mid", label: `10⁻⁴–10⁻² · ${this.bandPts.mid.length}` },
      { band: "rare", label: `<10⁻⁴ · ${this.bandPts.rare.length}` },
    ];
    for (const { band, label } of defs) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "interp-neuron-chip";
      btn.textContent = label;
      btn.setAttribute("aria-pressed", String(this.band === band));
      if (this.band === band) btn.classList.add("is-active");
      btn.addEventListener("click", () => {
        this.band = this.band === band ? "all" : band;
        this.hover = null;
        this.tooltip.style.visibility = "hidden";
        this.buildChips();
        this.pushLayers();
      });
      this.chipRoot.appendChild(btn);
    }
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

    const cap = (text: string, sx: number, sy: number) => {
      const el = document.createElement("div");
      el.className = "interp-neuron-axis";
      el.textContent = text;
      el.style.transform = `translate(${sx.toFixed(1)}px, ${sy.toFixed(1)}px)`;
      this.labelRoot.appendChild(el);
    };
    const [rx] = this.worldToScreen(this.maxX, this.dataCY());
    const [, by] = this.worldToScreen(this.dataCX(), this.minY);
    cap(`PC1 → · ${pc1}% var`, rx - 96, by + 14);
    const [tx, ty] = this.worldToScreen(this.dataCX(), this.maxY);
    cap(`PC2 ↑ · ${pc2}% var`, tx + 8, ty + 2);

    for (const p of this.anchors) {
      const [sx, sy] = this.worldToScreen(p.position[0], p.position[1]);
      const [r, g, b] = this.colorOf(p);
      const el = document.createElement("div");
      el.className = "interp-neuron-anchor";
      el.textContent = `#${p.id} ↑${fmtTok(p.topTok)}`;
      el.style.color = `rgb(${r},${g},${b})`;
      this.labelRoot.appendChild(el);
      const w = el.offsetWidth;
      const x = sx + 6 + w > this.cssW - 8 ? sx - w - 6 : sx + 6;
      el.style.transform = `translate(${x.toFixed(1)}px, ${(sy - 8).toFixed(1)}px)`;
    }
  }

  private onPointerMove(e: PointerEvent): void {
    if (!this.deck) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const info = this.deck.pickObject({ x, y, radius: 5, layerIds: ["sae-active"] }) as
      | PickingInfo
      | null;
    const p = (info?.object as FeaturePt | undefined) ?? null;
    const changed = (p?.id ?? -1) !== (this.hover?.id ?? -1);
    if (changed) {
      this.hover = p;
      this.pushLayers();
    }
    if (!p) {
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
    add("point-tooltip-label", `SAE feature ${p.id}`);
    add(
      "point-tooltip-conf",
      `PC1 ${p.position[0].toFixed(2)} · PC2 ${p.position[1].toFixed(2)} · PC3 ${p.z.toFixed(2)}`,
    );
    add("point-tooltip-conf", sparsityLine(p.logSp));
    add("point-tooltip-conf", `promotes “${fmtTok(p.topTok)}” · Δlogit +${p.topVal.toFixed(2)}`);
    add("point-tooltip-conf", `suppresses “${fmtTok(p.botTok)}” · Δlogit ${p.botVal.toFixed(2)}`);
    this.tooltip.style.visibility = "visible";
    const px = Math.min(x + 14, this.cssW - 250);
    const py = Math.min(y + 14, this.cssH - 96);
    this.tooltip.style.transform = `translate(${px.toFixed(1)}px, ${py.toFixed(1)}px)`;
    this.canvas.style.cursor = "crosshair";
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
    // static — one fixed projection of the decoder directions, no data-bearing motion
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
    this.tooltip?.remove();
    this.labelRoot?.remove();
    this.chipRoot?.remove();
    this.deck?.finalize();
    this.deck = null;
  }
}

function fmtTok(s: string): string {
  return s.replace(/^ /, "␣").replace(/\n/g, "⏎") || "∅";
}

/** Exact sparsity line: log₁₀ value + the firing fraction it implies. The
 *  release clamps at −10; features there are dead (never fired in eval). */
function sparsityLine(logSp: number): string {
  if (logSp <= -9.9) return "log₁₀ sparsity ≤ −10 (floor) · dead — never fired";
  const pct = 10 ** logSp * 100;
  const fmt =
    pct >= 0.01 ? `${pct.toFixed(2)}%` : `${pct.toExponential(1).replace("e-", "e−")}%`;
  return `log₁₀ sparsity ${logSp.toFixed(2)} · fires on ≈${fmt} of tokens`;
}
