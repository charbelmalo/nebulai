/** #2b OV Eigenvalue Constellation — every complex eigenvalue of every
 *  attention head's residual-space OV map, plotted in ℂ. An eigenvalue λ
 *  means: along its eigendirection the head writes back λ× what it reads —
 *  positive real copies, negative real inverts, |λ|>1 amplifies. This is the
 *  full spectrum behind the head fingerprints' copying scalar, and it shows
 *  what the scalar hides: L11H8 scores a mild +0.29 yet contains a single
 *  λ = −87.5 (one massively inverted direction), while L11H3's entire
 *  spectrum sits in [+3.3, +9.6] — a uniform copying amplifier.
 *
 *  Log-polar plot: angle = arg λ (exact), radius = log₁₀|λ| over a STATED
 *  clamp window [−2, +2] (|λ| from 0.01 to 100; 0.2% of points clamp at the
 *  center). The |λ| = 1 unit circle — the amplify/attenuate boundary — is the
 *  emphasized ring. Conjugate symmetry (mirror across the real axis) is a
 *  property of the real OV matrix, not decoration.
 *
 *  deck.gl (WebGL2), camera off, static. Source: ov_eigs.json (float64 eig at
 *  d_head×d_head, verified == full 768×768 eigendecomposition). */

import type { Deck, OrthographicView, PickingInfo } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { type OVEigsBundle, loadOVEigs } from "../../data/interp";
import type { InterpDriver } from "./InterpDriver";
import { LAYER_COLORS } from "./NeuronFieldDriver";

type LayersModule = typeof import("@deck.gl/layers");

const LOG_LO = -2; // |λ| = 0.01 at the center
const LOG_HI = 2; // |λ| = 100 at the rim
const DECADES = [-1, 0, 1, 2]; // gridline rings: |λ| = 0.1, 1, 10, 100
const MARGIN = 56; // px around the outer ring for labels
const DIM_RGBA: [number, number, number, number] = [118, 126, 158, 26];
const GRID_RGBA: [number, number, number, number] = [118, 126, 158, 30];
const UNIT_RGBA: [number, number, number, number] = [166, 173, 200, 80];

interface EigPt {
  position: [number, number];
  re: number;
  im: number;
  mag: number;
  argDeg: number;
  head: number; // 0..n-1, layer-major
  layer: number;
  headIdx: number; // h within its layer
  rank: number; // 0 = largest |λ| of its head
  id: number;
}

interface Seg {
  source: [number, number];
  target: [number, number];
  unit: boolean;
}

export class OVEigenDriver implements InterpDriver {
  readonly animated = false;
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private canvas!: HTMLCanvasElement;
  private tooltip!: HTMLElement;
  private labelRoot!: HTMLElement;
  private chipRoot!: HTMLElement;

  private bundle: OVEigsBundle | null = null;
  private pts: EigPt[] = []; // deterministic-shuffled draw order
  private byLayer: EigPt[][] = [];
  private byHead: EigPt[][] = [];
  private isolateLayer: number | null = null;
  private isolateHead: number | null = null;
  private dimPts: EigPt[] = [];
  private hover: EigPt | null = null;

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

  async setModel(model: string, _trace?: string): Promise<void> {
    const b = await loadOVEigs(model);
    this.bundle = b;
    const dh = b.d_head;
    const H = b.meta.n_head;
    const N = b.n * dh;
    const ordered: EigPt[] = new Array(N);
    const byLayer: EigPt[][] = Array.from({ length: b.meta.n_layer }, () => []);
    const byHead: EigPt[][] = Array.from({ length: b.n }, () => []);
    for (let i = 0; i < N; i++) {
      const re = b.re[i] ?? 0;
      const im = b.im[i] ?? 0;
      const head = Math.floor(i / dh);
      const p: EigPt = {
        position: [0, 0],
        re,
        im,
        mag: Math.hypot(re, im),
        argDeg: (Math.atan2(im, re) * 180) / Math.PI,
        head,
        layer: Math.floor(head / H),
        headIdx: head % H,
        rank: i % dh,
        id: i,
      };
      ordered[i] = p;
      byLayer[p.layer]?.push(p);
      byHead[head]?.push(p);
    }
    // deterministic draw-order shuffle: layer-ordered data would always paint
    // late layers on top (z-order bias = a truthfulness bug, same lesson as
    // the neuron field). 5323 is coprime with any n here; the permutation is
    // stable across loads.
    const pts: EigPt[] = new Array(N);
    for (let i = 0; i < N; i++) pts[i] = ordered[(i * 5323) % N] as EigPt;
    this.pts = pts;
    this.byLayer = byLayer;
    this.byHead = byHead;
    this.isolateLayer = null;
    this.isolateHead = null;
    this.dimPts = [];
    this.hover = null;

    this.layoutPoints();
    this.buildChips();
    this.deck?.setProps({ viewState: this.viewState() });
    this.pushLayers();
    this.positionLabels();
  }

  // ---- log-polar layout ------------------------------------------------------
  private center(): [number, number] {
    return [this.cssW / 2, this.cssH / 2 + 8];
  }
  private R(): number {
    return Math.max(40, Math.min(this.cssW, this.cssH) / 2 - MARGIN);
  }
  private rOf(mag: number): number {
    const lg = Math.log10(Math.max(mag, 1e-12));
    const t = (Math.min(Math.max(lg, LOG_LO), LOG_HI) - LOG_LO) / (LOG_HI - LOG_LO);
    return t * this.R();
  }
  private layoutPoints(): void {
    const [cx, cy] = this.center();
    for (const p of this.pts) {
      const r = this.rOf(p.mag);
      const th = Math.atan2(p.im, p.re);
      p.position = [cx + r * Math.cos(th), cy - r * Math.sin(th)];
    }
  }
  private viewState() {
    return {
      ortho: {
        target: [this.cssW / 2, this.cssH / 2, 0] as [number, number, number],
        zoom: 0,
      },
    };
  }

  private gridSegs(): Seg[] {
    const [cx, cy] = this.center();
    const segs: Seg[] = [];
    const STEPS = 96;
    for (const d of DECADES) {
      const r = ((d - LOG_LO) / (LOG_HI - LOG_LO)) * this.R();
      const unit = d === 0;
      for (let k = 0; k < STEPS; k++) {
        const a0 = (k / STEPS) * 2 * Math.PI;
        const a1 = ((k + 1) / STEPS) * 2 * Math.PI;
        segs.push({
          source: [cx + r * Math.cos(a0), cy - r * Math.sin(a0)],
          target: [cx + r * Math.cos(a1), cy - r * Math.sin(a1)],
          unit,
        });
      }
    }
    for (let k = 0; k < 8; k++) {
      const a = (k / 8) * 2 * Math.PI;
      segs.push({
        source: [cx, cy],
        target: [cx + this.R() * Math.cos(a), cy - this.R() * Math.sin(a)],
        unit: false,
      });
    }
    return segs;
  }

  private activePts(): EigPt[] {
    if (this.isolateHead !== null) return this.byHead[this.isolateHead] ?? [];
    if (this.isolateLayer !== null) return this.byLayer[this.isolateLayer] ?? [];
    return this.pts;
  }

  private pushLayers(): void {
    if (!this.deck || !this.pts.length) return;
    const { ScatterplotLayer, LineLayer } = this.layersMod;
    const hoverHead = this.hover ? (this.byHead[this.hover.head] ?? []) : [];

    this.deck.setProps({
      layers: [
        new LineLayer<Seg>({
          id: "ov-grid",
          data: this.gridSegs(),
          getSourcePosition: (s) => [s.source[0], s.source[1], 0],
          getTargetPosition: (s) => [s.target[0], s.target[1], 0],
          getColor: (s) => (s.unit ? UNIT_RGBA : GRID_RGBA),
          getWidth: (s) => (s.unit ? 1.4 : 1),
          widthUnits: "pixels",
          pickable: false,
        }),
        new ScatterplotLayer<EigPt>({
          id: "ov-dim",
          data: this.dimPts,
          getPosition: (p) => [p.position[0], p.position[1], 0],
          getFillColor: DIM_RGBA,
          getRadius: 2,
          radiusUnits: "pixels",
          pickable: false,
        }),
        new ScatterplotLayer<EigPt>({
          id: "ov-active",
          data: this.activePts(),
          getPosition: (p) => [p.position[0], p.position[1], 0],
          getFillColor: (p) => {
            const [r, g, b] = LAYER_COLORS[p.layer] ?? [205, 210, 224];
            return [r, g, b, 150];
          },
          getRadius: 2.2,
          radiusUnits: "pixels",
          pickable: true,
        }),
        // the hovered head's FULL spectrum, brightened — one head's 64
        // eigenvalues are the meaningful unit here, not a lone point
        new ScatterplotLayer<EigPt>({
          id: "ov-hover-head",
          data: hoverHead,
          getPosition: (p) => [p.position[0], p.position[1], 0],
          getFillColor: (p) => {
            const [r, g, b] = LAYER_COLORS[p.layer] ?? [205, 210, 224];
            return [r, g, b, 245];
          },
          getRadius: 3,
          radiusUnits: "pixels",
          pickable: false,
        }),
        new ScatterplotLayer<EigPt>({
          id: "ov-hover",
          data: this.hover ? [this.hover] : [],
          getPosition: (p) => [p.position[0], p.position[1], 0],
          getFillColor: [255, 255, 255, 235],
          getLineColor: [12, 14, 22, 235],
          stroked: true,
          lineWidthUnits: "pixels",
          getLineWidth: 1.4,
          getRadius: 4.2,
          radiusUnits: "pixels",
          pickable: false,
        }),
      ],
    });
  }

  private setIsolate(layer: number | null, head: number | null): void {
    this.isolateLayer = layer;
    this.isolateHead = head;
    if (head !== null) {
      const keep = new Set(this.byHead[head]?.map((p) => p.id));
      this.dimPts = this.pts.filter((p) => !keep.has(p.id));
    } else if (layer !== null) {
      this.dimPts = this.pts.filter((p) => p.layer !== layer);
    } else {
      this.dimPts = [];
    }
    this.hover = null;
    this.tooltip.style.visibility = "hidden";
    this.buildChips();
    this.pushLayers();
    this.positionLabels();
  }

  private buildChips(): void {
    this.chipRoot.textContent = "";
    if (!this.bundle) return;
    const mk = (label: string, active: boolean, onClick: () => void, dot?: number) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "interp-neuron-chip";
      btn.textContent = label;
      btn.setAttribute("aria-pressed", String(active));
      if (active) btn.classList.add("is-active");
      if (dot !== undefined) {
        const [r, g, b] = LAYER_COLORS[dot] ?? [205, 210, 224];
        btn.style.setProperty("--chip-dot", `rgb(${r},${g},${b})`);
      }
      btn.addEventListener("click", onClick);
      this.chipRoot.appendChild(btn);
    };
    const noIso = this.isolateLayer === null && this.isolateHead === null;
    mk("all", noIso, () => this.setIsolate(null, null));
    for (let l = 0; l < this.bundle.meta.n_layer; l++) {
      mk(`L${l}`, this.isolateLayer === l, () => this.setIsolate(this.isolateLayer === l ? null : l, null), l);
    }
    if (this.isolateHead !== null) {
      const H = this.bundle.meta.n_head;
      const hd = this.isolateHead;
      mk(`L${Math.floor(hd / H)}H${hd % H} ×`, true, () => this.setIsolate(null, null), Math.floor(hd / H));
    }
  }

  private positionLabels(): void {
    this.labelRoot.textContent = "";
    if (!this.bundle) return;
    const [cx, cy] = this.center();
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

    // decade ring labels along the NE ray (45° keeps them off the data axes)
    for (const d of DECADES) {
      const r = ((d - LOG_LO) / (LOG_HI - LOG_LO)) * this.R();
      const el = cap(d === 0 ? "|λ|=1" : `${10 ** d}`);
      const a = Math.PI / 4;
      place(el, cx + r * Math.cos(a) + 4, cy - r * Math.sin(a) - 16);
    }
    // axis direction tags — what the angle MEANS
    const east = cap("+Re → copies");
    place(east, cx + this.R() + 8, cy - 8);
    const west = cap("inverts ← −Re");
    this.labelRoot.appendChild(west);
    place(west, cx - this.R() - 8 - west.offsetWidth, cy - 8);
    const north = cap("+Im (rotates)");
    place(north, cx - north.offsetWidth / 2, cy - this.R() - 24);

    // anchors: the spectrum-wide real extremes, factual pointers
    let maxRe = this.pts[0] as EigPt;
    let minRe = this.pts[0] as EigPt;
    for (const p of this.pts) {
      if (p.re > maxRe.re) maxRe = p;
      if (p.re < minRe.re) minRe = p;
    }
    for (const [p, tag] of [
      [maxRe, "max Re λ"],
      [minRe, "min Re λ"],
    ] as Array<[EigPt, string]>) {
      const [r, g, b] = LAYER_COLORS[p.layer] ?? [205, 210, 224];
      const el = cap(`L${p.layer}H${p.headIdx} ${tag} ${p.re.toFixed(1)}`, "interp-neuron-anchor");
      el.style.color = `rgb(${r},${g},${b})`;
      const w = el.offsetWidth;
      const [sx, sy] = p.position;
      place(el, sx + 8 + w > this.cssW - 8 ? sx - w - 8 : sx + 8, sy - 9);
    }
  }

  private pick(e: PointerEvent): EigPt | null {
    if (!this.deck) return null;
    const rect = this.canvas.getBoundingClientRect();
    const info = this.deck.pickObject({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      radius: 4,
      layerIds: ["ov-active"],
    }) as PickingInfo | null;
    return (info?.object as EigPt | undefined) ?? null;
  }

  private onClick(e: PointerEvent): void {
    const p = this.pick(e);
    if (p) {
      // click a point → isolate its head's full spectrum; click again clears
      this.setIsolate(null, this.isolateHead === p.head ? null : p.head);
    } else if (this.isolateHead !== null) {
      this.setIsolate(null, null);
    }
  }

  private onPointerMove(e: PointerEvent): void {
    const p = this.pick(e);
    const changed = (p?.id ?? -1) !== (this.hover?.id ?? -1);
    if (changed) {
      this.hover = p;
      this.pushLayers();
    }
    if (!p || !this.bundle) {
      this.tooltip.style.visibility = "hidden";
      this.canvas.style.cursor = "";
      return;
    }
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    this.tooltip.innerHTML = "";
    const add = (cls: string, text: string) => {
      const el = document.createElement("div");
      el.className = cls;
      el.textContent = text;
      this.tooltip.appendChild(el);
    };
    const cp = this.bundle.copying[p.head] ?? 0;
    const sgn = p.im >= 0 ? "+" : "−";
    add("point-tooltip-label", `L${p.layer} · head ${p.headIdx} · λ${p.rank + 1} of ${this.bundle.d_head}`);
    add("point-tooltip-conf", `λ = ${p.re.toFixed(4)} ${sgn} ${Math.abs(p.im).toFixed(4)}i`);
    add("point-tooltip-conf", `|λ| = ${p.mag.toFixed(4)} · arg = ${p.argDeg.toFixed(1)}°`);
    add("point-tooltip-conf", `head copying ${cp >= 0 ? "+" : ""}${cp.toFixed(3)} (Σ Re λ / Σ |λ|)`);
    add("point-tooltip-conf", "click to isolate this head's spectrum");
    this.tooltip.style.visibility = "visible";
    const px = Math.min(x + 14, this.cssW - 280);
    const py = Math.min(y + 14, this.cssH - 110);
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
    // static — a weight property; nothing animates
  }

  resize(width: number, height: number, dpr: number): void {
    this.cssW = width;
    this.cssH = height;
    this.dpr = dpr;
    this.layoutPoints();
    this.deck?.setProps({
      width,
      height,
      useDevicePixels: Math.min(dpr, 2),
      viewState: this.viewState(),
    });
    // positions changed in place — new refs force attribute regeneration
    this.pts = this.pts.slice();
    this.byLayer = this.byLayer.map((l) => l.slice());
    this.byHead = this.byHead.map((h) => h.slice());
    if (this.dimPts.length) this.dimPts = this.dimPts.slice();
    this.pushLayers();
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
