/** #18 Probability Simplex — the model's next-token distribution at the final
 *  position, shown on a true 2-simplex (ternary plot) with NO renormalization.
 *
 *  The three corners are: top-1 token, top-2 token, and "all other tokens"
 *  (everything of rank ≥ 3, which absorbs the entire tail). The plotted point's
 *  barycentric coordinates are the EXACT probabilities (p₁, p₂, p_rest),
 *  p₁+p₂+p_rest = 1 — so a confident prediction sits in a corner and an
 *  unconfident one (GPT-2 small very often) sits deep toward the "other" vertex.
 *  This is the honest antidote to renormalizing the top-k: on "The Eiffel Tower
 *  is located in the city of", the top-3 capture only ~14% of the mass, so the
 *  point sits near "other", not near " Paris". The companion bars give the full
 *  top-12 with the ranks-13+ tail, so nothing is hidden. Flows across the five
 *  curated prompts via the selector (real distributions, no fake motion).
 *
 *  Source: trace_*.json → final_topk (top-12 (token,prob) at the last position,
 *  from softmax over the real final logits). deck.gl (WebGL2), camera off. */

import type { Deck, OrthographicView, PickingInfo } from "@deck.gl/core";
import type { GpuTier } from "@psychix/viz/capabilities";
import { loadTrace, type LensTopk, type TraceBundle } from "../../data/interp";
import {
  dashedSegment,
  GRID_RGBA,
  MARKER_HOT,
  markerRing,
  type Vec2,
  withAlpha,
} from "@psychix/viz/chart-theme";
import { InterpTooltip, type TipRow } from "@psychix/viz/chart-tooltip";
import type { InterpDriver } from "./InterpDriver";
import type { StatTile } from "@psychix/viz/StatStrip";

type LayersModule = typeof import("@deck.gl/layers");

const H = Math.sqrt(3) / 2; // height of the unit equilateral triangle
const SPLIT = 0.54; // fraction of stage width given to the ternary panel
const GL = 44; // px left gutter
const GT = 96; // px top gutter (vertex label + trace bar clearance)
const GB = 74; // px bottom gutter (bottom vertex labels)
const GMID = 26; // px gap between the triangle panel and the bars panel

const GOLD: [number, number, number] = [245, 195, 59]; // top-1
const STEEL: [number, number, number] = [96, 165, 224]; // top-2
const OTHER: [number, number, number] = [123, 130, 156]; // rank ≥3
const TAIL: [number, number, number] = [86, 92, 116]; // ranks 13+

interface Bar {
  rank: number; // 1-based; 0 = the tail row
  token: string;
  prob: number;
  rgb: [number, number, number];
}

export class ProbabilitySimplexDriver implements InterpDriver {
  readonly animated = false; // one static distribution per prompt; redraw on hover
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private makeView!: () => OrthographicView;
  private canvas!: HTMLCanvasElement;
  private overlay!: HTMLElement;
  private tooltip!: InterpTooltip;
  private labelRoot!: HTMLElement;
  private barRoot!: HTMLElement;

  private bundle: TraceBundle | null = null;
  private topk: LensTopk = [];
  private p1 = 0;
  private p2 = 0;
  private prest = 0;
  private tail = 0;
  private bars: Bar[] = [];
  private hoverBar = -1;
  private hoverPoint = false;

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
    this.labelRoot = document.createElement("div");
    this.labelRoot.className = "interp-simplex-labels";
    overlay.appendChild(this.labelRoot);
    this.barRoot = document.createElement("div");
    this.barRoot.className = "interp-simplex-bars";
    overlay.appendChild(this.barRoot);

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

  async setModel(model: string, trace?: string): Promise<void> {
    if (!trace) throw new Error("no forward trace selected");
    const b = await loadTrace(model, trace);
    this.bundle = b;
    this.topk = b.final_topk ?? [];
    const sum = this.topk.reduce((a, [, p]) => a + p, 0);
    this.tail = Math.max(0, 1 - sum);
    this.p1 = this.topk[0]?.[1] ?? 0;
    this.p2 = this.topk[1]?.[1] ?? 0;
    this.prest = Math.max(0, 1 - this.p1 - this.p2);
    this.hoverBar = -1;
    this.hoverPoint = false;

    // bars: top-12 individually, then the ranks-13+ tail as a final row
    this.bars = this.topk.map(([tok, p], i) => ({
      rank: i + 1,
      token: tok,
      prob: p,
      rgb: i === 0 ? GOLD : i === 1 ? STEEL : OTHER,
    }));
    this.bars.push({ rank: 0, token: "· all other tokens (rank ≥13)", prob: this.tail, rgb: TAIL });

    this.pushLayers();
    this.positionLabels();
    this.buildBars();
  }

  /** barycentric (p1 on A, p2 on B, prest on C) → world xy */
  private bary(p1: number, p2: number, prest: number): [number, number] {
    return [p2 + 0.5 * prest, H * prest];
  }

  private pushLayers(): void {
    if (!this.deck) return;
    const { PathLayer, PolygonLayer, ScatterplotLayer } = this.layersMod;
    const A: [number, number] = [0, 0];
    const B: [number, number] = [1, 0];
    const C: [number, number] = [0.5, H];

    // ternary gridlines: iso-probability lines at 0.2..0.8 for each of the three
    // coordinates (constant p1 ∥ BC, constant p2 ∥ AC, constant p_rest ∥ AB).
    // Rendered as subtle DASHED hairlines now (req 5); dash authored in px and
    // scaled to world units so it holds a constant on-screen size.
    const wpp = 1 / this.zoomPx();
    const grid: { path: [number, number][] }[] = [];
    const gridSeg = (a: Vec2, b: Vec2) => {
      for (const s of dashedSegment(a, b, 3 * wpp, 6 * wpp)) grid.push({ path: [s.source, s.target] });
    };
    for (let g = 1; g <= 4; g++) {
      const c = g * 0.2;
      // constant p_rest = c (horizontal)
      gridSeg(this.bary(1 - c, 0, c), this.bary(0, 1 - c, c));
      // constant p1 = c
      gridSeg(this.bary(c, 1 - c, 0), this.bary(c, 0, 1 - c));
      // constant p2 = c
      gridSeg(this.bary(1 - c, c, 0), this.bary(0, c, 1 - c));
    }

    const pt = this.bary(this.p1, this.p2, this.prest);
    // hover reticle: a MARKER_HOT ring framing the distribution point (req 4).
    // A crosshair is skipped — on a barycentric triangle rectilinear guides read
    // as false axes, so the LED reticle alone marks the locked datum.
    const ring = this.hoverPoint ? [{ path: markerRing(pt[0], pt[1], 8 * wpp) }] : [];

    this.deck.setProps({
      layers: [
        new PolygonLayer<{ polygon: [number, number][] }>({
          id: "simplex-fill",
          data: [{ polygon: [A, B, C] }],
          getPolygon: (d) => d.polygon,
          getFillColor: [140, 150, 180, 12],
          stroked: false,
          filled: true,
          pickable: false,
        }),
        new PathLayer<{ path: [number, number][] }>({
          id: "simplex-grid",
          data: grid,
          getPath: (d) => d.path,
          getColor: GRID_RGBA,
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        new PathLayer<{ path: [number, number][] }>({
          id: "simplex-edge",
          data: [{ path: [A, B, C, A] }],
          getPath: (d) => d.path,
          getColor: [180, 186, 205, 150],
          getWidth: 1.5,
          widthUnits: "pixels",
          pickable: false,
        }),
        // guide lines from the point to each edge would over-clutter; instead a
        // faint drop to the AB baseline shows the p_rest height honestly.
        new PathLayer<{ path: [number, number][] }>({
          id: "simplex-drop",
          data: [{ path: [pt, [pt[0], 0]] }],
          getPath: (d) => d.path,
          getColor: [245, 195, 59, 90],
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        new ScatterplotLayer<{ position: [number, number] }>({
          id: "simplex-point-halo",
          data: [{ position: pt }],
          getPosition: (d) => [d.position[0], d.position[1], 0],
          getFillColor: [245, 205, 120, this.hoverPoint ? 90 : 60],
          getRadius: this.hoverPoint ? 15 : 12,
          radiusUnits: "pixels",
          pickable: false,
          updateTriggers: { getRadius: this.hoverPoint, getFillColor: this.hoverPoint },
        }),
        new ScatterplotLayer<{ position: [number, number] }>({
          id: "simplex-point",
          data: [{ position: pt }],
          getPosition: (d) => [d.position[0], d.position[1], 0],
          getFillColor: [255, 236, 194, 255],
          getLineColor: [30, 24, 12, 220],
          stroked: true,
          lineWidthUnits: "pixels",
          getLineWidth: 1.5,
          getRadius: 5.5,
          radiusUnits: "pixels",
          pickable: true,
        }),
        new PathLayer<{ path: [number, number][] }>({
          id: "simplex-reticle",
          data: ring,
          getPath: (d) => d.path,
          getColor: withAlpha(MARKER_HOT, 0.9),
          getWidth: 1.2,
          widthUnits: "pixels",
          pickable: false,
        }),
      ],
    });
  }

  // ---- layout: fit the unit triangle into the LEFT panel --------------------
  private panelRight(): number {
    return this.cssW * SPLIT;
  }
  private availW(): number {
    return Math.max(1, this.panelRight() - GL - GMID);
  }
  private availH(): number {
    return Math.max(1, this.cssH - GT - GB);
  }
  private zoomPx(): number {
    return Math.max(4, Math.min(this.availW() / 1.0, this.availH() / H));
  }
  private cX(): number {
    return GL + this.availW() / 2;
  }
  private cY(): number {
    return GT + this.availH() / 2;
  }
  private worldToScreen(wx: number, wy: number): [number, number] {
    const z = this.zoomPx();
    return [this.cX() + (wx - 0.5) * z, this.cY() - (wy - H / 2) * z];
  }
  private viewState() {
    const z = this.zoomPx();
    return {
      ortho: {
        target: [
          0.5 + (this.cssW / 2 - this.cX()) / z,
          H / 2 + (this.cY() - this.cssH / 2) / z,
          0,
        ] as [number, number, number],
        zoom: Math.log2(z),
      },
    };
  }

  private positionLabels(): void {
    this.labelRoot.textContent = "";
    const mk = (cls: string, html: string, wx: number, wy: number, dx: number, dy: number) => {
      const el = document.createElement("div");
      el.className = cls;
      el.innerHTML = html;
      const [sx, sy] = this.worldToScreen(wx, wy);
      el.style.transform = `translate(${(sx + dx).toFixed(1)}px, ${(sy + dy).toFixed(1)}px)`;
      this.labelRoot.appendChild(el);
    };
    const t1 = fmtTok(this.topk[0]?.[0] ?? "∅");
    const t2 = fmtTok(this.topk[1]?.[0] ?? "∅");
    const pct = (p: number) => `${(p * 100).toFixed(1)}%`;
    // A (bottom-left) = top-1, B (bottom-right) = top-2, C (top) = other
    mk(
      "interp-simplex-vtx is-top1",
      `<b>${escapeHtml(t1)}</b><span>${pct(this.p1)}</span>`,
      0,
      0,
      -6,
      6,
    );
    mk(
      "interp-simplex-vtx is-top2 is-right",
      `<b>${escapeHtml(t2)}</b><span>${pct(this.p2)}</span>`,
      1,
      0,
      6,
      6,
    );
    mk(
      "interp-simplex-vtx is-other is-top",
      `<b>all other</b><span>${pct(this.prest)} · rank ≥3</span>`,
      0.5,
      H,
      0,
      -34,
    );
  }

  private buildBars(): void {
    this.barRoot.textContent = "";
    const scale = Math.max(this.p1, this.tail, 1e-6);
    const head = document.createElement("div");
    head.className = "interp-simplex-bars-head";
    head.textContent = "next-token distribution · final position";
    this.barRoot.appendChild(head);
    this.bars.forEach((bar, i) => {
      const row = document.createElement("div");
      row.className = `interp-simplex-bar${i === this.hoverBar ? " is-hot" : ""}${bar.rank === 0 ? " is-tail" : ""}`;
      row.dataset.i = String(i);
      const tok = document.createElement("span");
      tok.className = "interp-simplex-bar-tok";
      tok.textContent = bar.rank === 0 ? bar.token : `${bar.rank}. ${fmtTok(bar.token)}`;
      const track = document.createElement("span");
      track.className = "interp-simplex-bar-track";
      const fill = document.createElement("span");
      fill.className = "interp-simplex-bar-fill";
      fill.style.width = `${((bar.prob / scale) * 100).toFixed(1)}%`;
      fill.style.background = `rgb(${bar.rgb[0]},${bar.rgb[1]},${bar.rgb[2]})`;
      track.appendChild(fill);
      const val = document.createElement("span");
      val.className = "interp-simplex-bar-val";
      val.textContent = `${(bar.prob * 100).toFixed(bar.prob >= 0.1 ? 1 : 2)}%`;
      row.append(tok, track, val);
      row.addEventListener("pointerenter", () => {
        this.hoverBar = i;
        this.refreshBarHot();
      });
      row.addEventListener("pointerleave", () => {
        if (this.hoverBar === i) {
          this.hoverBar = -1;
          this.refreshBarHot();
        }
      });
      this.barRoot.appendChild(row);
    });
    this.positionBars();
  }

  private refreshBarHot(): void {
    for (const row of Array.from(this.barRoot.querySelectorAll<HTMLElement>(".interp-simplex-bar"))) {
      row.classList.toggle("is-hot", Number(row.dataset.i) === this.hoverBar);
    }
  }

  private positionBars(): void {
    const left = this.panelRight() + GMID;
    this.barRoot.style.left = `${left.toFixed(0)}px`;
    this.barRoot.style.top = `${GT.toFixed(0)}px`;
    this.barRoot.style.width = `${Math.max(120, this.cssW - left - 22).toFixed(0)}px`;
    this.barRoot.style.maxHeight = `${(this.cssH - GT - 20).toFixed(0)}px`;
  }

  private onPointerMove(e: PointerEvent): void {
    if (this.tooltip.pinned) return;
    if (!this.deck) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    if (!this.showTooltipFor(x, y)) {
      this.tooltip.hide();
      this.canvas.style.cursor = "";
    }
  }

  /** Touch-only: a tap pins the readout so it survives past the finger lifting
   *  (touch has no hover, so pointerleave would otherwise hide it instantly).
   *  A tap on empty space clears the pin and the hover highlight rather than
   *  leaving a stale one stuck. Mouse pointers no-op — the hover path serves them. */
  private onClick(e: PointerEvent): void {
    if (e.pointerType !== "touch" || !this.deck) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    if (this.showTooltipFor(x, y)) {
      this.tooltip.pinned = true;
    } else {
      this.tooltip.pinned = false;
      this.tooltip.hide();
      this.canvas.style.cursor = "";
    }
  }

  /** Pick the distribution point at (x, y), update the hover halo, and show
   *  the tooltip there. Returns whether the point was hit. */
  private showTooltipFor(x: number, y: number): boolean {
    if (!this.deck) return false;
    const info = this.deck.pickObject({ x, y, radius: 8, layerIds: ["simplex-point"] }) as
      | PickingInfo
      | null;
    const over = !!info?.object;
    if (over !== this.hoverPoint) {
      this.hoverPoint = over;
      this.pushLayers();
    }
    if (!over) return false;
    const t1 = fmtTok(this.topk[0]?.[0] ?? "∅");
    const t2 = fmtTok(this.topk[1]?.[0] ?? "∅");
    // each row's swatch is the exact corner color the datum is drawn against;
    // the top-1 probability is the hot reading (the confidence the point encodes)
    const rows: TipRow[] = [
      { kind: "label", text: "next-token distribution (exact)" },
      { text: t1, value: `${(this.p1 * 100).toFixed(2)}%`, hot: true, swatch: GOLD },
      { text: t2, value: `${(this.p2 * 100).toFixed(2)}%`, swatch: STEEL },
      { text: "all other", value: `${(this.prest * 100).toFixed(2)}% · Σ = 1`, swatch: OTHER },
    ];
    this.tooltip.show(rows);
    this.tooltip.move(x, y, this.cssW, this.cssH);
    this.canvas.style.cursor = "crosshair";
    return true;
  }

  private onLeave(): void {
    if (this.tooltip.pinned) return;
    if (this.hoverPoint) {
      this.hoverPoint = false;
      this.pushLayers();
    }
    this.tooltip.hide();
    this.canvas.style.cursor = "";
  }

  /** Footer strip. p1/p2/tail are the exact masses the simplex vertices are
   *  placed from, so the strip cannot disagree with the plot. The tail is
   *  labelled by its rank cut rather than as "other", because "other" reads as
   *  a residual the view measured when it is really everything past the top-k
   *  the bundle shipped. */
  stats(): StatTile[] {
    if (!this.bundle) return [];
    const pct = (p: number) => `${(p * 100).toFixed(1)}%`;
    const top1 = this.topk[0]?.[0];
    return [
      {
        label: "top-1",
        value: pct(this.p1),
        title: top1 != null ? `most likely next token: ${JSON.stringify(top1)}` : undefined,
      },
      { label: "top-2", value: pct(this.p2) },
      {
        label: `tail ≥${this.topk.length + 1}`,
        value: pct(this.tail),
        title: `mass below the top-${this.topk.length} the bundle ships`,
      },
      { label: "candidates", value: String(this.topk.length) },
      {
        label: "entropy ≥",
        value: `${this.entropyBits().toFixed(2)} bits`,
        title:
          "Shannon entropy over the shipped top-k with the tail as ONE bucket — " +
          "a lower bound on true next-token entropy, since the real tail is " +
          "spread over many tokens this bundle does not ship",
      },
    ];
  }

  /** H over the top-k plus the tail collapsed into a single bucket. Stated that
   *  way on the tile's hover: it is a LOWER bound on the true next-token
   *  entropy, since the real tail is spread over many tokens the bundle does
   *  not ship. Reporting it as "the" entropy would overclaim. */
  private entropyBits(): number {
    let h = 0;
    for (const [, p] of this.topk) if (p > 0) h -= p * Math.log2(p);
    if (this.tail > 0) h -= this.tail * Math.log2(this.tail);
    return h;
  }

  frame(_dt: number, _t: number): void {
    // static per prompt — no data-bearing motion
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
    this.positionBars();
  }

  dispose(): void {
    for (const d of this.disposers) d();
    this.disposers = [];
    this.tooltip?.dispose();
    this.labelRoot?.remove();
    this.barRoot?.remove();
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
