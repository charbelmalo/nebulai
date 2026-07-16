/** #23 Attention-Rollout Waterfall — attention rollout (Abnar & Zuidema, 2020):
 *  account for how information mixes across depth by taking the cumulative matrix
 *  product of the (residual-augmented, head-averaged) attention maps. Real
 *  quantity, computed in-browser from trace_*.json → attn[layer][head][i][j]:
 *
 *    A_l   = mean_h attn[l][h]                 (head-averaged, still row-stochastic)
 *    Ã_l   = 0.5·A_l + 0.5·I, row-normalized   (0.5 weight to the residual stream)
 *    R_d   = Ã_d · Ã_{d-1} · … · Ã_0           (cumulative through layer d)
 *
 *  R_d[i][j] is how much source token j contributes to destination token i's
 *  representation after layers 0..d — a proper distribution (each row sums to 1)
 *  and strictly causal (j ≤ i). Scrub / play the depth to watch the "waterfall":
 *  local structure at shallow depth cascading onto the first-token attention SINK
 *  by full depth (a known, honest property of rollout — not a bug). Color is
 *  log₁₀ (the values span orders of magnitude); hover reads the exact value.
 *  deck.gl (WebGL2), camera off. */

import type { Deck, OrthographicView } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { loadTrace, type TraceBundle } from "../../data/interp";
import { ACCENT, crosshair, MARKER_HOT, withAlpha } from "./chart-theme";
import { InterpTooltip } from "./chart-tooltip";
import type { InterpDriver } from "./InterpDriver";

type LayersModule = typeof import("@deck.gl/layers");

const GL = 96; // px left gutter — destination token labels
const GT = 66; // px top gutter — source token labels
const GR = 34; // px right gutter
const GB = 104; // px bottom gutter — depth control + axis caption
const LOG_FLOOR = 1e-4; // color floor: values below this map to the darkest step
const STEP_MS = 640; // auto-play cadence (one real layer per step — no interpolation)

// perceptual dark → gold ramp, evaluated on the log-normalized weight. The
// lowest step is deliberately lifted off the page background so faint (but real)
// cells stay visible as tiles rather than blending into the void.
const RAMP: Array<[number, [number, number, number]]> = [
  [0.0, [30, 38, 66]],
  [0.35, [50, 66, 120]],
  [0.6, [70, 150, 214]],
  [0.82, [232, 160, 60]],
  [1.0, [255, 236, 194]],
];

interface Cell {
  i: number; // destination (row) — token that receives information
  j: number; // source (col) — token information flows from
}

export class AttentionRolloutDriver implements InterpDriver {
  readonly animated = false; // static per depth; the play timer re-pushes on step
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private makeView!: () => OrthographicView;
  private canvas!: HTMLCanvasElement;
  private overlay!: HTMLElement;
  private tooltip!: InterpTooltip;
  private labelRoot!: HTMLElement;
  private ctrlRoot!: HTMLElement;

  private bundle: TraceBundle | null = null;
  private T = 0;
  private nLayer = 0;
  private rollouts: Float64Array[][] = []; // rollouts[d] = R_d (T rows of length T)
  private cells: Cell[] = [];
  private depth = 0; // current cumulative depth d (0..nLayer-1)
  private selRow = -1; // highlighted destination row (default: last token)
  private hover: Cell | null = null;
  private playing = false;
  private timer = 0;

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
    this.labelRoot.className = "interp-roll-labels";
    overlay.appendChild(this.labelRoot);
    this.ctrlRoot = document.createElement("div");
    this.ctrlRoot.className = "interp-roll-ctrl";
    overlay.appendChild(this.ctrlRoot);

    const onMove = (e: PointerEvent) => this.onPointerMove(e);
    const onLeave = () => this.hideTip();
    const onClick = (e: PointerEvent) => this.onClick(e);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    canvas.addEventListener("pointerdown", onClick);
    this.disposers.push(() => {
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("pointerdown", onClick);
    });
  }

  async setModel(model: string, trace?: string): Promise<void> {
    if (!trace) throw new Error("no forward trace selected");
    const b = await loadTrace(model, trace);
    this.bundle = b;
    this.T = b.meta.T;
    this.nLayer = b.meta.n_layer;
    this.computeRollouts();
    this.selRow = -1; // no row isolated by default — show the whole matrix
    this.hover = null;
    this.stopPlay();
    this.depth = 0; // start at the top of the cascade and auto-play down the stack
    // draw the lower-triangular cells once (rollout is causal → j ≤ i)
    this.cells = [];
    for (let i = 0; i < this.T; i++) for (let j = 0; j <= i; j++) this.cells.push({ i, j });
    this.buildControls();
    this.pushLayers();
    this.positionLabels();
    this.startPlay(); // the waterfall: R_0 → R_{L-1}, resting at the full rollout
  }

  /** Precompute R_0 … R_{L-1}. Each is head-averaged, residual-augmented, and
   *  cumulatively multiplied — so scrubbing depth is instant and exact. */
  private computeRollouts(): void {
    const b = this.bundle!;
    const T = this.T;
    const H = b.meta.n_head;
    this.rollouts = [];
    // running product R (starts at identity)
    let R = Array.from({ length: T }, (_, i) => {
      const r = new Float64Array(T);
      r[i] = 1;
      return r;
    });
    for (let l = 0; l < this.nLayer; l++) {
      // Ã_l = 0.5·mean_h(attn) + 0.5·I, row-normalized
      const Atil = Array.from({ length: T }, () => new Float64Array(T));
      for (let i = 0; i < T; i++) {
        let s = 0;
        for (let j = 0; j < T; j++) {
          let a = 0;
          for (let h = 0; h < H; h++) a += b.attn[l]![h]![i]![j]!;
          a /= H;
          const v = 0.5 * a + (i === j ? 0.5 : 0);
          Atil[i]![j] = v;
          s += v;
        }
        if (s > 0) for (let j = 0; j < T; j++) Atil[i]![j]! /= s;
      }
      // R ← Ã_l · R
      const next = Array.from({ length: T }, () => new Float64Array(T));
      for (let i = 0; i < T; i++) {
        for (let k = 0; k < T; k++) {
          const a = Atil[i]![k]!;
          if (a === 0) continue;
          const Rk = R[k]!;
          const ni = next[i]!;
          for (let j = 0; j < T; j++) ni[j]! += a * Rk[j]!;
        }
      }
      R = next;
      this.rollouts.push(R.map((row) => Float64Array.from(row)));
    }
  }

  private valOf(i: number, j: number): number {
    return this.rollouts[this.depth]?.[i]?.[j] ?? 0;
  }

  private colorOf = (d: Cell): [number, number, number, number] => {
    const v = this.valOf(d.i, d.j);
    // log-normalize: values span orders of magnitude (a real sink forms at j=0)
    const t = v <= LOG_FLOOR ? 0 : Math.min(1, (Math.log10(v) - Math.log10(LOG_FLOOR)) / -Math.log10(LOG_FLOOR));
    const [r, g, b] = ramp(t);
    let a = 255;
    if (this.selRow >= 0 && d.i !== this.selRow) a = 70; // isolate the selected row
    if (this.hover && this.hover.i === d.i && this.hover.j === d.j) a = 255;
    return [r, g, b, a];
  };

  private pushLayers(): void {
    if (!this.deck) return;
    const { PolygonLayer, PathLayer } = this.layersMod;
    const T = this.T;
    // hovered cell: a crisp MARKER_HOT reticle around the tile plus an ACCENT
    // crosshair through its centre (req 4). Dashes are authored in pixels and
    // scaled into this driver's world space (1 world unit = 1 cell).
    const hover = this.hover;
    const wpp = 1 / this.zoomPx();
    const hoverPath: { path: [number, number][] }[] = [];
    const cross: { path: [number, number][] }[] = [];
    if (hover) {
      const x = hover.j;
      const y = T - hover.i - 1;
      hoverPath.push({
        path: [
          [x, y],
          [x + 1, y],
          [x + 1, y + 1],
          [x, y + 1],
          [x, y],
        ],
      });
      for (const s of crosshair(x + 0.5, y + 0.5, { x0: 0, y0: 0, x1: T, y1: T }, 4 * wpp, 5 * wpp)) {
        cross.push({ path: [s.source, s.target] });
      }
    }
    this.deck.setProps({
      layers: [
        new PolygonLayer<Cell>({
          id: "roll-cells",
          data: this.cells,
          getPolygon: (d) => {
            const x = d.j;
            const y = T - d.i - 1; // row 0 (dest) at the TOP
            return [
              [x, y],
              [x + 1, y],
              [x + 1, y + 1],
              [x, y + 1],
            ];
          },
          getFillColor: this.colorOf,
          stroked: true,
          filled: true,
          getLineColor: [8, 10, 18, 160],
          lineWidthUnits: "pixels",
          getLineWidth: 1,
          lineWidthMinPixels: 0.5,
          pickable: true,
          updateTriggers: {
            getFillColor: `${this.depth}|${this.selRow}|${this.hover ? `${this.hover.i},${this.hover.j}` : "x"}`,
          },
        }),
        new PathLayer<{ path: [number, number][] }>({
          id: "roll-crosshair",
          data: cross,
          getPath: (d) => d.path,
          getColor: withAlpha(ACCENT, 0.5),
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        new PathLayer<{ path: [number, number][] }>({
          id: "roll-hover",
          data: hoverPath,
          getPath: (d) => d.path,
          getColor: withAlpha(MARKER_HOT, 0.86),
          getWidth: 1.4,
          widthUnits: "pixels",
          pickable: false,
        }),
      ],
    });
  }

  // ---- layout ---------------------------------------------------------------
  private zoomPx(): number {
    const availW = this.cssW - GL - GR;
    const availH = this.cssH - GT - GB;
    return Math.max(6, Math.min(availW / this.T, availH / this.T));
  }
  private targetX(): number {
    const centerScreenX = GL + (this.cssW - GL - GR) / 2;
    return this.T / 2 - (centerScreenX - this.cssW / 2) / this.zoomPx();
  }
  private targetY(): number {
    const centerScreenY = GT + (this.cssH - GT - GB) / 2;
    return this.T / 2 - (this.cssH / 2 - centerScreenY) / this.zoomPx();
  }
  private worldToScreen(wx: number, wy: number): [number, number] {
    const z = this.zoomPx();
    return [
      this.cssW / 2 + (wx - this.targetX()) * z,
      this.cssH / 2 - (wy - this.targetY()) * z,
    ];
  }
  private viewState() {
    return {
      ortho: {
        target: [this.targetX(), this.targetY(), 0] as [number, number, number],
        zoom: Math.log2(this.zoomPx()),
      },
    };
  }

  private positionLabels(): void {
    const b = this.bundle;
    if (!b) return;
    this.labelRoot.textContent = "";
    const T = this.T;
    for (let i = 0; i < T; i++) {
      // destination labels (left), one per row
      const dl = document.createElement("div");
      dl.className = `interp-roll-dst${i === this.selRow ? " is-sel" : ""}`;
      dl.textContent = fmtTok(b.token_strs[i] ?? "");
      const [lx, ly] = this.worldToScreen(0, T - i - 0.5);
      dl.style.transform = `translate(${(lx - 8).toFixed(1)}px, ${ly.toFixed(1)}px)`;
      this.labelRoot.appendChild(dl);
      // source labels (top), one per column, rotated
      const sl = document.createElement("div");
      sl.className = "interp-roll-src";
      sl.textContent = fmtTok(b.token_strs[i] ?? "");
      const [sx, sy] = this.worldToScreen(i + 0.5, T);
      sl.style.transform = `translate(${sx.toFixed(1)}px, ${(sy - 8).toFixed(1)}px) rotate(-52deg)`;
      this.labelRoot.appendChild(sl);
    }
    // Left caption only: "row = destination, col = source" is fully disambiguated
    // by the two token axes, the hover ("dst ← src"), and the blurb. A top caption
    // would collide with the prompt tracebar, so we dock this one in the far-left
    // gutter, clear of the right-aligned destination token labels.
    const dcap = document.createElement("div");
    dcap.className = "interp-roll-axis is-dst";
    dcap.textContent = "destination ← source";
    const [, dcy] = this.worldToScreen(0, T / 2);
    dcap.style.transform = `translate(20px, ${dcy.toFixed(1)}px) rotate(-90deg)`;
    this.labelRoot.appendChild(dcap);
  }

  private buildControls(): void {
    this.ctrlRoot.textContent = "";
    const play = document.createElement("button");
    play.type = "button";
    play.className = "interp-roll-play";
    play.textContent = this.playing ? "❚❚ pause" : "▶ cascade";
    play.addEventListener("click", () => this.togglePlay());
    const slider = document.createElement("input");
    slider.type = "range";
    slider.className = "interp-roll-slider";
    slider.min = "0";
    slider.max = String(Math.max(0, this.nLayer - 1));
    slider.value = String(this.depth);
    slider.addEventListener("input", () => {
      this.stopPlay();
      this.setDepth(Number(slider.value));
    });
    const read = document.createElement("span");
    read.className = "interp-roll-read";
    read.textContent = `cumulative through layer ${this.depth} / ${this.nLayer - 1}`;
    this.ctrlRoot.append(play, slider, read);
    this.ctrlEls = { play, slider, read };
  }
  private ctrlEls: { play: HTMLButtonElement; slider: HTMLInputElement; read: HTMLElement } | null = null;

  private setDepth(d: number): void {
    this.depth = Math.max(0, Math.min(this.nLayer - 1, d | 0));
    if (this.ctrlEls) {
      this.ctrlEls.slider.value = String(this.depth);
      this.ctrlEls.read.textContent = `cumulative through layer ${this.depth} / ${this.nLayer - 1}`;
    }
    this.pushLayers();
  }

  private togglePlay(): void {
    if (this.playing) this.stopPlay();
    else this.startPlay();
  }
  private startPlay(): void {
    if (this.playing || this.nLayer < 2) return;
    this.playing = true;
    if (this.ctrlEls) this.ctrlEls.play.textContent = "❚❚ pause";
    if (this.depth >= this.nLayer - 1) this.setDepth(0); // restart from the top
    this.timer = window.setInterval(() => {
      if (this.depth >= this.nLayer - 1) {
        this.stopPlay(); // stop at full depth (the complete rollout)
        return;
      }
      this.setDepth(this.depth + 1);
    }, STEP_MS);
  }
  private stopPlay(): void {
    this.playing = false;
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = 0;
    }
    if (this.ctrlEls) this.ctrlEls.play.textContent = "▶ cascade";
  }

  private pick(e: PointerEvent): Cell | null {
    if (!this.deck) return null;
    const rect = this.canvas.getBoundingClientRect();
    const info = this.deck.pickObject({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      radius: 1,
      layerIds: ["roll-cells"],
    });
    return (info?.object as Cell | undefined) ?? null;
  }

  private onClick(e: PointerEvent): void {
    const cell = this.pick(e);
    if (!cell) return;
    this.selRow = this.selRow === cell.i ? -1 : cell.i; // toggle row isolation
    this.pushLayers();
    this.positionLabels();
  }

  private onPointerMove(e: PointerEvent): void {
    if (!this.bundle) return;
    const cell = this.pick(e);
    const same = cell && this.hover && cell.i === this.hover.i && cell.j === this.hover.j;
    if (!same) {
      this.hover = cell;
      this.pushLayers();
    }
    if (!cell) {
      this.hideTip();
      return;
    }
    const b = this.bundle;
    const v = this.valOf(cell.i, cell.j);
    const dst = fmtTok(b.token_strs[cell.i] ?? "");
    const src = fmtTok(b.token_strs[cell.j] ?? "");
    // swatch = the exact log-normalized ramp color this cell was drawn with
    const t =
      v <= LOG_FLOOR
        ? 0
        : Math.min(1, (Math.log10(v) - Math.log10(LOG_FLOOR)) / -Math.log10(LOG_FLOOR));
    const [sr, sg, sb] = ramp(t);
    this.tooltip.show([
      { kind: "label", text: `dst “${dst}” ← src “${src}”`, swatch: [sr, sg, sb] },
      { text: "rollout", value: v.toFixed(4), hot: true },
      { text: "through", value: `L${this.depth}` },
      { text: "pos", value: `${cell.j}→${cell.i}` },
    ]);
    const rect = this.canvas.getBoundingClientRect();
    this.tooltip.move(e.clientX - rect.left, e.clientY - rect.top, this.cssW, this.cssH);
    this.canvas.style.cursor = "crosshair";
  }

  private hideTip(): void {
    this.tooltip?.hide();
    this.canvas.style.cursor = "";
    if (this.hover) {
      this.hover = null;
      this.pushLayers();
    }
  }

  frame(_dt: number, _t: number): void {
    // depth is stepped by the play timer, not the RAF — nothing to do here
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
    this.stopPlay();
    for (const d of this.disposers) d();
    this.disposers = [];
    this.tooltip?.dispose();
    this.labelRoot?.remove();
    this.ctrlRoot?.remove();
    this.deck?.finalize();
    this.deck = null;
  }
}

function ramp(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t));
  for (let s = 1; s < RAMP.length; s++) {
    const [t1, c1] = RAMP[s]!;
    if (x <= t1) {
      const [t0, c0] = RAMP[s - 1]!;
      const f = t1 === t0 ? 0 : (x - t0) / (t1 - t0);
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * f),
        Math.round(c0[1] + (c1[1] - c0[1]) * f),
        Math.round(c0[2] + (c1[2] - c0[2]) * f),
      ];
    }
  }
  return RAMP[RAMP.length - 1]![1];
}

function fmtTok(s: string): string {
  return s.replace(/^ /, "␣").replace(/\n/g, "⏎") || "∅";
}
