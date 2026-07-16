/** #14 Causal Patching Map — where does the answer causally live?
 *
 *  Activation patching (causal tracing): run a matched clean/corrupt prompt
 *  pair, then for every (layer, position) copy ONE clean residual row into
 *  the corrupt forward and let it finish. Each cell is a real intervention —
 *  a full patched forward resume — measured as normalized recovery of the
 *  clean-vs-corrupt answer logit difference. Unlike attribution (#13), this
 *  IS a causal quantity: r = 1 means that single residual row is sufficient
 *  to flip the corrupt run to the clean answer. The IOI pair reproduces the
 *  published shape by intervention: the swapped-name position carries
 *  everything through L8, then the signal hands off to the final position at
 *  L9 — the same layer #13's name-mover heads write the margin.
 *
 *  deck.gl (WebGL2), camera off, static. Source: patch.json. */

import type { Deck, OrthographicView, PickingInfo } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { type PatchBundle, type PatchPair, loadPatch } from "../../data/interp";
import {
  ACCENT,
  crosshair,
  dashedSegment,
  HOT,
  MARKER_HOT,
  type RGB,
  type Seg as ThemeSeg,
  withAlpha,
} from "./chart-theme";
import { InterpTooltip, type TipRow } from "./chart-tooltip";
import type { InterpDriver } from "./InterpDriver";

type LayersModule = typeof import("@deck.gl/layers");

const GL = 56; // px — patch-site row labels
const GR = 36;
const GT = 96; // px — header + column (token) labels
const GB = 92; // px — chip strip + collapsed legend pill

// same diverging convention as #13: ~0 is a distinct near-invisible grey so
// "the patch did nothing" never reads as "did a little"
const LOW: [number, number, number] = [64, 66, 96];
const POS: [number, number, number] = [245, 195, 59]; // recovers the clean answer
const NEG: [number, number, number] = [96, 150, 255]; // pushes further corrupt
const ZERO_RGBA: [number, number, number, number] = [118, 126, 158, 14];

interface Cell {
  poly: [number, number][];
  r: number; // exported normalized recovery
  ld: number; // exported raw patched logit-diff
  row: number; // 0 = emb … n_layer = final residual
  pos: number;
}

export class PatchingMapDriver implements InterpDriver {
  readonly animated = false;
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private canvas!: HTMLCanvasElement;
  private tooltip!: InterpTooltip;
  private labelRoot!: HTMLElement;
  private chipRoot!: HTMLElement;

  private bundle: PatchBundle | null = null;
  private pair: PatchPair | null = null;
  private nL = 12;
  private cells: Cell[] = [];
  private hover: Cell | null = null;

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
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    this.disposers.push(() => {
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
    });
  }

  async setModel(model: string): Promise<void> {
    if (!this.bundle || this.bundle.meta.model !== model) {
      this.bundle = await loadPatch(model);
    }
    this.nL = this.bundle.meta.n_layer;
    if (!this.pair || !this.bundle.pairs.includes(this.pair)) {
      this.pair = this.bundle.pairs[0] ?? null;
    }
    if (!this.pair) throw new Error("patch.json has no pairs");
    this.hover = null;
    this.layout();
    this.buildChips();
    this.deck?.setProps({ viewState: this.viewState() });
    this.pushLayers();
    this.positionLabels();
  }

  // ---- pixel-space layout ---------------------------------------------------
  private gb(): number {
    return this.cssW < 640 ? 150 : GB; // narrow: chip strip lifts above the pill
  }
  private nRows(): number {
    return this.nL + 1; // emb + 11 intermediate + final residual
  }
  private cellW(): number {
    const t = this.pair?.T ?? 1;
    return Math.min(110, (this.cssW - GL - GR) / t);
  }
  private rowH(): number {
    return Math.min(44, Math.max(10, (this.cssH - GT - this.gb()) / this.nRows()));
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
    const pr = this.pair;
    if (!pr) return;
    const cw = this.cellW();
    const rh = this.rowH();
    const cells: Cell[] = [];
    for (let i = 0; i < this.nRows(); i++) {
      const y0 = GT + i * rh;
      for (let p = 0; p < pr.T; p++) {
        const x0 = GL + p * cw;
        const k = i * pr.T + p;
        cells.push({
          poly: [
            [x0 + 0.5, y0 + 0.5],
            [x0 + cw - 0.5, y0 + 0.5],
            [x0 + cw - 0.5, y0 + rh - 1],
            [x0 + 0.5, y0 + rh - 1],
          ],
          r: pr.r[k] ?? 0,
          ld: pr.ld[k] ?? 0,
          row: i,
          pos: p,
        });
      }
    }
    this.cells = cells;
  }

  private colorOf(r: number): [number, number, number, number] {
    if (Math.abs(r) < 5e-3) return ZERO_RGBA; // < 0.5% recovery: nothing moved
    const t = Math.min(1, Math.abs(r)); // color clamps at |r| = 1 — stated
    const end = r > 0 ? POS : NEG;
    const cr = Math.round(LOW[0] + t * (end[0] - LOW[0]));
    const cg = Math.round(LOW[1] + t * (end[1] - LOW[1]));
    const cb = Math.round(LOW[2] + t * (end[2] - LOW[2]));
    return [cr, cg, cb, Math.round(70 + 185 * t)];
  }

  private pushLayers(): void {
    if (!this.deck || !this.cells.length) return;
    const { SolidPolygonLayer, LineLayer } = this.layersMod;
    const hover = this.hover;
    const rh = this.rowH();
    const cw = this.cellW();
    // hovered cell: keep the crisp outline but recolor to the danger LED, and
    // snap an ACCENT row/column crosshair onto the cell centre (req 4)
    const edges = hover
      ? hover.poly.map((p, i) => ({
          source: p,
          target: hover.poly[(i + 1) % hover.poly.length] as [number, number],
        }))
      : [];
    // vertical guide on the diff position(s): where the two prompts differ —
    // DASHED amber hairlines now (req 5), the semantic color kept
    const pr = this.pair;
    const guides: ThemeSeg[] = (pr?.diff_pos ?? []).flatMap((p) => [
      ...dashedSegment([GL + p * cw, GT], [GL + p * cw, GT + this.nRows() * rh]),
      ...dashedSegment([GL + (p + 1) * cw, GT], [GL + (p + 1) * cw, GT + this.nRows() * rh]),
    ]);
    const plotBounds = { x0: GL, y0: GT, x1: GL + (pr?.T ?? 0) * cw, y1: GT + this.nRows() * rh };
    const cross: ThemeSeg[] = hover
      ? crosshair(GL + hover.pos * cw + cw / 2, GT + hover.row * rh + rh / 2, plotBounds)
      : [];

    this.deck.setProps({
      layers: [
        new SolidPolygonLayer<Cell>({
          id: "patch-cells",
          data: this.cells,
          getPolygon: (c) => c.poly,
          getFillColor: (c) => this.colorOf(c.r),
          pickable: true,
        }),
        new LineLayer<ThemeSeg>({
          id: "patch-diff-guide",
          data: guides,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: withAlpha(HOT, 90 / 255),
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        new LineLayer<ThemeSeg>({
          id: "patch-crosshair",
          data: cross,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: withAlpha(ACCENT, 0.5),
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        new LineLayer<{ source: [number, number]; target: [number, number] }>({
          id: "patch-hover",
          data: edges,
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

  // ---- chips: pair selector ---------------------------------------------------
  private buildChips(): void {
    const b = this.bundle;
    if (!b) return;
    this.chipRoot.textContent = "";
    this.chipRoot.style.bottom = this.cssW < 640 ? "110px" : "";
    const narrow = this.cssW < 640;
    for (const pr of b.pairs) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "interp-neuron-chip";
      btn.textContent = narrow
        ? `${vis(pr.ans_clean[0])}/${vis(pr.ans_corrupt[0])}`
        : `${vis(pr.ans_clean[0])} vs ${vis(pr.ans_corrupt[0])}`;
      btn.title = `${pr.clean} → ${pr.ans_clean[0]} / ${pr.corrupt} → ${pr.ans_corrupt[0]}`;
      btn.setAttribute("aria-pressed", String(pr === this.pair));
      if (pr === this.pair) btn.classList.add("is-active");
      btn.addEventListener("click", () => {
        if (pr === this.pair) return;
        this.pair = pr;
        this.hover = null;
        this.tooltip.hide();
        this.layout();
        this.buildChips();
        this.pushLayers();
        this.positionLabels();
      });
      this.chipRoot.appendChild(btn);
    }
  }

  // ---- labels -----------------------------------------------------------------
  private rowName(i: number): string {
    return i === 0 ? "emb" : i === this.nL ? "fin" : `x${i}`;
  }
  private rowMeaning(i: number): string {
    return i === 0
      ? "token+position embeddings (before block 0)"
      : i === this.nL
        ? "final residual (after block " + (this.nL - 1) + ", before ln_f)"
        : `residual entering block ${i} (after block ${i - 1})`;
  }

  private positionLabels(): void {
    this.labelRoot.textContent = "";
    const pr = this.pair;
    const b = this.bundle;
    if (!pr || !b) return;
    const cw = this.cellW();
    const rh = this.rowH();
    const narrow = this.cssW < 640;

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

    // header: the designated contrast and both baselines
    const h1 = cap(
      narrow
        ? `patch · “${vis(pr.ans_clean[0])}” vs “${vis(pr.ans_corrupt[0])}” · ` +
            `LD ${sgn(pr.ld_clean, 2)}/${sgn(pr.ld_corrupt, 2)}`
        : `patch one clean residual row into the corrupt run · answers “${vis(pr.ans_clean[0])}” vs ` +
            `“${vis(pr.ans_corrupt[0])}” · LD clean ${sgn(pr.ld_clean, 4)} · LD corrupt ${sgn(pr.ld_corrupt, 4)}`,
    );
    h1.style.color = "rgb(245,195,59)";
    place(h1, narrow ? 12 : GL, GT - 44);
    place(
      cap(
        narrow
          ? `r=(LD−LDc)/(LDcl−LDc) · |r|≤1 · ranks ${pr.ans_clean[3]}/${pr.ans_corrupt[3]} · ` +
              `n=${this.nRows() * pr.T}`
          : `r = (LD_patched − LD_corrupt)/(LD_clean − LD_corrupt) · color clamps |r| ≤ 1, exact in hover · ` +
              `“${vis(pr.ans_clean[0])}” rank ${pr.ans_clean[3]} in clean run, “${vis(pr.ans_corrupt[0])}” ` +
              `rank ${pr.ans_corrupt[3]} in corrupt run · ${this.nRows() * pr.T} real patched forwards`,
      ),
      narrow ? 12 : GL,
      GT - 30,
    );

    // column labels: the clean tokens; diff positions show corrupt→clean in amber
    for (let p = 0; p < pr.T; p++) {
      const isDiff = pr.diff_pos.includes(p);
      const el = cap(
        isDiff ? `${vis(pr.corrupt_strs[p] ?? "")}→${vis(pr.clean_strs[p] ?? "")}` : vis(pr.clean_strs[p] ?? ""),
      );
      if (isDiff) {
        el.style.color = "rgb(245,195,59)";
        el.title = "the corrupted position: the patch restores the clean token's residual here";
      }
      place(el, GL + p * cw + 3, GT - 16);
      // the diff label is the key to the whole view — let it borrow half the
      // neighbor column before truncating
      const maxW = isDiff ? cw * 1.5 : cw - 6;
      el.style.maxWidth = `${maxW.toFixed(0)}px`;
      el.style.overflow = "hidden";
      el.style.textOverflow = "ellipsis";
    }

    // row labels
    for (let i = 0; i < this.nRows(); i++) {
      const el = cap(this.rowName(i));
      el.title = this.rowMeaning(i);
      place(el, 8, GT + i * rh + rh / 2 - 8);
    }

    // in-cell recovery values where they matter and fit
    if (rh >= 15 && cw >= 34) {
      for (const c of this.cells) {
        if (Math.abs(c.r) < 0.3) continue;
        const el = cap(sgn(c.r, 2));
        el.style.color = "rgba(255,255,255,0.92)";
        place(el, GL + c.pos * cw + Math.max(1, (cw - el.offsetWidth) / 2), GT + c.row * rh + rh / 2 - 8);
      }
    }
  }

  // ---- interaction --------------------------------------------------------------
  private onPointerMove(e: PointerEvent): void {
    if (!this.deck) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const info = this.deck.pickObject({ x, y, radius: 1, layerIds: ["patch-cells"] }) as
      | PickingInfo
      | null;
    const c = (info?.object as Cell | undefined) ?? null;
    if (c !== this.hover) {
      this.hover = c;
      this.pushLayers();
    }
    const pr = this.pair;
    if (!c || !pr) {
      this.tooltip.hide();
      this.canvas.style.cursor = "";
      return;
    }
    const cc = this.colorOf(c.r);
    const swatch: RGB = [cc[0], cc[1], cc[2]];
    const rows: TipRow[] = [
      {
        kind: "label",
        text:
          `${this.rowName(c.row)} patch · pos ${c.pos} “${vis(pr.clean_strs[c.pos] ?? "")}”` +
          (pr.diff_pos.includes(c.pos) ? ` (was “${vis(pr.corrupt_strs[c.pos] ?? "")}”)` : ""),
        swatch,
      },
      { text: "r", value: sgn(c.r, 4), hot: true },
      { text: `${sgn(c.r * 100, 1)}% of the LD gap recovered` },
      {
        text: `LD patched ${sgn(c.ld, 4)} (clean ${sgn(pr.ld_clean, 4)} · corrupt ${sgn(pr.ld_corrupt, 4)})`,
      },
      { text: this.rowMeaning(c.row) },
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
    // static — the grid only changes on pair switches
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
    if (this.bundle) this.buildChips(); // chip offset depends on width
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

/** Signed fixed-point: +1.00, −0.42 — recovery values are directions. */
function sgn(v: number, dp: number): string {
  return `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(dp)}`;
}

/** Visible-escape a token for labels: leading space → ␣, newline → ⏎. */
function vis(s: string): string {
  const t = s.replace(/\n/g, "⏎").replace(/^ /, "␣");
  return t || "·";
}
