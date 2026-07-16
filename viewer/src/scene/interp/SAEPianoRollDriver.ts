/** #4 SAE Firing Piano-Roll — the res-jb sparse autoencoder's encoder run on a
 *  REAL forward pass of one bundled prompt. Rows = the top features by peak
 *  activation (over positions ≥ 1), columns = token positions; cell brightness
 *  is the exact encoder activation acts = ReLU((x̄ − b_dec)·W_enc + b_enc) on
 *  the layer-8 residual stream, re-centered per position into the SAE's
 *  training basis (TransformerLens center_writing_weights — LayerNorm-invariant,
 *  exact). Each row is scaled to its own peak, PRINTED at the right edge (a
 *  board-wide scale is one chip away). GPT-2's first-token massive-activation
 *  outlier drives a handful of features 60–100× everything else; those ship as
 *  a separate labeled band rather than silently flooding the board. Below the
 *  grid, per-position L0 and reconstruction cosine are printed exactly — the
 *  honesty strip saying how much of the stream the SAE actually explains.
 *
 *  deck.gl (WebGL2), camera off, static. Source: sae_acts.json. */

import type { Deck, OrthographicView, PickingInfo } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { appStore, type InterpSelection } from "../../app/store";
import {
  type SAEActsBundle,
  type SAEActsFeature,
  type SAEActsTrace,
  isLiveTrace,
  livePromptFor,
  loadLiveSAEActs,
  loadSAEActs,
  putLiveSAEActs,
} from "../../data/interp";
import {
  ACCENT,
  crosshair,
  MARKER_HOT,
  type RGB,
  type Seg as ThemeSeg,
  withAlpha,
} from "./chart-theme";
import { InterpTooltip, type TipRow } from "./chart-tooltip";
import type { InterpDriver } from "./InterpDriver";

type LayersModule = typeof import("@deck.gl/layers");

const GL = 128; // px — row labels (feature id + top token)
const GR = 52; // px — per-row peak values
const GT = 100; // px — column token labels, below the prompt tracebar
const GB = 112; // px — chip row + the collapsed legend pill (bottom-right)
const BAND_GAP = 26; // px — gap + header above the sink band
const STRIP_GAP = 22; // px — gap + header above the L0/cos strips
const STRIP_H = 17; // px per text strip row

// intensity ramp endpoints — a zero cell is a distinct near-invisible grey so
// "didn't fire" never reads as "fired a little"
const LOW: [number, number, number] = [64, 66, 96];
const HIGH: [number, number, number] = [245, 195, 59];
const ZERO_RGBA: [number, number, number, number] = [118, 126, 158, 14];

interface Row {
  f: SAEActsFeature;
  band: "main" | "sink";
  idx: number; // row index within its band (after sorting)
}

interface Cell {
  poly: [number, number][];
  act: number;
  pos: number;
  row: Row;
}

export class SAEPianoRollDriver implements InterpDriver {
  readonly animated = false;
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private canvas!: HTMLCanvasElement;
  private tooltip!: InterpTooltip;
  private labelRoot!: HTMLElement;
  private chipRoot!: HTMLElement;

  private bundle: SAEActsBundle | null = null;
  private tr: SAEActsTrace | null = null;
  private rows: Row[] = [];
  private cells: Cell[] = [];
  private boardMax = 1;
  private scaleMode: "row" | "board" = "row";
  private hover: Cell | null = null;
  /** feature id the global cross-view selection outlines (if a row shows it) */
  private linked: number | null = null;

  private cssW = 1;
  private cssH = 1;
  private dpr = 1;
  private disposers: Array<() => void> = [];

  async init(canvas: HTMLCanvasElement, _tier: GpuTier, overlay: HTMLElement): Promise<void> {
    this.canvas = canvas;
    const [core, layers] = await Promise.all([import("@deck.gl/core"), import("@deck.gl/layers")]);
    this.layersMod = layers;
    // flipY:true + zoom 0 → world units are css pixels (same pixel-space
    // pattern as the head fingerprints; a grid has no isometric-axes claim).
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

  async setModel(model: string, trace?: string): Promise<void> {
    if (!this.bundle || this.bundle.meta.model !== model) {
      this.bundle = await loadSAEActs(model);
    }
    const traces = this.bundle.traces;
    let found = traces.find((t) => t.slug === trace);
    // 2c live SAE lens: a typed prompt's activations come from POST /live/sae
    // (same producer as the offline traces); never silently show a bundled
    // prompt under a live slug
    if (!found && trace && isLiveTrace(trace)) found = await this.fetchLiveTrace(model, trace);
    const tr = found ?? traces[0];
    if (!tr) throw new Error("sae_acts.json has no traces");
    this.tr = tr;

    // piano-roll reading order: rows sorted by WHERE they peak, then intensity,
    // so firing structure reads left→right like a score
    const argmax = (a: number[]) => a.indexOf(Math.max(...a));
    const main = [...tr.features]
      .sort((a, b) => argmax(a.acts) - argmax(b.acts) || b.max - a.max)
      .map((f, i): Row => ({ f, band: "main", idx: i }));
    const sink = tr.sink_features.map((f, i): Row => ({ f, band: "sink", idx: i }));
    this.rows = [...main, ...sink];
    this.boardMax = Math.max(1e-9, ...tr.features.map((f) => f.max));
    this.hover = null;

    this.layoutCells();
    this.buildChips();
    this.deck?.setProps({ viewState: this.viewState() });
    this.pushLayers();
    this.positionLabels();
  }

  /** Resolve a live slug: the in-memory cache first, else one POST /live/sae.
   *  The final read goes through loadLiveSAEActs so the ⤓ data capture records
   *  the synthetic live:// URL like any other bundle. */
  private async fetchLiveTrace(model: string, slug: string): Promise<SAEActsTrace> {
    try {
      return await loadLiveSAEActs(model, slug);
    } catch {
      // not cached — compute it now (needs the prompt text behind the slug)
    }
    const prompt = livePromptFor(model, slug);
    if (!prompt)
      throw new Error(
        "live SAE acts not loaded — type the prompt again (results live in memory only)",
      );
    const base = appStore.getState().probing.liveUrl.replace(/\/+$/, "");
    let res: Response;
    try {
      res = await fetch(`${base}/live/sae`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: prompt }),
      });
    } catch {
      throw new Error(
        `no live server at ${base} — start it: python -m nebulai.backend.interp.live_server`,
      );
    }
    const body = (await res.json()) as {
      error?: string;
      model?: string;
      trace?: SAEActsTrace;
    };
    if (!res.ok || !body.trace) throw new Error(body.error ?? `live server error (${res.status})`);
    if (body.model !== model)
      throw new Error(`live server runs ${body.model}, viewer shows ${model}`);
    putLiveSAEActs(model, slug, body.trace);
    return loadLiveSAEActs(model, slug);
  }

  // ---- pixel-space grid layout ---------------------------------------------
  private plotW(): number {
    return Math.max(1, this.cssW - GL - GR);
  }
  private plotH(): number {
    return Math.max(1, this.cssH - GT - GB);
  }
  private nMain(): number {
    return this.tr?.features.length ?? 0;
  }
  private nSink(): number {
    return this.tr?.sink_features.length ?? 0;
  }
  private rowH(): number {
    const fixed = (this.nSink() ? BAND_GAP : 0) + STRIP_GAP + 2 * STRIP_H;
    const h = (this.plotH() - fixed) / Math.max(1, this.nMain() + this.nSink());
    // low floor: on short stages thin rows beat overflowing the chip gutter
    return Math.min(22, Math.max(4, h));
  }
  private cellW(): number {
    return this.plotW() / Math.max(1, this.tr?.T ?? 1);
  }
  private rowY(r: Row): number {
    const rh = this.rowH();
    if (r.band === "main") return GT + r.idx * rh;
    return GT + this.nMain() * rh + BAND_GAP + r.idx * rh;
  }
  private stripY(): number {
    const rh = this.rowH();
    return (
      GT + this.nMain() * rh + (this.nSink() ? BAND_GAP + this.nSink() * rh : 0) + STRIP_GAP
    );
  }
  private viewState() {
    return {
      ortho: {
        target: [this.cssW / 2, this.cssH / 2, 0] as [number, number, number],
        zoom: 0,
      },
    };
  }

  private layoutCells(): void {
    const tr = this.tr;
    if (!tr) return;
    const cw = this.cellW();
    const rh = this.rowH();
    const cells: Cell[] = [];
    for (const row of this.rows) {
      const y0 = this.rowY(row);
      for (let pos = 0; pos < tr.T; pos++) {
        const x0 = GL + pos * cw;
        cells.push({
          poly: [
            [x0 + 0.5, y0 + 0.5],
            [x0 + cw - 0.5, y0 + 0.5],
            [x0 + cw - 0.5, y0 + rh - 1],
            [x0 + 0.5, y0 + rh - 1],
          ],
          act: row.f.acts[pos] ?? 0,
          pos,
          row,
        });
      }
    }
    this.cells = cells;
  }

  /** The scale a cell's brightness is normalized by — the sink band is ALWAYS
   *  per-row (its 60–100× outliers would otherwise erase the whole board). */
  private scaleOf(row: Row): number {
    if (row.band === "sink" || this.scaleMode === "row") return Math.max(1e-9, row.f.max);
    return this.boardMax;
  }

  private colorOf(c: Cell): [number, number, number, number] {
    if (c.act <= 0) return ZERO_RGBA;
    const t = Math.min(1, c.act / this.scaleOf(c.row));
    const r = Math.round(LOW[0] + t * (HIGH[0] - LOW[0]));
    const g = Math.round(LOW[1] + t * (HIGH[1] - LOW[1]));
    const b = Math.round(LOW[2] + t * (HIGH[2] - LOW[2]));
    return [r, g, b, Math.round(70 + 185 * t)];
  }

  private pushLayers(): void {
    if (!this.deck || !this.cells.length) return;
    const { SolidPolygonLayer, LineLayer } = this.layersMod;
    const hover = this.hover;
    const edges = hover
      ? hover.poly.map((p, i) => ({
          source: p,
          target: hover.poly[(i + 1) % hover.poly.length] as [number, number],
        }))
      : [];
    // ACCENT crosshair snapped to the hovered cell centre — this is a clean
    // rectilinear time×feature grid, so row/column guides read unambiguously
    // (req 4, mirroring the logit-attribution grid).
    const cw = this.cellW();
    const rh = this.rowH();
    const plotBounds = {
      x0: GL,
      y0: GT,
      x1: GL + this.plotW(),
      y1: this.stripY() - STRIP_GAP,
    };
    const cross: ThemeSeg[] = hover
      ? crosshair(GL + hover.pos * cw + cw / 2, this.rowY(hover.row) + rh / 2, plotBounds)
      : [];
    // cross-view linked feature — a steady ACCENT outline around its whole row
    const linkEdges: ThemeSeg[] = [];
    const linkedRow = this.linked !== null ? this.rows.find((r) => r.f.id === this.linked) : undefined;
    if (linkedRow) {
      const ly = this.rowY(linkedRow);
      const lx1 = GL + this.plotW();
      linkEdges.push(
        { source: [GL, ly], target: [lx1, ly] },
        { source: [lx1, ly], target: [lx1, ly + rh] },
        { source: [lx1, ly + rh], target: [GL, ly + rh] },
        { source: [GL, ly + rh], target: [GL, ly] },
      );
    }

    this.deck.setProps({
      layers: [
        new SolidPolygonLayer<Cell>({
          id: "pr-cells",
          data: this.cells,
          getPolygon: (c) => c.poly,
          getFillColor: (c) => this.colorOf(c),
          pickable: true,
        }),
        new LineLayer<ThemeSeg>({
          id: "pr-crosshair",
          data: cross,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: withAlpha(ACCENT, 0.5),
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        new LineLayer<ThemeSeg>({
          id: "pr-link",
          data: linkEdges,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: withAlpha(ACCENT, 0.95),
          getWidth: 1.6,
          widthUnits: "pixels",
          pickable: false,
        }),
        // hovered cell outline recolored to the danger LED reticle (req 4)
        new LineLayer<{ source: [number, number]; target: [number, number] }>({
          id: "pr-hover",
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

  private buildChips(): void {
    this.chipRoot.textContent = "";
    const mk = (label: string, mode: "row" | "board") => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "interp-neuron-chip";
      btn.textContent = label;
      btn.setAttribute("aria-pressed", String(this.scaleMode === mode));
      if (this.scaleMode === mode) btn.classList.add("is-active");
      btn.addEventListener("click", () => {
        if (this.scaleMode === mode) return;
        this.scaleMode = mode;
        this.hover = null;
        this.tooltip.hide();
        this.cells = this.cells.slice(); // new ref → colors regenerate
        this.buildChips();
        this.pushLayers();
      });
      this.chipRoot.appendChild(btn);
    };
    mk("scale: row peak", "row");
    mk(`scale: board peak ${fmt(this.boardMax)}`, "board");
  }

  private positionLabels(): void {
    this.labelRoot.textContent = "";
    const tr = this.tr;
    if (!tr) return;
    const rh = this.rowH();
    const cw = this.cellW();

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

    // column token labels — the prompt's real tokens, leading space shown as ␣
    for (let pos = 0; pos < tr.T; pos++) {
      const raw = tr.token_strs[pos] ?? "";
      const el = cap(raw.replace(/^ /, "␣"));
      el.style.maxWidth = `${Math.max(8, cw - 4).toFixed(0)}px`;
      el.style.overflow = "hidden";
      el.style.whiteSpace = "pre";
      el.title = JSON.stringify(raw);
      place(el, GL + pos * cw + Math.max(1, (cw - el.offsetWidth) / 2), GT - 20);
    }

    // row labels (feature id + direct-path top token) + per-row peak at right
    const showRowText = rh >= 10; // below this the text would overlap itself
    for (const row of this.rows) {
      const y = this.rowY(row) + rh / 2 - 8;
      if (showRowText) {
        const el = cap(`#${row.f.id} ↑${vis(row.f.top_tok)}`);
        el.style.maxWidth = `${GL - 12}px`;
        el.style.overflow = "hidden";
        el.style.whiteSpace = "pre";
        place(el, 8, y);
        const pk = cap(fmt(row.f.max));
        place(pk, GL + this.plotW() + 6, y);
      }
    }

    // sink band header — the outlier band is labeled, never silent
    if (this.nSink()) {
      const el = cap("first-token outlier features · always own row scale");
      el.style.color = "rgb(245,195,59)";
      place(el, GL, GT + this.nMain() * rh + 6);
    }

    // honesty strips: exact L0 and reconstruction cosine per position
    const sy = this.stripY();
    const rows: Array<[string, string, (i: number) => string]> = [
      ["L0", "features active (of 24576)", (i) => String(tr.l0[i] ?? "")],
      // 4 dp is the bundle's exact stored precision — 3 dp would round
      // cos 0.9999 up to a claimed-perfect "1.000"
      ["cos", "reconstruction cosine", (i) => (tr.cos[i] ?? 0).toFixed(4).replace(/^0/, "")],
    ];
    rows.forEach(([name, title, get], k) => {
      const lbl = cap(name);
      lbl.title = title;
      place(lbl, 8, sy + k * STRIP_H);
      for (let pos = 0; pos < tr.T; pos++) {
        const el = cap(get(pos));
        el.title = title;
        place(el, GL + pos * cw + Math.max(1, (cw - el.offsetWidth) / 2), sy + k * STRIP_H);
      }
    });
  }

  private onPointerMove(e: PointerEvent): void {
    if (!this.deck) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const info = this.deck.pickObject({ x, y, radius: 1, layerIds: ["pr-cells"] }) as
      | PickingInfo
      | null;
    const c = (info?.object as Cell | undefined) ?? null;
    const prev = this.hover;
    if (c !== prev && (c?.pos !== prev?.pos || c?.row !== prev?.row)) {
      this.hover = c;
      this.pushLayers();
    }
    if (!c || !this.tr) {
      this.tooltip.hide();
      this.canvas.style.cursor = "";
      return;
    }
    const f = c.row.f;
    const tok = this.tr.token_strs[c.pos] ?? "";
    const cc = this.colorOf(c);
    const swatch: RGB = [cc[0], cc[1], cc[2]];
    const firePct = 10 ** f.log_sparsity * 100;
    const rows: TipRow[] = [
      { kind: "label", text: `feature #${f.id} · “${tok}” (pos ${c.pos})`, swatch },
      { text: "act", value: c.act.toFixed(3), hot: true },
      { text: "row peak", value: fmt(f.max) },
      {
        text: `fires globally on ≈${firePct.toPrecision(2)}% of tokens (log₁₀ ${f.log_sparsity.toFixed(2)})`,
      },
      { text: `direct path ↑“${vis(f.top_tok)}” (+${f.top_val.toFixed(1)}) — skips blocks 8–11` },
    ];
    if (c.row.band === "sink") {
      rows.push({ text: "first-token outlier band (‖x₀‖ ≈ 30× other positions)" });
    }
    this.tooltip.show(rows);
    this.tooltip.move(x, y, this.cssW, this.cssH);
    this.canvas.style.cursor = "crosshair";
  }

  private onClick(e: PointerEvent): void {
    if (!this.deck) return;
    const rect = this.canvas.getBoundingClientRect();
    const info = this.deck.pickObject({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      radius: 1,
      layerIds: ["pr-cells"],
    }) as PickingInfo | null;
    const c = (info?.object as Cell | undefined) ?? null;
    if (!c) return;
    // toggle publish as the global cross-view SAE-feature selection
    const same = this.linked === c.row.f.id;
    appStore.getState().setInterpSelection(same ? null : { kind: "saeFeature", id: c.row.f.id });
  }

  /** Cross-view link: outline the selected feature's row when this prompt's
   *  top-k board actually contains it (the board only shows top features). */
  setSelection(sel: InterpSelection | null): void {
    const id = sel?.kind === "saeFeature" ? sel.id : null;
    if (id === this.linked) return;
    this.linked = id;
    if (this.cells.length) this.pushLayers();
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
    // static — cells only change on trace/scale switches
  }

  resize(width: number, height: number, dpr: number): void {
    this.cssW = width;
    this.cssH = height;
    this.dpr = dpr;
    this.layoutCells(); // rebuilds the array → fresh refs for deck
    this.hover = null;
    this.deck?.setProps({
      width,
      height,
      useDevicePixels: Math.min(dpr, 2),
      viewState: this.viewState(),
    });
    this.pushLayers();
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

/** Compact numeric format for peak labels: 497, 51.4, 8.31. */
function fmt(v: number): string {
  if (v >= 100) return v.toFixed(0);
  if (v >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

/** Visible-escape a token for labels: leading space → ␣, newline → ⏎. */
function vis(s: string): string {
  const t = s.replace(/\n/g, "⏎").replace(/^ /, "␣");
  return t || "·";
}
