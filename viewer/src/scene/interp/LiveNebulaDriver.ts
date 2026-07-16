/** #25 Live Prompt Nebula — a real forward pass on text YOU type.
 *
 *  The viewer sends the prompt to a local probe server (stdlib Python,
 *  `python -m nebulai.backend.interp.live_server`) that runs the SAME numpy
 *  GPT-2 forward as every offline trace bundle and returns per-(layer,
 *  position) logit-lens readouts: top-1 token + probability, Shannon entropy
 *  of the full 50,257-way lens distribution (absolute scale, 0 → log2 V =
 *  15.62 bits), and KL(final ‖ lens) at the same position. Nothing is
 *  precomputed; latency shown is the measured server compute time. When the
 *  server is offline the view says so — it never shows stale data as live
 *  or invents placeholders.
 *
 *  deck.gl (WebGL2), camera off, static — redraws only on new responses.
 *  Weights (~0.5 GB float32) stay on your machine; that's why this is a
 *  local process and not an in-browser port. */

import type { Deck, OrthographicView, PickingInfo } from "@deck.gl/core";
import { appStore } from "../../app/store";
import type { GpuTier } from "../../app/capabilities";
import { HOT, MARKER_HOT } from "./chart-theme";
import { InterpTooltip } from "./chart-tooltip";
import type { InterpDriver } from "./InterpDriver";

type LayersModule = typeof import("@deck.gl/layers");

const GL = 50;
const GR = 16;
const GT = 96; // header lines
const GB = 76; // status footer + collapsed legend pill
const IN_H = 42; // prompt input row (inside the plot band, below the header)

const AMBER = HOT; // shared --data-hot: sharp = low-entropy = bright amber
const CELL_LO: [number, number, number] = [40, 42, 60];

/** Cell tuple from the server: [top1_str, p, entropy_bits, kl_bits]. */
type LiveCell = [string, number, number, number];

interface LiveResponse {
  model: string;
  T: number;
  n_layer: number;
  truncated: boolean;
  max_tokens: number;
  ms: number;
  tokens: string[];
  final_top: [string, number][];
  cells: LiveCell[][];
  meta: { formula: string; entropy_max: number; vocab: number; pos0_note: string };
}

interface Rect {
  x0: number;
  y0: number;
  w: number;
  h: number;
}

interface Cell {
  poly: [number, number][];
  layer: number;
  t: number;
}

interface Bar {
  poly: [number, number][];
  rank: number;
}

const vis = (s: string): string =>
  s === ""
    ? "·"
    : s
        .replace(/\n/g, "⏎")
        .replace(/^ /, "␣")
        .replace(/[\x00-\x09\x0b-\x1f\x7f]/g, (c) =>
          String.fromCodePoint(0x2400 + (c === "\x7f" ? 0x21 : c.charCodeAt(0))),
        );

export class LiveNebulaDriver implements InterpDriver {
  readonly animated = false;
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private canvas!: HTMLCanvasElement;
  private tooltip!: InterpTooltip;
  private labelRoot!: HTMLElement;
  private inputRoot!: HTMLElement;
  private input!: HTMLInputElement;

  private resp: LiveResponse | null = null;
  private status: "boot" | "ok" | "computing" | "offline" = "boot";
  private lastError = "";
  private cells: Cell[] = [];
  private bars: Bar[] = [];
  private hoverCell: Cell | null = null;
  private hoverBar = -1;

  private debounceId: number | null = null;
  private inflight: AbortController | null = null;

  private cssW = 1;
  private cssH = 1;
  private dpr = 1;
  private disposers: Array<() => void> = [];

  private base(): string {
    return (appStore.getState().probing.liveUrl || "http://127.0.0.1:8123").replace(/\/+$/, "");
  }

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

    // prompt input — the feature's primary control, driver-owned overlay DOM
    this.inputRoot = document.createElement("div");
    this.inputRoot.style.position = "absolute";
    this.inputRoot.style.pointerEvents = "auto";
    this.input = document.createElement("input");
    this.input.type = "text";
    // starts EMPTY on purpose: the placeholder + "type a prompt above" status are
    // the empty state — a hardcoded default prompt fired an unrequested forward
    // on mount and forced demo tooling to suppress it
    this.input.value = "";
    this.input.placeholder = "type a prompt — every keystroke runs a real forward pass";
    this.input.setAttribute("aria-label", "live prompt");
    this.input.spellcheck = false;
    Object.assign(this.input.style, {
      width: "100%",
      boxSizing: "border-box",
      font: "600 12px/1.4 var(--font-mono, ui-monospace, monospace)",
      color: "var(--text, #e8eaf6)",
      background: "rgba(16, 20, 34, 0.85)",
      border: "1px solid rgba(148, 156, 190, 0.35)",
      borderRadius: "8px",
      padding: "7px 10px",
      outline: "none",
    } as CSSStyleDeclaration);
    this.input.addEventListener("focus", () => {
      this.input.style.borderColor = "rgb(245,195,59)";
    });
    this.input.addEventListener("blur", () => {
      this.input.style.borderColor = "rgba(148, 156, 190, 0.35)";
    });
    this.input.addEventListener("input", () => this.schedule(450));
    this.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") this.schedule(0);
    });
    this.inputRoot.appendChild(this.input);
    overlay.appendChild(this.inputRoot);

    const onMove = (e: PointerEvent) => this.onPointerMove(e);
    const onLeave = () => this.onLeave();
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    this.disposers.push(() => {
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
    });
  }

  async setModel(_model: string, _trace?: string): Promise<void> {
    // the server decides what it serves (response.model is displayed as-is);
    // probe health, then run the current prompt for real
    this.status = "boot";
    this.positionLabels();
    try {
      const ctl = new AbortController();
      const tid = window.setTimeout(() => ctl.abort(), 2500);
      const res = await fetch(`${this.base()}/live/health`, { signal: ctl.signal });
      window.clearTimeout(tid);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      this.schedule(0);
    } catch {
      this.status = "offline";
      this.lastError = "health check failed";
      this.rebuild();
    }
  }

  private schedule(delay: number): void {
    if (this.debounceId != null) window.clearTimeout(this.debounceId);
    this.debounceId = window.setTimeout(() => {
      this.debounceId = null;
      void this.request();
    }, delay);
  }

  private async request(): Promise<void> {
    const text = this.input.value;
    if (!text) {
      this.status = "ok"; // empty box: keep last grid, status says waiting
      this.positionLabels();
      return;
    }
    this.inflight?.abort();
    const ctl = new AbortController();
    this.inflight = ctl;
    this.status = "computing";
    this.positionLabels();
    try {
      const res = await fetch(`${this.base()}/live/forward`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
        signal: ctl.signal,
      });
      const body = (await res.json()) as LiveResponse & { error?: string };
      if (ctl.signal.aborted) return;
      if (!res.ok) throw new Error(body.error ?? `HTTP ${res.status}`);
      this.resp = body;
      this.status = "ok";
      this.hoverCell = null;
      this.hoverBar = -1;
      this.rebuild();
    } catch (e) {
      if (ctl.signal.aborted) return; // superseded by a newer keystroke
      this.status = "offline";
      this.lastError = e instanceof Error ? e.message : String(e);
      this.rebuild();
    } finally {
      if (this.inflight === ctl) this.inflight = null;
    }
  }

  // ---- regions ---------------------------------------------------------------
  private narrow(): boolean {
    return this.cssW < 640;
  }
  private gb(): number {
    // narrow: the collapsed legend pill owns the bottom-right band — clear it
    return this.narrow() ? 118 : GB;
  }
  private gridRect(): Rect {
    const top = GT + IN_H + 18; // 18px column-token row above the grid
    if (this.narrow()) {
      const h = Math.max(60, (this.cssH - this.gb() - top) * 0.58);
      return { x0: GL - 8, y0: top, w: this.cssW - GL - GR + 8, h };
    }
    const w = Math.max(120, (this.cssW - GL - GR) * 0.66);
    return { x0: GL - 8, y0: top, w: w + 8, h: Math.max(60, this.cssH - this.gb() - top) };
  }
  private barsRect(): Rect {
    const g = this.gridRect();
    if (this.narrow()) {
      const y0 = g.y0 + g.h + 30;
      return { x0: GL, y0, w: this.cssW - GL - GR, h: Math.max(40, this.cssH - this.gb() - y0) };
    }
    const x0 = g.x0 + g.w + 52;
    return { x0, y0: g.y0 + 18, w: Math.max(60, this.cssW - GR - x0), h: g.h - 18 };
  }
  /** Candidate-bar geometry: one column of 8 (wide) or two columns of 4
   *  (narrow — 8 rows would crush to ~10 px). Same 0→pmax scale both columns. */
  private barSlot(br: Rect, i: number): { x0: number; y0: number; w: number; rowH: number; labelW: number } {
    if (!this.narrow()) {
      const rowH = Math.min(26, br.h / 8);
      return { x0: br.x0, y0: br.y0 + i * rowH, w: br.w, rowH, labelW: 56 };
    }
    const colW = br.w / 2 - 8;
    const rowH = Math.min(26, br.h / 4);
    return {
      x0: br.x0 + (i < 4 ? 0 : colW + 16),
      y0: br.y0 + (i % 4) * rowH,
      w: colW,
      rowH,
      labelW: 50,
    };
  }
  private viewState() {
    return {
      ortho: { target: [this.cssW / 2, this.cssH / 2, 0] as [number, number, number], zoom: 0 },
    };
  }

  // ---- geometry --------------------------------------------------------------
  private rebuild(): void {
    this.cells = [];
    this.bars = [];
    const r = this.resp;
    if (r && this.status !== "offline") {
      const g = this.gridRect();
      const cw = g.w / r.T;
      const rh = g.h / (r.n_layer + 1);
      for (let L = 0; L <= r.n_layer; L++) {
        const y0 = g.y0 + L * rh;
        for (let t = 0; t < r.T; t++) {
          const x0 = g.x0 + t * cw;
          this.cells.push({
            poly: [
              [x0 + 0.5, y0 + 0.5],
              [x0 + cw - 0.5, y0 + 0.5],
              [x0 + cw - 0.5, y0 + rh - 0.5],
              [x0 + 0.5, y0 + rh - 0.5],
            ],
            layer: L,
            t,
          });
        }
      }
      const br = this.barsRect();
      const pmax = Math.max(1e-9, r.final_top[0]?.[1] ?? 0);
      r.final_top.forEach(([, p], i) => {
        const s = this.barSlot(br, i);
        const w = Math.max(1, (p / pmax) * (s.w - s.labelW - 8));
        this.bars.push({
          poly: [
            [s.x0 + s.labelW, s.y0 + 3],
            [s.x0 + s.labelW + w, s.y0 + 3],
            [s.x0 + s.labelW + w, s.y0 + s.rowH - 3],
            [s.x0 + s.labelW, s.y0 + s.rowH - 3],
          ],
          rank: i,
        });
      });
    }
    this.pushLayers();
    this.positionLabels();
  }

  private cellColor(c: Cell): [number, number, number, number] {
    const r = this.resp;
    if (!r) return [0, 0, 0, 0];
    const cell = r.cells[c.layer]?.[c.t];
    if (!cell) return [0, 0, 0, 0];
    const hmax = r.meta.entropy_max || 15.617;
    const conf = 1 - Math.min(1, cell[2] / hmax); // bright = LOW entropy (sharp)
    const k = this.hoverCell && this.hoverCell !== c ? 0.55 : 1;
    return [
      (CELL_LO[0] + (AMBER[0] - CELL_LO[0]) * conf) * k,
      (CELL_LO[1] + (AMBER[1] - CELL_LO[1]) * conf) * k,
      (CELL_LO[2] + (AMBER[2] - CELL_LO[2]) * conf) * k,
      70 + 185 * conf,
    ];
  }

  private pushLayers(): void {
    if (!this.deck) return;
    const { PolygonLayer, SolidPolygonLayer } = this.layersMod;
    const r = this.resp;
    const layers = [];

    if (r && this.cells.length) {
      const fin = r.cells[r.n_layer];
      layers.push(
        new SolidPolygonLayer<Cell>({
          id: "ln-cells",
          data: this.cells,
          getPolygon: (d) => d.poly,
          getFillColor: (d) => this.cellColor(d),
          pickable: true,
          updateTriggers: { getFillColor: [this.hoverCell, r] },
        }),
        // white outline = lens top-1 at (L,t) equals the model's final top-1 at t
        new PolygonLayer<Cell>({
          id: "ln-match",
          data: this.cells.filter((c) => {
            const cell = r.cells[c.layer]?.[c.t];
            return !!cell && !!fin?.[c.t] && cell[0] === fin[c.t]?.[0];
          }),
          getPolygon: (d) => d.poly,
          filled: false,
          stroked: true,
          getLineColor: [255, 255, 255, 150],
          getLineWidth: 1,
          lineWidthUnits: "pixels",
          pickable: false,
        }),
      );
      // in-cell token text is HTML (positionLabels) — a deck TextLayer's default
      // font atlas is ASCII-only and silently DROPS ␣/⏎/…/unicode glyphs, which
      // renders " the" as "the" (a different token) — a truthfulness bug
      // candidate bars (final softmax, exact p printed in labels)
      layers.push(
        new SolidPolygonLayer<Bar>({
          id: "ln-bars",
          data: this.bars,
          getPolygon: (d) => d.poly,
          getFillColor: (d) =>
            d.rank === this.hoverBar ? [255, 224, 130, 235] : [AMBER[0], AMBER[1], AMBER[2], 190 - d.rank * 12],
          pickable: true,
          updateTriggers: { getFillColor: [this.hoverBar] },
        }),
      );
      if (this.hoverCell) {
        layers.push(
          new PolygonLayer<Cell>({
            id: "ln-hover",
            data: [this.hoverCell],
            getPolygon: (d) => d.poly,
            filled: false,
            stroked: true,
            // red LED reticle locks the cursor cell — deliberately NOT white, so
            // it never reads as the semantic white "matches final top-1" outline
            getLineColor: [MARKER_HOT[0], MARKER_HOT[1], MARKER_HOT[2], 255],
            getLineWidth: 1.6,
            lineWidthUnits: "pixels",
            pickable: false,
          }),
        );
      }
    }
    this.deck.setProps({ layers });
  }

  // ---- labels ----------------------------------------------------------------
  private positionLabels(): void {
    this.labelRoot.textContent = "";
    const narrow = this.narrow();
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
    const fit = (variants: string[], maxPx: number): string =>
      variants.find((v) => v.length * 6.9 <= maxPx) ?? variants[variants.length - 1] ?? "";

    // input row placement (kept in sync with the label pass)
    this.inputRoot.style.left = `${GL}px`;
    this.inputRoot.style.top = `${GT + 2}px`;
    this.inputRoot.style.width = `${this.cssW - GL - GR}px`;

    const r = this.resp;
    const hmax = (r?.meta.entropy_max ?? 15.617).toFixed(1);
    const h1 = cap(
      fit(
        [
          `live prompt nebula · one real forward pass per edit · logit-lens top-1 per (layer, position)`,
          `live forward pass · logit-lens top-1 per (layer, position)`,
          `live forward · logit lens`,
        ],
        this.cssW - GL - GR,
      ),
    );
    h1.style.color = "rgb(245,195,59)";
    place(h1, GL, GT - 44);
    place(
      cap(
        fit(
          [
            `computed on request by a local probe server (same numpy forward as every bundle) · nothing precomputed`,
            `local probe server · same numpy forward as the bundles · nothing precomputed`,
            `local probe server · nothing precomputed`,
          ],
          this.cssW - GL - GR,
        ),
      ),
      GL,
      GT - 30,
    );
    place(
      cap(
        fit(
          [
            `color: entropy of the full ${r ? r.meta.vocab.toLocaleString() : "50,257"}-way lens distribution — dark ${hmax} bits (uniform) → bright 0 (certain) · outline = matches final top-1`,
            `color: lens entropy — dark ${hmax} bits → bright 0 · outline = matches final`,
            `dark ${hmax} bits → bright 0 (entropy)`,
          ],
          this.cssW - GL - GR,
        ),
      ),
      GL,
      GT - 16,
    );

    // status line (bottom-left; bottom-right belongs to the legend pill)
    const st = cap("", "interp-neuron-axis");
    if (this.status === "offline") {
      st.textContent = fit(
        [
          `server offline (${this.lastError}) — run: python -m nebulai.backend.interp.live_server · endpoint in Settings → Model Probing`,
          `server offline — run: python -m nebulai.backend.interp.live_server`,
          `server offline — start live_server`,
        ],
        this.cssW - GL - GR - 150,
      );
      st.style.color = "rgb(240,110,110)";
    } else if (this.status === "computing" || this.status === "boot") {
      st.textContent = this.status === "boot" ? "probing server…" : "computing (real forward pass)…";
      st.style.color = "rgb(166,173,200)";
    } else if (r) {
      st.textContent = fit(
        [
          `${r.ms.toFixed(0)} ms server compute · ${r.T} tokens${r.truncated ? ` (truncated to ${r.max_tokens})` : ""} · ${r.model} · ${this.base().replace(/^https?:\/\//, "")}`,
          `${r.ms.toFixed(0)} ms · ${r.T} tok${r.truncated ? " (trunc)" : ""} · ${r.model}`,
        ],
        this.cssW - GL - GR - 150,
      );
      st.style.color = "rgb(166,173,200)";
    } else {
      st.textContent = "type a prompt above";
      st.style.color = "rgb(166,173,200)";
    }
    place(st, GL, this.cssH - this.gb() + (this.narrow() ? 92 : 26));

    if (!r || this.status === "offline") return;

    // grid row + column labels
    const g = this.gridRect();
    const rh = g.h / (r.n_layer + 1);
    const rstep = Math.max(1, Math.ceil(14 / rh));
    for (let L = 0; L <= r.n_layer; L += rstep) {
      const el = cap(L === r.n_layer ? "fin" : `L${L}`);
      el.style.color = "rgb(166,173,200)";
      place(el, g.x0 - 30, g.y0 + L * rh + rh / 2 - 6);
    }
    const cw = g.w / r.T;
    const cstep = Math.max(1, Math.ceil(26 / cw));
    const maxCh = Math.max(2, Math.floor((cw * cstep) / 7));
    const trunc = (s: string, n: number) => (s.length > n ? `${s.slice(0, Math.max(1, n - 1))}…` : s);
    for (let t = 0; t < r.T; t += cstep) {
      const s = trunc(vis(r.tokens[t] ?? ""), maxCh);
      const el = cap(s);
      el.style.color = "rgb(200,205,228)";
      place(el, g.x0 + t * cw + Math.max(1, (cw - s.length * 6.9) / 2), g.y0 - 15);
    }
    // in-cell lens top-1 tokens where they fit (HTML so ␣/⏎/unicode render;
    // a deck TextLayer's ASCII atlas would silently drop them)
    if (cw >= 26 && rh >= 12) {
      const hm = r.meta.entropy_max || 15.617;
      const cellCh = Math.max(2, Math.floor(cw / 7));
      for (const c of this.cells) {
        const cell = r.cells[c.layer]?.[c.t];
        if (!cell) continue;
        const s = trunc(vis(cell[0]), cellCh);
        const el = cap(s);
        const conf = 1 - Math.min(1, cell[2] / hm);
        el.style.color = conf > 0.55 ? "rgba(20,20,28,0.95)" : "rgba(230,233,245,0.92)";
        place(el, g.x0 + c.t * cw + Math.max(1, (cw - s.length * 6.9) / 2), g.y0 + c.layer * rh + rh / 2 - 7);
      }
    }

    // candidates panel: title + token/p labels per bar
    const br = this.barsRect();
    const pmax = r.final_top[0]?.[1] ?? 0;
    const bt = cap(
      fit(
        [
          `next token — final softmax · bars 0 → p=${pmax.toFixed(3)}`,
          `next token · bars 0→${pmax.toFixed(3)}`,
        ],
        narrow ? this.cssW - GL - GR : br.w,
      ),
    );
    bt.style.color = "rgb(200,205,228)";
    place(bt, br.x0, br.y0 - 15);
    r.final_top.forEach(([tok, p], i) => {
      const s = this.barSlot(br, i);
      const lab = cap(vis(tok).slice(0, narrow ? 6 : 7));
      lab.style.color = "rgb(226,230,246)";
      place(lab, s.x0, s.y0 + s.rowH / 2 - 6);
      const pv = cap(p.toFixed(3));
      const barW = Math.max(1, (p / Math.max(1e-9, pmax)) * (s.w - s.labelW - 8));
      // value label rides the bar end; when the bar fills its slot (rank 0
      // always does) it moves INSIDE the bar in dark text instead of clipping
      const outX = s.x0 + s.labelW + barW + 6;
      if (outX + 40 > s.x0 + s.w) {
        pv.style.color = "rgba(20,20,28,0.95)";
        place(pv, s.x0 + s.labelW + barW - 42, s.y0 + s.rowH / 2 - 6);
      } else {
        pv.style.color = "rgb(166,173,200)";
        place(pv, outX, s.y0 + s.rowH / 2 - 6);
      }
    });
  }

  // ---- hover ------------------------------------------------------------------
  private onPointerMove(e: PointerEvent): void {
    if (!this.deck) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const info = this.deck.pickObject({ x, y, radius: 2, layerIds: ["ln-cells", "ln-bars"] }) as PickingInfo | null;
    const obj = info?.object as Cell | Bar | undefined;
    const c = obj && "layer" in obj ? obj : null;
    const b = obj && "rank" in obj ? obj : null;
    if (c !== this.hoverCell || (b?.rank ?? -1) !== this.hoverBar) {
      this.hoverCell = c;
      this.hoverBar = b?.rank ?? -1;
      this.pushLayers();
    }
    const r = this.resp;
    if ((!c && !b) || !r) {
      this.tooltip.hide();
      this.canvas.style.cursor = "";
      return;
    }
    if (c) {
      const cell = r.cells[c.layer]?.[c.t];
      const fin = r.cells[r.n_layer]?.[c.t];
      if (!cell || !fin) {
        this.tooltip.hide();
        return;
      }
      // brighter (lower-entropy) lens reading is the "hot" datum the cursor locks
      const sharp = cell[2] < (r.meta.entropy_max || 15.617) * 0.5;
      this.tooltip.show([
        {
          kind: "label",
          text: `${c.layer === r.n_layer ? "final" : `L${c.layer}`} · pos ${c.t} “${vis(r.tokens[c.t] ?? "")}”`,
        },
        { text: `lens → “${vis(cell[0])}”`, value: `p ${cell[1].toFixed(4)}`, hot: sharp },
        { text: "entropy", value: `${cell[2].toFixed(3)} bits` },
        { text: "KL(final ‖ lens)", value: `${cell[3].toFixed(3)} bits` },
        { text: `final → “${vis(fin[0])}”`, value: `p ${fin[1].toFixed(4)}` },
      ]);
    } else if (b) {
      const ft = r.final_top[b.rank];
      if (!ft) {
        this.tooltip.hide();
        return;
      }
      this.tooltip.show([
        { kind: "label", swatch: AMBER, text: `next-token candidate #${b.rank + 1}` },
        { text: `“${vis(ft[0])}”`, value: `${(ft[1] * 100).toFixed(2)}%`, hot: b.rank === 0 },
        { kind: "conf", text: "final softmax at the last position — full 50,257-way" },
      ]);
    }
    this.tooltip.move(x, y, this.cssW, this.cssH);
    this.canvas.style.cursor = "crosshair";
  }

  private onLeave(): void {
    if (this.hoverCell || this.hoverBar >= 0) {
      this.hoverCell = null;
      this.hoverBar = -1;
      this.pushLayers();
    }
    this.tooltip.hide();
    this.canvas.style.cursor = "";
  }

  frame(_dt: number, _t: number): void {
    // static — redraws only on responses/hover
  }

  resize(width: number, height: number, dpr: number): void {
    this.cssW = width;
    this.cssH = height;
    this.dpr = dpr;
    this.hoverCell = null;
    this.hoverBar = -1;
    this.deck?.setProps({
      width,
      height,
      useDevicePixels: Math.min(dpr, 2),
      viewState: this.viewState(),
    });
    this.rebuild();
  }

  dispose(): void {
    if (this.debounceId != null) window.clearTimeout(this.debounceId);
    this.inflight?.abort();
    for (const d of this.disposers) d();
    this.disposers = [];
    this.tooltip.dispose();
    this.labelRoot.remove();
    this.inputRoot.remove();
    this.deck?.finalize();
    this.deck = null;
  }
}
