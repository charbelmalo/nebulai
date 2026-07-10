/** #12 Decoder Cosine Web — feature splitting, measured.
 *
 *  One point per SAE feature: x = the release's measured firing fraction
 *  (log₁₀), y = the maximum cosine between its decoder direction and any
 *  other feature's (all 24,576² ordered pairs scanned). No layout, no
 *  projection — both axes are computed quantities. The measured random-pair
 *  baseline (mean / p99 / p99.9) is drawn as guides, so "unusually close" has
 *  a yardstick instead of an eyeballed threshold. Hovering a feature draws
 *  the line to its nearest neighbor with both features' readouts; chips pin
 *  the top twin pairs.
 *
 *  deck.gl (WebGL2), camera off, static. Source: sae_web.json ⋈ sae.json. */

import type { Deck, OrthographicView, PickingInfo } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { type SAEBundle, type SAEWebBundle, loadSAE, loadSAEWeb } from "../../data/interp";
import type { InterpDriver } from "./InterpDriver";

type LayersModule = typeof import("@deck.gl/layers");

const GL = 46; // px — y-axis (cosine) tick labels
const GR = 16;
const GT = 96; // px — header summary
const GB = 92; // px — twin-pair chips + collapsed legend pill

const AMBER: [number, number, number] = [245, 195, 59];
const SLATE: [number, number, number] = [138, 146, 178];
const GUIDE: [number, number, number, number] = [118, 126, 158, 130];

interface Pt {
  i: number; // feature index
}

interface Seg {
  source: [number, number];
  target: [number, number];
}

export class SAEWebDriver implements InterpDriver {
  readonly animated = false;
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private canvas!: HTMLCanvasElement;
  private tooltip!: HTMLElement;
  private labelRoot!: HTMLElement;
  private chipRoot!: HTMLElement;

  private web: SAEWebBundle | null = null;
  private sae: SAEBundle | null = null;
  private pts: Pt[] = []; // deterministic shuffled draw order
  private px: Float32Array = new Float32Array(0); // layout positions (x,y)
  private layoutGen = 0; // bumped per layout — updateTrigger for accessors
  private hover: number | null = null; // feature index
  private sel: [number, number] | null = null; // pinned pair

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

  async setModel(model: string): Promise<void> {
    if (!this.web || this.web.meta.model !== model) {
      const [web, sae] = await Promise.all([loadSAEWeb(model), loadSAE(model)]);
      this.web = web;
      this.sae = sae;
      const n = web.meta.d_sae;
      // deterministic draw-order shuffle — index-ordered data would paint a
      // systematic z-bias (the #6/#2b lesson)
      this.pts = Array.from({ length: n }, (_, k) => ({ i: (k * 5323) % n }));
      // default pinned pair: the strongest twin that is NOT an exact duplicate
      // (cos < 0.999) — the semantic-split story; the ⏎ clique gets chip #1
      const distinct = this.distinctPairs();
      const sem = distinct.find((p) => p.cos < 0.999) ?? distinct[0];
      this.sel = sem ? [sem.i, sem.j] : null;
    }
    this.hover = null;
    this.layout();
    this.buildChips();
    this.deck?.setProps({ viewState: this.viewState() });
    this.pushLayers();
    this.positionLabels();
  }

  // ---- pixel-space layout ---------------------------------------------------
  private narrow(): boolean {
    return this.cssW < 640;
  }
  private gb(): number {
    return this.narrow() ? 150 : GB;
  }
  private plotW(): number {
    return Math.max(40, this.cssW - GL - GR);
  }
  private plotH(): number {
    return Math.max(40, this.cssH - GT - this.gb() - 18); // 18 = x tick row
  }
  private xOf(ls: number): number {
    const c = Math.max(-10, Math.min(0, ls));
    return GL + ((c + 10) / 10) * this.plotW();
  }
  private yOf(cos: number): number {
    const c = Math.max(0, Math.min(1, cos));
    return GT + (1 - c) * this.plotH();
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
    const web = this.web;
    const sae = this.sae;
    if (!web || !sae) return;
    const n = web.meta.d_sae;
    this.px = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      this.px[i * 2] = this.xOf(sae.log_sparsity[i] ?? -10);
      this.px[i * 2 + 1] = this.yOf(web.nn_cos[i] ?? 0);
    }
    this.layoutGen++;
  }

  /** top pairs deduped by feature AND by readout pair — the ⏎ duplicate
   *  clique otherwise fills every chip slot with the same direction */
  private distinctPairs(): { i: number; j: number; cos: number; mutual: boolean }[] {
    const web = this.web;
    const sae = this.sae;
    if (!web || !sae) return [];
    const seen = new Set<number>();
    const seenTok = new Set<string>();
    const out: { i: number; j: number; cos: number; mutual: boolean }[] = [];
    for (const p of web.pairs) {
      if (seen.has(p.i) || seen.has(p.j)) continue;
      const sig = `${sae.top_tok[p.i]}~${sae.top_tok[p.j]}`;
      if (seenTok.has(sig)) continue;
      seen.add(p.i);
      seen.add(p.j);
      seenTok.add(sig);
      out.push(p);
      if (out.length >= 5) break;
    }
    return out;
  }

  private pushLayers(): void {
    if (!this.deck || !this.web) return;
    const { ScatterplotLayer, LineLayer } = this.layersMod;
    const web = this.web;
    const b = web.meta.baseline;

    const pair = (a: number, c: number): Seg => ({
      source: [this.px[a * 2] ?? 0, this.px[a * 2 + 1] ?? 0],
      target: [this.px[c * 2] ?? 0, this.px[c * 2 + 1] ?? 0],
    });
    const selSegs: Seg[] = this.sel ? [pair(this.sel[0], this.sel[1])] : [];
    const hovSegs: Seg[] =
      this.hover != null ? [pair(this.hover, web.nn_idx[this.hover] ?? this.hover)] : [];
    const rings: { i: number; hov: boolean }[] = [];
    if (this.sel) rings.push({ i: this.sel[0], hov: false }, { i: this.sel[1], hov: false });
    if (this.hover != null) {
      rings.push(
        { i: this.hover, hov: true },
        { i: web.nn_idx[this.hover] ?? this.hover, hov: true },
      );
    }
    // measured random-pair baseline guides — the yardstick lines
    const guides: Seg[] = [b.mean, b.p99, b.p999].map((v) => ({
      source: [GL, this.yOf(v)],
      target: [GL + this.plotW(), this.yOf(v)],
    }));

    this.deck.setProps({
      layers: [
        new LineLayer<Seg>({
          id: "web-guides",
          data: guides,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: GUIDE,
          getWidth: 1,
          widthUnits: "pixels",
          updateTriggers: { getSourcePosition: this.layoutGen, getTargetPosition: this.layoutGen },
          pickable: false,
        }),
        new ScatterplotLayer<Pt>({
          id: "web-pts",
          data: this.pts,
          getPosition: (p) => [this.px[p.i * 2] ?? 0, this.px[p.i * 2 + 1] ?? 0, 0],
          getFillColor: (p) =>
            web.mutual[p.i]
              ? [AMBER[0], AMBER[1], AMBER[2], 175]
              : [SLATE[0], SLATE[1], SLATE[2], 110],
          getRadius: this.narrow() ? 1.4 : 1.7,
          radiusUnits: "pixels",
          updateTriggers: { getPosition: this.layoutGen },
          pickable: true,
        }),
        new LineLayer<Seg>({
          id: "web-sel",
          data: selSegs,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: [AMBER[0], AMBER[1], AMBER[2], 230],
          getWidth: 1.4,
          widthUnits: "pixels",
          pickable: false,
        }),
        new LineLayer<Seg>({
          id: "web-hov",
          data: hovSegs,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: [255, 255, 255, 200],
          getWidth: 1.2,
          widthUnits: "pixels",
          pickable: false,
        }),
        new ScatterplotLayer<{ i: number; hov: boolean }>({
          id: "web-rings",
          data: rings,
          getPosition: (r) => [this.px[r.i * 2] ?? 0, this.px[r.i * 2 + 1] ?? 0, 0],
          getFillColor: [0, 0, 0, 0],
          getLineColor: (r) => (r.hov ? [255, 255, 255, 220] : [AMBER[0], AMBER[1], AMBER[2], 230]),
          getRadius: 5,
          radiusUnits: "pixels",
          stroked: true,
          filled: false,
          getLineWidth: 1.4,
          lineWidthUnits: "pixels",
          pickable: false,
        }),
      ],
    });
  }

  // ---- chips: pinned twin pairs ------------------------------------------------
  private buildChips(): void {
    const web = this.web;
    const sae = this.sae;
    if (!web || !sae) return;
    this.chipRoot.textContent = "";
    this.chipRoot.style.bottom = this.narrow() ? "110px" : "";
    const pairs = this.distinctPairs();
    pairs.forEach((p) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "interp-neuron-chip";
      btn.textContent = this.narrow()
        ? p.cos.toFixed(2)
        : `${p.cos.toFixed(4)} ${vis(sae.top_tok[p.i] ?? "")}~${vis(sae.top_tok[p.j] ?? "")}`;
      btn.title =
        `#${p.i} (↑${vis(sae.top_tok[p.i] ?? "")}) ~ #${p.j} (↑${vis(sae.top_tok[p.j] ?? "")}) — ` +
        `cos ${p.cos}${p.mutual ? ", mutual" : ""} · top pairs listed once per feature`;
      const active = this.sel?.[0] === p.i && this.sel?.[1] === p.j;
      btn.setAttribute("aria-pressed", String(active));
      if (active) btn.classList.add("is-active");
      btn.addEventListener("click", () => {
        this.sel = active ? null : [p.i, p.j];
        this.buildChips();
        this.pushLayers();
        this.positionLabels();
      });
      this.chipRoot.appendChild(btn);
    });
  }

  // ---- labels -----------------------------------------------------------------
  private positionLabels(): void {
    this.labelRoot.textContent = "";
    const web = this.web;
    const sae = this.sae;
    if (!web || !sae) return;
    const narrow = this.narrow();
    const b = web.meta.baseline;

    const cap = (text: string, cls = "interp-neuron-axis") => {
      const el = document.createElement("div");
      el.className = cls;
      el.textContent = text;
      el.style.overflow = "hidden";
      el.style.textOverflow = "ellipsis";
      el.style.whiteSpace = "nowrap";
      this.labelRoot.appendChild(el);
      return el;
    };
    const place = (el: HTMLElement, x: number, y: number) => {
      el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
    };

    const top = web.pairs[0];
    const compact = this.cssW < 1000;
    const hMax = `${(this.cssW - (narrow ? 12 : GL) - 12).toFixed(0)}px`;
    const nTwins = web.nn_cos.filter((v) => v > 0.9).length;
    const h1 = cap(
      narrow
        ? `decoder web · ${nTwins} twins cos>0.9 · top ${top ? top.cos.toFixed(2) : "—"}`
        : compact
          ? `decoder cosine web · nn cos over all ${web.meta.d_sae.toLocaleString()}² pairs · ` +
              `${nTwins} twins >0.9 · top #${top?.i}~#${top?.j} ${top?.cos.toFixed(4)}`
          : `decoder cosine web · nearest-neighbor cosine over all ${web.meta.d_sae.toLocaleString()}² ` +
              `decoder-direction pairs · ${nTwins} features have a twin above 0.9 · ` +
              `top pair #${top?.i}~#${top?.j} cos ${top?.cos.toFixed(4)} (↑${vis(sae.top_tok[top?.i ?? 0] ?? "")})`,
    );
    h1.style.color = "rgb(245,195,59)";
    h1.style.maxWidth = hMax;
    place(h1, narrow ? 12 : GL, GT - 44);
    const h2 = cap(
      narrow
        ? `y = max cos · rand mean ${b.mean} · p99.9 ${b.p999}`
        : compact
          ? `x = firing fraction (log₁₀) · y = maxⱼ cos W_dec · ` +
              `rand mean ${b.mean} · p99.9 ${b.p999} · n ${b.n_pairs.toLocaleString()}`
          : `x = firing fraction (log₁₀, release eval set) · y = maxⱼ cos(W_dec[i], W_dec[j]) · ` +
              `random-pair baseline mean ${b.mean} · p99 ${b.p99} · p99.9 ${b.p999} ` +
              `(n ${b.n_pairs.toLocaleString()}, seed ${b.seed}) · ${web.meta.mutual_count.toLocaleString()} mutual`,
    );
    h2.style.maxWidth = hMax;
    place(h2, narrow ? 12 : GL, GT - 30);
    // pinned-pair caption — third header line
    if (this.sel) {
      const [i, j] = this.sel;
      const cos = web.nn_idx[i] === j ? web.nn_cos[i] : web.nn_cos[j];
      // narrow: shorter form (no "pair", tokens truncated with a visible "…",
      // no fires) so the honesty numbers never get eaten by CSS ellipsis
      const tok = (t: string) => (narrow && t.length > 8 ? `${t.slice(0, 7)}…` : t);
      const h3 = cap(
        `${narrow ? "" : "pair "}#${i}~#${j} · cos ${(cos ?? 0).toFixed(4)} · ` +
          `↑${tok(vis(sae.top_tok[i] ?? ""))} / ↑${tok(vis(sae.top_tok[j] ?? ""))}` +
          (narrow
            ? ""
            : ` · fires ${pct(sae.log_sparsity[i] ?? -10)} / ${pct(sae.log_sparsity[j] ?? -10)}`),
      );
      h3.style.color = "rgb(245,195,59)";
      h3.style.maxWidth = hMax;
      place(h3, narrow ? 12 : GL, GT - 16);
    }

    // y ticks (cosine) + baseline guide labels
    for (let v = 0; v <= 10; v += 2) {
      place(cap((v / 10).toFixed(1)), Math.max(2, GL - 26), this.yOf(v / 10) - 7);
    }
    // on narrow the point cloud fills the right half — guide labels go left,
    // into the empty low-firing region, instead of on top of the data
    const gLab = cap(compact ? `rand p99.9 ${b.p999}` : `random-pair p99.9 = ${b.p999} (measured)`);
    gLab.style.color = "rgb(118,126,158)";
    place(gLab, narrow ? GL + 8 : GL + this.plotW() - (compact ? 110 : 210), this.yOf(b.p999) - 14);
    const gLab2 = cap(`p99 ${b.p99}`);
    gLab2.style.color = "rgb(118,126,158)";
    place(gLab2, narrow ? GL + 8 : GL + this.plotW() - 60, this.yOf(b.p99) - 14);
    // mean line (cos ≈ 0.004) sits on the x-axis — its value lives in the h2
    // header; an in-plot label would collide with the "0 = 100%" tick.

    // x ticks: log10 firing fraction, with % translations where round
    const xt: [number, string][] = narrow
      ? [
          [-10, "dead"],
          [-6, "10⁻⁶"],
          [-4, "10⁻⁴"],
          [-2, "1%"],
          [0, "100%"],
        ]
      : [
          [-10, "−10 (dead floor)"],
          [-8, "−8"],
          [-6, "−6"],
          [-4, "10⁻⁴ = 0.01%"],
          [-2, "10⁻² = 1%"],
          [0, "0 = 100%"],
        ];
    // x tick labels sit just above the axis INSIDE the plot — the band below
    // it belongs to the collapsed legend pill (the recurring #13 lesson), and
    // the lower-right of the plot is empty (few features have nn_cos < 0.2)
    const yTick = GT + this.plotH() - 16;
    for (const [v, s] of xt) {
      place(cap(s), this.xOf(v) - (v === 0 ? (narrow ? 34 : 58) : 4), yTick);
    }
    if (!narrow && !compact) {
      place(
        cap("x = release-measured firing fraction · every point is one SAE feature — no layout, no projection"),
        GL + 4,
        GT + 2,
      );
    }
  }

  // ---- interaction --------------------------------------------------------------
  private pick(e: PointerEvent): number | null {
    if (!this.deck) return null;
    const rect = this.canvas.getBoundingClientRect();
    const info = this.deck.pickObject({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      radius: 4,
      layerIds: ["web-pts"],
    }) as PickingInfo | null;
    const p = info?.object as Pt | undefined;
    return p ? p.i : null;
  }

  private onPointerMove(e: PointerEvent): void {
    const web = this.web;
    const sae = this.sae;
    if (!web || !sae) return;
    const i = this.pick(e);
    if (i !== this.hover) {
      this.hover = i;
      this.pushLayers();
    }
    if (i == null) {
      this.tooltip.style.visibility = "hidden";
      this.canvas.style.cursor = "";
      return;
    }
    const j = web.nn_idx[i] ?? i;
    this.tooltip.innerHTML = "";
    const add = (cls: string, text: string) => {
      const el = document.createElement("div");
      el.className = cls;
      el.textContent = text;
      this.tooltip.appendChild(el);
    };
    add(
      "point-tooltip-label",
      `#${i} — nn cos ${(web.nn_cos[i] ?? 0).toFixed(4)} → #${j}${web.mutual[i] ? " · mutual" : ""}`,
    );
    add(
      "point-tooltip-conf",
      `#${i}: fires ${pct(sae.log_sparsity[i] ?? -10)} · ↑“${vis(sae.top_tok[i] ?? "")}” ${(sae.top_val[i] ?? 0).toFixed(2)}`,
    );
    add(
      "point-tooltip-conf",
      `#${j}: fires ${pct(sae.log_sparsity[j] ?? -10)} · ↑“${vis(sae.top_tok[j] ?? "")}” ${(sae.top_val[j] ?? 0).toFixed(2)}`,
    );
    add(
      "point-tooltip-conf",
      `random-pair mean ${this.web?.meta.baseline.mean} · p99.9 ${this.web?.meta.baseline.p999} · click to pin the pair`,
    );
    this.tooltip.style.visibility = "visible";
    const rect = this.canvas.getBoundingClientRect();
    const px = Math.min(e.clientX - rect.left + 14, this.cssW - 330);
    const py = Math.min(e.clientY - rect.top + 14, this.cssH - 110);
    this.tooltip.style.transform = `translate(${px.toFixed(1)}px, ${py.toFixed(1)}px)`;
    this.canvas.style.cursor = "pointer";
  }

  private onClick(e: PointerEvent): void {
    const web = this.web;
    if (!web) return;
    const i = this.pick(e);
    if (i == null) return;
    const j = web.nn_idx[i] ?? i;
    const same = this.sel && this.sel[0] === i && this.sel[1] === j;
    this.sel = same ? null : [i, j];
    this.buildChips();
    this.pushLayers();
    this.positionLabels();
  }

  private onLeave(): void {
    if (this.hover != null) {
      this.hover = null;
      this.pushLayers();
    }
    this.tooltip.style.visibility = "hidden";
    this.canvas.style.cursor = "";
  }

  frame(_dt: number, _t: number): void {
    // static — redraws on hover/selection/resize only
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
    if (this.web) this.buildChips();
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

/** measured firing fraction as a percentage string from log10 sparsity */
function pct(ls: number): string {
  if (ls <= -9.5) return "~never (dead)";
  const p = 10 ** ls * 100;
  return p >= 0.01 ? `${p.toFixed(p >= 1 ? 2 : 3)}%` : `${p.toExponential(1)}%`;
}

/** Visible-escape a token for labels: leading space → ␣, newline → ⏎. */
function vis(s: string): string {
  const t = s.replace(/\n/g, "⏎").replace(/^ /, "␣");
  return t || "·";
}
