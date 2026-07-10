/** #22 Direction Compass — where SAE feature directions point.
 *
 *  One point per SAE feature: x = the exact max cosine between its decoder
 *  direction and ANY of the 50,257 token embeddings (W_E rows), y = the exact
 *  max cosine vs ANY of the 36,864 MLP neuron write directions (c_proj rows).
 *  Both axes are exhaustively computed cosines on the same 0–1 scale, so the
 *  diagonal y = x is meaningful: above it, the feature is closer to a neuron
 *  than to any token. The yardstick is a MEASURED baseline — random unit
 *  directions scanned the same way — drawn as guide lines per axis. Color is
 *  the best neuron's layer (same viridis ramp as #6); the SAE reads the
 *  residual BEFORE block 8, so best-neurons in L8–11 are flagged on hover as
 *  geometric-only (they write after the hook).
 *
 *  deck.gl (WebGL2), camera off, static. Source: compass.json ⋈ sae.json. */

import type { Deck, OrthographicView, PickingInfo } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { type CompassBundle, type SAEBundle, loadCompass, loadSAE } from "../../data/interp";
import type { InterpDriver } from "./InterpDriver";
import { LAYER_COLORS } from "./NeuronFieldDriver";

type LayersModule = typeof import("@deck.gl/layers");

const GL = 46; // px — y-axis (neuron cos) tick labels
const GR = 16;
const GT = 96; // px — header summary
const GB = 92; // px — exemplar chips + collapsed legend pill

const AMBER: [number, number, number] = [245, 195, 59];
const GUIDE: [number, number, number, number] = [118, 126, 158, 130];
const DIAG: [number, number, number, number] = [138, 146, 178, 90];

interface Pt {
  i: number; // feature index
}

interface Seg {
  source: [number, number];
  target: [number, number];
}

export class CompassDriver implements InterpDriver {
  readonly animated = false;
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private canvas!: HTMLCanvasElement;
  private tooltip!: HTMLElement;
  private labelRoot!: HTMLElement;
  private chipRoot!: HTMLElement;

  private com: CompassBundle | null = null;
  private sae: SAEBundle | null = null;
  private pts: Pt[] = []; // deterministic shuffled draw order
  private px: Float32Array = new Float32Array(0);
  private layoutGen = 0;
  private hover: number | null = null; // feature index
  private sel: number | null = null; // pinned feature
  private nAsNeuron = 0; // nc > 0.9
  private nInBox = 0; // both cosines below the random p99
  private aboveFrac = 0; // fraction with nc > tc (above the diagonal)

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
    if (!this.com || this.com.meta.model !== model) {
      const [com, sae] = await Promise.all([loadCompass(model), loadSAE(model)]);
      this.com = com;
      this.sae = sae;
      const n = com.meta.d_sae;
      // deterministic draw-order shuffle — index-ordered data would paint a
      // systematic z-bias (the #6/#2b lesson); 5323 is coprime to 24576
      this.pts = Array.from({ length: n }, (_, k) => ({ i: (k * 5323) % n }));
      const bn = com.meta.baseline.neuron.p99;
      const bt = com.meta.baseline.token.p99;
      this.nAsNeuron = com.nc.filter((v) => v > 0.9).length;
      this.nInBox = com.nc.filter((v, i) => v < bn && (com.tc[i] ?? 1) < bt).length;
      this.aboveFrac = com.nc.filter((v, i) => v > (com.tc[i] ?? 0)).length / n;
      // default pin: the strongest neuron alignment — the "this feature IS a
      // neuron direction" story
      this.sel = com.exemplars[0]?.f ?? null;
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
    return this.narrow() ? 128 : GB;
  }
  private plotW(): number {
    return Math.max(40, this.cssW - GL - GR);
  }
  private plotH(): number {
    return Math.max(40, this.cssH - GT - this.gb() - 18); // 18 = x tick row
  }
  /** both axes are cosine on an identical 0–1 scale — the diagonal is honest */
  private xOf(cos: number): number {
    return GL + Math.max(0, Math.min(1, cos)) * this.plotW();
  }
  private yOf(cos: number): number {
    return GT + (1 - Math.max(0, Math.min(1, cos))) * this.plotH();
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
    const com = this.com;
    if (!com) return;
    const n = com.meta.d_sae;
    this.px = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      this.px[i * 2] = this.xOf(com.tc[i] ?? 0);
      this.px[i * 2 + 1] = this.yOf(com.nc[i] ?? 0);
    }
    this.layoutGen++;
  }

  private layerOf(i: number): number {
    const com = this.com;
    if (!com) return 0;
    return Math.floor((com.ni[i] ?? 0) / com.meta.d_mlp);
  }

  private pushLayers(): void {
    if (!this.deck || !this.com) return;
    const { ScatterplotLayer, LineLayer } = this.layersMod;
    const com = this.com;
    const b = com.meta.baseline;

    // measured random-direction baselines: horizontal for the neuron family,
    // vertical for the token family — the box near the origin is "random"
    const guides: Seg[] = [
      ...[b.neuron.mean, b.neuron.p99].map(
        (v): Seg => ({
          source: [GL, this.yOf(v)],
          target: [GL + this.plotW(), this.yOf(v)],
        }),
      ),
      ...[b.token.mean, b.token.p99].map(
        (v): Seg => ({
          source: [this.xOf(v), GT],
          target: [this.xOf(v), GT + this.plotH()],
        }),
      ),
    ];
    const diag: Seg[] = [
      { source: [this.xOf(0), this.yOf(0)], target: [this.xOf(1), this.yOf(1)] },
    ];
    const rings: { i: number; hov: boolean }[] = [];
    if (this.sel != null) rings.push({ i: this.sel, hov: false });
    if (this.hover != null) rings.push({ i: this.hover, hov: true });

    this.deck.setProps({
      layers: [
        new LineLayer<Seg>({
          id: "compass-diag",
          data: diag,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: DIAG,
          getWidth: 1,
          widthUnits: "pixels",
          updateTriggers: { getSourcePosition: this.layoutGen, getTargetPosition: this.layoutGen },
          pickable: false,
        }),
        new LineLayer<Seg>({
          id: "compass-guides",
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
          id: "compass-pts",
          data: this.pts,
          getPosition: (p) => [this.px[p.i * 2] ?? 0, this.px[p.i * 2 + 1] ?? 0, 0],
          getFillColor: (p) => {
            const [r, g, bl] = LAYER_COLORS[this.layerOf(p.i)] ?? [205, 210, 224];
            return [r, g, bl, 140];
          },
          getRadius: this.narrow() ? 1.3 : 1.6,
          radiusUnits: "pixels",
          updateTriggers: { getPosition: this.layoutGen },
          pickable: true,
        }),
        new ScatterplotLayer<{ i: number; hov: boolean }>({
          id: "compass-rings",
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
          updateTriggers: { getPosition: this.layoutGen },
          pickable: false,
        }),
      ],
    });
  }

  // ---- chips: exemplar features ------------------------------------------------
  private buildChips(): void {
    const com = this.com;
    if (!com) return;
    this.chipRoot.textContent = "";
    this.chipRoot.style.bottom = this.narrow() ? "110px" : "";
    // one row only — wrapped chips collide with the feature pill. ~115px/chip
    const per = this.narrow() ? 2 : Math.max(2, Math.min(4, Math.floor((this.cssW - 40) / 230)));
    const picks = [
      ...com.exemplars.filter((e) => e.kind === "neuron").slice(0, per),
      ...com.exemplars.filter((e) => e.kind === "token").slice(0, per),
    ];
    for (const e of picks) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "interp-neuron-chip";
      const L = this.layerOf(e.f);
      const unit = (com.ni[e.f] ?? 0) % com.meta.d_mlp;
      const tok = vis(com.tok_strs[com.ti_u[e.f] ?? 0] ?? "");
      btn.textContent =
        e.kind === "neuron"
          ? this.narrow()
            ? `n ${e.cos.toFixed(2)}`
            : `${e.cos.toFixed(4)} → L${L}/${unit}`
          : this.narrow()
            ? `t ${e.cos.toFixed(2)}`
            : `${e.cos.toFixed(4)} → “${tok}”`;
      btn.title =
        e.kind === "neuron"
          ? `#${e.f} — max cos ${e.cos} with neuron L${L}/${unit} · top alignments, one chip per distinct neuron`
          : `#${e.f} — max cos ${e.cos} with token “${tok}” · top alignments, one chip per distinct token`;
      const active = this.sel === e.f;
      btn.setAttribute("aria-pressed", String(active));
      if (active) btn.classList.add("is-active");
      btn.addEventListener("click", () => {
        this.sel = active ? null : e.f;
        this.buildChips();
        this.pushLayers();
        this.positionLabels();
      });
      this.chipRoot.appendChild(btn);
    }
  }

  // ---- labels -----------------------------------------------------------------
  private positionLabels(): void {
    this.labelRoot.textContent = "";
    const com = this.com;
    const sae = this.sae;
    if (!com || !sae) return;
    const narrow = this.narrow();
    const b = com.meta.baseline;

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

    // measured label width: ~6.74 px/char (6.9 with margin). Pick the longest
    // variant that actually fits — thresholds by guessed width ellipsized
    // honesty tails twice during review
    const fit = (s: string) => s.length * 6.9 + (narrow ? 24 : GL + 12) <= this.cssW;
    const pick3 = (full: string, mid: string, short: string) =>
      fit(full) ? full : fit(mid) ? mid : short;
    const hMax = `${(this.cssW - (narrow ? 12 : GL) - 12).toFixed(0)}px`;
    const upPct = (com.meta.upstream_frac * 100).toFixed(1);
    const h1 = cap(
      pick3(
        `direction compass · max cos of ${com.meta.d_sae.toLocaleString()} decoder dirs vs all ` +
          `${com.meta.n_neurons.toLocaleString()} neuron write-dirs and ${com.meta.n_tokens.toLocaleString()} ` +
          `token embs · ${this.nAsNeuron} feats ≈ a neuron (>0.9) · ${upPct}% upstream`,
        `compass · ${com.meta.d_sae.toLocaleString()} dirs vs ` +
          `${com.meta.n_neurons.toLocaleString()} neurons + ${com.meta.n_tokens.toLocaleString()} tokens · ` +
          `${this.nAsNeuron} ≈ a neuron · ${upPct}% upstream`,
        `compass · ${this.nAsNeuron} ≈ a neuron · ${upPct}% upstream`,
      ),
    );
    h1.style.color = "rgb(245,195,59)";
    h1.style.maxWidth = hMax;
    place(h1, narrow ? 12 : GL, GT - 44);
    const h2 = cap(
      pick3(
        `x = max cos vs W_E · y = vs c_proj (same 0–1 scale) · ` +
          `rand-dir max-cos mean/p99: neuron ${b.neuron.mean}/${b.neuron.p99} · ` +
          `token ${b.token.mean}/${b.token.p99} · ${this.nInBox} in random box`,
        `x = max cos vs W_E · y = vs c_proj · rand mean n ${b.neuron.mean} t ${b.token.mean} · ${this.nInBox} in box`,
        `x = max cos vs W_E · y = vs c_proj · rand ~${b.neuron.mean}`,
      ),
    );
    h2.style.maxWidth = hMax;
    place(h2, narrow ? 12 : GL, GT - 30);
    // pinned-feature caption — third header line, built from exact data
    if (this.sel != null) {
      const i = this.sel;
      const L = this.layerOf(i);
      const unit = (com.ni[i] ?? 0) % com.meta.d_mlp;
      const tok = vis(com.tok_strs[com.ti_u[i] ?? 0] ?? "");
      const own = vis(sae.top_tok[i] ?? "");
      const short3 = `#${i} · n L${L} ${(com.nc[i] ?? 0).toFixed(3)} · t ${(com.tc[i] ?? 0).toFixed(3)}`;
      const h3 = cap(
        pick3(
          `#${i} · neuron L${L}/${unit} cos ${(com.nc[i] ?? 0).toFixed(4)}${L >= 8 ? " (after hook)" : ""} · ` +
            `token “${tok}” cos ${(com.tc[i] ?? 0).toFixed(4)} · ↑“${own}” · fires ${pct(sae.log_sparsity[i] ?? -10)}`,
          `#${i} · neuron L${L}/${unit} ${(com.nc[i] ?? 0).toFixed(4)} · token “${tok}” ${(com.tc[i] ?? 0).toFixed(4)}`,
          short3,
        ),
      );
      h3.style.color = "rgb(245,195,59)";
      h3.style.maxWidth = hMax;
      place(h3, narrow ? 12 : GL, GT - 16);
    }

    // ticks: identical 0–1 cosine scale on both axes
    for (let v = 0; v <= 10; v += 2) {
      place(cap((v / 10).toFixed(1)), Math.max(2, GL - 26), this.yOf(v / 10) - 7);
    }
    const yTick = GT + this.plotH() + 4;
    for (let v = 0; v <= 10; v += 2) {
      place(cap((v / 10).toFixed(1)), this.xOf(v / 10) - 6, yTick);
    }

    // baseline guide labels — data lives left of x≈0.7, so the right side is free
    const gNFull = this.plotW() > 640;
    const gN = cap(
      gNFull ? `random-dir p99 (neuron) = ${b.neuron.p99} (measured)` : `rand p99 n ${b.neuron.p99}`,
    );
    gN.style.color = "rgb(118,126,158)";
    place(gN, GL + this.plotW() - (gNFull ? 310 : 118), this.yOf(b.neuron.p99) - 14);
    const gT = cap(
      narrow ? `t ${String(b.token.p99).replace(/^0\./, ".")}` : `rand p99 (token) ${b.token.p99}`,
    );
    gT.style.color = "rgb(118,126,158)";
    place(gT, this.xOf(b.token.p99) + 4, GT + this.plotH() - 16);
    // diagonal label — x > 0.7 is guaranteed empty (max token cos ≈ 0.67), so
    // the label rides above the line there without covering data
    if (!narrow) {
      const dFull = `y=x · above: nearer a neuron (${(this.aboveFrac * 100).toFixed(1)}%)`;
      const dX = this.xOf(0.7) + 8;
      const dLab = cap(dX + dFull.length * 6.9 <= this.cssW - 4 ? dFull : "y = x");
      dLab.style.color = "rgb(138,146,178)";
      place(dLab, dX, this.yOf(0.7) - 16);
    }

    // layer color legend — swatches carry exact counts; hook boundary marked.
    // Sits in the empty lower-right (all data has nc ≥ 0.14, tc ≤ 0.68 — the
    // corner below the diagonal is guaranteed empty)
    const leg = document.createElement("div");
    leg.className = "interp-neuron-axis";
    leg.style.display = "flex";
    leg.style.alignItems = "center";
    leg.style.gap = narrow ? "2px" : "3px";
    for (let L = 0; L < com.layer_counts.length; L++) {
      if (L === 8) {
        const bar = document.createElement("span");
        bar.textContent = "│";
        bar.title = "SAE hook: resid_pre 8 — layers right of this write AFTER the hook";
        bar.style.color = "rgb(245,195,59)";
        leg.appendChild(bar);
      }
      const sw = document.createElement("span");
      const [r, g, bl] = LAYER_COLORS[L] ?? [205, 210, 224];
      sw.style.width = narrow ? "8px" : "9px";
      sw.style.height = narrow ? "8px" : "9px";
      sw.style.borderRadius = "2px";
      sw.style.background = `rgb(${r},${g},${bl})`;
      sw.title = `L${L}: best match for ${com.layer_counts[L]} features`;
      leg.appendChild(sw);
    }
    const legTxt = document.createElement("span");
    const legStr =
      narrow || this.cssW < 900 ? "L0→11 │=hook" : "color = best-neuron layer L0→L11 · │ = hook";
    legTxt.textContent = legStr;
    legTxt.style.marginLeft = "4px";
    this.labelRoot.appendChild(leg);
    leg.appendChild(legTxt);
    // right-anchored so it never clips; both placements sit in provably empty
    // plot area: no feature has nc < 0.1397, so the strip below cos ≈ 0.13 is
    // data-free (on wide the lower-right is empty too — max token cos ≈ 0.67)
    const legW = 12 * (narrow ? 10 : 12) + 12 + legStr.length * 6.9 + 8;
    place(
      leg,
      this.cssW - legW - 6,
      narrow ? this.yOf(0.05) - 5 : GT + this.plotH() - 40,
    );
  }

  // ---- interaction --------------------------------------------------------------
  private pick(e: PointerEvent): number | null {
    if (!this.deck) return null;
    const rect = this.canvas.getBoundingClientRect();
    const info = this.deck.pickObject({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      radius: 4,
      layerIds: ["compass-pts"],
    }) as PickingInfo | null;
    const p = info?.object as Pt | undefined;
    return p ? p.i : null;
  }

  private onPointerMove(e: PointerEvent): void {
    const com = this.com;
    const sae = this.sae;
    if (!com || !sae) return;
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
    const L = this.layerOf(i);
    const unit = (com.ni[i] ?? 0) % com.meta.d_mlp;
    const tok = vis(com.tok_strs[com.ti_u[i] ?? 0] ?? "");
    this.tooltip.innerHTML = "";
    const add = (cls: string, text: string) => {
      const el = document.createElement("div");
      el.className = cls;
      el.textContent = text;
      this.tooltip.appendChild(el);
    };
    add(
      "point-tooltip-label",
      `#${i} — best neuron L${L}/${unit} · cos ${(com.nc[i] ?? 0).toFixed(4)}`,
    );
    if (L >= 8) add("point-tooltip-conf", `L${L} writes AFTER the hook — geometric only, not a source`);
    add("point-tooltip-conf", `best token “${tok}” · cos ${(com.tc[i] ?? 0).toFixed(4)}`);
    add(
      "point-tooltip-conf",
      `↑“${vis(sae.top_tok[i] ?? "")}” ${(sae.top_val[i] ?? 0).toFixed(2)} · fires ${pct(sae.log_sparsity[i] ?? -10)}`,
    );
    add(
      "point-tooltip-conf",
      `rand-dir mean: n ${com.meta.baseline.neuron.mean} · t ${com.meta.baseline.token.mean} · click to pin`,
    );
    this.tooltip.style.visibility = "visible";
    const rect = this.canvas.getBoundingClientRect();
    const px = Math.min(e.clientX - rect.left + 14, this.cssW - 330);
    const py = Math.min(e.clientY - rect.top + 14, this.cssH - 120);
    this.tooltip.style.transform = `translate(${px.toFixed(1)}px, ${py.toFixed(1)}px)`;
    this.canvas.style.cursor = "pointer";
  }

  private onClick(e: PointerEvent): void {
    if (!this.com) return;
    const i = this.pick(e);
    if (i == null) return;
    this.sel = this.sel === i ? null : i;
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
    if (this.com) this.buildChips();
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

/** Visible-escape a token for labels: leading space → ␣, newline → ⏎, other
 *  C0 controls → their Unicode control pictures (␋ etc — some W_E best-matches
 *  are control-char tokens that would otherwise render as nothing). */
function vis(s: string): string {
  const t = s
    .replace(/\n/g, "⏎")
    .replace(/^ /, "␣")
    .replace(/[\x00-\x09\x0b-\x1f\x7f]/g, (c) =>
      String.fromCodePoint(0x2400 + (c === "\x7f" ? 0x21 : c.charCodeAt(0))),
    );
  return t || "·";
}
