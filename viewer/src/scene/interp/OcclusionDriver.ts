/** #19 Occlusion Vignette — leave-one-token-out causal attribution.
 *
 *  For the selected prompt, each position is occluded — substituted with
 *  <|endoftext|> ("sub", positions preserved) or deleted outright ("del",
 *  later positions shift) — and one real forward measures how far the
 *  baseline top-1 next token's log-probability falls. Bars diverge from a
 *  zero axis: up (amber) = the token supported the prediction, down (blue) =
 *  it was suppressing it. A white outline marks positions whose occlusion
 *  flips the top-1 prediction — the flip target is shown.
 *
 *  Completes the causal-lens set: input occlusion (here), residual patching
 *  (#14), head ablation (#17), direct logit attribution (#13).
 *
 *  deck.gl (WebGL2), camera off, static. Source: occlusion.json. */

import type { Deck, OrthographicView, PickingInfo } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { type OcclusionBundle, type OcclusionPrompt, loadOcclusion } from "../../data/interp";
import type { InterpDriver } from "./InterpDriver";

type LayersModule = typeof import("@deck.gl/layers");

const GL = 46; // px — y-axis scale labels (wide) / token labels start (narrow)
const GR = 16;
const GT = 96; // px — header summary, below the prompt tracebar
const GB = 92; // px — mode chips + collapsed legend pill

const AMBER: [number, number, number] = [245, 195, 59];
const BLUE: [number, number, number] = [96, 165, 250];
const GUIDE: [number, number, number, number] = [118, 126, 158, 110];

type Mode = "sub" | "del";
const MODES: Mode[] = ["sub", "del"];

interface Bar {
  poly: [number, number][]; // the bar rect
  hoverPoly: [number, number][]; // full slot column/row for picking
  i: number; // position index
  v: number; // drop_lp, nats
  flipped: boolean;
  tip: [number, number]; // bar end, for flip labels
}

interface Seg {
  source: [number, number];
  target: [number, number];
}

export class OcclusionDriver implements InterpDriver {
  readonly animated = false;
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private canvas!: HTMLCanvasElement;
  private tooltip!: HTMLElement;
  private labelRoot!: HTMLElement;
  private chipRoot!: HTMLElement;

  private bundle: OcclusionBundle | null = null;
  private entry: OcclusionPrompt | null = null;
  private mode: Mode = "sub";
  private bars: Bar[] = [];
  private hover: Bar | null = null;
  private maxPos = 1; // max positive drop of current entry+mode (scale, stated)
  private maxNeg = 0; // max |negative| drop (same nats-per-px as maxPos)

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
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    this.disposers.push(() => {
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
    });
  }

  async setModel(model: string, trace?: string): Promise<void> {
    if (!this.bundle || this.bundle.meta.model !== model) {
      this.bundle = await loadOcclusion(model);
    }
    this.entry =
      this.bundle.prompts.find((e) => e.slug === trace) ?? this.bundle.prompts[0] ?? null;
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
  private plotH(): number {
    return Math.max(40, this.cssH - GT - this.gb());
  }

  private viewState() {
    return {
      ortho: {
        target: [this.cssW / 2, this.cssH / 2, 0] as [number, number, number],
        zoom: 0,
      },
    };
  }

  private drops(): number[] {
    return this.entry?.[this.mode].drop_lp ?? [];
  }
  private newTop(i: number, mode: Mode = this.mode): [string, number] {
    return this.entry?.[mode].new_top[i] ?? ["", 0];
  }
  private isFlip(i: number, mode: Mode = this.mode): boolean {
    const e = this.entry;
    return !!e && this.newTop(i, mode)[0] !== e.top1[0];
  }

  /** One linear scale both directions: nats-per-px is identical above and
   *  below the axis; the axis sits where the data's own +/− extents put it. */
  private layout(): void {
    const e = this.entry;
    this.bars = [];
    if (!e) return;
    const d = this.drops();
    this.maxPos = Math.max(0.001, ...d.filter((v) => v > 0));
    this.maxNeg = Math.max(0, ...d.map((v) => (v < 0 ? -v : 0)));
    const T = e.T;
    if (this.narrow()) {
      // rows: token labels left, horizontal bars diverge from a vertical axis
      const labelW = 86;
      const x0 = GL + labelW;
      const avail = Math.max(40, this.cssW - x0 - GR - 8);
      const s = avail / (this.maxPos + this.maxNeg); // px per nat
      const axisX = x0 + this.maxNeg * s;
      const rowH = Math.min(30, this.plotH() / T);
      for (let i = 0; i < T; i++) {
        const y0 = GT + i * rowH + 3;
        const y1 = GT + (i + 1) * rowH - 3;
        const v = d[i] ?? 0;
        const bx0 = v >= 0 ? axisX : axisX - -v * s;
        const bx1 = v >= 0 ? axisX + v * s : axisX;
        this.bars.push({
          poly: [
            [bx0, y0],
            [bx1, y0],
            [bx1, y1],
            [bx0, y1],
          ],
          hoverPoly: [
            [GL, GT + i * rowH],
            [this.cssW - GR, GT + i * rowH],
            [this.cssW - GR, GT + (i + 1) * rowH],
            [GL, GT + (i + 1) * rowH],
          ],
          i,
          v,
          flipped: this.isFlip(i),
          tip: [v >= 0 ? bx1 : bx0, (y0 + y1) / 2],
        });
      }
    } else {
      // columns: bars diverge from a horizontal axis, token labels beneath
      const labelH = 24;
      const slotW = Math.min(120, (this.cssW - GL - GR) / T);
      const barZone = Math.max(40, this.plotH() - labelH - 8);
      const s = barZone / (this.maxPos + this.maxNeg); // px per nat
      const axisY = GT + this.maxPos * s;
      for (let i = 0; i < T; i++) {
        const x0 = GL + i * slotW + 4;
        const x1 = GL + (i + 1) * slotW - 4;
        const v = d[i] ?? 0;
        const by0 = v >= 0 ? axisY - v * s : axisY;
        const by1 = v >= 0 ? axisY : axisY + -v * s;
        this.bars.push({
          poly: [
            [x0, by0],
            [x1, by0],
            [x1, by1],
            [x0, by1],
          ],
          hoverPoly: [
            [GL + i * slotW, GT],
            [GL + (i + 1) * slotW, GT],
            [GL + (i + 1) * slotW, GT + barZone],
            [GL + i * slotW, GT + barZone],
          ],
          i,
          v,
          flipped: this.isFlip(i),
          tip: [(x0 + x1) / 2, v >= 0 ? by0 : by1],
        });
      }
    }
  }

  private axisSegs(): Seg[] {
    const e = this.entry;
    if (!e) return [];
    if (this.narrow()) {
      const x0 = GL + 86;
      const avail = Math.max(40, this.cssW - x0 - GR - 8);
      const axisX = x0 + (this.maxNeg / (this.maxPos + this.maxNeg)) * avail;
      const rowH = Math.min(30, this.plotH() / e.T);
      return [{ source: [axisX, GT], target: [axisX, GT + e.T * rowH] }];
    }
    const barZone = Math.max(40, this.plotH() - 24 - 8);
    const axisY = GT + (this.maxPos / (this.maxPos + this.maxNeg)) * barZone;
    const slotW = Math.min(120, (this.cssW - GL - GR) / e.T);
    return [{ source: [GL, axisY], target: [GL + e.T * slotW, axisY] }];
  }

  private pushLayers(): void {
    if (!this.deck || !this.entry) return;
    const { SolidPolygonLayer, LineLayer } = this.layersMod;

    const outline = (poly: [number, number][]): Seg[] =>
      poly.map((p, i) => ({
        source: p,
        target: poly[(i + 1) % poly.length] as [number, number],
      }));
    const flipSegs: Seg[] = [];
    for (const bar of this.bars) if (bar.flipped) flipSegs.push(...outline(bar.poly));
    const hoverSegs: Seg[] = this.hover ? outline(this.hover.hoverPoly) : [];

    this.deck.setProps({
      layers: [
        new SolidPolygonLayer<Bar>({
          id: "occ-bars",
          data: this.bars,
          getPolygon: (b) => b.poly,
          getFillColor: (b) =>
            b.v >= 0 ? [AMBER[0], AMBER[1], AMBER[2], 215] : [BLUE[0], BLUE[1], BLUE[2], 215],
          updateTriggers: { getPolygon: this.mode, getFillColor: this.mode },
          pickable: false,
        }),
        new SolidPolygonLayer<Bar>({
          id: "occ-slots",
          data: this.bars,
          getPolygon: (b) => b.hoverPoly,
          getFillColor: [0, 0, 0, 1], // effectively invisible; exists for picking
          pickable: true,
        }),
        new LineLayer<Seg>({
          id: "occ-axis",
          data: this.axisSegs(),
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: GUIDE,
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
        new LineLayer<Seg>({
          id: "occ-flip",
          data: flipSegs,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: [255, 255, 255, 210],
          getWidth: 1.2,
          widthUnits: "pixels",
          updateTriggers: { getSourcePosition: this.mode, getTargetPosition: this.mode },
          pickable: false,
        }),
        new LineLayer<Seg>({
          id: "occ-hover",
          data: hoverSegs,
          getSourcePosition: (e) => [e.source[0], e.source[1], 0],
          getTargetPosition: (e) => [e.target[0], e.target[1], 0],
          getColor: [255, 255, 255, 160],
          getWidth: 1,
          widthUnits: "pixels",
          pickable: false,
        }),
      ],
    });
  }

  // ---- chips: occlusion mode --------------------------------------------------
  private buildChips(): void {
    if (!this.bundle) return;
    this.chipRoot.textContent = "";
    this.chipRoot.style.bottom = this.narrow() ? "110px" : "";
    for (const mode of MODES) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "interp-neuron-chip";
      btn.textContent = this.narrow()
        ? mode
        : mode === "sub"
          ? "substitute ⟨endoftext⟩"
          : "delete & shift";
      btn.title =
        mode === "sub"
          ? "replace the token with <|endoftext|> — positions preserved"
          : "delete the token — later tokens shift left, positional embeddings move";
      btn.setAttribute("aria-pressed", String(mode === this.mode));
      if (mode === this.mode) btn.classList.add("is-active");
      btn.addEventListener("click", () => {
        if (mode === this.mode) return;
        this.mode = mode;
        this.hover = null;
        this.tooltip.style.visibility = "hidden";
        this.layout();
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
    const b = this.bundle;
    const e = this.entry;
    if (!b || !e) return;
    const narrow = this.narrow();

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
    const signed = (v: number, dp = 2) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(dp)}`;

    const d = this.drops();
    let topI = 0;
    d.forEach((v, i) => {
      if (v > (d[topI] ?? 0)) topI = i;
    });
    const compact = this.cssW < 1000;
    const hMax = `${(this.cssW - (narrow ? 12 : GL) - 12).toFixed(0)}px`;
    const h1 = cap(
      narrow
        ? `occlude · top1 “${vis(e.top1[0])}” p ${e.top1[1].toFixed(3)} · top “${vis(e.token_strs[topI] ?? "")}” ${signed(d[topI] ?? 0)}`
        : compact
          ? `occlusion vignette · baseline top-1 “${vis(e.top1[0])}” p ${e.top1[1].toFixed(4)} · ` +
              `top ${this.mode} drop “${vis(e.token_strs[topI] ?? "")}” ${signed(d[topI] ?? 0)} nats`
          : `occlusion vignette · baseline top-1 “${vis(e.top1[0])}” p ${e.top1[1].toFixed(4)} ` +
              `(logit ${e.top1[2].toFixed(2)}) · top ${this.mode}-occlusion drop “${vis(e.token_strs[topI] ?? "")}” ` +
              `${signed(d[topI] ?? 0, 4)} nats`,
    );
    h1.style.color = "rgb(245,195,59)";
    h1.style.maxWidth = hMax;
    place(h1, narrow ? 12 : GL, GT - 44);
    const modeNote =
      this.mode === "sub" ? "⟨endoftext⟩ substitute, positions kept" : "deleted, positions shift";
    const h2 = cap(
      narrow
        ? `drop = Δ log p(“${vis(e.top1[0])}”) · ` +
            `${this.mode === "sub" ? "⟨eot⟩ sub" : "del, pos shift"} · ${b.meta.n_forward} fwd`
        : compact
          ? `drop = log p_base − log p_occ (baseline argmax) · ` +
              `${this.mode === "sub" ? "⟨eot⟩ sub, positions kept" : "deleted, positions shift"} · ` +
              `${b.meta.n_forward} fwd · drift ${b.meta.causal_drift}`
          : `drop = log p_base(“${vis(e.top1[0])}”) − log p_occluded, one real forward per bar · ` +
              `${modeNote} · white outline = top-1 flips · ${b.meta.n_forward} forwards (all prompts) · ` +
              `causal drift ${b.meta.causal_drift}`,
    );
    h2.style.maxWidth = hMax;
    place(h2, narrow ? 12 : GL, GT - 30);

    if (narrow) {
      // rows — token labels in the left gutter, scale ticks along the top edge
      const labelW = 86;
      const rowH = Math.min(30, this.plotH() / e.T);
      for (let i = 0; i < e.T; i++) {
        const el = cap(`${i} ${vis(e.token_strs[i] ?? "")}`);
        el.style.maxWidth = `${labelW - 4}px`;
        if (i === topI) el.style.color = "rgb(245,195,59)";
        place(el, GL, GT + i * rowH + rowH / 2 - 8);
      }
      const axis = this.axisSegs()[0];
      if (axis) {
        place(cap("0"), axis.source[0] - 3, GT - 14);
        place(cap(signed(this.maxPos, this.maxPos >= 1 ? 1 : 2)), this.cssW - GR - 40, GT - 14);
        // the −extent tick only fits when the axis sits clear of the left edge
        if (this.maxNeg > 0 && axis.source[0] - (GL + labelW) >= 34) {
          place(cap(signed(-this.maxNeg, this.maxNeg >= 1 ? 1 : 2)), GL + labelW, GT - 14);
        }
      }
    } else {
      // columns — token labels beneath the bar zone, y scale on the left
      const slotW = Math.min(120, (this.cssW - GL - GR) / e.T);
      const labelH = 24;
      const barZone = Math.max(40, this.plotH() - labelH - 8);
      for (let i = 0; i < e.T; i++) {
        const el = cap(vis(e.token_strs[i] ?? ""));
        el.style.maxWidth = `${(slotW - 8).toFixed(0)}px`;
        if (i === topI) el.style.color = "rgb(245,195,59)";
        place(el, GL + i * slotW + 4, GT + barZone + 6);
      }
      const axis = this.axisSegs()[0];
      const axisY = axis ? axis.source[1] : GT + barZone;
      if (axis) {
        place(cap("0"), Math.max(2, GL - 14), axisY - 7);
        place(cap(signed(this.maxPos, this.maxPos >= 1 ? 1 : 2)), 2, GT - 2);
        // the −extent tick only fits when the axis sits clear of the bottom
        if (this.maxNeg > 0 && GT + barZone - axisY >= 24) {
          place(cap(signed(-this.maxNeg, this.maxNeg >= 1 ? 1 : 2)), 2, GT + barZone - 12);
        }
      }
      // per-bar values + flip targets, only when slots have room; labels that
      // would escape the panel (tallest bar → header, tiny bar → token row)
      // move inside/above instead
      if (slotW >= 56) {
        for (const bar of this.bars) {
          if (Math.abs(bar.v) < 0.005) continue;
          let vy = bar.v >= 0 ? bar.tip[1] - 15 : bar.tip[1] + 3;
          let fy = bar.v >= 0 ? bar.tip[1] - 29 : bar.tip[1] + 17;
          let inside = false;
          if (bar.v >= 0 && fy < GT + 2) {
            inside = true; // tallest bars: labels sit inside the amber bar
            vy = bar.tip[1] + 4;
            fy = bar.tip[1] + 18;
          } else if (bar.v < 0 && fy > GT + barZone - 14) {
            vy = axisY - 15; // tiny negative bars: use the empty space above
            fy = axisY - 29;
          }
          const el = cap(signed(bar.v));
          el.style.color = inside
            ? "rgba(20,22,34,0.95)"
            : bar.v >= 0
              ? "rgb(245,195,59)"
              : "rgb(96,165,250)";
          place(el, bar.tip[0] - 14, vy);
          if (bar.flipped) {
            const fl = cap(`→${vis(this.newTop(bar.i)[0])}`);
            fl.style.color = inside ? "rgba(20,22,34,0.9)" : "rgba(255,255,255,0.9)";
            fl.style.maxWidth = `${(slotW - 4).toFixed(0)}px`;
            place(fl, bar.tip[0] - 14, fy);
          }
        }
      }
      if (!compact) {
        place(cap("drop in log p of the baseline top-1 (nats, linear) · one bar = one real occluded forward"), GL + 4, GT + barZone - 12);
      }
    }
  }

  // ---- interaction --------------------------------------------------------------
  private pick(e: PointerEvent): Bar | null {
    if (!this.deck) return null;
    const rect = this.canvas.getBoundingClientRect();
    const info = this.deck.pickObject({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      radius: 1,
      layerIds: ["occ-slots"],
    }) as PickingInfo | null;
    return (info?.object as Bar | undefined) ?? null;
  }

  private onPointerMove(ev: PointerEvent): void {
    const e = this.entry;
    if (!e) return;
    const bar = this.pick(ev);
    if (bar !== this.hover) {
      this.hover = bar;
      this.pushLayers();
    }
    if (!bar) {
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
    const signed = (v: number, dp = 4) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(dp)}`;
    const i = bar.i;
    const dLp = e[this.mode].drop_lp[i] ?? 0;
    const dLogit = e[this.mode].drop_logit[i] ?? 0;
    const [nt, ntP] = this.newTop(i);
    const other: Mode = this.mode === "sub" ? "del" : "sub";
    const [ont] = this.newTop(i, other);
    add(
      "point-tooltip-label",
      `pos ${i} “${vis(e.token_strs[i] ?? "")}” — ${this.mode} drop ${signed(dLp)} nats (Δlogit ${signed(dLogit)})`,
    );
    add(
      "point-tooltip-conf",
      this.isFlip(i)
        ? `top-1 FLIPS: “${vis(e.top1[0])}” → “${vis(nt)}” p ${ntP.toFixed(4)}`
        : `occluded top-1 stays “${vis(nt)}” p ${ntP.toFixed(4)} (baseline p ${e.top1[1].toFixed(4)})`,
    );
    add(
      "point-tooltip-conf",
      `${other}: ${signed(e[other].drop_lp[i] ?? 0)} · top-1 “${vis(ont)}”${this.isFlip(i, other) ? " (flip)" : ""}`,
    );
    add(
      "point-tooltip-conf",
      "drop > 0 = this token supported the prediction · one real forward",
    );
    this.tooltip.style.visibility = "visible";
    const rect = this.canvas.getBoundingClientRect();
    const px = Math.min(ev.clientX - rect.left + 14, this.cssW - 330);
    const py = Math.min(ev.clientY - rect.top + 14, this.cssH - 110);
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
    // static — redraws on mode/trace switches only
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
    if (this.bundle) this.buildChips();
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

/** Visible-escape a token for labels: leading space → ␣, newline → ⏎. */
function vis(s: string): string {
  const t = s.replace(/\n/g, "⏎").replace(/^ /, "␣");
  return t || "·";
}
