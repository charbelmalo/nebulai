/** #2c Composition Web — which heads feed which heads, from weights alone.
 *
 *  Q/K/V composition scores (Elhage et al. 2021) between every cross-layer
 *  head pair: how much of head 1's OV write lands in head 2's query, key, or
 *  value channel. Raw Frobenius composition has a positive floor for
 *  UNRELATED maps, so the bundle ships that floor measured over seeded random
 *  factor pairs, and this view only draws arcs at a stated multiple of it —
 *  everything below is honestly not drawn (nodes still show and report their
 *  strongest partners on hover). Arc geometry is layout, not data: layers are
 *  columns, heads are rows, the bulge only avoids occlusion. The induction
 *  signature is visible and directional: L4H11 (the prev-token head) is
 *  K-elevated into L5H1/L5H5 but not Q-elevated — induction composes through
 *  keys, exactly as the theory says.
 *
 *  deck.gl (WebGL2), camera off, static. Source: comp.json. */

import type { Deck, OrthographicView, PickingInfo } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { type CompBundle, loadComp } from "../../data/interp";
import type { InterpDriver } from "./InterpDriver";

type LayersModule = typeof import("@deck.gl/layers");

const GL = 64; // px — head-row labels
const GR = 24;
const GT = 88; // px — header + layer labels
const GB = 92; // px — chips + collapsed legend pill
const NODE_R = 4.5;

const LOW: [number, number, number] = [64, 66, 96];
const HIGH: [number, number, number] = [245, 195, 59];
const RATIO_HI = 6; // color/width ramp clamps at 6× the random floor (stated)

type CompType = "q" | "k" | "v";

interface Node {
  layer: number;
  head: number;
  position: [number, number];
}

interface Arc {
  path: [number, number][];
  s: number; // exact exported score
  ratio: number; // s / baseline_mean
  i: number; // earlier layer
  h1: number;
  j: number; // later layer
  h2: number;
  idx: number; // flat index into the bundle arrays
}

export class CompositionWebDriver implements InterpDriver {
  readonly animated = false;
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private canvas!: HTMLCanvasElement;
  private tooltip!: HTMLElement;
  private labelRoot!: HTMLElement;
  private chipRoot!: HTMLElement;

  private bundle: CompBundle | null = null;
  private nL = 12;
  private nH = 12;
  private type: CompType = "k";
  // 2× default: the induction arcs L4H11→L5H1/L5H5 sit at 2.8×/2.7× the
  // floor — a 3× default would hide the feature's own headline story
  private thresh = 2;
  private nodes: Node[] = [];
  private arcs: Arc[] = [];
  private isolate: Node | null = null; // click-pinned head
  private hoverArc: Arc | null = null;
  private hoverNode: Node | null = null;

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
    canvas.addEventListener("click", onClick as EventListener);
    this.disposers.push(() => {
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("click", onClick as EventListener);
    });
  }

  async setModel(model: string): Promise<void> {
    if (!this.bundle || this.bundle.meta.model !== model) {
      this.bundle = await loadComp(model);
    }
    this.nL = this.bundle.meta.n_layer;
    this.nH = this.bundle.meta.n_head;
    this.isolate = null;
    this.hoverArc = null;
    this.hoverNode = null;
    this.layout();
    this.buildChips();
    this.deck?.setProps({ viewState: this.viewState() });
    this.pushLayers();
    this.positionLabels();
  }

  // ---- layout ---------------------------------------------------------------
  /** On narrow stages the wrapped chip strip is lifted above the collapsed
   *  legend pill, so the reserved bottom band must grow with it. */
  private gb(): number {
    return this.cssW < 640 ? 150 : GB;
  }
  private colW(): number {
    return (this.cssW - GL - GR) / this.nL;
  }
  private rowH(): number {
    return (this.cssH - GT - this.gb()) / this.nH;
  }
  private nodeXY(layer: number, head: number): [number, number] {
    return [GL + layer * this.colW() + this.colW() / 2, GT + head * this.rowH() + this.rowH() / 2];
  }
  private viewState() {
    return {
      ortho: {
        target: [this.cssW / 2, this.cssH / 2, 0] as [number, number, number],
        zoom: 0,
      },
    };
  }

  private flatIdx(pi: number, h1: number, h2: number): number {
    return pi * this.nH * this.nH + h1 * this.nH + h2;
  }

  /** Horizontal-tangent cubic bezier — pure layout (stated in the legend).
   *  y(t) is a convex combination of the endpoint/control ys, so arcs can
   *  never overshoot the row band into the header or chip strip. A small sag
   *  ∝ horizontal span separates same-row arcs of different spans. */
  private bezier(a: [number, number], b: [number, number]): [number, number][] {
    const dx = b[0] - a[0];
    const sag = Math.min(10, 0.03 * dx);
    const c1: [number, number] = [a[0] + 0.4 * dx, a[1] + sag];
    const c2: [number, number] = [b[0] - 0.4 * dx, b[1] + sag];
    const pts: [number, number][] = [];
    for (let s = 0; s <= 24; s++) {
      const t = s / 24;
      const u = 1 - t;
      pts.push([
        u * u * u * a[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t * t * t * b[0],
        u * u * u * a[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t * t * t * b[1],
      ]);
    }
    return pts;
  }

  private layout(): void {
    const b = this.bundle;
    if (!b) return;
    this.nodes = [];
    for (let L = 0; L < this.nL; L++) {
      for (let h = 0; h < this.nH; h++) {
        this.nodes.push({ layer: L, head: h, position: this.nodeXY(L, h) });
      }
    }
    const arr = b[this.type];
    const bl = b.meta.baseline_mean;
    const cut = this.thresh * bl;
    const arcs: Arc[] = [];
    b.layer_pairs.forEach(([i, j], pi) => {
      for (let h1 = 0; h1 < this.nH; h1++) {
        for (let h2 = 0; h2 < this.nH; h2++) {
          const idx = this.flatIdx(pi, h1, h2);
          const s = arr[idx] ?? 0;
          if (s < cut) continue;
          arcs.push({
            path: this.bezier(this.nodeXY(i, h1), this.nodeXY(j, h2)),
            s,
            ratio: s / bl,
            i,
            h1,
            j,
            h2,
            idx,
          });
        }
      }
    });
    // draw weakest first so the strongest arcs paint last (order = data, not
    // insertion accident)
    arcs.sort((a, c) => a.s - c.s);
    this.arcs = arcs;
  }

  private touches(a: Arc, n: Node): boolean {
    return (a.i === n.layer && a.h1 === n.head) || (a.j === n.layer && a.h2 === n.head);
  }

  /** Nodes are rebuilt on every reflow — compare by coordinates, not identity,
   *  so a pinned isolate survives chip changes. */
  private sameNode(a: Node | null, b: Node | null): boolean {
    return !!a && !!b && a.layer === b.layer && a.head === b.head;
  }

  private arcColor(a: Arc): [number, number, number, number] {
    const focus = this.isolate ?? this.hoverNode;
    const t = Math.min(1, (a.ratio - this.thresh) / (RATIO_HI - this.thresh));
    const r = Math.round(LOW[0] + t * (HIGH[0] - LOW[0]));
    const g = Math.round(LOW[1] + t * (HIGH[1] - LOW[1]));
    const bch = Math.round(LOW[2] + t * (HIGH[2] - LOW[2]));
    let alpha = Math.round(60 + 150 * t);
    if (focus) alpha = this.touches(a, focus) ? Math.min(255, alpha + 70) : 18;
    if (this.hoverArc === a) alpha = 255;
    return [r, g, bch, alpha];
  }

  private pushLayers(): void {
    if (!this.deck || !this.bundle) return;
    const { PathLayer, ScatterplotLayer } = this.layersMod;
    const focus = this.isolate ?? this.hoverNode;
    // participation at the current threshold is data-derived, so it may show
    const active = new Set<string>();
    for (const a of this.arcs) {
      active.add(`${a.i}:${a.h1}`);
      active.add(`${a.j}:${a.h2}`);
    }

    this.deck.setProps({
      layers: [
        new PathLayer<Arc>({
          id: "comp-arcs",
          data: this.arcs,
          getPath: (a) => a.path,
          getColor: (a) => this.arcColor(a),
          getWidth: (a) => 1 + 2 * Math.min(1, (a.ratio - this.thresh) / (RATIO_HI - this.thresh)),
          widthUnits: "pixels",
          pickable: true,
          updateTriggers: {
            getColor: [this.isolate, this.hoverNode, this.hoverArc],
          },
        }),
        new ScatterplotLayer<Node>({
          id: "comp-nodes",
          data: this.nodes,
          getPosition: (n) => [n.position[0], n.position[1], 0],
          getRadius: NODE_R,
          radiusUnits: "pixels",
          getFillColor: (n) => {
            const on = active.has(`${n.layer}:${n.head}`);
            if (this.sameNode(n, focus)) return [255, 255, 255, 240];
            return on ? [166, 173, 200, 200] : [118, 126, 158, 80];
          },
          pickable: true,
          updateTriggers: {
            getFillColor: [this.isolate, this.hoverNode, this.type, this.thresh],
          },
        }),
      ],
    });
  }

  // ---- chips ----------------------------------------------------------------
  private buildChips(): void {
    const narrow = this.cssW < 640;
    this.chipRoot.textContent = "";
    // narrow stages: one short-label row lifted above the collapsed legend
    // pill (which is ~44px tall at bottom 16 + margin)
    this.chipRoot.style.bottom = narrow ? "110px" : "";
    const mk = (label: string, pressed: boolean, onClick: () => void, title = "") => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "interp-neuron-chip";
      btn.textContent = label;
      if (title) btn.title = title;
      btn.setAttribute("aria-pressed", String(pressed));
      if (pressed) btn.classList.add("is-active");
      btn.addEventListener("click", onClick);
      this.chipRoot.appendChild(btn);
    };
    const types: Array<[CompType, string, string]> = [
      ["q", narrow ? "Q" : "Q-comp", "head 1's write feeds head 2's QUERY"],
      ["k", narrow ? "K" : "K-comp", "head 1's write feeds head 2's KEY (induction)"],
      ["v", narrow ? "V" : "V-comp", "head 1's write feeds head 2's VALUE"],
    ];
    for (const [t, label, title] of types) {
      mk(label, this.type === t, () => {
        if (this.type === t) return;
        this.type = t;
        this.reflow();
      }, title);
    }
    for (const th of [2, 3]) {
      mk(narrow ? `≥${th}×` : `≥${th}× floor`, this.thresh === th, () => {
        if (this.thresh === th) return;
        this.thresh = th;
        this.reflow();
      }, "arcs shown only above this multiple of the measured random floor");
    }
    if (this.isolate) {
      mk(`L${this.isolate.layer}H${this.isolate.head} ×`, true, () => {
        this.isolate = null;
        this.reflow();
      });
    }
  }

  private reflow(): void {
    this.hoverArc = null;
    this.hoverNode = null;
    this.tooltip.style.visibility = "hidden";
    this.layout();
    this.buildChips();
    this.pushLayers();
    this.positionLabels();
  }

  // ---- labels ---------------------------------------------------------------
  private positionLabels(): void {
    this.labelRoot.textContent = "";
    const b = this.bundle;
    if (!b) return;
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

    const narrow = this.cssW < 640;
    const names: Record<CompType, string> = { q: "Q-composition", k: "K-composition", v: "V-composition" };
    const total = b.layer_pairs.length * this.nH * this.nH;
    const h1 = cap(
      narrow
        ? `${names[this.type]} · ${this.arcs.length}/${total} pairs ≥ ${this.thresh}× floor`
        : `${names[this.type]} · ${this.arcs.length} of ${total} cross-layer pairs ≥ ${this.thresh}× the random floor`,
    );
    h1.style.color = "rgb(245,195,59)";
    place(h1, narrow ? 12 : GL, GT - 44);
    place(
      cap(
        narrow
          ? `floor ${b.meta.baseline_mean.toFixed(4)}±${b.meta.baseline_std.toFixed(4)} n=${b.meta.baseline_n} · ` +
              "arcs=layout, not data"
          : `floor ${b.meta.baseline_mean.toFixed(4)} ± ${b.meta.baseline_std.toFixed(4)} measured over ` +
              `${b.meta.baseline_n} random factor pairs · color/width ramp ${this.thresh}×→${RATIO_HI}× floor · ` +
              "arc shape is layout, not data",
      ),
      narrow ? 12 : GL,
      GT - 30,
    );

    for (let L = 0; L < this.nL; L++) {
      const el = cap(`L${L}`);
      place(el, GL + L * this.colW() + this.colW() / 2 - 8, GT - 14);
    }
    // thin row labels when rows are tighter than the 10.5px line height
    const step = this.rowH() >= 13 ? 1 : this.rowH() >= 7 ? 2 : 4;
    for (let h = 0; h < this.nH; h += step) {
      place(cap(`H${h}`), 8, GT + h * this.rowH() + this.rowH() / 2 - 8);
    }
  }

  // ---- interaction ----------------------------------------------------------
  private pick(e: PointerEvent): { node?: Node; arc?: Arc } {
    if (!this.deck) return {};
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const info = this.deck.pickObject({ x, y, radius: 4, layerIds: ["comp-nodes", "comp-arcs"] }) as
      | PickingInfo
      | null;
    if (!info?.object) return {};
    if (info.layer?.id === "comp-nodes") return { node: info.object as Node };
    return { arc: info.object as Arc };
  }

  private onPointerMove(e: PointerEvent): void {
    const b = this.bundle;
    if (!b) return;
    const { node, arc } = this.pick(e);
    if (node !== this.hoverNode || arc !== this.hoverArc) {
      this.hoverNode = node ?? null;
      this.hoverArc = arc ?? null;
      this.pushLayers();
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
    if (arc) {
      add("point-tooltip-label", `L${arc.i}H${arc.h1} → L${arc.j}H${arc.h2}`);
      add("point-tooltip-conf", `${this.type.toUpperCase()}-comp ${arc.s.toFixed(4)} · ${arc.ratio.toFixed(1)}× random floor`);
      const pi = b.layer_pairs.findIndex(([i, j]) => i === arc.i && j === arc.j);
      const others = (["q", "k", "v"] as CompType[]).filter((t) => t !== this.type);
      add(
        "point-tooltip-conf",
        others.map((t) => `${t.toUpperCase()} ${(b[t][this.flatIdx(pi, arc.h1, arc.h2)] ?? 0).toFixed(4)}`).join(" · "),
      );
    } else if (node) {
      add("point-tooltip-label", `L${node.layer} · head ${node.head}`);
      const { best, count } = this.strongest(node);
      add("point-tooltip-conf", `${count} arc${count === 1 ? "" : "s"} at ≥${this.thresh}× floor (${this.type.toUpperCase()})`);
      if (best) {
        const dir = best.i === node.layer && best.h1 === node.head ? `→ L${best.j}H${best.h2}` : `← L${best.i}H${best.h1}`;
        add("point-tooltip-conf", `strongest ${dir} · ${best.s.toFixed(4)} (${best.ratio.toFixed(1)}×)`);
      }
      add("point-tooltip-conf", this.sameNode(this.isolate, node) ? "click to release" : "click to isolate");
    } else {
      this.tooltip.style.visibility = "hidden";
      this.canvas.style.cursor = "";
      return;
    }
    this.tooltip.style.visibility = "visible";
    const px = Math.min(x + 14, this.cssW - 300);
    const py = Math.min(y + 14, this.cssH - 100);
    this.tooltip.style.transform = `translate(${px.toFixed(1)}px, ${py.toFixed(1)}px)`;
    this.canvas.style.cursor = "pointer";
  }

  private strongest(n: Node): { best: Arc | null; count: number } {
    let best: Arc | null = null;
    let count = 0;
    for (const a of this.arcs) {
      if (!this.touches(a, n)) continue;
      count++;
      if (!best || a.s > best.s) best = a;
    }
    return { best, count };
  }

  private onClick(e: PointerEvent): void {
    const { node } = this.pick(e);
    if (node) {
      this.isolate = this.sameNode(this.isolate, node) ? null : node;
    } else if (this.isolate) {
      this.isolate = null;
    } else {
      return;
    }
    this.buildChips();
    this.pushLayers();
  }

  private onLeave(): void {
    if (this.hoverArc || this.hoverNode) {
      this.hoverArc = null;
      this.hoverNode = null;
      this.pushLayers();
    }
    this.tooltip.style.visibility = "hidden";
    this.canvas.style.cursor = "";
  }

  frame(_dt: number, _t: number): void {
    // static — arcs only change on chip/isolate changes
  }

  resize(width: number, height: number, dpr: number): void {
    this.cssW = width;
    this.cssH = height;
    this.dpr = dpr;
    this.layout();
    this.hoverArc = null;
    this.hoverNode = null;
    this.deck?.setProps({
      width,
      height,
      useDevicePixels: Math.min(dpr, 2),
      viewState: this.viewState(),
    });
    this.pushLayers();
    if (this.bundle) this.buildChips(); // chip labels/offset depend on width
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
