/** The hierarchical-network view — the video's third legend radio: a radial
 *  dendrogram of the cluster similarity structure. The tree is built
 *  client-side by single-linkage agglomeration over the exported
 *  `cluster_edges` (Kruskal over descending weight + union-find), so every
 *  join is honest: "these subtrees merge at similarity w" in the 10-D cluster
 *  space the export stamps (metric/space shown in the tooltip, never screen
 *  distance). Radius is display-only (normalized by the strongest merge);
 *  tooltips always carry the raw weight.
 *
 *  Rendering is deck.gl (lazy-imported so the atlas bundle stays lean):
 *  PathLayer elbows (arc along the parent radius, then radial to the child)
 *  + ScatterplotLayers for leaves and join points, with deck's built-in
 *  picking. Per the nebulai-viz laws deck owns no interaction state: the
 *  controller is off, the viewport is derived from stage size, and our own
 *  pointer handlers call pickObject and write to the shared store. */

import type { Deck, OrthographicView } from "@deck.gl/core";
import type { GpuTier } from "../../app/capabilities";
import { appStore, type Selection } from "../../app/store";
import type { Dataset } from "../../data/loader";
import { rampColor } from "../../styles/tokens";
import type { SceneDriver } from "../SceneDriver";
import { clusterColor } from "../layers/PointsLayer";

/** leaves sit on this world radius; the frustum is sized around it */
const LEAF_R = 1;
/** frustum half-extent along the shorter screen axis (matches ChordDriver) */
const HALF_MIN = 1.42;
const LABEL_MAX_CHARS = 26;

interface HierNode {
  id: number;
  /** leaf: index into columns.clusters; -1 for join nodes */
  clusterIdx: number;
  /** leaf: exported cluster id; -1 for join nodes */
  clusterId: number;
  /** merge similarity (raw, from the export); leaves = 1, synthetic root = 0 */
  weight: number;
  children: number[];
  parent: number;
  angle: number;
  x: number;
  y: number;
  leafStart: number;
  leafEnd: number;
}

interface HierLink {
  /** [x, y][] — arc along the parent radius, then radial to the child */
  path: [number, number][];
  childId: number;
  /** raw merge weight of the parent join */
  w: number;
}

type LayersModule = typeof import("@deck.gl/layers");

export class HierarchyDriver implements SceneDriver {
  private deck: Deck<OrthographicView[]> | null = null;
  private layersMod!: LayersModule;
  private makeView!: () => OrthographicView;
  private canvas!: HTMLCanvasElement;

  private nodes: HierNode[] = [];
  private links: HierLink[] = [];
  private leaves: number[] = []; // node ids, in DFS rim order
  private leafByClusterId = new Map<number, number>(); // clusterId → node id
  private wMax = 1;
  private ds: Dataset | null = null;

  /** node ids on the focused leaf's path to the root; null = no focus */
  private focusPath: Set<number> | null = null;
  private focusVersion = 0;

  private labelRoot!: HTMLElement;
  private labels: Array<{ el: HTMLElement; nodeId: number }> = [];
  private tooltip!: HTMLElement;

  private cssW = 1;
  private cssH = 1;
  private dpr = 1;
  /** world units per CSS px */
  private wpp = 0.01;
  private disposers: Array<() => void> = [];

  async init(canvas: HTMLCanvasElement, _tier: GpuTier): Promise<void> {
    this.canvas = canvas;
    // deck is ~300KB gz — lazy so the atlas bundle stays under budget. deck is
    // WebGL2-only, which is fine here: it rides the `webgl` rung untouched and
    // the hierarchy visuals deliberately don't depend on post FX.
    const [core, layers] = await Promise.all([import("@deck.gl/core"), import("@deck.gl/layers")]);
    this.layersMod = layers;
    this.makeView = () => new core.OrthographicView({ id: "ortho", flipY: false });

    this.deck = new core.Deck({
      canvas,
      views: [this.makeView()],
      viewState: this.viewState(),
      controller: false, // camera is app state, deck never owns interaction
      useDevicePixels: Math.min(this.dpr, 2),
      layers: [],
      width: this.cssW,
      height: this.cssH,
    }) as unknown as Deck<OrthographicView[]>;

    const overlay = document.getElementById("overlay-html")!;
    this.labelRoot = document.createElement("div");
    this.labelRoot.className = "hier-labels";
    overlay.appendChild(this.labelRoot);

    this.tooltip = document.createElement("div");
    this.tooltip.className = "point-tooltip hier-tooltip";
    this.tooltip.style.visibility = "hidden";
    overlay.appendChild(this.tooltip);

    const onMove = (e: PointerEvent) => this.onPointerMove(e);
    const onLeave = () => this.setHover(null);
    const onClick = (e: PointerEvent) => this.onClick(e);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    canvas.addEventListener("click", onClick as EventListener);
    this.disposers.push(() => {
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("click", onClick as EventListener);
    });

    // highlight follows the shared store, so Esc/mode-switch clears propagate
    let prevHover = appStore.getState().hover;
    let prevSel = appStore.getState().selection;
    this.disposers.push(
      appStore.subscribe((s) => {
        if (s.hover !== prevHover || s.selection !== prevSel) {
          prevHover = s.hover;
          prevSel = s.selection;
          this.applyHighlight();
        }
      }),
    );
  }

  // ── tree construction: single-linkage over cluster_edges ────────────────

  setDataset(ds: Dataset): void {
    this.ds = ds;
    const { clusters, edges } = ds.columns;
    this.nodes = [];
    this.links = [];
    this.leaves = [];
    this.leafByClusterId = new Map();

    // leaves — one per exported cluster
    for (let i = 0; i < clusters.length; i++) {
      const c = clusters[i]!;
      this.nodes.push({
        id: i,
        clusterIdx: i,
        clusterId: c.id,
        weight: 1,
        children: [],
        parent: -1,
        angle: 0,
        x: 0,
        y: 0,
        leafStart: 0,
        leafEnd: 0,
      });
      this.leafByClusterId.set(c.id, i);
    }

    // Kruskal on descending similarity: the first edge that bridges two
    // components is, by construction, their single-linkage merge weight
    const dsu: number[] = this.nodes.map((n) => n.id);
    const find = (i: number): number => {
      while (dsu[i] !== i) {
        dsu[i] = dsu[dsu[i]!]!;
        i = dsu[i]!;
      }
      return i;
    };
    const join = (a: number, b: number, w: number): void => {
      const nid = this.nodes.length;
      this.nodes.push({
        id: nid,
        clusterIdx: -1,
        clusterId: -1,
        weight: w,
        children: [a, b],
        parent: -1,
        angle: 0,
        x: 0,
        y: 0,
        leafStart: 0,
        leafEnd: 0,
      });
      this.nodes[a]!.parent = nid;
      this.nodes[b]!.parent = nid;
      dsu.push(nid);
      dsu[a] = nid;
      dsu[b] = nid;
    };

    const ce = edges?.clusterEdges;
    if (ce) {
      const triples: Array<[number, number, number]> = [];
      for (let i = 0; i < ce.length; i += 3) {
        const a = this.leafByClusterId.get(ce[i]!);
        const b = this.leafByClusterId.get(ce[i + 1]!);
        if (a === undefined || b === undefined) continue;
        triples.push([a, b, ce[i + 2]!]);
      }
      triples.sort((p, q) => q[2] - p[2]);
      for (const [a, b, w] of triples) {
        const ra = find(a);
        const rb = find(b);
        if (ra !== rb) join(ra, rb, w);
      }
    }

    // the top-k edge graph may be a forest — gather roots under one synthetic
    // root at weight 0 ("no exported similarity connects these components")
    const rootIds = new Set<number>();
    for (const n of this.nodes) if (n.parent === -1) rootIds.add(find(n.id));
    let root: number;
    if (rootIds.size === 1) {
      root = [...rootIds][0]!;
    } else {
      root = this.nodes.length;
      this.nodes.push({
        id: root,
        clusterIdx: -1,
        clusterId: -1,
        weight: 0,
        children: [...rootIds],
        parent: -1,
        angle: 0,
        x: 0,
        y: 0,
        leafStart: 0,
        leafEnd: 0,
      });
      for (const r of rootIds) this.nodes[r]!.parent = root;
    }

    this.layout(root);
    this.focusPath = null;
    this.focusVersion++;
    this.pushLayers();
    this.rebuildLabels();
  }

  /** Radial dendrogram: DFS leaf order → rim angles; join radius = weight
   *  normalized by the strongest merge (display-only — tooltips carry raw w);
   *  links are classic elbows (arc at the parent radius, radial drop). */
  private layout(root: number): void {
    this.leaves = [];
    const assign = (id: number): void => {
      const nd = this.nodes[id]!;
      nd.leafStart = this.leaves.length;
      if (nd.children.length === 0) this.leaves.push(id);
      for (const c of nd.children) assign(c);
      nd.leafEnd = this.leaves.length;
    };
    assign(root);

    const n = Math.max(this.leaves.length, 1);
    const leafAngle = (k: number) => -Math.PI / 2 + ((k + 0.5) / n) * Math.PI * 2;

    this.wMax = 1e-6;
    for (const nd of this.nodes)
      if (nd.children.length > 0 && nd.id !== root) this.wMax = Math.max(this.wMax, nd.weight);

    for (const nd of this.nodes) {
      if (nd.children.length === 0) {
        nd.angle = leafAngle(nd.leafStart);
      } else {
        // DFS makes every subtree's leaf range contiguous, so the midpoint
        // angle never wraps
        nd.angle = leafAngle((nd.leafStart + nd.leafEnd - 1) / 2);
      }
      const r =
        nd.children.length === 0
          ? LEAF_R
          : nd.id === root
            ? 0
            : Math.min(nd.weight / this.wMax, 1) * 0.9 * LEAF_R;
      nd.x = Math.cos(nd.angle) * r;
      nd.y = Math.sin(nd.angle) * r;
    }

    this.links = [];
    for (const nd of this.nodes) {
      if (nd.children.length === 0) continue;
      const rp = Math.hypot(nd.x, nd.y);
      for (const cid of nd.children) {
        const c = this.nodes[cid]!;
        const path: [number, number][] = [];
        if (rp < 0.02) {
          path.push([nd.x, nd.y]);
        } else {
          // arc along the parent radius from parent angle to child angle
          const steps = Math.max(2, Math.ceil(Math.abs(c.angle - nd.angle) / 0.05));
          for (let s = 0; s <= steps; s++) {
            const a = nd.angle + ((c.angle - nd.angle) * s) / steps;
            path.push([Math.cos(a) * rp, Math.sin(a) * rp]);
          }
        }
        path.push([c.x, c.y]);
        this.links.push({ path, childId: cid, w: nd.weight });
      }
    }
  }

  // ── deck layers ──────────────────────────────────────────────────────────

  private pushLayers(): void {
    if (!this.deck) return;
    const { PathLayer, ScatterplotLayer } = this.layersMod;
    const focus = this.focusPath;
    const wMax = this.wMax;
    const { clusters } = this.ds?.columns ?? { clusters: [] };
    const maxSize = Math.max(...clusters.map((c) => c.size), 1);

    const linkColor = (d: HierLink): [number, number, number, number] => {
      const wn = Math.min(d.w / wMax, 1);
      const [r, g, b] = rampColor(wn);
      const lit = !focus || focus.has(d.childId);
      const base = 46 + 150 * wn * wn;
      return [r * 255, g * 255, b * 255, lit ? (focus ? 235 : base) : 14];
    };

    const leafColor = (id: number): [number, number, number, number] => {
      const nd = this.nodes[id]!;
      const [r, g, b] = clusterColor(nd.clusterId);
      const lit = !focus || focus.has(id);
      return [r * 255, g * 255, b * 255, lit ? 235 : 50];
    };

    const joints = this.nodes.filter((nd) => nd.children.length > 0);

    this.deck.setProps({
      layers: [
        new PathLayer<HierLink>({
          id: "hier-links",
          data: this.links,
          getPath: (d) => d.path,
          getColor: linkColor,
          getWidth: (d) => 0.8 + 2.6 * Math.min(d.w / wMax, 1) ** 2,
          widthUnits: "pixels",
          jointRounded: true,
          pickable: false,
          updateTriggers: { getColor: this.focusVersion },
        }),
        new ScatterplotLayer<HierNode>({
          id: "hier-joints",
          data: joints,
          getPosition: (d) => [d.x, d.y],
          getRadius: 2.2,
          radiusUnits: "pixels",
          getFillColor: (d) =>
            !focus || focus.has(d.id) ? [157, 143, 166, focus ? 220 : 130] : [157, 143, 166, 26],
          pickable: true,
          updateTriggers: { getFillColor: this.focusVersion },
        }),
        new ScatterplotLayer<number>({
          id: "hier-leaves",
          data: this.leaves,
          getPosition: (id) => [this.nodes[id]!.x, this.nodes[id]!.y],
          getRadius: (id) => {
            const c = clusters[this.nodes[id]!.clusterIdx];
            return 2.5 + 5.5 * Math.sqrt((c?.size ?? 1) / maxSize);
          },
          radiusUnits: "pixels",
          getFillColor: leafColor,
          pickable: true,
          updateTriggers: { getFillColor: this.focusVersion },
        }),
      ],
    });
  }

  // ── rotated radial labels (HTML-first law, same recipe as ChordDriver) ──

  private rebuildLabels(): void {
    if (!this.labelRoot) return;
    this.labelRoot.textContent = "";
    this.labels = [];
    if (!this.ds || this.leaves.length === 0) return;

    const rimPx = LEAF_R / Math.max(this.wpp, 1e-6);
    const minGap = 13 / Math.max(rimPx, 1);
    const bySize = this.leaves
      .map((id) => ({ id, size: this.ds!.columns.clusters[this.nodes[id]!.clusterIdx]!.size }))
      .sort((a, b) => b.size - a.size);

    const kept: number[] = [];
    for (const { id } of bySize) {
      const ang = this.nodes[id]!.angle;
      let ok = true;
      for (const k of kept) {
        let d = Math.abs(ang - this.nodes[k]!.angle);
        d = Math.min(d, Math.PI * 2 - d);
        if (d < minGap) {
          ok = false;
          break;
        }
      }
      if (!ok) continue;
      kept.push(id);
      const el = document.createElement("div");
      el.className = "hier-label";
      const title = this.ds.columns.clusters[this.nodes[id]!.clusterIdx]!.title;
      el.textContent =
        title.length > LABEL_MAX_CHARS ? `${title.slice(0, LABEL_MAX_CHARS - 1)}…` : title;
      this.labelRoot.appendChild(el);
      this.labels.push({ el, nodeId: id });
    }
    this.positionLabels();
  }

  private positionLabels(): void {
    const cx = this.cssW / 2;
    const cy = this.cssH / 2;
    const rimPx = LEAF_R / Math.max(this.wpp, 1e-6);
    for (const { el, nodeId } of this.labels) {
      const nd = this.nodes[nodeId]!;
      const rad = rimPx + 14;
      const sx = cx + Math.cos(nd.angle) * rad;
      const sy = cy - Math.sin(nd.angle) * rad;
      const flip = Math.cos(nd.angle) < 0;
      const deg = (-nd.angle * 180) / Math.PI + (flip ? 180 : 0);
      el.style.transform =
        `translate(${sx.toFixed(1)}px, ${sy.toFixed(1)}px) rotate(${deg.toFixed(2)}deg) ` +
        `translate(${flip ? "-100%" : "0"}, -50%)`;
    }
  }

  // ── interaction: deck pickObject behind our own handlers ────────────────

  private hoverNode: number | null = null;

  private pickNode(x: number, y: number): HierNode | null {
    if (!this.deck) return null;
    const info = this.deck.pickObject({ x, y, radius: 7 });
    if (!info?.layer) return null;
    if (info.layer.id === "hier-leaves") return this.nodes[info.object as number] ?? null;
    if (info.layer.id === "hier-joints") return info.object as HierNode;
    return null;
  }

  private onPointerMove(e: PointerEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    const nd = this.pickNode(e.clientX - rect.left, e.clientY - rect.top);
    this.setHover(nd?.id ?? null);
    if (nd) {
      const edges = this.ds?.columns.edges;
      const provenance = edges ? ` (${edges.metric.replace(/_/g, " ")} in ${edges.space})` : "";
      this.tooltip.innerHTML = "";
      const line1 = document.createElement("div");
      line1.className = "point-tooltip-label";
      const line2 = document.createElement("div");
      line2.className = "point-tooltip-conf";
      if (nd.clusterIdx >= 0) {
        const c = this.ds!.columns.clusters[nd.clusterIdx]!;
        const pw = nd.parent >= 0 ? this.nodes[nd.parent]!.weight : 0;
        line1.textContent = c.title;
        line2.textContent = `${c.size.toLocaleString("en-US")} tokens · joins at ${pw.toFixed(2)}${provenance}`;
      } else {
        const beneath = nd.leafEnd - nd.leafStart;
        line1.textContent = `join · ${beneath} clusters beneath`;
        line2.textContent =
          nd.weight > 0
            ? `merge similarity ${nd.weight.toFixed(2)}${provenance}`
            : "root — components unlinked in the exported top-k edges";
      }
      this.tooltip.append(line1, line2);
      this.tooltip.style.visibility = "visible";
      const px = Math.min(e.clientX - rect.left + 14, this.cssW - 260);
      const py = Math.min(e.clientY - rect.top + 14, this.cssH - 56);
      this.tooltip.style.transform = `translate(${px.toFixed(1)}px, ${py.toFixed(1)}px)`;
      this.canvas.style.cursor = "pointer";
    } else {
      this.tooltip.style.visibility = "hidden";
      this.canvas.style.cursor = "";
    }
  }

  private setHover(nodeId: number | null): void {
    if (nodeId === this.hoverNode) return;
    this.hoverNode = nodeId;
    if (nodeId === null) this.tooltip.style.visibility = "hidden";
    const nd = nodeId === null ? null : this.nodes[nodeId]!;
    // join nodes highlight locally but aren't clusters — no store hover for them
    appStore
      .getState()
      .setHover(nd && nd.clusterIdx >= 0 ? { kind: "cluster", id: nd.clusterId } : null);
    if (nd && nd.clusterIdx < 0) {
      this.focusSubtree(nd.id);
    }
  }

  private onClick(e: PointerEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    const nd = this.pickNode(e.clientX - rect.left, e.clientY - rect.top);
    appStore
      .getState()
      .setSelection(nd && nd.clusterIdx >= 0 ? { kind: "cluster", id: nd.clusterId } : null);
  }

  /** Focus for a leaf = its ancestor chain (path to root); everything else
   *  recedes. Driven from the store so Esc/mode switches clear it. */
  private applyHighlight(): void {
    const { hover, selection } = appStore.getState();
    const sel =
      selection?.kind === "cluster" ? selection : hover?.kind === "cluster" ? hover : null;
    const leaf = sel ? this.leafByClusterId.get(sel.id) : undefined;
    if (leaf === undefined) {
      if (this.focusPath === null) return;
      this.focusPath = null;
    } else {
      this.focusPath = new Set<number>();
      for (let id: number = leaf; id !== -1; id = this.nodes[id]!.parent) this.focusPath.add(id);
    }
    this.focusVersion++;
    this.pushLayers();
    for (const { el, nodeId } of this.labels)
      el.classList.toggle("is-dim", this.focusPath !== null && !this.focusPath.has(nodeId));
  }

  /** Hovering a join node lights its whole subtree (not a store concern —
   *  join nodes aren't clusters, so this is view-local). */
  private focusSubtree(id: number): void {
    const nd = this.nodes[id]!;
    this.focusPath = new Set<number>();
    const stack = [id];
    while (stack.length) {
      const cur = this.nodes[stack.pop()!]!;
      this.focusPath.add(cur.id);
      for (const c of cur.children) stack.push(c);
    }
    for (let p = nd.parent; p !== -1; p = this.nodes[p]!.parent) this.focusPath.add(p);
    this.focusVersion++;
    this.pushLayers();
    for (const { el, nodeId } of this.labels)
      el.classList.toggle("is-dim", !this.focusPath.has(nodeId));
  }

  // ── SceneDriver plumbing ─────────────────────────────────────────────────

  /** Keyed by view id ("ortho") — deck's typed viewState form for view arrays. */
  private viewState() {
    const short = Math.max(Math.min(this.cssW, this.cssH), 1);
    return {
      ortho: {
        target: [0, 0, 0] as [number, number, number],
        zoom: Math.log2(short / (2 * HALF_MIN)),
      },
    };
  }

  frame(_dt: number, _t: number): void {
    // deck renders on its own demand-driven loop; nothing time-based here
    // (no post FX on the deck rung, and goldens need a static scene anyway)
  }

  resize(width: number, height: number, dpr: number): void {
    this.cssW = width;
    this.cssH = height;
    this.dpr = dpr;
    const short = Math.max(Math.min(width, height), 1);
    this.wpp = (HALF_MIN * 2) / short;
    this.deck?.setProps({
      width,
      height,
      useDevicePixels: Math.min(dpr, 2),
      viewState: this.viewState(),
    });
    this.rebuildLabels();
  }

  pick(x: number, y: number): Selection | null {
    const nd = this.pickNode(x, y);
    return nd && nd.clusterIdx >= 0 ? { kind: "cluster", id: nd.clusterId } : null;
  }

  snapshotForTransition(): HTMLCanvasElement | null {
    return null;
  }

  dispose(): void {
    for (const d of this.disposers) d();
    this.disposers = [];
    this.labelRoot?.remove();
    this.tooltip?.remove();
    this.deck?.finalize();
    this.deck = null;
  }
}
