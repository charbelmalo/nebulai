/** The chord view — the video's radial "hairball" scene (frames f04/f08):
 *  clusters as glowing rim nodes on a circle, similarity edges as thin static
 *  gradient bezier ribbons through the interior (weight maps to opacity and a
 *  hairline-to-thin width — no dashes, no motion, matching the atlas beams'
 *  minimal register), rotated radial HTML labels. three-TSL so it rides both
 *  the webgpu and forceWebGL rungs.
 *
 *  Honesty: rim order comes from each cluster's atlas 2-D centroid angle, so
 *  ring neighborhoods echo the map — but chord weights are similarity in the
 *  10-D cluster space (metric stamped in the export, same as the atlas beams),
 *  never screen distance. Layout at ~200 rim nodes is CPU-trivial; hover is a
 *  radial/angular test, not a per-point scan.
 *
 *  WebGPU budget: maxVertexBuffers is 8 and PlaneGeometry already binds
 *  position+normal+uv, so per-instance data is packed into two vec4
 *  attributes (seg, meta) instead of six scalars/vec2s. TSL tree-shakes
 *  unreferenced attributes, so an over-budget layout only fails once the
 *  full node graph is wired — and it fails silently (async pipeline). */

import * as THREE from "three/webgpu";
import {
  float,
  instancedDynamicBufferAttribute,
  mix,
  positionGeometry,
  texture,
  uniform,
  uv,
  vec2,
  vec3,
} from "three/tsl";
import type { GpuTier } from "../../app/capabilities";
import { appStore, type Selection } from "../../app/store";
import type { Dataset } from "../../data/loader";
import { rampTextureData } from "../../styles/tokens";
import type { SceneDriver } from "../SceneDriver";
import { clusterColor } from "../layers/PointsLayer";

const MAX_NODES = 512;
const MAX_CHORDS = 2048;
/** world radius of the rim circle; camera frustum is sized around it */
const RIM_R = 1;
/** frustum half-extent along the shorter screen axis */
const HALF_MIN = 1.42;
const LABEL_MAX_CHARS = 26;

interface RimNode {
  clusterIdx: number; // index into columns.clusters
  clusterId: number; // exported cluster id (edge endpoints reference this)
  angle: number; // final rim angle, radians
  x: number;
  y: number;
  sizePx: number; // sprite radius in CSS px
  degree: number; // chord count touching this node
}

export class ChordDriver implements SceneDriver {
  private renderer!: THREE.WebGPURenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
  private canvas!: HTMLCanvasElement;

  /** world units per CSS px — set on resize, feeds px-true widths */
  private readonly uWpp = uniform(0.01);

  // chords — packed to respect the 8-vertex-buffer WebGPU limit
  private chordMesh!: THREE.InstancedMesh;
  /** vec4 per chord: start.xy, end.xy */
  private chordSeg!: THREE.InstancedBufferAttribute;
  /** vec4 per chord: weight, rampA, rampB, active */
  private chordMeta!: THREE.InstancedBufferAttribute;
  /** per-chord rim indices, for highlight recompute */
  private chordEnds: Array<[number, number]> = [];

  // rim nodes
  private nodeMesh!: THREE.InstancedMesh;
  private nodePos!: THREE.InstancedBufferAttribute;
  private nodeColor!: THREE.InstancedBufferAttribute;
  private nodeSize!: THREE.InstancedBufferAttribute;
  private nodeActive!: THREE.InstancedBufferAttribute;

  private rim: RimNode[] = [];
  /** clusterId → rim index */
  private rimByClusterId = new Map<number, number>();
  private ds: Dataset | null = null;

  // rotated radial HTML labels
  private labelRoot!: HTMLElement;
  private labels: Array<{ el: HTMLElement; rimIdx: number }> = [];
  private tooltip!: HTMLElement;

  private cssW = 1;
  private cssH = 1;
  /** world units per CSS px, plain number twin of uWpp for CPU math */
  private wpp = 0.01;
  private disposers: Array<() => void> = [];
  private rampTex!: THREE.DataTexture;
  private materials: THREE.Material[] = [];

  async init(canvas: HTMLCanvasElement, tier: GpuTier): Promise<void> {
    this.canvas = canvas;
    this.renderer = new THREE.WebGPURenderer({
      canvas,
      antialias: true,
      alpha: true,
      forceWebGL: tier === "webgl",
    });
    await this.renderer.init();
    this.renderer.setClearColor(0x000000, 0);
    this.camera.position.z = 10;

    this.rampTex = new THREE.DataTexture(rampTextureData(), 256, 1, THREE.RGBAFormat);
    this.rampTex.needsUpdate = true;

    this.buildChords();
    this.buildNodes();

    const overlay = document.getElementById("overlay-html")!;
    this.labelRoot = document.createElement("div");
    this.labelRoot.className = "chord-labels";
    overlay.appendChild(this.labelRoot);

    this.tooltip = document.createElement("div");
    this.tooltip.className = "point-tooltip chord-tooltip";
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

  /** Bezier ribbon chords, adapted from BeamsLayer: per-instance endpoints on
   *  the rim, a control point pulled toward the center (long chords sweep
   *  deep, short ones stay shallow), the shared ramp as an endpoint-to-
   *  endpoint hue gradient, and weight mapped straight to opacity — static
   *  ribbons, nothing animated. */
  private buildChords(): void {
    this.chordSeg = new THREE.InstancedBufferAttribute(new Float32Array(MAX_CHORDS * 4), 4);
    this.chordMeta = new THREE.InstancedBufferAttribute(new Float32Array(MAX_CHORDS * 4), 4);

    const aSeg = instancedDynamicBufferAttribute<"vec4">(this.chordSeg, "vec4");
    const aMeta = instancedDynamicBufferAttribute<"vec4">(this.chordMeta, "vec4");
    const aStart = aSeg.xy;
    const aEnd = aSeg.zw;
    const aWeight = aMeta.x;
    const aRampA = aMeta.y;
    const aRampB = aMeta.z;
    const aActive = aMeta.w;

    const material = new THREE.MeshBasicNodeMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });

    const t = uv().x;
    const across = positionGeometry.y; // -0.5 … 0.5 on the plane strip
    const chordVec = aEnd.sub(aStart);
    const len = chordVec.length().max(1e-6);
    // quadratic bezier: control = midpoint pulled toward the circle center;
    // pull grows with chord length so cross-circle links dive through the
    // middle while neighbor links hug the rim (the video's interior weave)
    const sag = len.div(RIM_R * 2).clamp(0, 1).mul(0.85);
    const ctrl = aStart.add(aEnd).mul(0.5).mul(float(1).sub(sag));
    const p0 = mix(aStart, ctrl, t);
    const p1 = mix(ctrl, aEnd, t);
    const p = mix(p0, p1, t);
    // bezier tangent → ribbon normal
    const tangent = p1.sub(p0).normalize();
    const perp = vec2(tangent.y.negate(), tangent.x);
    // thin register: strong links a touch wider, weak ones hairlines
    const widthPx = mix(float(1.0), float(2.5), aWeight);
    const pos = p.add(perp.mul(across.mul(widthPx).mul(this.uWpp)));
    material.positionNode = vec3(pos, 0.02);

    // hue travels the shared ramp from cluster A's color slot to cluster B's
    material.colorNode = texture(this.rampTex, vec2(mix(aRampA, aRampB, t), 0.5)).rgb;

    const edgeFade = across.abs().mul(2).smoothstep(0.25, 1).oneMinus();
    const endFade = t.smoothstep(0, 0.04).mul(t.oneMinus().smoothstep(0, 0.04));
    // weight IS the opacity — one visual channel, honestly mapped, static
    const alpha = mix(float(0.16), float(0.62), aWeight);
    const focus = mix(float(0.05), float(1), aActive); // dim when unfocused
    material.opacityNode = edgeFade.mul(endFade).mul(alpha).mul(focus);

    this.materials.push(material);
    this.chordMesh = new THREE.InstancedMesh(new THREE.PlaneGeometry(1, 1, 32, 1), material, MAX_CHORDS);
    this.chordMesh.count = 0;
    this.chordMesh.frustumCulled = false;
    this.chordMesh.renderOrder = 1;
    this.scene.add(this.chordMesh);
  }

  /** Rim node sprites: SDF discs with a soft halo, cluster-identity colors
   *  (same golden-ratio ramp scramble as the atlas points). */
  private buildNodes(): void {
    this.nodePos = new THREE.InstancedBufferAttribute(new Float32Array(MAX_NODES * 2), 2);
    this.nodeColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_NODES * 3), 3);
    this.nodeSize = new THREE.InstancedBufferAttribute(new Float32Array(MAX_NODES), 1);
    this.nodeActive = new THREE.InstancedBufferAttribute(new Float32Array(MAX_NODES).fill(1), 1);

    const aPos = instancedDynamicBufferAttribute<"vec2">(this.nodePos, "vec2");
    const aColor = instancedDynamicBufferAttribute<"vec3">(this.nodeColor, "vec3");
    const aSize = instancedDynamicBufferAttribute<"float">(this.nodeSize, "float");
    const aActive = instancedDynamicBufferAttribute<"float">(this.nodeActive, "float");

    const material = new THREE.MeshBasicNodeMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
    });

    const grow = mix(float(1), float(1.25), aActive.smoothstep(0.99, 1)); // hovered node swells
    const offset = positionGeometry.xy.mul(aSize).mul(grow).mul(this.uWpp);
    material.positionNode = vec3(aPos.add(offset), 0.1);
    material.colorNode = aColor;

    const d = uv().sub(0.5).length().mul(2);
    const core = d.smoothstep(0.42, 0.6).oneMinus();
    const halo = d.smoothstep(0.1, 1).oneMinus().mul(0.3);
    material.opacityNode = core.add(halo).mul(mix(float(0.22), float(1), aActive));

    this.materials.push(material);
    this.nodeMesh = new THREE.InstancedMesh(new THREE.PlaneGeometry(2, 2), material, MAX_NODES);
    this.nodeMesh.count = 0;
    this.nodeMesh.frustumCulled = false;
    this.nodeMesh.renderOrder = 2;
    this.scene.add(this.nodeMesh);
  }

  setDataset(ds: Dataset): void {
    this.ds = ds;
    const { clusters, clusterId, pos2 } = ds.columns;

    // atlas 2-D centroids → rim order by angle around the map center
    const acc = new Map<number, { x: number; y: number; n: number }>();
    for (let i = 0; i < ds.columns.count; i++) {
      const cid = clusterId[i]!;
      if (cid < 0) continue;
      let a = acc.get(cid);
      if (!a) acc.set(cid, (a = { x: 0, y: 0, n: 0 }));
      a.x += pos2[i * 2]!;
      a.y += pos2[i * 2 + 1]!;
      a.n++;
    }
    let mx = 0;
    let my = 0;
    for (const a of acc.values()) {
      mx += a.x / a.n;
      my += a.y / a.n;
    }
    mx /= Math.max(acc.size, 1);
    my /= Math.max(acc.size, 1);

    const order = clusters
      .map((c, idx) => {
        const a = acc.get(c.id);
        const angle = a ? Math.atan2(a.y / a.n - my, a.x / a.n - mx) : 0;
        return { idx, id: c.id, size: c.size, angle };
      })
      .sort((a, b) => a.angle - b.angle)
      .slice(0, MAX_NODES);

    const n = order.length;
    const maxSize = Math.max(...order.map((o) => o.size), 1);
    this.rim = order.map((o, k) => {
      const angle = -Math.PI / 2 + (k / n) * Math.PI * 2; // rank-spaced, starts at top
      return {
        clusterIdx: o.idx,
        clusterId: o.id,
        angle,
        x: Math.cos(angle) * RIM_R,
        y: Math.sin(angle) * RIM_R,
        sizePx: 3 + 7.5 * Math.sqrt(o.size / maxSize),
        degree: 0,
      };
    });
    this.rimByClusterId = new Map(this.rim.map((r, i) => [r.clusterId, i]));

    // chords from cluster_edges (10-D space weights); strongest first if over cap
    this.chordEnds = [];
    const ce = ds.columns.edges?.clusterEdges;
    const triples: Array<[number, number, number]> = [];
    if (ce) {
      for (let i = 0; i < ce.length; i += 3) {
        const a = this.rimByClusterId.get(ce[i]!);
        const b = this.rimByClusterId.get(ce[i + 1]!);
        if (a === undefined || b === undefined) continue;
        triples.push([a, b, ce[i + 2]!]);
      }
      triples.sort((p, q) => q[2] - p[2]);
      triples.length = Math.min(triples.length, MAX_CHORDS);
    }

    const seg = this.chordSeg.array as Float32Array;
    const meta = this.chordMeta.array as Float32Array;
    for (let i = 0; i < triples.length; i++) {
      const [ai, bi, weight] = triples[i]!;
      const A = this.rim[ai]!;
      const B = this.rim[bi]!;
      // endpoints tucked just inside the rim so ribbons emerge from the discs
      seg[i * 4] = A.x * 0.985;
      seg[i * 4 + 1] = A.y * 0.985;
      seg[i * 4 + 2] = B.x * 0.985;
      seg[i * 4 + 3] = B.y * 0.985;
      meta[i * 4] = weight;
      meta[i * 4 + 1] = (A.clusterId * 0.61803398875) % 1;
      meta[i * 4 + 2] = (B.clusterId * 0.61803398875) % 1;
      meta[i * 4 + 3] = 1;
      A.degree++;
      B.degree++;
      this.chordEnds.push([ai, bi]);
    }
    this.chordSeg.needsUpdate = true;
    this.chordMeta.needsUpdate = true;
    this.chordMesh.count = triples.length;

    const np = this.nodePos.array as Float32Array;
    const nc = this.nodeColor.array as Float32Array;
    const nsz = this.nodeSize.array as Float32Array;
    for (let i = 0; i < n; i++) {
      const r = this.rim[i]!;
      np[i * 2] = r.x;
      np[i * 2 + 1] = r.y;
      const [cr, cg, cb] = clusterColor(r.clusterId);
      nc[i * 3] = cr;
      nc[i * 3 + 1] = cg;
      nc[i * 3 + 2] = cb;
      nsz[i] = r.sizePx;
    }
    this.nodePos.needsUpdate = true;
    this.nodeColor.needsUpdate = true;
    this.nodeSize.needsUpdate = true;
    this.nodeMesh.count = n;

    this.applyHighlight();
    this.rebuildLabels();
  }

  // ── rotated radial labels (HTML-first law) ───────────────────────────────

  /** Greedy label pick: biggest clusters first, keep only those whose rim
   *  angle clears the already-kept set by a px-derived angular gap. Re-run on
   *  resize — the gap depends on the projected rim radius. */
  private rebuildLabels(): void {
    if (!this.labelRoot) return;
    this.labelRoot.textContent = "";
    this.labels = [];
    if (!this.ds || this.rim.length === 0) return;

    const rimPx = RIM_R / Math.max(this.wpp, 1e-6);
    const minGap = 13 / Math.max(rimPx, 1);
    const bySize = this.rim
      .map((r, i) => ({ i, size: this.ds!.columns.clusters[r.clusterIdx]!.size }))
      .sort((a, b) => b.size - a.size);

    const kept: number[] = [];
    for (const { i } of bySize) {
      const ang = this.rim[i]!.angle;
      let ok = true;
      for (const k of kept) {
        let d = Math.abs(ang - this.rim[k]!.angle);
        d = Math.min(d, Math.PI * 2 - d);
        if (d < minGap) {
          ok = false;
          break;
        }
      }
      if (!ok) continue;
      kept.push(i);
      const el = document.createElement("div");
      el.className = "chord-label";
      const title = this.ds.columns.clusters[this.rim[i]!.clusterIdx]!.title;
      el.textContent = title.length > LABEL_MAX_CHARS ? `${title.slice(0, LABEL_MAX_CHARS - 1)}…` : title;
      this.labelRoot.appendChild(el);
      this.labels.push({ el, rimIdx: i });
    }
    this.positionLabels();
  }

  private positionLabels(): void {
    const cx = this.cssW / 2;
    const cy = this.cssH / 2;
    const rimPx = RIM_R / Math.max(this.wpp, 1e-6);
    for (const { el, rimIdx } of this.labels) {
      const r = this.rim[rimIdx]!;
      const rad = rimPx + r.sizePx + 10;
      const sx = cx + Math.cos(r.angle) * rad;
      const sy = cy - Math.sin(r.angle) * rad;
      const flip = Math.cos(r.angle) < 0;
      const deg = (-r.angle * 180) / Math.PI + (flip ? 180 : 0);
      el.style.transform =
        `translate(${sx.toFixed(1)}px, ${sy.toFixed(1)}px) rotate(${deg.toFixed(2)}deg) ` +
        `translate(${flip ? "-100%" : "0"}, -50%)`;
    }
  }

  // ── interaction: radial/angular hover, click to pin ─────────────────────

  /** screen CSS px → nearest rim node, or null when off the ring */
  private hitTest(x: number, y: number): number | null {
    if (this.rim.length === 0) return null;
    const wx = (x - this.cssW / 2) * this.wpp;
    const wy = -(y - this.cssH / 2) * this.wpp;
    const r = Math.hypot(wx, wy);
    if (Math.abs(r - RIM_R) > 0.14) return null;
    const ang = Math.atan2(wy, wx);
    let best = -1;
    let bestD = Infinity;
    for (let i = 0; i < this.rim.length; i++) {
      let d = Math.abs(ang - this.rim[i]!.angle);
      d = Math.min(d, Math.PI * 2 - d);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    const spacing = (Math.PI * 2) / this.rim.length;
    return bestD <= Math.max(spacing * 0.6, 0.02) ? best : null;
  }

  private hoverRim: number | null = null;

  private onPointerMove(e: PointerEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    const idx = this.hitTest(e.clientX - rect.left, e.clientY - rect.top);
    this.setHover(idx);
    if (idx !== null) {
      const r = this.rim[idx]!;
      const c = this.ds?.columns.clusters[r.clusterIdx];
      if (c) {
        this.tooltip.innerHTML = "";
        const line1 = document.createElement("div");
        line1.className = "point-tooltip-label";
        line1.textContent = c.title;
        const line2 = document.createElement("div");
        line2.className = "point-tooltip-conf";
        const edges = this.ds?.columns.edges;
        const provenance = edges
          ? ` (${edges.metric.replace(/_/g, " ")} in ${edges.space})`
          : "";
        line2.textContent = `${c.size.toLocaleString("en-US")} tokens · ${r.degree} links${provenance}`;
        this.tooltip.append(line1, line2);
        this.tooltip.style.visibility = "visible";
        const px = Math.min(e.clientX - rect.left + 14, this.cssW - 240);
        const py = Math.min(e.clientY - rect.top + 14, this.cssH - 56);
        this.tooltip.style.transform = `translate(${px.toFixed(1)}px, ${py.toFixed(1)}px)`;
      }
      this.canvas.style.cursor = "pointer";
    } else {
      this.tooltip.style.visibility = "hidden";
      this.canvas.style.cursor = "";
    }
  }

  private setHover(idx: number | null): void {
    if (idx === this.hoverRim) return;
    this.hoverRim = idx;
    if (idx === null) this.tooltip.style.visibility = "hidden";
    appStore
      .getState()
      .setHover(idx === null ? null : { kind: "cluster", id: this.rim[idx]!.clusterId });
  }

  private onClick(e: PointerEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    const idx = this.hitTest(e.clientX - rect.left, e.clientY - rect.top);
    appStore
      .getState()
      .setSelection(idx === null ? null : { kind: "cluster", id: this.rim[idx]!.clusterId });
  }

  /** Focus = selection if pinned, else hover. Chords touching the focused
   *  cluster stay lit, everything else recedes; no focus → everything lit. */
  private applyHighlight(): void {
    const { hover, selection } = appStore.getState();
    const focusSel = selection?.kind === "cluster" ? selection : hover?.kind === "cluster" ? hover : null;
    const focus = focusSel ? (this.rimByClusterId.get(focusSel.id) ?? null) : null;

    const na = this.nodeActive.array as Float32Array;
    const meta = this.chordMeta.array as Float32Array;
    if (focus === null) {
      na.fill(1);
      for (let i = 0; i < this.chordEnds.length; i++) meta[i * 4 + 3] = 1;
    } else {
      na.fill(0);
      na[focus] = 1;
      for (let i = 0; i < this.chordEnds.length; i++) {
        const [a, b] = this.chordEnds[i]!;
        const on = a === focus || b === focus;
        meta[i * 4 + 3] = on ? 1 : 0;
        if (on) {
          na[a] = Math.max(na[a]!, 0.95);
          na[b] = Math.max(na[b]!, 0.95);
        }
      }
    }
    this.nodeActive.needsUpdate = true;
    this.chordMeta.needsUpdate = true;
    for (const { el, rimIdx } of this.labels)
      el.classList.toggle("is-dim", focus !== null && na[rimIdx]! < 0.2);
  }

  // ── SceneDriver plumbing ─────────────────────────────────────────────────

  frame(_dt: number, _t: number): void {
    this.renderer.render(this.scene, this.camera);
  }

  resize(width: number, height: number, dpr: number): void {
    this.cssW = width;
    this.cssH = height;
    this.renderer.setPixelRatio(Math.min(dpr, 2));
    this.renderer.setSize(width, height, false);
    const short = Math.max(Math.min(width, height), 1);
    this.wpp = (HALF_MIN * 2) / short;
    this.uWpp.value = this.wpp;
    this.camera.left = (-width / 2) * this.wpp;
    this.camera.right = (width / 2) * this.wpp;
    this.camera.top = (height / 2) * this.wpp;
    this.camera.bottom = (-height / 2) * this.wpp;
    this.camera.updateProjectionMatrix();
    this.rebuildLabels();
  }

  pick(x: number, y: number): Selection | null {
    const idx = this.hitTest(x, y);
    return idx === null ? null : { kind: "cluster", id: this.rim[idx]!.clusterId };
  }

  snapshotForTransition(): HTMLCanvasElement | null {
    return null;
  }

  dispose(): void {
    for (const d of this.disposers) d();
    this.disposers = [];
    this.labelRoot?.remove();
    this.tooltip?.remove();
    for (const m of this.materials) m.dispose();
    this.chordMesh?.geometry.dispose();
    this.nodeMesh?.geometry.dispose();
    this.rampTex?.dispose();
    this.renderer?.dispose();
  }
}
