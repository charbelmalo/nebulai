/** The Atlas view — the video's "US map" scene. Owns the WebGPURenderer (or
 *  its forceWebGL rung), an orthographic camera driven by Camera2D, the
 *  points/territories layers, HTML label pills, kdbush hover picking, and the
 *  pointer gestures (drag pan, cursor-anchored wheel zoom, click select,
 *  eased fly-to). Writes hover/selection into the shared store; never talks
 *  to other drivers directly. */

import * as THREE from "three/webgpu";
import type { GpuTier } from "../../app/capabilities";
import { appStore, type Selection } from "../../app/store";
import { clusterDegrees, clusterNeighbors, formatCount, knnNeighbors } from "../../data/edges";
import type { Dataset } from "../../data/loader";
import { hullRadius, type ClusterHull } from "../../data/hulls";
import { BeamBadges, type BadgeSpec } from "../../chrome/BeamBadges";
import { Tooltip } from "../../chrome/Tooltip";
import { Camera2D, easeInOutCubic } from "../camera2d";
import { LabelOverlay } from "../labels/LabelOverlay";
import { BeamsLayer, type Beam } from "../layers/BeamsLayer";
import { FlareLayer } from "../layers/FlareLayer";
import { HaloLayer, type Halo } from "../layers/HaloLayer";
import { clusterColor, PointsLayer } from "../layers/PointsLayer";
import { TerritoriesLayer } from "../layers/TerritoriesLayer";
import { IdPicker, PointPicker } from "../picking";
import { createBloomPipeline, type BloomPipeline } from "../post/bloom";
import type { SceneDriver } from "../SceneDriver";

const POINT_PX = 4.5;
const HOVER_RADIUS_PX = 10;
const DRAG_THRESHOLD_PX = 3;
const MAX_CLUSTER_BEAMS = 12; // strongest neighbors of the selected hub
const HALO_HUBS = 8; // top clusters by summed edge weight get pulsing rings

// orbit (3-D only): middle/right-drag rotates azimuth+elevation; a trackpad
// two-finger horizontal swipe rotates azimuth. Elevation offset is clamped so
// the camera never dips under the map or snaps fully overhead.
const ORBIT_AZ_SPEED = 0.008; // rad per px of horizontal drag
const ORBIT_EL_SPEED = 0.006; // rad per px of vertical drag
const ORBIT_EL_MIN = -0.55;
const ORBIT_EL_MAX = 0.85;
const WHEEL_ORBIT_AZ = 0.004; // rad per px of horizontal wheel/swipe
const EL_CLAMP_MAX = 1.45; // ~83° from overhead — keep the horizon off-screen

// navigation smoothing: raw input moves *targets*; the rendered angles ease
// toward them, and a released drag coasts on its exponentially-decaying
// velocity. Reduced motion snaps instantly (no ease, no inertia, no auto-orbit).
const ORBIT_EASE_TAU_S = 0.08; // ease time-constant while following the target
const ORBIT_INERTIA_TAU_S = 0.35; // coast decay after release
const ORBIT_VEL_EPS = 1e-3; // rad/s — below this the coast is over
const ORBIT_SETTLE_EPS = 1e-4; // rad — target reached
const ZOOM_TAU_S = 0.05; // pending wheel factor settles in ~120 ms
const AUTO_ORBIT_RAD_S = 0.06; // base auto-orbit rate, scaled by orbitSpeed

// 2D↔3D morph: the camera lifts to this tilt while points glide pos2→pos3
const TILT_RAD = (38 * Math.PI) / 180;
const MORPH_MS = 900;
const ID_PICK_INTERVAL_MS = 33; // ~30Hz async id-buffer hover in 3D

export class AtlasDriver implements SceneDriver {
  readonly cam = new Camera2D();

  private canvas!: HTMLCanvasElement;
  private renderer!: THREE.WebGPURenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);

  private dataset: Dataset | null = null;
  private points: PointsLayer | null = null;
  private territories: TerritoriesLayer | null = null;
  private labels: LabelOverlay | null = null;
  private picker: PointPicker | null = null;
  private tooltip: Tooltip | null = null;
  private hullsById = new Map<number, ClusterHull>();

  // hero layers: beams + flare live for the driver's whole life; halos are
  // rebuilt per dataset (hub choice is data-driven)
  private beams: BeamsLayer | null = null;
  private flare: FlareLayer | null = null;
  private halos: HaloLayer | null = null;
  private badges: BeamBadges | null = null;
  private bloomPipe: BloomPipeline | null = null;
  private bloomOn = false;
  private tier: GpuTier = "webgpu";
  private reducedMotion = false;
  /** map extent (max bound dimension) — scale reference for flare sizing */
  private mapExtent = 1;

  private cameraDirty = true;
  private mouse: { x: number; y: number } | null = null;
  private hoverDirty = false;
  private hoveredIndex: number | null = null;

  private dragging = false;
  private pointerDown: { x: number; y: number } | null = null;
  private lastPointer: { x: number; y: number } | null = null;

  // orbit: user azimuth + extra elevation, scaled by morph so a flat 2-D map
  // stays exactly top-down (overlays project top-down and must not drift).
  // orbitAz/orbitEl are the *rendered* angles; input writes the targets and
  // stepNavigation() eases the rendered angles toward them each frame.
  private orbiting = false;
  private orbitLast: { x: number; y: number } | null = null;
  private orbitAz = 0;
  private orbitEl = 0;
  private orbitAzTarget = 0;
  private orbitElTarget = 0;
  /** release-inertia velocities (rad/s), EMA of the live drag velocity */
  private orbitAzVel = 0;
  private orbitElVel = 0;
  private orbitMoveAt = 0;
  /** wheel zoom: pending log-factor drained over ~120 ms, cursor-anchored */
  private zoomPending = 0;
  private zoomAnchor = { x: 0, y: 0 };

  /** dataset bounds in pos2 space; fit is deferred while the viewport is
   *  degenerate (booting in a hidden/zero-size tab) and applied on resize */
  private bounds: [number, number, number, number] | null = null;
  private fitPending = false;
  /** once the user pans/zooms/flies, resizes stop re-framing the whole map */
  private userDroveCamera = false;

  // 2D↔3D: eased morph value + xy bounds of pos3 (its frame differs from
  // pos2's PCA frame, so the camera flies to re-frame during the morph)
  private morph = 0;
  private morphTween: { from: number; to: number; start: number; duration: number } | null = null;
  private bounds3: [number, number, number, number] | null = null;
  private camDist = 30;
  private idPicker: IdPicker | null = null;
  private lastIdPickAt = 0;
  private idPickBusy = false;
  private projScratch = new THREE.Vector3();

  private abort = new AbortController();
  private unsubscribes: (() => void)[] = [];

  async init(canvas: HTMLCanvasElement, tier: GpuTier): Promise<void> {
    this.canvas = canvas;
    this.renderer = new THREE.WebGPURenderer({
      canvas,
      antialias: true,
      alpha: true, // transparent clear — the CSS vignette is the backdrop
      forceWebGL: tier === "webgl",
    });
    await this.renderer.init();
    this.renderer.setClearColor(0x000000, 0);

    this.camera.position.z = 10;
    this.tier = tier;
    this.reducedMotion = appStore.getState().capabilities?.reducedMotion ?? false;
    this.cam.reducedMotion = this.reducedMotion;

    const overlay = document.getElementById("overlay-html")!;
    this.tooltip = new Tooltip(overlay);
    this.badges = new BeamBadges(overlay);

    this.beams = new BeamsLayer();
    this.flare = new FlareLayer();
    this.beams.uWidthScale.value = appStore.getState().appearance.atlas.beamWidth;
    this.scene.add(this.beams.object, this.flare.group);

    // bloom rides only on real webgpu; the webgl rung renders direct (keeps
    // e2e goldens deterministic and the transpiled path lean)
    if (tier === "webgpu") {
      this.bloomPipe = createBloomPipeline(this.renderer, this.scene, this.camera, "full");
    }
    this.bloomOn = this.bloomPipe !== null && appStore.getState().settings.bloom;

    this.attachPointer();
    this.unsubscribes.push(
      appStore.subscribe((s, prev) => {
        if (s.selection !== prev.selection) {
          this.labels?.setSelected(s.selection?.kind === "cluster" ? s.selection.id : null);
          this.applySelection(s.selection);
        }
        if (s.toggles !== prev.toggles) {
          if (this.territories) this.territories.visible = s.toggles.territories;
          if (this.labels) this.labels.visible = s.toggles.labels;
          if (this.halos) this.halos.visible = s.toggles.halos;
          if (this.points) this.points.uNoiseVis.value = s.toggles.noise ? 1 : 0;
          this.applyBeamsVisibility(s.toggles.beams);
          this.cameraDirty = true;
        }
        if (s.settings !== prev.settings) {
          if (this.points) {
            this.points.uScale.value = s.settings.pointScale;
            this.points.uConfFloor.value = s.settings.confidenceFloor;
          }
          this.bloomOn = this.bloomPipe !== null && s.settings.bloom;
        }
        if (s.appearance !== prev.appearance) {
          if (this.beams) this.beams.uWidthScale.value = s.appearance.atlas.beamWidth;
          if (this.halos) this.halos.uIntensity.value = s.appearance.atlas.haloIntensity;
          this.cameraDirty = true;
        }
        if (s.dims !== prev.dims) this.onDimsChange(s.dims);
        if (s.mapQuery !== prev.mapQuery) {
          this.points?.setMatches(s.mapQuery.results?.matchIds ?? null);
          this.cameraDirty = true;
        }
      }),
    );
  }

  /** Eased morph value, 0 = flat map … 1 = flythrough (exposed for tests). */
  get morphValue(): number {
    return this.morph;
  }

  setDataset(ds: Dataset): void {
    this.clearLayers();
    this.dataset = ds;

    this.points = new PointsLayer(ds.columns);
    this.territories = new TerritoriesLayer(ds.hulls);
    this.scene.add(this.territories.group, this.points.object);

    this.picker = new PointPicker(ds.columns.pos2, ds.columns.count);
    this.hullsById = new Map(ds.hulls.map((h) => [h.clusterId, h]));

    const overlay = document.getElementById("overlay-html")!;
    this.labels = new LabelOverlay(overlay, ds.hulls, ds.columns.clusters, (cid) => {
      appStore.getState().setSelection({ kind: "cluster", id: cid });
      this.flyToCluster(cid);
    });

    const t = appStore.getState().toggles;
    this.territories.visible = t.territories;
    this.labels.visible = t.labels;

    // frame the whole map (deferred if the viewport has no size yet)
    const p = ds.columns.pos2;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (let i = 0; i < ds.columns.count; i++) {
      const x = p[i * 2]!, y = p[i * 2 + 1]!;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    this.bounds = [minX, minY, maxX, maxY];
    this.mapExtent = Math.max(maxX - minX, maxY - minY) || 1;

    // pos3 lives in its own (pre-PCA) frame — track its xy bounds so the
    // dimension morph can re-frame the camera onto the 3-D cloud
    const q = ds.columns.pos3;
    let m3x0 = Infinity, m3y0 = Infinity, m3x1 = -Infinity, m3y1 = -Infinity;
    for (let i = 0; i < ds.columns.count; i++) {
      const x = q[i * 3]!, y = q[i * 3 + 1]!;
      if (x < m3x0) m3x0 = x;
      if (x > m3x1) m3x1 = x;
      if (y < m3y0) m3y0 = y;
      if (y > m3y1) m3y1 = y;
    }
    this.bounds3 = [m3x0, m3y0, m3x1, m3y1];

    // the tilt orbit needs the camera pulled back past the 3-D cloud's depth
    this.camDist = this.mapExtent * 2;
    this.camera.near = 0.1;
    this.camera.far = this.mapExtent * 8;

    this.fitPending = true;
    this.userDroveCamera = false;
    this.applyFit();

    // GPU id-buffer picking for the 3D flythrough (2D stays on kdbush)
    this.idPicker = new IdPicker(this.renderer, this.points.createIdMesh());
    if (this.cam.viewportW >= 2) this.idPicker.setSize(this.cam.viewportW, this.cam.viewportH);

    // pulsing halos on the hubs — data-driven: top clusters by summed edge
    // weight in 10-D cluster space (no edges → no halos, honestly)
    if (ds.columns.edges) {
      const degrees = [...clusterDegrees(ds.columns.edges).entries()].sort((a, b) => b[1] - a[1]);
      const halos: Halo[] = [];
      for (const [cid] of degrees.slice(0, HALO_HUBS)) {
        const hull = this.hullsById.get(cid);
        if (!hull) continue;
        // clamp: a spatially spread cluster can have a hull spanning half the
        // map — the halo marks the hub, it shouldn't engulf the atlas
        const radius = Math.min(
          Math.max(hullRadius(hull), this.mapExtent * 0.01),
          this.mapExtent * 0.03,
        );
        halos.push({ pos: hull.anchor, radius, color: clusterColor(cid) });
      }
      this.halos = new HaloLayer(halos);
      if (this.reducedMotion) this.halos.uMotion.value = 0;
      this.halos.uIntensity.value = appStore.getState().appearance.atlas.haloIntensity;
      this.halos.visible = t.halos;
      this.scene.add(this.halos.object);
    }

    this.beams?.clear();
    this.flare?.clearTarget();
    this.badges?.clear();
    this.applyBeamsVisibility(t.beams);
    this.points.uNoiseVis.value = t.noise ? 1 : 0;
    const settings = appStore.getState().settings;
    this.points.uScale.value = settings.pointScale;
    this.points.uConfFloor.value = settings.confidenceFloor;

    // fresh layers start flat — re-apply the current dimension morph
    this.applyMorph();

    this.cameraDirty = true;
    this.hoverClear();
  }

  frame(dt: number, t: number): void {
    if (!this.dataset || !this.points) return;
    // never submit against a zero-size swapchain (hidden/collapsed tab)
    if (this.cam.viewportW < 2 || this.cam.viewportH < 2) return;

    const now = performance.now();
    const flying = this.cam.update(now);
    if (flying) this.cameraDirty = true;

    if (this.morphTween) {
      const tw = this.morphTween;
      const k = Math.min((now - tw.start) / tw.duration, 1);
      this.morph = tw.from + (tw.to - tw.from) * easeInOutCubic(k);
      if (k >= 1) this.morphTween = null;
      this.applyMorph();
      this.cameraDirty = true;
    }

    this.stepNavigation(dt);

    if (this.hoverDirty || (this.cameraDirty && this.mouse)) {
      this.updateHover();
      this.hoverDirty = false;
    }

    // scene time drives halo breathing (beams/flare are static by design);
    // ?frozen pins t to 0 upstream, reduced motion zeroes uMotion instead
    if (this.halos) this.halos.uTime.value = t;

    if (this.cameraDirty) {
      const [hx, hy] = this.cam.halfExtents();
      this.camera.left = -hx;
      this.camera.right = hx;
      this.camera.top = hy;
      this.camera.bottom = -hy;
      // tilt lifts the (still orthographic) camera off the map plane as the
      // morph progresses — the video's axonometric flythrough look. Orbit adds
      // azimuth + extra elevation; both are scaled by morph so at morph=0 the
      // camera is exactly overhead (flat map, no overlay drift).
      const el = Math.min(this.morph * (TILT_RAD + this.orbitEl), EL_CLAMP_MAX);
      const az = this.morph * this.orbitAz;
      const sinEl = Math.sin(el);
      this.camera.position.set(
        this.cam.cx + Math.sin(az) * sinEl * this.camDist,
        this.cam.cy - Math.cos(az) * sinEl * this.camDist,
        Math.cos(el) * this.camDist,
      );
      this.camera.lookAt(this.cam.cx, this.cam.cy, 0);
      this.camera.updateProjectionMatrix();
      this.points.uSize.value = POINT_PX * this.cam.wpp;
      if (this.beams) this.beams.uWpp.value = this.cam.wpp;
      this.labels?.update(this.cam);
      this.badges?.update(this.cam);
      this.cameraDirty = false;
    }

    if (this.bloomOn && this.bloomPipe) this.bloomPipe.post.render();
    else this.renderer.render(this.scene, this.camera);
  }

  /** Per-frame navigation integrator: auto-orbit, release inertia, easing of
   *  the rendered orbit angles toward their targets, and draining the pending
   *  cursor-anchored wheel zoom. Marks the camera dirty while anything is
   *  still in motion; reduced motion snaps instantly and never coasts. */
  private stepNavigation(dtMs: number): void {
    const dt = Math.min(dtMs / 1000, 0.1); // clamp tab-switch dt spikes
    if (dt <= 0) return;

    // auto-orbit (Settings → Appearance → Atlas): slow cinematic spin once
    // the flythrough is up; any active gesture pauses it
    const { orbitEnabled, orbitSpeed } = appStore.getState().appearance.atlas;
    if (
      orbitEnabled &&
      this.morph > 0.5 &&
      !this.orbiting &&
      !this.dragging &&
      !this.reducedMotion
    ) {
      this.orbitAzTarget += dt * AUTO_ORBIT_RAD_S * orbitSpeed;
    }

    // release inertia: the coast velocity keeps pushing the target, decaying
    if (!this.orbiting) {
      if (
        Math.abs(this.orbitAzVel) > ORBIT_VEL_EPS ||
        Math.abs(this.orbitElVel) > ORBIT_VEL_EPS
      ) {
        this.orbitAzTarget += this.orbitAzVel * dt;
        this.orbitElTarget = Math.min(
          Math.max(this.orbitElTarget + this.orbitElVel * dt, ORBIT_EL_MIN),
          ORBIT_EL_MAX,
        );
        const decay = Math.exp(-dt / ORBIT_INERTIA_TAU_S);
        this.orbitAzVel *= decay;
        this.orbitElVel *= decay;
      } else {
        this.orbitAzVel = 0;
        this.orbitElVel = 0;
      }
    }

    // ease the rendered angles toward the targets
    const dAz = this.orbitAzTarget - this.orbitAz;
    const dEl = this.orbitElTarget - this.orbitEl;
    if (Math.abs(dAz) > ORBIT_SETTLE_EPS || Math.abs(dEl) > ORBIT_SETTLE_EPS) {
      if (this.reducedMotion) {
        this.orbitAz = this.orbitAzTarget;
        this.orbitEl = this.orbitElTarget;
      } else {
        const k = 1 - Math.exp(-dt / ORBIT_EASE_TAU_S);
        this.orbitAz += dAz * k;
        this.orbitEl += dEl * k;
        // snap the last hair so the loop actually settles
        if (Math.abs(this.orbitAzTarget - this.orbitAz) < ORBIT_SETTLE_EPS)
          this.orbitAz = this.orbitAzTarget;
        if (Math.abs(this.orbitElTarget - this.orbitEl) < ORBIT_SETTLE_EPS)
          this.orbitEl = this.orbitElTarget;
      }
      this.cameraDirty = true;
    }

    // drain the pending wheel zoom, anchored where the cursor last was
    if (this.zoomPending !== 0) {
      const k = 1 - Math.exp(-dt / ZOOM_TAU_S);
      let step = this.zoomPending * k;
      if (Math.abs(this.zoomPending - step) < 1e-4) step = this.zoomPending;
      this.zoomPending -= step;
      this.cam.zoomAt(this.zoomAnchor.x, this.zoomAnchor.y, Math.exp(step));
      this.cameraDirty = true;
      this.hoverDirty = true;
    }
  }

  resize(width: number, height: number, dpr: number): void {
    if (width < 2 || height < 2) return; // hidden/collapsed tab — keep last real size
    this.renderer.setPixelRatio(Math.min(dpr, 2));
    this.renderer.setSize(width, height, false);
    this.cam.setViewport(width, height);
    this.idPicker?.setSize(width, height);
    // while the camera is still on the automatic overview, keep it framed
    // through resizes; once the user drives, preserve their view instead
    if (!this.userDroveCamera) this.fitPending = true;
    this.applyFit();
    this.cameraDirty = true;
  }

  private applyFit(): void {
    if (!this.fitPending) return;
    if (this.cam.viewportW < 2 || this.cam.viewportH < 2) return;
    // frame whichever cloud the current dimension shows
    const b = appStore.getState().dims === 3 ? this.bounds3 : this.bounds;
    if (!b) return;
    this.cam.fitBounds(b[0], b[1], b[2], b[3], 72);
    this.fitPending = false;
  }

  // ── 2D↔3D dimension morph ───────────────────────────────────────────────

  /** Dims changed: tween the morph and fly the camera to frame the target
   *  cloud in one synchronized cinematic move. */
  private onDimsChange(dims: 2 | 3): void {
    const to = dims === 3 ? 1 : 0;
    const now = performance.now();
    const duration = this.reducedMotion ? this.cam.reducedFlyMs : MORPH_MS;
    this.morphTween = { from: this.morph, to, start: now, duration };
    this.hoverClear();

    const b = dims === 3 ? this.bounds3 : this.bounds;
    if (b && this.cam.viewportW >= 2) {
      const pad = 72;
      const wpp = Math.max(
        (b[2] - b[0]) / Math.max(this.cam.viewportW - pad * 2, 1),
        (b[3] - b[1]) / Math.max(this.cam.viewportH - pad * 2, 1),
      );
      this.cam.flyTo((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, wpp, now, duration);
    }
    // the dimension switch re-frames — resizes keep auto-fitting again
    this.userDroveCamera = false;
  }

  /** Push the eased morph into every layer that cares. Territories, label
   *  pills, and halos are flat-map furniture, so they fade out on the lift. */
  private applyMorph(): void {
    const m = this.morph;
    if (this.points) this.points.uMorph.value = m;
    this.territories?.setFade(1 - m);
    this.labels?.setFade(1 - m);
    if (this.halos) this.halos.uFade.value = 1 - m;
    appStore.getState().setMorphT(m);
  }

  pick(x: number, y: number): Selection | null {
    // 3D positions only exist on the GPU — clicks use the async id-buffer
    // hover result instead of this synchronous kdbush path
    if (this.morph > 0.5) {
      return this.hoveredIndex !== null ? { kind: "point", id: this.hoveredIndex } : null;
    }
    if (!this.picker) return null;
    const [wx, wy] = this.cam.screenToWorld(x, y);
    const i = this.picker.nearest(wx, wy, HOVER_RADIUS_PX * this.cam.wpp);
    return i >= 0 ? { kind: "point", id: i } : null;
  }

  snapshotForTransition(): HTMLCanvasElement | null {
    return this.canvas ?? null;
  }

  dispose(): void {
    this.abort.abort();
    for (const u of this.unsubscribes) u();
    this.unsubscribes = [];
    this.clearLayers();
    if (this.beams) {
      this.scene.remove(this.beams.object);
      this.beams.dispose();
      this.beams = null;
    }
    if (this.flare) {
      this.scene.remove(this.flare.group);
      this.flare.dispose();
      this.flare = null;
    }
    this.badges?.dispose();
    this.badges = null;
    this.bloomPipe?.dispose();
    this.bloomPipe = null;
    this.tooltip?.dispose();
    this.tooltip = null;
    this.renderer?.dispose();
  }

  // ── hero: selection → beams + badges + flare ──────────────────────────

  /** Rebuild beams/flare for a new selection. Cluster → edges to neighbor
   *  hubs (badge = neighbor size); point → its kNN row (badge = similarity).
   *  Weights are gaussian sims in 10-D u_cluster space, never display space. */
  private applySelection(sel: Selection | null): void {
    if (!this.beams || !this.flare || !this.dataset) return;
    const edges = this.dataset.columns.edges;
    // beams/flare are drawn in the flat map plane — in 3D they'd anchor to
    // stale pos2 coordinates, so selections there render no edges (yet)
    const in3d = appStore.getState().dims === 3 || this.morph > 0.02;
    if (!sel || !edges || in3d) {
      this.beams.clear();
      this.flare.clearTarget();
      this.badges?.clear();
      return;
    }

    const beams: Beam[] = [];
    const badgeSpecs: BadgeSpec[] = [];

    if (sel.kind === "cluster") {
      const hull = this.hullsById.get(sel.id);
      if (!hull) return;
      for (const nb of clusterNeighbors(edges, sel.id).slice(0, MAX_CLUSTER_BEAMS)) {
        const other = this.hullsById.get(nb.other);
        if (!other) continue;
        beams.push({ start: hull.anchor, end: other.anchor, weight: nb.weight });
        badgeSpecs.push({ start: hull.anchor, end: other.anchor, text: formatCount(other.size) });
      }
      // the flare marks the anchor, it must not engulf the cluster
      const size = Math.min(Math.max(hullRadius(hull) * 0.6, this.mapExtent * 0.015), this.mapExtent * 0.06);
      this.flare.setTarget(hull.anchor[0], hull.anchor[1], size, clusterColor(sel.id));
    } else {
      const p = this.dataset.columns.pos2;
      const start: [number, number] = [p[sel.id * 2]!, p[sel.id * 2 + 1]!];
      for (const nb of knnNeighbors(edges, sel.id)) {
        const end: [number, number] = [p[nb.id * 2]!, p[nb.id * 2 + 1]!];
        beams.push({ start, end, weight: nb.sim });
        badgeSpecs.push({ start, end, text: nb.sim.toFixed(2) });
      }
      const cid = this.dataset.columns.clusterId[sel.id]!;
      const color = cid >= 0 ? clusterColor(cid) : ([0.9, 0.85, 0.95] as [number, number, number]);
      this.flare.setTarget(start[0], start[1], this.mapExtent * 0.02, color);
    }

    // normalize weights for display so solid-vs-dotted contrast survives even
    // when all sims cluster near 1 — badges keep the raw honest numbers
    if (beams.length > 1) {
      let lo = Infinity, hi = -Infinity;
      for (const b of beams) {
        if (b.weight < lo) lo = b.weight;
        if (b.weight > hi) hi = b.weight;
      }
      const span = hi - lo;
      if (span > 1e-6) {
        for (const b of beams) b.weight = 0.15 + 0.85 * ((b.weight - lo) / span);
      }
    }

    this.beams.setBeams(beams);
    this.badges?.setBadges(badgeSpecs);
    this.badges?.update(this.cam);
    this.applyBeamsVisibility(appStore.getState().toggles.beams);
  }

  private applyBeamsVisibility(on: boolean): void {
    if (this.beams) this.beams.visible = on;
    if (this.flare) this.flare.visible = on;
    if (this.badges) this.badges.visible = on;
  }

  /** Zoom onto one point's neighborhood (search-result click). The window is
   *  a fixed fraction of the map so nearby tokens stay in frame for context. */
  flyToPoint(id: number): void {
    if (!this.dataset) return;
    const p = this.dataset.columns.pos2;
    const x = p[id * 2];
    const y = p[id * 2 + 1];
    if (x === undefined || y === undefined) return;
    this.userDroveCamera = true;
    const fitPx = Math.min(this.cam.viewportW, this.cam.viewportH) * 0.55;
    const wpp = Math.max((this.mapExtent * 0.06) / fitPx, this.cam.minWpp);
    this.cam.flyTo(x, y, wpp, performance.now());
  }

  /** Cinematic zoom onto one cluster (pill click / future keyboard nav). */
  flyToCluster(clusterId: number): void {
    const hull = this.hullsById.get(clusterId);
    if (!hull) return;
    this.userDroveCamera = true;
    const fitPx = Math.min(this.cam.viewportW, this.cam.viewportH) * 0.55;
    const wpp = Math.max((hullRadius(hull) * 2) / fitPx, this.cam.minWpp);
    this.cam.flyTo(hull.anchor[0], hull.anchor[1], wpp, performance.now());
  }

  private clearLayers(): void {
    if (this.points) {
      this.scene.remove(this.points.object);
      this.points.dispose();
      this.points = null;
    }
    if (this.territories) {
      this.scene.remove(this.territories.group);
      this.territories.dispose();
      this.territories = null;
    }
    if (this.halos) {
      this.scene.remove(this.halos.object);
      this.halos.dispose();
      this.halos = null;
    }
    this.beams?.clear();
    this.flare?.clearTarget();
    this.badges?.clear();
    this.labels?.dispose();
    this.labels = null;
    this.picker = null;
    this.idPicker?.dispose();
    this.idPicker = null;
    this.hullsById.clear();
  }

  // ── pointer gestures ────────────────────────────────────────────────────

  private attachPointer(): void {
    const c = this.canvas;
    const opts = { signal: this.abort.signal };

    c.addEventListener(
      "pointerdown",
      (e) => {
        // middle (wheel-click) or right button → orbit the camera
        if (e.button === 1 || e.button === 2) {
          e.preventDefault();
          c.setPointerCapture(e.pointerId);
          this.orbiting = true;
          this.orbitLast = { x: e.clientX, y: e.clientY };
          this.orbitAzVel = 0;
          this.orbitElVel = 0;
          this.orbitMoveAt = performance.now();
          this.hoverClear();
          c.style.cursor = "move";
          return;
        }
        if (e.button !== 0) return;
        c.setPointerCapture(e.pointerId);
        this.pointerDown = { x: e.clientX, y: e.clientY };
        this.lastPointer = { x: e.clientX, y: e.clientY };
        this.dragging = false;
      },
      opts,
    );

    // right-drag orbits; suppress the browser context menu on the canvas
    c.addEventListener("contextmenu", (e) => e.preventDefault(), opts);

    c.addEventListener(
      "pointermove",
      (e) => {
        if (this.orbiting && this.orbitLast) {
          this.userDroveCamera = true;
          // orbiting is a 3-D affordance — on the first drag movement, lift a
          // flat map into the flythrough so the gesture is never a no-op (a
          // stray middle/right *click* without drag leaves the map alone)
          if (this.morph <= 0.02 && appStore.getState().dims !== 3) {
            appStore.getState().setDims(3);
          }
          const dAz = (e.clientX - this.orbitLast.x) * ORBIT_AZ_SPEED;
          const dEl = (e.clientY - this.orbitLast.y) * ORBIT_EL_SPEED;
          this.orbitAzTarget += dAz;
          this.orbitElTarget = Math.min(
            Math.max(this.orbitElTarget + dEl, ORBIT_EL_MIN),
            ORBIT_EL_MAX,
          );
          // EMA of the live velocity — becomes the coast speed on release
          const now = performance.now();
          const dtS = Math.max((now - this.orbitMoveAt) / 1000, 1e-3);
          this.orbitAzVel = this.orbitAzVel * 0.7 + (dAz / dtS) * 0.3;
          this.orbitElVel = this.orbitElVel * 0.7 + (dEl / dtS) * 0.3;
          this.orbitMoveAt = now;
          this.orbitLast = { x: e.clientX, y: e.clientY };
          this.cameraDirty = true;
          return;
        }
        if (this.pointerDown && this.lastPointer) {
          const dx = e.clientX - this.pointerDown.x;
          const dy = e.clientY - this.pointerDown.y;
          if (!this.dragging && Math.hypot(dx, dy) > DRAG_THRESHOLD_PX) {
            this.dragging = true;
            c.style.cursor = "grabbing";
            this.hoverClear();
          }
          if (this.dragging) {
            this.userDroveCamera = true;
            // tilted view foreshortens vertically — scale dy so the map
            // tracks the cursor instead of lagging it
            const el = Math.min(this.morph * (TILT_RAD + this.orbitEl), EL_CLAMP_MAX);
            const tiltComp = 1 / Math.max(Math.cos(el), 0.5);
            this.cam.panPixels(
              e.clientX - this.lastPointer.x,
              (e.clientY - this.lastPointer.y) * tiltComp,
            );
            this.lastPointer = { x: e.clientX, y: e.clientY };
            this.cameraDirty = true;
            return;
          }
          this.lastPointer = { x: e.clientX, y: e.clientY };
        }
        this.mouse = { x: e.clientX, y: e.clientY };
        this.hoverDirty = true;
      },
      opts,
    );

    c.addEventListener(
      "pointerup",
      (e) => {
        if (this.orbiting) {
          this.orbiting = false;
          this.orbitLast = null;
          c.style.cursor = "";
          // a pause before release means "stop here" — don't fling from a
          // velocity that's already stale
          if (performance.now() - this.orbitMoveAt > 90 || this.reducedMotion) {
            this.orbitAzVel = 0;
            this.orbitElVel = 0;
          }
          return;
        }
        const wasDrag = this.dragging;
        this.pointerDown = null;
        this.lastPointer = null;
        this.dragging = false;
        c.style.cursor = "";
        if (wasDrag) return;

        // click: select the picked point's cluster (noise → point selection)
        const picked = this.pick(e.clientX, e.clientY);
        const store = appStore.getState();
        if (!picked) {
          store.setSelection(null); // click empty space deselects
          return;
        }
        const cid = this.dataset?.columns.clusterId[picked.id] ?? -1;
        store.setSelection(cid >= 0 ? { kind: "cluster", id: cid } : picked);
      },
      opts,
    );

    c.addEventListener(
      "pointerleave",
      () => {
        this.mouse = null;
        this.hoverClear();
      },
      opts,
    );

    c.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.userDroveCamera = true;
        // in 3-D, a horizontal-dominant trackpad swipe orbits the azimuth;
        // pinch-zoom (ctrlKey) and vertical scroll / mouse-wheel keep zooming
        if (this.morph > 0.02 && !e.ctrlKey && Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
          this.orbitAzTarget += e.deltaX * WHEEL_ORBIT_AZ;
          this.cameraDirty = true;
          return;
        }
        // accumulate in log space, drained over ~120 ms in stepNavigation so
        // discrete wheel ticks read as one continuous glide
        if (this.reducedMotion) {
          this.cam.zoomAt(e.clientX, e.clientY, Math.exp(e.deltaY * 0.0012));
        } else {
          this.zoomPending += e.deltaY * 0.0012;
          this.zoomAnchor = { x: e.clientX, y: e.clientY };
        }
        this.cameraDirty = true;
        this.hoverDirty = true;
      },
      { signal: this.abort.signal, passive: false },
    );

    window.addEventListener(
      "keydown",
      (e) => {
        if (e.key === "Escape") appStore.getState().setSelection(null);
      },
      opts,
    );
  }

  private updateHover(): void {
    if (!this.mouse || !this.dataset || this.dragging) return;
    if (this.morph > 0.02) {
      this.updateHover3D();
      return;
    }
    const picked = this.pick(this.mouse.x, this.mouse.y);
    this.setHovered(picked?.id ?? null);
  }

  /** Async id-buffer hover for the flythrough, throttled to ~30Hz. Results
   *  that resolve after a dataset switch or a return to 2D are dropped. */
  private updateHover3D(): void {
    if (!this.idPicker || this.idPicker.broken || this.idPickBusy) return;
    const now = performance.now();
    if (now - this.lastIdPickAt < ID_PICK_INTERVAL_MS) return;
    this.lastIdPickAt = now;
    this.idPickBusy = true;
    const dataset = this.dataset;
    const mouse = this.mouse!;
    void this.idPicker
      .pick(this.camera, mouse.x, mouse.y)
      .then((i) => {
        if (this.dataset !== dataset || this.morph <= 0.02) return;
        this.setHovered(i >= 0 && i < dataset!.columns.count ? i : null);
      })
      .finally(() => {
        this.idPickBusy = false;
      });
  }

  /** Shared hover application: highlight, store, tooltip (projected through
   *  the live morphed position, so it tracks points in 2D and 3D alike). */
  private setHovered(index: number | null): void {
    // same non-null index still falls through: the tooltip re-anchors as the
    // camera moves under a held hover
    if (index === this.hoveredIndex && index === null) return;
    if (index !== this.hoveredIndex) {
      this.hoveredIndex = index;
      this.points?.setHover(index);
      appStore.getState().setHover(index !== null ? { kind: "point", id: index } : null);
    }

    if (index !== null && this.tooltip && this.dataset) {
      const cols = this.dataset.columns;
      const cid = cols.clusterId[index]!;
      const title = cid >= 0
        ? (cols.clusters.find((cl) => cl.id === cid)?.title ?? `cluster ${cid}`)
        : null;
      const [sx, sy] = this.projectPoint(index);
      this.tooltip.show(sx, sy, {
        label: cols.labels[index]!,
        clusterTitle: title,
        confidence: cols.confidence[index]! / 255,
      });
      this.canvas.style.cursor = "pointer";
    } else {
      this.tooltip?.hide();
      if (!this.dragging) this.canvas.style.cursor = "";
    }
  }

  /** Screen position of point i at the current morph, via the render camera
   *  (matches the GPU's mix(pos2, pos3, uMorph) exactly). */
  private projectPoint(i: number): [number, number] {
    const cols = this.dataset!.columns;
    const m = this.morph;
    const v = this.projScratch.set(
      cols.pos2[i * 2]! * (1 - m) + cols.pos3[i * 3]! * m,
      cols.pos2[i * 2 + 1]! * (1 - m) + cols.pos3[i * 3 + 1]! * m,
      cols.pos3[i * 3 + 2]! * m,
    );
    v.project(this.camera);
    return [(v.x * 0.5 + 0.5) * this.cam.viewportW, (-v.y * 0.5 + 0.5) * this.cam.viewportH];
  }

  private hoverClear(): void {
    this.hoveredIndex = null;
    this.points?.setHover(null);
    this.tooltip?.hide();
    appStore.getState().setHover(null);
  }
}
