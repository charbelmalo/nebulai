/** The Atlas view — the video's "US map" scene. Owns the WebGPURenderer (or
 *  its forceWebGL rung), an orthographic camera driven by Camera2D, the
 *  points/territories layers, HTML label pills, kdbush hover picking, and the
 *  pointer gestures (drag pan, cursor-anchored wheel zoom, click select,
 *  eased fly-to). Writes hover/selection into the shared store; never talks
 *  to other drivers directly. */

import * as THREE from "three/webgpu";
import type { GpuTier } from "@psychix/viz/capabilities";
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
const WHEEL_ORBIT_EL = 0.003; // rad per px of shift+vertical wheel/swipe
const EL_CLAMP_MAX = 1.45; // ~83° from overhead — keep the horizon off-screen

// wheel/trackpad. Browsers report deltas in three units (px / line / page) and
// only Chrome-on-mac reliably uses px, so everything is normalized to px before
// it reaches the zoom. Per-event log-zoom is clamped because a trackpad flick
// can deliver one 400px delta that would otherwise jump ~1.6× in a single tick.
const WHEEL_LINE_PX = 16;
const WHEEL_PAGE_PX = 100;
const WHEEL_ZOOM_GAIN = 0.0012; // log-factor per px of wheel delta
const PINCH_ZOOM_GAIN = 0.012; // trackpad pinch arrives as small ctrl+wheel
const WHEEL_ZOOM_MAX = 0.22; // max log-factor a single event may contribute

// touch: 1 finger pans, 2 fingers pinch-zoom + twist-orbit + drag-tilt (the
// Maps convention). Thresholds keep a shaky two-finger hold from spinning the
// camera before the user has actually committed to a twist.
const TOUCH_TWIST_DEADZONE = 0.06; // rad — twist ignored below this
const TOUCH_TILT_SPEED = 0.005; // rad per px of two-finger vertical drag
const TOUCH_ORBIT_AZ_SPEED = 0.005; // rad per px of two-finger horizontal drag

// orbit input is direct-drive: a drag maps 1:1 onto the rendered angles — no
// easing, no release inertia. Only the wheel zoom keeps a short glide.
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
  /** cluster centroids in pos3 (u3 display) space — 3D beam anchors */
  private centroid3ById = new Map<number, [number, number, number]>();
  /** per-cluster pos3 xy spread radius (RMS*2, clamped) — 3D fly-to framing */
  private radius3ById = new Map<number, number>();

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
  // Input writes orbitAz/orbitEl directly — what you drag is what renders.
  private orbiting = false;
  private orbitLast: { x: number; y: number } | null = null;
  private orbitAz = 0;
  private orbitEl = 0;
  // orbit pivot: the world point the camera rotates around. Resolved at
  // gesture start (raycasted node → selection → view-center cloud depth →
  // ground plane) and held in both frames because the rendered cloud is
  // mix(pos2, pos3, morph). The anchor stores the pivot's view-plane offsets
  // (world units along camera right/up); each frame the camera center is
  // re-solved so those offsets stay invariant — which pins the pivot to one
  // screen position while the angles change, i.e. the cloud visibly rotates
  // around it instead of swinging off-frame around a z=0 ground point.
  private orbitPivot: { p2: [number, number]; p3: [number, number, number] } | null = null;
  private orbitAnchor: { a: number; b: number } | null = null;
  private wheelOrbitAt = 0; // last wheel-orbit tick — a fresh swipe re-grabs
  /** wheel zoom: pending log-factor drained over ~120 ms, cursor-anchored */
  private zoomPending = 0;
  private zoomAnchor = { x: 0, y: 0 };

  // This driver's touch handling is intentionally private rather than adopting
  // the shared src/scene/gestures.ts recognizer — its pinch is entangled with
  // morph state and the id-picker — but it mirrors that recognizer's gesture
  // model (1 finger pan, 2 finger pinch+twist+drag, tap-vs-drag threshold).
  /** live touch points, in order of contact — 2+ entries switch to the pinch
   *  gesture. Mouse/pen pointers are never tracked here. */
  private touches = new Map<number, { x: number; y: number }>();
  /** two-finger gesture baseline, sampled when the second finger lands */
  private pinch: {
    dist: number;
    angle: number;
    cx: number;
    cy: number;
    /** accumulated twist and whether it has cleared the rotation deadzone */
    twist: number;
    twistOn: boolean;
  } | null = null;

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
  /** max xy dimension of the pos3 cloud — scale reference for 3D fly-to */
  private extent3 = 1;
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
    this.centroid3ById = new Map(ds.columns.clusters.map((c) => [c.id, c.centroid]));

    const overlay = document.getElementById("overlay-html")!;
    this.labels = new LabelOverlay(overlay, ds.hulls, ds.columns.clusters, (cid) => {
      appStore.getState().setSelection({ kind: "cluster", id: cid });
      // flyToCluster is morph-aware — it aims at the pos2 hull anchor when flat
      // and the pos3 centroid mid-flythrough, so pills fly correctly in both
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
    this.extent3 = Math.max(m3x1 - m3x0, m3y1 - m3y0) || 1;

    // per-cluster pos3 xy spread, so a 3D fly-to frames the neighborhood at a
    // sane zoom. RMS*2 (not max) so one stray point can't inflate the window;
    // the band clamp then guards both a lone tight cluster and a diffuse one.
    this.radius3ById.clear();
    {
      const cid = ds.columns.clusterId;
      const sumSq = new Map<number, number>();
      const counts = new Map<number, number>();
      for (let i = 0; i < ds.columns.count; i++) {
        const c = cid[i]!;
        if (c < 0) continue;
        const centroid = this.centroid3ById.get(c);
        if (!centroid) continue;
        const dx = q[i * 3]! - centroid[0];
        const dy = q[i * 3 + 1]! - centroid[1];
        sumSq.set(c, (sumSq.get(c) ?? 0) + dx * dx + dy * dy);
        counts.set(c, (counts.get(c) ?? 0) + 1);
      }
      const lo = this.extent3 * 0.01, hi = this.extent3 * 0.25;
      for (const [c, ss] of sumSq) {
        const n = counts.get(c)!;
        const rms2 = Math.sqrt(ss / n) * 2;
        this.radius3ById.set(c, Math.min(Math.max(rms2, lo), hi));
      }
    }

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
    this.applyOrbitPivot();

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
      const [az, el] = this.orbitAngles();
      const sinEl = Math.sin(el);
      const cosEl = Math.cos(el);
      const sinAz = Math.sin(az);
      const cosAz = Math.cos(az);
      this.camera.position.set(
        this.cam.cx + sinAz * sinEl * this.camDist,
        this.cam.cy - cosAz * sinEl * this.camDist,
        cosEl * this.camDist,
      );
      // This map is Z-up, so three's default camera.up (world +Y) lies *inside*
      // the map plane. Letting lookAt derive the roll from it pins world +Y to
      // screen-vertical at every azimuth and goes degenerate at the top of the
      // arc — the camera rocks instead of turning. Hand it the exact up of the
      // spherical frame instead: it is orthogonal to the view direction for
      // every (az, el), and at az=el=0 it is (0,1,0), so the flat 2-D map keeps
      // the orientation the HTML/SVG overlays project against.
      this.camera.up.set(-cosEl * sinAz, cosEl * cosAz, sinEl);
      this.camera.lookAt(this.cam.cx, this.cam.cy, 0);
      this.camera.updateProjectionMatrix();
      // refresh matrixWorldInverse now (render would too, but a frame later)
      // so projectWorld-anchored pills track this frame's camera, not last's
      this.camera.updateMatrixWorld();
      this.points.uSize.value = POINT_PX * this.cam.wpp;
      if (this.beams) this.beams.uWpp.value = this.cam.wpp;
      this.labels?.update(this.cam, this.morph, this.projectWorld);
      this.badges?.update(this.cam);
      this.cameraDirty = false;
    }

    if (this.bloomOn && this.bloomPipe) this.bloomPipe.post.render();
    else this.renderer.render(this.scene, this.camera);
  }

  /** The rendered spherical camera angles: azimuth around the map's +Z axis and
   *  elevation measured *from* overhead. Both are scaled by the morph so a flat
   *  2-D map stays exactly top-down, and elevation is capped short of the
   *  horizon. Every gesture that has to reason about the camera's ground frame
   *  (pan, hover, fly-to framing) reads them from here so they can't drift out
   *  of step with the matrix frame() builds. */
  private orbitAngles(): [az: number, el: number] {
    return [
      this.morph * this.orbitAz,
      Math.min(this.morph * (TILT_RAD + this.orbitEl), EL_CLAMP_MAX),
    ];
  }

  /** Pan by a screen-space drag delta. Camera2D pans along world X/Y, so once
   *  the orbit carries an azimuth a raw delta slides the map off at an angle —
   *  rotate the delta into the camera's ground frame first. The vertical
   *  component is additionally divided by cos(el) because a tilted view
   *  foreshortens depth, and without it the map lags the cursor. */
  private panScreen(dxPx: number, dyPx: number): void {
    this.clearOrbitPivot(); // panning moves the center deliberately
    const [az, el] = this.orbitAngles();
    const dy = dyPx / Math.max(Math.cos(el), 0.35);
    const cosAz = Math.cos(az);
    const sinAz = Math.sin(az);
    this.cam.panPixels(dxPx * cosAz + dy * sinAz, -dxPx * sinAz + dy * cosAz);
    this.userDroveCamera = true;
    this.cameraDirty = true;
  }

  /** Apply a raw orbit delta (radians) directly, clamping elevation. Shared by
   *  mouse drag, wheel/trackpad swipe and touch. */
  private orbitBy(dAz: number, dEl: number): void {
    this.orbitAz += dAz;
    this.orbitEl = Math.min(Math.max(this.orbitEl + dEl, ORBIT_EL_MIN), ORBIT_EL_MAX);
    this.userDroveCamera = true;
    this.cameraDirty = true;
  }

  /** Orbiting is a 3-D affordance: on the first movement of an orbit gesture,
   *  lift a flat map into the flythrough so the gesture is never a no-op. A
   *  stray click/tap without movement leaves the map alone. */
  private ensure3DForOrbit(): void {
    if (this.morph <= 0.02 && appStore.getState().dims !== 3) {
      appStore.getState().setDims(3);
    }
  }

  // ── orbit pivot ─────────────────────────────────────────────────────────

  /** Resolve the orbit pivot and capture its anchor. Called at the start of
   *  every orbit interaction, before hover is cleared. */
  private grabOrbitPivot(useHover = true): void {
    this.orbitPivot = this.resolveOrbitPivot(useHover);
    this.captureOrbitAnchor();
  }

  private clearOrbitPivot(): void {
    this.orbitPivot = null;
    this.orbitAnchor = null;
  }

  /** Fallback chain: the raycasted node under the mouse → the selected point /
   *  cluster centroid → the view-center at the visible cloud's median depth →
   *  the ground plane. `useHover` is off for auto-orbit, where a mouse merely
   *  resting on a point shouldn't hijack the spin. */
  private resolveOrbitPivot(
    useHover: boolean,
  ): { p2: [number, number]; p3: [number, number, number] } | null {
    const ds = this.dataset;
    if (!ds) return null;
    const p = ds.columns.pos2;
    const q = ds.columns.pos3;
    const point = (i: number) => ({
      p2: [p[i * 2]!, p[i * 2 + 1]!] as [number, number],
      p3: [q[i * 3]!, q[i * 3 + 1]!, q[i * 3 + 2]!] as [number, number, number],
    });
    if (useHover && this.hoveredIndex !== null) return point(this.hoveredIndex);
    const sel = appStore.getState().selection;
    if (sel?.kind === "point") return point(sel.id);
    if (sel?.kind === "cluster") {
      const c3 = this.centroid3ById.get(sel.id);
      const hull = this.hullsById.get(sel.id);
      if (c3 && hull) return { p2: [hull.anchor[0], hull.anchor[1]], p3: [...c3] };
    }
    // median depth of the points currently in frame, pivot at the view center
    const m = this.morph;
    const [hx, hy] = this.cam.halfExtents();
    const n = Math.min(p.length / 2, q.length / 3);
    const zs: number[] = [];
    for (let i = 0; i < n; i++) {
      const x = p[i * 2]! + (q[i * 3]! - p[i * 2]!) * m;
      const y = p[i * 2 + 1]! + (q[i * 3 + 1]! - p[i * 2 + 1]!) * m;
      if (Math.abs(x - this.cam.cx) <= hx && Math.abs(y - this.cam.cy) <= hy)
        zs.push(q[i * 3 + 2]!);
    }
    if (zs.length) {
      zs.sort((a, b) => a - b);
      const zMed = zs[zs.length >> 1]!;
      return {
        p2: [this.cam.cx, this.cam.cy],
        p3: [this.cam.cx, this.cam.cy, zMed],
      };
    }
    return { p2: [this.cam.cx, this.cam.cy], p3: [this.cam.cx, this.cam.cy, 0] };
  }

  /** The pivot's current world position — the same morph mix PointsLayer's
   *  positionNode applies, so the pivot tracks the rendered cloud even while
   *  a 2-D map is still lifting into the flythrough. */
  private pivotWorld(): [number, number, number] {
    const pv = this.orbitPivot!;
    const m = this.morph;
    return [
      pv.p2[0] + (pv.p3[0] - pv.p2[0]) * m,
      pv.p2[1] + (pv.p3[1] - pv.p2[1]) * m,
      pv.p3[2] * m,
    ];
  }

  /** Record the pivot's offsets along the camera's right/up axes (world
   *  units). Holding these invariant pins the pivot to one screen position. */
  private captureOrbitAnchor(): void {
    if (!this.orbitPivot) {
      this.orbitAnchor = null;
      return;
    }
    const [px, py, pz] = this.pivotWorld();
    const [az, el] = this.orbitAngles();
    const dx = px - this.cam.cx;
    const dy = py - this.cam.cy;
    this.orbitAnchor = {
      a: dx * Math.cos(az) + dy * Math.sin(az),
      b: (-dx * Math.sin(az) + dy * Math.cos(az)) * Math.cos(el) + pz * Math.sin(el),
    };
  }

  /** Re-solve the camera center so the pivot keeps its captured anchor at the
   *  current angles + morph. The orthographic projection makes this exact —
   *  rotating about a 3-D point is just az/el plus this compensating pan, so
   *  no drift accumulates. Runs every frame; a no-op once settled. */
  private applyOrbitPivot(): void {
    if (!this.orbitPivot || !this.orbitAnchor) return;
    const [px, py, pz] = this.pivotWorld();
    const [az, el] = this.orbitAngles();
    const b2 = (this.orbitAnchor.b - pz * Math.sin(el)) / Math.cos(el); // el clamped < 90°
    const cosAz = Math.cos(az);
    const sinAz = Math.sin(az);
    const cx = px - (this.orbitAnchor.a * cosAz - b2 * sinAz);
    const cy = py - (this.orbitAnchor.a * sinAz + b2 * cosAz);
    if (cx !== this.cam.cx || cy !== this.cam.cy) {
      this.cam.cx = cx;
      this.cam.cy = cy;
      this.cameraDirty = true;
    }
  }

  /** Wheel orbit has no gesture boundaries — treat a >400 ms gap as a fresh
   *  swipe and re-grab the pivot (the cursor may be on a different node). */
  private refreshWheelOrbitPivot(): void {
    const now = performance.now();
    if (!this.orbitPivot || now - this.wheelOrbitAt > 400) this.grabOrbitPivot();
    this.wheelOrbitAt = now;
  }

  /** Per-frame navigation integrator: auto-orbit and draining the pending
   *  cursor-anchored wheel zoom. Orbit gestures write the angles directly,
   *  so there is nothing to ease or coast here. */
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
      // spin around the cloud's local depth (or the user's last pivot), not
      // the z=0 ground plane — zoomed in, the latter swings the cloud away
      if (!this.orbitPivot) this.grabOrbitPivot(false);
      this.orbitAz += dt * AUTO_ORBIT_RAD_S * orbitSpeed;
      this.cameraDirty = true;
    }

    // auto-orbit winds the azimuth up without bound; fold by whole turns so a
    // long-running spin can't bleed float precision into the trig
    if (Math.abs(this.orbitAz) > Math.PI * 2) {
      this.orbitAz -= Math.trunc(this.orbitAz / (Math.PI * 2)) * Math.PI * 2;
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
    this.clearOrbitPivot();
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

    // an orbit gesture lifting a flat map keeps its zoom and its pivot — the
    // anchor compensation carries the grabbed node through the morph. Only a
    // deliberate dims toggle gets the cinematic full-cloud re-frame.
    const orbitLift =
      this.orbiting || this.pinch !== null || now - this.wheelOrbitAt < 400;
    if (orbitLift) return;

    this.clearOrbitPivot();
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

  /** Push the eased morph into every layer that cares. Territories and halos
   *  are flat-map furniture, so they fade out on the lift; label pills persist
   *  (they re-anchor to cluster centroids via projectWorld in frame()). */
  private applyMorph(): void {
    const m = this.morph;
    if (this.points) this.points.uMorph.value = m;
    if (this.beams) this.beams.uMorph.value = m;
    this.territories?.setFade(1 - m);
    if (this.halos) this.halos.uFade.value = 1 - m;
    this.applyBeamsVisibility(appStore.getState().toggles.beams);
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
   *  Weights are gaussian sims in 10-D u_cluster space, never display space.
   *  Every beam carries both pos2 and pos3 endpoints — uMorph glides them in
   *  sync with the points, so selections work flat, mid-morph, and in 3D. */
  private applySelection(sel: Selection | null): void {
    if (!this.beams || !this.flare || !this.dataset) return;
    const edges = this.dataset.columns.edges;
    if (!sel || !edges) {
      this.beams.clear();
      this.flare.clearTarget();
      this.badges?.clear();
      return;
    }

    const beams: Beam[] = [];
    const badgeSpecs: BadgeSpec[] = [];
    // 3D anchor for a cluster: its pos3 centroid (fall back to the flat
    // anchor at z 0 so a missing centroid degrades visibly, not wrongly)
    const anchor3 = (h: ClusterHull): [number, number, number] =>
      this.centroid3ById.get(h.clusterId) ?? [h.anchor[0], h.anchor[1], 0];

    if (sel.kind === "cluster") {
      const hull = this.hullsById.get(sel.id);
      if (!hull) return;
      for (const nb of clusterNeighbors(edges, sel.id).slice(0, MAX_CLUSTER_BEAMS)) {
        const other = this.hullsById.get(nb.other);
        if (!other) continue;
        beams.push({
          start: hull.anchor,
          end: other.anchor,
          start3: anchor3(hull),
          end3: anchor3(other),
          weight: nb.weight,
        });
        badgeSpecs.push({ start: hull.anchor, end: other.anchor, text: formatCount(other.size) });
      }
      // the flare marks the anchor, it must not engulf the cluster
      const size = Math.min(Math.max(hullRadius(hull) * 0.6, this.mapExtent * 0.015), this.mapExtent * 0.06);
      this.flare.setTarget(hull.anchor[0], hull.anchor[1], size, clusterColor(sel.id));
    } else {
      const p = this.dataset.columns.pos2;
      const q = this.dataset.columns.pos3;
      const at2 = (i: number): [number, number] => [p[i * 2]!, p[i * 2 + 1]!];
      const at3 = (i: number): [number, number, number] => [q[i * 3]!, q[i * 3 + 1]!, q[i * 3 + 2]!];
      const start = at2(sel.id);
      const start3 = at3(sel.id);
      for (const nb of knnNeighbors(edges, sel.id)) {
        const end = at2(nb.id);
        beams.push({ start, end, start3, end3: at3(nb.id), weight: nb.sim });
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
    // flare + badges anchor in the flat map plane (screen-projected top-down)
    // — past the morph midpoint they'd sit visibly off the 3D cloud, so they
    // hide honestly while the beams keep gliding with the points
    const flat = this.morph <= 0.5;
    if (this.flare) this.flare.visible = on && flat;
    if (this.badges) this.badges.visible = on && flat;
  }

  /** Zoom onto one point's neighborhood (search-result click). The window is
   *  a fixed fraction of the map so nearby tokens stay in frame for context. */
  flyToPoint(id: number): void {
    if (!this.dataset) return;
    this.clearOrbitPivot(); // the fly-to tween owns the camera center now
    this.userDroveCamera = true;
    const fitPx = Math.min(this.cam.viewportW, this.cam.viewportH) * 0.55;
    if (this.morph > 0.02) {
      // mid-flythrough cam.cx/cy live in the pos3 xy frame — aim there, not at
      // pos2. The point sits at z = pos3[id*3+2]; the camera's lookAt targets
      // the z=0 plane at (cx, cy), so there's a vertical offset — accepted, the
      // orthographic projection keeps the neighborhood in frame.
      const q = this.dataset.columns.pos3;
      const x = q[id * 3];
      const y = q[id * 3 + 1];
      if (x === undefined || y === undefined) return;
      const wpp = Math.max((this.extent3 * 0.06) / fitPx, this.cam.minWpp);
      this.cam.flyTo(x, y, wpp, performance.now());
      return;
    }
    const p = this.dataset.columns.pos2;
    const x = p[id * 2];
    const y = p[id * 2 + 1];
    if (x === undefined || y === undefined) return;
    const wpp = Math.max((this.mapExtent * 0.06) / fitPx, this.cam.minWpp);
    this.cam.flyTo(x, y, wpp, performance.now());
  }

  /** Cinematic zoom onto one cluster (pill click / future keyboard nav). */
  flyToCluster(clusterId: number): void {
    this.clearOrbitPivot(); // the fly-to tween owns the camera center now
    const fitPx = Math.min(this.cam.viewportW, this.cam.viewportH) * 0.55;
    const centroid3 = this.centroid3ById.get(clusterId);
    if (this.morph > 0.02 && centroid3) {
      // mid-flythrough cam.cx/cy live in the pos3 xy frame — aim at the pos3
      // centroid, not the pos2 hull anchor. The centroid sits at z =
      // centroid3[2]; the camera's lookAt targets the z=0 plane at (cx, cy), so
      // there's a vertical offset — accepted, the orthographic projection keeps
      // the neighborhood in frame.
      this.userDroveCamera = true;
      const r3 = this.radius3ById.get(clusterId) ?? this.extent3 * 0.04;
      const wpp = Math.max((r3 * 2) / fitPx, this.cam.minWpp);
      this.cam.flyTo(centroid3[0], centroid3[1], wpp, performance.now());
      return;
    }
    const hull = this.hullsById.get(clusterId);
    if (!hull) return;
    this.userDroveCamera = true;
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
        if (e.pointerType === "touch") {
          this.touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
          // a second finger ends whatever the first one was doing and starts
          // the pinch/twist/tilt gesture from a fresh baseline
          if (this.touches.size >= 2) {
            this.pointerDown = null;
            this.lastPointer = null;
            this.dragging = false;
            this.orbiting = false;
            this.orbitLast = null;
            this.beginPinch();
            this.hoverClear();
            return;
          }
        }
        // middle (wheel-click) or right button → orbit the camera
        if (e.button === 1 || e.button === 2) {
          e.preventDefault();
          c.setPointerCapture(e.pointerId);
          this.orbiting = true;
          this.orbitLast = { x: e.clientX, y: e.clientY };
          // before hoverClear — the node under the cursor heads the pivot chain
          this.grabOrbitPivot();
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
        if (e.pointerType === "touch" && this.touches.has(e.pointerId)) {
          this.touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
          if (this.touches.size >= 2) {
            this.stepPinch();
            return;
          }
        }
        if (this.orbiting && this.orbitLast) {
          this.ensure3DForOrbit();
          this.orbitBy(
            (e.clientX - this.orbitLast.x) * ORBIT_AZ_SPEED,
            (e.clientY - this.orbitLast.y) * ORBIT_EL_SPEED,
          );
          this.orbitLast = { x: e.clientX, y: e.clientY };
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
            this.panScreen(e.clientX - this.lastPointer.x, e.clientY - this.lastPointer.y);
            this.lastPointer = { x: e.clientX, y: e.clientY };
            return;
          }
          this.lastPointer = { x: e.clientX, y: e.clientY };
        }
        // touch has no hover state — a finger down is a gesture, not a cursor
        if (e.pointerType === "touch") return;
        this.mouse = { x: e.clientX, y: e.clientY };
        this.hoverDirty = true;
      },
      opts,
    );

    // a cancelled pointer (OS gesture, browser scroll takeover, palm reject)
    // never delivers pointerup — without this the drag/pinch state sticks
    c.addEventListener(
      "pointercancel",
      (e) => {
        this.touches.delete(e.pointerId);
        // re-sample rather than keep a baseline that names a gone finger — a
        // stale dist/angle would jump the camera on the next move
        this.pinch = null;
        if (this.touches.size >= 2) this.beginPinch();
        this.pointerDown = null;
        this.lastPointer = null;
        this.dragging = false;
        this.orbiting = false;
        this.orbitLast = null;
        c.style.cursor = "";
      },
      opts,
    );

    c.addEventListener(
      "pointerup",
      (e) => {
        if (e.pointerType === "touch") {
          this.touches.delete(e.pointerId);
          if (this.pinch) {
            // lifting out of a pinch: re-baseline if a finger remains, and
            // never let the leftover contact fall through to the tap path
            this.pinch = null;
            if (this.touches.size >= 2) this.beginPinch();
            else {
              const [only] = [...this.touches.values()];
              if (only) {
                this.pointerDown = { x: only.x, y: only.y };
                this.lastPointer = { x: only.x, y: only.y };
                this.dragging = true; // continue as a pan, not a fresh tap
              }
            }
            return;
          }
        }
        if (this.orbiting) {
          this.orbiting = false;
          this.orbitLast = null;
          c.style.cursor = "";
          return;
        }
        const wasDrag = this.dragging;
        this.pointerDown = null;
        this.lastPointer = null;
        this.dragging = false;
        c.style.cursor = "";
        if (wasDrag) return;

        // click/tap: select the picked point's cluster (noise → point selection)
        if (this.morph > 0.5 && this.hoveredIndex === null) {
          // the flythrough resolves picks from the id-buffer result that *hover*
          // populates — and a tap never hovers, so it has to run its own pick or
          // every touch in 3-D would read as a deselect
          void this.selectAtAsync(e.clientX, e.clientY);
          return;
        }
        this.applyPick(this.pick(e.clientX, e.clientY));
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
        // normalize to pixels first — Firefox reports lines and a page-scroll
        // wheel reports pages, both of which read as a near-dead zoom otherwise
        const unit =
          e.deltaMode === 1 ? WHEEL_LINE_PX : e.deltaMode === 2 ? WHEEL_PAGE_PX : 1;
        const dx = e.deltaX * unit;
        const dy = e.deltaY * unit;
        // trackpad pinch arrives as ctrl+wheel with a much smaller delta, so it
        // needs its own gain to feel 1:1 with the fingers
        const pinching = e.ctrlKey;

        if (this.morph > 0.02 && !pinching) {
          // in 3-D a horizontal-dominant two-finger swipe orbits the azimuth,
          // and shift+swipe takes elevation — vertical stays zoom, which is the
          // one gesture a plain mouse wheel also has to serve
          if (Math.abs(dx) > Math.abs(dy)) {
            this.refreshWheelOrbitPivot();
            this.ensure3DForOrbit();
            this.orbitBy(dx * WHEEL_ORBIT_AZ, 0);
            return;
          }
          if (e.shiftKey) {
            this.refreshWheelOrbitPivot();
            this.ensure3DForOrbit();
            this.orbitBy(0, dy * WHEEL_ORBIT_EL);
            return;
          }
        }

        // zoom re-centers on its own cursor anchor — the orbit pivot yields
        this.clearOrbitPivot();
        // accumulate in log space, drained over ~120 ms in stepNavigation so
        // discrete wheel ticks read as one continuous glide
        const step = Math.max(
          -WHEEL_ZOOM_MAX,
          Math.min(dy * (pinching ? PINCH_ZOOM_GAIN : WHEEL_ZOOM_GAIN), WHEEL_ZOOM_MAX),
        );
        if (this.reducedMotion) {
          this.cam.zoomAt(e.clientX, e.clientY, Math.exp(step));
        } else {
          this.zoomPending += step;
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

  /** Turn a picked point into a store selection: its cluster, or the bare point
   *  when it's noise. Null clears — clicking empty space deselects. */
  private applyPick(picked: Selection | null): void {
    const store = appStore.getState();
    if (!picked) {
      store.setSelection(null);
      return;
    }
    const cid = this.dataset?.columns.clusterId[picked.id] ?? -1;
    store.setSelection(cid >= 0 ? { kind: "cluster", id: cid } : picked);
  }

  /** One id-buffer pick for a tap in the flythrough. Results that resolve after
   *  a dataset switch or a return to 2-D are dropped, matching updateHover3D. */
  private async selectAtAsync(x: number, y: number): Promise<void> {
    const picker = this.idPicker;
    const dataset = this.dataset;
    if (!picker || picker.broken || !dataset) {
      this.applyPick(null);
      return;
    }
    this.idPickBusy = true;
    try {
      const i = await picker.pick(this.camera, x, y);
      if (this.dataset !== dataset || this.morph <= 0.5) return;
      this.applyPick(i >= 0 && i < dataset.columns.count ? { kind: "point", id: i } : null);
    } finally {
      this.idPickBusy = false;
    }
  }

  /** Sample the two-finger baseline: separation, twist angle and midpoint. */
  private beginPinch(): void {
    const [a, b] = [...this.touches.values()];
    if (!a || !b) return;
    this.pinch = {
      dist: Math.max(Math.hypot(b.x - a.x, b.y - a.y), 1),
      angle: Math.atan2(b.y - a.y, b.x - a.x),
      cx: (a.x + b.x) / 2,
      cy: (a.y + b.y) / 2,
      twist: 0,
      twistOn: false,
    };
    this.zoomPending = 0; // don't let a stale wheel glide fight the fingers
  }

  /** Two-finger frame: separation → cursor-anchored zoom, twist → azimuth,
   *  midpoint drag → tilt (2-D has no tilt, so there the midpoint pans). This
   *  is the Maps convention, and it keeps one finger free for plain panning. */
  private stepPinch(): void {
    const prev = this.pinch;
    if (!prev) return;
    const [a, b] = [...this.touches.values()];
    if (!a || !b) return;

    const dist = Math.max(Math.hypot(b.x - a.x, b.y - a.y), 1);
    const angle = Math.atan2(b.y - a.y, b.x - a.x);
    const cx = (a.x + b.x) / 2;
    const cy = (a.y + b.y) / 2;

    // spreading the fingers must zoom *in*, i.e. shrink world-units-per-pixel.
    // Deliberately NOT morph-gated: the camera stays orthographic even in the
    // flythrough (see the OrthographicCamera at ~line 81), so frustum scaling
    // is the only meaningful zoom at any tilt — wheel zoom (~line 1060) is
    // already ungated the same way.
    if (Math.abs(dist - prev.dist) > 0.5) {
      this.cam.zoomAt(cx, cy, prev.dist / dist);
      this.userDroveCamera = true;
      this.cameraDirty = true;
    }

    const dMid = { x: cx - prev.cx, y: cy - prev.cy };
    // twist → azimuth. atan2 wraps at ±π; fold the delta back into (−π, π] so
    // crossing the seam can't fling the camera a full turn.
    let dAngle = angle - prev.angle;
    if (dAngle > Math.PI) dAngle -= Math.PI * 2;
    else if (dAngle < -Math.PI) dAngle += Math.PI * 2;
    // a pinch is never perfectly parallel, so rotation stays latched off until
    // the accumulated twist clears the deadzone — otherwise every zoom would
    // also drift the azimuth. Once engaged it stays engaged for the gesture.
    const twist = prev.twist + dAngle;
    const engaged = prev.twistOn || Math.abs(twist) > TOUCH_TWIST_DEADZONE;

    if (this.morph > 0.02) {
      // twist drives azimuth once engaged; a two-finger horizontal drag also
      // orbits azimuth directly (undiscoverable twist shouldn't be the only
      // way in), and both contributions sum onto the same angles. Re-capture
      // the anchor every step so the pinch-zoom's own re-centering (above)
      // folds into it instead of fighting the pivot compensation.
      if (!this.orbitPivot) this.grabOrbitPivot();
      else this.captureOrbitAnchor();
      const dAz = (engaged ? dAngle : 0) + dMid.x * TOUCH_ORBIT_AZ_SPEED;
      this.orbitBy(dAz, dMid.y * TOUCH_TILT_SPEED);
    } else if (dMid.x !== 0 || dMid.y !== 0) {
      this.panScreen(dMid.x, dMid.y);
    }

    this.pinch = { dist, angle, cx, cy, twist, twistOn: engaged };
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
    return (
      this.projectWorld(
        cols.pos2[i * 2]! * (1 - m) + cols.pos3[i * 3]! * m,
        cols.pos2[i * 2 + 1]! * (1 - m) + cols.pos3[i * 3 + 1]! * m,
        cols.pos3[i * 3 + 2]! * m,
      ) ?? [-9999, -9999] // clipped — park the tooltip far offscreen
    );
  }

  /** Project a morph-space world position through the render camera; null
   *  when it lands outside the near/far clip range (hide, don't mis-place).
   *  Label pills use this so they track clusters through the flythrough. */
  private projectWorld = (x: number, y: number, z: number): [number, number] | null => {
    const v = this.projScratch.set(x, y, z).project(this.camera);
    if (v.z < -1 || v.z > 1) return null;
    return [(v.x * 0.5 + 0.5) * this.cam.viewportW, (-v.y * 0.5 + 0.5) * this.cam.viewportH];
  };

  private hoverClear(): void {
    this.hoveredIndex = null;
    this.points?.setHover(null);
    this.tooltip?.hide();
    appStore.getState().setHover(null);
  }
}
