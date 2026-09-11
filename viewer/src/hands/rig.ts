/** rig.ts — the bridge between the hand tracker and the running atlas.
 *
 *  Everything below this file deals in hands; everything above it deals in the
 *  store and the driver. This is the only module that knows both, and it is
 *  deliberately the only one: `channels.ts` maps gestures onto abstract channels
 *  without importing three.js or zustand, `tracking.ts` runs the camera without
 *  importing either, and the driver exposes a small offset API without knowing a
 *  hand exists. Cut this file out and the other three still compile.
 *
 *  Two rules shape what the bridge is allowed to do.
 *
 *  **The rig composes, it never replaces.** Every camera channel reaches the
 *  driver as an offset added to whatever the pointer path already did, so the
 *  mouse never loses authority mid-gesture and dropping every hand provably
 *  returns the rig's contribution to zero. The two latched channels — pan and
 *  zoom — are differentiated here and handed to the driver as deltas, because
 *  `panScreen` and `zoomAt` already own every clamp and every world conversion
 *  and a second copy of that arithmetic in this file would drift from them.
 *
 *  **A gesture may not write a setting.** This used to be the opposite rule —
 *  the bridge was careful to flip the *real* `toggles.territories` and
 *  `appearance.atlas.orbitEnabled` so the Settings page kept showing the truth —
 *  and being careful about it was answering the wrong question. Both of those
 *  already have a switch the operator can see, reach, and undo; putting a second
 *  one on a probabilistic hand classifier meant a misread pose silently changed
 *  what the map *was* while they were reading it, and the honesty of the
 *  checkbox afterwards did not help. The same argument retired the two
 *  cluster-stepping casts and the camera reset. What is left here writes the
 *  selection — which is what a lasso and a dwell are *for*, and which the
 *  operator asked for by aiming at something — and nothing else.
 */

import { requestFlyToCluster } from "../app/actions";
import { appStore } from "../app/store";
import type { AtlasDriver } from "../scene/drivers/AtlasDriver";
import type { HandChannelOutput } from "./channels";
import { pointInLasso, type Lasso, type StrokePoint } from "./strokes";
import { HandTracker, type HandTrackerStatus } from "./tracking";
import {
  EMPTY_HAND_DIAGNOSTICS,
  handControlAvailable,
  type HandTrackingDiagnostics,
  type HandTrackingPhase,
  type SpellEvent,
  type SpellId,
} from "./types";

/** Minimum points inside a lasso before it is read as a selection. */
const LASSO_MIN_POINTS = 3;

/**
 * Radius of the disc a dwelt-on waypoint tests, in mirrored viewport widths.
 *
 * Roughly a fingertip's worth of aiming slop. Small enough that pointing between
 * two clusters resolves to neither rather than to whichever happens to be
 * denser, which is the failure that makes a dwell control feel arbitrary.
 */
const WAYPOINT_RADIUS = 0.05;

export interface HandCastAnnouncement {
  id: SpellId;
  /** Monotonic, so a repeat of the same spell still reads as a new event. */
  seq: number;
}

export interface HandRigState {
  phase: HandTrackingPhase;
  error: string | null;
  diagnostics: HandTrackingDiagnostics;
  /** The most recent cast, for the HUD's live region. */
  cast: HandCastAnnouncement | null;
}

const IDLE_STATE: HandRigState = {
  phase: "idle",
  error: null,
  diagnostics: EMPTY_HAND_DIAGNOSTICS,
  cast: null,
};

/**
 * The rig singleton.
 *
 * One per document, because there is one camera and one webcam. It is created
 * eagerly — the object is a few fields — but the `HandTracker` underneath it,
 * and with it the 152 KB MediaPipe bundle, the 11 MB WASM runtime and the 8 MB
 * gesture model, is not constructed until something calls `enable()`.
 */
class HandRig {
  private tracker: HandTracker | null = null;
  private target: AtlasDriver | null = null;
  private listeners = new Set<(state: HandRigState) => void>();
  private state: HandRigState = IDLE_STATE;
  private castSeq = 0;

  /** Previous frame's latched channels, for differentiation. */
  private lastPanX = 0;
  private lastPanY = 0;
  private lastZoom = 0;

  private watching = false;
  private paused = false;

  get available(): boolean {
    return handControlAvailable();
  }

  get snapshot(): HandRigState {
    return this.state;
  }

  /** The tracker, or null while the rig has never been enabled. */
  get session(): HandTracker | null {
    return this.tracker;
  }

  subscribe(listener: (state: HandRigState) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private publish(next: HandRigState): void {
    this.state = next;
    for (const listener of this.listeners) listener(next);
  }

  /**
   * Points the rig at a driver, or at nothing.
   *
   * Called by the entry module once the atlas is up, and again with `null` when
   * a driver is torn down. Detaching returns the outgoing driver's offsets to
   * neutral rather than leaving them wherever the last frame put them: a driver
   * that is disposed mid-gesture would otherwise be recreated carrying a stale
   * point gain or a frozen shockwave nobody can see the cause of.
   */
  setTarget(target: AtlasDriver | null): void {
    if (this.target === target) return;
    const previous = this.target;
    this.target = target;
    if (previous) {
      previous.handPointGain(1);
      previous.handPulse(0, 0);
    }
    this.lastPanX = 0;
    this.lastPanY = 0;
    this.lastZoom = 0;
  }

  private ensureTracker(): HandTracker {
    const existing = this.tracker;
    if (existing) return existing;

    const settings = appStore.getState().settings;
    const tracker = new HandTracker({
      effects: settings.handEffects,
      onStep: (output) => {
        this.apply(output);
      },
      onSpell: (spell) => {
        this.cast(spell);
      },
      onLasso: (lasso) => {
        this.select(lasso);
      },
      onWaypoint: (point) => {
        this.flyTo(point);
      },
      onStatus: (status) => {
        this.report(status);
      },
    });
    this.tracker = tracker;
    tracker.setPaused(this.paused);

    appStore.subscribe((store, previous) => {
      if (store.settings.handEffects !== previous.settings.handEffects) {
        tracker.setEffectsEnabled(store.settings.handEffects);
      }
    });

    return tracker;
  }

  async enable(): Promise<void> {
    await this.ensureTracker().enable();
  }

  disable(): void {
    this.tracker?.disable();
  }

  /**
   * Suspends inference without releasing the camera.
   *
   * Remembered on the rig rather than only forwarded, because the tracker may
   * not exist yet: the atlas can be off-screen long before anyone turns the
   * toggle on, and a tracker constructed after that has to be born paused.
   */
  setPaused(paused: boolean): void {
    this.paused = paused;
    this.tracker?.setPaused(paused);
  }

  /**
   * Follows the Settings toggle and the current page. Called once by the entry
   * module, after `setTarget`.
   *
   * Kept out of module scope so importing this file — which the Settings page
   * does, for the availability probe — cannot start a camera. Nothing here
   * constructs the tracker until the toggle is actually on.
   *
   * The pause arm is not an optimisation detail: inference against a page the
   * operator has navigated away from is a running camera light with nothing
   * responding to it, which is the single most alarming thing a webcam feature
   * can do. It stops at the same moment the atlas stops being visible.
   */
  watchSettings(): void {
    if (this.watching) return;
    this.watching = true;

    const sync = (on: boolean): void => {
      if (!on) {
        this.disable();
        return;
      }
      this.enable().catch((cause: unknown) => {
        console.error("[nebulai] hand control failed to start", cause);
      });
    };

    const store = appStore.getState();
    let enabled = store.settings.handTracking;
    let visible = store.page === "map" && store.viewMode === "atlas";

    appStore.subscribe((next) => {
      const nextVisible = next.page === "map" && next.viewMode === "atlas";
      if (nextVisible !== visible) {
        visible = nextVisible;
        this.setPaused(!visible);
      }
      if (next.settings.handTracking !== enabled) {
        enabled = next.settings.handTracking;
        sync(enabled);
      }
    });

    if (!visible) this.setPaused(true);
    if (enabled) sync(true);
  }

  private report(status: HandTrackerStatus): void {
    this.publish({
      phase: status.phase,
      error: status.error,
      diagnostics: status.diagnostics,
      cast: this.state.cast,
    });
  }

  // ── per-frame application ────────────────────────────────────────────────

  /**
   * Writes one integration step onto the driver.
   *
   * The early return on `engaged` is what makes the whole rig free when it is
   * not in use: with no hand raised and everything settled, this touches no
   * uniform and marks nothing dirty, so a session that never enables hand
   * tracking renders through the identical path it did before the rig existed.
   */
  private apply(output: HandChannelOutput): void {
    const driver = this.target;
    if (!driver) return;

    if (!output.engaged) {
      // Keep the differentiator's baseline current so the first engaged frame
      // after a lull emits the delta since *that* frame rather than replaying
      // every step taken while disengaged.
      this.lastPanX = output.panX;
      this.lastPanY = output.panY;
      this.lastZoom = output.zoom;
      return;
    }

    driver.handPointGain(output.pointGain);

    // Pan arrives as a cumulative offset in viewport widths so the spring has
    // something to settle toward; the driver wants a per-frame pixel delta.
    // Both axes scale by the viewport *width* because that is the unit the
    // channel is declared in — using the height for y would make a diagonal
    // hand sweep bend on a non-square viewport.
    const { width } = driver.viewportSize();
    const dx = (output.panX - this.lastPanX) * width;
    const dy = (output.panY - this.lastPanY) * width;
    this.lastPanX = output.panX;
    this.lastPanY = output.panY;
    if (dx !== 0 || dy !== 0) driver.handPan(dx, dy);

    // Zoom is a log offset, so the frame's factor is the exponential of the
    // difference. Differencing in log space rather than in linear space is what
    // makes a steady push produce a steady *ratio* per second — the way a wheel
    // behaves — instead of accelerating as it zooms in.
    const dz = output.zoom - this.lastZoom;
    this.lastZoom = output.zoom;
    if (dz !== 0) driver.handZoom(Math.exp(dz));

    driver.handPulse(output.pulse.radius, output.pulse.amplitude);
  }

  // ── discrete casts ───────────────────────────────────────────────────────

  /**
   * Announces a cast, and anchors the one that needs anchoring.
   *
   * The switch used to have six arms and four of them moved the camera or wrote
   * a setting. Both survivors are purely visual, so this method now does almost
   * nothing — which is the point, and is why `onSpell` is documented as an
   * announcement hook rather than an action hook. If a future arm here needs the
   * store or the driver's camera, that is the signal it should not be a gesture.
   */
  private cast(spell: SpellEvent): void {
    if (spell.id === "shockwave") {
      // Anchor the wave before the first `handPulse` of the next step, so the
      // front starts from where the hand was rather than from where the camera
      // happens to be pointing a frame later.
      this.target?.handShockwave(spell.origin.x, spell.origin.y);
    }
    // `snap` is entirely a channel-layer effect: the flash rides the point gain,
    // which `apply` is already writing every step.

    this.castSeq += 1;
    this.publish({ ...this.state, cast: { id: spell.id, seq: this.castSeq } });
  }

  /**
   * Resolves an air-drawn lasso into a selection.
   *
   * The atlas selects one cluster or one point, not an arbitrary set, so the
   * honest reading of a loop is "the cluster this loop is mostly around".
   */
  private select(lasso: Lasso): void {
    const best = this.clusterWithin((x, y) => pointInLasso(lasso, x, y), LASSO_MIN_POINTS);
    if (best === null) {
      // An empty loop clears the selection: drawing a circle round nothing is a
      // deliberate act, and the only thing it can mean is "none of these".
      appStore.getState().setSelection(null);
      return;
    }
    appStore.getState().setSelection({ kind: "cluster", id: best });
    requestFlyToCluster(best);
  }

  /**
   * Flies to whatever a dwelt-on fingertip was resting on.
   *
   * Unlike a lasso, an empty result here does **not** clear the selection. A
   * loop drawn round nothing took effort and can only have been meant; a finger
   * resting over empty space is what a raised hand looks like when the operator
   * is thinking, and wiping their selection for it would punish holding still —
   * the exact posture this mode asks them to adopt. One point is enough to
   * commit, because pointing at a thing is already the aiming step.
   */
  private flyTo(point: StrokePoint): void {
    const best = this.clusterWithin(
      (x, y) => Math.hypot(x - point.x, y - point.y) <= WAYPOINT_RADIUS,
      1,
    );
    if (best === null) return;
    appStore.getState().setSelection({ kind: "cluster", id: best });
    requestFlyToCluster(best);
  }

  /**
   * The cluster most of the points in a region belong to, or null.
   *
   * Noise points are counted but cannot win: they have no hull to fly to and no
   * legend entry, so selecting "noise" would be a selection the rest of the UI
   * cannot show.
   */
  private clusterWithin(
    contains: (x: number, y: number) => boolean,
    minimum: number,
  ): number | null {
    const driver = this.target;
    if (!driver) return null;

    const inside = driver.handLassoPick(contains);
    if (inside.length < minimum) return null;

    const columns = appStore.getState().dataset?.columns;
    if (!columns) return null;

    const tally = new Map<number, number>();
    for (const index of inside) {
      const cluster = columns.clusterId[index];
      if (cluster === undefined || cluster < 0) continue;
      tally.set(cluster, (tally.get(cluster) ?? 0) + 1);
    }

    let best: number | null = null;
    let bestCount = 0;
    for (const [cluster, count] of tally) {
      if (count > bestCount) {
        best = cluster;
        bestCount = count;
      }
    }

    return best;
  }
}

export const handRig = new HandRig();
