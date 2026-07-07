/** The one law of this app: exactly one SceneDriver owns #scene-canvas at a
 *  time, and toolkits never share a GPU context. Drivers coordinate through
 *  the store and the shared camera, nothing else. */

import type { GpuTier } from "../app/capabilities";
import type { Selection } from "../app/store";
import type { Dataset } from "../data/loader";

export interface SceneDriver {
  /** Async because WebGPURenderer.init() is async. Must be called once. */
  init(canvas: HTMLCanvasElement, tier: GpuTier): Promise<void>;
  setDataset(ds: Dataset): void;
  /** dt in ms, t = elapsed seconds (pinned when ?frozen=1 for goldens). */
  frame(dt: number, t: number): void;
  resize(width: number, height: number, dpr: number): void;
  /** Screen-px → what's under the cursor, or null. */
  pick(x: number, y: number): Selection | null;
  /** For view-morph crossfades (M4/M6). */
  snapshotForTransition(): HTMLCanvasElement | null;
  dispose(): void;
}
