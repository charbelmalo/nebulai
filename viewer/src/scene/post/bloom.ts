/** Bloom post pipeline (three RenderPipeline + TSL bloom node — NOT the old
 *  EffectComposer). Threshold is set so only the brightest additive content
 *  blooms: the selection flare core and hot beam sources — never the base
 *  point cloud. Tiers: full (webgpu), half (half-res mips), off. */

import * as THREE from "three/webgpu";
import { pass } from "three/tsl";
import { bloom } from "three/addons/tsl/display/BloomNode.js";

export type BloomTier = "full" | "half" | "off";

const STRENGTH = 0.85;
const RADIUS = 0.4;
const THRESHOLD = 0.55;

/** Per-view overrides.
 *
 *  These are NOT a taste knob — they exist because bloom's effect depends on
 *  how densely the hot content is packed, which is a property of the scene, not
 *  of the effect. The atlas and the sessions flight path are sparse, so
 *  overlapping halos stay coloured. The compare field packs 837 concepts into a
 *  tight blob, and at the default threshold the summed halos saturated all
 *  three channels: measured 84% of lit pixels below 0.15 saturation, i.e. the
 *  model-identity hue — the entire point of that view — was gone. A view whose
 *  content is dense must raise the threshold so only true spikes bloom. */
export interface BloomOptions {
  strength?: number;
  radius?: number;
  threshold?: number;
}

export interface BloomPipeline {
  post: THREE.RenderPipeline;
  dispose(): void;
}

export function createBloomPipeline(
  renderer: THREE.WebGPURenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
  tier: Exclude<BloomTier, "off">,
  opts: BloomOptions = {},
): BloomPipeline {
  const post = new THREE.RenderPipeline(renderer);
  const scenePass = pass(scene, camera);
  const color = scenePass.getTextureNode("output");
  const bloomNode = bloom(
    color,
    opts.strength ?? STRENGTH,
    opts.radius ?? RADIUS,
    opts.threshold ?? THRESHOLD,
  );
  if (tier === "half") bloomNode.setResolutionScale(0.5);
  post.outputNode = color.add(bloomNode);
  return {
    post,
    dispose() {
      post.dispose();
    },
  };
}
