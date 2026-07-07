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

export interface BloomPipeline {
  post: THREE.RenderPipeline;
  dispose(): void;
}

export function createBloomPipeline(
  renderer: THREE.WebGPURenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
  tier: Exclude<BloomTier, "off">,
): BloomPipeline {
  const post = new THREE.RenderPipeline(renderer);
  const scenePass = pass(scene, camera);
  const color = scenePass.getTextureNode("output");
  const bloomNode = bloom(color, STRENGTH, RADIUS, THRESHOLD);
  if (tier === "half") bloomNode.setResolutionScale(0.5);
  post.outputNode = color.add(bloomNode);
  return {
    post,
    dispose() {
      post.dispose();
    },
  };
}
