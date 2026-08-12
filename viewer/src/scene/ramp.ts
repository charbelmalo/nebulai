/** The shared RAMP sampler for the `three/webgpu` stack.
 *
 *  `viz/tokens.ts` owns the ramp itself (amber → orange → pink → magenta →
 *  violet) and can bake it to a 256×1 RGBA buffer. Every GPU driver that wanted
 *  to sample it was repeating the same four lines — build a `DataTexture`, set
 *  `needsUpdate`, sample it at `vec2(t, 0.5)`, remember to dispose it — which is
 *  four chances to drift from the canonical ramp and one chance to leak.
 *
 *  What this does NOT do is decide what `t` means. The ramp is a DATA channel:
 *  `t` has to be a real normalized quantity the caller computed, and the caller
 *  still owns saying so in its legend. Chrome uses ACCENT, never this. */

import * as THREE from "three/webgpu";
import { texture, vec2 } from "three/tsl";
import { rampTextureData } from "@psychix/viz/tokens";

let shared: THREE.DataTexture | null = null;

/** The process-wide 256×1 ramp texture.
 *
 *  Deliberately a never-disposed singleton. It is 1 KB, it is immutable, and
 *  it outlives any one driver — refcounting it across drivers that mount and
 *  unmount independently would buy nothing and would make one driver's
 *  `dispose()` able to blank another's colours. Renderers release their own
 *  per-texture GPU handles on `renderer.dispose()` regardless. */
export function rampTexture(): THREE.DataTexture {
  if (!shared) {
    shared = new THREE.DataTexture(rampTextureData(), 256, 1, THREE.RGBAFormat);
    shared.needsUpdate = true;
  }
  return shared;
}

/** TSL node: the ramp colour at `t`, as rgb.
 *
 *  `t` is a node, so it can be a uniform, an attribute, or anything computed in
 *  the shader — but it must land in [0,1]; the texture clamps rather than
 *  wrapping, so out-of-range values silently pin to an end stop instead of
 *  reading as wrong. Normalize before you get here, and normalize against a
 *  bound you can name in the legend. */
export function rampNode(t: Parameters<typeof vec2>[0]) {
  return texture(rampTexture(), vec2(t, 0.5)).rgb;
}
