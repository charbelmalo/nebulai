---
name: nebulai-viz-wgsl
description: >-
  Raw WebGPU/WGSL patterns for the Nebul.AI viewer (~/Developer/nebulai/
  viewer/). Use when writing or debugging a hand-rolled WebGPU SceneDriver:
  instanced quad sprites, UBO layout/alignment, premultiplied alpha,
  multi-state morph attributes, DPR/resize, orbit cameras, and where CPU
  picking stops scaling. Distilled from the CompareDriver port of
  backend/viewer.py. Load nebulai-viz first for routing + the driver contract
  — raw WGSL drivers are WebGPU-only and must be capability-gated.
---

# Raw WGSL driver — field notes

Reference implementation: `viewer/src/scene/drivers/CompareDriver.ts`
(837-point cross-model compare; port of the deprecated
`src/nebulai/backend/viewer.py`).

## Device + canvas

- Request your **own adapter/device** — never share three's. One driver, one
  canvas, one device; `dispose()` destroys buffers and the device.
- Context config: `alphaMode: "premultiplied"`, format from
  `navigator.gpu.getPreferredCanvasFormat()`. Premultiplied blend =
  `{ srcFactor: "one", dstFactor: "one-minus-src-alpha" }` and the fragment
  shader multiplies rgb by alpha itself.
- TS lib.dom (5.x) ships the WebGPU *interfaces* but not the runtime constant
  namespaces (`GPUBufferUsage`, `GPUTextureUsage`) or the `"webgpu"`
  getContext overload — shim them in a local `.d.ts`
  (`scene/drivers/webgpu-globals.d.ts`) instead of pulling `@webgpu/types`
  (which duplicate-declares against lib.dom).

## Instanced quad sprites

- One 6-vertex quad vertex buffer (stepMode vertex) + one big instance buffer
  (stepMode instance). CompareDriver: **18 floats/instance** = 4 states ×
  vec3 position + color rgb + size + sourceIdx + sharedFlag.
- Billboard in view space: `viewPos.xy += quadCorner * size * pointScale`.
- Circular sprite: SDF disc in the fragment shader,
  `alpha = 1 - smoothstep(0.72, 1.0, d)` — WGSL smoothstep needs
  **increasing edges**.
- Multi-state morph: keep every layout state in the instance buffer; the
  vertex shader mixes `statePos(from)` → `statePos(to)` with a smoothstepped
  uniform `t` (900ms tween, 150ms under reduced motion). Retarget by setting
  `fromState = current interpolated state` and resetting the clock.

## UBO layout (the alignment trap)

- Layout structs by hand and check with an offset comment per field. mat4x4 =
  64 bytes each, then pack scalars into vec4-sized slots: CompareDriver's UBO
  is **192 bytes** = view(64) + proj(64) + params vec4 (fromState, toState,
  t, pointScale) + two vec4 visibility masks + flags vec4.
- vec3 aligns to 16 bytes — never lay a vec3 followed by a float you care
  about without accounting for padding. Prefer vec4 slots.
- Matrices are **column-major**; write perspective/lookAt helpers
  accordingly and test by projecting a known point on the CPU (the tooltip
  projection reuses exactly these matrices — if hover lands, the math is
  right).

## Resize / DPR

- Clamp `dpr ≤ 2`. On resize set canvas width/height (device px), destroy and
  recreate the depth texture (`depth24plus`), same size. Never recreate the
  pipeline.

## Picking + tooltips

- CPU projection hover (project every instance, nearest within threshold) is
  fine at ≤~1K points — CompareDriver does this at 837. It is **O(N) per
  pointermove** and was already the perf sin of the old standalone viewer at
  50K: past ~1K, build an id-buffer pass instead.
- Tooltip: project with only the populated perspective cells
  (`clipW = -viewZ`), place an absolutely-positioned div in `#overlay-html`,
  offset ~14px, clamp to stage.

## Gating

- Raw WGSL drivers are WebGPU-only by definition. The chrome option must be
  `disabled` with a hint on other tiers, and any deep link (`?view=...`) must
  re-check the tier — degrade to a disabled state, never a crash.
