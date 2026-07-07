---
name: nebulai-viz-threejs
description: >-
  three.js WebGPURenderer + TSL patterns for the Nebul.AI viewer
  (~/Developer/nebulai/viewer/, three r185). Use when writing or debugging a
  three-based SceneDriver: TSL node materials, the forceWebGL fallback rung,
  node-based post-processing/bloom, instanced points, GPU id-buffer picking,
  camera tweens. Every gotcha here was hit for real in the AtlasDriver or
  ChordDriver — check this before re-deriving. Load nebulai-viz first for the
  routing + driver contract.
---

# three.js WebGPU + TSL — field notes (r185)

The AtlasDriver (`viewer/src/scene/drivers/AtlasDriver.ts` and
`viewer/src/scene/layers/*`) is the reference implementation. These are the
traps and patterns, hardest-won first.

## Renderer + imports

- `WebGPURenderer` init is **async** (`await renderer.init()`), and so is the
  whole driver `init` — the app shell awaits it before first frame.
- Fallback rung: `new WebGPURenderer({ forceWebGL: true })` — TSL transpiles
  node materials to GLSL. Same scene graph, both rungs; test with
  `?gpu=webgl`.
- **Import from `three/webgpu` and `three/tsl` only.** Importing an addon
  that internally does a bare `import from "three"` pulls a **duplicate
  three core** into the bundle and breaks instanceof checks. If an addon does
  that, inline the needed piece instead.
- Post-processing on WebGPU is **node-based** (`PostProcessing` /
  `RenderPipeline` — the class was renamed across r18x releases, check the
  local three version), not `EffectComposer`. Bloom is a TSL bloom node with
  a threshold so only hot content (beams/halos/flare) blooms; quality tiers:
  full → half-res → off (reduced motion / webgl has no post at all — use
  additive blending so glow-ish content still reads).

## TSL

- **WebGPU `maxVertexBuffers` is 8, and it fails silently.** PlaneGeometry
  already binds position+normal+uv (3 buffers), so an instanced material gets
  ~5 more attribute buffers before `createRenderPipeline` rejects the layout.
  Three creates pipelines in an **async task**, so nothing throws at the draw
  call and a synchronous error scope around a frame catches nothing — the
  invalid pipeline just spams `uncapturederror` at submit and the mesh draws
  zero fragments. Worse, **TSL tree-shakes unreferenced attributes**, so every
  simplified diagnostic material drops enough buffers to fit the budget and
  compiles fine — only the full node graph fails, which makes bisection lie to
  you. Fix: pack per-instance data into vec4 attributes and swizzle in TSL
  (`aSeg.xy`/`aSeg.zw`, `aMeta.x/.y/.z/.w`) — see ChordDriver
  (`viewer/src/scene/drivers/ChordDriver.ts`), which packs six scalars/vec2s
  into two vec4s. To catch this class of bug directly, hook
  `device.createRenderPipeline`/`createRenderPipelineAsync` and wrap each call
  in `pushErrorScope("validation")`/`popErrorScope()` — that surfaces the real
  message ("Vertex buffer count (9) exceeds the maximum…").
- Live TSL debugging in a Vite preview eval: `await import("/@id/three/tsl")`
  resolves through the dev server, so you can build node graphs and swap
  materials on a running scene from the console. Two timing traps: GPU
  `uncapturederror` events and Preact renders flush **between** evals (counts
  read in eval N belong to eval N−1's frames), and async store transitions
  (e.g. `switchViewMode`) resolve after the eval returns.
- `smoothstep(a, b, x)` requires **increasing edges** (a < b) on the WebGPU
  backend — `smoothstep(1.0, 0.72, d)`-style reversed edges silently work on
  GLSL but not WGSL. Write `1 - smoothstep(0.72, 1.0, d)`.
- Morph = `positionNode: mix(pos2Attr, pos3Attr, uMorph)` with a single
  `uniform(0)` driven by an eased tween — no geometry rebuild.
- Share one 256×1 ramp `DataTexture` between points/beams/badges so colors
  can't drift from the CSS gradient (`tokens.ts` is unit-test-synced with
  `tokens.css`).

## Picking (id-buffer)

- Parallel scene with id-colored copies. **Share the position
  attribute/uniform nodes with the display mesh** — a separate uMorph
  uniform desyncs picks mid-morph.
- `readRenderTargetPixelsAsync` origin: **top-left on WebGPU,
  bottom-left (GL convention) on forceWebGL**. Branch on
  `renderer.backend.isWebGLBackend`. Verified empirically both rungs.
- RenderTarget `{depthBuffer: false}` is wrong for overlapping instances —
  keep depth so the topmost instance wins. Decode
  `px[0] + px[1]*256 + px[2]*65536 - 1` (0 = background).
- Throttle async picks to ~30Hz for hover; use kdbush over pos2 in 2D
  instead (exact and sync).

## Camera + interaction

- `userDroveCamera` pattern: any pointer/wheel input sets a flag that cancels
  in-flight programmatic tweens — never fight the user for the camera.
- Cinematic fly-to: cubic-eased tween of center+zoom; 2D↔3D morph syncs the
  uMorph tween with a camera tilt (38°) in one clock so nothing lags.

## Assorted traps

- Halo/annulus radii need clamping against cluster spread — a fixed
  world-radius halo looks absurd on tight clusters and invisible on spread
  ones.
- Instanced meshes with count 0 can stay `visible` — guard badge/overlay
  code on the instance count, not object visibility.
- Perf measurement in an occluded preview panel: rAF is frozen — measure p95
  with forced frames, and expect `window.__perf.p95FrameMs` from the rAF loop
  to read as garbage there.
