# Toolkit routing — full decision table

Decide per *visualization*, not per app. The question is always: what does
this view need that the cheapest tier can't give?

## Decision order

1. **Can HTML/SVG do it?** Labels, badges, tooltips, legends, dotted
   connectors, small (<~500 element) diagrams. If yes, stop — overlays are
   free to style, accessible, and testable with the DOM tools. The atlas label
   pills, beam count badges, and the planned chord *labels* live here.
2. **Does it need a WebGL fallback?** If the view must work on the `webgl`
   rung (anything load-bearing for all users), use **three.js WebGPU + TSL** —
   TSL transpiles to GLSL under `forceWebGL`, so one material tree serves both
   rungs. This is why the atlas (points, territories, beams, halos, bloom) is
   three-TSL.
3. **Is the pipeline bespoke and WebGPU-only acceptable?** Compute passes,
   exotic vertex pulling, multi-state morph attributes, tight UBO control —
   **raw WGSL**. Cheaper than fighting a framework, but you own everything
   (resize, DPR, depth, input, tooltips). The CompareDriver is the template;
   it's capability-gated so the WebGL rung shows a disabled option, not a
   crash.
4. **Is it layered 2D/2.5D data with picking?** Hierarchies, geo-style maps,
   many independent layers — **deck.gl**. Built-in picking and layer
   diffing beat hand-rolling; the cost is bundle weight (lazy-load it) and a
   custom Viewport to follow our camera.

## What each toolkit must never do

- three: never `new WebGLRenderer` — always WebGPURenderer (+`forceWebGL`
  rung). Never a second renderer on the same canvas.
- raw WGSL: never share a device with three; request its own adapter/device.
- deck.gl: never own the camera — it follows the store via an external
  Viewport. Never mount on a canvas another driver owns.
- overlays: never read GPU state — they project through the same
  camera matrices the drivers use, from the store.

## Cost cheat-sheet

| | three-TSL | raw WGSL | deck.gl | HTML/SVG |
|---|---|---|---|---|
| WebGL fallback | ✅ transpiles | ❌ gate it | ✅ | ✅ |
| Bundle cost | ~160KB (already paid) | ~0 | ~300KB (lazy) | 0 |
| Picking | id-buffer (build it once) | CPU ≤1K pts, else id-buffer | built-in | DOM events |
| Post FX (bloom) | ✅ node-based | build your own | ❌ practical | ❌ |
| Text | ❌ use overlay | ❌ use overlay | limited | ✅ native |
