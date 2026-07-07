---
name: nebulai-viz-deckgl
description: >-
  deck.gl / regl integration patterns for the Nebul.AI viewer
  (~/Developer/nebulai/viewer/). Use when building or changing a deck.gl-based
  SceneDriver (shipped: HierarchyDriver, the radial dendrogram view): when deck
  wins over three/WGSL,
  driving deck from the shared store camera via an external Viewport, the
  custom-layer/regl escape hatch, lazy-loading, and the interop rules that keep
  deck from colliding with the other toolkits. Load nebulai-viz first for the
  routing + driver contract.
---

# deck.gl in the multi-toolkit viewer

The HierarchyDriver (`viewer/src/scene/drivers/HierarchyDriver.ts`, deck 9.3)
is the reference implementation: a radial single-linkage dendrogram
(PathLayer elbows + two ScatterplotLayers) with deck picking behind the
app's own pointer handlers. Field notes from building it are at the bottom.

## When deck wins

- Many independent, declaratively-diffed **data layers** (scatter, lines,
  polygons, text) where deck's layer diffing replaces hand-written buffer
  management.
- **Built-in picking** — deck's pick maps are free; three/WGSL make you build
  an id-buffer.
- 2.5D "layered map" looks: hierarchy networks, geo-ish projections,
  aggregation layers.

deck loses when you need custom post FX (bloom), TSL-style shared shader
logic, or the WebGL/WebGPU dual rung with one material tree — that's
three-TSL territory.

## Integration rules (from the nebulai-viz laws)

1. **Own canvas.** `new Deck({ canvas })` on a canvas the view manager
   creates; never share three's or a WGSL driver's canvas/context.
2. **Camera follows the store.** deck must not own interaction state: build a
   custom/external `Viewport` (or OrbitView with controller disabled) from
   the same matrices the rest of the app uses, updated per frame from the
   zustand store. Input handlers write to the store; deck re-renders from it.
   Otherwise overlay pills/tooltips drift from the GPU layers.
3. **Lazy-load.** deck is ~300KB gz — `import()` it inside the driver's
   `init` so the atlas bundle stays under the 500KB budget. The chrome option
   stays enabled; first switch shows the loading pill.
4. **SceneDriver wrapper.** Wrap Deck in the standard interface
   (init/frame/resize/pick/dispose); `frame()` calls
   `deck.setProps({viewState})` + redraw; `dispose()` calls `deck.finalize()`.
5. **Reduced motion + `?frozen=1`** still apply — transitions off, time-based
   layer props pinned.

## Custom layer / regl escape hatch

- Prefer composing built-in layers; write a custom `Layer` (deck's own
  shader assembly) only when a built-in prop can't express it.
- Drop to regl only for a fully bespoke effect that still wants layer-diffing
  around it — at that point reconsider raw WGSL (nebulai-viz-wgsl) if
  WebGPU-only is acceptable: fewer abstraction layers, no GL-only lock-in.

## Known interop warnings

- deck is WebGL2 today: it always works on the `webgl` rung — good (no
  gating needed), but it will never get WebGPU bloom; design hierarchy
  visuals to not depend on post FX.
- deck's default `getCursor`/DOM event handlers grab the canvas — disable
  what the shared input model already handles (Esc deselect lives app-wide).
- Text layers are GPU-rendered; our law says persistent text is HTML pills —
  use deck TextLayer only for dense transient annotation, if ever.

## Field notes from the HierarchyDriver (deck 9.3, verified)

- **Typed viewState is keyed by view id.** With `Deck<OrthographicView[]>`,
  `viewState` must be `{ ortho: { target, zoom } }`, not a bare
  `{ target, zoom }` — the flat form type-errors (and TS is right; keyed is
  the multi-view-safe form). `zoom` is log2(px per world unit):
  `Math.log2(shortAxisPx / (2 * worldHalfExtent))` fits a world radius to
  the stage exactly like the three drivers' ortho frustum math.
- **`OrthographicView({ flipY: false })`** gives y-up world coords matching
  the three/WGSL drivers, so label projection math (`sy = cy − y/wpp`) is
  shared verbatim with ChordDriver.
- **Passive deck really works**: `controller: false`, no
  `onHover`/`onClick` props; our own canvas pointer handlers call
  `deck.pickObject({ x, y, radius: 7 })` and write to the zustand store.
  Deck never touches the cursor or swallows Esc.
- **Lazy-load pattern**: `await import("@deck.gl/core")` inside
  `init()` + *type-only* top-level imports (`import type { Deck }`) —
  type-only imports pull zero code, and Vite splits deck into its own
  chunks (~221KB gz) loaded on first view switch. Verified in the build
  output; main bundle stayed at 277KB gz.
- **Highlight = layer rebuild + `updateTriggers`.** Accessors close over a
  focus Set; bump a version counter in `updateTriggers: { getColor: v }`
  and call `deck.setProps({ layers: [...] })`. Layer construction is cheap
  (deck diffs); don't mutate buffers like the three drivers do.
- **`frame()` is a no-op** — deck renders on demand when props change. That
  makes `?frozen=1` goldens trivial, but it also means nothing time-based
  belongs in a deck driver (there's no post FX on this rung anyway).
- **deck stamps `deck-widget-container` on the canvas's parent** (#stage).
  Harmless, but don't write CSS/tests that assume the stage class list is
  exactly your mode classes.
- **`useDevicePixels` accepts a number** — pass `Math.min(dpr, 2)` to match
  the other drivers' DPR clamp.
- Honest dendrogram recipe: Kruskal over the exported `cluster_edges`
  sorted by descending weight + union-find *is* single-linkage
  agglomeration over the top-k graph — no client-side distance recompute,
  weights stay in the export's stamped space. A disconnected forest gets a
  synthetic root at weight 0 whose tooltip says the components are unlinked
  in the exported top-k edges.
