---
name: nebulai-viz
description: >-
  Rendering orchestrator for the Nebul.AI Phase-2 viewer (~/Developer/nebulai/
  viewer/). Use this skill whenever adding or changing a visualization in the
  viewer app: it decides which toolkit renders a given view (three.js
  WebGPU+TSL vs raw WGSL vs deck.gl vs HTML/SVG overlay), and holds the
  architecture laws every driver must obey (one driver per canvas, store-based
  camera sharing, layered DOM, fallback ladder). Route deeper toolkit questions
  to nebulai-viz-threejs / nebulai-viz-wgsl / nebulai-viz-deckgl; keep the
  nebulai hub skill loaded for the data contract and honesty guardrails.
---

# Nebul.AI viz — rendering orchestrator

The viewer (`viewer/`, Vite + TypeScript) is a multi-toolkit app: three.js
WebGPURenderer + TSL, raw WGSL, deck.gl, and plain HTML/SVG all coexist. This
skill is the router: given a new visualization, it picks the toolkit and holds
the contracts that keep the toolkits from colliding.

## Routing — which toolkit renders what

Condensed table; the full decision rationale is in
`references/routing-table.md`.

| The visualization needs… | Use | Skill |
|---|---|---|
| Instanced point clouds, morphs, bloom, **and a WebGL fallback** | three.js WebGPU + TSL | `nebulai-viz-threejs` |
| A bespoke pipeline where WebGPU-only is acceptable | raw WGSL | `nebulai-viz-wgsl` |
| Layered 2D/2.5D data layers with built-in picking (geo-ish, hierarchies) | deck.gl | `nebulai-viz-deckgl` |
| Text, pills, badges, tooltips, dotted connectors | HTML/SVG overlay first | this skill (§ overlays) |

Existing drivers: `AtlasDriver` (three-TSL — atlas + 2D↔3D morph),
`CompareDriver` (raw WGSL — cross-model compare). Chord (M6) is planned as
three 2.5D + SVG overlay; Hierarchy (M6) as deck.gl.

## Laws (violating any of these has bitten us already)

1. **One driver per canvas; toolkits never share a GPU context.** Each
   SceneDriver owns its `<canvas>` and its device/renderer. View switching is
   an opacity crossfade between canvases (300ms, 150ms reduced-motion), both
   drivers framing until `fadeUntil` passes — never a context handoff.
2. **Shared state, not shared context.** The `zustand/vanilla` store
   (`src/app/store.ts`) is the only coordination point: camera, dataset,
   viewMode, selection, hover, toggles, settings. three reads matrices,
   raw WGSL mirrors into its UBO, overlays project with the same matrices.
   Drivers subscribe to the store; they never import chrome, and chrome never
   imports drivers (commands go through `src/app/actions.ts`
   registerActions/requestX).
3. **Layered DOM**: `#stage` > canvases → `#overlay-svg` → `#overlay-html` →
   `#chrome`. Mode-scoped visibility via a stage class (e.g. `.mode-compare`
   hides atlas overlays in CSS) — don't unmount overlays per frame.
4. **Fallback ladder**: WebGPU → three `forceWebGL` → static. A WebGPU-only
   driver must be capability-gated in chrome (disabled option + hint), never a
   runtime crash. `?gpu=webgl|static` forces a rung for testing.
5. **Text is HTML first.** Persistent labels are collision-culled HTML pills
   (~200 max); SDF text only if >1K persistent labels ever become a real need.
6. **Honesty guardrails ride along** (from the `nebulai` hub): confidence →
   opacity, MetaLine always visible, edge weights computed in 10-D cluster
   space and labeled so, compare view is label space — say so in the UI.

## Driver + view-manager contract

Interface, store slices, crossfade choreography, overlay projection, and
picking tiers: `references/scene-driver-contract.md`. Read it before writing
a new driver — it is extracted from the two working drivers, not aspirational.

## Chrome and design review

Chrome components are Preact + @preact/signals (`src/chrome/`), styled by
`tokens.css` + `craft-tokens.css`. Any chrome change goes through the four
design skills (composable-product-design, design-process, interaction-craft,
visual-craft) and the component review gate (targets, keyboard path, focus,
radii, tabular-nums, empty states, reduced motion).

## Verification idioms (preview-panel testing)

- Occluded preview panels freeze rAF and throttle timers: drive frames
  synchronously (`while (performance.now()-t0 < N) driver.frame(...)`), stash
  async results in `window` globals, read them in a follow-up eval (promise
  continuations only run *between* evals).
- Fresh `window.__errs` console.error counter per page load; `window.__store`,
  `window.__driver`, `window.__compareDriver` are the debug handles.
- `?frozen=1` pins the time uniform for screenshot goldens.
