# SceneDriver contract + view manager

Extracted from the two working drivers (`AtlasDriver`, `CompareDriver`).

## Interface (`src/scene/SceneDriver.ts`)

```ts
init(canvas, tier): Promise<void>   // async — WebGPU adapter/device request
frame(dt, t): void                  // one render; t pinned to 0 by ?frozen=1
resize(w, h, dpr): void             // CSS px + dpr; clamp dpr ≤ 2
pick(x, y): Promise<PickResult>     // CSS px, canvas-relative
snapshotForTransition(): Snapshot   // outgoing state for morph handoffs
dispose(): void                     // abort listeners, unsubscribe, destroy GPU objects
```

Rules:

- The driver subscribes to the store in `init` and mirrors the slices it
  cares about (toggles, settings, selection, its own view slice). It never
  writes chrome state; it only writes scene-facing store fields (hover,
  selection) in response to input on *its* canvas.
- `dispose` must be complete: `AbortController` on all listeners, store
  unsubscribe, buffer/texture/device destroy. The view manager may create
  drivers lazily but never leaks them.

## View manager (lives in `main.ts`)

- One `<canvas>` per driver inside `#stage`; `#stage > canvas` is positioned
  absolute, full-bleed. Secondary canvases are created lazily on first switch.
- Crossfade: `FADE_MS = reducedMotion ? 150 : 300`; both canvases get
  `transition: opacity FADE_MSms ease`; swap `opacity` + `pointer-events`;
  set `fadeUntil = now + FADE_MS + 120` and frame **both** drivers until then:

```ts
const fading = now < fadeUntil;
if (activeMode !== "compare" || fading) atlas.frame(dt, t);
if ((activeMode === "compare" || fading) && compare) compare.frame(dt, t);
```

- Stamp a mode class on the stage (`#stage.mode-compare`) and scope overlay
  visibility in CSS — atlas pills/badges/tooltip hide in compare, the compare
  tooltip hides outside it.
- Deep links: `?view=<mode>` switches after boot; keep it capability-gated
  the same way the chrome control is.

## Store slices a driver touches

- Read: `dataset` (typed-array columns), `toggles`, `settings`
  (pointScale, confidenceFloor, bloom), `dims`/`morphT`, its view slice
  (e.g. `compare: {state, hiddenModels, sharedOnly}`).
- Write: `hover`, `selection` (from picking), nothing else.
- Chrome → driver commands that aren't state (e.g. "switch view") go through
  `actions.ts` `registerActions`/`requestX`, handled in `main.ts`.

## Picking tiers

| Scale | Technique |
|---|---|
| 2D, any N | kdbush static index over pos2 (exact, ~µs) |
| 3D, >1K | GPU id-buffer: parallel id-colored scene → 1×1 readback at ~30Hz |
| ≤1K pts | CPU projection loop is fine (CompareDriver hover) |

Id-buffer traps: the id mesh must share position attribute/uniform *nodes*
with the display mesh or morphs desync; `readRenderTargetPixelsAsync` origin
is **top-left on WebGPU, bottom-left on the WebGL rung** — branch on
`renderer.backend.isWebGLBackend`; decode `px[0] + px[1]*256 + px[2]*65536 - 1`.

## Overlay projection

HTML/SVG overlays project world→CSS px with the same matrices the driver
renders with (from the store / driver camera). Anchor with
`transform: translate(xpx, ypx)`, clamp to the stage, offset tooltips ~14px.
Never lerp overlay positions separately from the camera — reproject per frame.
