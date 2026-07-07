# Parity checklist — nebulai viewer vs. reference video

Reference: `nebulai.mp4` ("syncatlas", Fuselab Creative, 23s), frames
`f01–f23.jpg` (extracted at plan time). Key frames: **f04/f08** chord view,
**f15/f18/f21** map view, **f02–f06** hero beam sequence, **f10–f13** sidebar
chrome.

Scoring: **0** = absent, **1** = present but visibly different, **2** = matches
the reference's intent (not pixel-identical — same anatomy, motion, and tone).
Re-score at every milestone review. **Gate: M2 exit requires ≥80% on map-view
rows** (score ÷ max on rows marked `map`).

Honesty note: some rows *intentionally* diverge from the video where the video
would violate the project's guardrails (e.g. our beam badges show raw 10-D
similarities, never display-space distances; noise stays visible). Those rows
score against the visual anatomy only — the data semantics are ours.

## M2 scoring — 2026-07-07

| # | Feature (video evidence) | View | Score | Notes |
|---|---|---|---|---|
| 1 | Dark plum bg + radial vignette (all frames) | map | 2 | `--bg #150f17`, CSS radial vignette; verified via inspect |
| 2 | Dark raised territories with hairline coasts (f15, f18) | map | 2 | merged hull fills @ 0.85, 1px `--hairline` strokes, 2 draw calls |
| 3 | Point cloud: small glowing nodes, dim noise (f15, f21) | map | 2 | additive TSL points, confidence→opacity, noise desaturated/smaller |
| 4 | Cluster label pills: dark pill + dot + name (f15, f18) | map | 2 | HTML pills, collision-culled, zoom-band faded, `--bg-pill` + ramp dot |
| 5 | Hub halo: pulsing ring around major nodes (f18, f21) | map | 2 | 8 hub halos, TSL ring pulse; radius clamped for spread clusters |
| 6 | Beam gradient: yellow→orange→magenta→purple ramp along beam (f03–f05) | map | 2 | shared 256×1 ramp texture, t-mapped colorNode |
| 7 | Beam anatomy: thick at source, tapering, arcing (f03, f05) | map | 2 | 14px→3.5px taper, sin bow (6% of length) |
| 8 | Weak links dotted, strong links solid ribbons (f04, f15) | map | 2 | weight-gated dash mask (`smoothstep(0.45,0.85)`), dots scroll src→dst |
| 9 | Rotated count badges on beams ("4.9K", "25K") (f03, f05) | map | 2 | pooled HTML pills rotated to beam angle, tabular-nums, formatCount |
| 10 | Selection "sun": bloom flare + particle sparkle (f02, f06) | map | 2 | additive sprite flare + sparkle particles + TSL bloom (webgpu) |
| 11 | Bloom on hot content only, not the base cloud (f02–f06) | map | 2 | threshold 0.55 via RenderPipeline; tiers full/half/off; webgl = no post |
| 12 | Cinematic eased camera fly-to on selection (f02→f06) | map | 2 | camera2d flyTo tween, cubic ease; `?frozen` pins motion |
| 13 | Beam energy: motion along the links (f04→f05) | map | 1 | pulse + scrolling dots present; video's per-particle "comet" heads not replicated |
| 14 | Label density at overview matches video's sparse look (f15) | map | 1 | greedy collision culls to ~40 pills; video hand-curates fewer, larger labels |
| 15 | Top bar: logo mark + wordmark (f01, f10) | chrome | 2 | TopBar with mark + "nebul.ai" wordmark |
| 16 | Sidebar: Settings/Additional tabs (f10–f12) | chrome | 2 | Tabs component, keyboard nav |
| 17 | Sidebar select rows (Title/Type/Size/Dataset/Dimensions) (f11) | chrome | 2 | Dataset/Type/Dimensions selects; row anatomy matches (label left, control right) |
| 18 | Sidebar toggle rows (Active/Link/Legend…) (f12) | chrome | 2 | 5 toggles (Territories/Labels/Beams/Noise/Legend), video-style pill switches |
| 19 | Floating collapsible legend card, gradient bar + ticks (f13) | chrome | 2 | LegendCard: "Connections" ramp bar 0–1 ticks, collapsible |
| 20 | Legend radio group Hairball/Chord/Hierarchical (f13) | chrome | 2 | M6b: all three radios live (Atlas/Chord/Hierarchical network); hierarchy disables only on v1 exports with a "needs edges" hint |
| 21 | Meta/status line (our honesty addition; no video analogue) | chrome | 2 | dataset · space · pts · clusters · noise% · namer · edges · gpu — always visible |
| 22 | Chord view: radial rotated labels (f04, f08) | chord | 2 | M6a: rim-tangent rotated HTML labels, greedy collision-culled (86 @ 192 clusters), flip past 90° so text never inverts |
| 23 | Chord view: dotted gradient chords + thick strong ribbons (f04, f08) | chord | 2 | M6a: instanced bezier ribbons, endpoint-to-endpoint ramp hue, weight-gated dotted↔solid, hover/click focus dims the rest to 5% |
| 24 | Hierarchical network view (f13 radio) | hier | 2 | M6b: deck.gl radial dendrogram — single-linkage tree over the exported cluster_edges, elbow links ramp-colored by merge weight, cluster-colored leaf discs, rotated radial labels; hover lights a leaf's root path or a join's subtree |
| 25 | 2D↔3D morph / camera tilt (f16→f17 lift) | morph | 2 | M3: 900ms eased uMorph tween synced with camera fly; 38° tilt; flat-map furniture (territories/pills/halos) fades out; id-buffer hover picking in 3D |
| 26 | View-mode crossfade transitions (f07→f08) | morph | 2 | M4: 300ms opacity crossfade atlas↔compare (150ms reduced motion); both drivers frame through the fade; chrome swaps LegendCard↔ComparePanel |

## Gate math

- **Map-view rows (1–14)**: 26 / 28 = **93%** ✅ (gate ≥80%)
- Chrome rows (15–21): 14 / 14 = 100%
- Overall (all rows): 50 / 52 = **96%** after M6b. The two remaining 1s are
  deliberate scope calls: row 13 (per-particle comet heads) and row 14 (the
  video hand-curates fewer, larger overview labels).

## Deliberate divergences (scored as-is, not defects)

- **Badge values**: video badges show sizes/counts; ours show cluster sizes for
  cluster beams and raw cosine sims for kNN beams — never display-space
  distances (guardrail).
- **Beam display weights**: per-selection normalized to [0.15, 1] so the
  solid-vs-dotted split reads like the video even for tight similarity ranges;
  badges always show the raw values.
- **Edges provenance**: legend + meta line state `gaussian_euclidean@umap10`;
  the video shows no provenance. Extra chrome, kept.
- **Noise**: video hides weak points; we dim them (honesty guardrail). Toggle
  exists to hide.

## Verification log (M2)

- WebGPU tier: full interaction battery (select cluster → 11 beams + flare +
  badges; kNN select → 6 beams + sim badges; all toggles; settings sliders;
  Escape; dataset switch) — **0 console errors** (`window.__errs`).
- WebGL rung (`?gpu=webgl`): beams + halos render, bloom correctly disabled
  with sidebar hint, 0 errors.
- Perf (forced-frame, bloom + 11 beams + camera/label/badge update):
  p50 0.1ms, **p95 0.3ms**, max 4.1ms — budget p95 ≤16.6ms ✅. Boot 338ms,
  parse 253ms (budget <800ms ✅), cached-page dataset switch 503ms (budget
  <400ms is for *cached* switch — cache lands in M4; naive switch acceptable).
- `?frozen=1` pins uTime on beams/flare/halos — golden-ready.
- Unit: vitest 27/27 (edges math, columns, hulls, camera, tokens sync);
  `tsc --noEmit` clean.

## Verification log (M3)

- Morph: setDims(3) → 900ms easeInOutCubic tween completes (uMorph 0→1,
  store.morphT synced); camera tilts to 38° with exact expected position
  (posY = cy − sin38°·camDist, posZ = cos38°·camDist); territories hidden,
  label pills opacity 0 + pointer-events none, halo uFade 0.
- Id-buffer picking, both rungs: direct pick at a projected point returns the
  visually topmost instance whose re-projection lands **3.4px** from the
  cursor (WebGPU and `?gpu=webgl` return the identical id). Readback origin
  is **top-left on WebGPU, bottom-left on WebGL** — IdPicker branches on
  `backend.isWebGLBackend`.
- 3D hover end-to-end: pointermove → throttled 30Hz async pick →
  store.hover + .point-tooltip anchored via the same projection the GPU uses.
- 3D click select: store.selection set; beams/badges correctly suppressed in
  3D (count 0 — beams are flat-map furniture).
- Return path: setDims(2) restores territories/pills/halos to full opacity,
  clears selection; cluster select after the round-trip renders 5 beams +
  5 badges.
- 0 console errors across the entire battery (fresh `window.__errs` both
  tiers); vitest 27/27, `tsc --noEmit` clean.

## Verification log (M4)

- CompareDriver (raw WGSL, own device + `#compare-canvas`): renders the
  837-concept cloud in all four layout states; state radio → store →
  driver subscription → 900ms smoothstepped tween (verified from/to state
  retarget + completion for semantic, by_model, by_concept).
- Crossfade: sidebar Type → compare swaps canvas opacity (atlas 1→0,
  compare 0→1), toggles pointer-events, stamps `#stage.mode-compare`
  (hides label pills / badges / atlas tooltip via CSS), swaps
  LegendCard→ComparePanel, and flips the MetaLine to the honest compare
  line ("label space, not model geometry"). Return path restores all of it
  and hides the compare tooltip.
- ComparePanel: model legend toggles mirror to the driver visibility UBO
  (visible[0]=0 on toggle), "Shared concepts only" mirrors to the shared
  flag; shared/unique counts + Jaccard table render from stats.
- Hover: CPU-projection tooltip shows `title · model · shared/unique ·
  N tokens` (e.g. "Work & Workplace / gpt2 · shared concept · 115 tokens").
- Dataset switcher: uncached gpt2 load 188ms; **cached switch back 0.3ms**
  (budget <400ms ✅ — per-dataset column cache).
- CLI: `nebulai compare` now prints the unified-viewer URL
  (`?view=compare` deep link, verified working); `backend/viewer.py`
  marked deprecated, still writes index.html for one release.
- 0 console errors across the battery; `tsc --noEmit` clean (WebGPU
  constant shims in `webgpu-globals.d.ts`); vitest 27/27.

## Verification log (M6a — chord view)

- ChordDriver (three-TSL on `#chord-canvas`, both rungs): 192 rim nodes,
  610 chords, 86 collision-culled rotated radial labels on pythia-70m;
  ring order follows atlas 2-D centroid angles, rank-spaced.
- **Root-caused the invisible-chords bug**: WebGPU's `maxVertexBuffers`
  is 8; PlaneGeometry binds position+normal+uv, and six per-instance
  attributes made 9 → `createRenderPipeline` fails **in an async task**
  (no error surfaces at draw call; the invalid pipeline spams
  uncapturederror). TSL tree-shakes unreferenced attributes, so simplified
  diagnostic materials all stayed ≤8 and compiled — only the full graph
  failed. Fix: per-instance data packed into two vec4s (`seg` =
  start.xy/end.xy, `meta` = weight/rampA/rampB/active) → 5 buffers.
- Hover: radial/angular hitTest → tooltip `title · N tokens · d links
  (gaussian euclidean in umap10)` — metric read from `edges.metric/space`,
  not hardcoded; hovered node swells, non-neighbors dim to 5%, labels get
  `.is-dim` (83/86 dimmed on a degree-6 cluster).
- Click pins selection (survives pointer-leave: 6/610 chords lit);
  Escape clears (610/610 relit); store round-trip verified.
- Crossfade atlas↔chord both directions: canvas opacity/pointer-events
  swap, `#stage.mode-chord` scopes overlay visibility (atlas pills/badges
  hidden in chord; chord labels/tooltip hidden in atlas), Dimensions row
  is atlas-only, MetaLine persists.
- Dataset switch while in chord mode: distilgpt2 → 199 nodes / 637 chords /
  88 labels rebuilt, 0 GPU errors.
- `?gpu=webgl` rung: identical render (TSL→GLSL), 0 errors — ChordDriver
  is deliberately not WebGPU-gated (unlike CompareDriver).
- `?view=chord&frozen=1` deep link boots straight into the frozen chord
  scene — golden-ready. vitest 27/27; `tsc --noEmit` clean.

## Verification log (M6b — hierarchy view)

- HierarchyDriver (deck.gl 9.3, lazy-imported): single-linkage dendrogram
  built client-side by Kruskal + union-find over the exported `cluster_edges`
  (descending weight), so every join is the honest "these subtrees merge at
  similarity w" in umap10. Sparse top-k forests gather under a synthetic
  root at weight 0 whose tooltip says so. Join radius is normalized by the
  strongest merge (display-only); tooltips always carry the raw weight.
- pythia-70m: 192 leaves / 381 nodes / 380 links / 80 collision-culled
  rotated radial labels; distilgpt2 switch in-mode rebuilds to 199 leaves /
  397 nodes / 396 links / 88 labels — 0 errors both.
- Hover leaf: tooltip `title · N tokens · joins at 0.99 (gaussian euclidean
  in umap10)`, 21-node root path lit, 79/80 labels dimmed, store hover set.
- Hover join: `join · 8 clusters beneath · merge similarity 0.93 (…)`,
  subtree + ancestor chain focused (view-local; joins aren't clusters so no
  store hover). Click pins a leaf (survives pointer-leave); Escape clears
  via the store round-trip.
- deck integration per the nebulai-viz laws: own `#hier-canvas`, controller
  off, viewState derived from stage size (OrthographicView flipY:false),
  our pointer handlers call `deck.pickObject` — deck owns no interaction
  state. `frame()` is a no-op (deck renders on demand; nothing time-based,
  so `?frozen=1` is trivially golden-ready).
- Crossfade atlas↔hierarchy both directions clean; `.mode-hierarchy` stage
  class scopes overlays; `?view=hierarchy&frozen=1` deep link boots direct.
- `?gpu=webgl` rung: identical render (deck is WebGL2 on both rungs), meta
  line shows `gpu: webgl`, 0 errors.
- Bundle: deck lands in two lazy chunks (~221KB gz) loaded on first switch;
  main bundle 276.8KB gz — under the 500KB budget. vitest 27/27;
  `tsc --noEmit` clean.

## Verification log (task #10 — testing + parity pass)

Full battery, all green at close:

- **pytest 8/8** (`uv run pytest`, now a `[dependency-groups] dev` dep):
  edges symmetry/dedup/weight-range, kNN self-exclusion + flat lengths,
  backfill round-trip fixture.
- **vitest 27/27** (`npm run test`; vitest scoped to `tests/unit/**` in
  vite.config.ts so it never grabs the Playwright specs).
- **Playwright e2e: 14 passed / 8 skipped / 0 failed** (`npm run e2e`,
  projects `webgl` + `webgpu`). All 8 skips are by-design rung gating
  (compare is WebGPU-only; chrome/a11y/perf specs run once on webgl).
  Coverage: screenshot goldens per view per rung
  (`tests/e2e/goldens/{webgl,webgpu}/`), meta-line honesty substrings in
  every view, zero console/page/network(≥400) errors on every boot,
  hierarchy-radio gating, perf budgets, axe, keyboard, Escape,
  reduced motion.
- **Headless GPU story (the one environment trap worth recording)**:
  Playwright's default headless_shell renders WebGL on **SwiftShader** —
  3fps at 50K points, main thread saturated (keyboard events starve >80s,
  rAF perf window never fills, screenshots time out). The config uses
  `channel: "chromium"` + `--use-angle=metal` → real GPU ("ANGLE Metal
  Renderer: Apple M1 Pro", 93fps). Headless **WebGPU** works on the same
  channel with `--enable-unsafe-webgpu --enable-gpu` — the compare golden
  runs on a real adapter. Goldens are renderer-specific: regenerate
  (`npm run e2e:update`) if the renderer ever changes.
- **Perf budgets, measured on the webgl rung**: parse 318ms (<800),
  boot 579ms (<3000), steady **p95 10ms** (≤16.7), cached dataset switch
  <400ms (store-cache path, measured via `#sel-dataset` round-trip).
  Bundle 276.8KB gz main + 221KB gz lazy deck chunks (<500KB budget).
- **A11y**: axe-core — zero serious/critical; Tab order reaches sidebar
  selects and legend radios; Escape deselects via the store round-trip;
  `prefers-reduced-motion` drops crossfades to 150ms (asserted on the
  canvas transition style).
- **Component review gate** (checklist copied to `viewer/CONTRIBUTING.md`)
  run live against the chrome via computed styles — it caught two real
  defects, both fixed in chrome.css and re-verified:
  - `.ctl-switch` was a 34×20 target and `.ctl-radio` rows ~17px — below
    the 24px AA floor. Fixed with expanded hit areas (switch `::before`
    inset −4px → 42×28; radio rows padded to 24.8px), visuals unchanged.
  - `.boot-status` (the MetaLine — the chrome's most number-dense line)
    had `font-variant-numeric: normal`. Now `tabular-nums`.
  - Resize check also caught the MetaLine clipping off-viewport at 375px
    (nowrap, ~1131px wide). Now `width: max-content` +
    `max-width: calc(100vw − 24px)` → single line on desktop, wraps on
    narrow viewports. Goldens re-ran clean after all three fixes.

### Honest scope notes (not regressions — never built)

- `#scene-canvas` is **not** a focusable application region: the plan's
  "arrows nudge camera, Enter selects" keyboard-canvas mode didn't land.
  Mouse/touch + sidebar/legend keyboard paths are complete; the canvas
  itself is pointer-only. Recorded as future work, not claimed as done.
- Parity rows 13 (per-particle comet heads) and 14 (hand-curated overview
  label density) remain deliberate 1s — see gate math above (96% overall).
