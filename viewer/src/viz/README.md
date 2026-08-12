# src/viz — the shared chart/control kit

Design tokens, chart aesthetics, and Preact control primitives shared by
Nebulai and the future Seer app, imported as `@psychix/viz/*` (see
`tsconfig.json` / `vite.config.ts`). No barrel — import each module by name
so a scene driver never pulls Preact components in through one specifier.

viz imports nothing from `../app`, `../chrome`, `../scene`, `../seer`,
`../data` — it is a leaf. Keep it that way by hand; nothing here enforces it.

**Here:** `tokens.ts`/`tokens.css`/`craft-tokens.css` (color + ramp tokens,
CSS-synced), `chart-theme.ts`/`chart-tooltip.ts`/`logscale.ts` (chart
aesthetic, hover tooltip, log axis), `capabilities.ts` (GPU-tier probe),
`controls.tsx`/`StatStrip.tsx`/`ChartCard.tsx` (form controls, stat tiles, chart frame).

**Not here, on purpose:** `scene/interp/chart-stage.ts` and `field2d.ts` pull
in `appStore`/gestures/post-bloom — they're the render engine, not a
primitive. `chrome/Tooltip.ts` and `chrome/BeamBadges.ts` are typed against
atlas domain data (`clusterTitle`/`confidence`) — moving them here would leak
Nebulai's data model into a package Seer is meant to reuse untouched.
