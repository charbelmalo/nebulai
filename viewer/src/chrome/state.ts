/** zustand → @preact/signals bridge. One signal per store slice the chrome
 *  reads; a single store subscription keeps them in sync. Writes go the other
 *  way through store setters / app actions — signals are read-only mirrors. */

/*  The signals themselves now live in `state/`, one module per store slice, so
 *  the bridge splits along the same seam the store does and a future per-app
 *  entry can pull in only the mirrors it needs. This file is the barrel that
 *  keeps every existing `from "./state"` import working untouched — no
 *  consumer knows or cares which module a signal came from.
 *
 *  `state/sync.ts` holds the SINGLE `appStore.subscribe` that writes them all,
 *  in the exact order it always has. It is imported for its side effect below;
 *  do not give any state module a subscription of its own.
 *
 *  That side-effect import is load-bearing and silent if it breaks. Nothing
 *  imports a VALUE from sync.ts, so the only thing keeping it in the bundle is
 *  that this package is not marked side-effect-free and every consumer
 *  value-imports this barrel. Add `"sideEffects": false` to viewer/package.json,
 *  or reduce the last consumer to `import type`, and the subscribe is tree-shaken
 *  out: every signal freezes at its boot value and the whole chrome quietly stops
 *  re-rendering — with typecheck and vitest still green, because neither exercises
 *  a production bundle. If you ever need that flag, list this file in
 *  `sideEffects` explicitly. */

export * from "./state/shell";
export * from "./state/atlas";
export * from "./state/interp";
export * from "./state/probing";
export * from "./state/appearance";
export * from "./state/snapshot";
export * from "./state/sessions";
export * from "./state/seer";

import "./state/sync";
