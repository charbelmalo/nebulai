/** Runtime data root for the baked `out/` artifacts.
 *
 *  The app shell is served under Vite's BASE_URL — which may be the domain
 *  root ("/") during dev or a sub-path (e.g. "/psychiX/nebulai-maps/") in a
 *  static deploy. The baked artifact tree is served as `out/` directly beneath
 *  that base, so we derive the data root from BASE_URL rather than hardcoding
 *  an absolute "/out". This makes every static fetch (index.json, per-dataset
 *  nebulai.json, interp bundles, compare.json) resolve correctly whether the
 *  viewer is hosted at the root or under an arbitrary sub-path.
 *
 *  DATA_BASE is resolved to an *absolute* URL against the document location.
 *  This is load-bearing: the parse worker (data/parse.worker.ts) fetches
 *  `${DATA_BASE}/<path>`, and relative URLs inside a Worker resolve against the
 *  worker script URL — not the document — so a relative base like "./out" would
 *  fetch from the wrong origin/path. Anchoring on location.href keeps the base
 *  identical on the main thread and in the worker.
 *
 *  Dev: BASE_URL="./" -> "http://localhost:5173/out"  (served by the serveOut()
 *       vite middleware at /out)
 *  Prod (VITE_BASE=/psychiX/nebulai-maps/):
 *       BASE_URL="/psychiX/nebulai-maps/"
 *       -> "https://<host>/psychiX/nebulai-maps/out"
 */
const APP_BASE = new URL(import.meta.env.BASE_URL || "/", location.href);
export const DATA_BASE = new URL("out/", APP_BASE).href.replace(/\/+$/, "");
